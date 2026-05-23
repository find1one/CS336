# Tokenizer Progress Notes

This note reviews the current state of `cs336_basics/tokenizer.py` and records what is already in place, what still needs attention, and useful invariants to check next.

## Current Structure

- `PAT` is defined with the GPT-2-style pre-tokenization regex.
- `pretokenize_encode(text, special_tokens)` is separated from the `Tokenizer` class.
- `Tokenizer.__init__` accepts:
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`
  - `special_tokens: list[str] | None = None`
- `Tokenizer.__init__` copies `vocab` into `self.vocab`, stores `self.merges`, normalizes `special_tokens` to a list, builds `self.bytes_to_id`, and builds `self.merge_rank`.
- `Tokenizer.__init__` also appends missing special tokens to `self.vocab` and `self.bytes_to_id`.
- Method stubs exist for:
  - `from_files`
  - `encode`
  - `encode_iterable`
  - `decode`

## Good Direction

- Keeping `vocab`, `merges`, `special_tokens`, `bytes_to_id`, and `merge_rank` inside the class is the right object boundary.
- Copying `vocab` before appending special tokens avoids mutating the caller's original dictionary.
- Building `bytes_to_id` is useful because encoding eventually needs to convert final byte tokens into integer IDs.
- Building `merge_rank` is useful because encoding must respect the training order of merges.
- Separating pre-tokenization into a helper function is reasonable because it is a reusable text-splitting step.

## Current Blocking Issues

- `tokenizer.py` is not currently importable because `from_files`, `encode_iterable`, and `decode` have function headers but no body.
- `Iterable` and `Iterator` are used in type annotations but are not imported.
- `from_files` has no implementation yet.
- `encode_iterable` has no implementation yet.
- `decode` has no implementation yet.

These should be fixed before running tokenizer tests, because Python needs the file to parse before any test can execute.

## `pretokenize_encode` Notes

- The function currently returns only normal pre-tokens and skips special tokens with:
  - `part == "" or part in special_tokens`
- That means special tokens are removed from the returned sequence.
- Later, `encode` checks whether a token is special, but that branch is unlikely to run because the helper already skipped special tokens.
- Clarify the intended invariant:
  - For BPE training, special tokens should be excluded from ordinary pair statistics.
  - For tokenizer encoding, special tokens should usually remain in the token stream as atomic units.

This is an important conceptual difference between training pre-tokenization and encode-time pre-tokenization.

## `__init__` Notes

- `self.vocab = dict(vocab)` is good because special tokens may be appended without changing the original input.
- `self.bytes_to_id` is currently built from `vocab`, not `self.vocab`, and then manually updated for special tokens. This is fine as long as every later vocabulary addition is also reflected in both dictionaries.
- `self.special_tokens = [] if special_tokens is None else special_tokens` works, but remember that if the caller passes a list, `self.special_tokens` refers to the same list object.
- If special tokens can overlap, think about whether longer special tokens should be matched before shorter ones during splitting.

## `encode` Notes

- The high-level plan in the comments is right:
  - pre-tokenize text
  - convert each ordinary pre-token into byte-level pieces
  - apply BPE merges
  - map final byte tokens to integer IDs
  - preserve special tokens as single IDs
- The current merge loop needs careful review.
- `while i in range(len(token))` uses the string length, while the merge state is stored in `token2`, a tuple of byte tokens. For non-ASCII text, string length and UTF-8 byte length can differ.
- `token3` is reset inside the `while` loop, so earlier pieces from the same pre-token may be lost.
- In the non-merge branch, `i` is not advanced, which can lead to a loop that never finishes.
- In the non-merge branch and final-token branch, `token[i]` is a string character, not a `bytes` object. The final encoded pieces should stay in the same representation as `bytes_to_id` keys.
- `self.bytes_to_id(token.encode("utf-8"))` uses call syntax, but dictionaries are looked up with square brackets.
- Think about whether one left-to-right pass is enough. BPE encoding usually keeps applying the best available merge according to merge rank until no mergeable adjacent pair remains inside the current pre-token.

## Test Adapter Expectations

The test adapter expects a `get_tokenizer(vocab, merges, special_tokens)` function to return an object with these methods:

- `encode(text) -> list[int]`
- `decode(ids) -> str`
- `encode_iterable(iterable) -> Iterator[int]`

The test file also exercises:

- empty strings
- single ASCII characters
- Unicode characters
- longer ASCII and Unicode strings
- special tokens such as `<|endoftext|>`
- large-file behavior through `encode_iterable`

## Useful Invariants To Check

- `decode(encode(text)) == text` for small examples.
- Every item emitted by the merge process should be a `bytes` object before converting to token IDs.
- Every emitted byte token should exist in `self.bytes_to_id`.
- Special tokens should encode to exactly one token ID.
- Special tokens should decode back to the exact same string.
- BPE merges should not cross pre-token boundaries.
- UTF-8 multi-byte characters should round-trip correctly.
- `encode_iterable` should avoid loading an entire large file into memory at once.

## Suggested Next Debugging Order

1. Make the file syntactically importable by giving unfinished methods temporary bodies while you work.
2. Decide whether `pretokenize_encode` should return special tokens as separate pieces during encoding.
3. Test `pretokenize_encode` on small strings with and without special tokens.
4. Test ordinary byte conversion on ASCII and one Unicode character.
5. Test the merge loop on a tiny hand-made token sequence before using the full GPT-2 fixture.
6. Test `decode` independently on known token IDs.
7. Only after small checks pass, run `tests/test_tokenizer.py`.


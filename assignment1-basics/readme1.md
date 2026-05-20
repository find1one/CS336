# BPE Training Notes

This note summarizes the current `cs336_basics/train_bpe.py` state, the concepts it touches, and the remaining work.

## Knowledge Points

- **Byte-level BPE**
  - The assignment works with bytes, not Python characters.
  - A string must be encoded with UTF-8 before byte-level BPE can operate on it.
  - Initial vocabulary usually maps each byte id `0..255` to its one-byte `bytes` value.

- **Vocabulary**
  - `vocab` is currently represented as `dict[int, bytes]`.
  - The key is the token id.
  - The value is the token content in bytes.
  - New merged tokens are added by concatenating the bytes of the selected pair.

- **Merges**
  - `merges` should be a `list[tuple[bytes, bytes]]`.
  - Each entry records the two byte tokens selected for a merge.
  - The order of `merges` matters because encoding later applies merges in training order.

- **Token ids vs. token bytes**
  - Internal training can use token ids for easier list updates.
  - The public `merges` output must use bytes pairs, not int pairs.
  - The current code converts selected token ids back into bytes when appending to `merges`.

- **Merge operation**
  - `merge(indices, pair, new_index)` scans a token-id sequence left to right.
  - When it sees the target adjacent pair, it emits `new_index` and skips both old ids.
  - This avoids overlapping merges, e.g. merging `(1, 1)` in `[1, 1, 1]` should produce `[new, 1]`.

- **Pre-tokenization**
  - `pretokenize(text, special_tokens)` uses the GPT-2-style regex pattern from the handout.
  - It first splits around special tokens so special tokens do not participate in normal BPE pair statistics.
  - It returns a frequency table keyed by tuples of one-byte `bytes` objects.
  - Example verified behavior:
    - `"ab ab"` becomes `{(b'a', b'b'): 1, (b' ', b'a', b'b'): 1}`.
    - `"ab<|endoftext|> ab"` with `["<|endoftext|>"]` excludes the special token from the frequency table.

- **Special tokens**
  - Special tokens are added directly to `vocab`.
  - They should not be split into ordinary bytes for BPE training.
  - They should not be merged with neighboring ordinary tokens.

- **Frequency table**
  - The pre-token frequency table records how many times each pre-token appears.
  - Pair counts during BPE training should be computed inside each pre-token only.
  - If a pre-token appears multiple times, its internal pair counts should contribute by that frequency.

## Current Status

- `merge` has been changed from a `for` loop to a manually controlled `while` loop.
- `pretokenize` now handles the empty-special-token case instead of splitting on an empty regex.
- `pretokenize` converts matched pre-token strings into tuples of one-byte `bytes`.
- `train_bpe` reads the file contents from `input_path`.
- `train_bpe` initializes the byte vocabulary and appends special tokens.
- `train_bpe` returns `(vocab, merges)`, matching the adapter-level expected shape.

## Remaining TODO List

- **Use `freq_table` in training**
  - `train_bpe` currently computes `freq_table`, but then still trains on `indices = list(map(int, string.encode("utf-8")))`.
  - This means the current merge loop still treats the whole file as one continuous byte stream.
  - The next step is to make pair counting and merging operate on the pre-token frequency table.

- **Avoid cross-pre-token merges**
  - Current training can learn tokens like `b"de "` or combinations that cross natural pre-token boundaries.
  - Pair statistics should only be counted within each pre-token tuple.

- **Account for pre-token frequency**
  - When counting adjacent pairs inside a pre-token, multiply by the pre-token count.
  - Example: `{(b'a', b'b'): 1, (b' ', b'a', b'b'): 2}` contributes `(b'a', b'b')` three times total.

- **Update all pre-token sequences after each merge**
  - After selecting a pair, every pre-token tuple containing that pair must be updated.
  - The frequency count for that pre-token should remain attached to the updated token sequence.

- **Clarify representation during training**
  - Decide whether the active pre-token sequences store token ids or bytes.
  - Keep the representation consistent so pair counting, tie-breaking, vocab updates, and merge records agree.

- **Confirm tie-break rule**
  - Current code breaks frequency ties with `vocab[k[0]] + vocab[k[1]]`.
  - Recheck the handout rule: whether ties should prefer lexicographically greater or smaller byte pairs, and whether comparison should be pair-wise or concatenated bytes.

- **Handle special-token regex ordering**
  - If special tokens overlap, longer special tokens should be matched first.
  - This avoids splitting a long special token into shorter special-token pieces.

- **Clean unused code/imports**
  - `os`, `Counter`, `Iterable`, and `dataclass` are currently unused or only partly planned.
  - The commented `bpeParams` and `BPETokenizer` sections can stay while learning, but they are not part of the current `train_bpe` path.

- **Add small manual checks**
  - `merge([1, 2, 1, 2, 3], (1, 2), 9)` should produce `[9, 9, 3]`.
  - `merge([1, 1, 1], (1, 1), 9)` should produce `[9, 1]`.
  - `pretokenize("ab ab", [])` should separate `"ab"` and `" ab"`.
  - `pretokenize("ab<|endoftext|> ab", ["<|endoftext|>"])` should not include the special token.


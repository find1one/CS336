# BPE Training Debug Notes - 2026-05-22

This note records today's review of `cs336_basics/train_bpe.py`.
Use it as a checklist for the next debugging pass.

## Current Direction

- [ ] Use the pre-token frequency table as the main BPE training state.
- [ ] Avoid using one global `indices` sequence over the entire input file.
- [ ] Keep pair counting inside each pre-token boundary.
- [ ] Use token ids internally during training for simpler pair counting and merging.
- [ ] Convert selected token ids back to bytes only when updating `vocab` and `merges`.

## Representation Checklist

- [ ] Confirm the output of `pretokenize` is still:
  - `dict[tuple[bytes, ...], int]`
  - each key is one pre-token split into one-byte `bytes` objects
  - each value is the frequency of that pre-token
- [ ] Add a separate conversion step from byte-token keys to token-id keys.
- [ ] Do not mutate dictionary keys in place.
- [ ] Do not mutate tuple elements in place.
- [ ] After conversion, the training table should conceptually be:
  - `dict[tuple[int, ...], int]`
  - example: `(b'a', b'b') -> 3` becomes `(97, 98) -> 3`
- [ ] Make sure type annotations match the representation actually used.

## `countAdjacent` Checklist

- [ ] `countAdjacent` should operate on the token-id frequency table, not the byte frequency table.
- [ ] Its input should conceptually be `dict[tuple[int, ...], int]`.
- [ ] Its output should conceptually be `dict[tuple[int, int], int]`.
- [ ] For each pre-token sequence, count only adjacent pairs inside that sequence.
- [ ] Multiply each adjacent pair count by the pre-token frequency.
- [ ] Skip sequences whose length is less than 2.
- [ ] Check that pair counts from repeated pre-tokens accumulate correctly.

## `merge` Checklist

- [ ] `merge` is still needed, but it should update the frequency table instead of a global `indices` list.
- [ ] Its return type should be a new frequency table, not `list[int]`.
- [ ] Every old key must produce a new key, even if the selected pair does not appear in it.
- [ ] The old frequency count should stay attached to the new key.
- [ ] If two different old keys merge into the same new key, their counts should be added.
- [ ] A key with multiple non-overlapping occurrences of the selected pair should merge all of them in one pass.
- [ ] Overlapping behavior should match left-to-right BPE behavior.
- [ ] Sanity check: merging `(1, 1)` in `(1, 1, 1)` should produce `(new_id, 1)`, not `(new_id, new_id)`.
- [ ] Sanity check: merging `(1, 2)` in `(1, 2, 1, 2)` should produce `(new_id, new_id)`.

## `train_bpe` Loop Checklist

- [ ] Initialize `vocab` with byte ids `0..255`.
- [ ] Append `special_tokens` to `vocab` after the initial byte vocabulary.
- [ ] Build the byte-level pre-token frequency table with `pretokenize`.
- [ ] Convert that table into a token-id frequency table before the BPE merge loop.
- [ ] In each iteration, compute adjacent pair counts from the current token-id table.
- [ ] If there are no adjacent pairs left, decide whether to stop early.
- [ ] Select the best pair using the assignment's required frequency and tie-break rule.
- [ ] Assign the next available token id to the selected pair.
- [ ] Update all pre-token sequences through the table-level merge operation.
- [ ] Append the byte form of the selected pair to `merges`.
- [ ] Add the concatenated byte token to `vocab`.

## Important Bugs Found Today

- [ ] The current conversion attempt loops over `len(key)` directly, but `len(key)` is an integer.
- [ ] The current conversion attempt tries to mutate tuple elements, but tuples are immutable.
- [ ] The current conversion attempt tries to change dictionary keys in place, which is not valid.
- [ ] The current `merge` can drop pre-token sequences that do not contain the selected pair.
- [ ] The current `merge` does not correctly accumulate multiple replacements within one key.
- [ ] The current `merge` does not combine counts when multiple old keys become the same new key.
- [ ] The current `merge` type annotation does not match what it returns.
- [ ] The current `countAdjacent` type annotation still mentions byte keys, even though the intended state is token ids.
- [ ] The training loop should handle the case where `pairs` is empty before calling `max`.

## Manual Checks For Tomorrow

- [ ] Check `pretokenize("ab ab", [])`.
- [ ] Check `pretokenize("ab<|endoftext|> ab", ["<|endoftext|>"])`.
- [ ] Check byte-to-id conversion on `{(b'a', b'b'): 2}`.
- [ ] Check pair counting on `{(97, 98): 2, (32, 97, 98): 1}`.
- [ ] Check one merge on `{(97, 98): 2, (32, 97, 98): 1}` with selected pair `(97, 98)`.
- [ ] Check that no merge crosses pre-token boundaries.
- [ ] Check that special tokens are added to `vocab` but do not participate in ordinary pair statistics.
- [ ] Re-read the handout's tie-break rule and compare it against the current `max` key.

## Invariants To Keep In Mind

- [ ] `vocab[token_id]` should always return the full byte content of that token.
- [ ] `merges` should record byte pairs, not token-id pairs.
- [ ] The internal training table can use token ids, but the public output must match the adapter's expected format.
- [ ] Pair counting should never observe pairs that cross pre-token or special-token boundaries.
- [ ] After each merge, the total weighted number of pre-token occurrences should stay the same.
- [ ] Only the token sequence keys change during merging; their frequencies represent corpus counts and should be preserved or combined.


"""CI (no GPU): the network<->input_ids identity tie is prefix-cache robust.

Under prefix caching the engine copies only a request's UNCACHED tail into input_ids, so the
prompt HEAD is absent — the old prefix-keyed match missed every cache-hit request (the ~60%
ceiling). Both ties now key on the longest uncached SUFFIX / contiguous containment, which is
exact for any request with >= min_len uncached tokens. A FULLY cached request copies nothing
and is correctly NOT placed by content — that residual is precisely what OTel covers (off-wire
request_id), so the network face and OTel are complementary, not redundant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from observe.net_label import (NetRequest, match_to_input_ids,  # noqa: E402
                               reconstruct_seq_to_request)

P = list(range(1, 11))                        # 10-token shared prefix -> prefix-CACHED
TAILS = {0: [101, 102, 103, 104], 1: [201, 202, 203, 204, 205],
         2: [301, 302, 303, 304, 305, 306]}   # distinct uncached tails (>= min_len)


def _nr(i, toks):
    return NetRequest(fd=i, t_ns=i, method="POST", path="/v1/completions",
                      prompt="", arrival_idx=i, token_ids=toks)


def _net():
    net = [_nr(i, P + TAILS[i]) for i in (0, 1, 2)]
    net.append(_nr(9, list(P)))               # a FULLY cached request: nothing in input_ids
    return net


def test_reconstruct_suffix_keyed():
    # input_ids holds only the uncached tails, concatenated in seq order (head is cached away)
    vals = TAILS[0] + TAILS[1] + TAILS[2]
    seq2req = reconstruct_seq_to_request([{"dst": 100, "vals": vals}], _net())
    assert seq2req == {0: 0, 1: 1, 2: 2}, seq2req      # placed by tail; fully-cached (9) absent


def test_match_to_input_ids_containment():
    rows = [TAILS[0], TAILS[1], TAILS[2]]             # per-slot uncached chunks
    m = match_to_input_ids(_net(), rows, min_overlap=4)
    assert m == {0: 0, 1: 1, 2: 2}, m


def test_fully_cached_not_placed_by_content():
    # only the fully-cached request is active -> its tail isn't in input_ids -> no content tie
    vals = [101, 102, 103, 104]                       # some other request's tail
    seq2req = reconstruct_seq_to_request([{"dst": 100, "vals": vals}], [_nr(9, list(P))])
    assert seq2req == {}, seq2req                      # correctly unplaced -> OTel's job


if __name__ == "__main__":
    test_reconstruct_suffix_keyed()
    test_match_to_input_ids_containment()
    test_fully_cached_not_placed_by_content()
    print("PASS: suffix/containment ties place prefix-cached requests the old prefix-match "
          "missed; the fully-cached request is left for OTel.")

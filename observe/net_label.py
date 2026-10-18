"""Optional network-side request LABEL for eKV (prompt text and external request id).

Request identity itself is intrinsic to the location map: a request is keyed by its own physical
blocks, recovered with no engine hook and no tokenizer (the Triton pass writes the block
# table into its capture buffer; observe/zero_touch.py reads it). This module is
optional and adds only the human-facing label, for billing or log correlation. The eBPF http_label
collector captures the bytes the API server recv()s (plaintext or TLS); here we reassemble that byte
stream per connection, parse the HTTP request, and extract the prompt and path. Tokenizing the prompt
(optional) gives the token sequence the engine H2D-copies as input_ids, a content cross-check against
the eBPF-captured input_ids -- not the identity mechanism. The cross-check is `match_to_input_ids`
below."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class NetRequest:
    fd: int
    t_ns: int
    method: str
    path: str
    prompt: str
    arrival_idx: int
    token_ids: list[int] = field(default_factory=list)


def _content_length(head: str) -> int | None:
    for ln in head.split("\r\n")[1:]:
        if ln.lower().startswith("content-length:"):
            try:
                return int(ln.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _parse_request(buf: bytes):
    """Return (method, path, body) once a COMPLETE HTTP request is in buf, else None."""
    i = buf.find(b"\r\n\r\n")
    if i < 0:
        return None
    head = buf[:i].decode("latin1", "replace")
    body = buf[i + 4:]
    first = head.split("\r\n", 1)[0].split()
    if len(first) < 2:
        return None
    method, path = first[0], first[1]
    cl = _content_length(head)
    if cl is not None and len(body) < cl:
        return None                                   # body not fully arrived yet
    return method, path, (body[:cl] if cl is not None else body)


def extract_prompt(body: bytes) -> str | None:
    """The prompt from an OpenAI completions/chat body (request identity content)."""
    try:
        obj = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    if "prompt" in obj:
        p = obj["prompt"]
        if isinstance(p, str):
            return p
        return p[0] if isinstance(p, list) and p and isinstance(p[0], str) else None
    if "messages" in obj:
        return "\n".join(m.get("content", "") for m in obj["messages"]
                         if isinstance(m, dict) and isinstance(m.get("content"), str))
    return None


def parse_http_recs(path: str, tokenizer=None,
                    only_paths=("/v1/completions", "/v1/chat/completions")
                    ) -> list[NetRequest]:
    """Reassemble the http_label JSONL byte stream per (pid,fd) and emit one
    NetRequest per complete HTTP request (in arrival order). tokenizer (optional, e.g.
    a HF tokenizer) populates token_ids = the sequence the engine will see as input_ids."""
    streams: dict[tuple, bytearray] = {}
    reqs: list[NetRequest] = []
    idx = 0
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "data" not in r or "fd" not in r:
                continue
            key = (r.get("pid"), r["fd"])
            buf = streams.setdefault(key, bytearray())
            buf += bytes.fromhex(r["data"])
            parsed = _parse_request(bytes(buf))
            if parsed is None:
                continue
            method, p, body = parsed
            streams[key] = bytearray()                # consumed; ready for next (keep-alive)
            if only_paths and not any(p.startswith(x) for x in only_paths):
                continue
            prompt = extract_prompt(body)
            if prompt is None:
                continue
            nr = NetRequest(fd=r["fd"], t_ns=int(r["t_ns"]), method=method, path=p,
                            prompt=prompt, arrival_idx=idx)
            idx += 1
            if tokenizer is not None:
                nr.token_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            reqs.append(nr)
    return reqs


def _contig(hay: list[int], needle: list[int]) -> int:
    """Index of the first contiguous occurrence of `needle` in `hay`, or -1."""
    n = len(needle)
    if n == 0 or n > len(hay):
        return -1
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


def _longest_suffix_offset(vals: list[int], toks: list[int], min_len: int, cap: int = 64) -> int:
    """Offset in `vals` of the LONGEST suffix of `toks` (length >= min_len) that occurs
    contiguously in `vals`, or -1. Under prefix caching the engine copies only a request's
    UNCACHED tail into input_ids, and that tail is a SUFFIX of its prompt — so the longest
    prompt-suffix present in input_ids IS that tail. Keying on it (instead of the prompt's
    head, which the cache evicts) is what removes the prefix-caching identity ceiling; the
    tail's offset still orders the seqs (chunks are disjoint, so tail-order == start-order)."""
    hi = min(len(toks), cap)
    for L in range(hi, min_len - 1, -1):
        off = _contig(vals, toks[-L:])
        if off >= 0:
            return off
    return -1


def reconstruct_seq_to_request(meta_records: list[dict], net_reqs: list[NetRequest],
                               key_len: int = 12, min_len: int = 4) -> dict[int, int]:
    """{seq_idx -> net_req.arrival_idx} from the eBPF EKV_META records (the engine's
    input_ids H2D copy) + the network requests — zero engine touch.

    The input_ids copy holds the batch's COMPUTED prompt tokens concatenated in seq order.
    Under prefix caching the cached prefix is NOT copied, so a request contributes only its
    UNCACHED tail — a SUFFIX of its prompt. We key on the longest prompt-suffix present in the
    copy (matching the prompt HEAD was the old ~60% prefix-caching ceiling — the head is cached
    out of input_ids); the tail's offset still orders the seqs. A FULLY cached prompt copies
    nothing and cannot be placed here; that residual is flagged, not guessed (the one corner
    content matching cannot resolve, whether the wire is plaintext or TLS).
    So: seq s (s-th by offset) -> the network request whose tail matched. ``key_len`` caps the
    matched-tail length (enough to disambiguate without scanning whole long prompts)."""
    prompts = [(nr.arrival_idx, list(nr.token_ids))
               for nr in net_reqs if len(nr.token_ids) >= min_len]
    by_dst: dict[int, dict[int, int]] = {}        # dst -> {arrival_idx: offset}
    for r in meta_records:
        vals = r.get("vals") or []
        dst = r.get("dst")
        if dst is None or len(vals) < min_len:
            continue
        for aidx, toks in prompts:
            off = _longest_suffix_offset(vals, toks, min_len, cap=key_len)
            if off >= 0:
                by_dst.setdefault(dst, {}).setdefault(aidx, off)
    if not by_dst:
        return {}
    # input_ids = the copy that matched the most distinct prompts; offset -> seq order.
    best = max(by_dst.values(), key=len)
    order = sorted(best.items(), key=lambda kv: kv[1])     # (arrival_idx, offset) by offset
    return {seq: aidx for seq, (aidx, _off) in enumerate(order)}


def match_to_input_ids(net_reqs: list[NetRequest], rows: list[list[int]],
                       min_overlap: int = 8) -> dict[int, int]:
    """Map engine batch-row index -> net_reqs arrival_idx by token CONTAINMENT (prefix-cache
    robust).

    rows[s] = the input_ids the engine COMPUTED for batch row s (from the eBPF-captured
    input_ids H2D copy, sliced by query_start_loc) = the request's UNCACHED chunk, which under
    prefix caching is a contiguous SUBSEQUENCE (suffix/middle), NOT a prefix, of its prompt. So
    a network request matches row s when row s occurs contiguously in its tokens; the unique
    container wins (matching a prefix was the old prefix-caching ceiling). A row shorter than
    ``min_overlap`` (a nearly/fully cached request) carries too little to place by content; that
    residual (a fully cached prompt) is flagged rather than guessed."""
    out: dict[int, int] = {}
    for s, row in enumerate(rows):
        if len(row) < min_overlap:
            continue
        matches = [nr.arrival_idx for nr in net_reqs
                   if len(nr.token_ids) >= len(row) and _contig(nr.token_ids, row) >= 0]
        if len(matches) == 1:                  # unique container -> exact; ambiguous -> flagged
            out[s] = matches[0]
    return out

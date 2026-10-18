"""R1 gate: does the recorded tile log-sum-exp give the true attention share of each block?

The pass stores, on audit steps, the log of the total attention weight of every tile at [layer][row][block]
[query row] for the KV group ``layer mod groups``. This test calls vLLM's real
``kernel_unified_attention`` (2D and split-KV 3D forms) under the pass, one launch per layer with a
cache of its own, sets the audit flag, normalizes the records with ``observe.tile_shares``, and
compares the result with a float64 reference computed from the same tensors:

  true share  T[b] = mean over the heads h of the sampled group of sum over tokens i in block b
                     of softmax_h(s)[i]

The earlier statistic, a sum of exp(score - running max), had a total variation distance of 0.049
from T in the 2D kernel on real data (0.669 worst), and 0.843 in the 3D kernel, where it kept
0.147 of the share the best 25% of blocks hold. The gate here is exactness.

Inputs are the real queries, keys and values of Llama-3.1-8B-Instruct (every layer, six texts of
512 to 6,144 tokens) and two synthetic controls (flat scores, monotone ramp). Reported per case
(median and worst over launches):

  TV(obs,T)    total variation distance between the normalized distributions
  jac25        Jaccard of the top-25% block sets, observed versus true share
  kept         true share kept by the observed top-25% set, divided by the share kept by the true
               top-25% set (1.0 is optimal)
  recency, random   the same kept ratio for the newest 25% and for a random 25%

Env (the runner sets it): EKV_ATTN_SHARE=1 EKV_NCAP=512 EKV_MAX_ROWS=64 plus the pass env.
"""

import ctypes
import os
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ekv_vllm_plugin import install_pass
from observe.tile_shares import normalize_tile_lse, read_tile_lse, share_descriptor

BLOCK = 16
TOP_FRACTION = 0.25
TV_LIMIT = 1e-3  # the gate: total variation distance from the true share
KV_HEADS = 8
TEXT_TOKENS = (512, 1024, 2048, 3072, 4096, 6144)
LAYERS = tuple(range(32))
MODEL = os.environ.get("EKV_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


_cudart = ctypes.CDLL("libcudart.so.12")


def d2h_f32(dev_ptr: int, count: int) -> np.ndarray:
    torch.cuda.synchronize()
    out = np.zeros(count, dtype=np.float32)
    rc = _cudart.cudaMemcpy(
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_void_p(int(dev_ptr)),
        ctypes.c_size_t(out.nbytes),
        2,
    )
    assert rc == 0, f"cudaMemcpy rc={rc}"
    return out


def h2d(dev_ptr: int, data: np.ndarray) -> None:
    torch.cuda.synchronize()
    rc = _cudart.cudaMemcpy(
        ctypes.c_void_p(int(dev_ptr)), data.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(data.nbytes), 1
    )
    assert rc == 0, f"cudaMemcpy rc={rc}"


@dataclass(frozen=True)
class Sequence:
    """One decode request: the query of the last token and its full K/V history."""

    q: torch.Tensor  # [Hq, D] fp16
    k: torch.Tensor  # [L, Hkv, D] fp16
    v: torch.Tensor  # [L, Hkv, D] fp16


def load_texts() -> list[str]:
    """Four real text sources, each long enough for 6,144 tokens."""
    import pandas as pd
    from datasets import load_dataset

    wiki_path = next(
        (
            os.path.join(r, f)
            for r, _d, fs in os.walk(
                os.path.expanduser("~/.cache/huggingface/hub/datasets--Salesforce--wikitext")
            )
            for f in fs
            if f.startswith("test-") and "wikitext-2-raw" in r
        )
    )
    wiki = " ".join(pd.read_parquet(wiki_path)["text"].tolist())
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_text = "\n\n".join(
        f"Question: {r['question']}\nAnswer: {r['answer']}" for r in gsm.select(range(400))
    )
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    c4_text = " ".join(r["text"] for _i, r in zip(range(300), c4, strict=False))
    para = (
        "The committee reviewed the quarterly infrastructure report and recorded routine details "
        "about regions, maintenance windows, and neutral operating procedures. "
    )
    doc = para * 160 + " The verification code is ZXQ-17000. " + para * 160
    return [wiki, gsm_text, c4_text, doc]


def real_sequences() -> dict[int, list[Sequence]]:
    """Run Llama once per text; return per-layer sequences for the last-token decode step."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama import modeling_llama

    tok = AutoTokenizer.from_pretrained(MODEL)
    assert tok is not None
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
    captured: list[torch.Tensor] = []
    original = modeling_llama.apply_rotary_pos_emb

    def capture(q: torch.Tensor, k: torch.Tensor, *args: object, **kwargs: object):
        q_embed, k_embed = original(q, k, *args, **kwargs)
        captured.append(q_embed[0, :, -1, :].detach().clone())
        return q_embed, k_embed

    modeling_llama.apply_rotary_pos_emb = capture
    texts = load_texts()
    per_layer: dict[int, list[Sequence]] = {layer: [] for layer in LAYERS}
    try:
        for i, n_tokens in enumerate(TEXT_TOKENS):
            ids = tok(texts[i % len(texts)], return_tensors="pt").input_ids[:, :n_tokens].cuda()
            assert ids.shape[1] == n_tokens, f"text source {i % len(texts)} is shorter than {n_tokens} tokens"
            captured.clear()
            with torch.no_grad():
                out = model(ids, use_cache=True)
            cache = out.past_key_values
            for layer in LAYERS:
                per_layer[layer].append(
                    Sequence(
                        q=captured[layer],
                        k=cache.layers[layer].keys[0].transpose(0, 1).contiguous(),
                        v=cache.layers[layer].values[0].transpose(0, 1).contiguous(),
                    )
                )
    finally:
        modeling_llama.apply_rotary_pos_emb = original
    del model
    torch.cuda.empty_cache()
    return per_layer


def synthetic_sequences(kind: str, hq: int = 32, hkv: int = 8, d: int = 128) -> list[Sequence]:
    """Controls: ``flat`` has iid scores; ``ramp`` raises the running max in every tile."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    seqs = []
    for n_tokens in TEXT_TOKENS:
        q = torch.randn(hq, d, generator=gen, device="cuda", dtype=torch.float32)
        k = torch.randn(n_tokens, hkv, d, generator=gen, device="cuda", dtype=torch.float32)
        if kind == "ramp":
            q = torch.zeros(hq, d, device="cuda")
            q[:, 0] = 1.0
            k = torch.zeros(n_tokens, hkv, d, device="cuda")
            k[:, :, 0] = torch.linspace(0.0, 12.0 * d**0.5, n_tokens, device="cuda")[:, None]
        elif kind != "flat":
            raise ValueError(kind)
        v = torch.randn(n_tokens, hkv, d, generator=gen, device="cuda", dtype=torch.float32)
        seqs.append(Sequence(q.half(), k.half(), v.half()))
    return seqs


def reference(seq: Sequence, kv_head: int) -> np.ndarray:
    """Per-block true share T for one KV group: the mean over its heads of the softmax weight of each block."""
    n_tokens, hkv, d = seq.k.shape
    group = seq.q.shape[0] // hkv
    q = seq.q[kv_head * group : (kv_head + 1) * group].double()
    k = seq.k[:, kv_head, :].double()
    p = torch.softmax((q @ k.T) / d**0.5, dim=1).mean(dim=0)  # [L]
    n_blocks = -(-n_tokens // BLOCK)
    padded = torch.nn.functional.pad(p, (0, n_blocks * BLOCK - n_tokens))
    return padded.reshape(n_blocks, BLOCK).sum(dim=1).cpu().numpy()


def tiles(per_block: np.ndarray, tile_blocks: int) -> np.ndarray:
    """Sum a per-block vector over tiles of ``tile_blocks`` blocks; the last tile may be partial."""
    pad = -len(per_block) % tile_blocks
    return np.pad(per_block, (0, pad)).reshape(-1, tile_blocks).sum(axis=1)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a / a.sum() - b / b.sum()).sum())


def kept_ratio(true_share: np.ndarray, chosen: np.ndarray) -> float:
    k = max(1, round(TOP_FRACTION * len(true_share)))
    best = np.sort(true_share)[::-1][:k].sum()
    return float(true_share[np.argsort(chosen)[::-1][:k]].sum() / best)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    k = max(1, round(TOP_FRACTION * len(a)))
    sa, sb = set(np.argsort(a)[::-1][:k].tolist()), set(np.argsort(b)[::-1][:k].tolist())
    return len(sa & sb) / len(sa | sb)


class KernelRunner:
    """Calls vLLM's real unified-attention kernel and reads the pass's tile log-sum-exp records."""

    def __init__(self, split_kv: bool) -> None:
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention

        self.attention = unified_attention
        self.split_kv = split_kv
        self.descriptor: dict[str, int] | None = None
        self.pid = os.getpid()

    def _descriptor(self) -> dict[str, int]:
        """The shared statistic block the pass describes; every compiled variant publishes the same one."""
        if self.descriptor is None:
            entry = share_descriptor(self.pid)
            self.descriptor = {
                "lse": int(entry["tile_lse_dev"]),
                "flag": int(entry["stat_flag_dev"]),
                "table": int(entry["layer_table_dev"]),
                "layers": int(entry["tile_lse_layers"]),
                "group_rows": int(entry["tile_lse_group_rows"]),
                "rows": int(entry["max_rows"]),
                "width": int(entry["ncap"]),
                "tile_blocks": int(entry["attn_share_tile_blocks"]),
            }
        return self.descriptor

    def launch(self, seqs: list[Sequence], layer: int) -> np.ndarray:
        """One audit decode step as layer ``layer``; returns the attention share per [row, block]."""
        b = len(seqs)
        hq, d = seqs[0].q.shape
        hkv = seqs[0].k.shape[1]
        lens = [s.k.shape[0] for s in seqs]
        nblk = [-(-n // BLOCK) for n in lens]
        total = sum(nblk) + 1
        perm = torch.randperm(total - 1, device="cuda") + 1  # block 0 stays unused
        k_cache = torch.zeros(total, BLOCK, hkv, d, dtype=torch.float16, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        table = torch.zeros(b, max(nblk), dtype=torch.int32, device="cuda")
        cursor = 0
        for i, s in enumerate(seqs):
            ids = perm[cursor : cursor + nblk[i]]
            cursor += nblk[i]
            table[i, : nblk[i]] = ids.int()
            pad = nblk[i] * BLOCK - lens[i]
            kk = torch.nn.functional.pad(s.k, (0, 0, 0, 0, 0, pad)).reshape(nblk[i], BLOCK, hkv, d)
            vv = torch.nn.functional.pad(s.v, (0, 0, 0, 0, 0, pad)).reshape(nblk[i], BLOCK, hkv, d)
            k_cache[ids] = kk
            v_cache[ids] = vv
        q = torch.stack([s.q for s in seqs])  # [B, Hq, D]
        out = torch.zeros_like(q)
        cu = torch.arange(b + 1, dtype=torch.int32, device="cuda")
        used = torch.tensor(lens, dtype=torch.int32, device="cuda")
        threshold: int | None = None
        segments: int | None = None
        segm_output: torch.Tensor | None = None
        segm_max: torch.Tensor | None = None
        segm_expsum: torch.Tensor | None = None
        if self.split_kv:
            threshold, segments, pad_d = 16, 16, 128
            segm_output = torch.empty(threshold, hq, segments, pad_d, dtype=torch.float32, device="cuda")
            segm_max = torch.empty(threshold, hq, segments, dtype=torch.float32, device="cuda")
            segm_expsum = torch.empty(threshold, hq, segments, dtype=torch.float32, device="cuda")

        def call() -> None:
            self.attention(
                q=q,
                k=k_cache,
                v=v_cache,
                out=out,
                cu_seqlens_q=cu,
                max_seqlen_q=1,
                seqused_k=used,
                max_seqlen_k=max(lens),
                softmax_scale=d**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                seq_threshold_3D=threshold,
                num_par_softmax_segments=segments,
                softmax_segm_output=segm_output,
                softmax_segm_max=segm_max,
                softmax_segm_expsum=segm_expsum,
            )

        if self.descriptor is None:
            call()  # the first launch compiles the variant and publishes its descriptor; the flag is off
            torch.cuda.synchronize()
        desc = self._descriptor()
        layer_table = np.zeros(desc["layers"], dtype=np.uint64)
        layer_table[layer] = k_cache.data_ptr()  # the kernel finds its layer from its KV pointer
        h2d(desc["table"], layer_table)
        layer_cells = desc["rows"] * desc["width"] * desc["group_rows"]
        base = desc["lse"] + layer * layer_cells * 4
        for plane in (0, 1):  # the tile max and the tile sum: NaN until the launch writes them
            h2d(
                base + plane * desc["layers"] * layer_cells * 4,
                np.full(layer_cells, np.nan, dtype=np.float32),
            )
        h2d(desc["flag"], np.array([1], dtype=np.uint32))
        call()
        h2d(desc["flag"], np.array([0], dtype=np.uint32))
        lse = read_tile_lse(
            Copy(),
            base,
            desc["layers"],
            1,
            desc["rows"],
            desc["width"],
            desc["group_rows"],
            b,
            desc["tile_blocks"],
        )
        return normalize_tile_lse(lse, np.array(nblk), desc["tile_blocks"])


class Copy:
    """Device-to-host copies in the form ``read_tile_lse`` asks for."""

    def d2h_many(self, requests: list[tuple[int, int]]) -> list[bytes]:
        return [d2h_f32(address, size // 4).tobytes() for address, size in requests]


def run_case(name: str, launches: list[tuple[int, list[Sequence]]], split_kv: bool) -> None:
    runner = KernelRunner(split_kv)
    stats: dict[str, list[float]] = {k: [] for k in ("tv_T", "jac", "kept", "recency", "random")}
    rng = np.random.default_rng(0)
    for layer, seqs in launches:
        shares = runner.launch(seqs, layer)
        for row, s in enumerate(seqs):
            t = reference(s, layer % KV_HEADS)
            assert not np.isnan(shares[row]).all(), f"{name}: no statistic for row {row}"
            obs = np.nan_to_num(shares[row, : len(t)], nan=0.0)
            assert not np.nan_to_num(shares[row, len(t) :], nan=0.0).any(), (
                f"{name}: statistic outside the row"
            )
            # The pass records one value per tile, so the gate compares tiles, not blocks.
            obs, t = tiles(obs, runner.descriptor["tile_blocks"]), tiles(t, runner.descriptor["tile_blocks"])
            stats["tv_T"].append(tv(obs, t))
            stats["jac"].append(jaccard(obs, t))
            stats["kept"].append(kept_ratio(t, obs))
            stats["recency"].append(kept_ratio(t, np.arange(len(t), dtype=np.float64)))
            stats["random"].append(kept_ratio(t, rng.random(len(t))))

    def cell(key: str) -> str:
        v = stats[key]
        return f"{statistics.median(v):.3f}/{min(v) if key in ('jac', 'kept') else max(v):.3f}"

    print(
        f"  {name:<14} {'3D' if split_kv else '2D'}  n={len(stats['kept']):<4}"
        f" TV(obs,T) {cell('tv_T'):<13} jac25 {cell('jac'):<13} kept {cell('kept'):<13}"
        f" recency {statistics.median(stats['recency']):.3f} random {statistics.median(stats['random']):.3f}",
        flush=True,
    )
    worst = max(stats["tv_T"])
    assert worst < TV_LIMIT, f"{name}: total variation distance {worst:.2e} exceeds {TV_LIMIT:.0e}"
    assert min(stats["kept"]) > 1.0 - 1e-9, f"{name}: the observed top 25% is not the true top 25%"


def main() -> None:
    assert torch.cuda.is_available()
    install_pass()
    print("cells are median/worst over launches x requests (worst = max for TV, min for jac and kept)")
    real = real_sequences()
    controls = {"flat": synthetic_sequences("flat"), "ramp": synthetic_sequences("ramp")}
    for split_kv in (False, True):
        run_case("llama-8b real", [(layer, real[layer]) for layer in LAYERS], split_kv)
        for name, seqs in controls.items():
            run_case(name, [(0, seqs)], split_kv)
    print("PASS: tile log-sum-exp equals the true attention share on the real kernel")


if __name__ == "__main__":
    main()

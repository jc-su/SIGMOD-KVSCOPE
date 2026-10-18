"""The error bound behind the read policy, on real Llama-3.1-8B queries and values.

If a decode query drops keys that carry a fraction ``m`` of its softmax weight, the attention output changes by at
most ``2 * m * max_j ||v_j||``. This check measures, for real queries and values at layers across the model:

* whether the bound ever fails (it must not),
* how tight it is (the median of error over bound), and
* whether ranking blocks by measured attention drops less softmax weight, and so has less error, than the other choices at
  the same number of blocks: the newest blocks (recency) and a random choice.

The selections use the same unit as the system: 16-token blocks, a request's first block always kept, and
a share per block that is the mean over the query heads of one KV group.
"""

import statistics
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _attn_share_live_check import (  # ty: ignore[unresolved-import]  # sibling
    BLOCK,
    KV_HEADS,
    LAYERS,
    real_sequences,
)

FRACTIONS = (0.5, 0.25, 0.1)


def block_sums(per_key: torch.Tensor, n_blocks: int) -> torch.Tensor:
    pad = n_blocks * BLOCK - per_key.shape[-1]
    return torch.nn.functional.pad(per_key, (0, pad)).reshape(*per_key.shape[:-1], n_blocks, BLOCK).sum(-1)


def keep_mask(blocks: np.ndarray, n_tokens: int, n_blocks: int) -> torch.Tensor:
    kept = torch.zeros(n_blocks * BLOCK, dtype=torch.bool, device="cuda")
    for b in blocks:
        kept[b * BLOCK : (b + 1) * BLOCK] = True
    return kept[:n_tokens]


def main() -> None:
    rng = np.random.default_rng(0)
    real = real_sequences()
    stats: dict[tuple[float, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    violations = 0
    cases = 0
    for layer in LAYERS[::4]:
        for seq in real[layer]:
            n_tokens, _, d = seq.k.shape
            n_blocks = -(-n_tokens // BLOCK)
            group = seq.q.shape[0] // KV_HEADS
            for kv in range(KV_HEADS):
                q = seq.q[kv * group : (kv + 1) * group].double()
                keys, values = seq.k[:, kv].double(), seq.v[:, kv].double()
                p = torch.softmax(q @ keys.T / d**0.5, dim=1)  # [group, n]
                full = p @ values
                vmax = values.norm(dim=1).max()
                share = block_sums(p, n_blocks).mean(0).cpu().numpy()  # the controller's score of each block
                for fraction in FRACTIONS:
                    count = max(2, int(np.ceil(fraction * n_blocks)))
                    ranked = np.argsort(-share, kind="stable")
                    choices = {
                        "ranked by attention": np.unique(np.concatenate([[0], ranked[: count - 1]])),
                        "newest": np.unique(np.concatenate([[0], np.arange(n_blocks - count + 1, n_blocks)])),
                        "random": np.unique(
                            np.concatenate([[0], rng.choice(np.arange(1, n_blocks), count - 1, False)])
                        ),
                    }
                    for name, blocks in choices.items():
                        kept = keep_mask(blocks, n_tokens, n_blocks)
                        restricted = torch.softmax((q @ keys.T / d**0.5)[:, kept], dim=1) @ values[kept]
                        error = (restricted - full).norm(dim=1)
                        dropped = 1.0 - p[:, kept].sum(1)
                        bound = 2.0 * dropped * vmax
                        cases += int(error.numel())
                        violations += int((error > bound + 1e-9).sum())
                        cell = stats[(fraction, name)]
                        cell["dropped"] += dropped.cpu().tolist()
                        cell["error"] += error.cpu().tolist()
                        cell["tightness"] += (error / bound.clamp_min(1e-12)).cpu().tolist()
    print(f"cases {cases}, bound violations {violations}")
    print(
        f"{'blocks kept':<12} {'selection':<20} {'weight dropped':>14} {'output error':>13} {'error/bound':>12}"
    )
    for (fraction, name), cell in sorted(stats.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        print(
            f"{fraction:<12.2f} {name:<20} {statistics.mean(cell['dropped']):>13.4f}"
            f" {statistics.mean(cell['error']):>13.4f} {statistics.median(cell['tightness']):>12.3f}"
        )
    assert violations == 0, "the bound failed on a real query"
    for fraction in FRACTIONS:
        ranked = statistics.mean(stats[(fraction, "ranked by attention")]["error"])
        for other in ("newest", "random"):
            assert ranked < statistics.mean(stats[(fraction, other)]["error"]), (
                f"ranking by attention did not keep the error lower than {other} at {fraction}"
            )
    print("PASS: the bound holds on every real query, and ranking by attention keeps the error lowest")


if __name__ == "__main__":
    main()

"""Attention shares from tile log-sum-exp records: exactness and the handling of missing data."""

import numpy as np
import pytest

from observe.tile_shares import normalize_tile_lse as normalize_rows
from observe.tile_shares import read_tile_lse, read_tile_rows


def normalize_tile_lse(lse: np.ndarray, tile_blocks: int = 1) -> np.ndarray:
    """Normalize with every row as long as the table is wide."""
    full = np.full(lse.shape[1], lse.shape[2])
    return normalize_rows(lse, full, tile_blocks)


def block_lse(scores: np.ndarray, sizes: list[int]) -> np.ndarray:
    """The log-sum-exp of the scores in each block."""
    out, start = [], 0
    for n in sizes:
        out.append(np.log(np.exp(scores[start : start + n]).sum()))
        start += n
    return np.array(out)


def true_share(scores: np.ndarray, sizes: list[int]) -> np.ndarray:
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return np.array(
        [weights[s : s + n].sum() for s, n in zip(np.cumsum([0, *sizes[:-1]]), sizes, strict=True)]
    )


def test_shares_equal_the_softmax_mass_of_each_block():
    rng = np.random.default_rng(0)
    layers, rows, query_rows = 3, 2, 4
    sizes = [16, 16, 16, 7]
    lse = np.zeros((layers, rows, len(sizes), query_rows), dtype=np.float32)
    expected = np.zeros((rows, len(sizes)))
    for layer in range(layers):
        for row in range(rows):
            for q in range(query_rows):
                scores = rng.normal(0, 3, sum(sizes))
                lse[layer, row, :, q] = block_lse(scores, sizes)
                expected[row] += true_share(scores, sizes) / (layers * query_rows)
    assert normalize_tile_lse(lse) == pytest.approx(expected, abs=1e-5)


def test_each_row_sums_to_one():
    rng = np.random.default_rng(1)
    lse = rng.normal(0, 4, (4, 3, 10, 2)).astype(np.float32)
    assert normalize_tile_lse(lse).sum(axis=1) == pytest.approx(np.ones(3))


def test_a_constant_offset_in_one_query_row_changes_nothing():
    rng = np.random.default_rng(2)
    lse = rng.normal(0, 2, (2, 2, 6, 3)).astype(np.float32)
    shifted = lse.copy()
    shifted[1, 0, :, 2] += 40.0  # a different running max leaves the shares of that row unchanged
    assert normalize_tile_lse(shifted) == pytest.approx(normalize_tile_lse(lse), abs=1e-6)


def test_unwritten_cells_and_empty_blocks_carry_no_mass():
    lse = np.full((1, 2, 4, 2), np.nan, dtype=np.float32)
    lse[0, 0, :, 0] = [0.0, np.log(3.0), -np.inf, np.nan]
    result = normalize_tile_lse(lse)
    assert result[0] == pytest.approx([0.25, 0.75, 0.0, 0.0])
    assert np.isnan(result[1]).all()


def test_a_query_row_with_no_data_does_not_dilute_the_mean():
    lse = np.full((1, 1, 3, 2), np.nan, dtype=np.float32)
    lse[0, 0, :, 0] = [np.log(1.0), np.log(1.0), np.log(2.0)]
    assert normalize_tile_lse(lse)[0] == pytest.approx([0.25, 0.25, 0.5])


def test_the_table_is_read_per_layer_for_the_live_rows_only():
    table_layers, layers, rows, width, group, tile_blocks = 4, 3, 5, 4, 2, 2
    cells = table_layers * rows * width * group
    rng = np.random.default_rng(3)
    m = rng.normal(0, 3, cells).astype(np.float32)
    total = rng.uniform(0.5, 9, cells).astype(np.float32)
    total[7] = 0.0  # a tile with no keys
    table = np.concatenate([m, total])  # plane 0 the tile max, plane 1 the tile sum

    class FakeCuda:
        def d2h_many(self, requests):
            return [table.tobytes()[a : a + n] for a, n in requests]

    got = read_tile_lse(
        FakeCuda(), 0, table_layers, layers, rows, width, group, live_rows=2, tile_blocks=tile_blocks
    )
    with np.errstate(divide="ignore"):
        u = m + np.log(total) - np.log(tile_blocks)
    expected = u.reshape(table_layers, rows, width, group)[:layers, :2]
    assert got.shape == (layers, 2, width, group)
    assert got == pytest.approx(expected, rel=1e-6)
    assert got[0, 0, 3, 1] == -np.inf  # cell 7: l = 0 carries no weight


def test_reading_chosen_rows_copies_only_them_and_equals_the_full_read():
    table_layers, layers, rows, width, group = 3, 2, 6, 4, 2
    cells = table_layers * rows * width * group
    rng = np.random.default_rng(5)
    table = np.concatenate([rng.normal(0, 3, cells), rng.uniform(0.5, 9, cells)]).astype(np.float32)
    copied: list[int] = []

    class FakeCuda:
        def d2h_many(self, requests):
            copied.extend(n for _a, n in requests)
            return [table.tobytes()[a : a + n] for a, n in requests]

    full = read_tile_lse(
        FakeCuda(), 0, table_layers, layers, rows, width, group, live_rows=rows, tile_blocks=2
    )
    copied.clear()
    chosen = np.array([1, 2, 5])
    got = read_tile_rows(FakeCuda(), 0, table_layers, layers, rows, width, group, chosen, tile_blocks=2)
    assert got == pytest.approx(full[:, chosen], rel=1e-6)
    assert len(copied) == 2 * layers * 2  # rows 1-2 and row 5 are two runs, per layer and plane
    assert sum(copied) == 2 * layers * len(chosen) * width * group * 4
    with pytest.raises(ValueError):
        read_tile_rows(FakeCuda(), 0, table_layers, layers, rows, width, group, np.array([2, 1]), 2)


def test_an_unwritten_cell_reads_as_nan():
    table = np.full(2 * 4, np.nan, dtype=np.float32)

    class FakeCuda:
        def d2h_many(self, requests):
            return [table.tobytes()[a : a + n] for a, n in requests]

    assert np.isnan(read_tile_lse(FakeCuda(), 0, 1, 1, 1, 2, 2, live_rows=1, tile_blocks=1)).all()


def test_a_partial_last_tile_gets_its_whole_mass_and_the_cells_beyond_the_row_are_dropped():
    # Three tiles of two blocks with masses 1, 2, 5. The row has five blocks, so the last tile has one.
    masses = np.array([1.0, 2.0, 5.0])
    cells = np.repeat(np.log(masses / 2), 2).astype(np.float32)  # what the kernel stores, per block
    lse = cells.reshape(1, 1, 6, 1)
    got = normalize_rows(lse, np.array([5]), tile_blocks=2)
    assert got[0] == pytest.approx(np.array([0.5, 0.5, 1.0, 1.0, 5.0, 0.0]) / 8.0)


def test_a_complete_last_tile_is_left_alone():
    cells = np.repeat(np.log(np.array([3.0, 1.0]) / 2), 2).astype(np.float32).reshape(1, 1, 4, 1)
    got = normalize_rows(cells, np.array([4]), tile_blocks=2)
    assert got[0] == pytest.approx([0.375, 0.375, 0.125, 0.125])


def test_max_aggregation_keeps_a_block_that_only_one_head_needs():
    # Two query rows. The first attends block 0, the second attends block 1; block 2 is ignored by both.
    lse = np.log(np.array([[9.0, 1.0, 1.0], [1.0, 9.0, 1.0]])).T.astype(np.float32).reshape(1, 1, 3, 2)
    mean = normalize_tile_lse(lse)
    maxed = normalize_rows(lse, np.array([3]), 1, "max")
    assert mean[0] == pytest.approx(np.array([10.0, 10.0, 2.0]) / 22.0, abs=1e-6)
    assert maxed[0] == pytest.approx(np.array([9.0, 9.0, 1.0]) / 19.0, abs=1e-6)
    assert maxed.sum() == pytest.approx(1.0)


def test_an_unknown_aggregate_is_rejected():
    with pytest.raises(ValueError):
        normalize_rows(np.zeros((1, 1, 2, 1)), np.array([2]), 1, "median")  # ty: ignore[invalid-argument-type]


def test_a_wrong_rank_is_rejected():
    with pytest.raises(ValueError):
        normalize_rows(np.zeros((2, 3, 4)), np.array([1, 1, 1]), 1)


def test_a_wrong_row_length_vector_is_rejected():
    with pytest.raises(ValueError):
        normalize_rows(np.zeros((1, 3, 4, 1)), np.array([4, 4]), 1)

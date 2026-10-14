import torch

from src.evaluation.qwen2_token_pruning import (
    Qwen2FirstLayerTokenPruner,
    select_keep_indices,
)


def _select(scores, mask, visual, ratio=0.1, threshold=16, minimum=2):
    return select_keep_indices(
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(visual, dtype=torch.bool),
        keep_ratio=ratio,
        short_text_threshold=threshold,
        min_keep_tokens=minimum,
    )


def test_long_input_keeps_exact_ceiling_ratio_and_forces_last_token():
    scores = [[float(index) for index in range(20)]]
    # Give the required final token deliberately low importance: it must still
    # survive, with the remaining slot assigned to the strongest history token.
    scores[0][-1] = -100.0
    selected, bypassed = _select(
        scores,
        [[1] * 20],
        [False],
        ratio=0.1,
        threshold=4,
        minimum=2,
    )
    assert selected[0].tolist() == [18, 19]
    assert bypassed == [False]


def test_selection_excludes_left_padding_and_restores_original_order():
    selected, _ = _select(
        [[1000.0, 999.0, 0.1, 0.9, 0.8, -1.0]],
        [[0, 0, 1, 1, 1, 1]],
        [True],
        ratio=0.75,
        threshold=16,
        minimum=2,
    )
    # Three of four active tokens: top two historical positions plus final.
    assert selected[0].tolist() == [3, 4, 5]


def test_short_pure_text_bypasses_but_short_visual_input_is_pruned():
    scores = [[0.1, 0.8, 0.4, 0.2], [0.1, 0.8, 0.4, 0.2]]
    mask = [[1, 1, 1, 1], [1, 1, 1, 1]]
    selected, bypassed = _select(
        scores,
        mask,
        [False, True],
        ratio=0.25,
        threshold=4,
        minimum=2,
    )
    assert selected[0].tolist() == [0, 1, 2, 3]
    assert selected[1].tolist() == [1, 3]
    assert bypassed == [True, False]


def test_invalid_or_empty_inputs_are_rejected():
    try:
        _select([[0.0]], [[0]], [False])
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty row was not rejected")

    try:
        _select([[0.0]], [[1]], [False], ratio=0.0)
    except ValueError as error:
        assert "keep_ratio" in str(error)
    else:
        raise AssertionError("zero keep ratio was not rejected")


def test_repack_left_pads_and_preserves_original_positions():
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1)
    position_ids = torch.arange(6).view(1, 1, 6).expand(3, 2, 6).clone()
    cos = torch.arange(3 * 2 * 6 * 2, dtype=torch.float32).reshape(3, 2, 6, 2)
    sin = -cos
    selections = [torch.tensor([1, 4, 5]), torch.tensor([2, 5])]
    new_hidden, new_mask, new_positions, (new_cos, new_sin) = (
        Qwen2FirstLayerTokenPruner._repack(
            hidden,
            position_ids,
            (cos, sin),
            selections,
        )
    )
    assert new_mask.tolist() == [[1, 1, 1], [0, 1, 1]]
    assert new_hidden[:, -1, 0].tolist() == [5.0, 11.0]
    assert new_positions[0, 0].tolist() == [1, 4, 5]
    assert new_positions[0, 1].tolist() == [0, 2, 5]
    torch.testing.assert_close(new_cos[:, 0], cos[:, 0, [1, 4, 5]])
    torch.testing.assert_close(new_sin[:, 1, 1:], sin[:, 1, [2, 5]])

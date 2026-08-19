"""Regression tests for the bounded 7.58M micro data pilot."""

from scripts.training.diag_micro_data_ab import _valid_first_token_position


def test_first_token_metric_skips_answer_outside_truncated_window() -> None:
    assert _valid_first_token_position(answer_start=156, seq_len=128) is None
    assert _valid_first_token_position(answer_start=127, seq_len=128) == 126

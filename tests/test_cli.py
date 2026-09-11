import pytest

from miracle_ad.cli.train import _parse_args


def test_supervised_cli_uses_reported_learning_rates(tmp_path):
    _, args = _parse_args(
        [
            "--datasets",
            "ncmmsc",
            "--data-root",
            str(tmp_path),
        ]
    )
    assert args.learning_rate == pytest.approx(3e-2)
    assert args.final_learning_rate == pytest.approx(3e-5)
    assert args.epochs == 100
    assert args.patience == 15
    assert args.batch_size == 1
    assert args.accumulation_steps == 4

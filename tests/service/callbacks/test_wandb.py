"""Tests for callbacks.wandb module."""

from unittest.mock import MagicMock, patch

import pytest

# Import directly from the module to avoid service/__init__.py imports
from callbacks.wandb import WandBCallback


class TestWandBCallback:
    """Tests for WandBCallback class."""

    def test_initializes_with_project_name(self):
        """Test that callback initializes with project name."""
        callback = WandBCallback(project="test-project")
        assert callback.project == "test-project"

    def test_initializes_with_optional_run_name(self):
        """Test that callback initializes with optional run name."""
        callback = WandBCallback(project="test-project", run_name="test-run")
        assert callback.run_name == "test-run"

    def test_initializes_with_optional_config(self):
        """Test that callback initializes with optional config."""
        config = {"lr": 0.001, "batch_size": 32}
        callback = WandBCallback(project="test-project", config=config)
        assert callback.run is None  # wandb is not available in test

    @patch("callbacks.wandb.wandb", None)
    def test_handles_missing_wandb(self, caplog):
        """Test that callback handles missing W&B gracefully."""
        with patch("callbacks.wandb.wandb", None):
            callback = WandBCallback(project="test-project")
            assert callback.run is None

    @patch("callbacks.wandb.wandb")
    def test_initializes_wandb_run_when_available(self, mock_wandb):
        """Test that callback initializes wandb run when available."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.run.url = "https://wandb.ai/test/run"

        callback = WandBCallback(project="test-project", run_name="test-run", config={"lr": 0.001})

        assert callback.run is mock_run
        mock_wandb.init.assert_called_once_with(
            project="test-project",
            name="test-run",
            config={"lr": 0.001},
        )

    @patch("callbacks.wandb.wandb")
    def test_on_epoch_end_logs_metrics(self, mock_wandb):
        """Test that on_epoch_end logs metrics to wandb."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")

        metrics = {
            "epoch": 5,
            "loss": 0.45,
            "train_loss": 0.50,
            "val_loss": 0.40,
            "mAP50": 0.82,
            "mAP50-95": 0.65,
            "precision": 0.88,
            "recall": 0.75,
            "lr": 0.001,
        }
        callback.on_epoch_end(metrics)

        expected_log = {
            "loss": 0.45,
            "train_loss": 0.50,
            "val_loss": 0.40,
            "mAP50": 0.82,
            "mAP50-95": 0.65,
            "precision": 0.88,
            "recall": 0.75,
            "lr": 0.001,
        }
        mock_wandb.log.assert_called_once_with(expected_log, step=5)

    @patch("callbacks.wandb.wandb")
    def test_on_epoch_end_skips_missing_metrics(self, mock_wandb):
        """Test that missing metrics are skipped without error."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")

        # Only provide a subset of metrics
        metrics = {"epoch": 1, "loss": 0.5}
        callback.on_epoch_end(metrics)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=1)

    @patch("callbacks.wandb.wandb")
    def test_on_epoch_end_handles_none_run(self, mock_wandb):
        """Test that on_epoch_end handles None run gracefully."""
        callback = WandBCallback(project="test-project")
        callback.run = None

        # Should not raise any errors
        callback.on_epoch_end({"epoch": 1, "loss": 0.5})

        mock_wandb.log.assert_not_called()

    @patch("callbacks.wandb.wandb", None)
    def test_on_epoch_end_handles_none_wandb(self):
        """Test that on_epoch_end handles None wandb gracefully."""
        with patch("callbacks.wandb.wandb", None):
            callback = WandBCallback(project="test-project")

            # Should not raise any errors
            callback.on_epoch_end({"epoch": 1, "loss": 0.5})

    @patch("callbacks.wandb.wandb")
    def test_on_epoch_end_defaults_epoch_to_none(self, mock_wandb):
        """Test that epoch defaults to None when not in metrics."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")

        metrics = {"loss": 0.5}
        callback.on_epoch_end(metrics)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=None)

    @patch("callbacks.wandb.wandb")
    def test_on_train_end_finishes_run(self, mock_wandb):
        """Test that on_train_end finishes the wandb run."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")
        callback.on_train_end({})

        mock_run.finish.assert_called_once()

    @patch("callbacks.wandb.wandb")
    def test_on_train_end_handles_none_run(self, mock_wandb):
        """Test that on_train_end handles None run gracefully."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")
        callback.run = None
        callback.on_train_end({})

        # Should not raise any errors
        mock_run.finish.assert_not_called()

    @patch("callbacks.wandb.wandb")
    def test_logs_all_recognized_metrics(self, mock_wandb):
        """Test that all recognized metrics are logged."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run

        callback = WandBCallback(project="test-project")

        metrics = {
            "epoch": 1,
            "loss": 0.5,
            "train_loss": 0.6,
            "val_loss": 0.4,
            "mAP50": 0.85,
            "mAP50-95": 0.70,
            "precision": 0.90,
            "recall": 0.80,
            "lr": 0.001,
        }
        callback.on_epoch_end(metrics)

        logged = mock_wandb.log.call_args[0][0]
        for key in ["loss", "train_loss", "val_loss", "mAP50", "mAP50-95", "precision", "recall", "lr"]:
            assert key in logged

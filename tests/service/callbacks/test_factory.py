"""Tests for callbacks.factory module."""

from unittest.mock import MagicMock, patch

import pytest

# Import directly from the modules to avoid service/__init__.py imports
from callbacks.base import TrainingCallback
from callbacks.manager import CallbackManager
from callbacks.early_stopping import EarlyStoppingCallback
from callbacks.metrics_logger import MetricsLoggerCallback
from callbacks.tensorboard import TensorBoardCallback
from callbacks.wandb import WandBCallback
from callbacks.factory import create_default_callbacks


class TestCreateDefaultCallbacks:
    """Tests for create_default_callbacks factory function."""

    def test_returns_callback_manager(self):
        """Test that factory returns a CallbackManager instance."""
        manager = create_default_callbacks(output_dir="/tmp/test")
        assert isinstance(manager, CallbackManager)

    def test_always_adds_metrics_logger(self):
        """Test that MetricsLoggerCallback is always added."""
        manager = create_default_callbacks(output_dir="/tmp/test")

        assert any(isinstance(cb, MetricsLoggerCallback) for cb in manager.callbacks)

    def test_metrics_logger_receives_output_dir(self):
        """Test that MetricsLoggerCallback receives the output_dir."""
        manager = create_default_callbacks(output_dir="/tmp/test")

        for cb in manager.callbacks:
            if isinstance(cb, MetricsLoggerCallback):
                assert cb.output_dir == "/tmp/test"

    def test_adds_tensorboard_when_enabled(self):
        """Test that TensorBoardCallback is added when enabled."""
        manager = create_default_callbacks(output_dir="/tmp/test", tensorboard=True)

        assert any(isinstance(cb, TensorBoardCallback) for cb in manager.callbacks)

    def test_tensorboard_receives_log_dir(self):
        """Test that TensorBoardCallback receives the output_dir as log_dir."""
        manager = create_default_callbacks(output_dir="/tmp/test", tensorboard=True)

        for cb in manager.callbacks:
            if isinstance(cb, TensorBoardCallback):
                assert cb.log_dir == "/tmp/test"

    def test_skips_tensorboard_when_disabled(self):
        """Test that TensorBoardCallback is not added when disabled."""
        manager = create_default_callbacks(output_dir="/tmp/test", tensorboard=False)

        assert not any(isinstance(cb, TensorBoardCallback) for cb in manager.callbacks)

    @patch("callbacks.wandb.wandb")
    def test_adds_wandb_when_project_set(self, mock_wandb):
        """Test that WandBCallback is added when project is set."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.run.url = "https://wandb.ai/test/run"

        manager = create_default_callbacks(output_dir="/tmp/test", wandb_project="test-project")

        assert any(isinstance(cb, WandBCallback) for cb in manager.callbacks)

    @patch("callbacks.wandb.wandb")
    def test_wandb_receives_project_name(self, mock_wandb):
        """Test that WandBCallback receives the project name."""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.run.url = "https://wandb.ai/test/run"

        manager = create_default_callbacks(output_dir="/tmp/test", wandb_project="my-project")

        for cb in manager.callbacks:
            if isinstance(cb, WandBCallback):
                assert cb.project == "my-project"

    def test_skips_wandb_when_project_not_set(self):
        """Test that WandBCallback is not added when project is None."""
        manager = create_default_callbacks(output_dir="/tmp/test", wandb_project=None)

        assert not any(isinstance(cb, WandBCallback) for cb in manager.callbacks)

    def test_adds_early_stopping_when_enabled(self):
        """Test that EarlyStoppingCallback is added when enabled."""
        manager = create_default_callbacks(
            output_dir="/tmp/test", early_stopping=True, early_stopping_patience=15
        )

        assert any(isinstance(cb, EarlyStoppingCallback) for cb in manager.callbacks)

    def test_early_stopping_receives_patience(self):
        """Test that EarlyStoppingCallback receives the patience value."""
        manager = create_default_callbacks(
            output_dir="/tmp/test", early_stopping=True, early_stopping_patience=20
        )

        for cb in manager.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                assert cb.patience == 20

    def test_early_stopping_defaults_to_patience_10(self):
        """Test that EarlyStoppingCallback defaults to patience 10."""
        manager = create_default_callbacks(output_dir="/tmp/test", early_stopping=True)

        for cb in manager.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                assert cb.patience == 10

    def test_skips_early_stopping_when_disabled(self):
        """Test that EarlyStoppingCallback is not added when disabled."""
        manager = create_default_callbacks(output_dir="/tmp/test", early_stopping=False)

        assert not any(isinstance(cb, EarlyStoppingCallback) for cb in manager.callbacks)

    def test_all_callbacks_can_be_enabled(self):
        """Test that all callbacks can be enabled simultaneously."""
        with patch("callbacks.wandb.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.run.url = "https://wandb.ai/test/run"

            manager = create_default_callbacks(
                output_dir="/tmp/test",
                tensorboard=True,
                wandb_project="test-project",
                early_stopping=True,
                early_stopping_patience=5,
            )

            # Should have 4 callbacks: metrics logger, tensorboard, wandb, early stopping
            assert len(manager.callbacks) == 4
            assert any(isinstance(cb, MetricsLoggerCallback) for cb in manager.callbacks)
            assert any(isinstance(cb, TensorBoardCallback) for cb in manager.callbacks)
            assert any(isinstance(cb, WandBCallback) for cb in manager.callbacks)
            assert any(isinstance(cb, EarlyStoppingCallback) for cb in manager.callbacks)

    def test_only_metrics_logger_when_all_disabled(self):
        """Test that only MetricsLoggerCallback is added when all are disabled."""
        manager = create_default_callbacks(
            output_dir="/tmp/test",
            tensorboard=False,
            wandb_project=None,
            early_stopping=False,
        )

        assert len(manager.callbacks) == 1
        assert isinstance(manager.callbacks[0], MetricsLoggerCallback)

    def test_returns_manager_for_chaining(self):
        """Test that returned manager can be used for chaining."""
        manager = create_default_callbacks(output_dir="/tmp/test")

        # Should be able to chain
        extra_cb = MagicMock()
        result = manager.add(extra_cb)

        assert result is manager

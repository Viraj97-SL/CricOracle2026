"""Tests for prediction models."""

import pytest
import numpy as np
import pandas as pd
from src.models.win_predictor import WinPredictor


class TestWinPredictor:
    """Tests for the match outcome predictor."""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "team_batting_power": np.random.uniform(100, 200, n),
            "team_bowling_power": np.random.uniform(5, 10, n),
            "venue_avg_score": np.random.uniform(140, 180, n),
            "h2h_win_rate": np.random.uniform(0.3, 0.7, n),
            "toss_winner_is_team1": np.random.randint(0, 2, n),
        })
        y = pd.Series(np.random.randint(0, 2, n))
        return X, y

    def test_train_returns_metrics(self, dummy_data):
        """Training should return evaluation metrics."""
        X, y = dummy_data
        predictor = WinPredictor()
        metrics = predictor.train(X[:150], y[:150], X[150:], y[150:])
        assert "log_loss" in metrics
        assert "auc_roc" in metrics
        assert "accuracy" in metrics

    def test_predict_proba_bounds(self, dummy_data):
        """Predictions must be between 0 and 1."""
        X, y = dummy_data
        predictor = WinPredictor()
        predictor.train(X[:150], y[:150])
        probs = predictor.predict_proba(X[150:])
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_binary(self, dummy_data):
        """Binary predictions must be 0 or 1."""
        X, y = dummy_data
        predictor = WinPredictor()
        predictor.train(X[:150], y[:150])
        preds = predictor.predict(X[150:])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_feature_importance(self, dummy_data):
        """Feature importance should list all features."""
        X, y = dummy_data
        predictor = WinPredictor()
        predictor.train(X, y)
        imp = predictor.get_feature_importance()
        assert len(imp) <= len(X.columns)
        assert "feature" in imp.columns
        assert "importance" in imp.columns

    def test_save_and_load(self, dummy_data, tmp_path):
        """Model should be saveable and loadable."""
        X, y = dummy_data
        predictor = WinPredictor()
        predictor.train(X, y)
        save_path = tmp_path / "test_model.pkl"
        predictor.save(save_path)

        # Load into new instance
        loaded = WinPredictor()
        loaded.load(save_path)
        assert loaded.is_trained

        # Predictions should match
        orig_preds = predictor.predict_proba(X[:5])
        loaded_preds = loaded.predict_proba(X[:5])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_untrained_predict_raises(self):
        """Predicting without training should raise error."""
        predictor = WinPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict_proba(pd.DataFrame({"a": [1]}))

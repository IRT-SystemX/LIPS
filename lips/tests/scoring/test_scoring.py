import unittest
from unittest.mock import MagicMock
from lips.scoring import Scoring


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.get_option.side_effect = lambda key: {
            "thresholds": {
                "a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05]},
                "spearman_correlation_drag": {"comparison_type": "maximize", "thresholds": [0.8, 0.9]},
                "inference_time": {"comparison_type": "minimize", "thresholds": [500, 1000]},
            },
            "valuebycolor": {"green": 2, "orange": 1, "red": 0},
        }[key]
        self.scoring = Scoring(config=self.mock_config)

    def test_colorize_metric_value_minimize(self):
        self.assertEqual(self.scoring.colorize_metric_value(0.01, "minimize", [0.02, 0.05]), "green")
        self.assertEqual(self.scoring.colorize_metric_value(0.03, "minimize", [0.02, 0.05]), "orange")
        self.assertEqual(self.scoring.colorize_metric_value(0.06, "minimize", [0.02, 0.05]), "red")

    def test_colorize_metric_value_maximize(self):
        self.assertEqual(self.scoring.colorize_metric_value(0.95, "maximize", [0.8, 0.9]), "green")
        self.assertEqual(self.scoring.colorize_metric_value(0.85, "maximize", [0.8, 0.9]), "orange")
        self.assertEqual(self.scoring.colorize_metric_value(0.75, "maximize", [0.8, 0.9]), "red")

    def test_calculate_score_color(self):
        metrics = {"a_or": 0.03, "spearman_correlation_drag": 0.85, "inference_time": 600}
        expected_output = {"a_or": "orange", "spearman_correlation_drag": "orange", "inference_time": "orange"}
        self.assertEqual(self.scoring.calculate_score_color(metrics), expected_output)

    def test_invalid_comparison_type(self):
        with self.assertRaises(ValueError):
            self.scoring.colorize_metric_value(0.5, "invalid_type", [0.1, 0.2])

    def test_validate_thresholds_config_missing(self):
        self.mock_config.get_option.side_effect = lambda key: None if key == "thresholds" else {"green": 2, "orange": 1,
                                                                                                "red": 0}
        with self.assertRaises(ValueError):
            Scoring(config=self.mock_config)

    def test_validate_thresholds_config_mismatch(self):
        self.mock_config.get_option.side_effect = lambda key: {
            "thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02]}},
            "valuebycolor": {"green": 2, "orange": 1, "red": 0},
        }[key]
        with self.assertRaises(ValueError):
            Scoring(config=self.mock_config)


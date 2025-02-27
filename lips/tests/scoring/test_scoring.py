import unittest
from unittest.mock import MagicMock

from lips.scoring import Scoring


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.get_option.side_effect = lambda key: {
            "thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05]},
                           "spearman_correlation_drag": {"comparison_type": "maximize", "thresholds": [0.8, 0.9]},
                           "inference_time": {"comparison_type": "minimize", "thresholds": [500, 1000]},
                           "reference_mean_simulation_time": {"comparison_type": "ratio", "thresholds": [1500]},
                           "max_speed_ratio_allowed": {"comparison_type": "ratio", "thresholds": [10000]}},
            "valuebycolor": {"green": 2, "orange": 1, "red": 0},
            "coefficients": {"ML": {"value": 0.3}, "OOD": {"value": 0.3}, "Physics": {"value": 0.3},
                             "Speed": {"value": 0.1}}}[key]
        self.scoring = Scoring(config=self.mock_config)

    def test_colorize_metrics(self):
        metrics = {"ML": {"a_or": 0.01, "spearman_correlation_drag": 0.95},
                   "OOD": {"a_or": 0.03, "inference_time": 400},
                   "Physics": {"a_or": 0.06, "spearman_correlation_drag": 0.8}, "Speed": {"inference_time": 600}}
        expected_colorized_metrics = {"ML": {"a_or": "green", "spearman_correlation_drag": "green"},
                                      "OOD": {"a_or": "orange", "inference_time": "green"},
                                      "Physics": {"a_or": "red", "spearman_correlation_drag": "red"},
                                      "Speed": {"inference_time": "orange"}}
        colorized_metrics = self.scoring.colorize_metrics(metrics)
        self.assertEqual(colorized_metrics, expected_colorized_metrics)

    def test__colorize_metric_value(self):
        self.assertEqual(self.scoring._colorize_metric_value("a_or", 0.01), "green")
        self.assertEqual(self.scoring._colorize_metric_value("a_or", 0.04), "orange")
        self.assertEqual(self.scoring._colorize_metric_value("a_or", 0.06), "red")

        self.assertEqual(self.scoring._colorize_metric_value("spearman_correlation_drag", 0.92), "green")
        self.assertEqual(self.scoring._colorize_metric_value("spearman_correlation_drag", 0.85), "orange")
        self.assertEqual(self.scoring._colorize_metric_value("spearman_correlation_drag", 0.7), "red")

        self.assertEqual(self.scoring._colorize_metric_value("inference_time", 450), "green")
        self.assertEqual(self.scoring._colorize_metric_value("inference_time", 750), "orange")
        self.assertEqual(self.scoring._colorize_metric_value("inference_time", 1100), "red")

    def test__colorize_metric_value_invalid_comparison_type(self):
        self.mock_config.get_option.side_effect = lambda key: \
            {"thresholds": {"a_or": {"comparison_type": "invalid", "thresholds": [0.02, 0.05]}},
             "valuebycolor": {"green": 2, "orange": 1, "red": 0}, "coefficients": {}}[
                key]  # empty coefficients to avoid other errors.
        self.scoring = Scoring(config=self.mock_config)  # recreate scoring object with invalid config.
        with self.assertRaises(ValueError) as context:
            self.scoring._colorize_metric_value("a_or", 0.01)
        self.assertIn("Invalid comparison type", str(context.exception))

    def test__validate_configuration_missing_thresholds(self):
        self.mock_config.get_option.side_effect = lambda key: \
            {"thresholds": None, "valuebycolor": {"green": 2, "orange": 1, "red": 0}, "coefficients": {}}[key]
        with self.assertRaises(ValueError) as context:
            Scoring(config=self.mock_config)
        self.assertIn("Thresholds configuration is missing", str(context.exception))

    def test__validate_configuration_missing_valuebycolor(self):
        self.mock_config.get_option.side_effect = lambda key: \
            {"thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05]}}, "valuebycolor": None,
             "coefficients": {}}[key]
        with self.assertRaises(ValueError) as context:
            Scoring(config=self.mock_config)
        self.assertIn("Value by color configuration is missing", str(context.exception))

    def test__validate_configuration_invalid_threshold_data(self):
        self.mock_config.get_option.side_effect = lambda key: \
            {"thresholds": {"a_or": {"thresholds": [0.02, 0.05]}},  # Missing comparison_type
             "valuebycolor": {"green": 2, "orange": 1, "red": 0}, "coefficients": {}}[key]
        with self.assertRaises(ValueError) as context:
            Scoring(config=self.mock_config)
        self.assertIn(
            "Invalid thresholds data for metric 'a_or'. Must be a dict with 'thresholds' and 'comparison_type' keys.",
            str(context.exception))

    def test__validate_configuration_invalid_threshold_count(self):
        self.mock_config.get_option.side_effect = lambda key: \
            {"thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05, 0.1]}},
             # Too many thresholds
             "valuebycolor": {"green": 2, "orange": 1, "red": 0}, "coefficients": {}}[key]
        with self.assertRaises(ValueError) as context:
            Scoring(config=self.mock_config)
        self.assertIn("Metric 'a_or': Thresholds count must be 2 (length of ValueByColor - 1).", str(context.exception))

    def test__calculate_leaf_score(self):
        colors = ["green", "orange", "red"]
        expected_score = (2 + 1 + 0) / (3 * 2)  # (sum of values) / (number of colors * max value)
        self.assertEqual(self.scoring._calculate_leaf_score(colors), expected_score)

    def test_calculate_sub_scores_leaf(self):
        node = {"a_or": "green", "spearman_correlation_drag": "orange"}
        expected_score = self.scoring._calculate_leaf_score(list(node.values()))
        self.assertEqual(self.scoring.calculate_sub_scores(node), expected_score)

    def test_calculate_sub_scores_subtree(self):
        node = {"ML": {"a_or": "green", "spearman_correlation_drag": "orange"}, "OOD": {"inference_time": "red"}}
        expected_scores = {"ML": self.scoring._calculate_leaf_score(list(node["ML"].values())),
                           "OOD": self.scoring._calculate_leaf_score(list(node["OOD"].values()))}
        self.assertEqual(self.scoring.calculate_sub_scores(node), expected_scores)

    def test_calculate_sub_scores_invalid_input(self):
        with self.assertRaises(ValueError) as context:
            self.scoring.calculate_sub_scores("not a dict")
        self.assertIn("Input must be a dictionary.", str(context.exception))

    def test_calculate_sub_scores_inconsistent_branching(self):
        node = {"a": "green", "b": {"c": "red"}}
        with self.assertRaises(ValueError) as context:
            self.scoring.calculate_sub_scores(node)
        self.assertIn("Parent node is not uniformly branched", str(context.exception))

    def test_calculate_global_score_subtree(self):
        tree = {"ML": {"a_or": 0.01, "spearman_correlation_drag": 0.95}, "OOD": {"inference_time": 400},
                "Physics": {"a_or": 0.06, "spearman_correlation_drag": 0.75}, "Speed": {"inference_time": 600}, }
        tree_score = {"ML": 1, "OOD": 1, "Physics": 0, "Speed": 0.5}

        # Calculate expected score manually based on coefficients and leaf scores
        ml_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["ML"]).values()))
        ood_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["OOD"]).values()))
        physics_score = self.scoring._calculate_leaf_score(
            list(self.scoring.colorize_metrics(tree["Physics"]).values()))
        speed_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["Speed"]).values()))

        expected_global_score = (0.3 * ml_score + 0.3 * ood_score + 0.3 * physics_score + 0.1 * speed_score)

        global_score = self.scoring.calculate_global_score(tree_score)  # Operate on the original tree

        self.assertAlmostEqual(global_score, expected_global_score, places=6)  # Use assertAlmostEqual

    def test_calculate_global_score_subtree_with_missing_coefficient(self):
        tree = {"ML": {"a_or": 0.01, "spearman_correlation_drag": 0.95},  # Original float values
                "OOD": {"inference_time": 400},  # Original float values
                "Physics": {"a_or": 0.06, "spearman_correlation_drag": 0.75},  # Original float values
                "Speed": {"inference_time": 600},  # Original float values
                "NewComponent": {"a_or": 0.03}  # Original float values
                }
        tree_score = {"ML": 1, "OOD": 1, "Physics": 0, "Speed": 0.5, "NewComponent": 0.5}

        ml_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["ML"]).values()))
        ood_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["OOD"]).values()))
        physics_score = self.scoring._calculate_leaf_score(
            list(self.scoring.colorize_metrics(tree["Physics"]).values()))
        speed_score = self.scoring._calculate_leaf_score(list(self.scoring.colorize_metrics(tree["Speed"]).values()))
        new_component_score = self.scoring._calculate_leaf_score(
            list(self.scoring.colorize_metrics(tree["NewComponent"]).values()))

        expected_global_score = (
                0.3 * ml_score + 0.3 * ood_score + 0.3 * physics_score + 0.1 * speed_score + new_component_score
            # Default weight of 1
        )

        global_score = self.scoring.calculate_global_score(tree_score)  # Operate on the original tree
        self.assertAlmostEqual(global_score, expected_global_score, places=6)


if __name__ == '__main__':
    unittest.main()

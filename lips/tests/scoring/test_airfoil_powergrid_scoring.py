import json
import unittest
from unittest.mock import MagicMock, patch, mock_open

from lips.scoring import AirfoilPowerGridScoring


class TestAirfoilPowerGridScoring(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.get_option.side_effect = lambda key: {
            "thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05]},
                           "spearman_correlation_drag": {"comparison_type": "maximize", "thresholds": [0.8, 0.9]},
                           "spearman_correlation_lift": {"comparison_type": "maximize", "thresholds": [0.96, 0.99]},
                           "pressure": {"comparison_type": "minimize", "thresholds": [500, 1000]},
                           "pressure_surfacic": {"comparison_type": "minimize", "thresholds": [0.008, 0.02]},
                           "reference_mean_simulation_time": {"comparison_type": "ratio", "thresholds": [1500]},
                           "max_speed_ratio_allowed": {"comparison_type": "ratio", "thresholds": [10000]}},
            "valuebycolor": {"green": 2, "orange": 1, "red": 0},
            "coefficients": {"ML": {"value": 0.3, "Accuracy": {"value": 0.75}, "Speed": {"value": 0.25}},
                             "OOD": {"value": 0.3, "Accuracy": {"value": 0.75}, "Speed": {"value": 0.25}},
                             "Physics": {"value": 0.3}}}[key]
        self.scoring = AirfoilPowerGridScoring(config=self.mock_config)

    def test__reconstruct_ml_metrics(self):
        raw_metrics = {"fc_metrics_test": {"test": {
            "ML": {"MSE_normalized": {"a_or": 0.4, "spearman_correlation_drag": 0.6},
                   "MSE_normalized_surfacic": {"pressure": 0.2}}}}}
        ml_key_path = ["fc_metrics_test", "test", "ML", "MSE_normalized"]
        ml_metrics = self.scoring._reconstruct_ml_metrics(raw_metrics, ml_key_path)
        self.assertEqual(ml_metrics["ML"]["a_or"], 0.4)
        self.assertEqual(ml_metrics["ML"]["pressure_surfacic"], 0.2)

    def test__reconstruct_ml_metrics_invalid_path(self):
        raw_metrics = {}
        ml_key_path = ["invalid", "path"]
        with self.assertRaises(ValueError):
            self.scoring._reconstruct_ml_metrics(raw_metrics, ml_key_path)

    def test__reconstruct_ml_metrics_invalid_type(self):
        raw_metrics = {"fc_metrics_test": {"test": {"ML": {"MSE_normalized": "not a dict"}}}}
        ml_key_path = ["fc_metrics_test", "test", "ML", "MSE_normalized"]
        with self.assertRaises(TypeError):
            self.scoring._reconstruct_ml_metrics(raw_metrics, ml_key_path)

    def test__reconstruct_physic_metrics(self):
        raw_metrics = {"fc_metrics_test": {
            "test": {"Physics": {"spearman_correlation_drag": 0.7, "spearman_correlation_lift": 0.8}}}}
        physic_key_path = ["fc_metrics_test", "test", "Physics"]
        physic_metrics = self.scoring._reconstruct_physic_metrics(raw_metrics, physic_key_path)
        self.assertEqual(physic_metrics["Physics"]["spearman_correlation_drag"], 0.7)

    def test__reconstruct_ood_metrics(self):
        raw_metrics = {"fc_metrics_test_ood": {
            "test_ood": {"ML": {"MSE_normalized": {"a_or": 0.3}}, "Physics": {"spearman_correlation_drag": 0.9}}},
            "fc_metrics_test": {
                "test": {"ML": {"MSE_normalized": {"some_metric": 0.4}}, "Physics": {"some_metric": 0.7}}}}
        ml_ood_key_path = ["fc_metrics_test_ood", "test_ood", "ML", "MSE_normalized"]
        physic_ood_key_path = ["fc_metrics_test_ood", "test_ood", "Physics"]
        ood_metrics = self.scoring._reconstruct_ood_metrics(raw_metrics, ml_ood_key_path, physic_ood_key_path)
        self.assertEqual(ood_metrics["OOD"]["spearman_correlation_drag"], 0.9)

    def test_compute_speed_score(self):
        time_inference = 5.0
        speed_score = self.scoring.compute_speed_score(time_inference)
        self.assertAlmostEqual(speed_score, 0.61928031)

    def test_compute_speed_score_max_speed(self):
        time_inference = 1600
        speed_score = self.scoring.compute_speed_score(time_inference)
        self.assertAlmostEqual(speed_score, 0)

    @patch("builtins.open", new_callable=mock_open)
    def test_compute_scores_from_path(self, mock_file):
        mock_json_data = {"test_mean_simulation_time": 5.0, "fc_metrics_test": {
            "test": {"ML": {"MSE_normalized": {"a_or": 0.4}, "MSE_normalized_surfacic": {"pressure": 0.008}},
                     "Physics": {"spearman_correlation_drag": 0.7}}}, "fc_metrics_test_ood": {
            "test_ood": {"ML": {"MSE_normalized": {"a_or": 0.3}, "MSE_normalized_surfacic": {"pressure": 0.06}},
                         "Physics": {"spearman_correlation_lift": 0.9}}}}
        mock_file.return_value.read.return_value = json.dumps(mock_json_data)

        scores = self.scoring.compute_scores(metrics_path="dummy.json")
        self.assertAlmostEqual(scores["Global Score"], 0.484068188)

    def test_compute_scores_from_dict(self):
        metrics_dict = {"test_mean_simulation_time": 5.0, "fc_metrics_test": {
            "test": {"ML": {"MSE_normalized": {"a_or": 0.4}, "MSE_normalized_surfacic": {"pressure": 0.008}},
                     "Physics": {"spearman_correlation_drag": 0.7}}}, "fc_metrics_test_ood": {
            "test_ood": {"ML": {"MSE_normalized": {"a_or": 0.3}, "MSE_normalized_surfacic": {"pressure": 0.06}},
                         "Physics": {"spearman_correlation_lift": 0.9}}}}
        scores = self.scoring.compute_scores(metrics_dict=metrics_dict)
        self.assertAlmostEqual(scores["Global Score"], 0.484068188)

    def test_compute_scores_invalid_input(self):
        with self.assertRaises(ValueError):
            self.scoring.compute_scores()

    def test_compute_scores_missing_time(self):
        metrics_dict = {"fc_metrics_test": {
            "test": {"ML": {"MSE_normalized": {"some_metric": 0.4}}, "Physics": {"some_metric": 0.7}}},
            "fc_metrics_test_ood": {"test_ood": {"ML": {"MSE_normalized": {"some_metric": 0.3}},
                                                 "Physics": {"some_metric": 0.9}}}}  # Dummy metrics
        with self.assertRaises(KeyError):
            self.scoring.compute_scores(metrics_dict=metrics_dict)


if __name__ == '__main__':
    unittest.main()

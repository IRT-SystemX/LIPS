import json
import unittest
from unittest.mock import patch, MagicMock, mock_open

from lips.scoring import ML4PhysimPowerGridScoring


class TestML4PhysimPowerGridScoring(unittest.TestCase):

    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.get_option.side_effect = lambda key: {
            "thresholds": {"a_or": {"comparison_type": "minimize", "thresholds": [0.02, 0.05]},
                           "a_ex": {"comparison_type": "minimize", "thresholds": [0.03, 0.06]},
                           "p_or": {"comparison_type": "minimize", "thresholds": [0.04, 0.07]},
                           "p_ex": {"comparison_type": "minimize", "thresholds": [0.05, 0.08]},
                           "v_or": {"comparison_type": "minimize", "thresholds": [0.06, 0.09]},
                           "v_ex": {"comparison_type": "minimize", "thresholds": [0.07, 0.1]},

                           "CURRENT_POS": {"comparison_type": "minimize", "thresholds": [0.1, 0.2]},
                           "VOLTAGE_POS": {"comparison_type": "minimize", "thresholds": [0.2, 0.3]},
                           "LOSS_POS": {"comparison_type": "minimize", "thresholds": [0.3, 0.4]},
                           "DISC_LINES": {"comparison_type": "minimize", "thresholds": [0.4, 0.5]},
                           "CHECK_LOSS": {"comparison_type": "minimize", "thresholds": [0.5, 0.6]},
                           "CHECK_GC": {"comparison_type": "minimize", "thresholds": [0.6, 0.7]},
                           "CHECK_LC": {"comparison_type": "minimize", "thresholds": [0.7, 0.8]},
                           "CHECK_JOULE_LAW": {"comparison_type": "minimize", "thresholds": [0.8, 0.9]},

                           "reference_mean_simulation_time": {"comparison_type": "ratio", "thresholds": [1500]}},

            "valuebycolor": {"green": 2, "orange": 1, "red": 0},
            "coefficients": {"ID": {"value": 0.3, "ML": {"value": 0.4}, "Physics": {"value": 0.6}},
                             "OOD": {"value": 0.3, "ML": {"value": 0.66}, "Physics": {"value": 0.34}},
                             "SpeedUP": {"value": 0.4}}}[key]

        self.scoring = ML4PhysimPowerGridScoring(config=self.mock_config)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_ml_metrics_valid_path(self, mock_get_nested_value):
        raw_metrics = {"test": {
            "ML": {"MAPE_90_avg": {"a_or": 0.1, "a_ex": 0.2}, "MAPE_10_avg": {"p_or": 0.3, "p_ex": 0.4},
                   "MAE_avg": {"v_or": 0.5, "v_ex": 0.6}}}}
        mock_get_nested_value.return_value = raw_metrics["test"]["ML"]
        ml_section_path = ["test", "ML"]
        expected_result = {"ML": {"a_or": 0.1, "a_ex": 0.2, "p_or": 0.3, "p_ex": 0.4, "v_or": 0.5, "v_ex": 0.6}}
        result = self.scoring._reconstruct_ml_metrics(raw_metrics, ml_section_path)
        self.assertEqual(result, expected_result)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_ml_metrics_invalid_path(self, mock_get_nested_value):
        raw_metrics = {"test": {
            "ML": {"MAPE_90_avg": {"a_or": 0.1, "a_ex": 0.2}, "MAPE_10_avg": {"p_or": 0.3, "p_ex": 0.4},
                   "MAE_avg": {"v_or": 0.5, "v_ex": 0.6}}}}
        mock_get_nested_value.return_value = None
        ml_section_path = ["invalid", "path"]
        with self.assertRaises(ValueError):
            self.scoring._reconstruct_ml_metrics(raw_metrics, ml_section_path)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_ml_metrics_invalid_type(self, mock_get_nested_value):
        raw_metrics = {"test": {"ML": "invalid_type"}}
        mock_get_nested_value.return_value = raw_metrics["test"]["ML"]
        ml_section_path = ["test", "ML"]
        with self.assertRaises(TypeError):
            self.scoring._reconstruct_ml_metrics(raw_metrics, ml_section_path)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_physic_metrics_valid_path(self, mock_get_nested_value):
        raw_metrics = {"test": {"Physics": {"CURRENT_POS": {"a_or": {"Violation_proportion": 0.1}},
                                            "VOLTAGE_POS": {"v_or": {"Violation_proportion": 0.2}},
                                            "LOSS_POS": {"violation_proportion": 0.3},
                                            "DISC_LINES": {"violation_proportion": 0.4},
                                            "CHECK_LOSS": {"violation_percentage": 0.5},
                                            "CHECK_GC": {"violation_percentage": 0.6},
                                            "CHECK_LC": {"violation_percentage": 0.7},
                                            "CHECK_JOULE_LAW": {"violation_proportion": 0.8}}}}
        mock_get_nested_value.return_value = raw_metrics["test"]["Physics"]
        physic_section_path = ["test", "Physics"]
        expected_result = {"Physics": {"CURRENT_POS": 10.0, "VOLTAGE_POS": 20.0, "LOSS_POS": 30.0, "DISC_LINES": 40.0,
                                       "CHECK_LOSS": 0.5, "CHECK_GC": 0.6, "CHECK_LC": 0.7, "CHECK_JOULE_LAW": 80.0}}
        result = self.scoring._reconstruct_physic_metrics(raw_metrics, physic_section_path)
        self.assertEqual(result, expected_result)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_physic_metrics_invalid_path(self, mock_get_nested_value):
        raw_metrics = {"test": {"Physics": {"CURRENT_POS": {"a_or": {"Violation_proportion": 0.5}},
                                            "VOLTAGE_POS": {"v_or": {"Violation_proportion": 0.2}},
                                            "LOSS_POS": {"violation_proportion": 0.3},
                                            "DISC_LINES": {"violation_proportion": 0.4},
                                            "CHECK_LOSS": {"violation_percentage": 0.5},
                                            "CHECK_GC": {"violation_percentage": 0.6},
                                            "CHECK_LC": {"violation_percentage": 0.7},
                                            "CHECK_JOULE_LAW": {"violation_proportion": 0.8}}}}
        mock_get_nested_value.return_value = None
        physic_section_path = ["invalid", "path"]
        with self.assertRaises(ValueError):
            self.scoring._reconstruct_physic_metrics(raw_metrics, physic_section_path)

    @patch('lips.scoring.utils.get_nested_value')
    def test_reconstruct_physic_metrics_invalid_type(self, mock_get_nested_value):
        raw_metrics = {"test": {"Physics": "invalid_type"}}
        mock_get_nested_value.return_value = raw_metrics["test"]["Physics"]
        physic_section_path = ["test", "Physics"]
        with self.assertRaises(TypeError):
            self.scoring._reconstruct_physic_metrics(raw_metrics, physic_section_path)

    def test_calculate_speed_up(self):
        time_inference = 0.5
        expected_speed_up = 3000.0
        result = self.scoring._calculate_speed_up(time_inference)
        self.assertEqual(result, expected_speed_up)

    def test_compute_speed_score(self):
        time_inference = 0.5
        expected_speed_score = 1.0
        result = self.scoring.compute_speed_score(time_inference)
        self.assertEqual(result, expected_speed_score)

    def test_weibull(self):
        c = 5
        b = 1.7
        x = 3000.0
        expected_weibull_value = 1.0
        result = self.scoring._weibull(c, b, x)
        self.assertAlmostEqual(result, expected_weibull_value, places=5)

    @patch("builtins.open", new_callable=mock_open)
    def test_compute_scores(self, mock_file):
        mock_json_data = {"test": {
            "ML": {"MAPE_90_avg": {"a_or": 0.15, "a_ex": 0.25}, "MAPE_10_avg": {"p_or": 0.35, "p_ex": 0.45},
                   "MAE_avg": {"v_or": 0.55, "v_ex": 0.65}, "TIME_INF": 12.0},
            "Physics": {"CURRENT_POS": {"a_or": {"Violation_proportion": 0.15}},
                        "VOLTAGE_POS": {"v_or": {"Violation_proportion": 0.25}},
                        "LOSS_POS": {"violation_proportion": 0.35}, "DISC_LINES": {"violation_proportion": 0.45},
                        "CHECK_LOSS": {"violation_percentage": 0.55}, "CHECK_GC": {"violation_percentage": 0.65},
                        "CHECK_LC": {"violation_percentage": 0.75}, "CHECK_JOULE_LAW": {"violation_proportion": 0.85}}},
            "test_ood_topo": {
                "ML": {"MAPE_90_avg": {"a_or": 0.12, "a_ex": 0.22}, "MAPE_10_avg": {"p_or": 0.32, "p_ex": 0.42},
                       "MAE_avg": {"v_or": 0.52, "v_ex": 0.62}},
                "Physics": {"CURRENT_POS": {"a_or": {"Violation_proportion": 0.12}},
                            "VOLTAGE_POS": {"v_or": {"Violation_proportion": 0.22}},
                            "LOSS_POS": {"violation_proportion": 0.32}, "DISC_LINES": {"violation_proportion": 0.42},
                            "CHECK_LOSS": {"violation_percentage": 0.52}, "CHECK_GC": {"violation_percentage": 0.62},
                            "CHECK_LC": {"violation_percentage": 0.72},
                            "CHECK_JOULE_LAW": {"violation_proportion": 0.82}}}}

        mock_file.return_value.read.return_value = json.dumps(mock_json_data)

        scores = self.scoring.compute_scores(metrics_path="dummy.json")
        self.assertAlmostEqual(scores["Global Score"], 0.452874999)

    def test_compute_scores_invalid_input(self):
        with self.assertRaises(ValueError):
            self.scoring.compute_scores()

    def test_compute_scores_no_metrics(self):
        with self.assertRaises(ValueError):
            self.scoring.compute_scores()


if __name__ == '__main__':
    unittest.main()

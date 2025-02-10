import math
from lips.metrics.power_grid.compute_solver_time import compute_solver_time
from lips.metrics.power_grid.compute_solver_time_grid2op import compute_solver_time_grid2op

thresholds={"a_or":(0.02,0.05,"min"),
            "a_ex":(0.02,0.05,"min"),
            "p_or":(0.02,0.05,"min"),
            "p_ex":(0.02,0.05,"min"),
            "v_or":(0.2,0.5,"min"),
            "v_ex":(0.2,0.5,"min"),
            "CURRENT_POS":(1., 5.,"min"),
            "VOLTAGE_POS":(1.,5.,"min"),
            "LOSS_POS":(1.,5.,"min"),
            "DISC_LINES":(1.,5.,"min"),
            "CHECK_LOSS":(1.,5.,"min"),
            "CHECK_GC":(0.05,0.10,"min"),
            "CHECK_LC":(0.05,0.10,"min"),
            "CHECK_JOULE_LAW":(1.,5.,"min")
           }

configuration={
    "coefficients":{"test":0.3, "test_ood":0.3, "speed_up":0.4},
    "test_ratio":{"ml": 0.66, "physics":0.34},
    "test_ood_ratio":{"ml": 0.66, "physics":0.34},
    "value_by_color":{"g":2,"o":1,"r":0},
    "max_speed_ratio_allowed":50
}

def evaluate_model(benchmark, model):
    metrics = benchmark.evaluate_simulator(augmented_simulator=model,
                                           eval_batch_size=128,
                                           dataset="all",
                                           shuffle=False,
                                           save_path=None,
                                           save_predictions=False
                                          )
    return metrics

def compute_speed_up_metrics( metrics):
    speed_up = metrics["solver_time"] / metrics["test"]["ML"]["TIME_INF"]
    return speed_up

def compute_speed_up(config, metrics):
    # solver_time = compute_solver_time(nb_samples=int(1e5), config=config)
    solver_time = compute_solver_time_grid2op(config_path=config.path_config, benchmark_name=config.section_name, nb_samples=int(1e5))
    speed_up = solver_time / metrics["test"]["ML"]["TIME_INF"]
    return speed_up

def reconstruct_metric_dict(metrics, dataset: str="test"):
    rec_metrics = dict()
    rec_metrics["ML"] = dict()
    rec_metrics["Physics"] = dict()
    
    rec_metrics["ML"]["a_or"] = metrics[dataset]["ML"]["MAPE_90_avg"]["a_or"]
    rec_metrics["ML"]["a_ex"] = metrics[dataset]["ML"]["MAPE_90_avg"]["a_ex"]
    rec_metrics["ML"]["p_or"] = metrics[dataset]["ML"]["MAPE_10_avg"]["p_or"]
    rec_metrics["ML"]["p_ex"] = metrics[dataset]["ML"]["MAPE_10_avg"]["p_ex"]
    rec_metrics["ML"]["v_or"] = metrics[dataset]["ML"]["MAE_avg"]["v_or"]
    rec_metrics["ML"]["v_ex"] = metrics[dataset]["ML"]["MAE_avg"]["v_ex"]

    rec_metrics["Physics"]["CURRENT_POS"]     = metrics[dataset]["Physics"]["CURRENT_POS"]["a_or"]["Violation_proportion"] * 100.
    rec_metrics["Physics"]["VOLTAGE_POS"]     = metrics[dataset]["Physics"]["VOLTAGE_POS"]["v_or"]["Violation_proportion"] * 100.
    rec_metrics["Physics"]["LOSS_POS"]        = metrics[dataset]["Physics"]["LOSS_POS"]["violation_proportion"] * 100.
    rec_metrics["Physics"]["DISC_LINES"]      = metrics[dataset]["Physics"]["DISC_LINES"]["violation_proportion"] * 100.
    rec_metrics["Physics"]["CHECK_LOSS"]      = metrics[dataset]["Physics"]["CHECK_LOSS"]["violation_percentage"]
    rec_metrics["Physics"]["CHECK_GC"]        = metrics[dataset]["Physics"]["CHECK_GC"]["violation_percentage"]
    rec_metrics["Physics"]["CHECK_LC"]        = metrics[dataset]["Physics"]["CHECK_LC"]["violation_percentage"]
    rec_metrics["Physics"]["CHECK_JOULE_LAW"] = metrics[dataset]["Physics"]["CHECK_JOULE_LAW"]["violation_proportion"] * 100.

    return rec_metrics

def discretize_results(metrics):
    results=dict()
    for subcategoryName, subcategoryVal in metrics.items():
        results[subcategoryName]=[]
        for variableName, variableError in subcategoryVal.items():
            thresholdMin,thresholdMax,evalType=thresholds[variableName]
            if evalType=="min":
                if variableError<thresholdMin:
                    accuracyEval="g"
                elif thresholdMin<variableError<thresholdMax:
                    accuracyEval="o"
                else:
                    accuracyEval="r"
            elif evalType=="max":
                if variableError<thresholdMin:
                    accuracyEval="r"
                elif thresholdMin<variableError<thresholdMax:
                    accuracyEval="o"
                else:
                    accuracyEval="g"

            results[subcategoryName].append(accuracyEval)
    return results

# def SpeedMetric(speedUp,speedMax):
#     return max(min(math.log10(speedUp)/math.log10(speedMax),1),0)

# def SpeedMetric(speedUp, speedMax):
#     a=0.01 # 0.01
#     b=0.5 #0.5
#     c=0.1 #0.1
#     k=9
#     res = quadratic_function(speedUp, a=a, b=b, c=c, k=k) / quadratic_function(speedMax, a=a, b=b, c=c, k=k)
#     return max(min(res, 1), 0)

def quadratic_function(x, a, b, c, k):
    if x == 1.:
        return 0.
    else:
        return a*(x**2) + b*x + c + math.log10(k*x)
    
def weibull(c,b,x):
    a = c * ((-math.log(0.9)) ** (-1/b))
    return 1. - math.exp(-(x / a)**b)

def SpeedMetric(speedUp):
    res = weibull(5, 1.7, speedUp)
    return max(min(res, 1), 0)

def compute_ml_subscore(results, key: str="test_ratio"):
    test_ratio = configuration[key]
    value_by_color = configuration["value_by_color"]
    test_ml_res = sum([value_by_color[color] for color in results["ML"]])
    test_ml_subscore = (test_ml_res * test_ratio["ml"]) / (len(results["ML"])*max(value_by_color.values()))
    return test_ml_subscore

def compute_physics_subscore(results, key: str="test_ratio"):
    test_ratio = configuration[key]
    value_by_color = configuration["value_by_color"]
    test_physics_res = sum([value_by_color[color] for color in results["Physics"]])
    test_physics_subscore = (test_physics_res*test_ratio["physics"]) / (len(results["Physics"])*max(value_by_color.values()))
    return test_physics_subscore

def compute_global_score(metrics):
    coefficients = configuration["coefficients"]
    max_speed_ratio_allowed = configuration["max_speed_ratio_allowed"]

    test_metrics = reconstruct_metric_dict(metrics, "test")
    test_ood_metrics = reconstruct_metric_dict(metrics, "test_ood_topo")

    test_results_disc = discretize_results(test_metrics)
    test_ood_results_disc = discretize_results(test_ood_metrics)

    test_ml_subscore = compute_ml_subscore(test_results_disc, key="test_ratio")
    test_physics_subscore = compute_physics_subscore(test_results_disc, key="test_ratio")
    test_subscore = test_ml_subscore + test_physics_subscore

    test_ood_ml_subscore = compute_ml_subscore(test_ood_results_disc, key="test_ood_ratio")
    test_ood_physics_subscore = compute_physics_subscore(test_ood_results_disc, key="test_ood_ratio")
    test_ood_subscore = test_ood_ml_subscore + test_ood_physics_subscore

    speed_up = compute_speed_up_metrics( metrics)
    speedup_score = SpeedMetric(speedUp=speed_up)

    globalScore = 100*(coefficients["test"]*test_subscore+coefficients["test_ood"]*test_ood_subscore+coefficients["speed_up"]*speedup_score)
    print("GS :", globalScore)
    print("Test :", test_subscore)
    print("OOD :", test_ood_subscore)
    print("SpeedUp :", speedup_score)

    return globalScore, test_ml_subscore, test_physics_subscore, test_ood_ml_subscore, test_ood_physics_subscore, speedup_score
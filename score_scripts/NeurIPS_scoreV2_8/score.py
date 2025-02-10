#!/usr/bin/env python
# Scoring program for the AirfRANSModel challenge.
# Use directly the outputfile from the LIPS evaluation program.
# Some libraries and options
import os
from sys import argv
import sys
import json
import math


# Default I/O directories:
root_dir = "/app/"
default_input_dir = root_dir + "scoring_input"
default_output_dir = root_dir + "scoring_output"
default_input_dir = "./"
default_output_dir="./"

# Constant used for a missing score
missing_score = -0.999999

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

class ModelApiError(Exception):
    """Model api error"""

    def __init__(self, msg=""):
        self.msg = msg
        print(msg)


def import_metrics(input_dir):
    ## import parameters.json as a dictionary
    path_submission_parameters = os.path.join(input_dir, 'res', 'json_metrics.json')
    if not os.path.exists(path_submission_parameters):
        raise ModelApiError("Missing json_metrics.json file")
        exit_program()
    with open(path_submission_parameters) as json_file:
        metrics = json.load(json_file)
    return metrics

def exit_program():
    print("Error exiting")
    sys.exit(0)

def SpeedMetric(speedUp,speedMax):
    return max(min(math.log10(speedUp)/math.log10(speedMax),1),0)

# =============================== MAIN ========================================

if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        # Create the output directory, if it does not already exist and open output files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'w')

    # Init HTML
    html_file.write("<html><head><title>Scoring program output</title></head><body>")

    ## thresholds
    # First competition thresholds
    # thresholds={"x-velocity":(0.1,0.2,"min"),
    #             "y-velocity":(0.1,0.2,"min"),
    #             "pressure":(0.02,0.1,"min"),
    #             "pressure_surfacic":(0.08,0.2,"min"),
    #             "turbulent_viscosity":(0.5,1.0,"min"),
    #             "mean_relative_drag":(1.0,10.0,"min"),
    #             "mean_relative_lift":(0.2,0.5,"min"),
    #             "spearman_correlation_drag":(0.5,0.8,"max"),
    #             "spearman_correlation_lift":(0.94,0.98,"max")          
    #     }
    
    # NeurIPS competition thresholds
    thresholds={"x-velocity":(0.01,0.02,"min"),
            "y-velocity":(0.01,0.02,"min"),
            "pressure":(0.002,0.01,"min"),
            "pressure_surfacic":(0.008,0.02,"min"),
            "turbulent_viscosity":(0.05,0.1,"min"),
            "mean_relative_drag":(0.4,5.0,"min"),
            "mean_relative_lift":(0.1,0.3,"min"),
            "spearman_correlation_drag":(0.8,0.9,"max"),
            "spearman_correlation_lift":(0.96,0.99,"max")          
    }
    
    
    # scoring configuration
    configuration={
        "coefficients":{"ML":0.4,"OOD":0.3,"Physics":0.3},
        "ratioRelevance":{"Speed-up":0.25,"Accuracy":0.75},
        "valueByColor":{"g":2,"o":1,"r":0},
        "maxSpeedRatioAllowed":10000,
        "reference_mean_simulation_time":1500
        }

    coefficients=configuration["coefficients"]
    ratioRelevance=configuration["ratioRelevance"]
    valueByColor=configuration["valueByColor"]
    maxSpeedRatioAllowed=configuration["maxSpeedRatioAllowed"]

    #Import metrics
    metrics = import_metrics(input_dir)
    fc_metrics_test = metrics["fc_metrics_test"]
    fc_metrics_test_ood = metrics["fc_metrics_test_ood"]
    test_mean_simulation_time = metrics["test_mean_simulation_time"]
    test_ood_mean_simulation_time = metrics["test_ood_mean_simulation_time"]


    ## Extract metrics of interest
    ML_metrics = "ML"
    ml_metrics = fc_metrics_test["test"][ML_metrics]["MSE_normalized"]
    ml_metrics["pressure_surfacic"] = fc_metrics_test["test"][ML_metrics]["MSE_normalized_surfacic"]["pressure"]

    physic_compliances = "Physics"
    phy_variables_to_keep = ["mean_relative_drag","mean_relative_lift","spearman_correlation_drag","spearman_correlation_lift"]
    phy_metrics = {phy_variable:fc_metrics_test["test"][physic_compliances][phy_variable] for phy_variable in phy_variables_to_keep}

    ml_ood_metrics = fc_metrics_test_ood["test_ood"][ML_metrics]["MSE_normalized"]
    ml_ood_metrics["pressure_surfacic"] = fc_metrics_test_ood["test_ood"][ML_metrics]["MSE_normalized_surfacic"]["pressure"]
    phy_ood_metrics = {phy_variable:fc_metrics_test_ood["test_ood"][physic_compliances][phy_variable] for phy_variable in phy_variables_to_keep}
    ood_metrics = {**ml_ood_metrics,**phy_ood_metrics}

    allmetrics={
        ML_metrics:ml_metrics,
        physic_compliances:phy_metrics,
        "OOD":ood_metrics
    }

    print(allmetrics)

    ## Compute speed-up :
    

    reference_mean_simulation_time=configuration["reference_mean_simulation_time"]
    speedUp={
            ML_metrics:reference_mean_simulation_time/test_mean_simulation_time,
            "OOD":reference_mean_simulation_time/test_mean_simulation_time
            }
    
    accuracyResults=dict()


    for subcategoryName, subcategoryVal in allmetrics.items():
        accuracyResults[subcategoryName]=[]
        html_file.write("<h3>" + subcategoryName + "</h3>")
        for variableName, variableError in subcategoryVal.items():
            thresholdMin,thresholdMax,evalType = thresholds[variableName]
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

            accuracyResults[subcategoryName].append(accuracyEval)
            html_file.write("<p>" + subcategoryName + " - " + variableName + " - " + ": "+accuracyEval+"</p>")
        html_file.write("<br>")
        
   


    
    ## ML subscore
    

    mlSubscore=0

    #Compute accuracy
    accuracyMaxPoints=ratioRelevance["Accuracy"]
    accuracyResult=sum([valueByColor[color] for color in accuracyResults["ML"]])
    accuracyResult=accuracyResult*accuracyMaxPoints/(len(accuracyResults["ML"])*max(valueByColor.values()))
    mlSubscore+=accuracyResult

    #Compute speed-up
    speedUpMaxPoints=ratioRelevance["Speed-up"]
    speedUpResult=SpeedMetric(speedUp=speedUp["ML"],speedMax=maxSpeedRatioAllowed)
    speedUpResult=speedUpResult*speedUpMaxPoints
    mlSubscore+=speedUpResult

    ## compute Physics subscore
    # Compute accuracy
    accuracyResult=sum([valueByColor[color] for color in accuracyResults["Physics"]])
    accuracyResult=accuracyResult/(len(accuracyResults["Physics"])*max(valueByColor.values()))
    physicsSubscore=accuracyResult

    ## Compute OOD subscore

    oodSubscore=0

    #Compute accuracy
    accuracyMaxPoints=ratioRelevance["Accuracy"]
    accuracyResult=sum([valueByColor[color] for color in accuracyResults["OOD"]])
    accuracyResult=accuracyResult*accuracyMaxPoints/(len(accuracyResults["OOD"])*max(valueByColor.values()))
    oodSubscore+=accuracyResult

    #Compute speed-up
    speedUpMaxPoints=ratioRelevance["Speed-up"]
    speedUpResult=SpeedMetric(speedUp=speedUp["OOD"],speedMax=maxSpeedRatioAllowed)
    speedUpResult=speedUpResult*speedUpMaxPoints
    oodSubscore+=speedUpResult

    ## Compute global score
    globalScore=100*(coefficients["ML"]*mlSubscore+coefficients["Physics"]*physicsSubscore+coefficients["OOD"]*oodSubscore)
    
    print(globalScore)

    # Write scores in the output file
    # exemple write score
    score_file.write("global_warmup" + ": %0.12f\n" % globalScore)
    score_file.write("ML_warmup" + ": %0.12f\n" % mlSubscore)
    score_file.write("Physics_warmup" + ": %0.12f\n" % physicsSubscore)
    score_file.write("OOD_warmup" + ": %0.12f\n" % oodSubscore)

    html_file.write("<br>")
    html_file.write("<p>Global score: %0.12f</p>" % globalScore)
    html_file.write("<p>ML score: %0.12f</p>" % mlSubscore)
    html_file.write("<p>Physics score: %0.12f</p>" % physicsSubscore)
    html_file.write("<p>OOD score: %0.12f</p>" % oodSubscore)
    html_file.write("</body></html>")

    html_file.close()
    score_file.close()
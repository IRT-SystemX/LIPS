#!/usr/bin/env python
# Use directly the outputfile from the LIPS evaluation program.
# Some libraries and options
import os
from sys import argv
import sys
import json
import math

import utils.compute_score as cs

# Default I/O directories:
root_dir = "/app/"
default_input_dir = root_dir + "scoring_input"
default_output_dir = root_dir + "scoring_output"
default_input_dir = "./"
default_output_dir="./"



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
    # path_submission_parameters = os.path.join(input_dir, 'json_metrics.json')
    print(path_submission_parameters)
    if not os.path.exists(path_submission_parameters):
        raise ModelApiError("Missing json_metrics.json file")
        exit_program()
    with open(path_submission_parameters) as json_file:
        metrics = json.load(json_file)
    return metrics



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
    print("Scoring file : ", os.path.join(output_dir, 'scores.txt'))
    metrics = import_metrics(input_dir)
    metrics["solver_time"] = 32.79

    globalScore, test_ml_subscore, test_physics_subscore, test_ood_ml_subscore, test_ood_physics_subscore, speedup_score = cs.compute_global_score(metrics)

    print("scoring done ", globalScore, test_ml_subscore, test_physics_subscore, test_ood_ml_subscore, test_ood_physics_subscore, speedup_score)

    score_file.write("global" + ": %0.12f\n" % globalScore)
    score_file.write("ML_test" + ": %0.12f\n" % test_ml_subscore)
    score_file.write("Physics_test" + ": %0.12f\n" % test_physics_subscore)
    score_file.write("ML_ood" + ": %0.12f\n" % test_ood_ml_subscore)
    score_file.write("Physics_ood" + ": %0.12f\n" % test_ood_physics_subscore)
    score_file.write("speedup" + ": %0.12f\n" % speedup_score)

    html_file.write("<html><head><title>Scoring program output</title></head><body>")
    html_file.write("<h3>Global Score : </h3>")
    html_file.write("<p>" + ": %0.12f\n" % globalScore)
    html_file.write("</p></body></html>")
    
    html_file.close()
    score_file.close()
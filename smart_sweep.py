import os
import json

from joblib import Parallel, delayed

from main import model, save_data, plot

def do_sweep():

    # Prepare all folder destinations and configs

    # Use sure run to run all points



def sure_run(point_list):

    # Send command to run all points in parallel

    # Check which points failed to finish

    # Recursively run all incomplete points


def run_point():
    ### This function should be run in parallel

    # Load config from file

    # Run the simulation at this point and save the data

    # Check that the point has been run successfully
        # If yes, make the file completion.txt with a note of the success
        # If no, exit, logging whatever error to an error file


def check_complete(out_dir):

    folders_here = [f for f in out_dir.iterdir() if f.is_dir()]
    incomplete = [f for f in folders_here if not os.path.exists(os.path.join(f, 'completion.txt'))]

    # return point IDS


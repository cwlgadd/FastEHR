import os
from pathlib import Path
import sys
node_type = os.getenv('BB_CPU')
venv_dir = f'/rds/homes/g/gaddcz/Projects/CPRD/virtual-env-{node_type}'
venv_site_pkgs = Path(venv_dir) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
if venv_site_pkgs.exists():
    sys.path.insert(0, str(venv_site_pkgs))
    print(f"Added path '{venv_site_pkgs}' at start of search paths.")
else:
    print(f"Path '{venv_site_pkgs}' not found. Check that it exists and/or that it exists for node-type '{node_type}'.")


# !echo $SQLITE_TMPDIR
# !echo $TMPDIR
# !echo $USERPROFILE

import pytorch_lightning
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sqlite3
import logging
from SurvivEHR.SurvivEHR_ExampleData.database.build_static_table import Static
from SurvivEHR.SurvivEHR_ExampleData.database.build_diagnosis_table import Diagnoses
from SurvivEHR.SurvivEHR_ExampleData.database.build_valued_event_tables import Measurements

if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    path_to_directory = "/home/ubuntu/Documents/GitHub/SurvivEHR/SurvivEHR_ExampleData/example/data/"

    PATH_TO_DB = path_to_directory + "_built/example_database.db"
    PATH_TO_STATIC = path_to_directory + "baseline/static_data.csv"
    PATH_TO_DIAGNOSIS = path_to_directory + "diagnoses/diagnosis_data.csv"
    PATH_TO_DYNAMIC = path_to_directory + "timeseries/measurement_tests_medications/"

    load = False
    if load:
        logging.warning(f"Load is true, if you want to re-build database set to False")
    
    static = Static(PATH_TO_DB, PATH_TO_STATIC, load=load)
    diagnosis = Diagnoses(PATH_TO_DB, PATH_TO_DIAGNOSIS, load=load)
    measurements = Measurements(PATH_TO_DB, PATH_TO_DYNAMIC, load=load)

    for table in [static, diagnosis, measurements]:
        print(table)
    
    
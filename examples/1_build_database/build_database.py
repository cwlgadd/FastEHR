import os
import torch
import logging
from database.build_db.build_static_table import Static
from database.build_db.build_diagnosis_table import Diagnoses
from database.build_db.build_valued_event_tables import Measurements

if __name__ == "__main__":

    torch.manual_seed(1337)
    logging.basicConfig(level=logging.INFO)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}.")

    path_to_directory = os.getcwd() + "/../data/"

    PATH_TO_DB = path_to_directory + "_built/example_database.db"
    print(f"Saving databse to {PATH_TO_DB}")

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
    
    
import sqlite3
import pandas as pd
from tqdm import tqdm
import logging


class Diagnoses():

    def __init__(self, db_path, path_to_data, load=False):
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        self.connection_token = 'sqlite://' + self.db_path
        self.path_to_data = path_to_data

        if load is False:
            # Create table
            self.connect()
            logging.info("Creating diagnosis_table")
            self.cursor.execute("""
                CREATE TABLE diagnosis_table (
                    PRACTICE_ID integer,
                    PATIENT_ID integer,
                    EVENT text,
                    DATE text
                )
            """)
            self.build_table()
            self.disconnect()

    def __str__(self):
        self.connect()
        self.cursor.execute("SELECT COUNT(*) FROM diagnosis_table")
        s = f'Diagnosis table with {self.cursor.fetchone()[0]} records.'
        self.disconnect()
        return s

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            logging.debug("Connected to SQLite database")
        except sqlite3.Error as e:
            logging.warning(f"Error connecting to SQLite database: {e}")
            raise ConnectionError

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
            logging.debug("Disconnected from SQLite database")

    def build_table(self, verbose=1, **kwargs):
        """
        """
        self.connect()
        self._add_file_to_table(self.path_to_data, verbose=verbose, **kwargs)

        # Create index
        self._make_index()

        self.disconnect()

    def _make_index(self):
        # Create index
        logging.info("Creating indexes on diagnosis_table")
        query = """
        CREATE INDEX IF NOT EXISTS diagnosis_index
        ON diagnosis_table (PRACTICE_ID);
        """
        logging.debug(query)
        self.cursor.execute(query)
        self.connection.commit()

    def _add_file_to_table(self, fname, chunksize=200000, verbose=0, **kwargs):

        generator = pd.read_csv(
            fname,
            chunksize=chunksize,
            iterator=True,
            encoding='utf-8',
            low_memory=False,
            dtype={'PRACTICE_PATIENT_ID': 'str'},
        )
        # low_memory=False just silences an error, TODO: add dtypes
        for df in tqdm(generator, desc="Building diagnosis table"):

            # Start indices from 1
            df.index += 1

            #####################
            # Conditions
            #####################
            # Get diagnosis date columns
            date_columns = [
                cn for cn in df.columns
                if (
                        "BD_MEDI:" in cn and
                        "LEARNINGDISABILITY_BIRM_CAM_V3" not in cn
                )
            ]

            # Note: Excludes a specific code for backward compatibility
            # which may appear in DEXTER output, but is omitted here
            # for backward compatibility.

            # Clean condition names (specific to CPRD DEXTER output)
            # by removing pre-fix
            condition_names = [
                name.removeprefix('BD_MEDI:')
                for name in date_columns
            ]

            # Remove specific substrings that may pollute condition names
            # (specific to CPRD DEXTER output)
            replacements = [
                "_BHAM_CAM", "_FINAL", "_BIRM_CAM", "_MM",
                "_11_3_21", "_20092020", "_120421",
            ]
            for substr in replacements:
                condition_names = [
                    name.replace(substr, '') for name in condition_names
                ]

            # Strip condition number suffix (if exists),
            # retaining only the base name (specific to CPRD DEXTER output)
            condition_names = [
                name.split(":", 1)[0] for name in condition_names
            ]

            rename_dict = dict(zip(date_columns, condition_names))

            # Rename and subset dataframe
            df = df.rename(columns=rename_dict)

            # Add practice ID column
            df['PRACTICE_ID'] = (
                df['PRACTICE_PATIENT_ID']
                .str.split('_')
                .str[0]
                .str.lstrip('p')
            )

            df['PATIENT_ID'] = (
                df['PRACTICE_PATIENT_ID'].str.split('_')
                .str[1]
            )

            # Conditions
            ############
            df_conditions = df[["PRACTICE_ID", "PATIENT_ID"] + condition_names]

            for condition in df_conditions.columns[2:]:

                # Subset to the ID and condition
                df_one_condition = (
                    df_conditions[["PRACTICE_ID", "PATIENT_ID", condition]]
                    .dropna()
                )

                # Add condition as new column
                df_one_condition["condition"] = condition

                # and rename the condition column
                df_one_condition = (
                    df_one_condition
                    .rename(columns={condition: "age_at_diagnosis"})
                )

                # and order them as we want to see them in the table
                df_one_condition = (
                    df_one_condition[
                        ["PRACTICE_ID",
                         "PATIENT_ID",
                         "condition",
                         "age_at_diagnosis"]]
                    .copy()
                )
                # print(df_one_condition.head())

                # Pull records from df to update SQLite .db with
                #   records or rows in a list of tuples
                #   [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
                records = (
                    df_one_condition
                    .to_records(index=False,
                                # column_dtypes={
                                #     "PRACTICE_ID": "int64",
                                #     "PATIENT_ID": "int64",
                                # }
                                )
                )
                # Add rows to database
                self.cursor.executemany(
                    'INSERT INTO diagnosis_table VALUES(?,?,?,?);', records
                )

                if verbose > 1:
                    print(
                        f'Inserted {self.cursor.rowcount} {condition} records.'
                    )

            #####################
            # For death
            #####################
            # Subset to the ID and death
            df_death = (
                df[
                    ["PRACTICE_ID",
                     "PATIENT_ID",
                     "DEATH_DATE",
                     "YEAR_OF_BIRTH"]
                ]
                .dropna()
            )

            # add condition column
            df_death["condition"] = "DEATH"

            # # Subset to the IDs and event details
            df_death = (
                df_death[
                    ["PRACTICE_ID",
                     "PATIENT_ID",
                     "condition",
                     "DEATH_DATE"]
                ]
            )

            # Pull records from df to update SQLite .db with
            #   records or rows in a list of tuples
            #   [(ID, CONDITION, AGE_AT_DIAGNOSIS),]
            records = (
                df_death
                .to_records(index=False,
                            # column_dtypes={
                            #     "PRACTICE_ID": "int64",
                            #     "PATIENT_ID": "int64",
                            # }
                            )
            )

            # Add rows to database
            self.cursor.executemany(
                'INSERT INTO diagnosis_table VALUES(?,?,?,?);',
                records
            )

            if verbose > 1:
                print(
                    f'Inserted {self.cursor.rowcount} DEATH records.'
                )

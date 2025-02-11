import sqlite3
import polars as pl
import pandas as pd
import numpy as np
from typing import Optional, Any, Union, Callable
import logging
from FastEHR.database.build_static_table import Static
from FastEHR.database.build_diagnosis_table import Diagnoses
from FastEHR.database.build_valued_event_tables import Measurements
from tqdm import tqdm
from tdigest import TDigest

class SQLiteDataCollector(Static, Diagnoses, Measurements):
    """
    A class to interface with an SQLite database to collect and collate patient records.

    This class provides functionality for extracting structured patient data, aggregating
     medical events, and computing metadata for pre-processing from an SQLite database.

    Inherits from:
        - `Static`: Handles static patient data, such as birth year and ethnicity.
        - `Diagnoses`: Handles diagnosis-related records.
        - `Measurements`: Handles event-based measurements, which may optionally include an associated value.

    Attributes
    ----------
    db_path : str
        Path to the SQLite database file.
    connection : sqlite3.Connection.
        SQLite connection object, initialized when `connect()` is called.
    cursor : sqlite3.Cursor.
        Cursor for executing SQL queries.

    Methods
    -------
    connect()
        Establish the SQLite database connection.
    disconnect()
        Close the SQLite database connection.
    _extract_distinct(table_names, identifier_column, inclusion_conditions=None, combine_approach="AND")
        Extracts distinct values of a given column across multiple tables.
    _extract_AGG(table_name, identifier_column=None, aggregations="COUNT(*)", condition=None)
        Performs grouped aggregations over tables.
    _t_digest_values(table_name)
        Uses the `t-digest algorithm <https://github.com/tdunning/t-digest>` to approximate percentiles of a given measurement.
    _generate_lazy_by_distinct(distinct_values, identifier_column, include_diagnoses=True, include_measurements=True, conditions=None)
        Generates Polars LazyFrames for distinct patient or practice identifiers.
    _collate_lazy_tables(lazy_frames, study_inclusion_method=None, drop_empty_dynamic=True, drop_missing_data=True, **kwargs)
        Merges static and dynamic patient records into a single LazyFrame.
    get_meta_information(practice_ids=None, static=True, diagnoses=True, measurement=True)
        Collects metadata from the SQLite database, including distributions of diagnoses and measurements.
    """

    def __init__(self, db_path: str):
        """
        Initializes the SQLiteDataCollector.

        Parameters
        ----------
        db_path : str
            Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect(self):
        """
        Establishes a connection to the SQLite database.

        If the connection is already established, this method does nothing.

        Raises
        ------
        sqlite3.Error
            If an error occurs while connecting to the database.
        """
        if self.connection is None:
            try:
                self.connection = sqlite3.connect(self.db_path, timeout=20000)
                self.cursor = self.connection.cursor()
                logging.debug("Connected to SQLite database")
            except sqlite3.Error as e:
                logging.exception(f"Error connecting to SQLite database: {e}")
        else:
            logging.debug("Connection already established.")

    def disconnect(self):
        """
        Closes the SQLite database connection.

        This method ensures that both the connection and cursor are properly closed.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
            logging.debug("Disconnected from SQLite database")

    def _extract_distinct(self,
                          table_names:          list[str],
                          identifier_column:    str,
                          inclusion_conditions: Optional[list] = None,
                          combine_approach:     str = "AND"
                          ) -> list[str]:
        """
        Extracts distinct values of an identifier column across multiple tables.

        Parameters
        ----------
        table_names : list[str]
            List of table names to extract values from.
        identifier_column : str
            Column name to extract distinct values from.
        inclusion_conditions : list[str], optional
            List of SQL conditions to filter each table's records (default is None).
        combine_approach : str, optional
            How to combine results across tables ("AND" for intersection, "OR" for union). Default is "AND".

        Returns
        -------
        list[str]
            List of unique identifier values found across the specified tables.

        Raises
        ------
        NotImplementedError
            If an invalid `combine_approach` is provided.
        """
        # Initialize an empty set to store distinct values
        unique_distinct = set()
        
        # Iterate over each table
        for idx_table, table in enumerate(table_names):
            
            # Construct the SQL query to extract distinct `identifier_column` entries for each specified table
            query = f"SELECT DISTINCT {identifier_column} FROM {table}"

            # If we want to add a condition
            if inclusion_conditions is not None:
                if inclusion_conditions[idx_table] is not None:
                    query += f" WHERE {inclusion_conditions[idx_table]}"
            
            # Execute the query
            logging.debug(f"Query: {query[:100] if len(query) > 100 else query}")
            self.cursor.execute(query)
            
            # Fetch distinct query values for the current table
            new_distinct_values = [_dv[0] for _dv in self.cursor.fetchall()]

            # and update the set
            if combine_approach == "OR":
                # For example, can condition static table to get `identifier_column` values
                #      based in England  OR  with condition X
                unique_distinct.update(new_distinct_values)
            elif combine_approach == "AND":
                #      based in England  AND  with condition X
                unique_distinct = unique_distinct & set(new_distinct_values) if idx_table != 0 else set(new_distinct_values)
            else:
                raise NotImplementedError

        return list(unique_distinct)

    def _extract_AGG(self,
                     table_name:          str,
                     identifier_column:   Optional[str] = None,
                     aggregations:        str = "COUNT(*)",
                     condition:           Optional[str] = None,
                     ):
        """
        Performs aggregated calculations over a specified table.

        Example usages: (1) count number of each diagnosis, (2) total number of observed values for
         a measurement, etc.

        Parameters
        ----------
        table_name : str
            The name of the table to perform aggregation on.
        identifier_column : str, optional
            Column name for grouping the aggregation (default is None).
        aggregations : str, optional
            Aggregation function(s) to apply, such as `COUNT(*)` or `AVG(VALUE)` (default is "COUNT(*)").
        condition : str, optional
            SQL condition to filter rows before aggregation (default is None).

        Returns
        -------
        list
            The result of the aggregation query.

        """
        query = f"SELECT "
        
        if identifier_column is not None:
            query += f"{identifier_column}, "
            
        query += f"{aggregations} FROM {table_name}"

        # If we want to add a condition
        if condition is not None:
            query += f" WHERE {condition}"

        if identifier_column:
            query += f" GROUP BY {identifier_column}"
        
        # Execute the query
        logging.debug(f"Query: {query[:300] if len(query) > 300 else query}")
        self.cursor.execute(query)

        # Fetch unique prefixes for the current table and update the set
        result = self.cursor.fetchall()

        return result

    def _t_digest_values(self,
                         table_name:          str,
                         ):
        """
        Approximates percentiles using `Ted Dunning's t-digest algorithm <https://github.com/tdunning/t-digest>`.

        This method efficiently accumulates rank-based statistics from large datasets.

        Parameters
        ----------
        table_name : str
            Name of the table containing numeric values.

        Returns
        -------
        TDigest
            A t-digest object containing approximated percentiles.

        Notes
        -----
        The algorithm supports streaming updates for large datasets.
        """

        digest = TDigest()
        
        self.cursor.execute(f"SELECT VALUE FROM {table_name}")

        fetches_count = 0
        while True:
            records = self.cursor.fetchmany(10000)
            
            if not records or fetches_count > 1e5 / 10000:
                # exit loop when no more records to fetch, or we reach some approximating limit 
                break 
            values = np.array([_record[0] for _record in records if _record[0] is not None])
            try:
                digest.batch_update(values)
                fetches_count += 1
            except:
                logging.warning(f"Unable to batch update t-digest for values {values} from {table_name}, skipping batch")

        return digest
    
    def _generate_lazy_by_distinct(self,
                                   distinct_values: list,
                                   identifier_column: str,
                                   include_diagnoses: bool = True, 
                                   include_measurements: bool = True,
                                   conditions: Optional[list[str]] = None
                                   ) -> list[pl.LazyFrame]:
        """
        Generates Polars LazyFrames for each unique identifier.

        Parameters
        ----------
        distinct_values : list
            List of distinct values on which to partition the identifier_column.
        identifier_column : str
            The column used for partitioning data (e.g., `PATIENT_ID` or `PRACTICE_ID`).
        include_diagnoses : bool, optional
            Whether to include diagnosis records (default is True).
        include_measurements : bool, optional
            Whether to include measurement records (default is True).
        conditions : list[str], optional
            List of SQL conditions for each table (default is None). Each condition applies to a specific table.

        Yields
        ------
        tuple
            A tuple `(distinct_value, rows_by_table)`, where `rows_by_table` contains lazy-loaded data.

        Notes
        -----
        `conditions` is likely not required, as this remove neither full patients or practices - but singular events.
        """
        
        table_names = ["static_table"]
        if include_diagnoses:            
            table_names.append("diagnosis_table")
        if include_measurements:
            for measurement_table in self.measurement_table_names:
                table_names.append(measurement_table)   

        # Iterate over each
        for distinct_value in distinct_values:
            rows_by_table = {}
            for idx_table, table in enumerate(table_names):

                # Construct query for fetching rows with the current prefix for the current table
                if type(distinct_value) == list:
                    sep_list = ",".join([f"'{dv}'" for dv in distinct_value])
                    query = f"SELECT * FROM {table} WHERE {identifier_column} IN ({sep_list});"
                    
                else:
                    query = f"SELECT * FROM {table} WHERE {identifier_column} = '{distinct_value}'"
                
                if conditions is not None:
                    if conditions[idx_table] is not None:
                        #  conditions for each table, e.g. a query asking for only certain diagnoses or measurements to be included in the generator
                        query += f"AND {conditions[idx_table]}"
                                  
                logging.debug(f"Query: {query[:120] if len(query) > 120 else query}")

                # Load with polars
                #    This can cause timeout issues
                # df = pl.read_database(query=query, connection_uri='sqlite://' + self.db_path)

                # Load with pandas then convert to polars. 
                #   This lets us use the existing connection from the sqlite3 package and so we can specify longer timeout
                #   We also need to specify 'VALUE' is a float, as pandas will convert this to string (not all queries are on tables with VALUE)
                pandas_df = pd.read_sql_query(query, self.connection)
                if "VALUE" in pandas_df.columns:
                    pandas_df["VALUE"] = pandas_df["VALUE"].astype(float)
                df = pl.from_pandas(pandas_df)

                if len(df) > 0:
                    rows_by_table["lazy_" + table] = df.lazy()

            # Yield the fetched rows as a chunk
            yield distinct_value, rows_by_table
                
    def _collate_lazy_tables(self,
                             lazy_frames,   
                             study_inclusion_method              = None,
                             drop_empty_dynamic:            bool = True,
                             drop_missing_data:             bool = True,
                             **kwargs
                             ) -> pl.LazyFrame:
        """
        Merges LazyFrames containing patient records into a single frame.

        Parameters
        ----------
        lazy_frames : dict
            Dictionary of Polars LazyFrames containing static and dynamic (diagnosis and measurement table) records.
        study_inclusion_method : callable, optional
            Custom function to filter patient records (default is None).
        drop_empty_dynamic : bool, optional
            Whether to remove patients without dynamic events (default is True).
        drop_missing_data : bool, optional
            Whether to remove records with missing values (default is True).

        Returns
        -------
        pl.LazyFrame
            A combined LazyFrame containing static and dynamic patient data. See example.

        Example merged frame return
        ---------------------------

        .. table::

            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+
            | PRACTICE_ID | PATIENT_ID | VALUE                      | EVENT                                    | … | HEALTH_AU  | INDEX_DATE | START_DATE | END_DATE   |
            +=============+============+============================+==========================================+===+============+============+============+============+
            | 20429       | 22038164   | [60.0, 120.0, 100.0]       | ["Diastolic_blood_pressure_5", "..." ]  | … | South East | 2005-01-01 | 2005-01-01 | 2022-03-17 |
            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+
            | 20429       | 22038165   | [20.7, null, 144.0]        | ["Body_mass_index_3", "..."]            | … | South East | 2018-06-27 | 2018-06-27 | 2022-03-17 |
            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+
            | 20429       | 22038168   | [null, 90.0, 130.0]        | ["HYPERTENSION", "..."]                 | … | South East | 2011-04-23 | 2011-04-23 | 2022-03-17 |
            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+
            | 20429       | 22038169   | [25.9, 80.0, 120.0]        | ["Body_mass_index_3", "..."]            | … | South East | 2005-01-01 | 2005-01-01 | 2011-11-07 |
            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+
            | 20429       | 22038170   | [24.8, 76.0, null]         | ["Body_mass_index_3", "..."]            | … | South East | 2005-01-01 | 2005-01-01 | 2008-06-19 |
            +-------------+------------+----------------------------+------------------------------------------+---+------------+------------+------------+------------+

        Columns:
            - **PRACTICE_ID** (*int*): Unique identifier for the medical practice.
            - **PATIENT_ID** (*int*): Unique identifier for the patient.
            - **VALUE** (*list[float]*): Measurement values recorded.
            - **EVENT** (*list[str]*): Event descriptions.
            - **TODO: ADD MISSING**
            - **HEALTH_AU** (*str*): Health authority region.
            - **INDEX_DATE** (*datetime*): The index date of patient entry.
            - **START_DATE** (*datetime*): Start date of record.
            - **END_DATE** (*datetime*): End date of record.

        Notes:
            - `VALUE` contains lists of numeric measurements, where `null` represents missing values.
            - `EVENT` contains lists of medical events related to each patient.
            - The dataset is sorted by `PRACTICE_ID` and `PATIENT_ID`.
            - The Polars collection is **not** applied inside this function
            - We **do not** sort within this lazy operation, so the row order will not be deterministic
        """

        ##############################
        # GET THE LAZY POLARS FRAMES #
        ##############################

        # Static lazy frame, converting all dates to datetime format
        lazy_static = (
            lazy_frames["lazy_static_table"]
            .with_columns(pl.col("INDEX_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("START_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("END_DATE").str.to_datetime("%Y-%m-%d"))
            .with_columns(pl.col("YEAR_OF_BIRTH").str.to_datetime("%Y-%m-%d"))  
        )

        # Diagnosis lazy frame
        lazy_diagnosis = lazy_frames["lazy_diagnosis_table"] if "lazy_diagnosis_table" in lazy_frames.keys() else None

        # Measurement lazy frames
        measurement_keys = [key for key in lazy_frames if key.startswith("lazy_measurement_")]
        measurement_lazy_frames = [lazy_frames[key] for key in measurement_keys]
        lazy_measurement = pl.concat(measurement_lazy_frames) if len(measurement_lazy_frames) > 0 else None
        #    and optionally drop missing measurements
        if drop_missing_data and measurement_lazy_frames is not None:
            logging.debug("Dropping missing measurements")
            lazy_measurement = lazy_measurement.drop_nulls()

        #####################################
        # MERGE SOURCES OF TIME_SERIES DATA #
        #####################################
        
        # Merge all frames containing time series data
        if lazy_diagnosis is not None and lazy_measurement is not None:
            # Stacking the frames vertically. Value column in diagnostic is filled with null
            lazy_combined_frame = pl.concat([lazy_measurement, lazy_diagnosis], how="diagonal")
        elif lazy_measurement is not None:
            #
            lazy_combined_frame = lazy_measurement            
        elif lazy_diagnosis is not None:
            # Note: if we load diagnoses with no measurements, we are not concating values, so add this missing column
            lazy_combined_frame = lazy_diagnosis.with_columns(pl.lit(None).cast(pl.Utf8).alias('VALUE'))
        else:
            raise ValueError(f"Empty frames: lazy_diagnosis {lazy_diagnosis} and lazy_measurement {lazy_measurement}")

        # Dynamic lazy frame, converting all dates to datetime format
        lazy_combined_frame = (
            lazy_combined_frame
            .with_columns(pl.col("DATE").str.to_datetime("%Y-%m-%d"))
        )

        # Convert event date to time since birth by linking the dynamic diagnosis and measurement frames to the static one
        # Subtract the dates and create a new column for the result            
        lazy_combined_frame = (
            lazy_combined_frame
            .join(lazy_static.select(["PRACTICE_ID", "PATIENT_ID", "YEAR_OF_BIRTH"]),                     # Add birth year information to calculate relative event times
                  on=["PRACTICE_ID", "PATIENT_ID"], how="inner")
            .select([
                  (pl.col("DATE") - pl.col("YEAR_OF_BIRTH")).dt.days().alias("DAYS_SINCE_BIRTH"), "*"
                ])
            .drop("YEAR_OF_BIRTH")
            )

        # Drop events which occur at unrealistic ages
        lazy_combined_frame = (
            lazy_combined_frame
            .filter(pl.col("DAYS_SINCE_BIRTH") > -365)
            .filter(pl.col("DAYS_SINCE_BIRTH") < 125*365)
        )
        
        #################
        # FILTER FRAMES #
        #################

        # Reduce based on study criteria. You may pass your own custom criteria method
        if study_inclusion_method is not None:
            lazy_static, lazy_combined_frame = study_inclusion_method(lazy_static, lazy_combined_frame)

        # Remove patients without multiple events
        lazy_combined_frame = (
            lazy_combined_frame.groupby("PATIENT_ID")
                .agg(pl.count("PATIENT_ID").alias("count"))
                .filter(pl.col("count") > 1)
                .join(lazy_combined_frame, on="PATIENT_ID", how="inner")
                .select(lazy_combined_frame.columns)
        )
        
        #############
        # AGGREGATE #
        #############

        # Remove entries before conception (negative to include pregnancy period, e.g. diagnosed with genetic condition pre-birth)
        agg_cols = ["VALUE", "EVENT", "DAYS_SINCE_BIRTH", "DATE"]
        lazy_combined_frame = (
            lazy_combined_frame
            .sort("DAYS_SINCE_BIRTH")
            .groupby(["PRACTICE_ID", "PATIENT_ID"])
            .agg(agg_cols)                                                # Turn into lists
            .sort(["PRACTICE_ID", "PATIENT_ID"])                          # make lazy collection output deterministic
        )
        
        if drop_empty_dynamic:
            logging.debug("Removing patients with no observed events")
            lazy_combined_frame = lazy_combined_frame.drop_nulls()

        ############################
        # MERGE STATIC AND DYNAMIC #
        ############################
        # Align the polars frames, linking on patient idenfitifer, then concatentate into a single frame (dropping repeated identifier)
        #    If identifier exists in one but not the other, default behaviour is to fill with null, these are handled by filtering later
        #    All these operations are performed lazily
        lazy_combined_frame = lazy_combined_frame.join(lazy_static, on=["PRACTICE_ID", "PATIENT_ID"], how="inner")


        return lazy_combined_frame

    def get_meta_information(self,
                             practice_ids:          Optional[list] = None,
                             static:                bool = True,
                             diagnoses:             bool = True,
                             measurement:           bool = True,
                            ) -> dict:
        """
        Collects metadata from the SQLite database, such as distributions of diagnoses and measurements.

        Parameters
        ----------
        practice_ids : list, optional
            List of practice IDs to filter metadata collection (default is None).
        static : bool, optional
            Whether to collect static patient information (default is True).
        diagnoses : bool, optional
            Whether to collect diagnosis-related metadata (default is True).
        measurement : bool, optional
            Whether to collect measurement-related metadata (default is True).

        Returns
        -------
        dict
            A dictionary containing metadata tables.
        """

        # Standardisation is TODO
        logging.info("\n\nCollecting meta information from database. This will be used for tokenization and (optionally) standardisation.")
        
        # Initialise meta information 
        meta_information = {}

        # TODO: calculate these only on training splits! Especially if standardisation gets implemented
        if practice_ids is not None:
            raise NotImplementedError
            # condition = f"PRACTICE_ID IN ({",".join([f"'{dv}'" for dv in practice_ids])});" 

        if static is True:
            logging.info("\t Static meta information")
            static_meta = {}
            for categorical_covariate in ["SEX", "IMD", "ETHNICITY"]:                
                result = self._extract_AGG("static_table", identifier_column=categorical_covariate, aggregations="COUNT(*)")
                category, counts = zip(*result)
                static_meta[categorical_covariate] = pd.DataFrame({"category": category,
                                                                   "count": [i for i in counts],
                                                                   })
            meta_information["static_table"] = static_meta
            logging.debug(f"static_meta: \n{static_meta}")
        
        if diagnoses is True:
            logging.info("\t Diagnosis meta information")
            result = self._extract_AGG("diagnosis_table", identifier_column="EVENT", aggregations="COUNT(*)")
            diagnoses, counts = zip(*result)
            diagnosis_meta = pd.DataFrame({"event": diagnoses,
                                           "count": [i for i in counts],
                                           })
            meta_information["diagnosis_table"] = diagnosis_meta
            logging.debug(f"diagnosis_meta: \n{diagnosis_meta}")
            
        if measurement is True:
            logging.info("\t Measurements meta information")
            measurements = []                                                # List of measurements
            counts, obs_counts = [], []                                      # List of measurement counts, and how many of those have observed values
            obs_digest, obs_mins, obs_maxes, obs_means = [], [], [], []      # T-digest data structure for approximate quantiles, and exact statistics for observed values
            cutoff_lower, cutoff_upper = [], []                              # Cut-off values based on approximate quantiles, used for filtering and standardisation

            for table in tqdm(self.measurement_table_names, desc="Measurements".rjust(50), total=len(self.measurement_table_names)):

                # Get the measurement name from the table's name
                measurement = table[12:]

                result = self._extract_AGG(table, aggregations=f"COUNT(*), COUNT(VALUE)")  #  condition=condition
                table_counts, table_counts_obs = result[0]

                if table_counts_obs > 0:
                    # Online accumulation to approximate quantiles which will then be used for standardisation and outlier removal.
                    digest = self._t_digest_values(table)
                    # From summary statistics get the standardisation limits
                    iqr = digest.percentile(75) - digest.percentile(25)
                    digest_lower = digest.percentile(25) - 1.5*iqr
                    digest_upper = digest.percentile(75) + 1.5*iqr
                    
                    # Get total number of entries, number of observed values, and statistics, for each unique event in measurement table
                    result = self._extract_AGG(table, aggregations=f"MIN(VALUE), MAX(VALUE), AVG(VALUE)")  #  condition=condition
                    table_min_obs, table_max_obs, table_mean_obs = result[0]
                else:
                    # Catch cases where there are no values for this measurement
                    digest = None
                    digest_lower, digest_upper = None, None
                    # Get total number of entries, number of observed values, and statistics, for each unique event in measurement table
                    table_min_obs, table_max_obs, table_mean_obs = None, None, None
                    
                # Collate each tables summary statistics, here we have assumed that each measurement table contains a unique event
                measurements.append(measurement)
                counts.append(table_counts)
                obs_counts.append(table_counts_obs)
                obs_digest.append(digest)
                obs_mins.append(table_min_obs)
                obs_maxes.append(table_max_obs)
                obs_means.append(table_mean_obs)
                cutoff_lower.append(digest_lower)
                cutoff_upper.append(digest_upper)

            measurement_meta = pd.DataFrame({"event": measurements,
                                             "count": counts,
                                             "count_obs": obs_counts,
                                             "digest": obs_digest,
                                             "min": obs_mins,
                                             "max": obs_maxes,
                                             "mean": obs_means,
                                             "approx_lqr": cutoff_lower,
                                             "approx_uqr": cutoff_upper
                                             })

            meta_information["measurement_tables"] = measurement_meta
            logging.debug(f"measurement_meta: \n{measurement_meta}")
            
        return meta_information
                             
    
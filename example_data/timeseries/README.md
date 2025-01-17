# SurvivEHR-ExampleData

For each measurement, tests and medications considered create a separate .csv file.

Each of these .csv files should have three columns:

PRACTICE_PATIENT_ID: As elsewhere this is a concatenation of PRACTICE_ID and PATIENT_ID strings found in the baseline static data, e.g. p20960_1

EVENT_DATE: Formatted as "yyyy-mm-dd". This is the date the measurement, medication or test was taken. Unlike some other cases, there should be no missing data here.

Value: The numeric value associated with the measurement or test. For example, BMI. These can be left empty if the associated file either does not have a corresponding measurement (such as for some medications), or the value is missing from the database.

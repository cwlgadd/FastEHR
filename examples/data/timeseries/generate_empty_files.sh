#!/bin/bash

# Check if the input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_with_names.txt>"
    exit 1
fi

# Input file containing the list of filenames
input_file=$1

# Create the subfolder if it doesn't exist
output_folder="measurement_tests_medications"
mkdir -p "$output_folder"

# Loop through each line in the file
while IFS= read -r filename || [[ -n "$filename" ]]; do
    # Ensure the filename ends with .csv
    if [[ ! "$filename" =~ \.csv$ ]]; then
        filename="${filename}.csv"
    fi

    touch $output_folder/$filename
    
    # Create an empty CSV file with the specified columns in the subfolder
    #{
    #    echo -e "PRACTICE_PATIENT_ID,EVENT_DATE,Value\r"
    #    echo -e "p20960_1,2006-01-28,\r"  # Add example data to the file
    #} > "$output_folder/$filename"

    echo "Created file with columns: $output_folder/$filename"
done < "$input_file"

echo "All files created successfully in the folder: $output_folder."

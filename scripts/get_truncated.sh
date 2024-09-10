#!/bin/bash

# Check if the input directory argument is provided
if [ -z "$1" ]; then
    echo "Error: Input directory argument is missing."
    exit 1
fi

# Check if the input directory is a valid path and contains at least one detected_boundaries*.csv file
input_dir=$1
if [ ! -d "$input_dir" ] || [ -z "$(ls "$input_dir"/detected_boundaries*.csv 2>/dev/null)" ]; then
    echo "Error: The provided path is not a directory or contains no detected_boundaries*.csv files."
    exit 1
fi

output_file=${input_dir}/truncated_read_ids.csv

# Initialize the output file with a header
echo "read_id" > "$output_file"

# Get the total number of files
total_files=$(ls "$input_dir"/detected_boundaries*.csv 2>/dev/null | wc -l)
current_file=0

# Loop through each detected_boundaries.csv file in the directory
for file in "$input_dir"/detected_boundaries*.csv; do
    # Check if the file exists
    if [[ -f "$file" ]]; then
        # Extract read_id where polya_truncated is true and append to output file
        # col1 is read_id, col18 is polya_truncated
        awk -F, 'NR > 1 && $18 == "True" {print $1}' "$file" >> "$output_file"
        
        # Update progress
        current_file=$((current_file + 1))
        progress=$((current_file * 100 / total_files))
        printf "\rProcessing file %d of %d: %d%% complete" "$current_file" "$total_files" "$progress"
    fi
done

echo ""
echo "Output written to $output_file"
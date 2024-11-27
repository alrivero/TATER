#!/bin/bash

# Usage: ./filter.sh <txt_file_with_paths> <output_json> <prepend_string>
# Example: ./filter.sh file_paths.txt add_and_remove.json /path/to/prepend/

# Input arguments
TXT_FILE=$1
OUTPUT_JSON=$2
PREPEND_STRING=$3

# Check if the input file exists
if [ ! -f "$TXT_FILE" ]; then
    echo "Error: File $TXT_FILE not found!"
    exit 1
fi

# Ensure the output JSON file path is writable
if [ -f "$OUTPUT_JSON" ]; then
    echo "Output JSON file $OUTPUT_JSON already exists. It will be updated."
else
    echo "{}" > "$OUTPUT_JSON"  # Create an empty JSON if it doesn't exist
    echo "Created new JSON file: $OUTPUT_JSON"
fi

# Ensure the prepend string is valid
if [[ -z "$PREPEND_STRING" ]]; then
    echo "Error: Prepend string not provided!"
    exit 1
fi

# Ensure the prepend string ends with a '/'
PREPEND_STRING="${PREPEND_STRING%/}/"

# Loop through each line in the text file and call the Python script
while IFS= read -r file_path; do
    # Prepend the string to the file path
    modified_path="${PREPEND_STRING}${file_path}"
    echo "Processing file: $modified_path"
    python3 filter.py "$modified_path" "$OUTPUT_JSON"
    if [ $? -ne 0 ]; then
        echo "Error processing $modified_path. Skipping."
        continue
    fi
done < "$TXT_FILE"

echo "Processing complete!"
import json
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description="Swap 'training' and 'validation' in the subset of JSON data.")
parser.add_argument('input_file', type=str, help="Path to the input JSON file.")
parser.add_argument('output_file', type=str, help="Path to save the modified JSON file.")

# Parse the command-line arguments
args = parser.parse_args()

# Load the JSON data from the provided input file
with open(args.input_file, 'r') as f:
    data = json.load(f)

# Iterate through the 'database' entries and update the 'subset' field
for key, value in data['database'].items():
    if value['subset'] == 'training':
        value['subset'] = 'validation'
    elif value['subset'] == 'validation':
        value['subset'] = 'training'

# Save the updated JSON data to the provided output file
with open(args.output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Updated data saved to {args.output_file}")

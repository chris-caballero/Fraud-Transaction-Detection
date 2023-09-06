import csv
import json

# Define the CSV input file and JSON output file
csv_file = '../../data/transactions-cleaned.csv'
json_file = '../../data/transactions-cleaned.json'

# Initialize a list to store the JSON records
json_data = []

# Read the CSV file and convert it to JSON
with open(csv_file, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        json_data.append(row)

# Write the JSON data to a JSON file (optional)
with open(json_file, mode='w') as json_file:
    json.dump(json_data, json_file, indent=4)

# Print the JSON data (optional)
# print(json.dumps(json_data, indent=4))


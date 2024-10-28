import json
import os

def extract_first_json_format(dataset_folder):
    # Get the list of files in the dataset folder
    files = os.listdir(dataset_folder)
    
    # Filter for JSON files
    json_files = [f for f in files if f.endswith('.json')]
    
    # Check if there are any JSON files
    if not json_files:
        return None  # No JSON files found
    
    # Load the first JSON file
    first_json_file = json_files[0]
    with open(os.path.join(dataset_folder, first_json_file), 'r') as file:
        data = json.load(file)
    
    # Return only the keys of the JSON data
    return list(data.keys())  # Extracting only the keys

print(extract_first_json_format("Dataset"))

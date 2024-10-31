import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv

# Path to your FHIR JSON files
fhir_data_folder = "Dataset"
load_dotenv()
url = "https://api.arliai.com/v1/chat/completions"
ARLIAI_API_KEY = os.environ.get('API_KEY')

# Load all the JSON files in the directory
def load_fhir_data(folder_path):
    fhir_records = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                try:
                    record = json.load(f)
                    fhir_records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error reading {file_name}: {e}")
    return fhir_records

# Load the pre-trained model for creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_fhir(bundle):
    """
    Extract relevant fields based on the FHIR resource type from a Bundle.
    """
    extracted_texts = []  # Changed to store individual texts
    
    if bundle.get('resourceType') == 'Bundle' and 'entry' in bundle:
        for entry in bundle['entry']:
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType', 'Unknown')

            if resource_type == 'Patient':
                family_name = resource.get('name', [{}])[0].get('family', 'Unknown')
                given_names = resource.get('name', [{}])[0].get('given', [])
                name = f"{family_name}, {' '.join(given_names)}"
                gender = resource.get('gender', 'Unknown')
                birth_date = resource.get('birthDate', 'Unknown')
                extracted_texts.append(f"Patient Name: {name}, Gender: {gender}, Birth Date: {birth_date}")

            elif resource_type == 'Condition':
                condition_code = resource.get('code', {}).get('text', 'Unknown condition')
                onset_date = resource.get('onsetDateTime', 'Unknown onset date')
                extracted_texts.append(f"Condition for {name}: {condition_code}, Onset: {onset_date}")

            elif resource_type == 'MedicationRequest':
                medication = resource.get('medicationCodeableConcept', {}).get('text', 'Unknown medication')
                dosage = resource.get('dosageInstruction', [{}])[0].get('text', 'Unknown dosage')
                extracted_texts.append(f"Medication for {name}: {medication}, Dosage: {dosage}")

            elif resource_type == 'Observation':
                observation_type = resource.get('code', {}).get('text', 'Unknown observation')
                value = resource.get('valueQuantity', {}).get('value', 'Unknown value')
                extracted_texts.append(f"Observation for {name}: {observation_type}, Value: {value}")
    
    return extracted_texts  # Return a list of texts

# Load the FHIR data
fhir_data = load_fhir_data(fhir_data_folder)

# Extract texts for each record and create embeddings for all FHIR records
texts = []
for record in fhir_data:
    extracted_texts = extract_text_from_fhir(record)
    texts.extend(extracted_texts)  # Flatten the list to include all extracted texts


# Save the first patient's text to a file for reference
# with open('patient0.txt', 'w') as file:
#     file.write("\n".join(extract_text_from_fhir(fhir_data[0])))
# file.close()

# Create embeddings for each extracted text
embeddings = model.encode(texts, show_progress_bar=True)

# Convert embeddings to a numpy array and normalize for cosine similarity
embedding_matrix = np.array(embeddings, dtype='float32')
faiss.normalize_L2(embedding_matrix)  # Normalize embeddings

# Create a FAISS index using Inner Product for cosine similarity
d = embedding_matrix.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatIP(d)   # IP for cosine similarity
index.add(embedding_matrix)    # Add embeddings to the index

print(f"Total records indexed: {index.ntotal}")

# Function to query the index
def search(query, k=10):
    """
    Search for the top-k closest records to the given query.
    """
    # Create an embedding for the query and normalize it
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype='float32')
    faiss.normalize_L2(query_embedding)  # Normalize query embedding for cosine similarity
    
    # Perform the search
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the top-k closest texts
    results = [(texts[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    
    return results

def generate_answer_with_llm(query, retrieved_entries):
    payload = json.dumps({
    "model": "Meta-Llama-3.1-8B-Instruct",
    "messages": [
            {
            "role": "system",
            "content": f"""
            You are an assistant with medical knowledge. Review the provided FHIR data entries and answer the query concisely, drawing only from the information directly relevant to the question.

            User Query: "{query}"

            Retrieved Data:
            {retrieved_entries}

            Answer in a structured and clear format, highlighting key information without including unrelated data points.
            """
            },

    ],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 1024,
    "stream": False
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {ARLIAI_API_KEY}"
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    # Extract and return only the text from the response
    if response.status_code == 200:
        try:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')  # if the response is JSON
        except json.JSONDecodeError:
            return response.text  # if it's plain text
    else:
        return f"Error: {response.status_code} - {response.text}"

# Example: Query the index
query_text = input("Query :")

while query_text != "-1":
    top_k_results = search(query_text)
    retrieved_entries = top_k_results  # Keep all results
    print(f"\n\nRAG RESULT: {retrieved_entries}")
    print(f"\n\nQUERY: {query_text}")
    print(generate_answer_with_llm(query_text, retrieved_entries))
    print("\n\n")
    query_text = input("Query :")

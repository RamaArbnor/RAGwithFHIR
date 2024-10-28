import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to your FHIR JSON files
fhir_data_folder = "Dataset"

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
    extracted_text = []
    
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
                extracted_text.append(f"Patient: {name}, Gender: {gender}, Birth Date: {birth_date}")

            elif resource_type == 'Condition':
                condition_code = resource.get('code', {}).get('text', 'Unknown condition')
                onset_date = resource.get('onsetDateTime', 'Unknown onset date')
                extracted_text.append(f"Condition: {condition_code}, Onset: {onset_date}")

            elif resource_type == 'MedicationRequest':
                medication = resource.get('medicationCodeableConcept', {}).get('text', 'Unknown medication')
                dosage = resource.get('dosageInstruction', [{}])[0].get('text', 'Unknown dosage')
                extracted_text.append(f"Medication: {medication}, Dosage: {dosage}")

            elif resource_type == 'Observation':
                observation_type = resource.get('code', {}).get('text', 'Unknown observation')
                value = resource.get('valueQuantity', {}).get('value', 'Unknown value')
                extracted_text.append(f"Observation: {observation_type}, Value: {value}")
    
    return " ".join(extracted_text)

# Load the FHIR data
fhir_data = load_fhir_data(fhir_data_folder)

# Extract text and create embeddings for all FHIR records
texts = [extract_text_from_fhir(record) for record in fhir_data]
embeddings = model.encode(texts, show_progress_bar=True)

# Convert embeddings to a numpy array and normalize for cosine similarity
embedding_matrix = np.array(embeddings, dtype='float32')
faiss.normalize_L2(embedding_matrix)  # Normalize embeddings

# Create a FAISS index using Inner Product for cosine similarity
d = embedding_matrix.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatIP(d)   # IP for cosine similarity
index.add(embedding_matrix)    # Add embeddings to the index

print(f"Total records indexed: {index.ntotal}")

# Function to query the index
def search(query, k=5):
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
    prompt = f"""
    You are an assistant with medical knowledge. Use the following FHIR data entries to provide a summarized answer.

    User Query: "{query}"

    Retrieved Data:
    {retrieved_entries}

    Provide a structured answer relevant to the query.
    """
    
    answer = ''
    return answer

# Example: Query the index
query_text = "Retrieve information on general health metrics like height, weight, BMI, blood pressure, heart rate, and any recent viral infections for King743, Bart73 Caleb651"
top_k_results = search(query_text)

print(top_k_results[0])

# # Print the results
# for text, distance in top_k_results:
#     print(f"Distance (Cosine Similarity): {distance:.4f}\n{text}\n")

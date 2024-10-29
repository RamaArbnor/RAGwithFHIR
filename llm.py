import requests
import json
import os

ARLIAI_API_KEY = os.environ.get('API_KEY')
# ARLIAI_API_KEY = ""




url = "https://api.arliai.com/v1/chat/completions"

def generate_answer_with_llm(query, retrieved_entries):
    payload = json.dumps({
    "model": "Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": f"""
        You are an assistant with medical knowledge. Use the following FHIR data entries to provide a summarized answer.

        User Query: "{query}"

        Retrieved Data:
        {retrieved_entries}

        Provide a structured answer relevant to the query.
        """},

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

print(generate_answer_with_llm("Test", "Test"))
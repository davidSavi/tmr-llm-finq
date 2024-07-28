import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_json_opt(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None

def get_embedding(text, client):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model="text-embedding-3-small").data[0].embedding

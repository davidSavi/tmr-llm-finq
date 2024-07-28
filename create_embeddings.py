import pandas as pd
import uuid
import itertools
import os
import json
from openai import OpenAI

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from utils import get_embedding

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

def safe_create_index(index_name, pc):
  if index_name not in pc.list_indexes().names():
      pc.create_index(
          name=index_name,
          dimension=1536,
          metric="cosine",
          spec=ServerlessSpec(
              cloud='aws', 
              region='us-east-1'
          ) 
      ) 

def flatten_text_chunks(data):
    pre_text = data.get("pre_text", [])
    post_text = data.get("post_text", [])
    textChunk =  [
        {"id": str(uuid.uuid4()), "filename": data["filename"], "text": text}  
        for _, text in enumerate(pre_text + post_text)
        if len(text) >= 2
    ]
    textChunk.append({"id": str(uuid.uuid4()), "filename": data["filename"], "text":" ".join([" ".join(row) for row in data["table"]])})

    return textChunk

def process_traininig(data_array):
    textChunks = []
    seen_ids = set()
    for data in data_array:
        if data["filename"] not in seen_ids:
            textChunks.extend(flatten_text_chunks(data))
            seen_ids.add(data["filename"])
    
    return textChunks

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def upsert(index, textChunks):
  df = pd.json_normalize(textChunks)
  df['embedding'] = df['text'].apply(lambda x: get_embedding(x, client))
  records = [
      {
          'id': str(row['id']),
          'values': row['embedding'],
          'metadata': {'filename': row['filename']}
      }
      for _, row in df.iterrows()
  ]
  print(len(records))

  for ids_vectors_chunk in chunks(records, batch_size=240):
      index.upsert(vectors=ids_vectors_chunk) 

if __name__ == "__main__":
  train_array = load_json_file('data/train.json')
  index_name = "finq-index-all"
  textChunks = process_traininig(train_array)
  
  safe_create_index(index_name, pc)
  index = pc.Index(index_name)
  
  upsert(index, textChunks)

import json
from openai import OpenAI
from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
import os

from utils import get_embedding, load_json

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

index_name = "finq-index-all"
index = pc.Index(index_name)


def get_response(prompt, max_tokens):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=max_tokens,
        model="gpt-4o-mini",
        temperature=0.0
    )
    return response.choices[0].message.content

def find_similarities(text, top_k=5):
    return index.query(vector = get_embedding(text, client), top_k = top_k, include_metadata=True)

def get_llm_response(context_dict, question, top_k=5, max_response_tokens=150):
    query_result = find_similarities(question, top_k)
    context = " ".join(context_dict[item['metadata']['filename']]['context'] for item in query_result['matches'])
    prompt = f"Respond with the result only, no explanation of how the calculation was done. It might be a percentage, a decimal number or amount of money. If it's money, make sure you use a currency symbol and  Make sure any calculations are mathematically accurate to the second decimal: '{question}' from the following context: {context}."
    return {
        "response": get_response(prompt, max_response_tokens),
        "context": context
        }

def generate_llm_response_report(questions, context_dict):
    for item in questions:
        question = item['question']
        if question in question_lookup:
            item['answer'] = question_lookup[question]['answer']
            item['id'] = question_lookup[question]['id']

    llm_responses = []
    with tqdm(total=len(questions), desc="Processing relevant questions") as pbar:
      for q in questions:
        llm_response = get_llm_response(context_dict,q["question"])
        llm_responses.append({
              "question": q["question"],
              "answer":llm_response["response"],
              "context": llm_response["context"],
              "expectedAnswer":q["answer"],
              "id": q['id']
            })
        pbar.update(1)

    return llm_responses

def generate_similarity_report(questions):
    report = []
    with tqdm(total=len(questions), desc="Processing questions") as pbar:
      for row in questions:
          question = row['question']
          correct_filename = row['filename']
                
          query_result = find_similarities(question, 10)
          
          current_rank = 0
          current_score = 0.0
          
          matched = next((m for rank, m in enumerate(query_result['matches']) if m['metadata']['filename'] == correct_filename), None)
          if matched:
              current_rank = query_result['matches'].index(matched) + 1
              current_score = matched['score']

          report.append({
              "question": question,
              "filename": correct_filename,
              "rank": current_rank,
              "score": current_score
          })
          pbar.update(1)

    return report

if __name__ == "__main__":
    questions = load_json('data/questions.json')
    context_dict = {item['filename']: item for item in load_json('data/contexts.json')}
    similarity_report = generate_similarity_report(questions)
    
    question_lookup = {item['question']: item for item in questions}

    relevant_questions = list(filter(lambda x: x['rank'] != 0, similarity_report))
    llm_responses = generate_llm_response_report(relevant_questions, context_dict)

    with open('reports/similarity_report.json', 'w') as f:
        json.dump(similarity_report, f, indent=4)
    with open('reports/llm_response_report.json', 'w') as f:
        json.dump(llm_responses, f, indent=4)
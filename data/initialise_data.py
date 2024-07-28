import numpy as np
import pandas as pd
import json


def convert_table_to_json(table):
    first_column_name = table[0][0] if table[0][0] else "Category"
    headers = table[0][1:]
    result = []
    for row in table[1:]:
        row_dict = {first_column_name: row[0]}
        for i, value in enumerate(row[1:], start=1):
            row_dict[headers[i-1]] = value
        
        result.append(row_dict)
    
    return result

def stringify_array(array):
    return "\n".join(array)

def count_words(text):
    return len(text.split())

def create_context(data):
    pre_text = data.get("pre_text", [])
    post_text = data.get("post_text", [])
    table_json = convert_table_to_json(data.get("table", []))
    
    context = f"{stringify_array(pre_text)}\n{json.dumps(table_json, indent=4)}\n{stringify_array(post_text)}"
    token_amount = count_words(context)

    context = {
        "context": context,
        "filename": data["filename"],
        "tokenAmount": token_amount,
    }
    
    return context

def process_training(data_array):
    context = []
    seen_ids = set()
    for data in data_array:
        if data["filename"] not in seen_ids:
            context.append(create_context(data))
            seen_ids.add(data["filename"])
    
    return context


def extract_questions(item):
    id_base = item["id"]
    filename = item["filename"]
    qa_entries = []
    for key, value in item.items():
        if key.startswith("qa"):
            qa_entry = {
                "id": f"{id_base}_{key}",
                "filename": filename,
                "question": value["question"],
                "answer": value["answer"]
            }
            qa_entries.append(qa_entry)    
    return qa_entries

def load_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json_to_file(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
  data_array = load_json_file('train.json')
  questions = []
  for item in data_array:
      questions.extend(extract_questions(item))

  context = process_training(data_array)
  write_json_to_file(context, 'contexts.json')
  #Remove this limit to generate the full set uf 3900 questions
  limit = 200
  write_json_to_file(questions[:limit], 'questions.json')
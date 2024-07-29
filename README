# LLM based mathematical precision evaluator

# Overview

This repository contains a set of Python scripts designed to generate embeddings from financial data extracted from `train.json` and to create prompts with the relevant context. The primary goal is to generate numeric responses, typically financial calculations, and to measure how closely these responses match our expected answers.

## Key Components

1. **[OpenAI APIs](https://openai.com/)**: Used for generating embeddings and serving as the Large Language Model (LLM) for prompts.
2. **[Pinecone](https://www.pinecone.io/)**: Utilized for storing embeddings and providing fast similarity search to filter the context passed to the LLM. This method is similar to retrieval augmentation but without actual augmentation.

## How It Works

- **Embedding Generation**: Embeddings are generated per line of `pre_text`, `post_text` and `table` fields.
- **Prompt Creation**: For each prompt, the full context from the top N similar neighboring files is merged and passed to the LLM.

## Reports and Results

- **End-to-End Flow**: A limited set of 200 questions from `train.json` can be run as a test using GitHub Actions.
- **Full Dataset Reports**: The `reports_full_dataset` directory contains the results of running all 3900 questions. The `reports_full_dataset/process_*` scripts can be executed to inspect these results.

## Conclusion
I was able to achieve ~75% accuracy on the similarity search and ~30% accuracy (of the 75% questions) on the LLM response with a 3% precision error threahold. The number is slightly higher with 5%.

I was curious to see how accurate calculations we can get using prompt engineering from complex contexts. There're two ways of improving this score using this solution. One is to increase the similarity search accuracy, and the second one is to improve the prompt to get better calculation accuracy. Currently we might be passing down too much context to `gpt4o-small` and causing **information overload**.

## Table of Contents
- [Overview](#overview)
- [Scripts](#scripts)
  - [initialise_data.py](#initialise_data.py)
  - [create_embeddings.py](#create_embeddings.py)
  - [generate_reports.py](#generate_reports.py)
  - [process_similarity_report.py](#process_similarity_report.py)
  - [process_llm_response_report.py](#process_llm_response_report.py)
- [Usage](#usage)

## Scripts

### [initialise_data.py](data/initialise_data.py)
Parses and transforms the `train.json` and outputs and deduplicates the `questions.json`:   

```json
   [ {
        "id": "Single_AAPL/2002/page_23.pdf-1_qa",
        "filename": "AAPL/2002/page_23.pdf",
        "question": "what was the percentage change in net sales from 2000 to 2001?",
        "answer": "-32%"
    },...]
```
and the `contexts.json`:
```json
   [{
        "context": "during the years ended december 31...",
        "filename": "SLG/2013/page_133.pdf",
        "tokenAmount": 971
    },...]
```

### [create_embeddings.py](create_embeddings.py)
This script should only be ran if we need to re-create or update the embeddings vector database in Pinecone. It contains all the `pre-text`, `post-text`, and `table` fields of the items from `train.json` and can be associated with `questions.json` and `contexts.json` using the filename as a key. 

### [generate_reports.py](generate_reports.py)

This function requires the output of `initialise_data.py`.
It generates two reports: 
- `reports/similarity_report.json` report which compares the `filename` field from the `questions.json` with the `filename` field in the metadata from the similarity query results from the pinecone index. If there's no match in the top 10 results it returns a score of 0.
- `reports/llm_response_report.json` which processes only with the questions that had higher score than 0 (2900 out of 3600 in the current dataset). It has the answer from `questions.json` as the expeted answer and it also contains the answer from the OpenAI call. 

### [process_similarity_report.py](reports/process_similarity_report.py)

Processes te `similarity_report.json` file. Here's the example output from the full 3600 question dataset:
```
Total questions: 3965
Rank 1 Total: 1551 questions (39.12%):
  Score 0.7 - 0.75: 409 questions (10.32%)
  Score > 0.75: 600 questions (15.13%)
  Score 0.65 - 0.7: 320 questions (8.07%)
  Score 0.5 - 0.6: 66 questions (1.66%)
  Score 0.6 - 0.65: 155 questions (3.91%)
  Score < 0.5: 1 questions (0.03%)
Rank 2 Total: 520 questions (13.11%):
  Score > 0.75: 86 questions (2.17%)
  Score 0.65 - 0.7: 158 questions (3.98%)
  Score 0.6 - 0.65: 95 questions (2.40%)
  Score 0.5 - 0.6: 34 questions (0.86%)
  Score 0.7 - 0.75: 144 questions (3.63%)
  Score < 0.5: 3 questions (0.08%)
Rank 3 Total: 280 questions (7.06%):
  Score 0.7 - 0.75: 39 questions (0.98%)
  Score 0.65 - 0.7: 86 questions (2.17%)
  Score 0.6 - 0.65: 70 questions (1.77%)
  Score 0.5 - 0.6: 43 questions (1.08%)
  Score > 0.75: 37 questions (0.93%)
  Score < 0.5: 5 questions (0.13%)
Rank 4 Total: 155 questions (3.91%):
  Score 0.7 - 0.75: 26 questions (0.66%)
  Score 0.6 - 0.65: 50 questions (1.26%)
  Score 0.65 - 0.7: 50 questions (1.26%)
  Score 0.5 - 0.6: 20 questions (0.50%)
  Score > 0.75: 7 questions (0.18%)
  Score < 0.5: 2 questions (0.05%)
Rank 5 Total: 123 questions (3.10%):
  Score 0.65 - 0.7: 30 questions (0.76%)
  Score 0.6 - 0.65: 44 questions (1.11%)
  Score 0.7 - 0.75: 11 questions (0.28%)
  Score 0.5 - 0.6: 33 questions (0.83%)
  Score < 0.5: 1 questions (0.03%)
  Score > 0.75: 4 questions (0.10%)
Rank 6 Total: 87 questions (2.19%):
  Score 0.5 - 0.6: 14 questions (0.35%)
  Score 0.65 - 0.7: 31 questions (0.78%)
  Score 0.7 - 0.75: 6 questions (0.15%)
  Score > 0.75: 5 questions (0.13%)
  Score 0.6 - 0.65: 31 questions (0.78%)
Rank 7 Total: 70 questions (1.77%):
  Score 0.5 - 0.6: 25 questions (0.63%)
  Score 0.6 - 0.65: 18 questions (0.45%)
  Score 0.65 - 0.7: 14 questions (0.35%)
  Score 0.7 - 0.75: 9 questions (0.23%)
  Score > 0.75: 4 questions (0.10%)
Rank 8 Total: 68 questions (1.72%):
  Score 0.7 - 0.75: 7 questions (0.18%)
  Score 0.6 - 0.65: 26 questions (0.66%)
  Score < 0.5: 1 questions (0.03%)
  Score 0.65 - 0.7: 21 questions (0.53%)
  Score 0.5 - 0.6: 11 questions (0.28%)
  Score > 0.75: 2 questions (0.05%)
Rank 9 Total: 63 questions (1.59%):
  Score 0.6 - 0.65: 20 questions (0.50%)
  Score 0.5 - 0.6: 14 questions (0.35%)
  Score 0.7 - 0.75: 8 questions (0.20%)
  Score 0.65 - 0.7: 19 questions (0.48%)
  Score > 0.75: 2 questions (0.05%)
Rank 10 Total: 48 questions (1.21%):
  Score 0.5 - 0.6: 20 questions (0.50%)
  Score 0.65 - 0.7: 6 questions (0.15%)
  Score < 0.5: 5 questions (0.13%)
  Score > 0.75: 1 questions (0.03%)
  Score 0.6 - 0.65: 13 questions (0.33%)
  Score 0.7 - 0.75: 3 questions (0.08%)
Mismatched: 1000 questions (25.22%)
```
### [process_llm_response_report.py](reports/process_llm_response_report.py)

Processes te `llm_response_report.json` file. Here's the example output from the full 2965/3600 matched question dataset:
```
Total questions that pass similarity: 2965
Successfully met 3% error tolerance: 1022
Failed 3% error tolerance: 1943
```

## Usage
1. **Run the github action (limited to 200 questions)**
   https://github.com/davidSavi/tmr-llm-finq/actions/workflows/run_report.yml
2. **Register your own Pinecone and OpenAI account**
  You can use your own API keys `PINECONE_API_KEY` and `OPENAI_API_KEY` and once you ran [create_embeddings.py](create_embeddings.py) you can run the scripts the same way they're run in the Github action
3. **View the reports**
  You can find the reports generated in the `reports_full_dataset` directory and can run the process scripts to view the results
 


import unittest
from unittest.mock import patch
import os
# Sample data for testing
questions_json = [
    {
        "id": "Single_AAPL/2002/page_23.pdf-1_qa",
        "filename": "AAPL/2002/page_23.pdf",
        "question": "what was the percentage change in net sales from 2000 to 2001?",
        "answer": "-32%"
    },
    {
        "id": "Single_UPS/2009/page_33.pdf-2_qa",
        "filename": "UPS/2009/page_33.pdf",
        "question": "what was the difference in percentage cumulative return on investment for united parcel service inc . compared to the s&p 500 index for the five year period ended 12/31/09?",
        "answer": "-26.16%"
    },
    {
        "id": "Double_UPS/2009/page_33.pdf_qa_0",
        "filename": "UPS/2009/page_33.pdf",
        "question": "what is the roi of an investment in ups in 2004 and sold in 2006?",
        "answer": "-8.9%"
    },
]

contexts_json = [
    {
        "filename": "AAPL/2002/page_23.pdf",
        "context": "Context for AAPL 2002 page 23"
    },
    {
        "filename": "UPS/2009/page_33.pdf",
        "context": "Context for UPS 2009 page 33"
    }
]
context_dict = {item['filename']: item for item in contexts_json}

class TestLLMFunctions(unittest.TestCase):

    @patch('utils.load_json')
    @patch('utils.get_embedding')
    @patch('openai.OpenAI')
    @patch('pinecone.grpc.PineconeGRPC')
    def setUp(self, MockPinecone, MockOpenAI, mock_get_embedding, mock_load_json):
        os.environ['PINECONE_API_KEY'] = 'fake-pinecone-key'
        os.environ['OPENAI_API_KEY'] = 'fake-openai-key'
        
        # Setup mocks
        self.mock_pc = MockPinecone.return_value
        self.mock_client = MockOpenAI.return_value
        self.mock_index = self.mock_pc.Index.return_value

        mock_get_embedding.return_value = [0.1, 0.2, 0.3]
        mock_load_json.side_effect = lambda x: questions_json if 'questions' in x else contexts_json

        import generate_reports  # Import your main script/module here

        self.main_module = generate_reports
    
    def test_find_similarities(self):
        query_result = {
            'matches': [
                {'metadata': {'filename': 'AAPL/2002/page_23.pdf'}, 'score': 0.9},
                {'metadata': {'filename': 'UPS/2009/page_33.pdf'}, 'score': 0.8},
            ]
        }
        self.mock_index.query.return_value = query_result

        result = self.main_module.find_similarities("Test text")
        self.assertEqual(result, query_result)
    
    def test_get_llm_response(self):
        query_result = {
            'matches': [
                {'metadata': {'filename': 'AAPL/2002/page_23.pdf'}, 'score': 0.9},
                {'metadata': {'filename': 'UPS/2009/page_33.pdf'}, 'score': 0.8},
            ]
        }
        self.mock_index.query.return_value = query_result
        result = self.main_module.get_llm_response(context_dict, "Test question")
        self.assertIn("Context for AAPL 2002 page 23", result['context'])
    
    def test_generate_llm_response_report(self):
        query_result = {
            'matches': [
                {'metadata': {'filename': 'AAPL/2002/page_23.pdf'}, 'score': 0.9},
                {'metadata': {'filename': 'UPS/2009/page_33.pdf'}, 'score': 0.8},
            ]
        }
        self.mock_index.query.return_value = query_result

        response_mock = {
            'choices': [
                {'message': {'content': 'Test response'}},
            ]
        }
        self.mock_client.chat.completions.create.return_value = response_mock
        self.main_module.question_lookup = {item['question']: item for item in questions_json}
        result = self.main_module.generate_llm_response_report(questions_json, context_dict)
        self.assertEqual(len(result), len(questions_json))
        for item in result:
            self.assertIn("question", item)
            self.assertIn("answer", item)
            self.assertIn("context", item)
            self.assertIn("expectedAnswer", item)
            self.assertIn("id", item)

    def test_generate_similarity_report(self):
        query_result = {
            'matches': [
                {'metadata': {'filename': 'AAPL/2002/page_23.pdf'}, 'score': 0.9},
                {'metadata': {'filename': 'UPS/2009/page_33.pdf'}, 'score': 0.8},
            ]
        }
        self.mock_index.query.return_value = query_result

        result = self.main_module.generate_similarity_report(questions_json)

        self.assertEqual(len(result), len(questions_json))
        for item in result:
            self.assertIn("question", item)
            self.assertIn("filename", item)
            self.assertIn("rank", item)
            self.assertIn("score", item)
            self.assertIn("similarities_query_result", item)

if __name__ == '__main__':
    unittest.main()

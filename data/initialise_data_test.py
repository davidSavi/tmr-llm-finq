import unittest
import json
import os

from data.initialise_data import (convert_table_to_json, stringify_array, count_words,
                              create_context, process_training, extract_questions,
                              load_json_file, write_json_to_file)

class TestModuleFunctions(unittest.TestCase):

    def test_convert_table_to_json(self):
        table = [
            ["", "Year 1", "Year 2"],
            ["Category A", 100, 200],
            ["Category B", 150, 250]
        ]
        expected = [
            {"Category": "Category A", "Year 1": 100, "Year 2": 200},
            {"Category": "Category B", "Year 1": 150, "Year 2": 250}
        ]
        result = convert_table_to_json(table)
        self.assertEqual(result, expected)

    def test_stringify_array(self):
        array = ["Line 1", "Line 2", "Line 3"]
        expected = "Line 1\nLine 2\nLine 3"
        result = stringify_array(array)
        self.assertEqual(result, expected)

    def test_count_words(self):
        text = "This is a test text with seven words."
        expected = 8
        result = count_words(text)
        self.assertEqual(result, expected)

    def test_create_context(self):
        data = {
            "pre_text": ["Pre text line 1", "Pre text line 2"],
            "post_text": ["Post text line 1", "Post text line 2"],
            "table": [
                ["", "Year 1", "Year 2"],
                ["Category A", 100, 200],
                ["Category B", 150, 250]
            ],
            "filename": "test.pdf"
        }
        expected_context = {
            "context": "Pre text line 1\nPre text line 2\n[\n    {\n        \"Category\": \"Category A\",\n        \"Year 1\": 100,\n        \"Year 2\": 200\n    },\n    {\n        \"Category\": \"Category B\",\n        \"Year 1\": 150,\n        \"Year 2\": 250\n    }\n]\nPost text line 1\nPost text line 2",
            "filename": "test.pdf",
            "tokenAmount": 40
        }
        result = create_context(data)
        self.assertEqual(result, expected_context)

    def test_extract_questions(self):
        item = {
            "id": "Double_IPG/2015/page_24.pdf",
            "filename": "IPG/2015/page_24.pdf",
            "qa_0": {
                "question": "What is the total cash?",
                "answer": "44.0"
            },
            "qa_1": {
                "question": "What percentage?",
                "answer": "36.55%"
            }
        }
        expected = [
            {
                "id": "Double_IPG/2015/page_24.pdf_qa_0",
                "filename": "IPG/2015/page_24.pdf",
                "question": "What is the total cash?",
                "answer": "44.0"
            },
            {
                "id": "Double_IPG/2015/page_24.pdf_qa_1",
                "filename": "IPG/2015/page_24.pdf",
                "question": "What percentage?",
                "answer": "36.55%"
            }
        ]
        result = extract_questions(item)
        self.assertEqual(result, expected)

    def test_load_json_file(self):
        test_data = [{"key": "value"}]
        with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)
        
        result = load_json_file('test.json')
        self.assertEqual(result, test_data)
        
        os.remove('test.json')

    def test_write_json_to_file(self):
        test_data = [{"key": "value"}]
        write_json_to_file(test_data, 'test_output.json')
        
        with open('test_output.json', 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        self.assertEqual(result, test_data)
        
        os.remove('test_output.json')

if __name__ == "__main__":
    unittest.main()

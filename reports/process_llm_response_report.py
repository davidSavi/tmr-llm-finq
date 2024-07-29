import re
import json

def extract_number(s):
    """Extracts the first number found in a string."""
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        return float(match.group())
    return None

def compare_numbers(num1, num2, tolerance_percentage):
    difference = abs(num1 - num2)
    tolerance = (tolerance_percentage / 100) * min(abs(num1), abs(num2))
    
    return difference <= tolerance

def evaluate_list(items, tolerance_percentage):
    success_count = 0
    fail_count = 0
    
    for item in items:
        answer = extract_number(item['answer'])
        expected_answer = extract_number(item['expectedAnswer'])
        
        if answer is not None and expected_answer is not None:
            if compare_numbers(answer, expected_answer, tolerance_percentage):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1  # Treat as failure if either number is missing
    
    return success_count, fail_count

report_file = 'llm_response_report.json'  # Replace with the path to your report file
with open(report_file, 'r') as f:
    report = json.load(f)

tolerance_percentage = 3

success_count, fail_count = evaluate_list(report, tolerance_percentage)

print(f"Total questions that pass similarity: {len(report)}")
print(f"Successfully met {tolerance_percentage}% error tolerance: {success_count}")
print(f"Failed {tolerance_percentage}% error tolerance: {fail_count}")

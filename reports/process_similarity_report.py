import json
from collections import defaultdict

def count_correct_ranks(report):
    rank_counts = defaultdict(lambda: defaultdict(int))
    score_ranges = [
        ('> 0.75', lambda x: x >= 0.75),
        ('0.7 - 0.75', lambda x: 0.7 <= x < 0.75),
        ('0.65 - 0.7', lambda x: 0.65 <= x < 0.7),
        ('0.6 - 0.65', lambda x: 0.6 <= x < 0.65),
        ('0.5 - 0.6', lambda x: 0.5 <= x < 0.6),
        ('< 0.5', lambda x: x < 0.5),
    ]
    
    rank_0_total = 0

    for entry in report:
        rank = entry['rank']
        score = entry['score']
        if rank == 0:
            rank_0_total += 1
        else:
            for range_label, range_func in score_ranges:
                if range_func(score):
                    rank_counts[rank][range_label] += 1
                    break

    return dict(rank_counts), rank_0_total

report_file = 'similarity_report.json' 
with open(report_file, 'r') as f:
    report = json.load(f)

total_questions = len(report)

rank_counts, rank_0_total = count_correct_ranks(report)

print(f"Total questions: {total_questions}")
for rank, counts in sorted(rank_counts.items()):
    rank_total = sum(counts.values())
    print(f"Rank {rank} Total: {rank_total} questions ({(rank_total / total_questions) * 100:.2f}%):")
    for range_label, count in counts.items():
        percentage = (count / total_questions) * 100
        print(f"  Score {range_label}: {count} questions ({percentage:.2f}%)")
print(f"Mismatched: {rank_0_total} questions ({(rank_0_total / total_questions) * 100:.2f}%)")
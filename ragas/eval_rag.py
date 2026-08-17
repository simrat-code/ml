import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 1. Prepare your evaluation data
# Note: 'contexts' must be a list of lists (strings) because a RAG system can retrieve multiple text chunks.
data = {
    "question": [
        "What is the capital of France?",
        "Who wrote the play Romeo and Juliet?"
    ],
    "answer": [
        "The capital of France is Paris.",
        "Romeo and Juliet was written by William Shakespeare."
    ],
    "contexts": [
        ["Paris is the capital and most populous city of France."],
        ["William Shakespeare was an English playwright who wrote Romeo and Juliet."]
    ],
    "ground_truth": [
        "Paris",
        "William Shakespeare"
    ]
}

# 2. Convert your dictionary into a Hugging Face Dataset object (required by Ragas)
eval_dataset = Dataset.from_dict(data)

# 3. Define the metrics you want to calculate
metrics_to_use = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

print("Starting evaluation using OpenAI judge models...")

# 4. Run the evaluation
result = evaluate(
    dataset=eval_dataset,
    metrics=metrics_to_use
)

# 5. Convert results to a Pandas DataFrame and display/save them
df = result.to_pandas()

print("\n--- Evaluation Scores Summary ---")
print(result)

print("\n--- Detailed Row-by-Row Results ---")
print(df[["question", "faithfulness", "answer_relevancy"]])

# Optional: Save to a CSV file to inspect later
df.to_csv("ragas_eval_results.csv", index=False)
print("\nResults saved successfully to 'ragas_eval_results.csv'")

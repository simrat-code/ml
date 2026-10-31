from pathlib import Path

from sentence_transformers import SentenceTransformer

def print_embedding(embedding, short=False):
    if short:
        embed = embedding[:5] + embedding[-5:]

    for index in range(0, len(embed), 5):
        row = embed[index : index + 5]
        print(" ".join(str(value) for value in row))
    print(f"\nEmbedding ({len(embedding)} dimensions)\n")

def main(short=False) -> None:
    query = input("Enter a query: ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    models = {
        "MiniLM": ("all-MiniLM-L6-v2", {}),
        "MiniLM-onnx": ("all-MiniLM-L6-v2-onnx", {"backend": "onnx"}),
        "MiniLM-onnx-int8": (
            "all-MiniLM-L6-v2-onnx-int8",
            {
                "backend": "onnx",
                "model_kwargs": {"file_name": "onnx/model_quantized.onnx"},
            },
        ),
    }
    model_choices = {
        "1": "MiniLM",
        "2": "MiniLM-onnx",
        "3": "MiniLM-onnx-int8",
    }
    print("Available models:")
    for number, name in model_choices.items():
        print(f"  [{number}] {name}")
    choice = input("Choose a model [3]: ").strip() or "3"
    if choice not in model_choices:
        raise SystemExit(f"Invalid model selection: {choice}")
    model_choice = model_choices[choice]

    model_directory, model_options = models[model_choice]
    model_path = Path(__file__).resolve().parent / "models" / model_directory
    model = SentenceTransformer(str(model_path), **model_options)
    embedding = model.encode(query, show_progress_bar=False).tolist()

    print_embedding(embedding, short=short)


if __name__ == "__main__":
    choice = input(f"[=] to print short or full embedding? (s/f): ").strip().lower()
    short = choice in ("s", "short")
    count = 1
    while True:
        choice = input(f"run-{count} another query? (y/n): ").strip().lower()
        # if 'enter' is pressed without any input, continue to the next iteration
        if not choice:
            continue
        if choice not in ("y", "yes"):
            break
        main(short=short)
        count += 1

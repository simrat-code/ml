from pathlib import Path

from sentence_transformers import SentenceTransformer


def main() -> None:
    query = input("Enter a query: ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    model_path = Path(__file__).resolve().parent / "models" / "all-MiniLM-L6-v2-onnx-int8"
    model = SentenceTransformer(
        str(model_path),
        backend="onnx",
        model_kwargs={"file_name": "onnx/model_quantized.onnx"},
    )
    embedding = model.encode(query, show_progress_bar=False).tolist()

    for index in range(0, len(embedding), 5):
        row = embedding[index : index + 5]
        print(" ".join(str(value) for value in row))
    print(f"\nEmbedding ({len(embedding)} dimensions)")


if __name__ == "__main__":
    main()

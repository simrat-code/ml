
#
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install sentence-transformers
#

from sentence_transformers import SentenceTransformer

def all_minilm():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    model.save("models/all-MiniLM-L6-v2")

def all_minilm_onnx():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model {model_name}...", flush=True)
    model = SentenceTransformer(model_name, backend="onnx")

    print("Converting to ONNX format...", flush=True)
    model.save("models/all-MiniLM-L6-v2-onnx")


if __name__ == "__main__":
    # all_minilm()
    all_minilm_onnx()

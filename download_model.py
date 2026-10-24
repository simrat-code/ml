
#
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install sentence-transformers
#

import os
from pathlib import Path
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


def all_minilm_onnx_quantized():
    # We use the Xenova repository which contains pre-configured ONNX models
    model_name = "Xenova/all-MiniLM-L6-v2"
    
    print(f"Loading quantized INT8 model from {model_name}...", flush=True)
    # The 'onnx' backend combined with file_name ensures it pulls the 23MB quantized file
    model = SentenceTransformer(
        model_name, 
        backend="onnx", 
        model_kwargs={"file_name": "onnx/model_quantized.onnx"}
    )

    output_dir = "models/all-MiniLM-L6-v2-onnx-int8"
    print(f"Saving quantized model structure to '{output_dir}'...", flush=True)
    model.save(output_dir)
    
    # Print the resulting file size to confirm it worked
    onnx_file_path = os.path.join(output_dir, "model_quantized.onnx")
    if os.path.exists(onnx_file_path):
        size_mb = os.path.getsize(onnx_file_path) / (1024 * 1024)
        print(f"✅ Success! Saved ONNX model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    print(f"\n\t[1] MiniLM \n\t[2] MiniLM ONNX \n\t[3] MiniLM ONNX Quantized")
    choice = input("Which model should be downloaded? : ").strip().lower()

    if choice in ("1", "minilm"):
        all_minilm()
    elif choice in ("2", "minilm onnx", "onnx"):
        all_minilm_onnx()
    elif choice in ("3", "minilm onnx quantized", "onnx quantized"):
        all_minilm_onnx_quantized()
    else:
        raise ValueError("Invalid choice. Enter 1 for MiniLM, 2 for MiniLM ONNX, or 3 for MiniLM ONNX Quantized.")

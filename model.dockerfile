FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model during the image build phase using python
RUN python -c "from transformers import AutoTokenizer; import onnxruntime as ort; \
               from huggingface_hub import hf_hub_download; \
               hf_hub_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', filename='model.onnx', local_dir='/app/models/')"

COPY . .
CMD ["python", "main.py"]

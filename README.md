# API LLM HIT - Multi-Modal AI API Server

A powerful Flask-based REST API that integrates **Large Language Models (LLMs)**, **image inpainting**, and **vector embeddings** for advanced AI capabilities. This project combines state-of-the-art models for text generation, image manipulation, and semantic search.

## 🎯 Features

- **Text Generation**: Generate text responses using the Qwen2 language model
- **Image Inpainting**: Generate or modify images using the Kandinsky inpainting model
- **Semantic Search**: Leverage embeddings with Pinecone vector database for RAG capabilities
- **Streaming Responses**: Support for both streaming and non-streaming text generation
- **GPU Acceleration**: CUDA-optimized for fast inference
- **Health Checks**: Built-in endpoint for monitoring API status
- **Docker Support**: Easy deployment with Docker and Docker Compose

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- 16GB+ system RAM
- 16GB+ VRAM (for running models efficiently)

## 🚀 Installation

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd api_llm_hit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

Build and run with Docker Compose:
```bash
docker-compose up --build
```

## 🔧 Configuration

Set environment variables in a `.env` file:

```env
FLASK_DEBUG=False
PORT=6000
FLASK_ENV=production
```

## 📡 API Endpoints

### Health Check
**`GET /health`**
- Returns API status
- **Response**: `{"status": "ok"}`

### Text Generation (Streaming)
**`POST /api/llm-response`**

Request body (default streaming mode):
```json
{
    "prompt": "Your prompt here"
}
```

**Response**: Streams text chunks via Server-Sent Events

### Text Generation (Non-Streaming)
**`POST /api/llm-response`**

Request body with `extension` flag:
```json
{
    "extension": true,
    "prompt": "Your prompt here"
}
```

**Response**: 
```json
{
    "text": "Generated text response"
}
```

### Image Inpainting
**`POST /api/llm-response`**

Request body:
```json
{
    "initial_img": [array of pixel values],
    "masked_img": [array of mask values],
    "prompt": "Description of what to generate",
    "negative_prompt": "What to avoid"
}
```

**Response**:
```json
{
    "img": [array of inpainted image pixel values]
}
```

### Image Generation (Prompt Only)
**`POST /api/llm-response`**

Request body:
```json
{
    "only_prompt": "A beautiful sunset over mountains"
}
```

**Response**:
```json
{
    "img": [array of generated image pixel values]
}
```

### Vector Database Integration (Pinecone)
**`POST /api/llm-response`**

Request body:
```json
{
    "api_key": "your-pinecone-api-key"
}
```

Queries the Pinecone "quickstart" index and generates LLM responses based on vector similarity.

## 🛠️ Tech Stack

- **Framework**: Flask
- **Models**:
  - Text: Qwen2-0.5B-Instruct (Hugging Face)
  - Image: Kandinsky 2.2 (Diffusers)
  - Embeddings: all-mpnet-base-v2 (Sentence Transformers)
- **Vector DB**: Pinecone
- **Computation**: PyTorch with CUDA
- **Deployment**: Docker & Docker Compose
- **Server**: Gunicorn

## 📦 Dependencies

- `flask` - Web framework
- `torch` - Deep learning framework
- `diffusers` - Image generation models
- `transformers` - NLP models
- `langchain-ollama` - LLM integration
- `sentence-transformers` - Semantic embeddings
- `pinecone` - Vector database
- `pillow` - Image processing
- `numpy` - Numerical computing

## 🎓 Usage Examples

### Python Example - Text Generation

```python
import requests

response = requests.post(
    "http://localhost:6000/api/llm-response",
    json={"prompt": "What is artificial intelligence?"}
)

# For streaming responses
for chunk in response.iter_content(decode_unicode=True):
    print(chunk, end='', flush=True)
```

### Python Example - Non-Streaming

```python
response = requests.post(
    "http://localhost:6000/api/llm-response",
    json={
        "extension": True,
        "prompt": "Explain machine learning in 50 words"
    }
)

result = response.json()
print(result['text'])
```

## 🐳 Docker Deployment

The project includes Docker configuration for containerized deployment:

- **Dockerfile**: Sets up Python environment with all dependencies
- **docker-compose.yml**: Orchestrates service deployment

To run:
```bash
docker-compose up -d
```

The API will be available at `http://localhost:6000`

## 📝 Notebooks

- `experiment.ipynb` - Jupyter notebook for experimentation and testing

## 📂 Project Structure

```
api_llm_hit/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker image configuration
├── docker-compose.yml       # Docker Compose configuration
├── start_services.sh        # Service startup script
├── experiment.ipynb         # Experimentation notebook
└── README.md               # This file
```

## ⚠️ Important Notes

- **GPU Memory**: Models require CUDA. The app uses CPU offloading to optimize VRAM usage
- **First Run**: Model downloading may take time on first execution
- **Performance**: Text generation is optimized for CPU-friendly inference using Qwen2-0.5B
- **Rate Limiting**: Consider implementing rate limiting for production use

## 🔐 Security Considerations

- Use environment variables for sensitive data (API keys)
- Validate and sanitize all incoming requests
- Implement authentication for production deployments
- Enable CORS headers as needed

## 🐛 Troubleshooting

**CUDA Memory Issues**:
```python
# Handled automatically with `torch.cuda.empty_cache()`
```

**Slow Generation**:
- Reduce `max_tokens` parameter
- Use smaller model variants
- Enable GPU acceleration

**Model Download Failures**:
- Check internet connection
- Verify sufficient disk space
- Increase timeout for large models

## 📧 Support & Contact

For issues, questions, or contributions, please create an issue in the repository.

## 📄 License

[Add your license here]
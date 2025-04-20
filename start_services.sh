# start_services.sh
#!/bin/sh

# Start Ollama in background
ollama serve &

# Wait for Ollama to initialize
sleep 5

# Pull required models
ollama pull llama3:latest

# Start Flask application
python app.py

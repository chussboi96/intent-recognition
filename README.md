**Voice-Based LLM Assistant:**
A voice-enabled chatbot system using Whisper for speech recognition and Qwen LLM for response generation. Features parallel processing of queries through multiple specialized analysis paths.

Features
1. Voice input processing using Whisper Large V3
2. Query intent classification with spaCy
3. Parallel query processing using langgraph: analysis, fact-checking, classification, detailed explanation, and summarization
4. Response synthesis combining multiple analysis perspectives

Requirements
- CUDA 12.1
- Python 3.8+
- Ollama with Qwen2.5:3b model installed
- SpaCy intent classifier model
- Langchain, Langgraph


Quick start using
pip install -r requirements.txt,
then python main.py

-- Press Enter to start recording (10-second duration), system will process voice input and generate comprehensive responses.

# RAG Project — Colab Export

This folder contains everything needed to run the RAG pipeline in Google Colab.

## How to Use

1. **Upload this entire `colab_export/` folder** to Google Drive, or upload it directly to a Colab session.

2. **Open `rag_colab.ipynb`** in Google Colab.

3. **Run the cells in order:**
   - Install dependencies
   - Set your OpenAI API key (via Colab Secrets or directly in code)
   - Upload your documents (`.txt`, `.pdf`, `.md`)
   - Initialize the pipeline
   - Start asking questions

## Folder Structure

```
colab_export/
├── rag_colab.ipynb      # Main Colab notebook
├── config.py            # Configuration (adapted for Colab)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── documents/           # Place your documents here
└── src/                 # RAG pipeline source modules
    ├── __init__.py
    ├── chains.py
    ├── document_processor.py
    ├── memory.py
    ├── prompts.py
    ├── rag_pipeline.py
    └── vector_store.py
```

## Notes

- The `config.py` is adapted to resolve paths relative to the notebook location and to retrieve API keys from Colab Secrets.
- If using a local LLM server (e.g. LM Studio), set `OPENAI_API_BASE_URL` in the notebook. Note that Colab cannot reach `localhost` on your machine — you would need a tunneling solution like ngrok.
- Documents are not included. Upload your own via the notebook's upload widget or the Colab file browser.

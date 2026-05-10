# NoteBookLM RAG

A retrieval-augmented generation (RAG) system that lets you upload a PDF and ask questions grounded in that document.

## Stack

- **Express** — HTTP server
- **Multer** — PDF upload
- **LangChain PDFLoader + RecursiveCharacterTextSplitter** — chunking (chunkSize: 800, overlap: 150)
- **HuggingFace Transformers (Xenova/all-MiniLM-L6-v2)** — local embeddings
- **MemoryVectorStore** — in-memory vector store (no external DB needed)
- **Anthropic Claude** — grounded answer generation

## Local Setup

```bash
npm install
# create .env with:
# ANTHROPIC_API_KEY=your_key
node server.js
# open http://localhost:3000
```

## Folder Structure

```
.
├── server.js          # Express server (upload + ask endpoints)
├── public/
│   └── index.html     # Single-page UI
├── src/
│   ├── ingest.js      # Original Qdrant ingestion script
│   └── query.js       # Original Qdrant query script
├── uploads/           # Uploaded files (gitignored)
├── package.json
└── .env               # Not committed
```

## Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command**: `npm install --legacy-peer-deps`
   - **Start Command**: `node server.js`
   - **Environment**: Node
5. Add environment variable:
   - `ANTHROPIC_API_KEY` = your Anthropic API key
6. Click **Deploy**

> First upload will be slow (~1-2 min) as the embedding model downloads. Subsequent uploads are faster.

## Chunking Strategy

Uses **RecursiveCharacterTextSplitter** with:
- `chunkSize: 800` characters
- `chunkOverlap: 150` characters

This strategy recursively splits on `\n\n`, `\n`, ` `, and `` (empty string), preserving semantic boundaries (paragraphs first, then sentences, then words) before falling back to character-level splits.

## Notes

- The application works best with small to medium-sized PDFs on the deployed free-tier version.
- Very large PDFs (thousands of chunks) may take longer to process due to memory and timeout limitations of free cloud hosting.
- Locally, the application can successfully process much larger documents.

## API Endpoints

### `POST /upload`
- Body: `multipart/form-data` with field `file` (PDF)
- Response: `{ success: true, chunks: <number> }`

### `POST /ask`
- Body: `{ "question": "..." }`
- Response: `{ answer: "...", sources: [{ index, page, snippet }] }`

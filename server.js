import "dotenv/config";
import express from "express";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

import Anthropic from "@anthropic-ai/sdk";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;

app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// FILE UPLOAD SETUP
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage });

// EMBEDDINGS
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

//  CLIENT
const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// IN-MEMORY VECTOR STORE
let vectorStore = null;

// HEALTH CHECK
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// PDF UPLOAD + INGESTION
app.post("/upload", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded.",
      });
    }

    console.log("Loading PDF...");

    const loader = new PDFLoader(req.file.path);

    const docs = await loader.load();

    console.log("PDF Loaded");

    // CHUNKING
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 150,
    });

    const chunks = await splitter.splitDocuments(docs);

    console.log(`Created ${chunks.length} chunks`);

    // VECTOR STORE
    vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);

    console.log("Vector store ready");

    res.json({
      success: true,
      message: "Document uploaded and indexed successfully.",
      chunks: chunks.length,
    });
  } catch (err) {
    console.error("Upload Error:", err);

    res.status(500).json({
      error: err.message,
    });
  }
});

// ASK QUESTION
app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;

    if (!question) {
      return res.status(400).json({
        error: "No question provided.",
      });
    }

    if (!vectorStore) {
      return res.status(400).json({
        error: "No document uploaded yet. Please upload a PDF first.",
      });
    }

    // RETRIEVAL
    const retriever = vectorStore.asRetriever({
      k: 3,
    });

    const results = await retriever.invoke(question);

    // CONTEXT
    const context = results.map((doc) => doc.pageContent).join("\n\n");

    // PROMPT
    const prompt = `
You are a helpful AI assistant.

Answer ONLY from the provided context.

Be concise and avoid repeating duplicate information from the document.

If the answer is not present in the context, say:
"I could not find this information in the document."

Context:
${context}

Question:
${question}
`;

    // CLAUDE RESPONSE
    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 500,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    const answer = response.content[0].text;

    // SOURCES
    const sources = results.map((doc, index) => ({
      index: index + 1,
      page: doc.metadata?.loc?.pageNumber || "Unknown",
      snippet: doc.pageContent.slice(0, 120),
    }));

    res.json({
      answer,
      sources,
    });
  } catch (err) {
    console.error("Ask Error:", err);

    res.status(500).json({
      error: err.message,
    });
  }
});

// START SERVER
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

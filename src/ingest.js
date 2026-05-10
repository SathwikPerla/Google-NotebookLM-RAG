import "dotenv/config";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import { QdrantVectorStore } from "@langchain/qdrant";

const filePath = "./uploads/sample.pdf";

async function ingestDocument() {
  try {
    console.log("Loading PDF...");

    const loader = new PDFLoader(filePath);

    const docs = await loader.load();

    console.log("PDF Loaded");

    // CHUNKING
    console.log("Creating chunks...");

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 800,
      chunkOverlap: 150,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    console.log(`Created ${splitDocs.length} chunks`);

    // EMBEDDINGS
    console.log("Generating embeddings...");


const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

    // STORE IN QDRANT
    console.log("Storing in Qdrant...");

    await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
      url: process.env.QDRANT_URL,
      collectionName: "notebooklm",
    });

    console.log("Ingestion Completed Successfully");
  } catch (error) {
    console.error("Error:", error);
  }
}

ingestDocument();

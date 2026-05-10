import "dotenv/config";

import { QdrantVectorStore } from "@langchain/qdrant";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import Anthropic from "@anthropic-ai/sdk";

const userQuery = "What is the difference between a framework and a library?";

async function askQuestion() {
  try {
    console.log("Connecting to Qdrant...");

    // EMBEDDINGS
    const embeddings = new HuggingFaceTransformersEmbeddings({
      model: "Xenova/all-MiniLM-L6-v2",
    });

    // CONNECT EXISTING COLLECTION
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: process.env.QDRANT_URL,
        collectionName: "notebooklm",
      },
    );

    // RETRIEVER
    const retriever = vectorStore.asRetriever({
      k: 3,
    });

    console.log("Searching relevant chunks...");

    const results = await retriever.invoke(userQuery);

    // CONTEXT
    const context = results.map((doc) => doc.pageContent).join("\n\n");

    // GEMINI
    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    });

    const prompt = `
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not present in context, say:
"I could not find this information in the document."

Context:
${context}

Question:
${userQuery}
`;

    console.log("Generating answer...");

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 500,
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
    });

    const answer = response.content[0].text;

    console.log("\n===== ANSWER =====\n");

    console.log(answer);

    console.log("\n===== SOURCES =====\n");

    results.forEach((doc, index) => {
      console.log(
        `Source ${index + 1} | Page: ${
          doc.metadata?.loc?.pageNumber || "Unknown"
        }`,
      );
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

askQuestion();

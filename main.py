import os
import PyPDF2
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

class TuberculosisChatbot:
    def __init__(self):
        # Configuration
        self.PDF_FOLDER = "pdfs/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize models (100% offline)
        print("Loading AI models...")
        self.embedder = SentenceTransformer('all-mpnet-base-v2', device=self.device)
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Initialize text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
        )
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None

    def load_documents(self):
        """Load and process all PDFs"""
        print("Processing documents...")
        pdf_files = [f for f in os.listdir(self.PDF_FOLDER) if f.endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError(f"No PDFs found in {self.PDF_FOLDER}")
        
        for filename in pdf_files:
            try:
                with open(os.path.join(self.PDF_FOLDER, filename), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    text = " ".join(text.split())  # Clean whitespace
                    
                    chunks = self.text_splitter.split_text(text)
                    self.corpus.extend(chunks)
                    self.chunk_metadata.extend([{
                        "text": chunk,
                        "source": filename,
                        "chunk_id": f"{filename}_{i}"
                    } for i, chunk in enumerate(chunks)])
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
        print(f"Loaded {len(self.corpus)} text chunks")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Hybrid semantic + keyword search"""
        # Semantic search
        query_embedding = self.embedder.encode(query)
        doc_embeddings = self.embedder.encode(self.corpus)
        semantic_scores = np.dot(doc_embeddings, query_embedding)
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Combine scores
        combined_scores = 0.5*(semantic_scores/semantic_scores.max()) + 0.5*(bm25_scores/bm25_scores.max())
        top_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [{
            "text": self.corpus[idx],
            "source": self.chunk_metadata[idx]["source"],
            "score": float(combined_scores[idx])
        } for idx in top_indices]

    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using local LLM"""
        prompt = f"""Based on this medical context about Tuberculosis:
        
{context}

Question: {query}

Provide:
1. A direct answer
2. Source document
3. Key details"""
        
        return self.llm(
            prompt,
            max_length=256,
            temperature=0.3,
            do_sample=False
        )[0]["generated_text"]

    def chat(self):
        """Run the chatbot interface"""
        self.load_documents()
        print("\nTuberculosis Chatbot Ready! Ask your questions.")
        
        while True:
            try:
                question = input("\nAsk about TB (or 'quit'): ").strip()
                if question.lower() in ['quit', 'exit']:
                    break
                
                # Retrieve relevant context
                results = self.search(question)
                context = "\n\n".join([
                    f"Source: {res['source']}\n{res['text']}" 
                    for res in results
                ])
                
                # Generate and display answer
                answer = self.generate_answer(question, context)
                print("\n" + "="*60)
                print(f"ANSWER: {answer}")
                print("SOURCES:")
                print("\n".join(set(res['source'] for res in results)))
                print("="*60)
                
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    chatbot = TuberculosisChatbot()
    chatbot.chat()
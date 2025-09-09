import os
import PyPDF2
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import pickle
from pathlib import Path
import faiss
from time import time

class TuberculosisChatbot:
    def __init__(self):
        # Configuration
        self.PDF_FOLDER = "pdfs/"
        self.CACHE_FOLDER = "cache/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories
        Path(self.PDF_FOLDER).mkdir(exist_ok=True)
        Path(self.CACHE_FOLDER).mkdir(exist_ok=True)
        
        # Initialize models
        print("Loading AI models...")
        start_time = time()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print(f"Models loaded in {time()-start_time:.2f} seconds")
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
        )
        
        # Document storage
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None
        self.faiss_index = None
        self.embedding_dim = 384
        
        # Load or process documents
        self.load_or_process_documents()

    def load_or_process_documents(self):
        """Load documents from cache or process PDFs"""
        cache_file = os.path.join(self.CACHE_FOLDER, "document_cache.pkl")
        faiss_file = os.path.join(self.CACHE_FOLDER, "faiss_index.faiss")
        
        if os.path.exists(cache_file) and os.path.exists(faiss_file):
            print("Loading cached documents...")
            with open(cache_file, "rb") as f:
                self.corpus, self.chunk_metadata = pickle.load(f)
            self.faiss_index = faiss.read_index(faiss_file)
            self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
            print(f"Loaded {len(self.corpus)} document chunks")
        else:
            self.process_pdfs_and_create_embeddings()

    def process_pdfs_and_create_embeddings(self):
        """Process PDFs and create embeddings"""
        print("Processing PDF documents...")
        start_time = time()
        
        pdf_files = [f for f in os.listdir(self.PDF_FOLDER) if f.endswith(".pdf")]
        if not pdf_files:
            raise ValueError(f"No PDFs found in {self.PDF_FOLDER}")

        for filename in pdf_files:
            try:
                with open(os.path.join(self.PDF_FOLDER, filename), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    text = " ".join(text.split())
                    
                    chunks = self.text_splitter.split_text(text)
                    self.corpus.extend(chunks)
                    self.chunk_metadata.extend([{
                        "source": filename,
                        "chunk_id": f"{filename}_{i}"
                    } for i in range(len(chunks))])
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        # Create search indexes
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
        embeddings = self.embedder.encode(self.corpus, show_progress_bar=True)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Save to cache
        with open(os.path.join(self.CACHE_FOLDER, "document_cache.pkl"), "wb") as f:
            pickle.dump((self.corpus, self.chunk_metadata), f)
        
        faiss.write_index(self.faiss_index, os.path.join(self.CACHE_FOLDER, "faiss_index.faiss"))
        
        print(f"Processed {len(self.corpus)} chunks in {time()-start_time:.2f} seconds")

    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> list:
        """Retrieve relevant document chunks"""
        # Semantic search
        query_embedding = self.embedder.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, k*2)
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Combine scores
        combined_results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            semantic_score = distances[0][np.where(indices[0] == idx)[0][0]]
            keyword_score = bm25_scores[idx]
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            combined_results.append((idx, combined_score))
        
        # Sort and get top results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return [{
            "text": self.corpus[idx],
            "source": self.chunk_metadata[idx]["source"],
            "score": float(score)
        } for idx, score in combined_results[:k]]

    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""Generate a comprehensive answer about tuberculosis based on the provided context, with detailed information and references to the sources, in 3-5 complete sentences. 
                Context:
                {context}

                Question: {query}

                Answer in 3-5 complete sentences:"""
        
        return self.llm(
            prompt,
            max_new_tokens=512,  # Increased from 256
            num_beams=4,  # Better quality generation
            early_stopping=True,
            no_repeat_ngram_size=3  # Avoids repetition
        )[0]["generated_text"]

    def ask_question(self, question: str):
        """Full QA pipeline"""
        print(f"\nProcessing question: '{question}'")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question)
        print(f"\nFound {len(relevant_chunks)} relevant chunks:")
        for chunk in relevant_chunks:
            print(f"- From {chunk['source']} (score: {chunk['score']:.3f})")
        
        # Combine chunks into context
        context = "\n\n".join([
            f"Source: {chunk['source']}\n{chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        # Generate answer
        answer = self.generate_response(question, context)
        
        # Display results
        print("\nGenerated Answer:")
        print(answer.strip())
        
        print("\nSource Documents:")
        for source in set(chunk['source'] for chunk in relevant_chunks):
            print(f"- {source}")

if __name__ == '__main__':
    chatbot = TuberculosisChatbot()
    
    print("\nTuberculosis Chatbot Ready!")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Your question about tuberculosis: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        chatbot.ask_question(question)
        print("\n" + "="*80 + "\n")
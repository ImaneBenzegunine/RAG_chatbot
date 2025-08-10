from flask import Flask, request, jsonify, render_template, send_from_directory
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

app = Flask(__name__, static_folder='static', template_folder='templates')

class DocumentChatbot:
    def __init__(self):
        # Configuration
        self.PDF_FOLDER = "pdfs/"
        self.CACHE_FOLDER = "cache/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create directories if they don't exist
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
        
        # Initialize text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
        )
        
        # Initialize document storage
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None
        self.faiss_index = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Load or process documents
        self.load_or_process_documents()

    def load_or_process_documents(self):
        """Load documents from cache or process PDFs and create embeddings"""
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
        """Process all PDFs, create chunks, and generate embeddings"""
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
                    text = " ".join(text.split())  # Normalize whitespace
                    
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
        cache_file = os.path.join(self.CACHE_FOLDER, "document_cache.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump((self.corpus, self.chunk_metadata), f)
        
        faiss_file = os.path.join(self.CACHE_FOLDER, "faiss_index.faiss")
        faiss.write_index(self.faiss_index, faiss_file)
        
        print(f"Processed {len(self.corpus)} chunks in {time()-start_time:.2f} seconds")

    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> list:
        """Retrieve most relevant document chunks using hybrid search"""
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
        top_indices = [idx for idx, _ in combined_results[:k]]
        
        return [{
            "text": self.corpus[idx],
            "source": self.chunk_metadata[idx]["source"],
            "score": float(combined_results[i][1])
        } for i, idx in enumerate(top_indices)]

    def generate_response(self, query: str, context: str) -> str:
        """Generate answer using the local LLM"""
        prompt = f"""Based on the following context:
        
{context}

Question: {query}

Please provide:
1. A concise answer
2. Relevant details
3. Source information"""
        
        return self.llm(
            prompt,
            max_length=512,
            temperature=0.3,
            do_sample=False
        )[0]["generated_text"]

# Initialize the chatbot
chatbot = DocumentChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    
    # Step 1: Retrieve relevant document chunks
    relevant_chunks = chatbot.retrieve_relevant_chunks(question)
    
    # Step 2: Combine chunks into context
    context = "\n\n".join([
        f"Source: {chunk['source']}\n{chunk['text']}" 
        for chunk in relevant_chunks
    ])
    
    # Step 3: Generate answer
    answer = chatbot.generate_response(question, context)
    
    # Step 4: Return response
    return jsonify({
        'answer': answer,
        'sources': list(set(chunk['source'] for chunk in relevant_chunks))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
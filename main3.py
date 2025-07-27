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

class TuberculosisChatbot:
    def __init__(self):
        # Configuration
        self.PDF_FOLDER = "pdfs/"
        self.CACHE_FOLDER = "cache/"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create cache directory if it doesn't exist
        Path(self.CACHE_FOLDER).mkdir(exist_ok=True)
        
        # Initialize models (100% offline) with smaller embedding model
        print("Loading AI models...")
        start_time = time()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            model_kwargs={"low_cpu_mem_usage": True}
        )
        print(f"Models loaded in {time()-start_time:.2f} seconds")
        
        # Initialize text processing
        self.text_splitter = RecursiveCharacterTextSplitter(  # Fixed typo: was 'text_splitter'
            chunk_size=512,
            chunk_overlap=128,
            length_function=len
        )
        self.corpus = []
        self.chunk_metadata = []
        self.bm25 = None
        self.faiss_index = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.load_documents()

    def load_documents(self):
        """Load and process all PDFs with caching"""
        cache_file = os.path.join(self.CACHE_FOLDER, "document_cache.pkl")
        faiss_file = os.path.join(self.CACHE_FOLDER, "faiss_index.faiss")
        
        # Try to load from cache
        if os.path.exists(cache_file) and os.path.exists(faiss_file):
            print("Loading cached documents and embeddings...")
            start_time = time()
            with open(cache_file, "rb") as f:
                self.corpus, self.chunk_metadata = pickle.load(f)
            
            self.faiss_index = faiss.read_index(faiss_file)
            self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
            print(f"Loaded {len(self.corpus)} cached chunks in {time()-start_time:.2f} seconds")
            return
        
        # Process documents if cache doesn't exist
        print("Processing documents...")
        start_time = time()
        pdf_files = [f for f in os.listdir(self.PDF_FOLDER) if f.endswith(".pdf")]
        
        if not pdf_files:
            raise ValueError(f"No PDFs found in {self.PDF_FOLDER}")
        
        for filename in pdf_files:
            try:
                with open(os.path.join(self.PDF_FOLDER, filename), "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    text = " ".join(text.split())  # Clean whitespace
                    
                    chunks = self.text_splitter.split_text(text)  # Fixed: was 'text_splitter'
                    self.corpus.extend(chunks)
                    self.chunk_metadata.extend([{
                        "text": chunk,
                        "source": filename,
                        "chunk_id": f"{filename}_{i}"
                    } for i, chunk in enumerate(chunks)])
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        # Create BM25 index
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
        
        # Create and save FAISS index
        print("Generating document embeddings...")
        embeddings = self.embedder.encode(self.corpus, batch_size=32, show_progress_bar=True)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump((self.corpus, self.chunk_metadata), f)
        faiss.write_index(self.faiss_index, faiss_file)
        
        print(f"Processed {len(self.corpus)} chunks in {time()-start_time:.2f} seconds")

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Hybrid semantic + keyword search using FAISS"""
        start_time = time()
        
        # Semantic search with FAISS
        query_embedding = self.embedder.encode(query)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # FAISS search returns distances and indices
        distances, indices = self.faiss_index.search(query_embedding, k*2)  # Get extra for filtering
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Combine scores for the FAISS-retrieved documents only
        combined_scores = []
        for idx in indices[0]:
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            semantic_score = distances[0][np.where(indices[0] == idx)[0][0]]
            bm25_score = bm25_scores[idx]
            combined_score = 0.5*(semantic_score/semantic_score.max()) + 0.5*(bm25_score/bm25_scores.max())
            combined_scores.append((idx, combined_score))
        
        # Sort by combined score and take top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in combined_scores[:k]]
        
        print(f"Search completed in {time()-start_time:.4f} seconds")
        return [{
            "text": self.corpus[idx],
            "source": self.chunk_metadata[idx]["source"],
            "score": float(score)
        } for idx, score in combined_scores[:k]]

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

# Initialize the chatbot
chatbot = TuberculosisChatbot()

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']
    
    # Retrieve relevant context
    results = chatbot.search(message)
    context = "\n\n".join([
        f"Source: {res['source']}\n{res['text']}" 
        for res in results
    ])
    
    # Generate answer
    answer = chatbot.generate_answer(message, context)
    
    return jsonify({
        'answer': answer,
        'sources': list(set(res['source'] for res in results))
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    index
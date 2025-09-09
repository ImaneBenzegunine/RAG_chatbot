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
from flask import Flask, render_template, request, jsonify
import threading
import traceback

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
            model="google/flan-t5-large",
            device=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        print(f"Models loaded in {time()-start_time:.2f} seconds")
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
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
                        "page": i // 3 + 1,  # Approximate page number
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

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> list:
        """Retrieve relevant document chunks with improved scoring"""
        # Semantic search
        query_embedding = self.embedder.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, k*3)  # Get more candidates
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Combine scores with better weighting
        combined_results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.corpus):
                continue
            semantic_score = distances[0][np.where(indices[0] == idx)[0][0]]
            keyword_score = bm25_scores[idx]
            # Dynamic weighting based on query length
            semantic_weight = 0.6 if len(query.split()) > 3 else 0.4
            combined_score = (semantic_weight * semantic_score + 
                            (1-semantic_weight) * keyword_score)
            combined_results.append((idx, combined_score))
        
        # Sort and get top results
        combined_results.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in combined_results[:k]]
        
        return [{
            "text": self.corpus[idx],
            "source": self.chunk_metadata[idx]["source"],
            "page": self.chunk_metadata[idx]["page"],
            "score": float(score)
        } for idx, score in combined_results[:k]]

    def generate_response(self, query: str, context: str) -> str:
        """Generate focused answers with better context handling"""
        # Pre-process context to remove irrelevant parts
        context = self._clean_context(context)
        
        # Create focused prompt
        prompt = f"""Based on the following context about tuberculosis, answer this question: {query}

Context: {context[:2000]}

Answer the question clearly and accurately using only the provided context. Provide a detailed response with important facts if available."""

        try:
            response = self.llm(
                prompt,
                max_new_tokens=512,
                min_new_tokens=50,
                num_beams=4,
                do_sample=True,
                temperature=0.7,
                no_repeat_ngram_size=2
            )
            return self._postprocess_answer(response[0]["generated_text"])
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return self._fallback_response(context)

    def _clean_context(self, context: str) -> str:
        """Remove irrelevant sections from context"""
        lines = []
        for line in context.split('\n'):
            # Filter out table of contents-like entries
            if not any(x in line for x in ['......', '....', 'CONTENTS', 'CHAPTER']):
                lines.append(line)
        return '\n'.join(lines)

    def _postprocess_answer(self, answer: str) -> str:
        """Clean up model output"""
        # Remove repeated prompts
        if 'Question:' in answer:
            answer = answer.split('Question:')[0]
        if 'Context:' in answer:
            answer = answer.split('Context:')[0]
        # Trim after last complete sentence
        last_period = answer.rfind('.')
        if last_period > 0:
            answer = answer[:last_period+1]
        return answer.strip()

    def _fallback_response(self, context: str) -> str:
        """Provide basic info when generation fails"""
        if context:
            first_chunk = context.split('\n\n')[0][:500] if context else "No relevant information found."
            return f"I found relevant information but couldn't generate a complete answer. Here's what I found: {first_chunk}"
        return "I couldn't find enough relevant information to answer your question. Please try rephrasing or ask about a different aspect of tuberculosis."

    def ask_question(self, question: str):
        """Enhanced QA pipeline with better context handling"""
        print(f"\nProcessing question: '{question}'")
        
        try:
            # Retrieve relevant chunks with minimum score threshold
            relevant_chunks = [c for c in self.retrieve_relevant_chunks(question) 
                              if c['score'] > 0.3]  # Lower threshold to get more results
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find enough relevant information about tuberculosis to answer your question. Please try rephrasing or ask about a different aspect of tuberculosis.",
                    "sources": [],
                    "chunks_count": 0
                }
            
            # Combine chunks into context with better organization
            context = "MEDICAL CONTEXT ABOUT TUBERCULOSIS:\n\n"
            context += "\n\n".join([
                f"DOCUMENT: {chunk['source']} (Page ~{chunk['page']})\n"
                f"CONTENT: {chunk['text']}\n"
                for chunk in relevant_chunks
            ])
            
            # Limit context to avoid truncation while keeping important info
            max_context = 3000  # characters
            context = context[:max_context]
            
            # Generate answer
            answer = self.generate_response(question, context)
            
            # Prepare sources for display
            sources = {(chunk['source'], chunk['page']) for chunk in relevant_chunks}
            source_list = [{"document": source, "page": page} for source, page in sorted(sources)]
            
            print(f"Generated answer: {answer}")
            
            return {
                "answer": answer.strip(),
                "sources": source_list,
                "chunks_count": len(relevant_chunks)
            }
            
        except Exception as e:
            print(f"Error in ask_question: {str(e)}")
            print(traceback.format_exc())
            return {
                "answer": "Sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "chunks_count": 0
            }

# Initialize Flask app and chatbot
app = Flask(__name__)
chatbot = None
chatbot_ready = False

def initialize_chatbot():
    """Initialize the chatbot in a separate thread to avoid blocking"""
    global chatbot, chatbot_ready
    try:
        print("Initializing Tuberculosis Chatbot...")
        chatbot = TuberculosisChatbot()
        chatbot_ready = True
        print("Chatbot initialized successfully!")
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        print(traceback.format_exc())
        chatbot_ready = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        if not chatbot or not chatbot_ready:
            return jsonify({'error': 'Chatbot is still initializing. Please wait...'}), 503
        
        result = chatbot.ask_question(question)
        
        return jsonify({
            'answer': result['answer'],
            'sources': result['sources'],
            'chunks_count': result['chunks_count']
        })
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'An error occurred while processing your question',
            'answer': 'Sorry, I encountered a technical issue. Please try your question again.'
        }), 500

@app.route('/status')
def status():
    return jsonify({
        'initialized': chatbot_ready,
        'device': chatbot.device if chatbot else 'Not initialized'
    })

if __name__ == '__main__':
    # Initialize chatbot in background thread
    init_thread = threading.Thread(target=initialize_chatbot)
    init_thread.daemon = True
    init_thread.start()
    
    # Start Flask app
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
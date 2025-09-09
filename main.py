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
            model="google/flan-t5-large",
            #model="google/flan-t5-base",
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

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> list:  # Increased from 3 to 5
        """Retrieve relevant document chunks with improved scoring"""
        # Semantic search
        query_embedding = self.embedder.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.faiss_index.search(query_embedding, k*3)  # Get more candidates
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Combine scores with better weighting
        combined_results = []
        for idx in indices[0]:
            if idx < 0:
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
        prompt = f"""Question: {query}

        Relevant Context:
        {context[:2000]}  

        Instructions:
        - Write a detailed answer in at least 3 sentences and more than 50 words.
        - Use ONLY the provided context.
        - Include important facts, numbers, or examples if present.
        - Write in clear, well-structured paragraphs.


        Answer:"""
            
        try:
                response = self.llm(
                    prompt,
                    max_new_tokens=512,
                    min_new_tokens=80,   # ensures at least ~80 tokens (~50–60 words)
                    num_beams=5,
                    do_sample=True,      # allow more natural generation
                    temperature=0.7,      # less robotic
                    no_repeat_ngram_size=3
                    #do_sample=False  # More deterministic
                )
                return self._postprocess_answer(response[0]["generated_text"])
        except Exception as e:
                print(f"Generation error: {str(e)}")
                return self._fallback_response(relevant_chunks)

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
            # Trim after last complete sentence
            last_period = answer.rfind('.')
            if last_period > 0:
                answer = answer[:last_period+1]
            return answer.strip()

    def _fallback_response(self, chunks: list) -> str:
        """Provide basic info when generation fails"""
        first_chunk = context.split('\n\n')[0][:500] # Get first ~500 chars of the first chunk
        return f"I found the following relevant information, but could not generate a structured answer:\n\n{first_chunk}"

    def ask_question(self, question: str):
        """Enhanced QA pipeline with better context handling"""
        print(f"\nProcessing question: '{question}'")
        
        # Retrieve relevant chunks with minimum score threshold
        relevant_chunks = [c for c in self.retrieve_relevant_chunks(question) 
                          if c['score'] > 0.5]  # Filter low-quality matches
        
        if not relevant_chunks:
            print("No high-quality matches found for this question")
            return
        
        #print(f"\nFound {len(relevant_chunks)} relevant chunks:")
        #for chunk in relevant_chunks:
        #    print(f"- From {chunk['source']} (page ~{chunk['page']}, score: {chunk['score']:.3f})")
        
        # Combine chunks into context with better organization
        context = "MEDICAL CONTEXT ABOUT TUBERCULOSIS:\n\n"
        context += "\n\n".join([
            f"DOCUMENT: {chunk['source']} (Page ~{chunk['page']})\n"
            f"CONTENT: {chunk['text']}\n"
            for chunk in relevant_chunks
        ])
        
        # Limit context to avoid truncation while keeping important info
        max_context = 4000  # characters
        context = context[:max_context]
        
        # Generate answer
        answer = self.generate_response(question, context)
        
        # Post-process and display results
        print("\nDETAILED ANSWER:")
        print(answer.strip())
        
        #print("\nSOURCE DOCUMENTS:")
        #sources = {(chunk['source'], chunk['page']) for chunk in relevant_chunks}
        #for source, page in sorted(sources):
        #    print(f"- {source} (page ~{page})")

if __name__ == '__main__':
    chatbot = TuberculosisChatbot()
    
    print("\nTUBERCULOSIS EXPERT CHATBOT READY")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Your question about tuberculosis: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        chatbot.ask_question(question)
        print("\n" + "="*80 + "\n")


        
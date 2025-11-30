"""Production RAG Implementation with Embeddings and Vector Search
Ingests real data from CSVs and PDFs for the RAG system
"""

import numpy as np
import pickle
import glob
from openai import OpenAI
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DOCS_DIR,
    MANUALS_DIR,
    VECTOR_INDEX_DIR,
    EMBEDDING_MODEL,
    TIKTOKEN_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CONTRACT_FILES,
    MANUAL_MODELS,
    RELOAD_DATA,
    DEFAULT_TOP_K,
)
from prompts import format_rag_answer_prompt
from scraper import scrape_toyota_manual
from utils import call_llm


@dataclass
class Document:
    """Enhanced document with embeddings"""
    content: str
    source: str
    page: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class RAGTool:
    """RAG with embeddings and vector search"""
    
    def __init__(self, model_name: str = None, index_path: str = None, force_reload: bool = None):
        """
        Initialize RAG system
        
        Args:
            model_name: HuggingFace model for embeddings (default from config)
            index_path: Directory to store/load index (default from config)
            force_reload: If True, rebuild index. If None, uses RELOAD_DATA config.
        """
        model_name = model_name or EMBEDDING_MODEL
        self.index_path = Path(index_path) if index_path else VECTOR_INDEX_DIR
        reload = force_reload if force_reload is not None else RELOAD_DATA
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index_path.mkdir(exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            model_name=TIKTOKEN_MODEL
        )
                
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        
        if self._index_exists() and not reload:
            self._load_index()
        else:
            if reload and self._index_exists():
                print("Rebuilding index (RELOAD_DATA=true)...")
                self._delete_index()
            else:
                print("No existing index found. Building new index...")
            self._build_index()
    
    def _index_exists(self) -> bool:
        """Check if index files exist"""
        return (self.index_path / "faiss.index").exists() and \
               (self.index_path / "documents.pkl").exists()
    
    def _delete_index(self):
        """Delete existing index files"""
        index_file = self.index_path / "faiss.index"
        docs_file = self.index_path / "documents.pkl"
        
        if index_file.exists():
            index_file.unlink()
            print(f"  ✓ Deleted {index_file}")
        if docs_file.exists():
            docs_file.unlink()
            print(f"  ✓ Deleted {docs_file}")
    
    def _build_index(self):
        """Build vector index from scratch"""
        # Load documents from various sources
        self.documents = []
        
        # Load real contract PDFs
        print("Loading contract PDFs...")
        self._load_contract_pdfs()
        self._load_owners_manuals()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(self.documents)} documents...")
        texts = [doc.content for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Attach embeddings to documents
        for doc, emb in zip(self.documents, embeddings):
            doc.embedding = emb
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype('float32'))
        
        # 5. Save index
        self._save_index()
        print(f"✓ Index built with {len(self.documents)} documents")
    
    def _load_contract_pdfs(self):
        """Load and chunk real contract PDFs"""
        if not DOCS_DIR.exists():
            print(f"⚠️  Warning: Docs directory not found at {DOCS_DIR}")
            return
        
        for pdf_file in CONTRACT_FILES:
            pdf_path = DOCS_DIR / pdf_file
            
            if not pdf_path.exists():
                print(f"⚠️  Warning: {pdf_file} not found at {pdf_path}")
                continue
            
            try:
                print(f"  Loading {pdf_file}...")
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        chunks = self._chunk_text(text)
                        for chunk in chunks:
                            self.documents.append(Document(
                                content=chunk,
                                source=pdf_file,
                                page=page_num + 1,
                                metadata={"type": "contract", "format": "pdf"}
                            ))
                print(f"  \u2713 Loaded {pdf_file} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  \u274c Error loading {pdf_file}: {e}")

    def _load_owners_manuals(self):
        """Load owner's manuals from Toyota Europe website or local files"""
        for model, variants in MANUAL_MODELS.items():
            for model_type, generation in variants:
                model_type = model_type.replace(" ", "_")
                file_name_prefix = f"{model}_{model_type}"
                if not glob.glob(str(MANUALS_DIR / f"{file_name_prefix}*.pdf")):
                    print(f"Fetching manual for {model} {model_type} ({generation})...")
                    scrape_toyota_manual(
                        model=model,
                        model_type=model_type,
                        generation_text=generation
                    )
                else:
                    print(f"Manual for {model} {model_type} already exists locally.")

        manuals = glob.glob(str(MANUALS_DIR / "*.pdf"))
        
        for pdf_file in manuals:
            pdf_path = Path(pdf_file)
            try:
                print(f"  Loading {pdf_path.name}...")
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    total_chunks = 0
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        chunks = self._chunk_text(text)
                        for chunk in chunks:
                            self.documents.append(Document(
                                content=chunk,
                                source=pdf_path.name,
                                page=page_num + 1,
                                metadata={"type": "manual", "format": "pdf"}
                            ))
                            total_chunks += len(chunks)
                print(f"  ✓ Loaded {pdf_path.name} ({total_chunks} chunks)")
            except Exception as e:
                print(f"  ❌ Error loading {pdf_path.name}: {e}")

    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into 512-token chunks using splitter"""
        if not text or not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def _save_index(self):
        """Save FAISS index and documents"""
        faiss.write_index(self.index, str(self.index_path / "faiss.index"))
        
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"✓ Index saved to {self.index_path}")
    
    def _load_index(self):
        """Load existing FAISS index and documents"""
        print("Loading existing index...")
        self.index = faiss.read_index(str(self.index_path / "faiss.index"))
        
        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        print(f"✓ Loaded index with {len(self.documents)} documents")
    
    def retrieve(self, question: str, top_k: int = None) -> List[tuple[Document, float]]:
        """
        Retrieve most relevant documents
        
        Args:
            question: The query to search for
            top_k: Number of documents to retrieve (default from config)
        
        Returns:
            List of (document, similarity_score) tuples
        """
        top_k = top_k or DEFAULT_TOP_K
        
        # Encode query
        query_embedding = self.model.encode([question])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        
        # Return documents with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def add_documents(self, new_docs: List[Document]):
        """Add new documents to existing index"""
        if not new_docs:
            return
        
        # Generate embeddings
        texts = [doc.content for doc in new_docs]
        embeddings = self.model.encode(texts)
        
        # Attach embeddings
        for doc, emb in zip(new_docs, embeddings):
            doc.embedding = emb
        
        # Add to index
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings_normalized.astype('float32'))
        
        # Add to documents list
        self.documents.extend(new_docs)
        
        # Save
        self._save_index()
        print(f"✓ Added {len(new_docs)} documents to index")

    def answer(self, question: str, documents: List[Document], client: OpenAI) -> tuple[str, int, float]:
        """Generate answer from retrieved documents
        
        Returns:
            Tuple of (answer, tokens_used, cost_usd)
        """
        context = "\n\n".join([
            f"[{doc.source}, page {doc.page}]\n{doc.content}"
            for doc, _ in documents
        ])
        
        prompt = format_rag_answer_prompt(question, context)
        response = call_llm(client, prompt, verbosity_level="medium", reasoning_level="medium", return_usage=True)
        return response.text, response.total_tokens, response.calculate_cost()


# Demo usage
if __name__ == "__main__":
    print("Building Production RAG System...")
    print("=" * 70)
    
    rag = RAGTool()
    
    # Test retrieval
    test_queries = [
        "What is the Toyota warranty?",
        "Where is the tire repair kit?",
        "What is the maintenance schedule for RAV4?",
        "Tell me about hybrid battery warranty"
    ]
    
    print("\n" + "=" * 70)
    print("Testing Retrieval")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        results = rag.retrieve(query, top_k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. [{doc.source}, page {doc.page}] (score: {score:.3f})")
            print(f"   {doc.content[:150]}...")
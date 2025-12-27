import os

# Set CUDA device before importing torch - must be done first!
cuda_device = os.environ.get("CUDA_DEVICE", None)
if cuda_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
else:
    import random
    os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, 1))

import json
import logging
import argparse
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
import warnings
import re

import torch
import numpy as np
import faiss
import datasets
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, AutoModel, HfArgumentParser
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from collections import defaultdict
from bm25_index import BM25RetrieverLunce
from utils import load_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TOP_BM25_RETRIEVAL = 2000

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_corpus(corpus_path: str):
    """Load corpus using datasets library"""
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def load_model(model_path: str, use_fp16: bool = False):
    """Load transformer model and tokenizer"""
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """Apply pooling to transformer outputs"""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_docs(corpus, doc_idxs):
    """Load documents by indices"""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

class Encoder:
    """Text encoder for queries and documents"""
    
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """Encode texts to embeddings"""
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs['attention_mask'],
                self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        # Clean up GPU memory
        del inputs, output
        torch.cuda.empty_cache()
        
        return query_emb

class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        """Retrieve topk relevant documents in corpus."""
        pass

    def _batch_search(self, query_list, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)
    
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)


class DenseRetriever(BaseRetriever):
    """Dense retriever based on pre-built faiss index."""

    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            # Check if faiss-gpu is available
            if hasattr(faiss, 'GpuMultipleClonerOptions'):
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.shard = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            else:
                logger.warning("faiss-gpu not installed, falling back to CPU. Install faiss-gpu for GPU support.")

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method, 
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        
    def _get_doc(self, index: str):
        return load_docs(self.corpus, [index])[0]

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False, search_indices = []):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        batch_size = self.batch_size
        results = []
        scores = []

        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + batch_size]
            
            batch_emb = self.encoder.encode(query_batch)
            if len(search_indices) >0:
                sel = faiss.IDSelectorArray(search_indices)
                params = faiss.SearchParameters(sel=sel)
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num, params= params)
            else:
                batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            scores.extend(batch_scores)
            results.extend(batch_results)
            
            # Clean up GPU memory
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
        
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    """Get dense retriever (BM25 is handled separately via BM25RetrieverLunce)"""
    # Note: BM25 is now handled by BM25RetrieverLunce (pure Python, no Java)
    # This function only returns DenseRetriever
    return DenseRetriever(config)

class BaseCrossEncoder:
    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)

    def _passage_to_string(self, doc_item):
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        return f"(Title: {title}) {text}"

    def rerank(self, queries: List[str], documents: List[List[dict]]):
        """
        Rerank documents for each query
        documents: list of list of dicts, where each dict is a document
        """
        assert len(queries) == len(documents)

        pairs = []
        qids = []
        doc_ids = []
        for qid, query in enumerate(queries):
            for doc_item in documents[qid]:
                doc = self._passage_to_string(doc_item)
                pairs.append((query, doc))
                qids.append(qid)
                doc_ids.append(doc_item["id"])

        scores = self._predict(pairs)
        query_to_doc_scores = defaultdict(list)

        assert len(scores) == len(pairs) == len(qids)
        for i in range(len(pairs)):
            query, doc = pairs[i]
            score = scores[i] 
            qid = qids[i]
            doc_id = doc_ids[i]
            query_to_doc_scores[qid].append((doc, score, doc_id))

        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores

    def _predict(self, pairs: List[tuple]):
        raise NotImplementedError 

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        raise NotImplementedError

class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model, batch_size=32, device="cuda"):
        super().__init__(model, batch_size, device)

    def _predict(self, pairs: List[tuple]):
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        return scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for retriever (from retrieval_rerank_server.py style)"""
    retrieval_method: str = field(default="e5")
    retrieval_model_path: str = field(default="intfloat/e5-base-v2")
    retrieval_pooling_method: str = field(default="mean")
    retrieval_query_max_length: int = field(default=256)
    retrieval_use_fp16: bool = field(default=True)
    retrieval_batch_size: int = field(default=128)
    retrieval_topk: int = field(default=200)  # Get 35 for reranking
    index_path: str = field(default="indexes/dense/e5_base_v2.index")
    corpus_path: str = field(default="data/corpus/processed_corpus.jsonl")
    faiss_gpu: bool = field(default=True)

@dataclass
class RerankerConfig:
    """Configuration for reranker (from rerank_server.py)"""
    max_length: int = field(default=512)
    rerank_topk: int = field(default=10)  # Return top 10 after reranking
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

def convert_title_format(text):
    """Convert title format (from retrieval_rerank_server.py)"""
    # Use regex to extract the title and the content
    match = re.match(r'\(Title:\s*([^)]+)\)\s*(.+)', text, re.DOTALL)
    if match:
        title, content = match.groups()
        return f'"{title}"\n{content}'
    else:
        return text

# ============================================================================
# FastAPI Models
# ============================================================================

class SearchRequest(BaseModel):
    queries: List[str]
    topk_retrieval: Optional[int] = 200  # Dense retrieval candidates for reranking
    topk_rerank: Optional[int] = 10     # Final results after reranking
    return_scores: bool = False

class SearchResponse(BaseModel):
    result: List[List[Dict[str, Any]]]
    processing_time: float

class VisitRequest(BaseModel):
    url: str

class HealthResponse(BaseModel):
    status: str
    pipeline_loaded: bool
    device: Optional[str] = None

class StatsResponse(BaseModel):
    corpus_size: int
    index_size: int
    retriever_model: str
    reranker_model: str

# ============================================================================
# FastAPI Application
# ============================================================================

# Global components
retriever = None
reranker = None
retriever_config = None
reranker_config = None
bm25_retriever = None

# Initialize configurations from environment variables
retriever_config = RetrieverConfig(
    retrieval_method=os.environ.get("RETRIEVER_NAME", "e5"),
    index_path=os.environ.get("INDEX_PATH", "data/corpus/e5_Flat.index"),
    corpus_path=os.environ.get("CORPUS_PATH", "data/corpus/corpus.jsonl"),
    retrieval_topk=int(os.environ.get("RETRIEVAL_TOPK", "35")),  # 35 candidates for reranking
    faiss_gpu=os.environ.get("FAISS_GPU", "true").lower() == "true",
    retrieval_model_path=os.environ.get("RETRIEVER_MODEL", "intfloat/e5-base-v2"),
    retrieval_pooling_method=os.environ.get("RETRIEVAL_POOLING_METHOD", "mean"),
    retrieval_query_max_length=int(os.environ.get("RETRIEVAL_QUERY_MAX_LENGTH", "256")),
    retrieval_use_fp16=os.environ.get("RETRIEVAL_USE_FP16", "true").lower() == "true",
    retrieval_batch_size=int(os.environ.get("RETRIEVAL_BATCH_SIZE", "128")),
)

reranker_config = RerankerConfig(
    rerank_topk=int(os.environ.get("RERANKING_TOPK", "10")),  # Top 10 after reranking
    rerank_model_name_or_path=os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2"),
    batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "64")),
)

from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global retriever, reranker, retriever_config, reranker_config, bm25_retriever, doc_store
    
    logger.info("Loading retrieval and reranking components...")
    logger.info(f"Index path: {retriever_config.index_path}")
    logger.info(f"Corpus path: {retriever_config.corpus_path}")
    logger.info(f"CUDA device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # load json file mapping doc_id to full content
    doc_store = load_index("./saved_index_data")

    logger.info("Loading dense retriever...")
    retriever = get_retriever(retriever_config)
    
    # Load reranker
    logger.info("Loading reranker...")
    reranker = get_reranker(reranker_config)
    
    bm25_index_path = os.environ.get("BM25_INDEX_PATH", None)

    bm25_cache_dir = os.environ.get("BM25_CACHE_DIR", retriever_config.corpus_path + "_bm25_cache")
    logger.info(f"BM25 cache dir: {bm25_cache_dir}")
    logger.info("Loading/Building BM25 index with bm25s...")
    bm25_retriever = BM25RetrieverLunce(retriever.corpus, is_corpus=True, cache_dir=bm25_cache_dir)
    
    logger.info("FlashRAG pipeline initialized successfully")
    yield


app = FastAPI(title="FlashRAG-style Server", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    pass

@app.post("/visit")
async def visit(request: VisitRequest):
    id = request.url.split("_")[-1]
    return {
        "result": [[{
            "title": doc_store.get(str(id))['title'],
            "text": doc_store.get(str(id))['full_contents'],
        }]]
    }
    
@app.post("/retrieve", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Main search endpoint that combines retrieval and reranking
    Based on retrieval_rerank_server.py logic
    """
    if retriever is None or reranker is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = time.time()
        
        # get candidates using BM25 (using bm25s - pure Python, no Java needed)
        results_ids, _ = bm25_retriever._search(request.queries[0], TOP_BM25_RETRIEVAL)
        
        # Retrieve documents
        retrieved_docs = retriever.batch_search(
            query_list=request.queries,
            num=request.topk_retrieval,
            return_score=False,
            search_indices = results_ids
        )

        # Rerank documents
        reranked = reranker.rerank(request.queries, retrieved_docs)
        # Format response 
        response = []
        for i, doc_scores in reranked.items():
            combined = []
            seen_titles = set()
            
            for doc, score, doc_id in doc_scores:
                if len(combined) >= request.topk_rerank:
                    break
                
                converted_doc = convert_title_format(doc)
                lines = converted_doc.split('\n', 1)
                title = lines[0].strip('"') if lines else "No title"
                
                if title in seen_titles:
                    continue
                
                seen_titles.add(title)
                
                text = lines[1] if len(lines) > 1 else ""
                
                doc_dict = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "contents": converted_doc
                }
                
                if request.return_scores:
                    doc_dict["score"] = float(score)
                
                combined.append(doc_dict)
            
            response.append(combined)
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            result=response,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if retriever is not None and reranker is not None else "unhealthy",
        pipeline_loaded=retriever is not None and reranker is not None,
        device=str(next(iter(retriever.encoder.model.parameters())).device) if retriever and hasattr(retriever, 'encoder') else None
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics"""
    if retriever is None or reranker is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        corpus_size = len(retriever.corpus) if hasattr(retriever, 'corpus') else 0
        index_size = retriever.index.ntotal if hasattr(retriever, 'index') else 0
        
        return StatsResponse(
            corpus_size=corpus_size,
            index_size=index_size,
            retriever_model=retriever_config.retrieval_model_path,
            reranker_model=reranker_config.rerank_model_name_or_path
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main Function (for running with python directly)
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch FlashRAG-style server")
    
    # CUDA device argument
    parser.add_argument("--cuda_device", type=str, default=None, help="CUDA device to use (e.g., '0', '1', '0,1')")
    
    # Retriever arguments
    parser.add_argument("--index_path", type=str, default=None, help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default=None, help="Local corpus file.")
    parser.add_argument("--retrieval_topk", type=int, default=None, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default=None, help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default=None, help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', default=None, help='Use GPU for computation')
    
    # Reranker arguments  
    parser.add_argument("--reranking_topk", type=int, default=None, help="Number of reranked passages for one query.")
    parser.add_argument("--reranker_model", type=str, default=None, help="Path of the reranker model.")
    parser.add_argument("--reranker_batch_size", type=int, default=None, help="Batch size for the reranker inference.")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=2223, help="Server port")
    
    cmd_args = parser.parse_args()
    
    # Override environment variables with command line arguments if provided
    if cmd_args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.cuda_device
    if cmd_args.index_path is not None:
        os.environ["INDEX_PATH"] = cmd_args.index_path
    if cmd_args.corpus_path is not None:
        os.environ["CORPUS_PATH"] = cmd_args.corpus_path
    if cmd_args.retrieval_topk is not None:
        os.environ["RETRIEVAL_TOPK"] = str(cmd_args.retrieval_topk)
    if cmd_args.retriever_name is not None:
        os.environ["RETRIEVER_NAME"] = cmd_args.retriever_name
    if cmd_args.retriever_model is not None:
        os.environ["RETRIEVER_MODEL"] = cmd_args.retriever_model
    if cmd_args.faiss_gpu is not None:
        os.environ["FAISS_GPU"] = str(cmd_args.faiss_gpu).lower()
    if cmd_args.reranking_topk is not None:
        os.environ["RERANKING_TOPK"] = str(cmd_args.reranking_topk)
    if cmd_args.reranker_model is not None:
        os.environ["RERANKER_MODEL"] = cmd_args.reranker_model
    if cmd_args.reranker_batch_size is not None:
        os.environ["RERANKER_BATCH_SIZE"] = str(cmd_args.reranker_batch_size)
    
    # Reinitialize configs with updated environment variables
    retriever_config = RetrieverConfig(
        retrieval_method=os.environ.get("RETRIEVER_NAME", "e5"),
        index_path=os.environ.get("INDEX_PATH", "data/corpus/e5_Flat.index"),
        corpus_path=os.environ.get("CORPUS_PATH", "data/corpus/corpus.jsonl"),
        retrieval_topk=int(os.environ.get("RETRIEVAL_TOPK", "35")),  # 35 candidates for reranking
        faiss_gpu=os.environ.get("FAISS_GPU", "true").lower() == "true",
        retrieval_model_path=os.environ.get("RETRIEVER_MODEL", "intfloat/e5-base-v2"),
        retrieval_pooling_method=os.environ.get("RETRIEVAL_POOLING_METHOD", "mean"),
        retrieval_query_max_length=int(os.environ.get("RETRIEVAL_QUERY_MAX_LENGTH", "256")),
        retrieval_use_fp16=os.environ.get("RETRIEVAL_USE_FP16", "true").lower() == "true",
        retrieval_batch_size=int(os.environ.get("RETRIEVAL_BATCH_SIZE", "128")),
    )
    
    reranker_config = RerankerConfig(
        rerank_topk=int(os.environ.get("RERANKING_TOPK", "10")),  # Top 10 after reranking
        rerank_model_name_or_path=os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2"),
        batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", "64")),
    )
    
    # Launch the server
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port)
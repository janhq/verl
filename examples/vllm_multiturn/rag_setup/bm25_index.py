import bm25s
import json, re
import gzip
import os
import datasets
import Stemmer
import logging

logger = logging.getLogger(__name__)

def load_corpus(corpus_path: str):
    """Load corpus using datasets library"""
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus


def load_docs(corpus, doc_idxs):
    """Load documents by indices"""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

class BM25RetrieverLunce:
    def __init__(self, corpus_path_or_corpus, is_corpus=False, cache_dir=None):
        """
        Args:
            corpus_path_or_corpus: Either a path string or a pre-loaded corpus dataset
            is_corpus: If True, corpus_path_or_corpus is a pre-loaded corpus
            cache_dir: Directory to save/load BM25 index. If None, uses corpus_path + "_bm25_cache"
        """
        if is_corpus:
            self.corpus = corpus_path_or_corpus
            self.cache_dir = cache_dir
        else:
            logger.info("BM25: Loading corpus...")
            self.corpus = load_corpus(corpus_path=corpus_path_or_corpus)
            # Default cache dir based on corpus path
            if cache_dir is None:
                self.cache_dir = corpus_path_or_corpus + "_bm25_cache"
            else:
                self.cache_dir = cache_dir
        
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = self._load_or_build_index()
    
    def _load_or_build_index(self):
        """Load index from cache if exists, otherwise build and save"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            logger.info(f"BM25: Loading cached index from {self.cache_dir}...")
            retriever = bm25s.BM25.load(self.cache_dir, load_corpus=False)
            logger.info("BM25: Cached index loaded successfully!")
            return retriever
        else:
            logger.info(f"BM25: No cached index found, building new index...")
            retriever = self._build_index()
            
            # Save to cache
            if self.cache_dir:
                logger.info(f"BM25: Saving index to {self.cache_dir}...")
                os.makedirs(self.cache_dir, exist_ok=True)
                retriever.save(self.cache_dir)
                logger.info("BM25: Index saved to cache!")
            
            return retriever
    
    def _build_index(self):
        logger.info(f"BM25: Building index for {len(self.corpus)} documents...")
        
        # Extract texts directly from corpus (more efficient)
        corpus_texts = [re.sub(r'[^\w\s]', '', doc["contents"]) for doc in self.corpus]
        
        logger.info("BM25: Tokenizing corpus...")
        tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=self.stemmer)
        
        logger.info("BM25: Indexing tokens...")
        retriever = bm25s.BM25()
        retriever.index(tokens)
        
        logger.info("BM25: Index built successfully!")
        return retriever
    
    def _search(self, query: str, num: int):
        results, scores = self.retriever.retrieve(bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer), k=num)
        return results[0], scores[0]
    
    
    
if __name__ == "__main__":
    bm25_ = BM25RetrieverLunce("/mnt/nas/alex/deep-research/src/rag_setup/data/corpus/corpus.jsonl")
    print(bm25_._search("Mc Donald", 5))
    result = bm25_._search(" Donald", 5)
    print(load_docs(bm25_.corpus, result[0][0]))
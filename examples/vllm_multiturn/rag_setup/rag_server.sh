# corpus_file=/mnt/nas/bachvd/Code-Agent/verl/data/searchR1_processed_direct/data/wiki-18.jsonl # jsonl
# save_dir=data/corpus
# retriever_name=e5 # this is for indexing naming
# retriever_model=intfloat/e5-base-v2

echo "Starting FlashRAG server..."
# python flashrag_server.py \
#     --index_path $save_dir/${retriever_name}_Flat.index \
#     --corpus_path $corpus_file \
#     --retrieval_topk 25 \
#     --retriever_name $retriever_name \
#     --retriever_model $retriever_model \
#     --reranking_topk 10 \
#     --reranker_model "cross-encoder/ms-marco-MiniLM-L12-v2" \
#     --reranker_batch_size 64 \
#     --host "0.0.0.0" \
#     --port 3030 \
#     --faiss_gpu \
#     --workers 64

export CUDA_DEVICE=0 
export INDEX_PATH="/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchr1/data/wiki-18_e5.index" 
export CORPUS_PATH="/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchr1/data/wiki-18.jsonl" 
export RETRIEVAL_TOPK=200 
export RERANKING_TOPK=10
export BM25_CACHE_DIR="/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchr1/data/bm25_cache"
export FAISS_GPU=false 

uvicorn flashrag_server:app --host 0.0.0.0 --port 3030 --workers 1
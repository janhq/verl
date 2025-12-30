# preprocess_corpus.py
import json
import re
import os
from pathlib import Path
import shutil
import datasets
import logging
import uuid
from collections import defaultdict
import pickle

logger = logging.getLogger(__name__)


class DocumentStore:
    def __init__(self, id_to_title, title_to_content):
        self.id_to_title = id_to_title
        self.title_to_content = title_to_content

    def get(self, chunk_id):
        title = self.id_to_title.get(chunk_id)
        if title is None:
            return None
        
        return {
            'title': title,
            'full_contents': self.title_to_content.get(title, "")
        }

    def __contains__(self, chunk_id):
        return chunk_id in self.id_to_title

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def preprocess_corpus(input_file, output_file):
    """Preprocess corpus for indexing"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in):
            try:
                doc = json.loads(line)
                
                # Clean title and text
                title = clean_text(doc.get('title', ''))
                text = clean_text(doc.get('text', ''))
                
                # Create combined content for indexing
                content = f'"{title}"\n{text}' if title else text
                
                # Prepare output document
                output_doc = {
                    'id': doc.get('id', f'doc_{line_num}'),
                    'title': title,
                    'text': text,
                    'contents': content
                }
                
                f_out.write(json.dumps(output_doc) + '\n')
                
            except json.JSONDecodeError:
                print(f"Error parsing line {line_num}")
                continue

def build_and_save_index(corpus_path: str, save_dir: str, temp_dir: str = "temp_shards_storage", num_proc: int = 64):
    
    def create_title(batch):
        titles = []
        for content in batch['contents']:
            title = content.split("\n")[0].strip('"')
            titles.append(title)
        return {'title': titles}

    def clean_content(batch):
        contents = []
        for content in batch['contents']:
            parts = content.split("\n")
            if len(parts) > 1:
                content_new = "\n".join(parts[1:])
            else:
                content_new = ""
            contents.append(content_new)
        return {'contents': contents}
        
    def create_group_shards(batch, rank):
        local_group = defaultdict(list)
        iterator = zip(batch['id'], batch['title'], batch['contents'])
        
        for id, title, content in iterator:
            local_group[title].append((id, content))
        
        unique_name = f"{rank}_{uuid.uuid4().hex}.json"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        file_path = os.path.join(temp_dir, unique_name)
        
        with open(file_path, 'w') as f:
            json.dump(local_group, f)
            
        return batch

    print(f"Building index from: {corpus_path}")
    
    db = datasets.load_dataset('json', data_files=corpus_path, split="train", num_proc=16)
    db = db.map(create_title, batch_size=10000, num_proc=num_proc, batched=True)
    db = db.map(clean_content, batch_size=10000, num_proc=num_proc, batched=True)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

        db.map(
            create_group_shards, 
            batched=True, 
            batch_size=10000, 
            num_proc=num_proc, 
            with_rank=True 
        )

    raw_groups = defaultdict(list)
    shard_files = os.listdir(temp_dir)

    for filename in shard_files:
        file_path = os.path.join(temp_dir, filename)
        try:
            with open(file_path, 'r') as f:
                shard_data = json.load(f)
            
            for title, items in shard_data.items():
                raw_groups[title].extend(items)
        except Exception as e:
            logger.error(f"Error reading shard {filename}: {e}")

    id_to_title = {}
    title_to_content = {}

    for title, chunks in raw_groups.items():
        chunks.sort(key=lambda x: x[0])
        full_contents = " ".join([item[1].strip() for item in chunks])
        title_to_content[title] = full_contents
        
        for chunk_id, _ in chunks:
            id_to_title[chunk_id] = title
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Saving artifacts to {save_dir}...")
    
    with open(os.path.join(save_dir, "id_to_title.pkl"), "wb") as f:
        pickle.dump(id_to_title, f)
        
    with open(os.path.join(save_dir, "title_to_content.pkl"), "wb") as f:
        pickle.dump(title_to_content, f)
    
    print("Build complete.")

def load_index(save_dir: str):
    print(f"Loading index from {save_dir}...")
    
    with open(os.path.join(save_dir, "id_to_title.pkl"), "rb") as f:
        id_to_title = pickle.load(f)
        
    with open(os.path.join(save_dir, "title_to_content.pkl"), "rb") as f:
        title_to_content = pickle.load(f)
        
    print(f"Loaded. IDs: {len(id_to_title)}, Documents: {len(title_to_content)}")
    return DocumentStore(id_to_title, title_to_content)

if __name__ == "__main__":
    DATA_PATH = "/mnt/nas/bachvd/Code-Agent/verl/data/janv2_searchr1/data/wiki-18.jsonl"
    INDEX_DIR = "./saved_index_data"

    # build_and_save_index(DATA_PATH, INDEX_DIR, num_proc=64)

    if os.path.exists(INDEX_DIR):
        doc_store = load_index(INDEX_DIR)
        
        # Test
        test_id = "10" 
        result = doc_store.get(test_id)
        print(result)
        if result:
            print(f"Retrieved: {result['title']}")
    else:
        print("Index not found. Please run build_and_save_index first.")
import os
import numpy as np
import requests
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv
from semantic_text_splitter import MarkdownSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Helper to get embedding from OpenAI
def get_openai_embedding(text: str, api_key: str) -> List[float]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# Use MarkdownSplitter for better chunking

def get_chunks(file_path: str, chunk_size: int = 1500) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    splitter = MarkdownSplitter(chunk_size)
    chunks = splitter.chunks(content)
    return chunks

def process_markdown_folder(folder_path: str, source_label: str, chunk_size: int = 1500):
    all_chunks = []
    all_sources = []
    all_files = []
    all_urls = []
    abs_folder_path = os.path.abspath(folder_path) # Get absolute path for reliable comparison

    for fname in tqdm(os.listdir(folder_path)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(folder_path, fname) # Original logic for fpath is fine for listdir
        
        # If you intend to walk through subdirectories, the logic for fpath and is_root_file check would need adjustment.
        # Assuming for now os.listdir gives files directly in folder_path, so they are all effectively root files relative to folder_path.
        # If tools-in-data-science-public has subdirectories, os.walk would be needed here instead of os.listdir.
        # For simplicity with current os.listdir, all files are treated as being in the top level of folder_path.

        chunks = get_chunks(fpath, chunk_size=chunk_size)
        all_chunks.extend(chunks)
        all_sources.extend([source_label]*len(chunks))
        all_files.extend([fname]*len(chunks))
        
        # URL logic
        if source_label == "discourse":
            thread_id = fname.replace("thread_", "").replace(".md", "")
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{thread_id}"
            all_urls.extend([url]*len(chunks))
        elif source_label == "course":
            filename_without_ext = os.path.splitext(fname)[0]
            
            # Check if the file is a root-level README.md or index.md
            # This check assumes fname is directly in folder_path, which is true with os.listdir
            is_root_readme_or_index = (fname.lower() == "readme.md" or fname.lower() == "index.md")
            
            if is_root_readme_or_index:
                url = "https://tds.s-anand.net/#/README"
            else:
                url = f"https://tds.s-anand.net/#/../{filename_without_ext}"
            all_urls.extend([url]*len(chunks))
        else:
            all_urls.extend([""]*len(chunks))
    return all_chunks, all_sources, all_files, all_urls

def main():
    # Set your folders here
    discourse_md_folder = "discourse_threads_md"
    course_md_folder = "tools-in-data-science-public"
    
    # Chunk and collect all data
    discourse_chunks, discourse_sources, discourse_files, discourse_urls = process_markdown_folder(discourse_md_folder, "discourse", chunk_size=1500)
    course_chunks, course_sources, course_files, course_urls = process_markdown_folder(course_md_folder, "course", chunk_size=1500)
    all_chunks = discourse_chunks + course_chunks
    all_sources = discourse_sources + course_sources
    all_files = discourse_files + course_files
    all_urls = discourse_urls + course_urls

    # Generate embeddings
    embeddings = []
    for chunk in tqdm(all_chunks, desc="Embedding chunks"):
        emb = get_openai_embedding(chunk, OPENAI_API_KEY)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    all_chunks = np.array(all_chunks)
    all_sources = np.array(all_sources)
    all_files = np.array(all_files)
    all_urls = np.array(all_urls)

    # Save to npz
    np.savez("embeddings.npz", embeddings=embeddings, texts=all_chunks, sources=all_sources, files=all_files, urls=all_urls)
    print(f"Saved {len(all_chunks)} chunks and embeddings to embeddings.npz")

if __name__ == "__main__":
    main()
    # filepath = "tools-in-data-science-public/development-tools.md"
    # chunks = get_chunks(filepath, chunk_size=1000)
    # embedding = get_openai_embedding(chunks[0], OPENAI_API_KEY)
    # print(len(embedding), embedding[:10])   # Print first 10 dimensions of the embedding
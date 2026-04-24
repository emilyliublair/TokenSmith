#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata
"""

import os
import pickle
import pathlib
import re
import json
from typing import List, Dict, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig, print_chunk_stats
from src.preprocessing.extraction import extract_sections_from_markdown
from src.instrumentation.profiler import time_it

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']


def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
    chapters_to_index: Optional[List[int]] = None,
    use_doc2query: bool = False,
    gen_model_path: Optional[str] = None,
) -> None:
    """
    Extract sections, chunk, embed, and build both FAISS and BM25 indexes.

    Persists:
        - {prefix}.faiss
        - {prefix}_bm25.pkl
        - {prefix}_chunks.pkl
        - {prefix}_sources.pkl
        - {prefix}_meta.pkl
        - {prefix}_page_to_chunk_map.json
    """
    all_chunks: List[str] = []
    faiss_texts: List[str] = []
    bm25_texts: List[str] = []
    sources: List[str] = []
    metadata: List[Dict] = []

    # FOR LOGGING PURPOSES ONLY
    doc2query_log: Dict[int, List[str]] = {}

    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )
    
    if chapters_to_index:
        sections = [s for s in sections if s.get('chapter') in chapters_to_index]

    page_to_chunk_ids = {}
    current_page = 1
    total_chunks = 0
    heading_stack = []

    # Step 1: Chunk
    for i, c in enumerate(sections):
        current_level = c.get('level', 1)
        chapter_num = c.get('chapter', 0)

        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))

        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        sub_chunks = chunker.chunk(c['content'])
        page_pattern = re.compile(r'--- Page (\d+) ---')

        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            chunk_pages = set()
            fragments = page_pattern.split(sub_chunk)

            if fragments[0].strip():
                page_to_chunk_ids.setdefault(current_page, set()).add(total_chunks + sub_chunk_id)
                chunk_pages.add(current_page)

            for idx in range(1, len(fragments), 2):
                try:
                    new_page = int(fragments[idx]) + 1
                    if fragments[idx + 1].strip():
                        page_to_chunk_ids.setdefault(new_page, set()).add(total_chunks + sub_chunk_id)
                        chunk_pages.add(new_page)
                    current_page = new_page
                except (IndexError, ValueError):
                    continue

            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()

            if c["heading"] == "Introduction":
                continue

            expanded_text = clean_chunk

            if use_doc2query and gen_model_path:
                @time_it(stage_name="Doc2Query_Generation")
                def _generate_synthetic_queries():
                    from src.query_enhancement import generate_doc2query_questions
                    return generate_doc2query_questions(clean_chunk, gen_model_path)
                
                try:
                    synthetic_queries = _generate_synthetic_queries()
                    if synthetic_queries:
                        def _is_useful_question(question: str, chunk: str, min_overlap: int = 2) -> bool:
                            stopwords = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for", "with"}
                            chunk_words = {w.lower().strip('.,?!') for w in chunk.split() if w.lower() not in stopwords}
                            q_words = {w.lower().strip('.,?!') for w in question.split() if w.lower() not in stopwords}
                            return len(chunk_words & q_words) >= min_overlap

                        synthetic_queries = [q.strip() for q in synthetic_queries if len(q.strip().split()) >= 6]
                        doc2query_log[total_chunks + sub_chunk_id] = synthetic_queries

                        expanded_text = clean_chunk + "\n\n" + "\n".join(synthetic_queries)
                except Exception as e:
                    print(f"Warning: Doc2Query failed for chunk {total_chunks + sub_chunk_id}: {e}")

            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": total_chunks + sub_chunk_id,
            }

            chunk_prefix = (
                f"Description: {full_section_path} Content: "
                if use_headings else ""
            )

            original_idx = len(all_chunks)

            display_chunk = chunk_prefix + clean_chunk
            
            search_chunk = chunk_prefix + expanded_text 

            all_chunks.append(display_chunk) 
            sources.append(markdown_file)
            metadata.append(meta)
            
            faiss_texts.append(search_chunk)
            bm25_texts.append(search_chunk)

        total_chunks += len(sub_chunks)

    # Save page-to-chunk map
    final_map = {page: sorted(list(ids)) for page, ids in page_to_chunk_ids.items()}
    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    # Print chunk stats before embedding - TODO: wrap in some verbose cfg param
    # print_chunk_stats(all_chunks, chunk_size_in_chars=chunk_config.recursive_chunk_size)

    # Step 2: Load embedder
    print(f"Loading embedding model (n_ctx={embedding_model_context_window})...")
    embedder = SentenceTransformer(
        embedding_model_path,
        n_ctx=embedding_model_context_window,
    )
    print(f"Embedding {len(all_chunks):,} chunks sequentially...")

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(workers=4)
        try:
            embeddings = embedder.encode_multi_process(
                faiss_texts,
                pool,
                batch_size=4,
            )
        finally:
            embedder.stop_multi_process_pool(pool)
    else:
        embeddings = embedder.encode(
            faiss_texts, 
            batch_size=16, 
            show_progress_bar=True
        )

    # Step 3: Build FAISS index
    @time_it(stage_name="FAISS_Index_Build")
    def _build_faiss_index():
        print(f"Building FAISS index for {len(all_chunks):,} chunks...")
        dim = embeddings.shape[1]
        faiss_idx = faiss.IndexFlatL2(dim)
        faiss_idx.add(embeddings)
        faiss.write_index(faiss_idx, str(artifacts_dir / f"{index_prefix}.faiss"))
        print(f"FAISS Index built successfully: {index_prefix}.faiss")
        return faiss_idx
    
    index = _build_faiss_index() # Execute the wrapped function

    # Step 4: Build BM25 index
    @time_it(stage_name="BM25_Index_Build")
    def _build_bm25_index():
        print(f"Building BM25 index for {len(all_chunks):,} chunks...")
        tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in bm25_texts]
        bm25_idx = BM25Okapi(tokenized_chunks)
        with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
            pickle.dump(bm25_idx, f)
        print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")
        return bm25_idx
        
    bm25_index = _build_bm25_index() # Execute the wrapped function

    # Step 5: Persist remaining artifacts
    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

    output_file = artifacts_dir / f"{index_prefix}_info.json"
    index_info = {
        "textbooks": [
            {
                "markdown_file": markdown_file,
                "chapters": chapters_to_index if chapters_to_index else ["all"],
                "status": "partial" if chapters_to_index else "full"
            }
        ]
    }
    with open(output_file, "w") as f:
        json.dump(index_info, f, indent=2)
    print(f"Saved index information: {output_file}")

    if use_doc2query and doc2query_log:
        log_file = artifacts_dir / f"{index_prefix}_doc2query_samples.json"
        with open(log_file, "w") as f:
            json.dump(doc2query_log, f, indent=4)
        print(f"Saved Doc2Query generated questions to: {log_file}")

# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """Lowercase and tokenize text for BM25 indexing."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    return text.split()
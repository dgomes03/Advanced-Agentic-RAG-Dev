import os
import io
import json
import random
import string
import pickle
import numpy as np
from collections import defaultdict
import faiss
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import torch
import pdfplumber
import re
import pytesseract
from PIL import Image
import concurrent.futures
from nltk.tokenize import word_tokenize

# === Server Configuration ===
ENABLE_SERVER = False
SERVER_HOST = 'localhost'
SERVER_PORT = 5050

if ENABLE_SERVER:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import threading
    import time
    from flask import Response, stream_with_context

# Global RAG system for server mode
rag_system = None

num_threads = "1"

os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["PYTORCH_NUM_THREADS"] = num_threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

# === Constants ===
DOCUMENTS_DIR = "/Users/diogogomes/Documents/Uni/Tese Mestrado/RAG_database"
MODEL_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-8b-instruct-mixed-6-8-bit"
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MULTIVECTOR_INDEX_PATH = "Indexes/FAISS_index.pkl"
BM25_DATA_PATH = "Indexes/BM25_index.pkl"
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MAX_RESPONSE_TOKENS = 1000
EVAL = False


def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None
    
# Server routes (only if server is enabled)
if ENABLE_SERVER:
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin Resource Sharing

    @app.route('/query', methods=['POST'])
    def handle_query():
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        def generate():
            try:
                # Stream the response token by token
                for text_chunk in Generator.answer_query_with_llm(
                    query,
                    rag_system.llm_model, 
                    rag_system.llm_tokenizer,
                    rag_system.search_documents_tool,
                    []
                ):
                    yield f"data: {json.dumps({'text': text_chunk})}\n\n"
                    time.sleep(0.01)  # Small delay to make streaming visible
                    
                # Signal the end of the stream
                yield "data: [DONE]\n\n"
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({'status': 'ready'})

class Indexer:
    def __init__(self, documents_dir=DOCUMENTS_DIR, embedding_model_name=EMBEDDING_MODEL_NAME,
                 faiss_index_path=MULTIVECTOR_INDEX_PATH, bm25_data_path=BM25_DATA_PATH):
        self.documents_dir = documents_dir
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.bm25_data_path = bm25_data_path

    @staticmethod
    def process_single_pdf(file_info): 
        file_path, doc_id, max_chunk_length = file_info
        doc_chunks = []
        doc_metadata = []
        filename = os.path.basename(file_path)

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    elements = []

                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = '\n'.join([' | '.join(map(str, row)) for row in table])
                            elements.append({
                                'type': 'table',
                                'content': table_text,
                                'meta': {
                                    'page': page_number,
                                    'filename': filename,
                                    'doc_id': doc_id,
                                    'table_idx': table_idx
                                }
                            })

                    # Extract paragraph text
                    text = page.extract_text()
                    if text:
                        # Use sentence-aware splitting instead of fixed length
                        sentences = re.split(r'(?<=[.!?])\s+', text)  # Better sentence splitting
                        current_chunk = []
                        current_length = 0
                        chunks = []
                        
                        for sentence in sentences:
                            if current_length + len(sentence) > max_chunk_length and current_chunk:
                                chunks.append(" ".join(current_chunk))
                                current_chunk = [sentence]
                                current_length = len(sentence)
                            else:
                                current_chunk.append(sentence)
                                current_length += len(sentence) + 1  # +1 for space
                        
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        
                        for chunk_idx, chunk in enumerate(chunks):
                            elements.append({
                                'type': 'text',
                                'content': chunk,
                                'meta': {
                                    'page': page_number,
                                    'filename': filename,
                                    'doc_id': doc_id,
                                    'chunk_idx': chunk_idx
                                }
                            })

                    # OCR for images
                    if page.images:
                        for img_idx, img_dict in enumerate(page.images):
                            img_x0 = max(0, img_dict["x0"])
                            img_top = max(0, img_dict["top"])
                            img_x1 = min(page.width, img_dict["x1"])
                            img_bottom = min(page.height, img_dict["bottom"])
                            try:
                                image = page.crop((img_x0, img_top, img_x1, img_bottom)).to_image(resolution=150)
                                pil_img = image.original
                                ocr_text = pytesseract.image_to_string(pil_img).strip()
                                if ocr_text:
                                    elements.append({
                                        'type': 'image_ocr',
                                        'content': ocr_text,
                                        'meta': {
                                            'page': page_number,
                                            'filename': filename,
                                            'doc_id': doc_id,
                                            'img_idx': img_idx
                                        }
                                    })
                            except Exception as e:
                                print(f"Failed OCR on image {img_idx} of page {page_number} in {filename}: {e}")

                    # Save elements
                    for elem in elements:
                        doc_chunks.append(elem['content'])
                        meta = elem['meta'].copy()
                        meta['type'] = elem['type']
                        doc_metadata.append(meta)

            print(f"Processed {filename}")
        except Exception as e:
            print(f"Could not read/process {filename}: {e}")

        return doc_chunks, doc_metadata

    def load_and_content_chunk_pdfs_parallel(self, max_chunk_length=2000, max_workers=10): # multicore for individual pdfs
        if not os.path.isdir(self.documents_dir):
            print(f"Error: Directory not found at {self.documents_dir}")
            return [], []

        pdf_files = [os.path.join(self.documents_dir, f) for f in os.listdir(self.documents_dir) if f.endswith(".pdf")]

        file_info_list = [(file_path, idx, max_chunk_length) for idx, file_path in enumerate(pdf_files)]

        all_chunks = []
        all_metadata = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_pdf, file_info) for file_info in file_info_list]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs..."):
                chunks, metadata = future.result()
                all_chunks.extend(chunks)
                all_metadata.extend(metadata)

        return all_chunks, all_metadata

    def get_embeddings(self, chunks, model=None):
        if model is None:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            model = SentenceTransformer(self.embedding_model_name)
        print("Generating embeddings for text chunks...")
        return model.encode(chunks, show_progress_bar=True)

    @staticmethod
    def build_multi_vector_index(embeddings, metadata, texts):
        index = []
        for emb, meta, text in zip(embeddings, metadata, texts):
            index.append({
                "embedding": np.array(emb, dtype=np.float32),
                "text": text,
                "metadata": meta
            })
        return index

    @staticmethod
    def tokenize_for_bm25(documents):
        return [word_tokenize(doc.lower()) for doc in documents]

    def build_bm25_index(self, documents):
        tokenized_corpus = self.tokenize_for_bm25(documents)
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25

    def save_indices(self, multi_vector_index, bm25, faiss_index=None):
        save_pickle(multi_vector_index, self.faiss_index_path)
        save_pickle(bm25, self.bm25_data_path)
        if faiss_index:
            faiss.write_index(faiss_index, "Indexes/faiss_index.faiss")

    def load_indices(self):
        multi_vector_index = load_pickle(self.faiss_index_path)
        bm25 = load_pickle(self.bm25_data_path)
        faiss_index = None
        if os.path.exists("Indexes/faiss_index.faiss"):
            faiss_index = faiss.read_index("Indexes/faiss_index.faiss")
        return multi_vector_index, bm25, faiss_index
    
    @staticmethod
    def build_faiss_index(multi_vector_index):
        embeddings = np.array([entry["embedding"] for entry in multi_vector_index]).astype('float32')
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]

        # Use more efficient index type for large datasets
        nlist = min(100, len(multi_vector_index) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
        # Train the index
        samples = embeddings if len(embeddings) < 10000 else embeddings[np.random.choice(len(embeddings), 10000, replace=False)]
        index.train(samples)

        #index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Enable parallel search
        index.nprobe = max(1, nlist // 10)  # Number of clusters to search
        faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        return index

class Retriever:
    def __init__(self, multi_vector_index, embedding_model, faiss_index,
                 bm25,
                 llm_model, llm_tokenizer, reranker):
        self.multi_vector_index = multi_vector_index
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.reranker = reranker

    @staticmethod
    def normalize_scores(scores):
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score - min_score < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

    @staticmethod
    def retrieve_with_faiss(query_embedding, faiss_index, k=10):
        query_vec = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        distances, indices = faiss_index.search(query_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((idx, float(dist)))
        return results

    @staticmethod
    def retrieve_with_bm25(tokenized_query, bm25, top_k=10):
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]

    def expand_query(self, query):
        system_prompt = (
            "You are a query rephraser and expander for document search.\n"
            "Generate two alternative phrasings "
            "that capture different ways the same question could be asked for retrieving relevant context.\n"
            "Also provide between 1 to 5 buzz words from the phrases that can be relevant for retrieving relevant context.\n"
            "IMPORTANT: Dont include any other commentary like 'Buzz words:' at the end!"
        )
        
        message = (
            f"ORIGINAL QUERY:\n{query}")

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True, 
            tokenize=False
        )

        response = generate(
            self.llm_model, 
            self.llm_tokenizer, 
            prompt=prompt, 
            max_tokens=200,
            verbose=False)
        
        response = re.sub(r'(?i)\b(?:here\s+are\s+the\s+)?buzz\s*words?\s*:\s*', '', response)

        alternatives = []
        
        for line in response.split('\n'):
            line = line.strip('"-•* 0123456789.').strip()
            if 2 < len(line):
                alternatives.append(line)
        if not alternatives:
            alternatives = [query]
        
        #print("=== FINAL ALTERNATIVES ===") # ver as query expansions
        #for alt in alternatives:
        #    print(alt)
        
        return alternatives

    def rerank_chunks_with_metadata(self, query, chunks, metadata_list, top_n=None):
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        combined = list(zip(chunks, metadata_list, scores))
        reranked = sorted(combined, key=lambda x: x[2], reverse=True)
        if top_n:
            reranked = reranked[:top_n]
        return [(chunk, meta) for chunk, meta, score in reranked]


    def combined_retrieval(
            self,
            query,
            k=7,
            weight_dense=0.6,
            weight_sparse=0.4,
            rerank_top_n=5,
            use_summarization=False,
    ):
        # Expand query variants for retrieval
        expanded_queries = [query] + self.expand_query(query)

        # Dense retrieval scoring
        dense_scores = defaultdict(float)
        query_embeddings = self.embedding_model.encode(expanded_queries)
        for query_emb in query_embeddings:
            faiss_results = self.retrieve_with_faiss(query_emb, self.faiss_index, k=k)
            for idx, score in faiss_results:
                dense_scores[idx] += score
        dense_scores_norm = self.normalize_scores(dense_scores)

        # Sparse retrieval scoring (BM25)
        sparse_scores = defaultdict(float)
        for exp_query in expanded_queries:
            tokenized_query = exp_query.lower().split()
            bm25_results = self.retrieve_with_bm25(tokenized_query, self.bm25, top_k=k)
            for idx, score in bm25_results:
                sparse_scores[idx] += score
        sparse_scores_norm = self.normalize_scores(sparse_scores)

        # Combine scores
        all_doc_indices = set(dense_scores_norm.keys()) | set(sparse_scores_norm.keys())
        combined_scores = {}
        for idx in all_doc_indices:
            dscore = dense_scores_norm.get(idx, 0)
            sscore = sparse_scores_norm.get(idx, 0)
            combined_scores[idx] = weight_dense * dscore + weight_sparse * sscore

        # Select top-k indices
        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]

        # Retrieve chunks and their metadata for top candidates
        retrieved_chunks = [self.multi_vector_index[idx]["text"] for idx in top_indices]
        retrieved_metadata = [self.multi_vector_index[idx]["metadata"] for idx in top_indices]

        # Rerank chunks while preserving metadata
        reranked = self.rerank_chunks_with_metadata(query, retrieved_chunks, retrieved_metadata, top_n=rerank_top_n)

        # Store metadata for citation tracking
        self.last_retrieved_metadata = [meta for _, meta in reranked]
        
        if use_summarization:
            reranked_texts = [chunk for chunk, meta in reranked]
            final_context = Generator.summarize_passages(reranked_texts, self.llm_model, self.llm_tokenizer)
        else:
            final_context = "\n---\n".join([chunk for chunk, meta in reranked])
            
        return final_context
    
    def search_documents_tool(self, query: str) -> str:
        try:
            return self.combined_retrieval(
                query=query,
                k=80,
                weight_dense=0.6,
                weight_sparse=0.4,
                rerank_top_n=6, # final number chunks dado ao LLM
                use_summarization=False
            )
        except Exception as e:
            return f"Error: {str(e)}"

class Generator:
    @staticmethod
    def summarize_passages(passages, llm_model, llm_tokenizer):
        context = "\n".join(passages)
        prompt = (
            "Summarize the following retrieved passages.\n"
            f"{context}\n"
            "Summary:"
        )
        summary = generate(llm_model, llm_tokenizer, prompt=prompt, max_tokens=700, verbose=False)
        return summary

    def answer_query_with_llm(query, llm_model, llm_tokenizer, search_documents_tool, history):
        def search_wikipedia(query: str) -> str:
            """Fetch top Wikipedia results using Wikipedia API"""
            try:
                import urllib.parse
                import urllib.request
                import json
                
                # Prepare API request
                encoded_query = urllib.parse.quote(query)
                url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                    f"&srsearch={encoded_query}&format=json&srlimit=3"
                )
                
                # Add user-agent header (required by Wikipedia)
                headers = {'User-Agent': 'YourApp/1.0 (contact@example.com)'}
                req = urllib.request.Request(url, headers=headers)
                
                # Execute request with timeout
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode('utf-8'))
                
                # Process results
                if not data.get('query', {}).get('search'):
                    return "No relevant Wikipedia articles found."
                    
                results = []
                for i, item in enumerate(data['query']['search'], 1):
                    # Clean HTML snippets from Wikipedia
                    snippet = (
                        item['snippet']
                        .replace('<span class="searchmatch">', '**')
                        .replace('</span>', '**')
                    )
                    results.append(f"{i}. **{item['title']}**: {snippet}")
                
                return "Wikipedia Results:\n" + "\n".join(results)
                
            except Exception as e:
                return f"Wikipedia Error: {str(e)}. Try rephrasing your query."
        
        tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "Search for documents relevant to the user's query.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_wikipedia",
                        "description": "Search Wikipedia for factual information and general knowledge.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query string."}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        
        # FIX 1: Properly format conversation history as a conversation, not as query-response pairs
        conversation = [
            {"role": "system", "content": "You are a helpful assistant with document search and Wikipedia capabilities. If you use tools, answer based *only* on the provided results. After receiving tool results, provide a final answer, DO NOT make additional tool calls."}
        ]

        # FIX 2: Parse actual conversation history properly
        if history:
            # Assuming history is a list of dicts with 'query' and 'response' keys
            for item in history:
                conversation.append({"role": "user", "content": item["query"]})
                conversation.append({"role": "assistant", "content": item["response"]})
        
        # Add current query
        conversation.append({"role": "user", "content": query})

        # temperatura e top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)

        # First call: include tools definition
        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tools=tools,
            tokenize=False
        )
        
        prompt_cache = make_prompt_cache(llm_model)
        
        # Generate first response
        response = generate(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=prompt,
            max_tokens=MAX_RESPONSE_TOKENS,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=prompt_cache,
            verbose=True
        )
        
        # Check if response contains a tool call request
        response_text = response.strip()

        print("\nFORMATTED PROMPT:")
        print(prompt)
        
        if "[TOOL_CALLS][" in response_text:
            print("\nModel requested to use a tool. Processing tool call...")
            
            try:
                # Extract tool call JSON
                start_idx = response_text.find("[TOOL_CALLS][") + len("[TOOL_CALLS][")
                end_idx = response_text.find("]", start_idx)
                tool_call_str = response_text[start_idx:end_idx]
                
                # Parse as JSON (MAY contain arguments as objects)
                tool_calls = json.loads(tool_call_str)
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]

                # CRITICAL: Convert arguments to JSON strings IF they're dicts
                for call in tool_calls:
                    if isinstance(call.get("arguments"), dict):
                        # Convert dict → JSON string (with proper escaping)
                        call["arguments"] = json.dumps(call["arguments"])
                
                # Format for conversation (MUST preserve string format)
                formatted_tool_calls = []
                for call in tool_calls:
                    call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                    formatted_tool_calls.append({
                        "id": call_id,
                        "function": {
                            "name": call["name"],
                            "arguments": call["arguments"]  # NOW GUARANTEED TO BE STRING
                        },
                        "type": "function"
                    })
                
                # Add to conversation
                conversation.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": formatted_tool_calls
                })
                
                # Execute tools
                for tool_call in formatted_tool_calls:
                    tool_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]  # NOW ALWAYS STRING
                    
                    # Parse the JSON string to get query
                    try:
                        tool_args = json.loads(args_str)
                        query_str = tool_args["query"]
                    except (json.JSONDecodeError, KeyError) as e:
                        return f"Tool call error: Invalid arguments format - {str(e)}"
                    
                    print(f"Executing tool: {tool_name} with query: '{query_str}'")
                    
                    # Execute the tool
                    if tool_name == "search_documents":
                        tool_result = search_documents_tool(query_str)
                    elif tool_name == "search_wikipedia":
                        tool_result = search_wikipedia(query_str)
                    else:
                        tool_result = f"Error: Unknown tool: {tool_name}"

                    conversation.append({
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call["id"]
                    })

                # Generate final response without tool results
                print("\nGenerating final response with tool results...")
                prompt = llm_tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                prompt_cache = make_prompt_cache(llm_model)
                final_response = generate(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_RESPONSE_TOKENS,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    prompt_cache=prompt_cache,
                    verbose=True
                )

                print("\nFORMATTED PROMPT:")
                print(prompt)

                return final_response, prompt
                
            except Exception as e:
                print(f"Error processing tool calls: {str(e)}")
                import traceback
                traceback.print_exc()
                return response
        return response
            
    def eval(query, llm_model, llm_tokenizer, rag_response):
        system_prompt = f"""
            You are a RAG evaluator judge. Please evaluate the following response based on these criteria with a 0-10 scale:
            1. Context Relevance: How well retrieved documents align with the user's query and are able to address it.
                Example: Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The biggest earthquake in Lisbon was in 1755, which killed more than 30000 people. Score: 10; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The white house is where the president lives. Score: 0; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The earthquake. Lisbon. Earthquake in Lisbon. The earthquake in Lisbon. The earthquake in Lisbon. The biggest earthquake in Lisbon Score: 3;
            2. Groundedness: How accurately the response is based on the retrieved context.
            3. Answer Relevance: How well the response addresses the original query.
            4. Faithfulness: Is the output contradicting the retrieved facts?
            5. Contextual Recall: Did we retrieve ALL the info needed?
            6. Contextual Relevancy: What % of retrieved chunks actually matter?
            Be precise and objective on your scores! Be very critic.
        """

        user_prompt = f"""
            \n\nUser Query: {query}\n\n
            Response and context: {rag_response}
        """
    
        eval_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
        
        sampler = make_sampler(temp=0.2, top_k=30, top_p=0.6)

        prompt = llm_tokenizer.apply_chat_template(
            eval_conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        eval_response = generate(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=prompt,
            sampler=sampler,
            verbose=False
        )
        print("Evaluation Results:\n", eval_response)

# ==== Main Runtime ====
if __name__ == "__main__":
    print("Attempting to load saved FAISS index and BM25 index...")
    indexer = Indexer()
    multi_vector_index, bm25, faiss_index = indexer.load_indices()

    if multi_vector_index is None or bm25 is None:
        print("No saved indices found. Proceeding to build indices.")
        print("Loading and chunking documents...")
        chunks, chunk_metadata = indexer.load_and_content_chunk_pdfs_parallel()
        if not chunks:
            print("No documents were loaded. Exiting.")
            exit()

        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(MODEL_PATH)
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

        print("\nGenerating embeddings...")
        chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
        
        print("\nBuilding the multi-vector index...")
        multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)
        indexer.save_indices(multi_vector_index, None, None)  # Save BM25

        print("\nBuilding BM25 index...")
        bm25 = indexer.build_bm25_index(chunks)
        indexer.save_indices(multi_vector_index, bm25, None)

    else:
        print("Loaded saved FAISS and BM25 indices from disk.")
        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(MODEL_PATH) #, adapter_path='/Users/diogogomes/Documents/Uni/Tese Mestrado/Fine-tuning/adapters')
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if faiss_index is None and multi_vector_index is not None:
        print("Building FAISS index...")
        faiss_index = Indexer.build_faiss_index(multi_vector_index)
        indexer.save_indices(multi_vector_index, bm25, faiss_index)

    print("\nCreating Retriever...")
    retriever = Retriever(
        multi_vector_index,
        embedding_model,
        faiss_index,  #FAISS index
        bm25,
        llm_model,
        llm_tokenizer,
        reranker,
    )

    print(f"Index contains {len(multi_vector_index)} chunks.")
    # Force cleanup
    import gc
    gc.collect()


    if ENABLE_SERVER:
        rag_system = retriever
        print(f"Starting RAG server on http://{SERVER_HOST}:{SERVER_PORT}")
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False)
    else:
        # Run in interactive console mode
        print("\nReady to answer queries. (Type 'exit' to quit)")
        history = []
        try:
            while True:
                query = input("\nEnter your query: ")
                if query.lower() == "exit":
                    break

                #context = "[NO DOCUMENTS]"

                rag_response = Generator.answer_query_with_llm(
                    query,
                    llm_model, 
                    llm_tokenizer,
                    retriever.search_documents_tool,
                    history
                    )
                
                #print("\n\nDEBUG:")
                #print(rag_response)

                # Extract just the response text, not the tuple
                if isinstance(rag_response, tuple):
                    response_text = rag_response[0]  # First element is the response text
                else:
                    response_text = rag_response

                # Store only the text in history
                history.append({"query": query, "response": response_text})
                history = history[-6:] # mantem apenas ultimas 3 msgs no chat history
                
                ##################### EVAL BREAKS AFTER 1 RESPONSE!!!! #########################

                if EVAL:
                    print("==========")
                    eval = Generator.eval(
                        query,
                        llm_model, 
                        llm_tokenizer,
                        rag_response
                    )

        except KeyboardInterrupt:
            print("\nExiting program.")
import os
import faiss
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import torch
from PIL import Image
from nltk.tokenize import word_tokenize

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
MAX_RESPONSE_TOKENS = 1000
eval_path = '/Users/diogogomes/Documents/Uni/Tese Mestrado/eval.json'

llm_model, llm_tokenizer = load(MODEL_PATH)

# Load and parse structured data
import json

# Load and parse structured data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
    
    # Check if it's a valid JSON array
    try:
        if content.startswith('[') and content.endswith(']'):
            return json.loads(content)
    except:
        pass
    
    # If not, try to parse as multiple JSON objects separated by commas
    # Add brackets to make it a valid JSON array
    if not content.startswith('['):
        content = '[' + content + ']'
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If still failing, try to extract individual JSON objects
        pattern = r'\{[^}]+\}'
        matches = re.findall(pattern, content)
        data_list = []
        
        for match in matches:
            try:
                data = json.loads(match)
                data_list.append(data)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                match = match.replace('\n', '\\n').replace('\t', '\\t')
                try:
                    data = json.loads(match)
                    data_list.append(data)
                except:
                    print(f"Failed to parse: {match[:100]}...")
                    continue
        
        return data_list

# Load data
data_list = load_json_data(eval_path)
print(f"Found {len(data_list)} evaluation items")

# Process each evaluation item
for i, data in enumerate(data_list):
    print(f"Processing evaluation item {i+1}/{len(data_list)}")
    print(f"Question: {data['question'][:100]}...")
    
    # Format retrieved chunks for better readability
    formatted_chunks = "\n".join([f"Chunk {i+1}: {chunk}" 
                                for i, chunk in enumerate(data["retrieved_chunks"])])

    conversation = [
        {"role": "system", "content": """
        You are a RAG evaluator judge. Please evaluate the following response based on these criteria with a 0-10 scale:
        1. Context Relevance: How well retrieved documents align with the user's query and are able to address it.
            Example: Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The biggest earthquake in Lisbon was in 1755, which killed more than 30000 people. Score: 10; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The white house is where the president lives. Score: 0; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The earthquake. Lisbon. Earthquake in Lisbon. The earthquake in Lisbon. The earthquake in Lisbon. The biggest earthquake in Lisbon Score: 3;
        2. Groundedness: How accurately the response is based on the retrieved context.
        3. Answer Relevance: How well the response addresses the original query.
        Be precise and objective on your scores!
        Provide only the scores in JSON format:
        {"context_relevance": score, "groundedness": score, "answer_relevance": score}
        """},
        {"role": "user", "content": f"""
        Question: {data['question']}
        
        Retrieved Context:
        {formatted_chunks}
        
        Generated Answer:
        {data['answer']}
        
        Please provide scores:
        """}
    ]

    # temperatura e top sampling
    sampler = make_sampler(temp=0.1, top_k=10, top_p=0.3)

    prompt = llm_tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    prompt_cache = make_prompt_cache(llm_model)

    response = generate(
        model=llm_model,
        tokenizer=llm_tokenizer,
        prompt=prompt,
        max_tokens=MAX_RESPONSE_TOKENS,
        sampler=sampler,
        prompt_cache=prompt_cache,
        verbose=False
    )

    try:
        scores = json.loads(response)
        print(f"Context Relevance: {scores['context_relevance']}/10")
        print(f"Groundedness: {scores['groundedness']}/10")
        print(f"Answer Relevance: {scores['answer_relevance']}/10")
    except json.JSONDecodeError:
        print("Raw response:", response)
    
    print("-" * 50)
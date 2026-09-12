This repository contains the implementation and experimental notebooks for the master's thesis:
**"Optimizing a Dense-Vector Retrieval-Augmented Generation Pipeline for Domain-Specific Question Answering in Central Bank Speeches."**

The project investigates a dense-vector RAG pipeline for domain-specific question answering over central bank speeches.
The experiments evaluate the effects of text splitting, chunk size, chunk overlap, embedding models, vector database backends, domain-adaptive embedding fine-tuning, and cross-encoder reranking on retrieval and generated-answer performance.
The pipeline uses locally executable embedding and language models. The models were downloaded and executed directly within Google Colab without relying on external inference APIs.
---

## Baseline Configuration
The baseline RAG configuration uses:
- Character-based text splitting
- Chunk size: 1000 characters
- Chunk overlap: 30 characters
- `sentence-transformers/all-mpnet-base-v2`
- Qdrant vector database
- Top-4 dense retrieval
- Meta-Llama-3-8B-Instruct
- 4-bit quantization
- No embedding fine-tuning
- No cross-encoder reranking
The evaluation benchmark contains 150 curated question-answer pairs.
---

## Experimental Configurations
The experiments investigate the following dimensions:
### 1. Text Splitting
- Character-based splitting (baseline)
- Recursive splitting
- Token-based splitting

### 2. Embedding Models
- `all-mpnet-base-v2` (baseline)
- BAAI embedding model
- Google EmbeddingGemma

### 3. Vector Databases
- Qdrant (baseline)
- FAISS
- Weaviate

### 4. Chunk Size
- 500 characters
- 1000 characters (baseline)
- 2000 characters

### 5. Chunk Overlap
- 30 characters (baseline)
- 100 characters
- 200 characters

### 6. Domain-Adaptive Embedding Fine-Tuning
The baseline MPNet embedding model is fine-tuned using domain-specific question-passage pairs to investigate its effect on retrieval and downstream answer generation.

### 7. Cross-Encoder Reranking
Cross-encoder reranking is evaluated as an optional post-retrieval component.
The reranking configuration retrieves an initial candidate pool of 50 chunks, reranks them using a cross-encoder, and supplies the top 4 reranked chunks to the language model.
---

## Evaluation
The evaluation is performed directly within each experimental notebook and uses a combination of retrieval-level and end-to-end metrics.
The evaluation framework includes:
- BERTScore Precision, Recall, and F1
- Median BERTScore F1
- Sample standard deviation of BERTScore F1
- Answer Coverage
- OAP (Overall Answering Performance) Index
- IDK (abstention) detection and response counts
- Retrieval Precision@4
- Retrieval Success@4 using the eRAG framework
BERTScore is computed over automatically detected non-IDK responses, while Coverage and OAP account for abstained responses. Retrieval-level metrics are evaluated using the eRAG framework.---

## Repository Structure
```text
central-bank-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
└── notebooks/
    ├── 00_baseline/
    ├── 01_text_splitting/
    ├── 02_embedding_models/
    ├── 03_vector_databases/
    ├── 04_chunk_size/
    ├── 05_chunk_overlap/
    ├── 06_embedding_finetuning/
    └── 07_cross_encoder_reranking/
Each notebook contains the corresponding experimental pipeline, evaluation procedure, and experimental outputs.

Running the Notebooks
The notebooks were developed in Google Colab and can be executed in a compatible Python/Jupyter environment.

Install the required dependencies with:
pip install -r requirements.txt

Before running the notebooks, prepare the required data according to the instructions in:
data/README.md

Reproducibility
The repository provides the complete experimental notebooks, dependency specifications, data preparation instructions, evaluation procedures, and configuration details required to inspect and reproduce the experiments.
Large model files, model caches, local vector database files, and temporary generated files are not included in the repository.

Software
The implementation uses Python and libraries including:

PyTorch
Hugging Face Transformers
Sentence Transformers
LangChain
Qdrant
FAISS
Weaviate
BERTScore
pandas
NumPy
scikit-learn
Exact packages are specified in requirements.txt. 

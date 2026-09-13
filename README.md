# Optimizing a Dense-Vector Retrieval-Augmented Generation Pipeline for Domain-Specific Question Answering in Central Bank Speeches

This repository contains the source code and experimental notebooks for the thesis:

**"Optimizing a Dense-Vector Retrieval-Augmented Generation Pipeline for Domain-Specific Question Answering in Central Bank Speeches."**

The project investigates the optimization of a dense-vector Retrieval-Augmented Generation (RAG) pipeline for domain-specific question answering over central bank speeches.

The evaluated pipeline uses locally executable embedding and language models and examines the effect of different retrieval and generation components on answer quality, answer coverage, abstention behavior, and retrieval performance.

The experiments were developed and tested in Google Colab using locally executable models. The language and embedding models are downloaded and executed directly within the computational environment without relying on external inference APIs.

---


## Experimental Scope

The experiments evaluate the effect of the following components and configurations:

1. Text splitting strategy
2. Embedding model
3. Vector storage backend
4. Chunk size
5. Chunk overlap
6. Domain-adaptive embedding fine-tuning
7. Cross-encoder reranking

The experiments are conducted using a common question-answering benchmark consisting of **150 curated domain-specific question-answer pairs** derived from central bank speeches.

The evaluated configurations are designed to isolate the effect of individual RAG components while keeping the remaining pipeline components as consistent as possible.

---


## Baseline Configuration

The baseline RAG configuration uses:

- **Text splitter:** Character-based splitting
- **Chunk size:** 1000 characters
- **Chunk overlap:** 30 characters
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Vector storage:** Qdrant
- **Retrieval:** Top-4 dense retrieval
- **Reranking:** Not used
- **Embedding fine-tuning:** Not used
- **Language model:** Meta-Llama-3-8B-Instruct
- **Quantization:** 4-bit
- **Evaluation benchmark:** 150 curated question-answer pairs

The baseline configuration is provided in:

`notebooks/00_baseline/baseline_rag.ipynb`

---


## Experimental Configurations

### 1. Text Splitting

The effect of different text splitting strategies is evaluated using:

- Character-based splitting
- Recursive character splitting
- Token-based splitting

The corresponding notebooks are located in:

`notebooks/01_text_splitting/`

---


### 2. Embedding Models

The following embedding models are compared:

- `sentence-transformers/all-mpnet-base-v2`
- BAAI BGE
- EmbeddingGemma

The corresponding notebooks are located in:

`notebooks/02_embedding_models/`

---


### 3. Vector Storage Backends

The following vector storage/retrieval backends are evaluated:

- Qdrant
- FAISS
- Weaviate

The corresponding notebooks are located in:

`notebooks/03_vector_databases/`

---


### 4. Chunk Size

The effect of different chunk sizes is evaluated using:

- 500 characters
- 1000 characters (baseline)
- 2000 characters

The corresponding notebooks are located in:

`notebooks/04_chunk_size/`

---


### 5. Chunk Overlap

The following chunk-overlap configurations are evaluated:

- 30 characters (baseline)
- 100 characters
- 200 characters

The corresponding notebooks are located in:

`notebooks/05_chunk_overlap/`

---


### 6. Domain-Adaptive Embedding Fine-Tuning

A domain-adaptive fine-tuning experiment is conducted using the MPNet embedding model.

The fine-tuned embedding model is evaluated within the RAG pipeline to examine whether domain-specific adaptation improves retrieval and downstream answering performance.

The corresponding notebook is located in:

`notebooks/06_embedding_finetuning/`

---


### 7. Cross-Encoder Reranking

A cross-encoder is evaluated as an optional post-retrieval component.

The reranking configuration retrieves an initial pool of **50 candidate chunks**, applies a cross-encoder to reorder the candidates, and retains the **top 4 passages** as the final contextual input to the language model.

The corresponding notebook is located in:

`notebooks/07_cross_encoder_reranking/`

---


## Evaluation

The RAG configurations are evaluated using both answer-level and retrieval-level measures.

### Answer-Level Evaluation

The following metrics are used:

- BERTScore Precision
- BERTScore Recall
- BERTScore F1
- Median BERTScore F1
- Sample standard deviation of BERTScore F1
- Answer Coverage
- Overall Answering Performance (OAP)
- "I don't know" (IDK) response detection and counts

BERTScore is calculated using `roberta-large` with layer 17.

For each configuration, BERTScore is calculated over the automatically detected non-IDK responses. Therefore, the number and identity of evaluated responses may differ between configurations.

The mean BERTScore represents semantic similarity conditional on the system providing a substantive answer, while Coverage and the OAP Index account for abstained responses.

### Retrieval-Level Evaluation

Retrieval performance is additionally evaluated using:

- Retrieval Precision@4
- Retrieval Success@4

These retrieval metrics are calculated using the eRAG evaluation procedure.

---


## Repository Structure

```text
central-bank-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    ├── 00_baseline/
    │   └── baseline_rag.ipynb
    ├── 01_text_splitting/
    │   ├── recursive_splitter.ipynb
    │   └── token_splitter.ipynb
    ├── 02_embedding_models/
    │   ├── baai-bge.ipynb
    │   └── embeddinggemma.ipynb
    ├── 03_vector_databases/
    │   ├── faiss.ipynb
    │   └── weaviate.ipynb
    ├── 04_chunk_size/
    │   ├── chunk_2000.ipynb
    │   └── chunk_500.ipynb
    ├── 05_chunk_overlap/
    │   ├── overlap_100.ipynb
    │   └── overlap_200.ipynb
    ├── 06_embedding_finetuning/
    │   └── finetuned_mpnet.ipynb
    └── 07_cross_encoder_reranking/
        └── cross_encoder_reranking.ipynb
Root Files and Directories
README.md — Project overview, experimental configurations, evaluation methodology, repository structure, and execution instructions.
requirements.txt — Python dependencies required to run the notebooks.
.gitignore — Files and directories excluded from version control.
data/ — Data preparation instructions and information about the source dataset.
notebooks/ — Baseline and experimental notebooks organized according to the evaluated RAG components.
Notebook Directories
00_baseline/ — Baseline RAG configuration.
01_text_splitting/ — Experiments comparing text splitting strategies.
02_embedding_models/ — Experiments comparing embedding models.
03_vector_databases/ — Experiments comparing vector storage backends.
04_chunk_size/ — Experiments evaluating different chunk sizes.
05_chunk_overlap/ — Experiments evaluating different chunk-overlap configurations.
06_embedding_finetuning/ — Domain-adaptive embedding fine-tuning experiment.
07_cross_encoder_reranking/ — Cross-encoder reranking experiment.

Each notebook contains the corresponding experimental pipeline, evaluation procedure, configuration settings, and experimental outputs.

Running the Notebooks

The notebooks were developed and tested in Google Colab and can be executed in a compatible Python/Jupyter environment.

To start the project:

Clone the repository.
Install the required Python dependencies:
pip install -r requirements.txt
Prepare the required data according to the instructions provided in:

data/README.md

Open the desired notebook under:

notebooks/

Run the notebook cells sequentially.

The baseline configuration is provided in:

notebooks/00_baseline/baseline_rag.ipynb

The remaining notebook directories contain the comparative experiments for the individual RAG components.

Some experiments require downloading pretrained embedding, reranking, or language models. These models are loaded and executed locally within the computational environment.

Data

The original central bank speech corpus is not redistributed in this repository.

Instructions for obtaining and preparing the dataset used in the study are provided in:

data/README.md

The experimental benchmark consists of 150 curated question-answer pairs derived from central bank speeches.

Users attempting to reproduce the experiments should follow the data preparation instructions and use the same dataset configuration described in the thesis.

Reproducibility

The repository provides the complete experimental notebooks, dependency specifications, data preparation instructions, evaluation procedures, and configuration details required to inspect and reproduce the experimental pipeline.

The notebooks contain the implementation of the baseline system and the comparative experiments described in the thesis.

Large model files, model caches, local vector database files, temporary generated files, and other environment-specific artifacts are not included in the repository.

The experiments were originally developed and tested in Google Colab. Reproduction may therefore require a compatible GPU environment and sufficient computational resources, particularly for experiments involving the Llama-3-8B-Instruct language model and embedding fine-tuning.

Software and Libraries

The project uses the following main software libraries and frameworks:

Python
PyTorch
Hugging Face Transformers
Sentence Transformers
Accelerate
BitsAndBytes
LangChain
Qdrant
FAISS
Weaviate
BERTScore
pandas
NumPy
scikit-learn
Jupyter / Google Colab

The required Python packages are specified in:

requirements.txt

Notes on Model Execution

The language and embedding models used in this project are locally executable within the computational environment.

The experiments do not depend on external inference APIs for model generation. Pretrained models are downloaded and executed directly in the Google Colab environment or another compatible local/Jupyter environment.

The specific model versions and configurations used in the thesis should be considered when reproducing the reported results.

Citation

If you use this repository or build upon the experimental implementation, please cite the associated thesis.

License

This project is released under the MIT License. See the LICENSE file for details.


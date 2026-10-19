# Role Simulation: Data Engineer — NLP ETL, Tokenization & RAG Chunking Specification

**Project**: Project 03 · NLP + LLM Fine-Tuning (Text Summarization & Domain RAG System)  
**Owner Role**: Data Engineer  
**DagsHub MLflow**: [`https://dagshub.com/29omchavan/mlops-project-3-nlp-rag-summarizer`](https://dagshub.com/29omchavan/mlops-project-3-nlp-rag-summarizer)  
**Hugging Face Hub**: [`https://huggingface.co/29Oms/mlops-project-3-lora-summarizer`](https://huggingface.co/29Oms/mlops-project-3-lora-summarizer)

---

## 1. Dual-Pipeline Data Architecture
Project 03 ingests and validates two complementary NLP datasets:
1. **Seq2Seq Dialogue Summarization Corpus (`summarization_train.json` / `summarization_val.json`)**: Modeled on SAMSum / MLOps engineering incident transcripts (`id`, `domain`, `dialogue`, `summary`) for PEFT LoRA fine-tuning of `google/flan-t5-small`.
2. **Multi-Document Technical Knowledge Base (`domain_knowledge_chunks.json`)**: Structured technical documentation chunks with metadata (`doc_id`, `title`, `category`, `source`, `chunk_text`) indexed into **Qdrant** and **BM25**.

---

## 2. Byte-Pair Encoding (BPE) Tokenization from First Principles (Andrej Karpathy Lecture)

Why do modern LLMs use subword tokenization (BPE / SentencePiece) instead of character-level or word-level vocabularies?
- **Word-level**: Suffers from massive vocabulary sizes ($>500\text{k}$) and fails with `<UNK>` on rare technical identifiers (`ks_2samp`, `ERR_CUDA_OOM_404`).
- **Character/Byte-level**: Vocabulary is tiny ($256$ UTF-8 bytes), but sequence length explodes ($T \times 4$), causing quadratic $O(T^2)$ self-attention bottlenecks.
- **Byte-Pair Encoding (`BytePairTokenizerFromScratch`)**:
  1. Initialize vocabulary $\mathcal{V}_0 = \{0, 1, \dots, 255\}$ with raw UTF-8 bytes.
  2. Count frequencies of all adjacent token pairs $(t_i, t_{i+1})$ in the corpus.
  3. Merge the most frequent pair $(p_1, p_2)$ into a new token ID $v_{\text{new}} = |\mathcal{V}|$ and repeat for $M = |\mathcal{V}_{\text{target}}| - 256$ merges.

---

## 3. Sequence Truncation & Padding Masking (`-100`)
When preparing target labels for `AutoModelForSeq2SeqLM` (`google/flan-t5-small`, `vocab_size = 32,128`), padding tokens (`tokenizer.pad_token_id == 0`) must be replaced with `-100`:

```python
labels[labels == tokenizer.pad_token_id] = -100
```

PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)` automatically excludes `-100` positions from gradient computation so the LoRA adapters do not waste capacity predicting padding tokens.

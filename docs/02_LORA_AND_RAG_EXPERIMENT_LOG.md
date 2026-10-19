# Role Simulation: ML Researcher — Self-Attention, LoRA Math & Hybrid RAG Experiment Log

**Project**: Project 03 · NLP + LLM Fine-Tuning (Text Summarization & Domain RAG System)  
**Owner Role**: ML Researcher  
**Hugging Face Model**: [`29Oms/mlops-project-3-lora-summarizer`](https://huggingface.co/29Oms/mlops-project-3-lora-summarizer)

---

## 1. Mathematical Foundations (Umar Jamil & Andrej Karpathy Reference Lectures)

### 1.1 Scaled Dot-Product Multi-Head Self-Attention
Given input sequence embeddings $X \in \mathbb{R}^{T \times d_{\text{model}}}$, each attention head $i \in \{1, \dots, h\}$ projects $X$ into Queries, Keys, and Values ($Q_i = X W_i^Q, K_i = X W_i^K, V_i = X W_i^V \in \mathbb{R}^{T \times d_k}$ where $d_k = d_{\text{model}} / h$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

Why divide by $\sqrt{d_k}$? If components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$, their dot product $q \cdot k = \sum_{j=1}^{d_k} q_j k_j$ has mean $0$ and variance $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance ($\sigma^2 = 1$), preventing the softmax from saturating into regions with vanishingly small gradients.

### 1.2 Low-Rank Adaptation (LoRA) Decomposition
During fine-tuning, weight updates $\Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ have low "intrinsic rank". LoRA constrains the update by decomposing $\Delta W = B A$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, and rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

- **Initialization**: $A \sim \mathcal{N}(0, \sigma^2)$ (Kaiming uniform) and $B = \mathbf{0}$, guaranteeing $\Delta W = B A = 0$ at step $t = 0$ so the fine-tuning run starts identically to the pretrained model.
- **Parameter Reduction on `google/flan-t5-small`**:
  - Total parameters: `~77,305,216`
  - Trainable LoRA parameters ($r = 8, \alpha = 16$ on `["q", "v"]`): `~344,064` (**`0.4451%`** of total parameters!).

---

## 2. Advanced Hybrid RAG: Reciprocal Rank Fusion (RRF) & Re-Ranking (James Briggs Series)

Naive dense vector search struggles with exact technical identifiers (such as `ERR_CUDA_OOM_404` or `ks_2samp`), while BM25 lexical search misses semantic paraphrases. Our `HybridRAGEngine` merges **Qdrant Cosine Vector Search** ($r_{\text{dense}}$) and **BM25Okapi** ($r_{\text{bm25}}$) via **Reciprocal Rank Fusion**:

$$\text{RRF}(d) = \frac{1}{k + r_{\text{dense}}(d)} + \frac{1}{k + r_{\text{bm25}}(d)}, \qquad k = 60$$

Followed by a Cross-Feature Lexical Overlap Re-Ranker and `[DOC-ID]` citation injection.

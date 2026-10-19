"""
First-Principles LLM Building Blocks (Andrej Karpathy + Umar Jamil Reference Implementations):
1. BytePairTokenizerFromScratch: Iterative byte-pair frequency counting & merging (Karpathy GPT Tokenizer)
2. ScaledDotProductMultiHeadAttention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
3. LoRALinearFromScratch: Low-Rank Adaptation W = W_0 + (alpha / r) * B @ A with frozen W_0
   and zero-initialized B so delta_W = 0 at step 0 (Umar Jamil LoRA from Scratch)
"""
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BytePairTokenizerFromScratch:
    """
    Implements Andrej Karpathy's Byte-Pair Encoding (BPE) algorithm over UTF-8 byte sequences (0..255).
    Iteratively merges the most frequent adjacent token pair into a new vocabulary ID >= 256.
    """

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _get_pair_stats(ids: List[int]) -> Counter:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge_pair(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int = 280) -> None:
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode("utf-8"))
        for i in range(num_merges):
            stats = self._get_pair_stats(ids)
            if not stats:
                break
            top_pair = stats.most_common(1)[0][0]
            new_id = 256 + i
            ids = self._merge_pair(ids, top_pair, new_id)
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._get_pair_stats(ids)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge_pair(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: List[int]) -> str:
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors="replace")


class ScaledDotProductMultiHeadAttention(nn.Module):
    """
    First-Principles Multi-Head Self-Attention (Vaswani et al. / Umar Jamil / Karpathy):
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + M) @ V
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, causal_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.w_o(context), attn_weights


class LoRALinearFromScratch(nn.Module):
    """
    First-Principles Low-Rank Adaptation (Hu et al., 2021 / Umar Jamil):
    Given frozen pretrained weight W_0 in R^(d_out x d_in), parameterizes update delta_W = B @ A
    where A in R^(r x d_in) ~ N(0, sigma^2), B in R^(d_out x r) = 0, and r << min(d_in, d_out):
        h = W_0 x + (alpha / r) * (B @ A) x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        pretrained_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r)

        self.pretrained = nn.Linear(in_features, out_features, bias=False)
        if pretrained_weight is not None:
            with torch.no_grad():
                self.pretrained.weight.copy_(pretrained_weight)
        # Freeze W_0
        self.pretrained.weight.requires_grad = False

        # Low-rank adapters A (Gaussian init) and B (Zero init so delta_W = 0 at t=0)
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.pretrained(x)
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

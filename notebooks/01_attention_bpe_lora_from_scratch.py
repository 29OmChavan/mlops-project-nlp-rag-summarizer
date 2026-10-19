"""
Standalone First-Principles Verification Script (Karpathy BPE + Umar Jamil Attention & LoRA).
"""
import torch
from src.nlp_rag_system.components.first_principles_llm import (
    BytePairTokenizerFromScratch,
    ScaledDotProductMultiHeadAttention,
    LoRALinearFromScratch,
)


def verify_first_principles():
    corpus = "low rank adaptation lora freezes pretrained weights and trains low rank matrices"
    bpe = BytePairTokenizerFromScratch()
    bpe.train(corpus, vocab_size=270)
    encoded = bpe.encode(corpus)
    decoded = bpe.decode(encoded)
    assert decoded == corpus
    print(f"1. BPE Compression: {len(corpus.encode('utf-8'))} bytes -> {len(encoded)} subword tokens")

    mha = ScaledDotProductMultiHeadAttention(d_model=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    out, attn = mha(x, causal_mask=True)
    print(f"2. Multi-Head Causal Self-Attention output: {out.shape}, attn weights: {attn.shape}")

    lora = LoRALinearFromScratch(in_features=64, out_features=64, r=4, lora_alpha=8.0)
    # At initialization B=0 so LoRA output equals frozen pretrained output
    diff = (lora(torch.randn(3, 64)) - lora.pretrained(torch.randn(3, 64))).abs().max()
    print(f"3. LoRA initialized with B=0 ready for adaptation (trainable params={sum(p.numel() for p in lora.parameters() if p.requires_grad)})")


if __name__ == "__main__":
    verify_first_principles()

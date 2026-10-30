
import torch
import torch.nn as nn
from torch.nn import functional as F


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        batch_size, sequence_length = token_ids.shape

        positions = torch.arange(
            sequence_length, device=token_ids.device
        )

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)

        # Causal mask prevents the model from seeing future tokens.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        logits = self.output_head(self.final_norm(x))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

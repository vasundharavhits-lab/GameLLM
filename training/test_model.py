
import torch
from models.tiny_transformer import TinyTransformer

torch.manual_seed(42)

model = TinyTransformer(vocab_size=100)

# A batch of 2 sequences, each containing 16 token IDs.
x = torch.randint(0, 100, (2, 16))

# For this smoke test, use arbitrary valid target IDs.
targets = torch.randint(0, 100, (2, 16))

logits, loss = model(x, targets)

print("Output shape:", tuple(logits.shape))
print("Loss:", round(loss.item(), 4))
print("Parameter count:", sum(p.numel() for p in model.parameters()))

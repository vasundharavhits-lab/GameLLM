
from pathlib import Path
import json
import torch
import torch.nn.functional as F

from models.tiny_transformer import TinyTransformer
from tokenizer.char_tokenizer import CharTokenizer

# ---------- Project paths ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATHS = [
    ROOT / "data" / "raw" / "npc_dialogues.txt",
    ROOT / "data" / "raw" / "npc_dialogues_extra.txt",
]
MODEL_PATH = ROOT / "models" / "tiny_transformer.pt"
VOCAB_PATH = ROOT / "tokenizer" / "vocab.json"

# ---------- Training settings ----------
# ---------- Training settings ----------
CONFIG_PATH = ROOT / "configs" / "tiny_config.json"

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

SEED = config.get("seed", 42)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if config.get("device", "auto") == "auto"
    and torch.cuda.is_available()
    else "cpu"
    if config.get("device", "auto") == "auto"
    else config.get("device", "cpu")
)

BLOCK_SIZE = config.get("block_size", 64)
BATCH_SIZE = config.get("batch_size", 8)
MAX_STEPS = config.get("max_iters", 1000)
EVAL_INTERVAL = config.get("eval_interval", 100)
EVAL_BATCHES = config.get("eval_batches", 10)
LEARNING_RATE = config.get("learning_rate", 0.001)

EMBEDDING_DIM = config.get("n_embd", 64)
NUM_HEADS = config.get("n_head", 4)
NUM_LAYERS = config.get("n_layer", 2)
DROPOUT = config.get("dropout", 0.1)
# ---------- Load and tokenize dialogue ----------
text = "\n".join(
    path.read_text(encoding="utf-8-sig")
    for path in DATA_PATHS
)
tokenizer = CharTokenizer(text)

all_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

if len(all_ids) <= BLOCK_SIZE + 10:
    raise ValueError("Dataset is too small. Add more NPC dialogue first.")

split_index = int(0.9 * len(all_ids))
train_data = all_ids[:split_index]
val_data = all_ids[split_index:]

print("Device:", DEVICE)
print("Characters:", len(text))
print("Vocabulary size:", tokenizer.vocab_size)
print("Training characters:", len(train_data))
print("Validation characters:", len(val_data))


def get_batch(data):
    starts = torch.randint(
        0, len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,)
    )

    x = torch.stack([data[i:i + BLOCK_SIZE] for i in starts])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts])

    return x.to(DEVICE), y.to(DEVICE)


# ---------- Initialize model from scratch ----------
model = TinyTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE
)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}

    for name, data in [
        ("train", train_data),
        ("validation", val_data),
    ]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results


# ---------- Training loop ----------
model.train()

for step in range(1, MAX_STEPS + 1):
    if step == 1 or step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"Step {step:03d} | "
            f"train loss: {losses['train']:.4f} | "
            f"validation loss: {losses['validation']:.4f}"
        )

    x, y = get_batch(train_data)
    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# ---------- Save model and vocabulary ----------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "block_size": BLOCK_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "chars": tokenizer.chars,
    },
    MODEL_PATH,
)

VOCAB_PATH.write_text(
    json.dumps(tokenizer.chars, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# ---------- Generate a sample ----------
model.eval()
context = torch.tensor(
    [tokenizer.encode("\nPlayer:")], dtype=torch.long, device=DEVICE
)

with torch.no_grad():
    for _ in range(250):
        context_crop = context[:, -BLOCK_SIZE:]
        logits, _ = model(context_crop)
        next_logits = logits[:, -1, :]
        probabilities = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        context = torch.cat([context, next_id], dim=1)
generated = tokenizer.decode(context[0].tolist())

print("\n--- Generated sample ---")
print(generated)
print("\nModel saved to:", MODEL_PATH)
print("Vocabulary saved to:", VOCAB_PATH)

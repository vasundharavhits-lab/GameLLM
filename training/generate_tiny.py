
from pathlib import Path
import torch
import torch.nn.functional as F

from models.tiny_transformer import TinyTransformer
from tokenizer.char_tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tiny_transformer.pt"
DATA_PATH = ROOT / "data" / "raw" / "npc_dialogues.txt"

DEVICE = "cpu"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

text = DATA_PATH.read_text(encoding="utf-8")
tokenizer = CharTokenizer(text)

model = TinyTransformer(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("GameLLM tiny model loaded.")
print("Type a prompt using characters found in the training data.")
print("Type exit to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.strip().lower() == "exit":
        break

    try:
        ids = tokenizer.encode(prompt)
    except ValueError as error:
        print("Prompt contains characters outside the vocabulary:", error)
        continue

    if not ids:
        print("Please enter a prompt.")
        continue

    context = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            context_crop = context[:, -checkpoint["block_size"]:]
            logits, _ = model(context_crop)

            next_logits = logits[:, -1, :] / TEMPERATURE
            # Keep only the 5 most likely next characters.
            k = min(5, next_logits.size(-1))
            top_values, top_indices = torch.topk(next_logits, k=k, dim=-1)
            filtered_logits = torch.full_like(next_logits, float("-inf"))
            filtered_logits.scatter_(1, top_indices, top_values)
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)


            context = torch.cat([context, next_id], dim=1)

    print("GameLLM:", tokenizer.decode(context[0].tolist()), "\n")

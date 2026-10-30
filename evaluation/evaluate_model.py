
from pathlib import Path
import sys
import math
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tiny_transformer import TinyTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tiny_transformer.pt"
DATA_PATHS = [
    ROOT / "data" / "raw" / "npc_dialogues.txt",
    ROOT / "data" / "raw" / "npc_dialogues_extra.txt",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 8


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}\n"
            "Train the tiny model first."
        )

    text = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in DATA_PATHS
        if path.exists()
    )

    if not text:
        raise ValueError("No training dialogue files were found.")

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )

    chars = checkpoint["chars"]
    char_to_id = {char: i for i, char in enumerate(chars)}
    ids = [char_to_id[c] for c in text if c in char_to_id]

    if len(ids) < checkpoint["block_size"] + 2:
        raise ValueError("Not enough text to evaluate this checkpoint.")

    # Evaluate on the final 10% of the available dialogue text.
    split_index = int(len(ids) * 0.9)
    validation_ids = torch.tensor(ids[split_index:], dtype=torch.long)

    block_size = checkpoint["block_size"]
    if len(validation_ids) <= block_size:
        raise ValueError("Validation data is too small.")

    model = TinyTransformer(
        vocab_size=checkpoint["vocab_size"],
        block_size=block_size,
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    losses = []

    with torch.no_grad():
        for start in range(0, len(validation_ids) - block_size, block_size):
            chunk = validation_ids[start:start + block_size + 1]

            if len(chunk) < block_size + 1:
                continue

            x = chunk[:-1].unsqueeze(0).to(DEVICE)
            y = chunk[1:].unsqueeze(0).to(DEVICE)

            _, loss = model(x, y)
            losses.append(loss.item())

    if not losses:
        raise ValueError("Could not create validation batches.")

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(mean_loss, 100))

    print("GameLLM Tiny Transformer Evaluation")
    print("----------------------------------")
    print(f"Device: {DEVICE}")
    print(f"Validation characters: {len(validation_ids)}")
    print(f"Validation batches: {len(losses)}")
    print(f"Mean validation loss: {mean_loss:.4f}")
    print(f"Approximate perplexity: {perplexity:.2f}")
    print("\nNote: Validation uses the final 10% of the combined text.")


if __name__ == "__main__":
    main()

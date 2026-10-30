
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.char_tokenizer import CharTokenizer

DATA_PATH = ROOT / "data" / "raw" / "npc_dialogues.txt"

text = DATA_PATH.read_text(encoding="utf-8")
tokenizer = CharTokenizer(text)

sample = "Eldrin: Welcome!"
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Vocabulary size:", tokenizer.vocab_size)
print("Original text:", sample)
print("Encoded IDs:", encoded)
print("Decoded text:", decoded)
print("Round-trip successful:", decoded == sample)

assert decoded == sample, "Tokenizer round-trip failed!"
print("Tokenizer test passed.")

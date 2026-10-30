
import json
from pathlib import Path


class NPCMemory:
    """Stores and retrieves simple facts during a game session."""

    def __init__(self, memory_file=None):
        if memory_file is None:
            project_root = Path(__file__).resolve().parents[1]
            memory_file = project_root / "memory" / "player_memory.json"

        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory = {}
        self.load()

    def load(self):
        """Load previously saved memories, if available."""
        if self.memory_file.exists():
            try:
                self.memory = json.loads(
                    self.memory_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self.memory = {}

    def save(self):
        """Save memories to a JSON file."""
        self.memory_file.write_text(
            json.dumps(self.memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def remember(self, key, value):
        """Remember a fact about the player."""
        key = key.strip().lower()
        value = value.strip()

        if not key or not value:
            return False

        self.memory[key] = value
        self.save()
        return True

    def recall(self, key):
        """Retrieve a remembered fact."""
        return self.memory.get(key.strip().lower())

    def forget(self, key):
        """Forget one fact."""
        key = key.strip().lower()

        if key in self.memory:
            del self.memory[key]
            self.save()
            return True

        return False

    def get_all(self):
        """Return all remembered facts."""
        return self.memory.copy()

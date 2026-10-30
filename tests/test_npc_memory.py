
from memory.npc_memory import NPCMemory


def test_remember_and_recall(tmp_path):
    memory = NPCMemory(tmp_path / "memory.json")

    assert memory.remember("player_name", "Arjun")
    assert memory.recall("player_name") == "Arjun"


def test_memory_survives_reload(tmp_path):
    memory_file = tmp_path / "memory.json"

    memory = NPCMemory(memory_file)
    memory.remember("weapon", "Silver sword")

    reloaded_memory = NPCMemory(memory_file)
    assert reloaded_memory.recall("weapon") == "Silver sword"


def test_forget_memory(tmp_path):
    memory = NPCMemory(tmp_path / "memory.json")
    memory.remember("weapon", "Silver sword")

    assert memory.forget("weapon")
    assert memory.recall("weapon") is None


def test_unknown_memory_returns_none(tmp_path):
    memory = NPCMemory(tmp_path / "memory.json")

    assert memory.recall("unknown_fact") is None


def test_empty_values_are_rejected(tmp_path):
    memory = NPCMemory(tmp_path / "memory.json")

    assert not memory.remember("", "Arjun")
    assert not memory.remember("player_name", "")

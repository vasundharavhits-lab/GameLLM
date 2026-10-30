
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "game" / "main.py"


def run_chat(*messages):
    input_text = "\n".join([*messages, "exit", ""])
    return subprocess.run(
        [sys.executable, str(MAIN)],
        input=input_text,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=15,
        check=True,
    ).stdout


def test_npc_remembers_and_recalls_weapon():
    output = run_chat(
        "I'm carrying a silver sword",
        "Do you remember my weapon?",
    )

    assert "I'll remember that you're carrying a silver sword." in output
    assert "Your weapon is a silver sword." in output
    assert output.count("Your weapon is a silver sword.") == 1


def test_npc_remembers_player_name():
    output = run_chat(
        "My name is Arjun",
        "What is my name?",
    )

    assert "Nice to meet you, Arjun!" in output
    assert "Your name is Arjun." in output


def test_game_knowledge_still_works():
    output = run_chat("Where is the blacksmith?")

    assert "beside the village gate in Eldoria" in output

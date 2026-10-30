
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge import gamellm_chat
from memory.npc_memory import NPCMemory


def main():
    memory = NPCMemory()

    print("=" * 45)
    print("       GameLLM - NPC Chatbot")
    print("=" * 45)
    print("Ask about quests, weapons, healing, or magic.")
    print("Tell Eldrin about yourself so he can remember.")
    print("Commands: 'memory', 'forget weapon', or 'exit'.\n")

    while True:
        question = input("Player: ").strip()

        if question.lower() in {"exit", "quit", "bye"}:
            print("Eldrin: Farewell, adventurer!")
            break

        if not question:
            continue

        normalized = question.lower().strip()
        handled = False

        # Show all saved memories.
        if normalized in {"memory", "what do you remember about me?",
                          "what do you know about me?"}:
            saved = memory.get_all()

            if saved:
                print("Eldrin: Here is what I remember about you:")
                for key, value in saved.items():
                    print(f"  {key}: {value}")
            else:
                print("Eldrin: I haven't remembered anything about you yet.")
            handled = True

        # Forget one saved memory.
        elif normalized.startswith("forget "):
            key = question[7:].strip().lower()
            if memory.forget(key):
                print(f"Eldrin: I've forgotten your {key}.")
            else:
                print(f"Eldrin: I don't have a memory saved under '{key}'.")
            handled = True

        # Save the player's name.
        elif normalized.startswith("my name is "):
            player_name = question[len("my name is "):].strip()
            if memory.remember("player_name", player_name):
                print(f"Eldrin: Nice to meet you, {player_name}!")
            handled = True

        elif normalized.startswith("i am ") or normalized.startswith("i'm "):
            prefix = "i am " if normalized.startswith("i am ") else "i'm "
            detail = question[len(prefix):].strip()

            if detail:
                if detail.lower().startswith("carrying "):
                    item = detail[len("carrying "):].strip()
                    if memory.remember("weapon", item):
                        print(f"Eldrin: I'll remember that you're carrying {item}.")
                else:
                    if memory.remember("player_name", detail):
                        print(f"Eldrin: Nice to meet you, {detail}!")
            else:
                print("Eldrin: Tell me a little more.")
            handled = True

        # Save the player's weapon.
        elif normalized.startswith("remember that my weapon is "):
            weapon = question[len("remember that my weapon is "):].strip()
            if memory.remember("weapon", weapon):
                print(f"Eldrin: I'll remember that your weapon is {weapon}.")
            handled = True

        elif normalized.startswith("my weapon is "):
            weapon = question[len("my weapon is "):].strip()
            if memory.remember("weapon", weapon):
                print(f"Eldrin: I'll remember that your weapon is {weapon}.")
            handled = True

        # Recall the player's name.
        elif normalized in {
            "what is my name", "what's my name", "what is my name?"
        } or "what is my name?" in normalized:
            name = memory.recall("player_name")
            if name:
                print(f"Eldrin: Your name is {name}.")
            else:
                print("Eldrin: You haven't told me your name yet.")
            handled = True

        # Recall the player's weapon using different phrasings.
        
        # Recall the player's weapon using different phrasings.
        elif any(phrase in normalized.rstrip("?!.,") for phrase in (
            "what weapon do i have",
            "what is my weapon",
            "what's my weapon",
            "do you remember my weapon",
            "do you remember what weapon",
            "what weapon am i carrying",
        )):
            weapon = memory.recall("weapon")
            if weapon:
                print(f"Eldrin: Your weapon is {weapon}.")
            else:
                print("Eldrin: You haven't told me your weapon yet.")
            handled = True


        # Preserve the existing game knowledge responses.
        if not handled:
            print(gamellm_chat(question))

        print()


if __name__ == "__main__":
    main()


import json
from pathlib import Path

# Locate the project's game_data folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = PROJECT_ROOT / "game_data" / "game_knowledge.json"


def load_knowledge():
    """Load NPC knowledge from the JSON file."""
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)




def get_game_fact(question):
    """Match common ways of asking about known game topics."""
    question = question.lower().strip()

    # Branna's location
    if (
        "branna" in question
        and any(word in question for word in ["where", "find", "location"])
    ) or any(
        phrase in question
        for phrase in [
            "where is the blacksmith",
            "where is branna",
            "blacksmith location",
            "find the blacksmith",
        ]
    ):
        return load_knowledge()["branna_location"]["answer"]

    # Weapons and sword shop
    if any(
        phrase in question
        for phrase in [
            "sword", "weapon", "blacksmith",
            "buy a weapon", "buy weapon", "buy a sword",
        ]
    ):
        return load_knowledge()["sword_shop"]["answer"]

    # Healing
    if any(
        word in question
        for word in ["health", "healing", "heal", "wounds", "potion"]
    ):
        return load_knowledge()["healing"]["answer"]

    # Quests
    if any(
        word in question
        for word in ["quest", "herbs", "reward", "mission"]
    ):
        return load_knowledge()["herb_quest"]["answer"]

    # Magic
    if any(
        phrase in question
        for phrase in [
            "magic", "spell", "teach me", "learn magic",
        ]
    ):
        return load_knowledge()["magic_training"]["answer"]

    return None


def gamellm_chat(question, npc="Eldrin"):
    """Answer from trusted game facts; avoid unsupported claims."""
    fact = get_game_fact(question)

    if fact:
        return f"{npc}: {fact}"

    return (
        f"{npc}: I don't know that yet, adventurer. "
        "Ask me about quests, healing, weapons, or magic."
    )


if __name__ == "__main__":
    questions = [
        "Can you give me a quest?",
        "Where can I buy a sword?",
        "Where can I find Branna?",
        "What is your favorite food?",
        "Where is the blacksmith?",
        "I need a healing potion.",
        "Can you give me a mission?",
        "Can you teach me a spell?",
    ]

    for question in questions:
        print(f"\nPlayer: {question}")
        print(gamellm_chat(question))

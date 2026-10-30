
from game.knowledge import gamellm_chat


def test_quest_question():
    answer = gamellm_chat("Can you give me a quest?")
    assert "missing herbs" in answer.lower()


def test_sword_question():
    answer = gamellm_chat("Where can I buy a sword?")
    assert "branna" in answer.lower()
    assert "swords" in answer.lower()


def test_blacksmith_location():
    answer = gamellm_chat("Where is the blacksmith?")
    assert "village gate" in answer.lower()


def test_healing_question():
    answer = gamellm_chat("I need a healing potion.")
    assert "health potion" in answer.lower()


def test_magic_question():
    answer = gamellm_chat("Can you teach me a spell?")
    assert "magic" in answer.lower()


def test_unknown_question_uses_safe_fallback():
    answer = gamellm_chat("What is your favorite food?")
    assert "don't know that yet" in answer.lower()

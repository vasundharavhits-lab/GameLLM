# GameLLM — Game Dialogue AI

GameLLM is an experimental AI project for creating interactive NPC (Non-Player Character) conversations in games. It explores training a small Transformer from scratch, fine-tuning a pretrained language model, and using a knowledge base to provide reliable in-game answers.

## Features

- **Custom Transformer:** A small character-level Transformer implemented in PyTorch.
- **LLM fine-tuning:** Experiments with fine-tuning DistilGPT2 using NPC dialogue data.
- **Knowledge-based responses:** Loads game facts from a JSON knowledge base.
- **NPC dialogue:** Supports questions about quests, healing, weapons, NPC locations, and magic.
- **Safe fallback:** Returns a predefined response when the knowledge base cannot answer a question.
- **Automated tests:** Uses pytest to verify supported questions and fallback behavior.

## Project Structure

```text
GameLLM/
├── configs/          # Model and training configuration
├── data/             # Training data
├── evaluation/       # Evaluation utilities
├── game/             # Game dialogue and knowledge logic
├── game_data/        # JSON knowledge base
├── memory/           # Memory-related components
├── models/           # Transformer implementation
├── tests/            # Automated tests
├── tokenizer/        # Character tokenizer
├── training/         # Training scripts
├── .gitignore
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.12
- pip
- PyTorch
- pytest

A GPU is not required to run the current knowledge-based chatbot or experiment with the tiny Transformer, although training speed depends on your hardware.

## Setup

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GameLLM
```

Create and activate a virtual environment on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Knowledge-Based Chatbot

```powershell
python .\game\knowledge.py
```

The chatbot loads game facts from `game_data/game_knowledge.json`.

Example questions:

- Can you give me a quest?
- Where can I buy a sword?
- Where can I find Branna?
- I need a healing potion.
- Can you teach me a spell?

Questions outside the current knowledge base receive a safe fallback response.

## Run Automated Tests

```powershell
python -m pytest -v
```

The current test suite has six tests covering quests, weapons, location, healing, magic, and unsupported questions.

## Model Experiments

### 1. Transformer from Scratch

The project includes a small character-level Transformer built with PyTorch. It is intended for learning the fundamentals of tokenization, attention, training, and text generation.

### 2. Fine-Tuning a Pretrained Language Model

The project also explores fine-tuning DistilGPT2 using NPC dialogue data in a Kaggle GPU environment. The fine-tuned model is experimental and may produce inaccurate or incoherent responses.

The current knowledge-based chatbot is a separate component; it does not automatically load the fine-tuned DistilGPT2 model.

## Current Limitations

- The game knowledge base contains only a small set of predefined facts.
- Keyword-based matching does not understand every paraphrase.
- The fine-tuned language model requires more suitable training data and evaluation.
- Full game-engine integration has not yet been implemented.

## Future Improvements

- Expand the game world, NPC personalities, quests, and dialogue dataset.
- Improve retrieval and intent matching.
- Add NPC memory and conversation history.
- Evaluate response quality and factual accuracy.
- Integrate with a game engine such as Unity or Godot.

## License

A license has not yet been selected. Add a license before distributing the project if you intend to make its reuse terms explicit.

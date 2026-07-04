# cli-agent

Personal project building a CLI agent in Python, following along with a tutorial.

**Tutorial:** [YouTube](https://www.youtube.com/watch?v=3GjE_YAs03s) — progress: **50:06**

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # Windows: copy .env.example .env
```
## Run

```bash
python main.py
```

## Project layout

```
client/
  llm_client.py   # Async OpenAI client (OpenRouter)
  response.py     # StreamEvent, TokenUsage, etc.
main.py           # Entry point
```

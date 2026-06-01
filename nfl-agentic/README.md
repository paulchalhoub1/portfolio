# NFL Agentic

This folder contains a Python agent that collects 2025-26 NFL regular-season game logs from ESPN and uses Gemini function calling to work through each week of the season.

## Files

- `01-NFL_Agent.py` - main data collection agent
- `requirements.txt` - Python package dependencies
- `env.example` - example environment variable file
- `.gitignore` - ignores local secrets, Python cache files, and generated JSON outputs

## Prerequisites

- Python 3.10 or newer
- A Gemini API key
- Git installed locally

## Setup

Clone the portfolio repository and move into this project folder:

```bash
git clone https://github.com/paulchalhoub1/portfolio.git
cd portfolio/nfl-agentic
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file from the example:

```bash
cp env.example .env
```

Then edit `.env` and replace the placeholder with your actual Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

Do not commit `.env`. It is intentionally ignored by Git.

## Run the Agent

From the `portfolio/nfl-agentic` folder, run:

```bash
python 01-NFL_Agent.py
```

The script will:

1. Ask the agent to process regular-season weeks 1-18.
2. Fetch each week's NFL schedule from ESPN.
3. Fetch game stats for each scheduled game.
4. Save collected game logs to `nfl_game_logs.json`.
5. Save flagged games to `nfl_flagged_games.json` if any games need review.

## Generated Outputs

The generated JSON files are ignored by Git by default:

- `nfl_game_logs.json`
- `nfl_flagged_games.json`

This keeps the repository clean while allowing you to rerun the agent locally whenever you need fresh output.

## Notes

The full run may take a while because the script pauses between model calls. Keep the terminal open until you see the completion message.

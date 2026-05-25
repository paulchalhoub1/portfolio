# Portfolio

This repository contains a Streamlit app for analyzing 2026 WNBA team data and generating matchup predictions.

## Files

- `01-WNBA2026.py` - the Streamlit app
- `requirements.txt` - Python dependencies needed to run the app

## Prerequisites

Before running the app, make sure you have:

- Python 3.10+ installed
- `pip` available in your terminal
- Internet access while the app runs, because `nba_api` pulls live league data

You can check your Python version with:

```bash
python --version
```

If your system uses `python3` instead of `python`, use `python3` in the commands below.

## 1. Clone the repository

```bash
git clone https://github.com/paulchalhoub1/portfolio.git
cd portfolio/wnba-ml
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

After activation, your terminal should show that you are inside `.venv`.

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the Streamlit app

```bash
streamlit run 01-WNBA2026.py
```

Streamlit will print a local URL in your terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Notes

- The app uses `nba_api`, so it may fail if the NBA stats service is temporarily unavailable or if your network blocks that traffic.
- The app trains several machine learning models the first time it runs, so startup may take a little longer than a basic Streamlit app.
- If `streamlit` is not found after installation, try:

```bash
python -m streamlit run 01-WNBA2026.py
```

## Deactivate the virtual environment

When you are done, you can leave the virtual environment with:

```bash
deactivate
```

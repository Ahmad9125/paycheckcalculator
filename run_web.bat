@echo off
cd /d %~dp0

REM --- Create venv if it doesn't exist ---
if not exist .venv (
  python -m venv .venv
)

REM --- Activate venv ---
call .venv\Scripts\activate.bat

REM --- Install dependencies ---
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

REM --- Launch the website UI ---
streamlit run app.py

pause

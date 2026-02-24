# Deploy for Free (Streamlit Community Cloud)

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**.
4. Select your repository, branch, and set the main file to `app.py`.
5. Click **Deploy**.

## Files already prepared

- `requirements.txt` for Python dependencies
- `packages.txt` for OS packages (`tesseract-ocr`)
- `runtime.txt` to pin Python 3.11

## iPhone usage

- Open the deployed URL in Safari on iPhone.
- Optional: Share -> Add to Home Screen to use it like an app icon.

## Notes

- The app uses Streamlit session state for settings in hosted mode.
- OCR depends on Tesseract being available in the deployment image.

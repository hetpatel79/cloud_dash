Open `index.html` in a browser while the backend is running on **:8001**.

For best results (CORS and `fetch` behavior), serve this folder instead of double-clicking the file:

```bash
cd ui
python -m http.server 3000
```

Then open **http://127.0.0.1:3000** in your browser.

Ensure `uvicorn api.main:app` is running from the project root on port **8001** so the UI can reach `/health`, `/conversations`, and related routes.

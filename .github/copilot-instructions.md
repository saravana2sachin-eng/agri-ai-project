## Quick orientation (what this repo is)

This is a small ML-powered Streamlit project for crop recommendation. There are two model tracks:
- A classical ML pipeline (scikit-learn RandomForest) used by the Streamlit UI.
- A deep-learning fusion model (PyTorch) under `model.py` / `train_dl_model.py` used for experimental work.

Key runtime artifacts
- `crop_model.pkl` — joblib dump of the trained RandomForest. Loaded by `app.py`.
- `dl_crop_model.pth` — (optional) saved PyTorch weights from `train_dl_model.py`.
- `Crop_recommendation.csv` — dataset used by `train_model.py`.

Key files
- `app.py` — Streamlit UI. Loads `crop_model.pkl` and calls model.predict on arrays shaped as [N, P, K, temperature, humidity, ph, rainfall].
- `train_model.py` — trains and saves the RandomForest (`crop_model.pkl`). Uses columns ['N','P','K','temperature','humidity','ph','rainfall'].
- `model.py` — PyTorch module(s): CNNModel, LSTMModel, SoilModel, CropModel (fusion). Used by `train_dl_model.py`.
- `train_dl_model.py` — example DL training loop (dummy data currently). Saves `dl_crop_model.pth`.
- `ndvi.py` — placeholder: `get_ndvi(lat, lon)` returns a fixed value (0.5).
- `weather.py` — `get_weather(lat, lon)` uses public Open-Meteo API (no API key required). Returns (temperature, rainfall).

Developer workflows (concrete commands)
- Run the Streamlit app locally:

  streamlit run app.py

- Rebuild the classical model (regenerate `crop_model.pkl`):

  python train_model.py

- Train the deep-learning fusion model (example script; uses dummy data):

  python train_dl_model.py

Smoke checks
- Verify the saved classical model loads:

  python -c "import joblib; joblib.load('crop_model.pkl'); print('loaded')"

Patterns & conventions specific to this repo
- Input order is important: features are [N,P,K,temperature,humidity,ph,rainfall]. `app.py` constructs a NumPy row array in that order before calling `model.predict`.
- Models and artifacts are referenced by relative path at repo root. Keep `crop_model.pkl`, `dl_crop_model.pth`, and `Crop_recommendation.csv` in the project root when running scripts.
- External dependency: `weather.py` calls Open-Meteo (no auth). Network access is required if the app uses `get_weather`.

Integration and coupling notes
- The Streamlit app is tightly coupled to the scikit-learn model saved as `crop_model.pkl` and to the CSV schema. If you change features or column names in the CSV, you must retrain with `train_model.py` and update `app.py` accordingly.
- The DL path (`model.py` + `train_dl_model.py`) is experimental/demonstration. `train_dl_model.py` imports `CropModel` and expects the model code to be runnable and correctly shaped.

Concrete gotchas discovered in the code (what to check before running)
- `model.py` contains obvious issues to fix before using the DL training script: the CNN `forward` method appears indented/structured incorrectly and contains debug prints; the fully-connected input size (`nn.Linear(800, 64)`) is a hard-coded value that depends on convolution output shapes — recalc or compute dynamically.
- `ndvi.py` is a stub (returns 0.5). If you rely on NDVI for predictions, replace the stub with real data loading or an API.
- `app.py` expects `crop_model.pkl` to be present; missing file causes the Streamlit app to crash on load. Use `train_model.py` to regenerate.

Examples (explicit snippets from this repo)
- App prediction (app.py):

  data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
  result = model.predict(data)

- Weather helper (weather.py):

  temperature, rainfall = get_weather(lat, lon)  # calls Open-Meteo

Extension notes for contributors
- If you modify feature columns, add a small adapter in `app.py` mapping UI fields to model input array. Keep a single place that defines the column order.
- When improving the DL model: add unit tests that import `model.py` and run a forward pass with dummy tensors to catch shape/indentation errors early.

Where to start when you open this repo
1. Run `python train_model.py` to produce `crop_model.pkl` (fast). 2. Start `streamlit run app.py` and exercise the UI. 3. If you work on DL, open `model.py`, fix the CNN `forward` indentation and verify shapes with a small forward pass.

Questions or missing pieces for the next iteration
- Should NDVI be wired into the prediction pipeline? If yes, provide the data source or method to compute NDVI.
- Are there preferred Python versions / CI steps to add? The repo doesn't contain CI or tests; tell me whether to add a minimal smoke test or CI job.

If anything in this file looks incomplete or you want me to expand any section (examples, tests, CI), tell me what to include and I'll iterate.

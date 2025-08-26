# Streamlit Movie Recommender (Local, No-Cloud)

A complete, local movie recommendation system you can run on AxOS (Arch-based).  
It uses the **MovieLens Latest Small** dataset and builds **content-based** and **lightweight personalized** recommendations with TF‑IDF.

## Features
- Search movies and get **similar titles** (content-based).
- Build **personalized recommendations** from your own ratings (no heavy matrix factorization).
- Clean Streamlit UI with fast local inference.
- Works fully offline after the first dataset download.

## Project Structure
```
movie-recs-streamlit/
├── app/
│   ├── streamlit_app.py
│   └── prepare_data.py
├── data/                # dataset and preprocessed cache will be created here
├── requirements.txt
└── README.md
```

## 1) Create a Python virtualenv (recommended)
```bash
# Install system deps (optional; python, pip should already be present)
sudo pacman -S --needed python python-pip

# From project folder:
python -m venv .venv
source .venv/bin/activate

# Upgrade pip (important on Arch)
pip install --upgrade pip wheel setuptools
```

## 2) Install Python dependencies
```bash
pip install -r requirements.txt
```

## 3) Prepare data (downloads MovieLens "ml-latest-small", then preprocesses TF-IDF)
```bash
python app/prepare_data.py
```
This will create `data/movielens/` with the raw CSVs and a cached `data/processed.pkl` (~tens of MB).

> If you're behind a proxy or the download fails, manually download from GroupLens (https://grouplens.org/datasets/movielens/) and place the CSVs inside `data/movielens/`, then rerun `prepare_data.py`.

## 4) Run the app
```bash
streamlit run app/streamlit_app.py
```
Streamlit will print a local URL (usually http://localhost:8501).

## Tips
- The **Personalized** tab works best after you add ~5–10 ratings (use the “Rate Movies” tool in the sidebar).
- Everything stays local—no data is sent anywhere.
- If you change the dataset, delete `data/processed.pkl` to rebuild the cache.

## Troubleshooting
- If you see a NumPy 1.x vs 2.x compiled extension error from other packages, try ensuring you’re using a clean virtualenv and reinstall:
  ```bash
  pip uninstall -y numpy
  pip install --no-binary :all: numpy  # if you need a source build (usually not necessary)
  pip install -r requirements.txt
  ```
- If Streamlit fails to open a browser, copy the URL printed in the terminal and paste into your browser.

# streamlit_app.py
import streamlit as st
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "processed.pkl"

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_cache():
    if not CACHE_PATH.exists():
        st.error("Cache not found. Run `python -m app.preprocess` first.")
        st.stop()
    with open(CACHE_PATH, "rb") as f:
        payload = pickle.load(f)

    # Reconstruct CSR matrix
    X = sparse.csr_matrix(
        (payload["X_data"], payload["X_indices"], payload["X_indptr"]),
        shape=payload["X_shape"]
    )

    # Handle optional keys
    tags = payload.get("tags", pd.DataFrame())
    id2row = payload.get("id2row", {})
    row2id = payload.get("row2id", {})

    return {
        "movies": payload["movies"],
        "ratings": payload["ratings"],
        "tags": tags,
        "X": X,
        "id2row": id2row,
        "row2id": row2id
    }

data = load_cache()
movies: pd.DataFrame = data["movies"]
ratings: pd.DataFrame = data["ratings"]
X = data["X"]
id2row: Dict[int,int] = data["id2row"]
row2id: Dict[int,int] = data["row2id"]

# Build helpers
title_to_ids = movies.groupby("title")["movieId"].apply(list).to_dict()
movie_lookup = movies.set_index("movieId")

# Session state for user ratings
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}  # movieId -> rating (0.5..5.0)

def id_from_title(title: str) -> int:
    ids = title_to_ids.get(title, [])
    return ids[0] if ids else None

def similar_by_id(movie_id: int, top_k: int = 20) -> pd.DataFrame:
    if movie_id not in id2row:
        return pd.DataFrame()
    row = id2row[movie_id]
    v = X.getrow(row)
    sims = cosine_similarity(v, X).ravel()
    idx = np.argpartition(-sims, range(top_k+1))[:top_k+1]
    idx_sorted = idx[np.argsort(-sims[idx])]
    mids = [row2id[i] for i in idx_sorted if row2id[i] != movie_id][:top_k]
    df = movie_lookup.loc[mids].reset_index()
    df["similarity"] = sims[[id2row[mid] for mid in df["movieId"]]]
    return df[["movieId", "title", "similarity"]]

def user_profile_vector(user_ratings: Dict[int, float]) -> np.ndarray:
    if not user_ratings:
        return None
    liked = [mid for mid, r in user_ratings.items() if r >= 4.0 and mid in id2row]
    disliked = [mid for mid, r in user_ratings.items() if r <= 2.0 and mid in id2row]
    if not liked and not disliked:
        return None
    vec = np.zeros((1, X.shape[1]), dtype=np.float32)
    if liked:
        rows = [id2row[m] for m in liked]
        vec += X[rows].mean(axis=0)
    if disliked:
        rows = [id2row[m] for m in disliked]
        vec -= X[rows].mean(axis=0)
    # Normalize
    denom = np.sqrt((vec.multiply(vec)).sum()) if sparse.issparse(vec) else np.linalg.norm(vec)
    if denom != 0:
        vec = vec / denom
    return vec

def recommend_for_user(user_ratings: Dict[int, float], top_k: int = 30) -> pd.DataFrame:
    profile = user_profile_vector(user_ratings)
    if profile is None:
        return pd.DataFrame()
    sims = cosine_similarity(profile, X).ravel()
    rated_ids = set(user_ratings.keys())
    # Remove already rated movies
    mask = np.ones_like(sims, dtype=bool)
    for mid in rated_ids:
        if mid in id2row:
            mask[id2row[mid]] = False
    sims_masked = np.where(mask, sims, -np.inf)
    idx = np.argpartition(-sims_masked, range(top_k))[:top_k]
    idx_sorted = idx[np.argsort(-sims_masked[idx])]
    mids = [row2id[i] for i in idx_sorted]
    df = movie_lookup.loc[mids].reset_index()
    df["score"] = sims[[id2row[mid] for mid in df["movieId"]]]
    return df[["movieId", "title", "score"]]

with st.sidebar:
    st.header("Rate Movies")
    q = st.text_input("Search title")
    if q:
        subset = movies[movies["title"].str.contains(q, case=False, na=False)].head(20)
        for _, row in subset.iterrows():
            cols = st.columns([3,1.5])
            with cols[0]:
                st.caption(f'{row["title"]}')
            with cols[1]:
                rating = st.number_input(
                    f'⭐ {row["movieId"]}',
                    min_value=0.0, max_value=5.0, step=0.5,
                    value=float(st.session_state.user_ratings.get(row["movieId"], 0.0)),
                    label_visibility="collapsed"
                )
                if rating > 0:
                    st.session_state.user_ratings[row["movieId"]] = rating
                elif row["movieId"] in st.session_state.user_ratings:
                    del st.session_state.user_ratings[row["movieId"]]
    st.divider()
    st.write("You have rated", len(st.session_state.user_ratings), "movies.")

st.title("🎬 Movie Recommender")
tab1, tab2, tab3 = st.tabs(["🔎 Search & Similar", "🧑‍🤝‍🧑 Personalized", "📊 Dataset Overview"])

with tab1:
    st.subheader("Find similar movies by title")
    title = st.selectbox("Pick a movie", options=sorted(title_to_ids.keys()))
    if title:
        mid = id_from_title(title)
        res = similar_by_id(mid, top_k=20)
        st.write(f"Top similar to **{title}**")
        st.dataframe(res)

with tab2:
    st.subheader("Personalized recommendations")
    st.write("Add ratings from the sidebar, then click **Recommend**.")
    k = st.slider("How many recommendations?", 5, 50, 20)
    if st.button("Recommend", type="primary"):
        recs = recommend_for_user(st.session_state.user_ratings, top_k=k)
        if recs.empty:
            st.info("Please rate a few movies first (ideally 5–10 with ⭐≥4.0).")
        else:
            st.dataframe(recs)

with tab3:
    st.subheader("Dataset quick stats")
    st.write("Movies:", len(movies))
    st.write("Unique genres:", sorted({g for gs in movies.get("genres", pd.Series()).dropna().tolist() for g in gs.split('|') if g!='(no genres listed)'}))
    st.dataframe(movies.head(20))

import pandas as pd
from scipy import sparse
import pickle
import os

def build_movielens_100k():
    # Paths
    ratings_path = "data/movielens/ml-100k/u.data"
    movies_path = "data/movielens/ml-100k/u.item"

    # Load ratings
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"]
    )

    # Load movies
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1, 2],  # movieId, title, genres
        names=["movieId", "title", "genres"]
    )

    # Build id2row mapping for movies
    movie_ids = ratings["movieId"].unique()
    id2row = {mid: i for i, mid in enumerate(movie_ids)}
    row2id = {i: mid for mid, i in id2row.items()}

    # Build sparse matrix (movies × users)
    rows = [id2row[m] for m in ratings["movieId"]]
    cols = [u - 1 for u in ratings["userId"]]  # userId starts at 1
    data = ratings["rating"].astype(float)

    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(movie_ids), ratings["userId"].nunique()))

    # Package payload
    payload = {
        "X_data": X.data,
        "X_indices": X.indices,
        "X_indptr": X.indptr,
        "X_shape": X.shape,
        "movies": movies,
        "ratings": ratings,
        "id2row": id2row,
        "row2id": row2id,
        "tags": pd.DataFrame()  # ML-100k has no tags
    }

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/processed.pkl", "wb") as f:
        pickle.dump(payload, f)

    print("✅ Preprocessing complete! Saved to data/processed.pkl")


if __name__ == "__main__":
    build_movielens_100k()

import pandas as pd
import os

DATA_DIR = "data/movielens/ml-100k"

def load_movielens_100k():
    # Ratings file (userId, movieId, rating, timestamp)
    ratings_path = os.path.join(DATA_DIR, "u.data")
    ratings = pd.read_csv(
        ratings_path, 
        sep="\t", 
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python"
    )

    # Movies file (movieId, title, release_date, video_release, IMDb URL, genres...)
    movies_path = os.path.join(DATA_DIR, "u.item")
    movies = pd.read_csv(
        movies_path,
        sep="|",
        encoding="latin-1",
        names=[
            "movieId", "title", "release_date", "video_release_date", "IMDb_URL"
        ] + [f"genre_{i}" for i in range(19)],  # 19 genre flags
        engine="python"
    )

    # Keep only movieId + title
    movies = movies[["movieId", "title"]]

    return ratings, movies

if __name__ == "__main__":
    ratings, movies = load_movielens_100k()
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print(ratings.head())
    print(movies.head())

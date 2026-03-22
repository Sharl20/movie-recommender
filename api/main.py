
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(title="Movie Recommender API")

# Load models
mf          = joblib.load("models/mf_model.pkl")
content_sim = joblib.load("models/content_sim.pkl")
movies      = joblib.load("models/movies.pkl")
ratings     = joblib.load("models/ratings.pkl")

movie_idx_map = {mid: idx for idx, mid in enumerate(movies["movie_id"])}

class CFRequest(BaseModel):
    user_id: int
    n: int = 10

class HybridRequest(BaseModel):
    user_id: int
    n: int = 10
    cf_weight: float = 0.7
    cb_weight: float = 0.3

@app.get("/")
def root():
    return {"message": "Movie Recommender API ✅"}

@app.post("/recommend/cf")
def recommend_cf(req: CFRequest):
    watched = ratings[ratings["user_id"] == req.user_id]["movie_id"].tolist()
    unseen  = [m for m in ratings["movie_id"].unique() if m not in watched]
    preds   = [(m, mf.predict(req.user_id, m)) for m in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)
    top     = preds[:req.n]
    result  = movies[movies["movie_id"].isin([m for m, _ in top])][["movie_id", "title"]]
    return {"recommendations": result.to_dict(orient="records")}

@app.post("/recommend/hybrid")
def recommend_hybrid(req: HybridRequest):
    watched = ratings[ratings["user_id"] == req.user_id]["movie_id"].tolist()
    unseen  = [m for m in ratings["movie_id"].unique() if m not in watched]

    cf_scores = np.array([mf.predict(req.user_id, m) for m in unseen])

    top_watched = (ratings[ratings["user_id"] == req.user_id]
                   .sort_values("rating", ascending=False)
                   .head(5)["movie_id"].tolist())

    cb_scores = []
    for m in unseen:
        if m not in movie_idx_map:
            cb_scores.append(0)
            continue
        sims = [content_sim[movie_idx_map[w]][movie_idx_map[m]]
                for w in top_watched if w in movie_idx_map]
        cb_scores.append(np.mean(sims) if sims else 0)
    cb_scores = np.array(cb_scores)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    hybrid    = req.cf_weight * norm(cf_scores) + req.cb_weight * norm(cb_scores)
    top_idx   = np.argsort(hybrid)[::-1][:req.n]
    top_ids   = [unseen[i] for i in top_idx]
    top_scores = [float(hybrid[i]) for i in top_idx]

    result = movies[movies["movie_id"].isin(top_ids)][["movie_id", "title"]].copy()
    result["score"] = result["movie_id"].map(dict(zip(top_ids, top_scores)))
    result = result.sort_values("score", ascending=False)
    return {"recommendations": result.to_dict(orient="records")}

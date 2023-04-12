import pandas as pd
from data.db import DATA
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.DataFrame()
# data["title"]=DATA["title"].astype(str).apply(lambda x: x.lower())
# data["description"]=DATA["description"].astype(str).apply(lambda x: x.lower())
data["text"] = DATA["title"].astype(str) + DATA["description"].astype(str)
data["text"] = data["text"].apply(lambda x: x.lower())
data["text"] = data["text"].str.replace("[^\w\s]", "")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["text"])
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(tfidf_matrix)
app = FastAPI()


@app.get("/")
async def get_search_form():
    return FileResponse("index.html")


@app.get("/search")
def search(q: str):
    

    query_tfidf = vectorizer.transform([q])
    try:
        distances, top_indices = knn.kneighbors(query_tfidf)
        results = []
        for i, idx in enumerate(top_indices[0]):
            title = DATA.iloc[idx]["title"]
            description = DATA.iloc[idx]["description"]
            distance = distances[0][i]
            results.append((title, description, distance))
        # Sort by whether query appears in title and then by cosine distance
        results.sort(key=lambda x: (q in x[0], x[2]))
        top_texts = [r[0] for r in results]
        top_description = [r[1] for r in results]
        # _, top_indices = knn.kneighbors(query_tfidf)
        # distances = knn.kneighbors(query_tfidf, return_distance=True)[0][0]
        # results = list(zip(top_indices[0], distances))
        # sorted_results = sorted(results, key=lambda x: x[1])
        # top_indices_sorted = [x[0] for x in sorted_results]
        # top_texts = DATA.iloc[top_indices_sorted]["title"].tolist()
        # top_description = DATA.iloc[top_indices_sorted]["description"].tolist()
        #top_texts = DATA.iloc[top_indices[0]]["title"].tolist()
        #top_description = DATA.iloc[top_indices[0]]["description"].tolist()
    except ValueError:
        top_texts, top_description = [], []
    return {
        "results": [
            {"title": x, "description": y} for x, y in zip(top_texts, top_description)
        ]
    }

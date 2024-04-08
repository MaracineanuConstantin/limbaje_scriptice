from fastapi import FastAPI
from typing import Union
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from pyod.models.knn import KNN
from sklearn.model_selection import train_test_split
import os

app = FastAPI()

neigh = None
clf = None


@app.on_event("startup")
def load_train_model():
    df = pd.read_csv("iris_ok.csv")
    global neigh
    neigh = KNeighborsClassifier(n_neighbors=len(np.unique(df['y'])))
    neigh.fit(df[df.columns[:4]].values.tolist(), df['y'])
    print(f'MOdel finished the training')

    global clf
    clf = KNN()
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:4]].values.tolist(), df['y'], test_size=0.33, shuffle=True, random_state=42)
    clf.fit(X_train, y_train)
    print("Incercarea de anomalie")

@app.get("/predict")
def predict(p1: float, p2: float, p3: float, p4: float):
    pred = neigh.predict([[p1, p2, p3, p4]])
    return "{}".format(pred[0])


@app.get("/anomalie")
def detect_anomaly(p1: float, p2: float, p3: float, p4: float):
    pred = neigh.predict([[p1, p2, p3, p4]])
    return "{}".format(pred[0])

@app.get("/")
def read_root():
    return {"Hello": "World"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ['HOST'], port=os.environ['PORT'])
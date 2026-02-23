import pandas as pd
from sklearn.linear_model import LinearRegression

print("Starting training...")

# Dummy dataset
data = {
    "hours": [1, 2, 3, 4, 5],
    "score": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

X = df[["hours"]]
y = df["score"]

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")
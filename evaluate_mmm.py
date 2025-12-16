import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "compressed_data.csv"
CUTOFF_DATE = datetime(2023, 12, 31)
MODEL_PATH = Path("linear_mmm_model.pkl")
FEATURES_PATH = Path("mmm_model_features.json")

FEATURES = [
    "GOOGLE_PAID_SEARCH_SPEND",
    "GOOGLE_SHOPPING_SPEND",
    "GOOGLE_PMAX_SPEND",
    "META_FACEBOOK_SPEND",
    "META_INSTAGRAM_SPEND",
    "EMAIL_CLICKS",
    "ORGANIC_SEARCH_CLICKS",
    "DIRECT_CLICKS",
    "BRANDED_SEARCH_CLICKS",
    "year",
    "month",
    "day_of_week",
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE_DAY"] = pd.to_datetime(df["DATE_DAY"], format="%d-%m-%Y")
    df["revenue"] = df["ALL_PURCHASES_ORIGINAL_PRICE"] - df["ALL_PURCHASES_GROSS_DISCOUNT"]
    df["total_spend"] = df[
        [
            "GOOGLE_PAID_SEARCH_SPEND",
            "GOOGLE_SHOPPING_SPEND",
            "GOOGLE_PMAX_SPEND",
            "META_FACEBOOK_SPEND",
            "META_INSTAGRAM_SPEND",
        ]
    ].fillna(0).sum(axis=1)
    df["roi"] = df["revenue"] / (df["total_spend"] + 1)
    df["year"] = df["DATE_DAY"].dt.year
    df["month"] = df["DATE_DAY"].dt.month
    df["day_of_week"] = df["DATE_DAY"].dt.dayofweek
    return df


def train_eval(df: pd.DataFrame):
    feature_cols = FEATURES[:-3]  # spend/clicks
    df[feature_cols] = df[feature_cols].fillna(0)
    df = df.dropna(subset=["revenue"])

    df_train = df[df["DATE_DAY"] <= CUTOFF_DATE]
    df_test = df[df["DATE_DAY"] > CUTOFF_DATE]

    X_train = df_train[FEATURES]
    y_train = df_train["revenue"]
    X_test = df_test[FEATURES]
    y_test = df_test["revenue"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    return model, r2, rmse, len(df_train), len(df_test)


def main():
    df = load_data(DATA_PATH)
    model, r2, rmse, n_train, n_test = train_eval(df)
    # persist artifacts for API consumption
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURES))

    print(f"Train rows: {n_train}, Test rows: {n_test}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved features -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import timedelta
from collections import defaultdict
import hashlib
import os
import re
from heapq import nlargest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

path = 'data_anonymized_with_user_ids.csv'

#loading
df = pd.read_csv(path)

#base
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    x = x.replace(" ", "")    
    x = re.sub(r"[^\d.]", "", x)  
    try:
        return float(x)
    except:
        return np.nan

df["Price"] = df["Price"].apply(clean_price)
df = df.dropna(subset=["Price"])

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Price"])
df = df.sort_values(["user_id", "Date"])
df["item_id"] = df["Item"].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())

#parsing categories
def parse_item(s):
    s = str(s).lower()
    cat = "other"
    mat = "other"
    if "кольц" in s or "ring" in s: cat = "ring"
    elif "серьг" in s or "ear" in s: cat = "earrings"
    elif "цеп" in s or "chain" in s: cat = "chain"
    elif "брасл" in s or "brace" in s: cat = "bracelet"
    elif "подвес" in s or "pendant" in s: cat = "pendant"
    if "585" in s or "зол" in s or "gold" in s: mat = "au585"
    elif "925" in s or "сереб" in s or "silver" in s: mat = "ag925"
    return pd.Series({"category": cat, "material": mat})

df[["category", "material"]] = df["Item"].apply(parse_item)

#calendar features
cut_date = pd.Timestamp("2025-10-10")
forecast_start = cut_date + pd.Timedelta(days=1)
forecast_end   = pd.Timestamp("2025-10-23")
print(f"Cut date: {cut_date.date()} | Прогнозируем покупки с {forecast_start.date()} по {forecast_end.date()}")
# df["day_of_week"] = df["Date"].dt.weekday # 0=Mon, 6=Sun
# df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
# df["month"] = df["Date"].dt.month
# df["quarter"] = df["Date"].dt.quarter

#%%
events = df.sort_values("Date")[["user_id","item_id","Date"]]
half_life = pd.Timedelta(days=90)
now = events["Date"].max()
events["w"] = np.exp(-(now - events["Date"]).dt.days / half_life.days)
pop = events.groupby("item_id")["w"].sum()

from collections import defaultdict
co = defaultdict(float)
user_groups = events.groupby("user_id")["item_id"].agg(list)
for items in user_groups:
    uniq = list(dict.fromkeys(items))  
    for i, a in enumerate(uniq):
        for b in uniq[i+1:]:
            co[(a,b)] += 1.0
            co[(b,a)] += 1.0

from heapq import nlargest
neighbors = defaultdict(list)
for (a,b), c in co.items():
    neighbors[a].append((b, c))
for a in neighbors:
    neighbors[a] = nlargest(100, neighbors[a], key=lambda x: x[1])

def recommend_next(user_id, k=10):
    last_items = user_groups.get(user_id, [])[-5:]
    cand = defaultdict(float)
    for li in last_items:
        for nb, score in neighbors.get(li, []):
            cand[nb] += score
    for it, p in pop.nlargest(100).items():
        cand[it] += 0.01 * p
    bought = set(user_groups.get(user_id, []))
    recs = [it for it,_ in sorted(cand.items(), key=lambda x: -x[1]) if it not in bought][:k]
    return recs




#%%
#features
hist = df[df["Date"] <= cut_date].copy()

def user_features(df_hist, as_of):
    h = df_hist[df_hist["Date"] < as_of]
    if h.empty:
        return pd.DataFrame(columns=[
            "user_id","days_since_last","orders_30d","orders_90d",
            "avg_price_90d","total_spent_90d","unique_categories_180d","tradein_ratio_180d"
        ])
    last_date = h.groupby("user_id")["Date"].max().rename("last_date")
    u = last_date.to_frame()
    u["days_since_last"] = (as_of - u["last_date"]).dt.days
    for days in [30,90]:
        start = as_of - pd.Timedelta(days=days)
        cnt = h[h["Date"] >= start].groupby("user_id")["Date"].count()
        u[f"orders_{days}d"] = cnt
    h90 = h[h["Date"] >= as_of - pd.Timedelta(days=90)]
    price_stats = h90.groupby("user_id")["Price"].agg(["mean","sum"]).rename(
        columns={"mean":"avg_price_90d","sum":"total_spent_90d"}
    )
    u = u.join(price_stats, how="left")
    h180 = h[h["Date"] >= as_of - pd.Timedelta(days=180)]
    u["unique_categories_180d"] = h180.groupby("user_id")["category"].nunique()
    if "Format" in h180.columns:
        trade_ratio = (h180["Format"].astype(str).str.lower().str.contains("trade")).groupby(h180["user_id"]).mean()
        u["tradein_ratio_180d"] = trade_ratio
    else:
        u["tradein_ratio_180d"] = 0.0
    u = u.fillna(0.0).reset_index()
    return u

X_users = user_features(hist, cut_date)

#%%
buyers = set(df[(df["Date"] > cut_date) & (df["Date"] <= forecast_end)]["user_id"])
X_users["label"] = X_users["user_id"].map(lambda u: 1 if u in buyers else 0)
print(f"Реальных покупателей с 11 по 23 октября: {len(buyers)}")


#%%
#training

HORIZONS = [7, 14, 30, 60, 90]
FEATURES = ["days_since_last","orders_30d","orders_90d",
             "avg_price_90d","total_spent_90d","unique_categories_180d","tradein_ratio_180d"]

def build_labels(df_full, as_of, horizon):
    """Кто купил в течение horizon дней после as_of."""
    end = as_of + pd.Timedelta(days=horizon)
    buyers = set(df_full[(df_full["Date"] > as_of) & (df_full["Date"] <= end)]["user_id"])
    return buyers

def assemble_dataset(df_full, cuts, horizons):
    rows = []
    for t_cut in cuts:
        feats = user_features(df_full, t_cut)
        for N in horizons:
            buyers = build_labels(df_full, t_cut, N)
            featsN = feats.copy()
            featsN["cut_time"] = t_cut
            featsN["horizon"] = N
            featsN["label"] = featsN["user_id"].map(lambda u: 1 if u in buyers else 0)
            rows.append(featsN)
    return pd.concat(rows, ignore_index=True)

cut_points = pd.date_range(df["Date"].min() + pd.Timedelta(days=180),
                           df["Date"].max() - pd.Timedelta(days=30), freq="30D")

dataset = assemble_dataset(df, cut_points, HORIZONS)

X_future = user_features(df, cut_date)

os.makedirs("buyers_predictions", exist_ok=True)



for N in HORIZONS:
    sub = dataset[dataset["horizon"]==N]
    X_train, y_train = sub[FEATURES].values, sub["label"].values

    if len(np.unique(y_train)) < 2:
        print(f"⚠️ Горизонт {N}: пропущен (мало данных)")
        continue

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)

    X_future["prob"] = pipe.predict_proba(X_future[FEATURES])[:, 1]

    X_future["predicted_next_date"] = cut_date + pd.to_timedelta((N * X_future["prob"]).round(0), unit="D")

    likely = X_future[X_future["prob"] >= 0.8].sort_values("prob", ascending=False).copy()
    likely["probability_percent"] = (likely["prob"] * 100).round(1)

    out_path = f"buyers_predictions/buyers_h{N}.csv"
    likely[["user_id", "predicted_next_date", "probability_percent"]].to_csv(out_path, index=False)

    print(f"\nbuyers_h{N}.csv сохранён ({len(likely)} клиентов):")
    print(likely.head(10)[["user_id", "predicted_next_date", "probability_percent"]])

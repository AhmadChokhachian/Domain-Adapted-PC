from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ---------- paths ----------
ROOT = Path.cwd().resolve()
if not (ROOT / "data").exists():
    ROOT = ROOT.parent

DATA_DIR     = ROOT / "data"
RESULTS_DIR  = ROOT / "results" / "intermediate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SETUP_SUFFIX = "_dfp"
TERRAIN_PATH = DATA_DIR / "weightedTerrainData.csv"

# --- year argument: python code/Table2_P_XGBoost_dfp.py 2017
args        = sys.argv[1:]
TEST_YEARS  = [int(args[0])] if args else [2017, 2018]
year_suffix = f"_{args[0]}" if args else ""

OUT_LONG = RESULTS_DIR / f"Table2_P_XGBoost_detail{SETUP_SUFFIX}{year_suffix}.csv"
OUT_SUMM = RESULTS_DIR / f"Table2_P_XGBoost_summary{SETUP_SUFFIX}{year_suffix}.csv"

# ---------- IDs ----------
ALL_IDS   = list(range(1, 67))
TEST_IDS  = list(range(38, 45))
TRAIN_IDS = list(sorted(set(ALL_IDS) - set(TEST_IDS)))

# ---------- terrain ----------
terrain_df   = pd.read_csv(TERRAIN_PATH)
terrain_cols = terrain_df.columns[1:4]
terrain_mat  = terrain_df.loc[:65, terrain_cols].values

FEATURES = ["wind_speed", "temperature", "turbulence_intensity", "std_wind_direction",
            "wind_direction_sin", "wind_direction_cos"]


# ---------- utils ----------
def rmse(yhat, y):
    yhat = np.asarray(yhat, float)
    y    = np.asarray(y,    float)
    m    = np.isfinite(yhat) & np.isfinite(y)
    return float(np.sqrt(np.mean((yhat[m] - y[m])**2))) if np.any(m) else np.nan

def load_turbine_csv(tid, year):
    f = DATA_DIR / f"Turbine{tid}_{year}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    needed = ["wind_speed", "temperature", "turbulence_intensity",
              "std_wind_direction", "wind_direction", "power"]
    df = df[needed].copy()
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None
    rad = np.deg2rad(df["wind_direction"].values)
    df["wind_direction_sin"] = np.sin(rad)
    df["wind_direction_cos"] = np.cos(rad)
    df = df.drop(columns=["wind_direction"])
    return df

def fit_lgbm(X_train, y_train, X_test, seed):
    model = LGBMRegressor(
        objective="regression", n_estimators=200, learning_rate=0.1,
        max_depth=8, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)

def append_row(row, path):
    row_df = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    row_df.to_csv(path, mode="a", header=write_header, index=False)


# ---------- cache ----------
print("[INFO] Loading data cache ...", flush=True)
data_cache = {}
for tid in ALL_IDS:
    for yr in (2017, 2018):
        data_cache[(tid, yr)] = load_turbine_csv(tid, yr)

# ---------- build TRAIN ONLY pooled 2017 ----------
X_list, y_list, tid_list = [], [], []
for tid in TRAIN_IDS:
    df = data_cache[(tid, 2017)]
    if df is None:
        continue
    X_list.append(df[FEATURES].values)
    y_list.append(df["power"].values)
    tid_list.append(np.full(len(df), tid))

Xtrain    = np.vstack(X_list)
ytrain    = np.concatenate(y_list)

# ---------- resume ----------
done_keys = set()
if OUT_LONG.exists() and OUT_LONG.stat().st_size > 0:
    try:
        prev = pd.read_csv(OUT_LONG)
        if {"target", "year"}.issubset(prev.columns):
            done_keys = set(f"{int(r.target)}|{int(r.year)}" for _, r in prev.iterrows())
            print(f"[INFO] Resuming — {len(done_keys)} rows already done.", flush=True)
    except Exception:
        pass

# ---------- main loop ----------
for year in TEST_YEARS:
    for i in TEST_IDS:
        pair_key = f"{i}|{year}"
        if pair_key in done_keys:
            print(f"  [SKIP] Turbine {i} Year {year}", flush=True)
            continue

        df_test = data_cache[(i, year)]
        if df_test is None:
            continue

        X_test = df_test[FEATURES].values
        y_test = df_test["power"].values

        print(f"DFP: Turbine {i} Year {year} — P_XGBoost(x)", flush=True)
        t0      = time.time()
        pred_x  = fit_lgbm(Xtrain, ytrain, X_test, seed=i)
        runtime = time.time() - t0

        append_row({"method": "P_XGBoost(x)", "target": i, "year": year,
                    "rmse": rmse(pred_x, y_test), "runtime_sec": runtime}, OUT_LONG)
        done_keys.add(pair_key)
        print(f"  -> RMSE: {rmse(pred_x, y_test):.4f}", flush=True)

# ---------- summary ----------
if OUT_LONG.exists() and OUT_LONG.stat().st_size > 0:
    detail_df  = pd.read_csv(OUT_LONG)
    summary_df = detail_df.groupby(["method", "year"], as_index=False).agg(
        avg_rmse=("rmse", "mean"),
        total_runtime_sec=("runtime_sec", "sum"),
    )
    summary_df.to_csv(OUT_SUMM, index=False)
    print("[DONE] Summary saved.", flush=True)

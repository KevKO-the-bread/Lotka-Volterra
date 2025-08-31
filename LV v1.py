#!/usr/bin/env python3
"""
Oil (CL=F) vs Oil Producers (XOP) — LV on Cycles
- Weekly data for 20 years
- Log + HP/Band-pass cycle extraction
- Scale to (0,1]
- Fit Lotka–Volterra (LV) params (α,β,γ,δ ≥0) on train
- One-step ahead forecasts on test
- Detect turning points (peaks/troughs)
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares

# optional HP filter
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

# -------------------- Data --------------------

def _download_weekly_close(ticker, name, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    if df.empty or "Close" not in df.columns:
        return None
    s = df["Close"].dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s.resample("W-FRI").last().dropna()
    s.name = name
    return s

def get_oil_and_producers(start, end):
    oil = _download_weekly_close("CL=F", "WTI Crude", start, end)
    prod = _download_weekly_close("XOP", "Oil Producers ETF", start, end)
    if oil is None or prod is None:
        raise RuntimeError("Could not fetch CL=F or XOP")
    return oil, prod

# -------------------- Cycles --------------------

def hp_cycle(series, lam=1_000_000):
    if not HAVE_SM:
        return None
    cycle, trend = sm.tsa.filters.hpfilter(np.log(series.clip(lower=1e-6)), lamb=lam)
    return pd.Series(np.asarray(cycle), index=series.index, name=f"{series.name}_cycle")

def bandpass_cycle(series, low_weeks=8, high_weeks=104):
    s = np.log(series.clip(lower=1e-6)).values
    if len(s) < 20:
        return pd.Series(np.zeros_like(s), index=series.index)
    fs = 1.0
    nyq = 0.5 * fs
    low = 1.0 / high_weeks
    high = 1.0 / low_weeks
    low = max(low, 1e-6)
    high = min(high, nyq - 1e-6)
    if not (low < high):
        low, high = 1e-4, 0.2
    b, a = butter(N=3, Wn=[low/nyq, high/nyq], btype="band")
    cyc = filtfilt(b, a, s)
    return pd.Series(cyc, index=series.index, name=f"{series.name}_cycle")

def extract_cycle(series):
    c = hp_cycle(series)
    if c is None:
        c = bandpass_cycle(series)
    c = c - np.nanmin(c)
    denom = np.nanmax(c)
    if denom <= 0:
        c = pd.Series(np.ones_like(c), index=c.index)
    else:
        c = c / denom
        c = c.clip(lower=1e-6)
    c.name = f"{series.name}_cycle01"
    return c

# -------------------- LV Estimation --------------------

def fit_lv_nls(X, Y):
    X = X.values; Y = Y.values
    X_next, Y_next = X[1:], Y[1:]
    X_t, Y_t = X[:-1], Y[:-1]
    def residuals(theta):
        a,b,g,d = np.maximum(theta,0.0)
        X_pred = X_t + (a*X_t - b*X_t*Y_t)
        Y_pred = Y_t + (g*X_t*Y_t - d*Y_t)
        return np.concatenate([X_pred - X_next, Y_pred - Y_next])
    theta0 = np.array([0.10,0.10,0.10,0.10])
    bounds = (np.zeros(4), np.ones(4)*10.0)
    sol = least_squares(residuals, theta0, bounds=bounds, max_nfev=5000)
    a,b,g,d = np.maximum(sol.x,0.0)
    return a,b,g,d, sol.cost, sol.success

def one_step_forecast(X,Y,params):
    a,b,g,d = params
    X_t, Y_t = X.values[:-1], Y.values[:-1]
    X_hat = X_t + (a*X_t - b*X_t*Y_t)
    Y_hat = Y_t + (g*X_t*Y_t - d*Y_t)
    idx = X.index[1:]
    return pd.Series(X_hat,index=idx,name="X_hat"), pd.Series(Y_hat,index=idx,name="Y_hat")

def detect_turns(s):
    ds = s.diff(); sign = np.sign(ds)
    flips = (sign * sign.shift(1) < 0).fillna(False)
    return list(s.index[flips])

# -------------------- Plot --------------------

def plot_cycles(train_idx, Xc, Yc, X_hat, Y_hat):
    fig, axes = plt.subplots(2,2,figsize=(16,10))
    fig.suptitle("Oil (CL=F) vs Producers (XOP) — LV on Cycles",fontsize=16,fontweight="bold")

    ax1=axes[0,0]
    ax1.plot(Xc.index,Xc,label="Oil cycle",color="black")
    ax1.plot(Yc.index,Yc,label="Producers cycle",color="darkred")
    ax1.plot(X_hat.index,X_hat,"--",label="Oil one-step",color="black",alpha=0.7)
    ax1.plot(Y_hat.index,Y_hat,"--",label="Producers one-step",color="darkred",alpha=0.7)
    ax1.axvline(train_idx[-1],color="gray",ls="--",alpha=0.5,label="train/test split")
    ax1.set_title("Cycles + one-step LV forecasts")
    ax1.legend(); ax1.grid(True,alpha=0.3); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2=axes[0,1]
    ax2.scatter(Xc,Yc,s=10,alpha=0.6)
    ax2.set_xlabel("Oil cycle"); ax2.set_ylabel("Producers cycle")
    ax2.set_title("Phase diagram"); ax2.grid(True,alpha=0.3)

    ax3=axes[1,0]
    errX=(Xc-X_hat).dropna().loc[X_hat.index]
    errY=(Yc-Y_hat).dropna().loc[Y_hat.index]
    ax3.plot(errX.index,errX,label="Oil error")
    ax3.plot(errY.index,errY,label="Producers error")
    ax3.axhline(0,color="black",alpha=0.3)
    ax3.set_title("One-step errors"); ax3.legend(); ax3.grid(True,alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax4=axes[1,1]
    roll=Xc.rolling(26).corr(Yc)
    ax4.plot(roll.index,roll,color="purple")
    ax4.axhline(0,color="black",alpha=0.3)
    ax4.set_ylim(-1,1); ax4.set_title("Rolling correlation (26w)")
    ax4.grid(True,alpha=0.3); ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout(); plt.show()

# -------------------- Main --------------------

def main(years_back=20, train_frac=0.8):
    end = datetime.now()
    start = end - timedelta(days=365*years_back)

    # Force Series with correct names
    oil, prod = get_oil_and_producers(start, end)
    oil = pd.Series(oil.squeeze(), name="X_oil")
    prod = pd.Series(prod.squeeze(), name="Y_prod")

    df = pd.concat([oil, prod], axis=1, join="inner").dropna()

    # Debug
    print("Columns in df:", df.columns.tolist())
    print(df.head())

    # --- Cycle extraction ---
    Xc = extract_cycle(df["X_oil"])
    Yc = extract_cycle(df["Y_prod"])
    cyc = pd.concat([Xc, Yc], axis=1, join="inner").dropna()
    Xc, Yc = cyc.iloc[:, 0], cyc.iloc[:, 1]

    # --- Train/test split ---
    split = int(len(cyc) * train_frac)
    train_idx = cyc.index[:split]
    test_idx = cyc.index[split:]

    # --- Parameter estimation ---
    a, b, g, d, cost, ok = fit_lv_nls(Xc.loc[train_idx], Yc.loc[train_idx])
    print("=== LV params ===")
    print(f"alpha={a:.4f}, beta={b:.4f}, gamma={g:.4f}, delta={d:.4f} (success={ok})")

    # --- Forecasting ---
    X_hat, Y_hat = one_step_forecast(Xc, Yc, (a, b, g, d))

    # --- Turning points ---
    turns_X_act = detect_turns(Xc.loc[test_idx])
    turns_X_hat = detect_turns(X_hat.loc[test_idx])
    print(f"Test turns (Oil) — model {len(turns_X_hat)} vs actual {len(turns_X_act)}")

    # --- Plots ---
    plot_cycles(train_idx, Xc, Yc, X_hat, Y_hat)

    # --- Prediction Quality ---
    corr = X_hat.loc[test_idx].corr(Xc.loc[test_idx])
    rmse = np.sqrt(((X_hat.loc[test_idx] - Xc.loc[test_idx])**2).mean())
    mae = (X_hat.loc[test_idx] - Xc.loc[test_idx]).abs().mean()

    ss_res = ((Xc.loc[test_idx] - X_hat.loc[test_idx])**2).sum()
    ss_tot = ((Xc.loc[test_idx] - Xc.loc[test_idx].mean())**2).sum()
    r2 = 1 - ss_res/ss_tot

    acc = len(set(turns_X_hat).intersection(set(turns_X_act))) / max(1, len(turns_X_act))

    # Tolerant turn accuracy (±2 weeks)
    k = 2
    matched = 0
    for t in turns_X_hat:
        if any(abs((t - act).days) <= k*7 for act in turns_X_act):
            matched += 1
    acc_tol = matched / max(1, len(turns_X_act))

    print("\n=== Prediction Quality ===")
    print(f"Correlation (test): {corr:.3f}")
    print(f"RMSE (test): {rmse:.3f}")
    print(f"MAE (test): {mae:.3f}")
    print(f"R² (test): {r2:.3f}")
    print(f"Turn prediction accuracy (exact): {acc:.2%}")
    print(f"Turn prediction accuracy (±{k} weeks): {acc_tol:.2%}")

# -------------------- Main --------------------

def main(years_back=20, train_frac=0.8):
    end = datetime.now()
    start = end - timedelta(days=365*years_back)

    # --- Fetch oil & producers ---
    oil, prod = get_oil_and_producers(start, end)
    oil = pd.Series(oil.squeeze(), name="X_oil")
    prod = pd.Series(prod.squeeze(), name="Y_prod")

    df = pd.concat([oil, prod], axis=1, join="inner").dropna()

    # Debug
    print("Columns in df:", df.columns.tolist())
    print(df.head())

    # --- Cycle extraction ---
    Xc = extract_cycle(df["X_oil"])
    Yc = extract_cycle(df["Y_prod"])
    cyc = pd.concat([Xc, Yc], axis=1, join="inner").dropna()
    Xc, Yc = cyc.iloc[:, 0], cyc.iloc[:, 1]

    # --- Train/test split ---
    split = int(len(cyc) * train_frac)
    train_idx = cyc.index[:split]
    test_idx = cyc.index[split:]

    # --- Parameter estimation ---
    a, b, g, d, cost, ok = fit_lv_nls(Xc.loc[train_idx], Yc.loc[train_idx])
    print("=== LV params ===")
    print(f"alpha={a:.4f}, beta={b:.4f}, gamma={g:.4f}, delta={d:.4f} (success={ok})")

    # --- Forecasting ---
    X_hat, Y_hat = one_step_forecast(Xc, Yc, (a, b, g, d))

    # --- Turning points ---
    turns_X_act = detect_turns(Xc.loc[test_idx])
    turns_X_hat = detect_turns(X_hat.loc[test_idx])
    print(f"Test turns (Oil) — model {len(turns_X_hat)} vs actual {len(turns_X_act)}")

    # --- Prediction Quality ---
    try:
        # Align indices
        X_hat_test = X_hat.reindex(test_idx)
        Xc_test = Xc.reindex(test_idx)

        # Core metrics
        corr = X_hat_test.corr(Xc_test)
        rmse = np.sqrt(((X_hat_test - Xc_test) ** 2).mean())
        mae = (X_hat_test - Xc_test).abs().mean()

        ss_res = ((Xc_test - X_hat_test) ** 2).sum()
        ss_tot = ((Xc_test - Xc_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

        # Turn accuracy (exact match on weeks)
        acc = len(set(turns_X_hat).intersection(set(turns_X_act))) / max(1, len(turns_X_act))

        # Tolerant turn accuracy (±2 weeks)
        k = 2
        matched = 0
        for t in turns_X_hat:
            for act in turns_X_act:
                if abs((t - act).days) <= k * 7:
                    matched += 1
                    break
        acc_tol = matched / max(1, len(turns_X_act))

        # Print metrics
        print("\n=== Prediction Quality ===")
        print(f"Correlation (test): {corr:.3f}")
        print(f"RMSE (test): {rmse:.3f}")
        print(f"MAE (test): {mae:.3f}")
        print(f"R² (test): {r2:.3f}")
        print(f"Turn prediction accuracy (exact): {acc:.2%}")
        print(f"Turn prediction accuracy (±{k} weeks): {acc_tol:.2%}")

    except Exception as e:
        import traceback
        print("\nError while computing metrics:", e)
        traceback.print_exc()

    # --- Plots (at the end) ---
    plot_cycles(train_idx, Xc, Yc, X_hat, Y_hat)


if __name__ == "__main__":
    main()


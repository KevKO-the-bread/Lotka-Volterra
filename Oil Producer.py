#!/usr/bin/env python3
"""
Oil (CL=F) vs Oil Producers (XOP) — Plot & Lotka–Volterra (LV) Quick View
- Fetches CL=F and XOP from Yahoo
- Aligns/cleans data
- Handles negative oil prices (Apr 2020) for logs
- Plots: prices (dual-axis), normalized series, phase-space (LV-style)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import yfinance as yf
from scipy.integrate import odeint
from sklearn.preprocessing import StandardScaler


# ---------- Helpers ----------

def _clean_series(s, name):
    """Return a 1D Series with DatetimeIndex and a stable name."""
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = name
    return s.sort_index()

def fetch_series(ticker, name, start, end):
    """Download one ticker and return Close as a clean Series."""
    df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No data for {ticker}")
    return _clean_series(df["Close"], name)

def align(oil, prod):
    """Inner-join by date."""
    return pd.concat([oil.rename("oil"), prod.rename("producers")], axis=1, join="inner").dropna()

def prep_lv(df):
    """Log->standardize->shift positive for LV-like phase plot."""
    oil, prod = df["oil"].copy(), df["producers"].copy()
    # handle negative/zero oil (Apr 2020)
    if (oil <= 0).any():
        oil = oil + abs(oil.min()) + 1e-6
    if (prod <= 0).any():
        prod = prod + abs(min(0, prod.min())) + 1e-6
    log_oil = np.log(oil)
    log_prod = np.log(prod)
    z = StandardScaler().fit_transform(np.column_stack([log_oil, log_prod]))
    lv = pd.DataFrame(
        {"oil": z[:, 0] + 4.0, "producers": z[:, 1] + 4.0},
        index=df.index
    )
    return lv

# Optional: simple LV simulation with fixed params (for visualization)
def lv_simulate(alpha, beta, gamma, delta, init_o, init_p, n):
    def derivs(state, t):
        o, p = state
        return [alpha*o - beta*o*p, gamma*o*p - delta*p]
    t = np.linspace(0, n-1, n)
    return odeint(derivs, [init_o, init_p], t)


# ---------- Main ----------

def main(years_back=18):
    end = datetime.now()
    start = end - timedelta(days=years_back*365)

    # Fetch exactly CL=F and XOP (no fallbacks, simpler & stable)
    oil = fetch_series("CL=F", "WTI Crude Oil ($)", start, end)
    prod = fetch_series("XOP", "Oil Producers ETF ($)", start, end)

    df = align(oil, prod)
    print(f"Aligned: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  n={len(df)}")

    # Build LV-ready normalized data
    lv = prep_lv(df)

    # ----- PLOTTING (always returns a graph) -----
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    fig.suptitle("Oil (CL=F) vs Oil Producers (XOP) — Predator–Prey View", fontsize=16, fontweight="bold")

    # (1) Prices dual-axis
    ax1 = fig.add_subplot(gs[0, :])
    ax1_t = ax1.twinx()
    ax1.plot(df.index, df["oil"], lw=2, label="WTI Crude (CL=F)", color="black")
    ax1_t.plot(df.index, df["producers"], lw=2, label="XOP (Producers)", color="darkred")
    ax1.set_ylabel("Oil ($)", color="black")
    ax1_t.set_ylabel("XOP ($)", color="darkred")
    ax1.set_title("Prices")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lines = ax1.get_lines() + ax1_t.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")

    # (2) Normalized comparison (index to 100)
    ax2 = fig.add_subplot(gs[1, 0])
    idx_oil = df["oil"] / df["oil"].iloc[0] * 100
    idx_prod = df["producers"] / df["producers"].iloc[0] * 100
    ax2.plot(df.index, idx_oil, lw=2, label="Oil (idx=100)", color="black")
    ax2.plot(df.index, idx_prod, lw=2, label="XOP (idx=100)", color="darkred")
    ax2.set_title("Indexed to 100")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # (3) Phase-space (LV-style)
    ax3 = fig.add_subplot(gs[1, 1])
    sc = ax3.scatter(lv["oil"], lv["producers"], c=np.arange(len(lv)), cmap="plasma", s=6, alpha=0.8)
    ax3.set_xlabel("Oil (normalized, >0)")
    ax3.set_ylabel("Producers (normalized, >0)")
    ax3.set_title("Phase Space (Predator–Prey Orbit)")
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("Time →")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

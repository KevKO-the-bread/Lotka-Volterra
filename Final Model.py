"""
VFGLVM: Multi-Frequency Comparison (Daily, Weekly, Monthly)
===========================================================
Uses the EXACT same approach as the original VFGLVM code.
Generates 6-panel comparison plot showing actual vs fitted.
"""

from __future__ import annotations
import math, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gammaln

# ============================================================================
# CONFIG
# ============================================================================

START_DATE = "2016-01-01"
END_DATE   = "2025-09-30"

PATH_GOLD_ETF   = r"C:\Users\Kevin\Downloads\Thesis\Data\Gold Volume.xlsx"
PATH_LBMA_JSON  = r"C:\Users\Kevin\Downloads\Thesis\Data\LBMA gold price 12.10.2025.json"
PATH_CRYPTO_TSV = r"C:\Users\Kevin\Downloads\Thesis\Data\coin-metrics-new-chart.tsv.tsv"

WINSOR_PCT = 0.005

OUTDIR = Path.cwd() / "output" / "vfglvm_multi_frequency"
OUTDIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 100,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})
plt.style.use("seaborn-v0_8-darkgrid")

# ============================================================================
# CORE VFGLVM (EXACT COPY FROM ORIGINAL)
# ============================================================================

def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)

def _ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    XtX = X.T @ X
    XtX[np.diag_indices_from(XtX)] += alpha
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)

def _rolling_slope(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    out = np.full(n, np.nan)
    t = np.arange(n, dtype=float)
    for end in range(window - 1, n):
        start = end - window + 1
        yy = y[start:end+1]
        tt = t[start:end+1]
        mask = np.isfinite(yy)
        if mask.sum() < max(2, window // 2):
            continue
        yyv = yy[mask]
        ttv = tt[mask]
        ym = np.mean(yyv)
        tm = np.mean(ttv)
        num = np.sum((ttv - tm) * (yyv - ym))
        den = np.sum((ttv - tm)**2)
        out[end] = np.nan if den == 0 else num / den
    return out

def estimate_q_t(series_matrix: np.ndarray, window: int,
                 clip: Tuple[float,float]=(0.3,0.7), smooth_ma: int=4) -> np.ndarray:
    n, k = series_matrix.shape
    if window > k // 2:
        window = max(4, k // 4)
    slopes_abs = []
    for i in range(n):
        s = _rolling_slope(series_matrix[i], window)
        slopes_abs.append(np.abs(s))
    S = np.nanmean(np.vstack(slopes_abs), axis=0)
    valid = S[np.isfinite(S)]
    if len(valid) < 5:
        q = np.full(k, np.mean(clip))
    else:
        p75 = np.nanpercentile(valid, 75)
        if not np.isfinite(p75) or p75 <= 0:
            q = np.full(k, np.mean(clip))
        else:
            q = S / p75
        q = pd.Series(q).ffill().bfill().values
        q = np.clip(q, clip[0], clip[1])
    if smooth_ma and smooth_ma > 1:
        q = pd.Series(q).rolling(smooth_ma, min_periods=1, center=True).mean().values
        q = np.clip(q, clip[0], clip[1])
    return q

def fractional_accum_variable_order(x: np.ndarray, q_t: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(x)
    if len(q_t) != n:
        raise ValueError("q_t must have same length as x")
    out = np.zeros(n, dtype=float)
    for k in range(n):
        qk = float(np.clip(q_t[k], 0.05, 0.95))
        i_start = max(0, k - max_lag + 1)
        acc = 0.0
        for i in range(i_start, k+1):
            dist = k - i
            lg = gammaln(qk + dist) - gammaln(qk) - gammaln(dist + 1.0)
            coeff = math.exp(lg)
            acc += coeff * x[i]
        out[k] = acc
    return out

def build_Z_q(levels: List[np.ndarray], q_t: np.ndarray, max_lag: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    Xq = [fractional_accum_variable_order(x, q_t, max_lag=max_lag) for x in levels]
    Zq = []
    for xq in Xq:
        z = np.full_like(xq, np.nan)
        z[1:] = 0.5 * (xq[1:] + xq[:-1])
        Zq.append(z)
    return Xq, Zq

@dataclass
class VFGLVMParams:
    a: np.ndarray
    b: np.ndarray
    C: np.ndarray

def fit_vfglvm_params(levels: List[np.ndarray], Zq: List[np.ndarray], ridge_alpha: float = 1e-5) -> VFGLVMParams:
    n = len(levels)
    T = min(len(x) for x in levels)
    X0 = np.vstack([_as_array(x) for x in levels])[:, :T]
    Z  = np.vstack([_as_array(z) for z in Zq])[:, :T]
    idx_all = np.arange(1, T-1)
    mask = np.ones(len(idx_all), dtype=bool)
    for i in range(n):
        mask &= np.isfinite(Z[i, idx_all]) & np.isfinite(X0[i, idx_all+1])
    idx = idx_all[mask]
    if len(idx) < 5:
        raise RuntimeError(f"Not enough usable points to fit VFGLVM ({len(idx)})")
    a_hat = np.zeros(n)
    b_hat = np.zeros(n)
    C_hat = np.zeros((n, n))
    for i in range(n):
        zi = Z[i, idx]
        cols = [zi, -(zi**2)]
        for j in range(n):
            if j == i:
                continue
            zj = Z[j, idx]
            cols.append(-(zi * zj))
        Bi = np.column_stack(cols)
        Mi = X0[i, idx+1]
        theta = _ridge(Bi, Mi, alpha=ridge_alpha)
        a_hat[i] = theta[0]
        b_hat[i] = theta[1]
        c_idx = 2
        for j in range(n):
            if j == i:
                continue
            C_hat[i, j] = theta[c_idx]
            c_idx += 1
    return VFGLVMParams(a=a_hat, b=b_hat, C=C_hat)

def vfglvm_fitted_values(params: VFGLVMParams, levels: List[np.ndarray], Zq: List[np.ndarray]) -> List[np.ndarray]:
    n = len(levels)
    T = len(levels[0])
    fitted = [np.full(T, np.nan) for _ in range(n)]
    for k in range(1, T):
        z_k = np.array([Zq[i][k] for i in range(n)])
        if not np.all(np.isfinite(z_k)):
            continue
        for i in range(n):
            pred = params.a[i]*z_k[i] - params.b[i]*(z_k[i]**2)
            for j in range(n):
                if j != i:
                    pred -= params.C[i,j]*z_k[i]*z_k[j]
            if k+1 < T:
                fitted[i][k+1] = pred
    return fitted

def _default_max_lag(n_obs: int) -> int:
    return 36 if n_obs <= 200 else 156

def metrics_levels(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return dict(RMSE=np.nan, MAE=np.nan, MAPE=np.nan)
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yt - yp
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs(err)/np.abs(yt)
        ape = ape[np.isfinite(ape)]
        mape = float(np.mean(ape)*100) if len(ape)>0 else np.nan
    return dict(RMSE=rmse, MAE=mae, MAPE=mape)

# ============================================================================
# DATA LOADING (EXACT COPY FROM ORIGINAL)
# ============================================================================

def month_end_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="ME")

def daily_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")

def align_level_to_eom(series: pd.Series, eom_idx: pd.DatetimeIndex) -> pd.Series:
    series = series.sort_index()
    out = []
    for ts in eom_idx:
        hist = series.loc[series.index <= ts]
        out.append(hist.iloc[-1] if len(hist) else np.nan)
    return pd.Series(out, index=eom_idx)

def _winsorize(s: pd.Series, p: float | None) -> pd.Series:
    if p is None or p <= 0:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def load_gold_etf_holdings_monthly(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[["Date", "Ounces"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ounces"] = pd.to_numeric(df["Ounces"], errors="coerce")
    df = df.dropna().sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(START_DATE)) &
            (df["Date"] <= pd.to_datetime(END_DATE))]
    eom_idx = month_end_index(START_DATE, END_DATE)
    s_oz = pd.Series(df["Ounces"].values, index=df["Date"].values)
    s_eom_oz = align_level_to_eom(s_oz, eom_idx)
    return pd.DataFrame({"gold_etf_ounces": s_eom_oz}, index=eom_idx)

def load_lbma_gold_price_daily(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for rec in data:
        d = rec.get("d")
        vals = rec.get("v", [])
        usd = vals[0] if isinstance(vals, list) and len(vals) > 0 else None
        rows.append((d, usd))
    df = pd.DataFrame(rows, columns=["date", "usd_per_oz"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["usd_per_oz"] = pd.to_numeric(df["usd_per_oz"], errors="coerce")
    df = df.dropna().sort_values("date")
    df = df[(df["date"] >= pd.to_datetime(START_DATE)) &
            (df["date"] <= pd.to_datetime(END_DATE))]
    df = df.set_index("date")
    full_idx = daily_index(START_DATE, END_DATE)
    df_daily = df.reindex(full_idx).ffill()
    df_daily.columns = ["gold_usd_per_oz"]
    return df_daily

def load_crypto_daily(path: str) -> pd.DataFrame:
    encodings = ['utf-16','utf-16-le','utf-16-be','utf-8-sig','latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError("Could not read crypto TSV.")
    need = ["Time", "BTC / Market Cap (USD)", "ETH / Market Cap (USD)"]
    df = df[need].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["BTC / Market Cap (USD)"] = pd.to_numeric(df["BTC / Market Cap (USD)"], errors="coerce")
    df["ETH / Market Cap (USD)"] = pd.to_numeric(df["ETH / Market Cap (USD)"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")
    df = df[(df["Time"] >= pd.to_datetime(START_DATE)) &
            (df["Time"] <= pd.to_datetime(END_DATE))]
    df = df.set_index("Time")
    daily = df.resample("D").last()
    if WINSOR_PCT and WINSOR_PCT > 0:
        daily["BTC / Market Cap (USD)"] = _winsorize(daily["BTC / Market Cap (USD)"], WINSOR_PCT)
        daily["ETH / Market Cap (USD)"] = _winsorize(daily["ETH / Market Cap (USD)"], WINSOR_PCT)
    daily["crypto_total_usd"] = (daily["BTC / Market Cap (USD)"].fillna(0) +
                                 daily["ETH / Market Cap (USD)"].fillna(0))
    return daily[["crypto_total_usd"]]

def interpolate_ounces_daily(gold_ounces_monthly: pd.DataFrame) -> pd.DataFrame:
    s_monthly = gold_ounces_monthly["gold_etf_ounces"].copy()
    s_monthly = s_monthly.dropna()
    full_idx = daily_index(START_DATE, END_DATE)
    s_daily = s_monthly.reindex(full_idx)
    s_daily = s_daily.interpolate(method="time")
    return pd.DataFrame({"gold_etf_ounces_daily": s_daily}, index=full_idx)

def build_gold_aum_daily(ounces_daily: pd.DataFrame, gold_price_daily: pd.DataFrame) -> pd.DataFrame:
    df = ounces_daily.join(gold_price_daily, how="inner")
    df["gold_etf_mcap_usd"] = df["gold_etf_ounces_daily"] * df["gold_usd_per_oz"]
    return df[["gold_etf_mcap_usd"]]

# ============================================================================
# PANEL BUILDERS (using original scaling approach)
# ============================================================================

def build_panel_and_scale(gold_aum: pd.DataFrame, crypto: pd.DataFrame, freq_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    print(f"\n[{freq_name}] Building scaled panel...")
    df = gold_aum.join(crypto[["crypto_total_usd"]], how="inner").sort_index()
    panel_raw = df[["gold_etf_mcap_usd", "crypto_total_usd"]].dropna(how="any").copy()
    
    # EXACT scaling from original
    eps = 1e-12
    med_gold  = np.nanmedian(panel_raw["gold_etf_mcap_usd"])
    med_crypto = np.nanmedian(panel_raw["crypto_total_usd"])
    if not (np.isfinite(med_gold) and med_gold > 0):   med_gold = 1.0
    if not (np.isfinite(med_crypto) and med_crypto > 0): med_crypto = 1.0
    
    panel_scaled = pd.DataFrame({
        "x": panel_raw["gold_etf_mcap_usd"] / (med_gold + eps),
        "y": panel_raw["crypto_total_usd"]   / (med_crypto + eps),
    }, index=panel_raw.index)
    
    scale_info = {
        "gold_median_usd": med_gold,
        "crypto_total_median_usd": med_crypto
    }
    
    print(f"   {freq_name}: {len(panel_scaled)} observations")
    print(f"   Gold median:  {med_gold:.3g} USD")
    print(f"   Crypto median: {med_crypto:.3g} USD")
    
    return panel_raw, panel_scaled, scale_info

def build_monthly():
    gold_ounces_m = load_gold_etf_holdings_monthly(PATH_GOLD_ETF)
    lbma_price_d  = load_lbma_gold_price_daily(PATH_LBMA_JSON)
    lbma_monthly = lbma_price_d.resample("ME").last()
    
    gold_aum = gold_ounces_m.join(lbma_monthly, how="inner")
    gold_aum["gold_etf_mcap_usd"] = gold_aum["gold_etf_ounces"] * gold_aum["gold_usd_per_oz"]
    gold_aum = gold_aum[["gold_etf_mcap_usd"]]
    
    crypto_d = load_crypto_daily(PATH_CRYPTO_TSV)
    crypto_monthly = crypto_d.resample("ME").last()
    
    return build_panel_and_scale(gold_aum, crypto_monthly, "MONTHLY")

def build_weekly():
    # Build from daily then resample
    gold_ounces_m = load_gold_etf_holdings_monthly(PATH_GOLD_ETF)
    gold_ounces_d = interpolate_ounces_daily(gold_ounces_m)
    lbma_price_d  = load_lbma_gold_price_daily(PATH_LBMA_JSON)
    gold_aum_d = build_gold_aum_daily(gold_ounces_d, lbma_price_d)
    
    crypto_d = load_crypto_daily(PATH_CRYPTO_TSV)
    
    # Resample to weekly
    gold_aum_w = gold_aum_d.resample("W-FRI").last()
    crypto_w = crypto_d.resample("W-FRI").last()
    
    return build_panel_and_scale(gold_aum_w, crypto_w, "WEEKLY")

def build_daily():
    gold_ounces_m = load_gold_etf_holdings_monthly(PATH_GOLD_ETF)
    gold_ounces_d = interpolate_ounces_daily(gold_ounces_m)
    lbma_price_d  = load_lbma_gold_price_daily(PATH_LBMA_JSON)
    gold_aum_d = build_gold_aum_daily(gold_ounces_d, lbma_price_d)
    crypto_d = load_crypto_daily(PATH_CRYPTO_TSV)
    
    return build_panel_and_scale(gold_aum_d, crypto_d, "DAILY")

# ============================================================================
# FIT VFGLVM (using original approach)
# ============================================================================

def fit_vfglvm_frequency(panel_scaled: pd.DataFrame, panel_raw: pd.DataFrame, scale_info: Dict,
                         freq_name: str, window_q: int, smooth_ma_q: int) -> Dict:
    print(f"\n[{freq_name}] Fitting VFGLVM...")
    
    levels = [_as_array(panel_scaled["x"].values), _as_array(panel_scaled["y"].values)]
    X0_full = np.vstack(levels)
    
    # Estimate q(t) - EXACT from original
    q_t = estimate_q_t(X0_full, window=window_q, smooth_ma=smooth_ma_q)
    if not np.isfinite(q_t).any():
        q_t = np.full(len(panel_scaled), 0.5)
    
    max_lag = _default_max_lag(len(panel_scaled))
    Xq, Zq = build_Z_q(levels, q_t, max_lag=max_lag)
    
    # Fit parameters
    params = fit_vfglvm_params(levels, Zq, ridge_alpha=1e-5)
    
    # Fitted values (scaled)
    fitted_list = vfglvm_fitted_values(params, levels, Zq)
    
    # Unscale to original units
    med_gold = scale_info["gold_median_usd"]
    med_crypto = scale_info["crypto_total_median_usd"]
    
    gold_fitted_raw = fitted_list[0] * med_gold
    crypto_fitted_raw = fitted_list[1] * med_crypto
    
    # Create output dataframe
    df_out = pd.DataFrame({
        "gold_actual": panel_raw["gold_etf_mcap_usd"],
        "gold_fitted": gold_fitted_raw,
        "crypto_actual": panel_raw["crypto_total_usd"],
        "crypto_fitted": crypto_fitted_raw,
    }, index=panel_raw.index)
    
    # Calculate metrics
    gold_metrics = metrics_levels(panel_raw["gold_etf_mcap_usd"].values, gold_fitted_raw)
    crypto_metrics = metrics_levels(panel_raw["crypto_total_usd"].values, crypto_fitted_raw)
    
    print(f"   Gold:   MAPE={gold_metrics['MAPE']:.2f}%, RMSE=${gold_metrics['RMSE']/1e9:.2f}B")
    print(f"   Crypto: MAPE={crypto_metrics['MAPE']:.2f}%, RMSE=${crypto_metrics['RMSE']/1e9:.2f}B")
    
    return {
        "df_fit": df_out,
        "params": params,
        "q_t": q_t,
        "gold_mape": gold_metrics['MAPE'],
        "gold_rmse": gold_metrics['RMSE'],
        "crypto_mape": crypto_metrics['MAPE'],
        "crypto_rmse": crypto_metrics['RMSE'],
    }

# ============================================================================
# 6-PANEL PLOT
# ============================================================================

def plot_6_panel(results_daily, results_weekly, results_monthly):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Gold
    for i, (freq_name, results) in enumerate([
        ("Daily", results_daily),
        ("Weekly", results_weekly),
        ("Monthly", results_monthly)
    ]):
        ax = axes[0, i]
        df = results["df_fit"]
        
        ax.plot(df.index, df["gold_actual"]/1e9, label="Actual", 
                lw=2, alpha=0.7, color='goldenrod')
        ax.plot(df.index, df["gold_fitted"]/1e9, label="VFGLVM",
                lw=1.5, alpha=0.9, linestyle='--', color='darkgoldenrod')
        
        ax.set_title(f"Gold ETF AUM - {freq_name}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Billions USD", fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        
        ax.text(0.02, 0.98, f"MAPE: {results['gold_mape']:.1f}%\nRMSE: ${results['gold_rmse']/1e9:.1f}B", 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.6))
    
    # Row 2: Crypto
    for i, (freq_name, results) in enumerate([
        ("Daily", results_daily),
        ("Weekly", results_weekly),
        ("Monthly", results_monthly)
    ]):
        ax = axes[1, i]
        df = results["df_fit"]
        
        ax.plot(df.index, df["crypto_actual"]/1e9, label="Actual",
                lw=2, alpha=0.7, color='steelblue')
        ax.plot(df.index, df["crypto_fitted"]/1e9, label="VFGLVM",
                lw=1.5, alpha=0.9, linestyle='--', color='navy')
        
        ax.set_title(f"Total Crypto (BTC+ETH) - {freq_name}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Billions USD", fontsize=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        
        ax.text(0.02, 0.98, f"MAPE: {results['crypto_mape']:.1f}%\nRMSE: ${results['crypto_rmse']/1e9:.1f}B",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "vfglvm_6panel_comparison.png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved: {OUTDIR / 'vfglvm_6panel_comparison.png'}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" VFGLVM MULTI-FREQUENCY COMPARISON")
    print(" (Using Original VFGLVM Approach)")
    print("="*70)
    
    # Build panels with original scaling
    panel_raw_monthly, panel_scaled_monthly, scale_monthly = build_monthly()
    panel_raw_weekly, panel_scaled_weekly, scale_weekly = build_weekly()
    panel_raw_daily, panel_scaled_daily, scale_daily = build_daily()
    
    # Fit VFGLVM with original parameters
    results_monthly = fit_vfglvm_frequency(
        panel_scaled_monthly, panel_raw_monthly, scale_monthly,
        "MONTHLY", window_q=12, smooth_ma_q=4
    )
    
    results_weekly = fit_vfglvm_frequency(
        panel_scaled_weekly, panel_raw_weekly, scale_weekly,
        "WEEKLY", window_q=20, smooth_ma_q=6
    )
    
    results_daily = fit_vfglvm_frequency(
        panel_scaled_daily, panel_raw_daily, scale_daily,
        "DAILY", window_q=30, smooth_ma_q=8
    )
    
    # 6-panel plot
    plot_6_panel(results_daily, results_weekly, results_monthly)
    
    # Summary
    print("\n" + "="*70)
    print(" PERFORMANCE SUMMARY")
    print("="*70)
    
    summary = pd.DataFrame([
        {"Frequency": "Monthly", "Asset": "Gold", "MAPE_%": results_monthly["gold_mape"], "RMSE_B": results_monthly["gold_rmse"]/1e9},
        {"Frequency": "Monthly", "Asset": "Crypto", "MAPE_%": results_monthly["crypto_mape"], "RMSE_B": results_monthly["crypto_rmse"]/1e9},
        {"Frequency": "Weekly", "Asset": "Gold", "MAPE_%": results_weekly["gold_mape"], "RMSE_B": results_weekly["gold_rmse"]/1e9},
        {"Frequency": "Weekly", "Asset": "Crypto", "MAPE_%": results_weekly["crypto_mape"], "RMSE_B": results_weekly["crypto_rmse"]/1e9},
        {"Frequency": "Daily", "Asset": "Gold", "MAPE_%": results_daily["gold_mape"], "RMSE_B": results_daily["gold_rmse"]/1e9},
        {"Frequency": "Daily", "Asset": "Crypto", "MAPE_%": results_daily["crypto_mape"], "RMSE_B": results_daily["crypto_rmse"]/1e9},
    ])
    
    print("\n", summary.round(2).to_string(index=False))
    summary.to_csv(OUTDIR / "performance_summary.csv", index=False)
    
    # Save fitted data
    results_monthly["df_fit"].to_csv(OUTDIR / "fitted_monthly.csv")
    results_weekly["df_fit"].to_csv(OUTDIR / "fitted_weekly.csv")
    results_daily["df_fit"].to_csv(OUTDIR / "fitted_daily.csv")
    
    # Parameter comparison
    print("\n" + "="*70)
    print(" PARAMETER COMPARISON")
    print("="*70)
    
    param_df = pd.DataFrame({
        "Frequency": ["Monthly", "Weekly", "Daily"],
        "a_gold": [results_monthly["params"].a[0], results_weekly["params"].a[0], results_daily["params"].a[0]],
        "a_crypto": [results_monthly["params"].a[1], results_weekly["params"].a[1], results_daily["params"].a[1]],
        "b_gold": [results_monthly["params"].b[0], results_weekly["params"].b[0], results_daily["params"].b[0]],
        "b_crypto": [results_monthly["params"].b[1], results_weekly["params"].b[1], results_daily["params"].b[1]],
        "C_crypto_to_gold": [results_monthly["params"].C[0,1], results_weekly["params"].C[0,1], results_daily["params"].C[0,1]],
        "C_gold_to_crypto": [results_monthly["params"].C[1,0], results_weekly["params"].C[1,0], results_daily["params"].C[1,0]],
    })
    
    print("\n", param_df.round(6).to_string(index=False))
    param_df.to_csv(OUTDIR / "parameter_comparison.csv", index=False)
    
    print("\n✅ All outputs saved to:", OUTDIR)
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

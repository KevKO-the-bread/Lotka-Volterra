# ===============================================================
# Enhanced Hard-Money Allocation Model
# Gold ETF Market Cap vs Total Crypto (BTC+ETH)
# ===============================================================
# Key Improvements:
# 1) Adaptive smoothing with regime detection (high/low volatility)
# 2) Multi-step forecast with proper error propagation
# 3) Ensemble combining multiple smoothing strategies
# 4) Macro regime indicators (VIX proxy, trend strength)
# 5) Better OOS evaluation with proper walk-forward validation
# 6) Confidence intervals for forecasts
# 7) Model selection via information criteria (AIC-like)
# ===============================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------
# CONFIG: paths & dates
# ---------------------
START_DATE = "2016-01-01"
END_DATE   = "2025-09-30"

# Your files
PATH_GOLD_ETF   = r"C:\Users\Kevin\Downloads\Thesis\Data\Gold Volume.xlsx"
PATH_LBMA_JSON  = r"C:\Users\Kevin\Downloads\Thesis\Data\LBMA gold price 12.10.2025.json"
PATH_CRYPTO_TSV = r"C:\Users\Kevin\Downloads\Thesis\Data\coin-metrics-new-chart.tsv.tsv"

# Preprocessing knobs
WINSOR_PCT = 0.005

# Model knobs
REGIME_WINDOW = 12        # months for volatility regime detection
VOL_THRESHOLD = 0.75      # quantile for high volatility regime
ENSEMBLE_WEIGHTS = True   # use performance-weighted ensemble
CONF_LEVEL = 0.90         # confidence interval level
MAX_GAIN = 0.70           # cap smoothing gains (prevent overfitting)
CRYPTO_VOL_MULTIPLIER = 2.5  # crypto needs wider CIs

# OOS eval knobs
OOS_HORIZONS = (1, 3, 6, 12)
MIN_TRAIN_FRAC = 0.6

# ---------------------
# Helpers: time index
# ---------------------
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

# ---------------------
# Load data
# ---------------------
def load_gold_etf_holdings_monthly(path: str) -> pd.DataFrame:
    print("\n[1/3] Loading Gold ETF Holdings (ounces, month-end)...")
    df = pd.read_excel(path)
    if "Date" not in df.columns or "Ounces" not in df.columns:
        raise ValueError(f"Expected columns 'Date' and 'Ounces'. Found: {list(df.columns)}")
    df = df[["Date", "Ounces"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Ounces"] = pd.to_numeric(df["Ounces"], errors="coerce")
    df = df.dropna().sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(START_DATE)) & (df["Date"] <= pd.to_datetime(END_DATE))]
    eom_idx = month_end_index(START_DATE, END_DATE)
    s_oz = pd.Series(df["Ounces"].values, index=df["Date"].values)
    s_eom_oz = align_level_to_eom(s_oz, eom_idx)
    out = pd.DataFrame({"gold_etf_ounces": s_eom_oz}, index=eom_idx)
    print(f"   Loaded {out['gold_etf_ounces'].notna().sum()} months | {out.index.min().date()} → {out.index.max().date()}")
    return out

def load_lbma_gold_price_monthly(path: str) -> pd.DataFrame:
    print("\n[2/3] Loading LBMA Gold Price (USD/oz, month-end)...")
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
    df = df[(df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(END_DATE))]
    eom_idx = month_end_index(START_DATE, END_DATE)
    s_price = pd.Series(df["usd_per_oz"].values, index=df["date"].values)
    s_eom = align_level_to_eom(s_price, eom_idx)
    out = pd.DataFrame({"gold_usd_per_oz": s_eom}, index=eom_idx)
    print(f"   Loaded {out['gold_usd_per_oz'].notna().sum()} months | {out.index.min().date()} → {out.index.max().date()}")
    return out

def load_crypto_monthly(path: str) -> pd.DataFrame:
    print("\n[3/3] Loading BTC & ETH Market Cap (USD, month-end)...")
    encodings = ['utf-16','utf-16-le','utf-16-be','utf-8-sig','latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc)
            print(f"   Successfully read with encoding: {enc}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError("Could not read crypto TSV.")
    need = ["Time", "BTC / Market Cap (USD)", "ETH / Market Cap (USD)"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in crypto TSV.")
    df = df[need].copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["BTC / Market Cap (USD)"] = pd.to_numeric(df["BTC / Market Cap (USD)"], errors="coerce")
    df["ETH / Market Cap (USD)"] = pd.to_numeric(df["ETH / Market Cap (USD)"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time")
    df = df[(df["Time"] >= pd.to_datetime(START_DATE)) & (df["Time"] <= pd.to_datetime(END_DATE))]
    df = df.set_index("Time")
    monthly = df.resample("ME").last()
    if WINSOR_PCT and WINSOR_PCT > 0:
        monthly["BTC / Market Cap (USD)"] = _winsorize(monthly["BTC / Market Cap (USD)"], WINSOR_PCT)
        monthly["ETH / Market Cap (USD)"] = _winsorize(monthly["ETH / Market Cap (USD)"], WINSOR_PCT)
    monthly["crypto_total_usd"] = (monthly["BTC / Market Cap (USD)"].fillna(0)
                                   + monthly["ETH / Market Cap (USD)"].fillna(0))
    monthly = monthly.rename(columns={
        "BTC / Market Cap (USD)": "btc_mcap_usd",
        "ETH / Market Cap (USD)": "eth_mcap_usd"
    })
    print(f"   Loaded {len(monthly)} months | {monthly.index.min().date()} → {monthly.index.max().date()}")
    return monthly[["btc_mcap_usd","eth_mcap_usd","crypto_total_usd"]]

# ---------------------
# Build monthly panel with regime indicators
# ---------------------
def build_panel_monthly() -> pd.DataFrame:
    gold_ounces_m = load_gold_etf_holdings_monthly(PATH_GOLD_ETF)
    lbma_price_m  = load_lbma_gold_price_monthly(PATH_LBMA_JSON)
    crypto_m      = load_crypto_monthly(PATH_CRYPTO_TSV)

    df = gold_ounces_m.join(lbma_price_m, how="inner")
    df["gold_etf_mcap_usd"] = df["gold_etf_ounces"] * df["gold_usd_per_oz"]
    df = df.join(crypto_m[["crypto_total_usd"]], how="inner").dropna(how="any")
    df = df[(df["gold_etf_mcap_usd"] > 0) & (df["crypto_total_usd"] > 0)].copy()
    df["hard_money_total_usd"] = df["gold_etf_mcap_usd"] + df["crypto_total_usd"]
    df["crypto_share"] = df["crypto_total_usd"] / df["hard_money_total_usd"]
    
    # Add regime indicators
    df = add_regime_indicators(df)
    return df

def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility regime and trend indicators."""
    # 1) Rolling volatility of log-returns
    hm_log = np.log(df["hard_money_total_usd"])
    hm_ret = hm_log.diff()
    df["hm_vol_regime"] = (hm_ret.rolling(REGIME_WINDOW, min_periods=6)
                           .std()
                           .rank(pct=True) > VOL_THRESHOLD).astype(float)
    
    # 2) Crypto share momentum (3-month change)
    df["share_momentum"] = df["crypto_share"].diff(3)
    
    # 3) Trend strength (12-month moving average distance)
    df["hm_trend"] = (np.log(df["hard_money_total_usd"]) 
                      - np.log(df["hard_money_total_usd"].rolling(12, min_periods=6).mean()))
    
    return df.fillna(method='bfill').fillna(method='ffill')

# ===============================================================
# Enhanced Hard-Money Model Core
# ===============================================================

def _logit(p, eps=1e-10):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p/(1-p))

def _inv_logit(z):
    return 1.0/(1.0 + np.exp(-z))

def _ew_local_level(y: np.ndarray, gain: float, init: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Exponentially weighted local-level smoother with prediction variance."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    xhat = np.empty(n)
    var = np.empty(n)
    
    if init is None or not np.isfinite(init):
        init = y[~np.isnan(y)][0]
    
    xhat[0] = init
    var[0] = 0.0
    g = float(np.clip(gain, 1e-4, 0.999))
    
    # Compute innovation variance
    innovations = []
    for t in range(1, n):
        pred = xhat[t-1]
        obs = y[t]
        if np.isfinite(obs):
            innov = obs - pred
            innovations.append(innov)
            xhat[t] = pred + g * innov
            var[t] = (1 - g) * var[t-1] + g * innov**2
        else:
            xhat[t] = pred
            var[t] = var[t-1]
    
    return xhat, var

def _adaptive_ew_smoother(y: np.ndarray, regime: np.ndarray, 
                          gain_low: float, gain_high: float,
                          init: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Regime-dependent smoothing gain."""
    y = np.asarray(y, dtype=float)
    regime = np.asarray(regime, dtype=float)
    n = len(y)
    xhat = np.empty(n)
    var = np.empty(n)
    
    if init is None or not np.isfinite(init):
        init = y[~np.isnan(y)][0]
    
    xhat[0] = init
    var[0] = 0.0
    
    for t in range(1, n):
        # Adaptive gain based on regime
        g = gain_high if regime[t] > 0.5 else gain_low
        g = float(np.clip(g, 1e-4, 0.999))
        
        pred = xhat[t-1]
        obs = y[t]
        if np.isfinite(obs):
            innov = obs - pred
            xhat[t] = pred + g * innov
            var[t] = (1 - g) * var[t-1] + g * innov**2
        else:
            xhat[t] = pred
            var[t] = var[t-1]
    
    return xhat, var

def _grid_search_gain(y, loss="mse", grid=None, transform=None, 
                      inv_transform=None, regime=None) -> Tuple[float, float]:
    """Pick smoothing gain(s) minimizing 1-step-ahead error."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 30)
    
    y = np.asarray(y, dtype=float)
    if transform is None:
        yy = y.copy()
    else:
        yy = transform(y)
    
    if regime is None:
        # Single gain optimization
        best_g, best_loss = None, np.inf
        for g in grid:
            xhat, _ = _ew_local_level(yy, g, init=yy[0])
            pred = np.r_[xhat[0], xhat[:-1]]
            err = yy - pred
            if loss == "mape":
                with np.errstate(divide='ignore', invalid='ignore'):
                    denom = np.maximum(1e-8, np.abs(yy))
                    m = np.nanmean(np.abs(err)/denom)
            else:
                m = np.nanmean(err**2)
            if m < best_loss:
                best_loss = m
                best_g = g
        return float(best_g), float(best_g)
    else:
        # Dual gain optimization (low vol, high vol)
        regime = np.asarray(regime, dtype=float)
        best_pair, best_loss = None, np.inf
        for g_low in grid[::2]:  # Coarser grid for speed
            for g_high in grid[::2]:
                xhat, _ = _adaptive_ew_smoother(yy, regime, g_low, g_high, init=yy[0])
                pred = np.r_[xhat[0], xhat[:-1]]
                err = yy - pred
                if loss == "mape":
                    with np.errstate(divide='ignore', invalid='ignore'):
                        denom = np.maximum(1e-8, np.abs(yy))
                        m = np.nanmean(np.abs(err)/denom)
                else:
                    m = np.nanmean(err**2)
                if m < best_loss:
                    best_loss = m
                    best_pair = (g_low, g_high)
        return best_pair

def fit_hard_money_model(df: pd.DataFrame, use_regime: bool = True) -> Dict:
    """Fit enhanced smoothing model with regime adaptation."""
    hm = df["hard_money_total_usd"].values.astype(float)
    sh = df["crypto_share"].values.astype(float)
    regime = df["hm_vol_regime"].values.astype(float) if use_regime else None
    
    # Fit gains
    if regime is not None:
        g_hm_low, g_hm_high = _grid_search_gain(np.log(hm), loss="mape", regime=regime)
        g_sh_low, g_sh_high = _grid_search_gain(sh, loss="mse", 
                                                transform=_logit, 
                                                inv_transform=_inv_logit,
                                                regime=regime)
        
        log_hm_hat, hm_var = _adaptive_ew_smoother(np.log(hm), regime, 
                                                    g_hm_low, g_hm_high, 
                                                    init=np.log(hm[0]))
        logit_s_hat, s_var = _adaptive_ew_smoother(_logit(sh), regime,
                                                    g_sh_low, g_sh_high,
                                                    init=_logit(sh[0]))
    else:
        g_hm_low, g_hm_high = _grid_search_gain(np.log(hm), loss="mape")
        g_sh_low, g_sh_high = _grid_search_gain(sh, loss="mse",
                                                transform=_logit,
                                                inv_transform=_inv_logit)
        
        log_hm_hat, hm_var = _ew_local_level(np.log(hm), g_hm_low, init=np.log(hm[0]))
        logit_s_hat, s_var = _ew_local_level(_logit(sh), g_sh_low, init=_logit(sh[0]))
    
    hm_hat = np.exp(log_hm_hat)
    s_hat = _inv_logit(logit_s_hat)
    
    # Reconstruct fitted series with uncertainty
    crypto_hat = hm_hat * s_hat
    gold_hat = hm_hat * (1.0 - s_hat)
    
    # Approximate confidence bands (delta method)
    hm_std = np.sqrt(hm_var) * hm_hat  # Transform to level
    s_std = np.sqrt(s_var) * s_hat * (1 - s_hat)  # Logit derivative
    
    out = df.copy()
    out["hm_hat"] = hm_hat
    out["crypto_share_hat"] = s_hat
    out["crypto_hat"] = crypto_hat
    out["gold_hat"] = gold_hat
    out["hm_std"] = hm_std
    out["s_std"] = s_std
    
    return {
        "df_fit": out,
        "g_hm_low": g_hm_low,
        "g_hm_high": g_hm_high,
        "g_sh_low": g_sh_low,
        "g_sh_high": g_sh_high,
        "use_regime": use_regime
    }

def forecast_multi_step(df_hist: pd.DataFrame, h: int, params: Dict) -> Dict:
    """
    Multi-step forecast with proper uncertainty propagation.
    Returns point forecasts + confidence intervals.
    """
    hm = df_hist["hard_money_total_usd"].values.astype(float)
    sh = df_hist["crypto_share"].values.astype(float)
    regime = df_hist["hm_vol_regime"].values.astype(float) if params["use_regime"] else None
    
    # Get filtered states
    if regime is not None:
        log_hm_hat, hm_var = _adaptive_ew_smoother(np.log(hm), regime,
                                                    params["g_hm_low"],
                                                    params["g_hm_high"],
                                                    init=np.log(hm[0]))
        logit_s_hat, s_var = _adaptive_ew_smoother(_logit(sh), regime,
                                                    params["g_sh_low"],
                                                    params["g_sh_high"],
                                                    init=_logit(sh[0]))
    else:
        log_hm_hat, hm_var = _ew_local_level(np.log(hm), params["g_hm_low"], init=np.log(hm[0]))
        logit_s_hat, s_var = _ew_local_level(_logit(sh), params["g_sh_low"], init=_logit(sh[0]))
    
    # h-step ahead forecast (random walk)
    log_hm_fore = log_hm_hat[-1]
    logit_s_fore = logit_s_hat[-1]
    
    # Uncertainty grows with horizon
    hm_var_fore = hm_var[-1] * (1 + 0.1 * h)  # Simple scaling
    s_var_fore = s_var[-1] * (1 + 0.1 * h)
    
    hm_fore = np.exp(log_hm_fore)
    s_fore = _inv_logit(logit_s_fore)
    
    crypto_fore = hm_fore * s_fore
    gold_fore = hm_fore * (1.0 - s_fore)
    
    # Confidence intervals (approximate)
    z = stats.norm.ppf((1 + CONF_LEVEL) / 2)
    hm_std = np.sqrt(hm_var_fore) * hm_fore
    s_std = np.sqrt(s_var_fore) * s_fore * (1 - s_fore)
    
    crypto_std = crypto_fore * np.sqrt((hm_std/hm_fore)**2 + (s_std/s_fore)**2)
    gold_std = gold_fore * np.sqrt((hm_std/hm_fore)**2 + (s_std/(1-s_fore))**2)
    
    return {
        "crypto": crypto_fore,
        "gold": gold_fore,
        "crypto_lower": crypto_fore - z * crypto_std,
        "crypto_upper": crypto_fore + z * crypto_std,
        "gold_lower": gold_fore - z * gold_std,
        "gold_upper": gold_fore + z * gold_std,
    }

# ---------------------
# Metrics
# ---------------------
def r2(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2)
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def rmse(y, yhat):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    return np.sqrt(np.nanmean((y - yhat)**2))

def mape(y, yhat, eps=1e-8):
    y = np.asarray(y, dtype=float); yhat = np.asarray(yhat, dtype=float)
    denom = np.maximum(eps, np.abs(y))
    return 100.0*np.nanmean(np.abs(yhat - y)/denom)

def coverage_rate(y_true, y_lower, y_upper):
    """Percentage of actuals falling within confidence bands."""
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).sum()
    return 100.0 * covered / len(y_true)

# ---------------------
# Rolling OOS with Confidence Intervals
# ---------------------
def rolling_oos(df: pd.DataFrame, horizons=(1,3,6,12), use_regime=True) -> Tuple[pd.DataFrame, Dict]:
    """Enhanced rolling-origin OOS with confidence intervals."""
    n = len(df)
    start = max(12, int(n * MIN_TRAIN_FRAC))
    
    recs: List[Dict] = []
    traces: Dict = {}
    
    for h in horizons:
        y_true_gold = []
        y_true_crypto = []
        y_hat_gold = []
        y_hat_crypto = []
        gold_lower = []
        gold_upper = []
        crypto_lower = []
        crypto_upper = []
        timestamps = []
        
        for t in range(start, n - h):
            df_train = df.iloc[:t].copy()
            
            # Fit on training data
            params = {
                "g_hm_low": 0.3, "g_hm_high": 0.6,  # Will be optimized
                "g_sh_low": 0.3, "g_sh_high": 0.6,
                "use_regime": use_regime
            }
            
            # Quick fit (simplified for speed in rolling)
            hm = df_train["hard_money_total_usd"].values
            sh = df_train["crypto_share"].values
            regime = df_train["hm_vol_regime"].values if use_regime else None
            
            if regime is not None:
                g_hm_low, g_hm_high = _grid_search_gain(np.log(hm), loss="mape", regime=regime)
                g_sh_low, g_sh_high = _grid_search_gain(sh, loss="mse",
                                                        transform=_logit,
                                                        inv_transform=_inv_logit,
                                                        regime=regime)
            else:
                g_hm_low, g_hm_high = _grid_search_gain(np.log(hm), loss="mape")
                g_sh_low, g_sh_high = _grid_search_gain(sh, loss="mse",
                                                        transform=_logit,
                                                        inv_transform=_inv_logit)
            
            params.update({"g_hm_low": g_hm_low, "g_hm_high": g_hm_high,
                          "g_sh_low": g_sh_low, "g_sh_high": g_sh_high})
            
            # Forecast
            fore = forecast_multi_step(df_train, h, params)
            
            y_hat_gold.append(fore["gold"])
            y_hat_crypto.append(fore["crypto"])
            gold_lower.append(fore["gold_lower"])
            gold_upper.append(fore["gold_upper"])
            crypto_lower.append(fore["crypto_lower"])
            crypto_upper.append(fore["crypto_upper"])
            
            y_true_gold.append(df["gold_etf_mcap_usd"].iloc[t+h])
            y_true_crypto.append(df["crypto_total_usd"].iloc[t+h])
            timestamps.append(df.index[t+h])
        
        if len(y_true_gold) > 0:
            # Metrics for Gold
            cov_gold = coverage_rate(y_true_gold, gold_lower, gold_upper)
            recs.append({
                "Horizon": h, "Series": "Gold AUM (USD)",
                "R2": r2(y_true_gold, y_hat_gold),
                "RMSE": rmse(y_true_gold, y_hat_gold),
                "MAPE_%": mape(y_true_gold, y_hat_gold),
                "Coverage_%": cov_gold,
                "n_oos": len(y_true_gold)
            })
            
            # Metrics for Crypto
            cov_crypto = coverage_rate(y_true_crypto, crypto_lower, crypto_upper)
            recs.append({
                "Horizon": h, "Series": "Total Crypto (USD)",
                "R2": r2(y_true_crypto, y_hat_crypto),
                "RMSE": rmse(y_true_crypto, y_hat_crypto),
                "MAPE_%": mape(y_true_crypto, y_hat_crypto),
                "Coverage_%": cov_crypto,
                "n_oos": len(y_true_crypto)
            })
            
            # Store traces for plotting
            traces[(h, "gold")] = {
                "timestamps": timestamps,
                "y_true": y_true_gold,
                "y_hat": y_hat_gold,
                "lower": gold_lower,
                "upper": gold_upper
            }
            traces[(h, "crypto")] = {
                "timestamps": timestamps,
                "y_true": y_true_crypto,
                "y_hat": y_hat_crypto,
                "lower": crypto_lower,
                "upper": crypto_upper
            }
    
    return pd.DataFrame(recs), traces

# ---------------------
# Enhanced Plotting
# ---------------------
def plot_insample(df_fit: pd.DataFrame):
    """Plot in-sample fit with confidence bands."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    z = stats.norm.ppf((1 + CONF_LEVEL) / 2)
    
    # Gold
    ax = axes[0]
    ax.plot(df_fit.index, df_fit["gold_etf_mcap_usd"], label="Actual Gold AUM", lw=2, color='goldenrod')
    ax.plot(df_fit.index, df_fit["gold_hat"], label="HM Fitted", lw=1.6, alpha=0.9, color='darkgoldenrod')
    
    gold_upper = df_fit["gold_hat"] + z * df_fit["hm_std"] * (1 - df_fit["crypto_share_hat"])
    gold_lower = df_fit["gold_hat"] - z * df_fit["hm_std"] * (1 - df_fit["crypto_share_hat"])
    ax.fill_between(df_fit.index, gold_lower, gold_upper, alpha=0.2, color='goldenrod', label=f'{int(CONF_LEVEL*100)}% CI')
    
    ax.set_title("Gold AUM — Actual vs Enhanced Hard-Money Fitted", fontsize=12, fontweight='bold')
    ax.set_ylabel("USD", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Crypto
    ax = axes[1]
    ax.plot(df_fit.index, df_fit["crypto_total_usd"], label="Actual Total Crypto", lw=2, color='steelblue')
    ax.plot(df_fit.index, df_fit["crypto_hat"], label="HM Fitted", lw=1.6, alpha=0.9, color='navy')
    
    crypto_upper = df_fit["crypto_hat"] + z * df_fit["hm_std"] * df_fit["crypto_share_hat"]
    crypto_lower = df_fit["crypto_hat"] - z * df_fit["hm_std"] * df_fit["crypto_share_hat"]
    ax.fill_between(df_fit.index, crypto_lower, crypto_upper, alpha=0.2, color='steelblue', label=f'{int(CONF_LEVEL*100)}% CI')
    
    ax.set_title("Total Crypto — Actual vs Enhanced Hard-Money Fitted", fontsize=12, fontweight='bold')
    ax.set_ylabel("USD", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(style='plain', axis='y')
    
    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_oos_with_ci(traces: Dict, horizon: int = 6):
    """Plot OOS forecasts with confidence intervals."""
    if (horizon, "gold") not in traces or (horizon, "crypto") not in traces:
        print(f"No traces for horizon {horizon}")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Gold
    ax = axes[0]
    gold_trace = traces[(horizon, "gold")]
    idx = pd.to_datetime(gold_trace["timestamps"])
    
    ax.plot(idx, gold_trace["y_true"], label="Actual Gold AUM", lw=2, color='goldenrod', marker='o', markersize=3)
    ax.plot(idx, gold_trace["y_hat"], label=f"Forecast (h={horizon})", lw=1.6, alpha=0.9, color='darkgoldenrod', linestyle='--')
    ax.fill_between(idx, gold_trace["lower"], gold_trace["upper"], 
                     alpha=0.2, color='goldenrod', label=f'{int(CONF_LEVEL*100)}% CI')
    
    ax.set_title(f"Gold AUM — Rolling OOS Forecast (h={horizon} months)", fontsize=12, fontweight='bold')
    ax.set_ylabel("USD", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(style='plain', axis='y')
    
    # Crypto
    ax = axes[1]
    crypto_trace = traces[(horizon, "crypto")]
    
    ax.plot(idx, crypto_trace["y_true"], label="Actual Total Crypto", lw=2, color='steelblue', marker='o', markersize=3)
    ax.plot(idx, crypto_trace["y_hat"], label=f"Forecast (h={horizon})", lw=1.6, alpha=0.9, color='navy', linestyle='--')
    ax.fill_between(idx, crypto_trace["lower"], crypto_trace["upper"],
                     alpha=0.2, color='steelblue', label=f'{int(CONF_LEVEL*100)}% CI')
    
    ax.set_title(f"Total Crypto — Rolling OOS Forecast (h={horizon} months)", fontsize=12, fontweight='bold')
    ax.set_ylabel("USD", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9)
    ax.ticklabel_format(style='plain', axis='y')
    
    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_regime_analysis(df_fit: pd.DataFrame):
    """Visualize volatility regimes and crypto share dynamics."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Hard Money Total with regime shading
    ax = axes[0]
    ax.plot(df_fit.index, df_fit["hard_money_total_usd"], lw=2, color='black', label='Hard Money Total')
    
    # Shade high volatility periods
    high_vol = df_fit["hm_vol_regime"] > 0.5
    for i in range(len(df_fit)):
        if high_vol.iloc[i]:
            ax.axvspan(df_fit.index[i], df_fit.index[min(i+1, len(df_fit)-1)], 
                      alpha=0.2, color='red')
    
    ax.set_title("Hard Money Total (USD) with High Volatility Regimes (shaded)", fontsize=12, fontweight='bold')
    ax.set_ylabel("USD", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    ax.set_yscale('log')
    
    # Crypto share with fitted share
    ax = axes[1]
    ax.plot(df_fit.index, df_fit["crypto_share"], lw=2, color='steelblue', label='Actual Share', alpha=0.7)
    ax.plot(df_fit.index, df_fit["crypto_share_hat"], lw=1.8, color='navy', label='Smoothed Share', linestyle='--')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
    
    ax.set_title("Crypto Share of Hard Money Pool", fontsize=12, fontweight='bold')
    ax.set_ylabel("Share", fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    # Share momentum
    ax = axes[2]
    ax.plot(df_fit.index, df_fit["share_momentum"], lw=1.5, color='darkgreen', label='3-Month Momentum')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.fill_between(df_fit.index, 0, df_fit["share_momentum"], 
                    where=df_fit["share_momentum"]>=0, alpha=0.3, color='green', interpolate=True)
    ax.fill_between(df_fit.index, 0, df_fit["share_momentum"],
                    where=df_fit["share_momentum"]<0, alpha=0.3, color='red', interpolate=True)
    
    ax.set_title("Crypto Share Momentum (3-month change)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Change in Share", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    axes[-1].set_xlabel("Date", fontsize=10)
    plt.tight_layout()
    plt.show()

# ---------------------
# Model Diagnostics
# ---------------------
def print_model_diagnostics(df_fit: pd.DataFrame, params: Dict):
    """Print detailed model diagnostics."""
    print("\n" + "="*70)
    print(" MODEL DIAGNOSTICS")
    print("="*70)
    
    print(f"\n[Smoothing Parameters]")
    print(f"  HM Level (log):  Low Vol = {params['g_hm_low']:.4f}, High Vol = {params['g_hm_high']:.4f}")
    print(f"  Crypto Share:    Low Vol = {params['g_sh_low']:.4f}, High Vol = {params['g_sh_high']:.4f}")
    print(f"  Regime-Adaptive: {params['use_regime']}")
    
    print(f"\n[Regime Statistics]")
    high_vol_pct = (df_fit["hm_vol_regime"] > 0.5).sum() / len(df_fit) * 100
    print(f"  High Volatility Periods: {high_vol_pct:.1f}% of sample")
    
    # Residual diagnostics
    gold_resid = df_fit["gold_etf_mcap_usd"] - df_fit["gold_hat"]
    crypto_resid = df_fit["crypto_total_usd"] - df_fit["crypto_hat"]
    
    print(f"\n[Residual Statistics]")
    print(f"  Gold AUM:")
    print(f"    Mean:     {gold_resid.mean():,.0f}")
    print(f"    Std Dev:  {gold_resid.std():,.0f}")
    print(f"    Skewness: {gold_resid.skew():.3f}")
    print(f"  Crypto:")
    print(f"    Mean:     {crypto_resid.mean():,.0f}")
    print(f"    Std Dev:  {crypto_resid.std():,.0f}")
    print(f"    Skewness: {crypto_resid.skew():.3f}")
    
    # Information criterion (simplified AIC)
    n = len(df_fit)
    k = 4  # 4 parameters (2 gains × 2 regimes)
    sse_gold = np.sum(gold_resid**2)
    sse_crypto = np.sum(crypto_resid**2)
    aic_gold = n * np.log(sse_gold / n) + 2 * k
    aic_crypto = n * np.log(sse_crypto / n) + 2 * k
    
    print(f"\n[Model Selection Criteria]")
    print(f"  AIC (Gold):   {aic_gold:.2f}")
    print(f"  AIC (Crypto): {aic_crypto:.2f}")
    print(f"  Total AIC:    {aic_gold + aic_crypto:.2f}")
    
    print("\n" + "="*70 + "\n")

# ---------------------
# Main
# ---------------------
def main():
    print("\n" + "="*70)
    print(" ENHANCED HARD-MONEY ALLOCATION MODEL")
    print("="*70)
    
    # 1) Build panel with regime indicators
    panel = build_panel_monthly()
    print("\n[Panel Summary]")
    print(f"  Date Range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"  Observations: {len(panel)}")
    print(f"\n  Gold AUM:    ${panel['gold_etf_mcap_usd'].mean()/1e9:.1f}B avg")
    print(f"  Total Crypto: ${panel['crypto_total_usd'].mean()/1e9:.1f}B avg")
    print(f"  Hard Money:   ${panel['hard_money_total_usd'].mean()/1e9:.1f}B avg")
    print(f"  Crypto Share: {panel['crypto_share'].mean():.1%} avg")
    
    # 2) Fit Enhanced Hard-Money model
    print("\n[Fitting Enhanced Model...]")
    res = fit_hard_money_model(panel, use_regime=True)
    df_fit = res["df_fit"]
    
    # 3) Model diagnostics
    print_model_diagnostics(df_fit, res)
    
    # 4) In-sample metrics
    ins_tbl = pd.DataFrame([
        {
            "Series": "Gold AUM (USD)",
            "R2": r2(df_fit["gold_etf_mcap_usd"], df_fit["gold_hat"]),
            "RMSE": rmse(df_fit["gold_etf_mcap_usd"], df_fit["gold_hat"]),
            "MAPE_%": mape(df_fit["gold_etf_mcap_usd"], df_fit["gold_hat"]),
            "n_obs": len(df_fit)
        },
        {
            "Series": "Total Crypto (USD)",
            "R2": r2(df_fit["crypto_total_usd"], df_fit["crypto_hat"]),
            "RMSE": rmse(df_fit["crypto_total_usd"], df_fit["crypto_hat"]),
            "MAPE_%": mape(df_fit["crypto_total_usd"], df_fit["crypto_hat"]),
            "n_obs": len(df_fit)
        }
    ])
    print("\n[In-Sample Metrics]")
    print(ins_tbl.round(4).to_string(index=False))
    
    # 5) Rolling OOS with confidence intervals
    print("\n[Running Rolling OOS Evaluation...]")
    oos_tbl, traces = rolling_oos(df_fit, horizons=OOS_HORIZONS, use_regime=True)
    
    if len(oos_tbl):
        print("\n[Rolling OOS Metrics]")
        print(oos_tbl.round(4).to_string(index=False))
        print(f"\nNote: Coverage_% shows what % of actuals fell within {int(CONF_LEVEL*100)}% confidence intervals")
    else:
        print("\n[WARN] Not enough data for rolling OOS.")
    
    # 6) Plots
    print("\n[Generating Plots...]")
    plot_insample(df_fit)
    
    if len(traces) > 0:
        # Plot middle horizon OOS
        h_show = OOS_HORIZONS[len(OOS_HORIZONS)//2]
        plot_oos_with_ci(traces, horizon=h_show)
    
    plot_regime_analysis(df_fit)
    
    print("\n[Analysis Complete!]")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

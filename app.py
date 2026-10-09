# -*- coding: utf-8 -*-
"""
NEXUS Terminal Master Edition — النسخة الاحترافية المكتملة والمطورة (Multi-TF Scanner)
========================================================================================
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import ccxt
except ImportError:
    ccxt = None
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

# ──────────────────────────────────────────────────────────────────────────────
# الثوابت والتنسيق الجمالي
# ──────────────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_state.json")
UP, DOWN, GOLD, MUTED = "#0ecb81", "#f6465d", "#fcd535", "#848e9c"
BG, PANEL, LINE = "#0b0e11", "#161a1e", "#2b3139"
SPLITS = (0.4, 0.3, 0.3)
TF_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
HTF_MAP = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h", "30m": "4h", "1h": "4h", "4h": "1d", "1d": "1w"}
STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "AEUR", "USD1", "XUSD", "PYUSD", "USDE", "UST", "EURI", "RLUSD"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="st-"] { font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif; }
.stApp { background: #0b0e11; color: #eaecef; }
.block-container { padding-top: 1.1rem; max-width: 1550px; }
footer, #MainMenu { visibility: hidden; }
[data-testid="stSidebar"] { background: #0f1317; border-left: 1px solid #2b3139; }
[data-testid="stSidebar"] * { color: #eaecef; }
h1, h2, h3, h4 { color: #eaecef; letter-spacing: 0; }
input, textarea, [data-baseweb="select"] > div { background: #161a1e !important; color: #eaecef !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #2b3139; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; color: #848e9c; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #fcd535 !important; }
[data-testid="stMetric"] { background: #161a1e; border: 1px solid #2b3139; border-radius: 6px; padding: 10px 14px; }
[data-testid="stMetricLabel"] { color: #848e9c; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-size: 1.35rem; }
.brand { font-size: 1.55rem; font-weight: 700; color: #fcd535; line-height: 1; padding: 4px 0 10px; }
.brand span { color: #848e9c; font-size: .8rem; font-weight: 500; margin-right: 8px; }
.sig { direction: rtl; text-align: right; background: #161a1e; border: 1px solid #2b3139; border-top: 3px solid var(--c);
       border-radius: 6px; padding: 12px 14px; margin-bottom: 12px; }
.sig-top { display: flex; justify-content: space-between; align-items: center; }
.sym { font-size: 1.2rem; font-weight: 700; }
.sym small { color: #848e9c; font-weight: 500; font-size: .75rem; margin-right: 3px; }
.badge { padding: 3px 10px; border-radius: 4px; font-weight: 700; font-size: .85rem; }
.tf-badge { background: #2b3139; color: #fcd535; padding: 2px 8px; border-radius: 4px; font-size: .75rem; font-weight: 600; }
.sig-kind { margin-top: 8px; font-weight: 600; color: #eaecef; font-size: .9rem; }
.sig-strat { color: #848e9c; font-size: .8rem; margin-bottom: 8px; }
.bar { height: 5px; background: #2b3139; border-radius: 3px; overflow: hidden; }
.bar i { display: block; height: 100%; }
.sig-conf { color: #848e9c; font-size: .78rem; margin: 4px 0 8px; }
.lv { display: flex; justify-content: space-between; font-size: .82rem; padding: 2px 0; font-variant-numeric: tabular-nums; }
.lv span { color: #848e9c; }
.note { direction: rtl; text-align: right; color: #848e9c; font-size: .85rem; }
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# أدوات الفحص المتقدمة وموّرد البيانات
# ──────────────────────────────────────────────────────────────────────────────
def fp(x) -> str:
    try: x = float(x)
    except (TypeError, ValueError): return "-"
    if not np.isfinite(x): return "-"
    a = abs(x)
    if a >= 1000: return f"{x:,.2f}"
    if a >= 1: return f"{x:,.4f}"
    if a >= 0.01: return f"{x:.5f}"
    return f"{x:.8f}"

def fv(x) -> str:
    x = float(x or 0)
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= d: return f"{x / d:.2f}{s}"
    return f"{x:.0f}"

def show_df(df, **kw):
    try: return st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception: return st.dataframe(df, use_container_width=True, hide_index=True, **kw)

def show_plot(fig, **kw):
    try: return st.plotly_chart(fig, width="stretch", **kw)
    except Exception: return st.plotly_chart(fig, use_container_width=True, **kw)

def ensure_theme():
    try:
        d = os.path.join(os.getcwd(), ".streamlit")
        p = os.path.join(d, "config.toml")
        if not os.path.exists(p):
            os.makedirs(d, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write('[theme]\nbase="dark"\nprimaryColor="#fcd535"\nbackgroundColor="#0b0e11"\n'
                        'secondaryBackgroundColor="#161a1e"\ntextColor="#eaecef"\n')
    except Exception: pass

DEMO_BASES = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "TRX", "ATOM", "NEAR",
              "APT", "ARB", "OP", "INJ", "SUI", "FIL", "UNI", "AAVE", "ETC", "XLM", "HBAR", "PEPE", "SEI", "TIA",
              "RUNE", "TON", "SHIB", "BCH", "ICP", "IMX", "FET", "RNDR", "WLD", "JUP", "ENA", "ORDI", "ORCA", "DRIFT", "ACE", "XAU"]

def synth_price(sym: str) -> float:
    h = zlib.crc32(sym.encode())
    base = 10 ** (np.random.default_rng(h).uniform(-1.8, 4.6))
    ph, t = (h % 628) / 100, time.time()
    return float(base * (1 + 0.012 * math.sin(t / 90 + ph) + 0.002 * math.sin(t / 7 + ph * 2)))

def synth_ohlcv(sym: str, tf: str, n: int = 300) -> pd.DataFrame:
    sec = TF_SEC.get(tf, 900)
    rng = np.random.default_rng(zlib.crc32(sym.encode()) % (2 ** 32) + sec)
    k = math.sqrt(sec / 900)
    t = np.arange(n)
    ret = rng.normal(0, 0.0045 * k, n) + 0.0035 * k * np.sin(t / rng.uniform(14, 40)) + 0.002 * k * np.sin(t / 7)
    close = np.exp(np.cumsum(ret))
    close = close * (synth_price(sym) / close[-1])
    now = int(time.time() // sec * sec)
    open_ = np.r_[close[0], close[:-1]]
    sp = np.abs(rng.normal(0, 0.0022 * k, n))
    high = np.maximum(open_, close) * (1 + sp)
    low = np.minimum(open_, close) * (1 - sp)
    vol = rng.lognormal(10, 0.5, n) * (1 + 3 * (rng.random(n) > 0.94))
    ts = (now - sec * np.arange(n - 1, -1, -1)) * 1000
    df = pd.DataFrame({"Timestamp": ts, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    return df

class Hub:
    def __init__(self, demo: bool = False):
        self.demo = demo or ccxt is None
        self.error = ""
        self._c: dict = {}
        self._lock = threading.Lock()
        self.ex = {}
        if not self.demo:
            self.ex = {"spot": ccxt.binance({"enableRateLimit": True}),
                       "futures": ccxt.binanceusdm({"enableRateLimit": True})}

    def _cached(self, key, ttl, fn):
        now = time.time()
        with self._lock:
            hit = self._c.get(key)
            if hit and now - hit[0] < ttl: return hit[1]
        val = fn()
        with self._lock: self._c[key] = (now, val)
        return val

    @staticmethod
    def sym(mk: str, base: str) -> str:
        return f"{base}/USDT:USDT" if mk == "futures" else f"{base}/USDT"

    def tickers(self, mk: str) -> dict:
        def f():
            if self.demo:
                out = {}
                for b in DEMO_BASES:
                    s = self.sym(mk, b)
                    d = synth_ohlcv(s, "15m", 120)
                    last = float(d.Close.iloc[-1])
                    r = np.random.default_rng(zlib.crc32(b.encode()))
                    out[s] = {"symbol": s, "last": last, "percentage": (last / float(d.Close.iloc[-96]) - 1) * 100,
                              "quoteVolume": float(r.uniform(2e5, 9e8)), "high": float(d.High.tail(96).max()),
                              "low": float(d.Low.tail(96).min())}
                return out
            return self.ex[mk].fetch_tickers()
        try: return self._cached(("t", mk), 20, f)
        except Exception as e:
            self.error = str(e)[:300]
            return {}

    def universe(self, mk: str, min_vol: float, sort_asc: bool = False) -> list:
        tk = self.tickers(mk)
        suffix = "/USDT:USDT" if mk == "futures" else "/USDT"
        out = []
        for s, t in tk.items():
            if not s.endswith(suffix): continue
            base = s.split("/")[0]
            if base in STABLES or base.endswith(("BULL", "BEAR")): continue
            qv = t.get("quoteVolume") or 0
            if qv >= min_vol: out.append((s, float(qv)))
        out.sort(key=lambda x: x[1] if sort_asc else -x[1])
        return out

    def ohlcv(self, mk: str, sym: str, tf: str, limit: int = 300, ttl: int = 15):
        def f():
            if self.demo: return synth_ohlcv(sym, tf, limit)
            raw = self.ex[mk].fetch_ohlcv(sym, tf, limit=limit)
            df = pd.DataFrame(raw, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
            return df
        try: return self._cached(("o", mk, sym, tf, limit), ttl, f).copy()
        except Exception as e:
            self.error = str(e)[:300]
            return None

    def prices(self, mk: str, syms: list) -> dict:
        if not syms: return {}
        try:
            if self.demo: return {s: synth_price(s) for s in syms}
            t = self.ex[mk].fetch_tickers(syms)
            return {s: float(v["last"]) for s, v in t.items() if v.get("last")}
        except Exception:
            out = {}
            for s in syms:
                try: out[s] = float(self.ex[mk].fetch_ticker(s)["last"])
                except Exception: pass
            return out

@st.cache_resource(show_spinner=False)
def get_hub(demo: bool) -> Hub: return Hub(demo)

# ──────────────────────────────────────────────────────────────────────────────
# حساب المؤشرات + القاع المحمي + خطوط الاتجاه المصلحة
# ──────────────────────────────────────────────────────────────────────────────
def calc_volume_profile(df: pd.DataFrame, bins: int = 30):
    if df is None or len(df) < 10: return None, None, None, None
    p_min, p_max = df["Low"].min(), df["High"].max()
    if p_min == p_max: return None, None, None, None
    price_bins = np.linspace(p_min, p_max, bins + 1)
    vol_profile = np.zeros(bins)
    
    for _, row in df.iterrows():
        h, l, v = row["High"], row["Low"], row["Volume"]
        if h == l:
            idx = min(int((h - p_min) / (p_max - p_min) * bins), bins - 1)
            vol_profile[idx] += v
        else:
            mask = (price_bins[:-1] <= h) & (price_bins[1:] >= l)
            count = np.sum(mask)
            if count > 0: vol_profile[mask] += v / count

    poc_idx = np.argmax(vol_profile)
    poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

    total_vol = np.sum(vol_profile)
    target_vol = total_vol * 0.70
    sorted_indices = np.argsort(vol_profile)[::-1]
    
    accum_vol, va_indices = 0, []
    for idx in sorted_indices:
        accum_vol += vol_profile[idx]
        va_indices.append(idx)
        if accum_vol >= target_vol: break
            
    val_price = price_bins[min(va_indices)]
    vah_price = price_bins[max(va_indices) + 1]

    return price_bins, vol_profile, poc_price, (vah_price, val_price)

def nw_envelope(close: pd.Series, h: float = 8.0, win: int = 60, mult: float = 2.2):
    y = close.values.astype(float)
    n = len(y)
    idx = np.arange(n)
    diff = idx[:, None] - idx[None, :]
    w = np.exp(-0.5 * (diff / h) ** 2)
    w = np.where((diff >= 0) & (diff <= win), w, 0.0)
    w /= w.sum(axis=1, keepdims=True)
    yhat = w @ y
    err = pd.Series(np.abs(y - yhat)).rolling(win, min_periods=10).mean().values
    return yhat, yhat + mult * err, yhat - mult * err

def find_swings(df: pd.DataFrame, window: int = 5):
    highs, lows = df['High'].values, df['Low'].values
    n = len(df)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(window, n - window):
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            sh[i] = highs[i]
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            sl[i] = lows[i]
    df['swing_high'] = sh
    df['swing_low'] = sl
    return df

def calculate_accurate_trendlines(df: pd.DataFrame, window: int = 5):
    d = find_swings(df.copy(), window=window)
    sh_idx = np.where(~np.isnan(d['swing_high']))[0]
    sl_idx = np.where(~np.isnan(d['swing_low']))[0]
    
    res_line = np.full(len(df), np.nan)
    sup_line = np.full(len(df), np.nan)
    
    if len(sh_idx) >= 2:
        i1, i2 = sh_idx[-2], sh_idx[-1]
        y1, y2 = d['swing_high'].iloc[i1], d['swing_high'].iloc[i2]
        slope = (y2 - y1) / max((i2 - i1), 1)
        for x in range(i1, len(df)):
            res_line[x] = y1 + slope * (x - i1)
            
    if len(sl_idx) >= 2:
        i1, i2 = sl_idx[-2], sl_idx[-1]
        y1, y2 = d['swing_low'].iloc[i1], d['swing_low'].iloc[i2]
        slope = (y2 - y1) / max((i2 - i1), 1)
        for x in range(i1, len(df)):
            sup_line[x] = y1 + slope * (x - i1)
            
    return res_line, sup_line

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index(drop=True).copy()
    o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume
    for n in (9, 21, 50, 200): d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    d["RSI"] = rsi.where(dn != 0, 100.0).fillna(50.0)
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD"], d["MACDs"] = m, m.ewm(span=9, adjust=False).mean()
    d["MACDh"] = d.MACD - d.MACDs
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_M"], d["BB_U"], d["BB_L"] = ma, ma + 2 * sd, ma - 2 * sd
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["ATR"] = atr
    tp = (h + l + c) / 3
    day = d.Timestamp.dt.date
    d["VWAP"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=d.index)
    mdm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=d.index)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    d["VolMA"] = v.rolling(20).mean()
    d["VolR"] = (v / d.VolMA.replace(0, np.nan)).fillna(1.0)
    rg = (h - l).replace(0, np.nan)
    d["LowWick"] = ((np.minimum(o, c) - l) / rg).fillna(0)
    d["UpWick"] = ((h - np.maximum(o, c)) / rg).fillna(0)
    d["NW"], d["NW_U"], d["NW_L"] = nw_envelope(c)

    # ICT & SMC Elements
    d["FVG_Bull"] = l > h.shift(2)
    d["FVG_Bear"] = h < l.shift(2)
    d["FVG_Bull_Top"] = np.where(d["FVG_Bull"], l, np.nan)
    d["FVG_Bull_Bot"] = np.where(d["FVG_Bull"], h.shift(2), np.nan)
    d["FVG_Bull_CE"] = (d["FVG_Bull_Top"] + d["FVG_Bull_Bot"]) / 2

    d["BSL"] = h.shift(1).rolling(20).max()
    d["SSL"] = l.shift(1).rolling(20).min()
    d["BOS_Bull"] = c > d["BSL"]
    d["BOS_Bear"] = c < d["SSL"]

    # Protected Low / High المنقحة
    prot_lows = np.full(len(d), np.nan)
    prot_highs = np.full(len(d), np.nan)
    last_pl, last_ph = np.nan, np.nan
    for idx in range(20, len(d)):
        if d["BOS_Bull"].iloc[idx]:
            last_pl = l.iloc[idx-15:idx].min()
        if d["BOS_Bear"].iloc[idx]:
            last_ph = h.iloc[idx-15:idx].max()
        prot_lows[idx] = last_pl
        prot_highs[idx] = last_ph

    d["Protected_Low"] = pd.Series(prot_lows, index=d.index).ffill()
    d["Protected_High"] = pd.Series(prot_highs, index=d.index).ffill()
    d["Is_Above_Protected_Low"] = c > d["Protected_Low"]
    d["Is_Below_Protected_High"] = c < d["Protected_High"]

    range_high = h.rolling(50).max()
    range_low = l.rolling(50).min()
    eq_level = (range_high + range_low) / 2
    d["Is_Discount"] = c < eq_level
    d["Is_Premium"] = c > eq_level

    d["Upper_Wick_1H"] = h.shift(1).rolling(12).max()
    d["Lower_Wick_1H"] = l.shift(1).rolling(12).min()
    d["Upper_Wick_4H"] = h.shift(1).rolling(16).max()
    d["Lower_Wick_4H"] = l.shift(1).rolling(16).min()
    d["Upper_Wick_1D"] = h.shift(1).rolling(6).max()
    d["Lower_Wick_1D"] = l.shift(1).rolling(6).min()

    _, _, poc_p, _ = calc_volume_profile(d.tail(100), bins=20)
    d["POC_Level"] = poc_p if poc_p is not None else np.nan

    res_line, sup_line = calculate_accurate_trendlines(d)
    d["Trend_Res"] = res_line
    d["Trend_Sup"] = sup_line

    return d

# ──────────────────────────────────────────────────────────────────────────────
# الاستراتيجيات المتكاملة
# ──────────────────────────────────────────────────────────────────────────────
def _b(x: pd.Series) -> pd.Series: return x.fillna(False).astype(bool)
def _recent(x: pd.Series, n: int) -> pd.Series: return x.astype(float).rolling(n, min_periods=1).max().fillna(0) > 0

def s_trendline_breakout(d: pd.DataFrame):
    prev_c, c = d.Close.shift(1), d.Close
    res, sup = d.Trend_Res, d.Trend_Sup
    lo = (prev_c <= res) & (c > res) & (d.VolR > 1.2) & (d.RSI > 50)
    sh = (prev_c >= sup) & (c < sup) & (d.VolR > 1.2) & (d.RSI < 50)
    score = 68 + (d.VolR > 1.8) * 10
    return _b(lo), _b(sh), score, score

def s_protected_low(d: pd.DataFrame):
    near_prot_low = (d.Low <= d.Protected_Low * 1.008) & (d.Close >= d.Protected_Low)
    lo = near_prot_low & d.Is_Above_Protected_Low & (d.Close > d.Open) & d.Is_Discount
    sh = (d.High >= d.Protected_High * 0.992) & d.Is_Below_Protected_High & (d.Close < d.Open) & d.Is_Premium
    score = 75 + (d.VolR > 1.3) * 10 + (d.RSI < 45) * 8
    return _b(lo), _b(sh), score, score

def s_scalp(d):
    cu = (d.EMA9 > d.EMA21) & (d.EMA9.shift(1) <= d.EMA21.shift(1))
    cd = (d.EMA9 < d.EMA21) & (d.EMA9.shift(1) >= d.EMA21.shift(1))
    lo = _recent(cu, 4) & (d.EMA9 > d.EMA21) & (d.Close > d.VWAP) & d.RSI.between(50, 72) & (d.MACDh > 0)
    sh = _recent(cd, 4) & (d.EMA9 < d.EMA21) & (d.Close < d.VWAP) & d.RSI.between(28, 50) & (d.MACDh < 0)
    base = 58 + (d.VolR > 2) * 8 + (d.ADX > 22) * 8
    return _b(lo), _b(sh), base, base

def s_bottom(d):
    ext = (d.RSI < 35) | (d.Low <= d.BB_L) | (d.Close < d.NW_L)
    lo = _b(ext & (d.LowWick > 0.4) & (d.Close > d.Open) & d.Is_Discount)
    sc = 55 + (d.RSI < 30) * 10 + (d.VolR > 1.5) * 8
    return lo, _b(d.Close < -1), sc, sc * 0

def s_top(d):
    ext = (d.RSI > 65) | (d.High >= d.BB_U) | (d.Close > d.NW_U)
    sh = _b(ext & (d.UpWick > 0.4) & (d.Close < d.Open) & d.Is_Premium)
    sc = 55 + (d.RSI > 70) * 10 + (d.VolR > 1.5) * 8
    return _b(d.Close < -1), sh, sc * 0, sc

def s_hunter(d):
    swl = _b((d.Low < d.SSL) & (d.Close > d.SSL) & (d.LowWick > 0.35))
    swh = _b((d.High > d.BSL) & (d.Close < d.BSL) & (d.UpWick > 0.35))
    lo = swl.shift(1, fill_value=False) & (d.Close > d.High.shift(1)) & d.Is_Discount
    sh = swh.shift(1, fill_value=False) & (d.Close < d.Low.shift(1)) & d.Is_Premium
    return _b(lo), _b(sh), 65, 65

def s_ict_short_sweep(d: pd.DataFrame):
    bsl_50 = d.High.shift(1).rolling(50).max()
    swing_low_20 = d.Low.shift(1).rolling(20).min()
    wave_range = (bsl_50 - swing_low_20).clip(lower=0.0001)
    projected_short_entry = bsl_50 + (wave_range * 0.272)
    sweep_condition = (d.High >= bsl_50) | (d.High >= projected_short_entry * 0.99)
    overbought_condition = (d.RSI > 75) | ((d.RSI > 70) & (d.VolR > 1.8))
    short_signal = sweep_condition & overbought_condition & d.Is_Premium
    score = 65 + (d.RSI > 80) * 15 + (d.VolR > 2.0) * 10
    return pd.Series(False, index=d.index), _b(short_signal), score * 0, score

def s_smc(d):
    fvg_bull_active = d.FVG_Bull.ffill(limit=10)
    lo = d.BOS_Bull.rolling(15).max().astype(bool) & fvg_bull_active & (d.Low <= d.FVG_Bull_CE.ffill(limit=10)) & d.Is_Above_Protected_Low
    sh = d.BOS_Bear.rolling(15).max().astype(bool) & d.FVG_Bear.ffill(limit=10) & d.Is_Below_Protected_High
    score = 65 + (d.VolR > 1.3) * 10
    return _b(lo), _b(sh), score, score

def s_wick_scalp(d):
    lo = (d.Close > d.Upper_Wick_1H) & (d.RSI > 50) & (d.VolR > 1.2)
    sh = (d.Close < d.Lower_Wick_1H) & (d.RSI < 50) & (d.VolR > 1.2)
    score = 60 + (d.VolR > 1.8) * 10
    return _b(lo), _b(sh), score, score

def s_wick_breakout(d):
    lo = (d.Close > d.Upper_Wick_4H) & (d.RSI > 50) & (d.VolR > 1.0)
    sh = (d.Close < d.Lower_Wick_4H) & (d.RSI < 50) & (d.VolR > 1.0)
    score = 63 + (d.VolR > 1.5) * 10
    return _b(lo), _b(sh), score, score

def s_wick_swing(d):
    lo = (d.Close > d.Upper_Wick_1D) & (d.RSI > 52) & (d.VolR > 1.1)
    sh = (d.Close < d.Lower_Wick_1D) & (d.RSI < 48) & (d.VolR > 1.1)
    score = 68 + (d.VolR > 1.5) * 10
    return _b(lo), _b(sh), score, score

def s_poc_rebound(d: pd.DataFrame):
    poc = d.POC_Level.ffill()
    near_poc = (d.Close - poc).abs() < (0.6 * d.ATR)
    lo = near_poc & (d.Close > poc) & (d.RSI > 50)
    sh = near_poc & (d.Close < poc) & (d.RSI < 50)
    score = 65 + (d.VolR > 1.5) * 10
    return _b(lo), _b(sh), score, score

def s_wyckoff(d):
    rl, rh = d.Low.shift(1).rolling(50).min(), d.High.shift(1).rolling(50).max()
    spring = (d.Low < rl) & (d.Close > rl) & (d.VolR > 1.5)
    thrust = (d.High > rh) & (d.Close < rh) & (d.VolR > 1.5)
    return _b(spring), _b(thrust), 70, 70

def s_break(d):
    dh, dl = d.High.shift(1).rolling(20).max(), d.Low.shift(1).rolling(20).min()
    lo = (d.Close > dh) & (d.ADX > 22) & (d.VolR > 1.3)
    sh = (d.Close < dl) & (d.ADX > 22) & (d.VolR > 1.3)
    return _b(lo), _b(sh), 60, 60

@dataclass(frozen=True)
class Strat:
    key: str
    label: str
    kind: str
    fn: Callable
    sl: float
    tps: tuple

STRATS = {s.key: s for s in [
    Strat("trendline_break", "📐 كسر خط الاتجاه والمثلثات", "trend", s_trendline_breakout, 1.5, (2.0, 3.8, 5.5)),
    Strat("prot_low", "🛡 الارتداد من القاع المحمي", "reversal", s_protected_low, 1.2, (2.0, 3.5, 5.5)),
    Strat("scalp", "⚡ مضاربة سريعة EMA/VWAP", "scalp", s_scalp, 1.0, (1.0, 1.8, 3.0)),
    Strat("wick_scalp", "⚡ اختراق ذيل 1H (5m)", "scalp", s_wick_scalp, 1.2, (1.5, 2.

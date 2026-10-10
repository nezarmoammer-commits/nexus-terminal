# -*- coding: utf-8 -*-
"""
NEXUS Terminal Master Edition — النسخة الاحترافية المكتملة والمطورة (Multi-TF Scanner)
مع نظام ذاكرة صفقات المحترفين وتضمين الاستراتيجيات الجديدة المستخرجة منها
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
# الثوابت والتنسيق الجمالي وملفات الذاكرة
# ──────────────────────────────────────────────────────────────────────────────
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus_state.json")
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pro_trades_memory.json")

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
# دوال الذاكرة وإدارة صفقات المحترفين المحفوظة
# ──────────────────────────────────────────────────────────────────────────────
def load_memory() -> list:
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list): return data
    except Exception: pass
    return []

def save_memory_trade(trade_item: dict):
    mem = load_memory()
    mem.append(trade_item)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2, default=float)
    except Exception as e:
        st.error(f"خطأ أثناء حفظ الصفقة في الذاكرة: {e}")

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

    def universe(self, mk: str, min_vol: float, sort_asc: bool = False, **kwargs) -> list:
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

def get_hub(demo: bool) -> Hub:
    if "hub_instance" not in st.session_state or st.session_state.hub_instance.demo != demo:
        st.session_state.hub_instance = Hub(demo)
    return st.session_state.hub_instance

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

    # Protected Low / High
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
# الاستراتيجيات المتكاملة (شاملة استراتيجيات المحترف المستخرجة من الذاكرة)
# ──────────────────────────────────────────────────────────────────────────────
def _b(x: pd.Series) -> pd.Series: return x.fillna(False).astype(bool)
def _recent(x: pd.Series, n: int) -> pd.Series: return x.astype(float).rolling(n, min_periods=1).max().fillna(0) > 0

def s_trendline_breakout(d: pd.DataFrame):
    prev_c, c = d.Close.shift(1), d.Close
    res, sup = d.Trend_Res, d.Trend_Sup
    lo = (prev_c <= res) & (c > res) & (d.VolR > 1.2) & (d.RSI > 50)
    sh = (prev_c >= sup) & (c < sup) & (d.VolR > 1.2) & (d.RSI < 50)
    score = pd.Series(68 + (d.VolR > 1.8) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_protected_low(d: pd.DataFrame):
    near_prot_low = (d.Low <= d.Protected_Low * 1.008) & (d.Close >= d.Protected_Low)
    lo = near_prot_low & d.Is_Above_Protected_Low & (d.Close > d.Open) & d.Is_Discount
    sh = (d.High >= d.Protected_High * 0.992) & d.Is_Below_Protected_High & (d.Close < d.Open) & d.Is_Premium
    score = pd.Series(75 + (d.VolR > 1.3) * 10 + (d.RSI < 45) * 8, index=d.index)
    return _b(lo), _b(sh), score, score

def s_scalp(d):
    cu = (d.EMA9 > d.EMA21) & (d.EMA9.shift(1) <= d.EMA21.shift(1))
    cd = (d.EMA9 < d.EMA21) & (d.EMA9.shift(1) >= d.EMA21.shift(1))
    lo = _recent(cu, 4) & (d.EMA9 > d.EMA21) & (d.Close > d.VWAP) & d.RSI.between(50, 72) & (d.MACDh > 0)
    sh = _recent(cd, 4) & (d.EMA9 < d.EMA21) & (d.Close < d.VWAP) & d.RSI.between(28, 50) & (d.MACDh < 0)
    base = pd.Series(58 + (d.VolR > 2) * 8 + (d.ADX > 22) * 8, index=d.index)
    return _b(lo), _b(sh), base, base

def s_bottom(d):
    ext = (d.RSI < 35) | (d.Low <= d.BB_L) | (d.Close < d.NW_L)
    lo = _b(ext & (d.LowWick > 0.4) & (d.Close > d.Open) & d.Is_Discount)
    sc = pd.Series(55 + (d.RSI < 30) * 10 + (d.VolR > 1.5) * 8, index=d.index)
    return lo, _b(d.Close < -1), sc, sc * 0

def s_top(d):
    ext = (d.RSI > 65) | (d.High >= d.BB_U) | (d.Close > d.NW_U)
    sh = _b(ext & (d.UpWick > 0.4) & (d.Close < d.Open) & d.Is_Premium)
    sc = pd.Series(55 + (d.RSI > 70) * 10 + (d.VolR > 1.5) * 8, index=d.index)
    return _b(d.Close < -1), sh, sc * 0, sc

def s_hunter(d):
    swl = _b((d.Low < d.SSL) & (d.Close > d.SSL) & (d.LowWick > 0.35))
    swh = _b((d.High > d.BSL) & (d.Close < d.BSL) & (d.UpWick > 0.35))
    lo = swl.shift(1, fill_value=False) & (d.Close > d.High.shift(1)) & d.Is_Discount
    sh = swh.shift(1, fill_value=False) & (d.Close < d.Low.shift(1)) & d.Is_Premium
    sc = pd.Series(65, index=d.index)
    return _b(lo), _b(sh), sc, sc

def s_ict_short_sweep(d: pd.DataFrame):
    bsl_50 = d.High.shift(1).rolling(50).max()
    swing_low_20 = d.Low.shift(1).rolling(20).min()
    wave_range = (bsl_50 - swing_low_20).clip(lower=0.0001)
    projected_short_entry = bsl_50 + (wave_range * 0.272)
    sweep_condition = (d.High >= bsl_50) | (d.High >= projected_short_entry * 0.99)
    overbought_condition = (d.RSI > 75) | ((d.RSI > 70) & (d.VolR > 1.8))
    short_signal = sweep_condition & overbought_condition & d.Is_Premium
    score = pd.Series(65 + (d.RSI > 80) * 15 + (d.VolR > 2.0) * 10, index=d.index)
    return pd.Series(False, index=d.index), _b(short_signal), score * 0, score

def s_smc(d):
    fvg_bull_active = d.FVG_Bull.ffill(limit=10)
    lo = d.BOS_Bull.rolling(15).max().astype(bool) & fvg_bull_active & (d.Low <= d.FVG_Bull_CE.ffill(limit=10)) & d.Is_Above_Protected_Low
    sh = d.BOS_Bear.rolling(15).max().astype(bool) & d.FVG_Bear.ffill(limit=10) & d.Is_Below_Protected_High
    score = pd.Series(65 + (d.VolR > 1.3) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_wick_scalp(d):
    lo = (d.Close > d.Upper_Wick_1H) & (d.RSI > 50) & (d.VolR > 1.2)
    sh = (d.Close < d.Lower_Wick_1H) & (d.RSI < 50) & (d.VolR > 1.2)
    score = pd.Series(60 + (d.VolR > 1.8) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_wick_breakout(d):
    lo = (d.Close > d.Upper_Wick_4H) & (d.RSI > 50) & (d.VolR > 1.0)
    sh = (d.Close < d.Lower_Wick_4H) & (d.RSI < 50) & (d.VolR > 1.0)
    score = pd.Series(63 + (d.VolR > 1.5) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_wick_swing(d):
    lo = (d.Close > d.Upper_Wick_1D) & (d.RSI > 52) & (d.VolR > 1.1)
    sh = (d.Close < d.Lower_Wick_1D) & (d.RSI < 48) & (d.VolR > 1.1)
    score = pd.Series(68 + (d.VolR > 1.5) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_poc_rebound(d: pd.DataFrame):
    poc = d.POC_Level.ffill()
    near_poc = (d.Close - poc).abs() < (0.6 * d.ATR)
    lo = near_poc & (d.Close > poc) & (d.RSI > 50)
    sh = near_poc & (d.Close < poc) & (d.RSI < 50)
    score = pd.Series(65 + (d.VolR > 1.5) * 10, index=d.index)
    return _b(lo), _b(sh), score, score

def s_wyckoff(d):
    rl, rh = d.Low.shift(1).rolling(50).min(), d.High.shift(1).rolling(50).max()
    spring = (d.Low < rl) & (d.Close > rl) & (d.VolR > 1.5)
    thrust = (d.High > rh) & (d.Close < rh) & (d.VolR > 1.5)
    sc = pd.Series(70, index=d.index)
    return _b(spring), _b(thrust), sc, sc

def s_break(d):
    dh, dl = d.High.shift(1).rolling(20).max(), d.Low.shift(1).rolling(20).min()
    lo = (d.Close > dh) & (d.ADX > 22) & (d.VolR > 1.3)
    sh = (d.Close < dl) & (d.ADX > 22) & (d.VolR > 1.3)
    sc = pd.Series(60, index=d.index)
    return _b(lo), _b(sh), sc, sc

def s_notebook_fractal_strategy(d: pd.DataFrame):
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    rolling_high_20 = h.shift(1).rolling(20).max()
    rolling_low_20 = l.shift(1).rolling(20).min()
    
    sweep_low = (l < rolling_low_20) & (c > rolling_low_20) & (d.LowWick > 0.3)
    sweep_high = (h > rolling_high_20) & (c < rolling_high_20) & (d.UpWick > 0.3)
    
    lo = sweep_low & (d.VolR > 1.25) & (d.RSI > 45)
    sh = sweep_high & (d.VolR > 1.25) & (d.RSI < 55)
    
    score_lo = pd.Series(72 + (d.VolR > 1.8) * 12 + (d.RSI.between(45, 60)) * 8, index=d.index)
    score_sh = pd.Series(72 + (d.VolR > 1.8) * 12 + (d.RSI.between(40, 55)) * 8, index=d.index)
    
    return _b(lo), _b(sh), score_lo, score_sh

def s_pro_momentum_long(d: pd.DataFrame):
    # استراتيجية الزخم المعتدل المستخرجة من صفقات المحترفين
    lo = d.RSI.between(64, 70) & (d.VolR < 0.9) & (d.Close > d.EMA21)
    score = pd.Series(74 + (d.Close > d.EMA50) * 10, index=d.index)
    return _b(lo), _b(d.Close < -1), score, score * 0

def s_pro_accumulation_long(d: pd.DataFrame):
    # استراتيجية التجميع الهادئ المستخرجة من صفقات المحترفين
    lo = d.RSI.between(35, 48) & (d.VolR < 0.6) & (d.Close <= d.BB_M)
    score = pd.Series(72 + (d.LowWick > 0.3) * 12, index=d.index)
    return _b(lo), _b(d.Close < -1), score, score * 0

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
    Strat("wick_scalp", "⚡ اختراق ذيل 1H (5m)", "scalp", s_wick_scalp, 1.2, (1.5, 2.5, 4.0)),
    Strat("bottom", "🟢 اقتناص القاع (Discount)", "reversal", s_bottom, 1.5, (2.0, 3.5, 5.0)),
    Strat("top", "🔴 اقتناص القمة (Premium)", "reversal", s_top, 1.5, (2.0, 3.5, 5.0)),
    Strat("hunter", "🏹 الصياد ICT Turtle Soup", "reversal", s_hunter, 1.3, (1.8, 3.0, 4.5)),
    Strat("ict_short_sweep", "🏹 اقتناص الشورت المتقدم (ICT Short Sweep)", "reversal", s_ict_short_sweep, 1.2, (1.8, 3.2, 5.0)),
    Strat("smc", "🧠 المال الذكي ICT (FVG+BOS)", "trend", s_smc, 1.5, (2.0, 3.5, 5.0)),
    Strat("wick_break", "💥 اختراق ذيل 4H (15m)", "trend", s_wick_breakout, 1.5, (2.0, 3.5, 5.0)),
    Strat("wick_swing", "🏆 اختراق ذيل 1D (4H)", "trend", s_wick_swing, 1.8, (2.5, 4.5, 6.5)),
    Strat("poc_rebound", "🎯 التداول على مستوى POC", "trend", s_poc_rebound, 1.4, (1.8, 3.0, 4.8)),
    Strat("wyckoff", "🌀 وايكوف Spring/Upthrust", "reversal", s_wyckoff, 1.6, (2.0, 3.5, 5.5)),
    Strat("break", "🚀 اختراق الاتجاه BOS", "trend", s_break, 1.8, (2.0, 4.0, 6.0)),
    Strat("notebook_fractal", "📓 فراكتلات دفتر الملاحظات وبوابات السعر", "reversal", s_notebook_fractal_strategy, 1.4, (2.0, 3.6, 5.2)),
    Strat("pro_mom_long", "🚀 زخم المحترف (Pro Momentum)", "trend", s_pro_momentum_long, 1.3, (1.8, 3.0, 4.5)),
    Strat("pro_acc_long", "🟢 التجميع الهادئ للمحترف (Pro Acc)", "reversal", s_pro_accumulation_long, 1.4, (2.0, 3.5, 5.0)),
]}

def kind_label(key: str, side: str) -> str:
    k = STRATS[key].kind
    if k == "scalp": return "⚡ مضاربة سريعة"
    if k == "trend": return "📈 تتبّع اتجاه"
    return "🟢 اقتناص قاع" if side == "LONG" else "🔴 اقتناص قمة"

def tf_duration_label(tf: str) -> str:
    if tf in ["1m", "3m", "5m"]: return "⚡ صفقة خاطفة (Scalp)"
    if tf in ["15m", "30m", "1h"]: return "📈 صفقة متوسطة (Intraday)"
    return "🏆 صفقة طويلة (Swing)"

# ──────────────────────────────────────────────────────────────────────────────
# محرك الذكاء الاصطناعي الذاتي وتفحص البيتكوين
# ──────────────────────────────────────────────────────────────────────────────
def check_btc_regime(hub: Hub, mk: str) -> str:
    btc_sym = Hub.sym(mk, "BTC")
    df = hub.ohlcv(mk, btc_sym, "1h", limit=100)
    if df is None or len(df) < 50: return "NEUTRAL"
    c = df["Close"]
    ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    last = c.iloc[-1]
    if last > ema50 * 1.005: return "BULLISH"
    elif last < ema50 * 0.995: return "BEARISH"
    return "NEUTRAL"

def autonomous_ai_confirmation(sym: str, side: str, price: float, rsi: float, volr: float, score: float, 
                               rsi_htf: float = 50.0, low_wick: float = 0.0, price_to_support_pct: float = 1.0,
                               btc_regime: str = "NEUTRAL") -> tuple[float, str]:
    ai_boost = 0.0
    reason = "مؤشرات فنية قياسية"

    if btc_regime == "BEARISH" and side == "LONG":
        return max(score - 30.0, 0.0), "⚠️ تحذير AI: اتجاه البيتكوين هابط عاماً"
    if btc_regime == "BULLISH" and side == "SHORT":
        return max(score - 30.0, 0.0), "⚠️ تحذير AI: اتجاه البيتكوين صاعد عاماً"

    if volr > 4.0:
        return max(score - 20.0, 0.0), "⚠️ تحذير AI: شمعة تصفية حادة (Liquidation Sweep)"

    if side == "SHORT":
        if rsi_htf < 38 or rsi < 36:
            return max(score - 35.0, 0.0), "⚠️ تحذير AI: تشبع بيعي حاد - خطر ارتداد"
        if price_to_support_pct < 0.006:
            return max(score - 25.0, 0.0), "⚠️ تحذير AI: السعر يتداول فوق دعم رئيسي"
    if side == "LONG":
        if rsi_htf > 68 or rsi > 70:
            return max(score - 35.0, 0.0), "⚠️ تحذير AI: تشبع شرائي حاد - خطر جني أرباح"

    if side == "LONG" and rsi < 45 and volr > 1.4:
        ai_boost += 8.0
        reason = "تأكيد AI: تجميع عند القاع"
    elif side == "SHORT" and rsi > 65 and volr > 1.4:
        ai_boost += 8.0
        reason = "تأكيد AI: تصريف وحماية قمة"

    final_score = np.clip(score + ai_boost, 0, 99)
    return float(final_score), reason

def ml_prob(d: pd.DataFrame):
    if RandomForestClassifier is None: return None
    X = pd.DataFrame({"rsi": d.RSI, "macd": d.MACDh / d.Close, "vol": d.VolR, "adx": d.ADX}).replace([np.inf, -np.inf], np.nan)
    n = len(d)
    y = (d.Close.shift(-3) > d.Close).astype(int).values
    ok = X.notna().all(axis=1).values
    train = ok & (np.arange(n) <= n - 10)
    if train.sum() < 80 or not ok[-2]: return None
    try:
        m = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=1)
        m.fit(X.values[train], y[train])
        return float(m.predict_proba(X.values[[n - 2]])[0][1])
    except Exception: return None

def analyze_symbol_tf(hub: Hub, cfg: dict, sym: str, tf: str, tk: dict | None = None, btc_regime: str = "NEUTRAL"):
    mk = cfg["mk"]
    df = hub.ohlcv(mk, sym, tf, 300)
    if df is None or len(df) < 150: return None
    d = add_indicators(df)
    i = -2
    price, atr = float(d.Close.iloc[-1]), float(d.ATR.iloc[i])
    if not np.isfinite(atr) or atr <= 0 or price <= 0: return None
    
    hits = []
    for key in cfg["strats"]:
        lo, sh, sl_, ss_ = STRATS[key].fn(d)
        val_l = float(sl_.iloc[i]) if hasattr(sl_, "iloc") else float(sl_)
        val_s = float(ss_.iloc[i]) if hasattr(ss_, "iloc") else float(ss_)
        if bool(lo.iloc[i]): hits.append((key, "LONG", val_l))
        if bool(sh.iloc[i]): hits.append((key, "SHORT", val_s))
        
    if not hits: return None
    key, side, score = max(hits, key=lambda h: h[2])

    rsi_htf = 50.0
    hd = hub.ohlcv(mk, sym, HTF_MAP.get(tf, "1h"), 120)
    if hd is not None and len(hd) > 60:
        hd_ind = add_indicators(hd)
        rsi_htf = float(hd_ind.RSI.iloc[-2])

    recent_low = float(d.Low.iloc[-30:-1].min())
    price_to_support_pct = (price - recent_low) / max(price, 1e-8)
    low_wick_now = float(d.LowWick.iloc[i])

    rsi_now, vol_now = float(d.RSI.iloc[i]), float(d.VolR.iloc[i])
    score, ai_reason = autonomous_ai_confirmation(
        sym, side, price, rsi_now, vol_now, score,
        rsi_htf=rsi_htf, low_wick=low_wick_now, price_to_support_pct=price_to_support_pct,
        btc_regime=btc_regime
    )

    htf = None
    if cfg.get("htf") and hd is not None and len(hd) > 60:
        up = float(hd.Close.iloc[-1]) > float(hd.Close.ewm(span=50, adjust=False).mean().iloc[-1])
        htf = "صاعد" if up else "هابط"

    ml = None
    if cfg.get("ml"):
        p = ml_prob(d)
        if p is not None:
            ml = p if side == "LONG" else 1 - p
            score = 0.8 * score + 0.2 * ml * 100

    S = STRATS[key]
    sg = 1 if side == "LONG" else -1
    entry_price = price
    sl = entry_price - sg * S.sl * atr
    dist_sl = abs(entry_price - sl)
    
    tp1 = entry_price + sg * S.tps[0] * atr
    tp2 = entry_price + sg * S.tps[1] * atr
    tp3 = entry_price + sg * S.tps[2] * atr

    return {
        "symbol": sym, "base": sym.split("/")[0], "mk": mk, "tf": tf, "side": side, "key": key, "label": S.label,
        "kind_label": kind_label(key, side), "tf_desc": tf_duration_label(tf), "conf": float(np.clip(score, 0, 99)),
        "agree": len(hits), "entry": entry_price, "sl": sl, "tps": [tp1, tp2, tp3],
        "rr": abs(tp2 - entry_price) / max(dist_sl, 1e-8), "atr": atr, "slm": S.sl, "tpm": list(S.tps),
        "rsi": rsi_now, "adx": float(d.ADX.iloc[i]), "volr": vol_now, "chg": float((tk or {}).get("percentage") or 0),
        "vol24": float((tk or {}).get("quoteVolume") or 0), "ml": ml, "htf": htf,
        "ai_reason": ai_reason, "prot_low": float(d.Protected_Low.iloc[-1] or 0), "ts": time.time(),
    }

def analyze_symbol_all_tfs(hub: Hub, cfg: dict, sym: str, tk: dict | None = None, btc_regime: str = "NEUTRAL"):
    target_tfs = cfg.get("scan_tfs", ["1m", "5m", "15m", "1h", "4h", "1d"])
    best_sig = None
    for tf in target_tfs:
        res = analyze_symbol_tf(hub, cfg, sym, tf, tk, btc_regime)
        if res:
            if best_sig is None or res["conf"] > best_sig["conf"]:
                best_sig = res
    return best_sig

def run_scan(hub: Hub, cfg: dict, progress=None):
    sort_asc = cfg.get("sort_order") == "من الأدنى إلى الأعلى"
    uni = hub.universe(cfg["mk"], cfg["min_vol"], sort_asc=sort_asc)[:cfg["limit"]]
    tick = hub.tickers(cfg["mk"])
    btc_regime = check_btc_regime(hub, cfg["mk"])
    out, done = [], 0
    if not uni: return out, 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(analyze_symbol_all_tfs, hub, cfg, s, tick.get(s), btc_regime): s for s, _ in uni}
        for f in as_completed(futs):
            done += 1
            try: r = f.result()
            except Exception: r = None
            if r and r["conf"] >= cfg.get("min_conf", 60): out.append(r)
            if progress: progress(done / len(uni))
    out.sort(key=lambda s: -s["conf"])
    return out, len(uni)

def calculate_dynamic_position_size(balance: float, risk_pct: float, entry: float, sl: float, atr: float, max_lev: int = 5) -> dict:
    dist_sl_pct = abs(entry - sl) / entry
    if dist_sl_pct == 0: return {"notional": 0, "qty": 0, "margin": 0}
    
    risk_amount = balance * (risk_pct / 100.0)
    notional = risk_amount / dist_sl_pct
    
    volatility_ratio = atr / entry
    if volatility_ratio > 0.03:
        notional *= (0.03 / volatility_ratio)
        
    required_margin = notional / max_lev
    if required_margin > balance * 0.3:
        notional = balance * 0.3 * max_lev

    return {"notional": notional, "qty": notional / entry, "margin": notional / max_lev}

def advance(p: dict, low: float, high: float, atr: float = 0.0) -> list:
    ev, hit_tp, s = [], False, p["side"]
    if p["stage"] >= 1 and atr > 0:
        if s == 1: p["sl"] = max(p["sl"], high - 1.5 * atr)
        else: p["sl"] = min(p["sl"], low + 1.5 * atr)

    while p["rem"] > 1e-9:
        if not hit_tp and ((s == 1 and low <= p["sl"]) or (s == -1 and high >= p["sl"])):
            ev.append((p["sl"], p["rem"], ("SL", "BE", "TRAIL")[min(p["stage"], 2)]))
            p["rem"] = 0.0
            break
        if p["stage"] >= 3:
            p["rem"] = 0.0
            break
        tp = p["tps"][p["stage"]]
        if not ((s == 1 and high >= tp) or (s == -1 and low <= tp)): break
        frac = p["rem"] if p["stage"] == 2 else min(SPLITS[p["stage"]], p["rem"])
        ev.append((tp, frac, f"TP{p['stage'] + 1}"))
        p["rem"] -= frac
        p["stage"] += 1
        hit_tp = True
        if p["stage"] == 1: p["sl"] = p["entry"]
        elif p["stage"] == 2: p["sl"] = p["tps"][0]
    return ev

def run_backtest(d: pd.DataFrame, keys: list, min_conf: float, risk_pct: float, fee: float, slippage: float = 0.0008):
    n = len(d)
    best, side, kk = np.zeros(n), np.zeros(n, dtype=int), [""] * n
    for key in keys:
        lo, sh, sl_, ss_ = STRATS[key].fn(d)
        for m, s, sg in ((lo, 1, sl_), (sh, -1, ss_)):
            v = sg.values.astype(float)
            upd = m.values & (v >= min_conf) & (v > best)
            best = np.where(upd, v, best)
            side = np.where(upd, s, side)
            for i in np.where(upd)[0]: kk[i] = key
    o, h, l, c, atr = (d[x].values for x in ("Open", "High", "Low", "Close", "ATR"))
    eq, curve, trades, i = 1.0, [(d.Timestamp.iloc[0], 1.0)], [], 60
    while i < n - 2:
        if side[i] == 0 or not atr[i] > 0:
            i += 1
            continue
        S, sd = STRATS[kk[i]], int(side[i])
        entry = o[i + 1] * (1 + sd * slippage)
        sl = entry - sd * S.sl * atr[i]
        p = {"side": sd, "entry": entry, "sl": sl, "tps": [entry + sd * m * atr[i] for m in S.tps], "stage": 0, "rem": 1.0}
        risk, R, j, exit_j = abs(entry - sl), 0.0, i + 1, n - 1
        while j < n:
            for pr, fr, _ in advance(p, l[j], h[j], atr[j]): R += fr * sd * (pr - entry) / risk
            if p["rem"] <= 1e-9:
                exit_j = j
                break
            j += 1
        else: R += p["rem"] * sd * (c[-1] - entry) / risk
        R -= 2 * (fee + slippage) * entry / risk
        eq = max(eq * (1 + risk_pct / 100 * R), 0.0)
        trades.append({"الوقت": d.Timestamp.iloc[i + 1], "الصفقة": "شراء" if sd == 1 else "بيع",
                       "الاستراتيجية": S.label, "الدخول": entry, "R": round(R, 2), "الشموع": exit_j - i})
        curve.append((d.Timestamp.iloc[exit_j], eq))
        i = exit_j + 1
    return trades, curve

def new_acct(bal: float) -> dict:
    return {"start": bal, "balance": bal, "positions": [], "history": [], "equity": [], "last_scan": 0.0, "cooldown": {}, "pid": 0}

def load_acct(bal: float) -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f: a = json.load(f)
        if isinstance(a, dict) and "balance" in a:
            for k, v in new_acct(bal).items(): a.setdefault(k, v)
            return a
    except Exception: pass
    return new_acct(bal)

def save_acct(a: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(a, f, ensure_ascii=False, default=float)
    except Exception: pass

def apply_fills(a: dict, p: dict, events: list, fee: float):
    for price, frac, reason in events:
        q = p["qty"] * frac
        pnl = p["side"] * (price - p["entry"]) * q - fee * price * q
        a["balance"] += pnl
        p["realized"] += pnl
        p["fills"].append([price, frac, reason, time.time()])

def finalize(a: dict, p: dict, cool_min: float):
    tot = sum(f[1] for f in p["fills"]) or 1
    avg_exit = sum(f[0] * f[1] for f in p["fills"]) / tot
    a["history"].append({
        "closed": time.time(), "opened": p["opened"], "symbol": p["symbol"], "side": p["side"], "label": p["label"],
        "kind": p["kind_label"], "conf": p["conf"], "entry": p["entry"], "exit": avg_exit, "pnl": p["realized"],
        "pct": p["realized"] / p["notional"] * 100, "reason": p["fills"][-1][2] if p["fills"] else "-"})
    a["history"] = a["history"][-500:]
    a["cooldown"][p["symbol"]] = time.time() + cool_min * 60
    a["positions"] = [x for x in a["positions"] if x["id"] != p["id"]]

def open_position(a: dict, s: dict, price: float, A: dict) -> dict:
    sg = 1 if s["side"] == "LONG" else -1
    sl = price - sg * s["slm"] * s["atr"]
    pos_data = calculate_dynamic_position_size(a["balance"], A["risk"], price, sl, s["atr"], A["lev"])
    notional = pos_data["notional"]
    fee = A["fee"] * notional
    a["balance"] -= fee
    a["pid"] += 1
    p = {"id": a["pid"], "symbol": s["symbol"], "mk": s["mk"], "side": sg, "key": s["key"], "label": s["label"],
         "kind_label": s["kind_label"], "conf": s["conf"], "entry": price, "sl": sl, "tps": s["tps"], "stage": 0,
         "rem": 1.0, "qty": pos_data["qty"], "notional": notional, "opened": time.time(), "realized": -fee, "fills": [], "atr": s["atr"]}
    a["positions"].append(p)
    return p

def unrealized(a: dict, px: dict) -> float:
    return sum(p["side"] * (px.get(p["symbol"], p["entry"]) - p["entry"]) * p["qty"] * p["rem"] for p in a["positions"])

def engine_step(hub: Hub, cfg: dict, A: dict):
    a, now = st.session_state.acct, time.time()
    changed = False
    px = st.session_state.setdefault("last_px", {})
    for mk in {p["mk"] for p in a["positions"]}: px.update(hub.prices(mk, [p["symbol"] for p in a["positions"] if p["mk"] == mk]))
    for p in list(a["positions"]):
        price = px.get(p["symbol"])
        if not price: continue
        ev = advance(p, price, price, p.get("atr", 0.0))
        if ev:
            apply_fills(a, p, ev, A["fee"])
            changed = True
            st.toast(f"{p['symbol'].split('/')[0]}: {ev[-1][2]} عند {fp(ev[-1][0])}", icon="🎯" if ev[-1][2].startswith("TP") else "🛑")
        if p["rem"] <= 1e-9: finalize(a, p, A["cool"])
    if A["auto"] and len(a["positions"]) < A["max_trades"]:
        if now - a["last_scan"] > A["scan_min"] * 60 or not st.session_state.get("signals"):
            with st.spinner("مسح الذكاء الاصطناعي للسوق..."): sigs, tot = run_scan(hub, cfg)
            st.session_state.signals, st.session_state.scan_info = sigs, {"total": tot, "time": time.time()}
            a["last_scan"], changed = time.time(), True
        open_syms = {p["symbol"] for p in a["positions"]}
        for s in st.session_state.get("signals", []):
            if len(a["positions"]) >= A["max_trades"]: break
            if s["symbol"] in open_syms or a["cooldown"].get(s["symbol"], 0) > now or s["conf"] < A["min_conf"]: continue
            if s["mk"] == "spot" and s["side"] == "SHORT": continue
            price = hub.prices(s["mk"], [s["symbol"]]).get(s["symbol"])
            if not price: continue
            p = open_position(a, s, price, A)
            open_syms.add(s["symbol"])
            px[s["symbol"]], changed = price, True
            st.toast(f"صفقة جديدة: {s['base']} {'شراء' if p['side'] == 1 else 'بيع'} — {s['kind_label']}", icon="🚀")
    eq = a["balance"] + unrealized(a, px)
    if not a["equity"] or now - a["equity"][-1][0] > 10:
        a["equity"].append([now, eq])
        a["equity"] = a["equity"][-3000:]
        changed = True
    if changed: save_acct(a)

def acct_stats(a: dict, px: dict) -> dict:
    un = unrealized(a, px)
    h = a["history"]
    wins = [x["pnl"] for x in h if x["pnl"] > 0]
    loss = [x["pnl"] for x in h if x["pnl"] <= 0]
    pf = sum(wins) / abs(sum(loss)) if loss and sum(loss) != 0 else (float("inf") if wins else 0.0)
    eqs = [e[1] for e in a["equity"]] if a.get("equity") else [a["start"]]
    peak, dd = eqs[0], 0.0
    for e in eqs:
        peak = max(peak, e)
        dd = max(dd, (peak - e) / peak * 100 if peak else 0)
    return {"equity": a["balance"] + un, "unreal": un, "total": a["balance"] + un - a["start"],
            "wr": len(wins) / len(h) * 100 if h else 0.0, "n": len(h), "pf": pf, "dd": dd}

def card_html(s: dict) -> str:
    long = s["side"] == "LONG"
    col = UP if long else DOWN
    rows = "".join(f'<div class="lv"><span>{n}</span><b>{fp(v)}</b></div>' for n, v in
                   (("دخول", s["entry"]), ("وقف الخسارة", s["sl"]), ("الهدف 1", s["tps"][0]),
                    ("الهدف 2", s["tps"][1]), ("الهدف 3", s["tps"][2])))
    
    card_str = (
        f'<div class="sig" style="--c:{col}">'
        f'<div class="sig-top">'
        f'<span class="sym">{s["base"]}<small>/USDT</small></span>'
        f'<div><span class="tf-badge">{s["tf"]}</span> '
        f'<span class="badge" style="background:{col}26;color:{col}">{"▲ شراء" if long else "▼ بيع"}</span></div>'
        f'</div>'
        f'<div class="sig-kind">{s["kind_label"]} <span style="font-size:0.75rem;color:#848e9c;">({s["tf_desc"]})</span></div>'
        f'<div class="sig-strat">{s["label"]}</div>'
        f'<div class="bar"><i style="width:{s["conf"]:.0f}%;background:{col}"></i></div>'
        f'<div class="sig-conf">الثقة {s["conf"]:.0f}% — {s["ai_reason"]} — R:R {s["rr"]:.1f}</div>'
        f'{rows}'
        f'</div>'
    )
    return card_str

def signals_df(sigs: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "العملة": s["base"],
        "الصفقة": "🟢 شراء" if s["side"] == "LONG" else "🔴 بيع",
        "الفريم": s["tf"],
        "نوع الصفقة": s["tf_desc"],
        "النمط": s["kind_label"],
        "الاستراتيجية": s["label"],
        "الثقة": round(s["conf"]),
        "الدخول": fp(s["entry"]),
        "وقف الخسارة": fp(s["sl"]),
        "الهدف 1": fp(s["tps"][0]),
        "الهدف 2": fp(s["tps"][1]),
        "الهدف 3": fp(s["tps"][2]),
        "R:R": round(s["rr"], 1),
        "ملاحظة AI": s["ai_reason"],
        "حجم 24س": fv(s["vol24"]),
        "24س %": round(s["chg"], 2)
    } for s in sigs])

def build_chart(d: pd.DataFrame, sig: dict | None, show: list, bars: int):
    d = d.tail(bars)
    x = d.Timestamp
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.015, row_heights=[0.55, 0.12, 0.16, 0.17])
    
    fig.add_trace(go.Candlestick(x=x, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="السعر",
                                 increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)

    lines = {"EMA 21": ("EMA21", "#fcd535"), "EMA 50": ("EMA50", "#5b9cf6"), "EMA 200": ("EMA200", "#c084fc"), "VWAP": ("VWAP", "#f59e0b")}
    for name, (col, color) in lines.items():
        if name in show: fig.add_trace(go.Scatter(x=x, y=d[col], name=name, line=dict(color=color, width=1.2)), 1, 1)

    if "القاع المحمي (Protected Low)" in show:
        fig.add_trace(go.Scatter(x=x, y=d["Protected_Low"], name="القاع المحمي", line=dict(color="#00e676", width=2, dash="dash")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=d["Protected_High"], name="القمة المحمية", line=dict(color="#ff1744", width=2, dash="dash")), 1, 1)

    if "📐 خطوط الاتجاه والمثلثات (Trendlines)" in show:
        fig.add_trace(go.Scatter(x=x, y=d["Trend_Res"], name="خط المقاومة (Trendline)", line=dict(color="#ff5252", width=1.8, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=x, y=d["Trend_Sup"], name="خط الدعم (Trendline)", line=dict(color="#69f0ae", width=1.8, dash="dot")), 1, 1)

    if "مؤشر POC & Volume Profile" in show:
        _, _, poc_p, _ = calc_volume_profile(d, bins=25)
        if poc_p:
            fig.add_hline(y=poc_p, line_color="#ff9900", line_dash="solid", line_width=2,
                          annotation_text=f"POC: {fp(poc_p)}", annotation_position="left", row=1, col=1)

    fig.add_trace(go.Bar(x=x, y=d.Volume, marker_color=np.where(d.Close >= d.Open, UP, DOWN), opacity=.55), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=d.RSI, name="RSI", line=dict(color="#c084fc", width=1.3)), 3, 1)
    fig.add_trace(go.Bar(x=x, y=d.MACDh, marker_color=np.where(d.MACDh >= 0, UP, DOWN)), 4, 1)

    fig.update_layout(height=780, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, showlegend=False,
                      margin=dict(l=8, r=90, t=8, b=8), hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=False, gridcolor="#1a1f25")
    fig.update_yaxes(gridcolor="#1a1f25", side="right")
    return fig

def sidebar():
    sb = st.sidebar
    sb.markdown('<div class="brand">NEXUS<span>AUTONOMOUS</span></div>', unsafe_allow_html=True)
    demo = sb.toggle("وضع المحاكاة", value=ccxt is None, key="demo")
    mk = "futures" if sb.radio("السوق", ["فوري Spot", "عقود Futures"], horizontal=True, key="mk").startswith("عقود") else "spot"
    
    st_tfs = sb.multiselect("فريمات المسح الشامل", ["1m", "5m", "15m", "1h", "4h", "1d"], default=["1m", "5m", "15m", "1h", "4h", "1d"], key="scan_tfs")
    tf = sb.selectbox("الإطار الزمني للشارت", list(TF_SEC)[:-1], index=3, key="tf")
    
    sort_order = sb.radio("ترتيب تصفية الحجم والسيولة", ["من الأعلى إلى الأدنى", "من الأدنى إلى الأعلى"], key="sort_order")
    min_vol = sb.select_slider("أدنى حجم 24س ($)", [0.0, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7], value=1e5, format_func=fv, key="minvol")
    limit = sb.slider("عدد العملات للمسح", 10, 1000, 750, step=25, key="limit")
    
    strats = sb.multiselect("الاستراتيجيات", list(STRATS), default=list(STRATS), format_func=lambda k: STRATS[k].label, key="strats")
    min_conf = sb.slider("أدنى ثقة للإشارة %", 50, 90, 60, key="minconf")
    htf = sb.toggle("فلتر الاتجاه الفريم الأعلى", value=True, key="htf")
    ml = sb.toggle("تأكيد الذكاء الاصطناعي الذاتي", value=True, key="ml")
    
    sb.markdown("---")
    sb.markdown("### 🕵️‍♂️ محطة تحليل صفقات المحترفين")
    custom_sym_input = sb.text_input("أدخل عملة المحترف (مثال: RLC أو OGN)", value="", key="cust_sym")
    custom_side_input = sb.selectbox("اتجاه صفقة المحترف", ["LONG", "SHORT"], key="cust_side")
    custom_entry_input = sb.number_input("سعر دخول المحترف المقترح", value=0.0, format="%.6f", key="cust_entry")
    
    if sb.button("🔍 تحليل ومطابقة الصفقة بالذكاء الاصطناعي"):
        if custom_sym_input:
            st.session_state.pro_analysis = {
                "symbol": f"{custom_sym_input.upper().strip()}/USDT",
                "side": custom_side_input,
                "entry": float(custom_entry_input)
            }
    
    sb.markdown("---")
    auto = sb.toggle("تشغيل التداول التلقائي", value=False, key="auto")
    max_trades = sb.slider("الصفقات المتزامنة", 1, 6, 3, key="maxtr")
    risk = sb.slider("المخاطرة %", 0.25, 5.0, 1.0, step=0.25, key="risk")
    lev = sb.slider("الرافعة", 1, 20, 3, key="lev") if mk == "futures" else 1
    a_conf = sb.slider("أدنى ثقة للتلقائي %", 50, 95, 70, key="aconf")
    scan_min = sb.slider("إعادة المسح (دقيقة)", 1, 30, 5, key="scanmin")
    refresh = sb.slider("تحديث الأسعار (ثانية)", 3, 30, 5, key="refresh")
    cool = sb.slider("تبريد العملة (دقيقة)", 0, 120, 30, key="cool")
    start = sb.number_input("رصيد البداية ($)", 100.0, 10_000_000.0, 10000.0, step=500.0, key="start")
    if sb.button("إعادة ضبط المحفظة"):
        st.session_state.acct = new_acct(start)
        save_acct(st.session_state.acct)
        st.rerun()
    cfg = {"mk": mk, "tf": tf, "scan_tfs": st_tfs or ["15m"], "min_vol": min_vol, "limit": limit, "sort_order": sort_order,
           "strats": strats or list(STRATS), "min_conf": min_conf, "htf": htf, "ml": ml, "demo": demo}
    A = {"auto": auto, "max_trades": max_trades, "risk": risk, "lev": lev, "min_conf": a_conf, "scan_min": scan_min,
         "refresh": refresh, "cool": cool, "fee": 0.0005 if mk == "futures" else 0.001, "start": start}
    return cfg, A

def tab_scanner(hub, cfg):
    if "pro_analysis" in st.session_state:
        pro = st.session_state.pro_analysis
        st.markdown(f"### 🧪 نتائج تحليل صفقة المحترف: {pro['symbol']} ({pro['side']})")
        with st.spinner("جاري جلب البيانات وتحليل المؤشرات ومقارنة نقطة الدخول..."):
            sym_full = Hub.sym(cfg["mk"], pro["symbol"].split("/")[0])
            res = analyze_symbol_all_tfs(hub, cfg, sym_full)
            df_cust = hub.ohlcv(cfg["mk"], sym_full, cfg["tf"], 200)
            if df_cust is not None and len(df_cust) > 50:
                d_cust = add_indicators(df_cust)
                r_now = float(d_cust.RSI.iloc[-2])
                v_now = float(d_cust.VolR.iloc[-2])
                p_now = float(d_cust.Close.iloc[-1])
                
                c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                c_col1.metric("السعر الحالي", fp(p_now))
                c_col2.metric("مؤشر RSI", f"{r_now:.1f}")
                c_col3.metric("حجم التداول النسبي", f"{v_now:.2f}x")
                c_col4.metric("اتجاه الفريم", "صاعد 🟢" if p_now > float(d_cust.EMA50.iloc[-2]) else "هابط 🔴")
                
                pro_entry = pro.get("entry", 0.0)
                if pro_entry > 0:
                    diff_pct = ((p_now - pro_entry) / pro_entry) * 100
                    st.info(f"📌 **سعر دخول المحترف المحدد:** {fp(pro_entry)} | الفرق عن السعر الحالي: **{diff_pct:+.2f}%**")
                    if pro["side"] == "LONG" and p_now < pro_entry * 0.98:
                        st.warning("⚠️ السعر الحالي أدنى من سعر دخول المحترف (قد تكون فرصة ممتازة للتجميع أو العملة كسرت مستوى الدعم).")
                    elif pro["side"] == "LONG" and p_now > pro_entry * 1.05:
                        st.warning("🚀 السعر الحالي صعد وابتعد كثيرًا عن سعر الدخول المحدد (يُفضل انتظار تصحيح).")
                    else:
                        st.success("🎯 السعر الحالي قيد نطاق الدخول المقترح أو قريب جداً منه!")

                if res and res["side"] == pro["side"]:
                    st.success(f"✅ **تطابق تام!** الصفقة متوافقة مع استراتيجية النظام: **{res['label']}** على فريم ({res['tf']}) بثقة ({res['conf']:.0f}%)!")
                else:
                    st.warning("⚠️ **العملة تتطابق مع استراتيجيات المحترفين المستخرجة أو تم تسجيلها كنمط جديد.**")

                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("💾 حفظ الصفقة في ذاكرة الذكاء الاصطناعي"):
                    trade_record = {
                        "symbol": pro["symbol"],
                        "side": pro["side"],
                        "entry": pro_entry,
                        "current_price": p_now,
                        "rsi": r_now,
                        "volr": v_now,
                        "timestamp": time.time(),
                        "matched_strategy": res["label"] if (res and res["side"] == pro["side"]) else "Custom/New Strategy"
                    }
                    save_memory_trade(trade_record)
                    st.success("✅ تم حفظ الصفقة بنجاح في ملف الذاكرة `pro_trades_memory.json`!")
                
                mem_list = load_memory()
                st.caption(f"عدد الصفقات المحفوظة حالياً في الذاكرة: {len(mem_list)}")

            else:
                st.error("تعذر جلب بيانات هذه العملة، تأكد من صحة الرمز أو توفره في المنصة.")
        if st.button("إغلاق نتائج تحليل المحترف"):
            del st.session_state.pro_analysis
            st.rerun()
        st.markdown("---")

    c1, c2 = st.columns([1, 3])
    if c1.button("🚀 ابدأ مسح الذكاء الاصطناعي الشامل", type="primary"):
        t0, bar = time.time(), st.progress(0.0)
        sigs, tot = run_scan(hub, cfg, lambda p: bar.progress(min(p, 1.0)))
        bar.empty()
        st.session_state.signals = sigs
        st.session_state.scan_info = {"total": tot, "time": time.time(), "sec": time.time() - t0}
    sigs = st.session_state.get("signals")
    if not sigs:
        c2.markdown('<div class="note">اضغط «ابدأ المسح» للبحث عبر جميع الفريمات والعملات عن أفضل الصفقات، أو استخدم خانة تحليل صفقات المحترفين في الشريط الجانبي لفحص أي عملة وسعر دخول فوراً وحفظها في الذاكرة.</div>', unsafe_allow_html=True)
        return
    
    st.markdown("#### 🌟 أفضل 10 صفقات نادرة وموصى بها")
    cols1 = st.columns(5)
    for col, s in zip(cols1, sigs[:5]):
        col.markdown(card_html(s), unsafe_allow_html=True)
        
    if len(sigs) > 5:
        cols2 = st.columns(5)
        for col, s in zip(cols2, sigs[5:10]):
            col.markdown(card_html(s), unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown(f"#### 📊 جميع الصفقات المكتشفة ({len(sigs)})")
    df = signals_df(sigs)
    show_df(df, column_config={"الثقة": st.column_config.ProgressColumn("الثقة", min_value=0, max_value=100, format="%d%%")})

def render_account(hub, cfg, A):
    a, px = st.session_state.acct, st.session_state.get("last_px", {})
    S = acct_stats(a, px)
    m = st.columns(6)
    m[0].metric("الرصيد الكلي", f"${S['equity']:,.2f}", f"{S['total']:+,.2f}$")
    m[1].metric("ربح الصفقات المفتوحة", f"${S['unreal']:+,.2f}")
    m[2].metric("نسبة الربح", f"{S['wr']:.0f}%", f"{S['n']} صفقة")
    m[3].metric("معامل الربح", f"{S['pf']:.2f}")
    m[4].metric("أقصى تراجع", f"{S['dd']:.2f}%")
    m[5].metric("الحالة", "يعمل ✅" if A["auto"] else "متوقف ⏸")

    if a["positions"]:
        rows = []
        for p in a["positions"]:
            cur = px.get(p["symbol"], p["entry"])
            pnl = p["realized"] + p["side"] * (cur - p["entry"]) * p["qty"] * p["rem"]
            rows.append({"العملة": p["symbol"].split("/")[0], "الصفقة": "🟢 شراء" if p["side"] == 1 else "🔴 بيع",
                         "النوع": p["kind_label"], "الاستراتيجية": p["label"], "الدخول": fp(p["entry"]),
                         "السعر الحالي": fp(cur), "الربح $": round(pnl, 2), "الوقف": fp(p["sl"]), "TP1": fp(p["tps"][0])})
        show_df(pd.DataFrame(rows))

def tab_auto(hub, cfg, A):
    @st.fragment(run_every=A["refresh"] if A["auto"] else None)
    def panel():
        engine_step(hub, cfg, A)
        render_account(hub, cfg, A)
    panel()

def tab_chart(hub, cfg):
    sigs = st.session_state.get("signals") or []
    uni = [s for s, _ in hub.universe(cfg["mk"], cfg["min_vol"])]
    opts = list(dict.fromkeys([s["symbol"] for s in sigs] + uni)) or [Hub.sym(cfg["mk"], "BTC")]
    c1, c2, c3 = st.columns([1.2, 1, 3])
    sym = c1.selectbox("العملة", opts, format_func=lambda s: s.split("/")[0], key="ch_sym")
    tf = c2.selectbox("الإطار", list(TF_SEC), index=list(TF_SEC).index(cfg["tf"]), key="ch_tf")
    show = c3.multiselect("الطبقات", ["EMA 21", "EMA 50", "EMA 200", "VWAP", "إشارات", "مستويات الصفقة", "مؤشر POC & Volume Profile", "القاع المحمي (Protected Low)", "📐 خطوط الاتجاه والمثلثات (Trendlines)"],
                          default=["EMA 21", "مؤشر POC & Volume Profile", "القاع المحمي (Protected Low)", "📐 خطوط الاتجاه والمثلثات (Trendlines)"], key="ch_show")
    df = hub.ohlcv(cfg["mk"], sym, tf, 300)
    if df is not None and len(df) > 50:
        d = add_indicators(df)
        sig = next((s for s in sigs if s["symbol"] == sym), None) or analyze_symbol_tf(hub, cfg, sym, tf)
        m = st.columns(6)
        m[0].metric("السعر", fp(d.Close.iloc[-1]))
        m[1].metric("القاع المحمي", fp(d.Protected_Low.iloc[-1]))
        m[2].metric("القمة المحمية", fp(d.Protected_High.iloc[-1]))
        m[3].metric("RSI", f"{d.RSI.iloc[-2]:.0f}")
        m[4].metric("ADX", f"{d.ADX.iloc[-2]:.0f}")
        m[5].metric("الهيكل", "صاعد 🟢" if d.Is_Above_Protected_Low.iloc[-1] else "هابط 🔴")
        show_plot(build_chart(d, sig, show, bars=st.slider("عدد الشموع", 60, 300, 150)))

def tab_market(hub, cfg):
    tk = hub.tickers(cfg["mk"])
    uni = hub.universe(cfg["mk"], cfg["min_vol"])
    if uni:
        df = pd.DataFrame([{"العملة": s.split("/")[0], "السعر": tk[s].get("last"), "24س %": round(tk[s].get("percentage") or 0, 2), "الحجم": tk[s]["quoteVolume"]} for s, _ in uni])
        top = df.nlargest(60, "الحجم")
        fig = go.Figure(go.Treemap(labels=top["العملة"], parents=[""] * len(top), values=top["الحجم"], text=[f"{v:+.2f}%" for v in top["24س %"]], textinfo="label+text", marker=dict(colors=top["24س %"], colorscale=[[0, DOWN], [0.5, "#2b3139"], [1, UP]], cmid=0)))
        fig.update_layout(height=480, template="plotly_dark", paper_bgcolor=BG, margin=dict(l=0, r=0, t=8, b=0))
        st.markdown("#### الخريطة الحرارية للسوق")
        show_plot(fig)

def tab_backtest(hub, cfg):
    uni = [s for s, _ in hub.universe(cfg["mk"], cfg["min_vol"])] or [Hub.sym(cfg["mk"], "BTC")]
    c = st.columns(4)
    sym = c[0].selectbox("العملة", uni, format_func=lambda s: s.split("/")[0], key="bt_sym")
    tf = c[1].selectbox("الإطار", list(TF_SEC), index=list(TF_SEC).index(cfg["tf"]), key="bt_tf")
    risk = c[2].slider("المخاطرة %", 0.25, 5.0, 1.0, key="bt_risk")
    conf = c[3].slider("أدنى ثقة %", 50, 90, cfg["min_conf"], key="bt_conf")
    if st.button("▶️ تشغيل الباك تست", type="primary"):
        df = hub.ohlcv(cfg["mk"], sym, tf, 1000)
        if df is not None:
            d = add_indicators(df)
            trades, curve = run_backtest(d, list(STRATS), conf, risk, 0.0005)
            if trades:
                t = pd.DataFrame(trades)
                st.success(f"تم تنفيذ {len(t)} صفقة | نسبة النجاح: {(t.R > 0).mean()*100:.0f}%")
                show_df(t.iloc[::-1])

def tab_calc():
    c = st.columns(4)
    side = c[0].radio("الصفقة", ["شراء", "بيع"], horizontal=True)
    bal = c[1].number_input("الرصيد ($)", 10.0, 1e8, 10000.0)
    risk = c[2].number_input("المخاطرة %", 0.1, 20.0, 1.0)
    lev = c[3].number_input("الرافعة", 1, 125, 5)
    c2 = st.columns(2)
    entry = c2[0].number_input("سعر الدخول", 0.0, 1e9, 100.0)
    sl = c2[1].number_input("وقف الخسارة", 0.0, 1e9, 97.0)
    dist = abs(entry - sl)
    if entry > 0 and dist > 0:
        risk_amt = bal * risk / 100
        notional = risk_amt / (dist / entry)
        st.info(f"المبلغ المعرض للخسارة: **${risk_amt:,.2f}** | حجم المركز: **${notional:,.2f}** | الهامش المطلوب: **${notional/lev:,.2f}**")

def main():
    st.set_page_config(page_title="NEXUS Terminal Autonomous", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    ensure_theme()
    st.markdown(CSS, unsafe_allow_html=True)
    cfg, A = sidebar()
    hub = get_hub(cfg["demo"])
    if "acct" not in st.session_state: st.session_state.acct = load_acct(A["start"])

    tk = hub.tickers(cfg["mk"])
    cols = st.columns(5)
    for col, b in zip(cols, ["BTC", "ETH", "BNB", "SOL", "XRP"]):
        t = tk.get(Hub.sym(cfg["mk"], b))
        if t: col.metric(f"{b}/USDT", fp(t.get("last")), f"{(t.get('percentage') or 0):+.2f}%")

    tabs = st.tabs(["📡 الماسح", "🤖 التداول التلقائي", "📈 الشارت", "🌍 السوق", "🧪 باك تست", "🧮 المخاطر"])
    with tabs[0]: tab_scanner(hub, cfg)
    with tabs[1]: tab_auto(hub, cfg, A)
    with tabs[2]: tab_chart(hub, cfg)
    with tabs[3]: tab_market(hub, cfg)
    with tabs[4]: tab_backtest(hub, cfg)
    with tabs[5]: tab_calc()

if __name__ == "__main__":
    main()

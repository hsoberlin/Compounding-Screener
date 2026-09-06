"""
COMPOUNDING SCREENER - INSTITUTIONAL BREAKOUT EDITION
=====================================================
Sistem penyaringan saham berbasis momentum absolut:
1. Filter Makro: Deteksi IHSG (Bear Regime Hysteresis -20% / -10%)
2. Tahap 0: Likuiditas >= Rp1M & Struktur Aman (Harga > LL20)
3. Tahap 1: Wyckoff Absorption (No Supply, Vol Ratio < 1.0)
4. Tahap 2: Breakout HH20 dengan Ambang Batas Dinamis
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import timedelta, timezone

st.set_page_config(page_title="Compounding Screener", layout="centered", initial_sidebar_state="collapsed")
WIB = timezone(timedelta(hours=7))

# =====================================================================
# TEMA VISUAL UI (INSTITUTIONAL DARK MODE)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #060709;
  --card: #0D1117;
  --card-line: #30363D;
  --accent-1: #FF7B72; 
  --accent-2: #58A6FF; 
  --text: #C9D1D9;
  --text-dim: #8B949E;
  --green: #3FB950;
  --red: #F85149;
  --bear: #E5534B;
  --bull: #2EA043;
}

.stApp { background-color: var(--bg); }
.stApp, .stApp p, .stApp span, .stApp div { color: var(--text); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; padding-bottom: 0px; }
[data-testid="stMetricValue"], code, .stMarkdown code { font-family: 'IBM Plex Mono', monospace !important; }

[data-testid="collapsedControl"] { display: none; }

/* Panel Makro */
.macro-panel {
    background: #161B22; border: 1px solid var(--card-line);
    padding: 16px; border-radius: 8px; margin-bottom: 24px;
    text-align: center;
}
.macro-title { font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px;}
.macro-status { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; margin-top: 4px; }

.card-container {
  background: var(--card); border: 1px solid var(--card-line);
  padding: 16px; border-radius: 8px; margin-bottom: 16px;
  display: flex; flex-direction: column; gap: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  border-left: 4px solid var(--accent-2);
}

.header-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.stock-title { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: #FFFFFF; }

.badge { font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 4px 8px; border-radius: 4px; white-space: nowrap; font-weight: 600;}
.badge-sektor { background: rgba(255, 255, 255, 0.1); color: var(--text-dim); }

.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
.metric-item { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--text-dim); display: flex; flex-direction: column; }
.metric-val { color: #FFFFFF; font-weight: 600; font-size: 14px; margin-top: 2px;}

.conf-header { display: flex; justify-content: space-between; font-size: 12px; font-family: 'IBM Plex Mono', monospace; margin-bottom: 4px; }
.conf-bar-bg { background: #21262D; border-radius: 4px; width: 100%; height: 6px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease-in-out; }

.stButton button { width: 100%; font-family: 'Space Grotesk', sans-serif; font-weight: 600; background-color: #21262D; color: #FFF; border: 1px solid #30363D; padding: 12px;}
.stButton button:hover { border-color: var(--accent-2); color: var(--accent-2); }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PARAMETER SISTEM MUTLAK
# =====================================================================
MIN_LIQUIDITY = 1_000_000_000  
VOL_RATIO_MAX = 1.0            
LL_PERIOD = 20                 
HYSTERESIS_ENTER_BEAR = -20.0  # Masuk Bear jika IHSG drop 20% dari puncak
HYSTERESIS_EXIT_BEAR = -10.0   # Keluar Bear jika IHSG pulih ke batas -10%

# =====================================================================
# DATA TICKER BEI
# =====================================================================
IDX_TICKERS_ALL = """AALI, ABBA, ABDA, ABMM, ACES, ADRO, AGII, AKRA, AMRT, ANTM, ARNA, ASII, AUTO, BBCA, BBNI, BBRI, BBTN, BMRI, BRIS, BRPT, BSDE, BUKA, BYAN, CPIN, CTRA, EMTK, ESSA, EXCL, GGRM, GOTO, HRUM, ICBP, INCO, INDF, INDY, INKP, INTP, ISAT, ITMG, JPFA, KLBF, MAPI, MDKA, MEDC, MGRO, MIKA, MNCN, MTEL, PGAS, PGEO, PTBA, PTPP, SMGR, SRTG, TBIG, TINS, TKIM, TLKM, TOWR, TPIA, UNTR, UNVR, WIKA""" # Diperpendek untuk simulasi, masukkan 750+ ticker Anda di sini
TICKERS = list(set([t.strip().upper() for t in IDX_TICKERS_ALL.replace('\n', ',').split(',') if t.strip()]))

# =====================================================================
# ENGINE & HYSTERESIS
# =====================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers, period="6mo"):
    tickers_jk = [f"{t}.JK" for t in tickers]
    try:
        df = yf.download(tickers_jk, period=period, interval="1d", progress=False, group_by="ticker", threads=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_macro_regime():
    try:
        ihsg = yf.download("^JKSE", period="1y", progress=False)
        if ihsg.empty: return "NORMAL", 0.0
        
        ihsg['Peak'] = ihsg['Close'].cummax()
        ihsg['Drawdown'] = (ihsg['Close'] - ihsg['Peak']) / ihsg['Peak'] * 100
        
        regime = "NORMAL"
        for dd in ihsg['Drawdown']:
            if regime == "NORMAL" and dd <= HYSTERESIS_ENTER_BEAR:
                regime = "BEAR"
            elif regime == "BEAR" and dd >= HYSTERESIS_EXIT_BEAR:
                regime = "NORMAL"
                
        current_dd = float(ihsg['Drawdown'].iloc[-1])
        return regime, current_dd
    except:
        return "NORMAL", 0.0

def calculate_stochastic(high, low, close, k_window=10, smooth_k=5):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    stoch_k = fast_k.rolling(window=smooth_k).mean()
    return stoch_k

def analyze_stock(df_stock, ticker, regime):
    if df_stock is None or len(df_stock.dropna()) < 35: return None
        
    close = df_stock['Close'].dropna()
    high = df_stock['High'].dropna()
    low = df_stock['Low'].dropna()
    volume = df_stock['Volume'].dropna()
    
    common_idx = close.index.intersection(volume.index)
    close, high, low, volume = close[common_idx], high[common_idx], low[common_idx], volume[common_idx]
    if len(close) < 35: return None

    # TAHAP 0 & 1: Filter Inti
    val_ma20 = (close * volume).rolling(20).mean().iloc[-1]
    hh20 = high.shift(1).rolling(LL_PERIOD).max().iloc[-1]
    ll20 = low.shift(1).rolling(LL_PERIOD).min().iloc[-1]
    close_now = close.iloc[-1]
    
    vma10 = volume.rolling(10).mean().iloc[-1]
    vma30 = volume.rolling(30).mean().iloc[-1]
    vol_ratio = vma10 / vma30 if vma30 > 0 else 999
    
    # Syarat Mutlak Breakout
    if val_ma20 < MIN_LIQUIDITY: return None
    if vol_ratio >= VOL_RATIO_MAX: return None
    if close_now < ll20: return None # Struktur tidak aman
    if close_now < hh20: return None # Bukan breakout

    stoch_k = calculate_stochastic(high, low, close).iloc[-1]
    
    # Kalkulasi Skor Baru (Dominasi Momentum & No Supply)
    # Vol Ratio menyumbang 50%: Makin mendekati 0, makin tinggi skor
    vol_score = max(0, (1.0 - vol_ratio) * 50) 
    # Momentum menyumbang 50%: Stochastic makin tinggi (kuat), makin dipercaya
    momentum_score = min(50, (stoch_k / 100) * 50) 
    buy_conf = min(99, vol_score + momentum_score)
    
    # GERBANG KONDISI MAKRO (Confidence Threshold)
    if regime == "BEAR" and buy_conf < 80:
        return None # Blokir saham medioker saat krisis

    return {
        "ticker": ticker, "close": float(close_now), "val_ma20": float(val_ma20),
        "hh20": float(hh20), "ll20": float(ll20), "vol_ratio": float(vol_ratio), 
        "stoch_k": float(stoch_k), "buy_confidence": float(buy_conf)
    }

# =====================================================================
# ANTARMUKA APLIKASI
# =====================================================================
st.title("Compounding Screener")
st.markdown("Sistem Breakout Murni dengan Hysteresis Makro & Volume No-Supply.")

# Cek Makro IHSG
regime, current_dd = get_macro_regime()
regime_color = "var(--bear)" if regime == "BEAR" else "var(--bull)"

st.markdown(f"""
<div class="macro-panel">
    <div class="macro-title">Status Makro IHSG Saat Ini</div>
    <div class="macro-status" style="color: {regime_color};">{regime} REGIME</div>
    <div style="color: var(--text-dim); font-size: 13px; font-family: 'IBM Plex Mono', monospace; margin-top: 8px;">
        IHSG Drawdown: <span style="color: {'var(--red)' if current_dd < -5 else 'var(--text)'}">{current_dd:.2f}%</span> 
        | Syarat Confidence: {'<b>≥ 80%</b> (Sangat Ketat)' if regime == 'BEAR' else '<b>Semua Valid</b> (Normal)'}
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Pindai Pasar (Breakout Mode)"):
    with st.spinner(f"Memindai pasar dalam kondisi {regime}..."):
        df_market = fetch_market_data(TICKERS)
    if df_market is None: 
        st.error("Gagal menarik data. Periksa koneksi internet Anda.")
        st.stop()

    k_breakout = []
    
    for t in TICKERS:
        df_stock = df_market[f"{t}.JK"] if isinstance(df_market.columns, pd.MultiIndex) and f"{t}.JK" in df_market.columns.get_level_values(0) else df_market if not isinstance(df_market.columns, pd.MultiIndex) else None
        if df_stock is None: continue
            
        m = analyze_stock(df_stock, t, regime)
        if m: k_breakout.append(m)

    if not k_breakout:
        st.warning(f"Tidak ada peluru emas hari ini. Modal aman di Kas. (Mode: {regime})")
        st.stop()

    k_breakout.sort(key=lambda x: x["buy_confidence"], reverse=True)

    st.subheader(f"⚡ Sinyal Eksekusi ({len(k_breakout)} Saham)")
    st.caption("SOP Sizing: Tetap Rp 10 Juta/Posisi. Harga telah menembus HH20 dengan Volume Ratio < 1.0.")
    
    for c in k_breakout:
        conf_col = "var(--green)" if c['buy_confidence'] > 80 else ("#D2A8FF" if c['buy_confidence'] > 60 else "var(--accent-2)")
        st.markdown(f"""
        <div class="card-container">
            <div class="header-row">
                <span class="stock-title">{c['ticker']}</span>
            </div>
            <div style="grid-column: 1 / -1; margin-top: 4px; margin-bottom: 8px;">
                <div class="conf-header">
                    <span style="color: var(--text-dim);">Konfirmasi Kekuatan Momentum & Volume</span>
                    <span style="color:{conf_col}; font-weight:700;">{c['buy_confidence']:.0f}%</span>
                </div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: {c['buy_confidence']}%; background: {conf_col};"></div></div>
            </div>
            <div class="metric-grid">
                <div class="metric-item">Harga Entry <span class="metric-val" style="color:var(--green)">{c['close']:,.0f}</span></div>
                <div class="metric-item">Stochastic (Kekuatan) <span class="metric-val" style="color:var(--accent-2)">{c['stoch_k']:.1f}</span></div>
                <div class="metric-item">No Supply Ratio <span class="metric-val">{c['vol_ratio']:.2f}x</span></div>
                <div class="metric-item">Nilai Trx 20H <span class="metric-val">Rp {c['val_ma20']/1e9:.1f} M</span></div>
                <div class="metric-item">Batas Cut Loss (LL20) <span class="metric-val" style="color:var(--red)">{c['ll20']:,.0f}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info("SOP KELUAR (EXIT): Take Profit (TP) saat Stochastic K > 80 selama 2 hari beruntun. Cut Loss (CL) mutlak jika harga menyentuh angka LL20.")

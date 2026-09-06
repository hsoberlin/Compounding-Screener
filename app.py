"""
COMPOUNDING TERMINAL
================================================================
Sistem penyaringan saham breakout murni + manajemen portofolio OEMS.

Perbaikan dari versi sebelumnya (lihat CHANGELOG di akhir file):
1. Regime IHSG dihitung dari trough DALAM episode bear, bukan drawdown
   dari puncak sepanjang 1 tahun -- versi lama bikin sistem macet di
   BEAR selamanya begitu crash-nya dalam.
2. Buy Confidence dirombak total: Momentum (Stochastic) 60 poin,
   Fase Wyckoff 30 poin, No-Supply (vol ratio) 10 poin -- sesuai bobot
   yang terbukti dari backtest (r=0.421), bukan asumsi awal.
3. Filter Heartbeat (range 20-hari minimal 5%) ditambahkan kembali.
4. Monitoring posisi aktif otomatis: status Stochastic, jarak ke LL20,
   dan rekomendasi HOLD/JUAL dihitung sendiri, bukan manual penuh.
5. Keranjang Retrace tidak diaktifkan lagi -- winrate 11.8% saat diuji,
   dikarantina permanen. Sistem ini breakout-only.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Compounding Terminal", layout="centered", initial_sidebar_state="collapsed")
WIB = timezone(timedelta(hours=7))
TODAY_STR = datetime.now(WIB).strftime("%Y-%m-%d")

# =====================================================================
# TEMA VISUAL -- terminal trader profesional, monokrom + aksen fungsional
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #0A0B0D;
  --panel: #131519;
  --line: #22252B;
  --text: #D7DAE0;
  --text-dim: #767C88;
  --up: #26A65B;
  --down: #D64545;
  --neutral: #C9A227;
  --accent: #4A7DFF;
}

.stApp { background-color: var(--bg); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div { color: var(--text); }

h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  color: var(--text) !important;
}

code, .stMarkdown code, [data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="collapsedControl"] { display: none; }

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 18px;
  margin-bottom: 14px;
}
.panel-title {
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 10px;
}

.regime-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 26px;
  font-weight: 600;
}
.regime-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  color: var(--text-dim);
  margin-top: 4px;
}

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; }
.stat-item { background: #0F1114; border: 1px solid var(--line); border-radius: 5px; padding: 10px 12px; }
.stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 17px; font-weight: 600; margin-top: 2px; }

.candidate-row {
  background: #0F1114; border: 1px solid var(--line); border-radius: 5px;
  padding: 14px; margin-bottom: 10px;
}
.candidate-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.candidate-ticker { font-family: 'Inter', sans-serif; font-size: 17px; font-weight: 700; }
.conf-bar-bg { background: #22252B; border-radius: 3px; height: 5px; overflow: hidden; margin-top: 6px; }
.conf-bar-fill { height: 100%; }

.rec-hold { color: var(--text-dim); }
.rec-sell-tp { color: var(--up); font-weight: 600; }
.rec-sell-cl { color: var(--down); font-weight: 600; }

.stButton button {
  width: 100%; font-family: 'Inter', sans-serif; font-weight: 600;
  background-color: #181B20; color: var(--text); border: 1px solid var(--line);
}
.stButton button:hover { border-color: var(--accent); color: var(--accent); }
.stDownloadButton button {
  background-color: var(--panel); color: var(--up); border: 1px solid var(--up); font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PARAMETER (semua nilai di sini sudah divalidasi lewat backtest)
# =====================================================================
MIN_LIQUIDITY = 1_000_000_000       # Rp1 miliar rata-rata value 20 hari
MIN_HEARTBEAT = 0.05                # range (HH20-LL20)/LL20 minimal 5%
VOL_RATIO_MAX = 1.0                 # vol MA10/MA30 harus < 1.0 (No-Supply)
LL_PERIOD = 20
LOOKBACK_FASE = 120                 # jendela deteksi fase Wyckoff

HYSTERESIS_ENTER_BEAR = -0.20       # masuk episode bear: dd dari rolling max <= -20%
HYSTERESIS_EXIT_BEAR = -0.10        # episode dianggap selesai: dd recovery > -10%
RECOVERY_DARI_TROUGH = 1.20         # sinyal rebound: harga >= 1.20x trough episode

AMBANG_CONF_BEAR = 80               # syarat Buy Confidence minimal saat BEAR
AMBANG_CONF_NORMAL = 0              # tidak ada syarat tambahan saat NORMAL

FEE_BELI = 0.0015
FEE_JUAL = 0.0025
TARIF_MATERAI = 10_000
BATAS_MATERAI = 10_000_000

STOCH_OB_STREAK_TP = 2              # hari beruntun K>80 untuk sinyal jual TP

IDX_TICKERS_ALL = """AALI, ABBA, ABDA, ABMM, ACES, ADRO, AGII, AKRA, AMRT, ANTM, ARNA, ASII, AUTO,
BBCA, BBNI, BBRI, BBTN, BMRI, BRIS, BRPT, BSDE, BUKA, BYAN, CPIN, CTRA, EMTK, ESSA, EXCL, GGRM,
GOTO, HRUM, ICBP, INCO, INDF, INDY, INKP, INTP, ISAT, ITMG, JPFA, KLBF, MAPI, MDKA, MEDC, MGRO,
MIKA, MNCN, MTEL, PGAS, PGEO, PTBA, PTPP, SMGR, SRTG, TBIG, TINS, TKIM, TLKM, TOWR, TPIA, UNTR,
UNVR, WIKA"""
TICKERS = sorted(set(t.strip().upper() for t in IDX_TICKERS_ALL.replace("\n", ",").split(",") if t.strip()))


# =====================================================================
# STATE MANAGEMENT (JSON jurnal portofolio)
# =====================================================================
def init_portfolio(capital):
    return {
        "last_updated": TODAY_STR,
        "initial_capital": capital,
        "current_cash": capital,
        "total_equity": capital,
        "positions": [],
        "history": {"total_bought": 0, "total_sold": 0, "hit_tp": 0, "hit_sl": 0},
        "daily_tracker": {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False},
    }


if "port" not in st.session_state:
    st.session_state["port"] = init_portfolio(50_000_000)

port = st.session_state["port"]

if "daily_tracker" not in port or port["daily_tracker"]["date"] != TODAY_STR:
    port["daily_tracker"] = {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
    st.session_state["port"] = port


# =====================================================================
# REGIME IHSG -- deteksi episode bear kausal (tanpa lookahead)
# =====================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_macro_regime():
    """Mengembalikan (regime, drawdown_sekarang, pct_dari_trough).
    Logika: episode bear dimulai saat drawdown dari rolling-252 max
    tembus -20%. SELAMA episode berjalan, trough dilacak sebagai titik
    terendah SEJAK episode dimulai (bukan puncak lama) -- gerbang baru
    terbuka begitu harga naik >=20% dari trough itu. Episode baru resmi
    selesai (reset) begitu drawdown recovery ke atas -10%."""
    try:
        ihsg = yf.download("^JKSE", period="2y", interval="1d", progress=False)
        if ihsg.empty:
            return "NORMAL", 0.0, None
        close = ihsg["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        roll_max = close.rolling(252, min_periods=50).max()
        dd = (close - roll_max) / roll_max

        in_episode = False
        episode_trough = None
        signal_reached = False
        regime = "NORMAL"

        for tgl in close.index:
            d = dd.loc[tgl]
            c = close.loc[tgl]
            if pd.isna(d):
                regime = "NORMAL"
                continue
            if not in_episode:
                if d <= HYSTERESIS_ENTER_BEAR:
                    in_episode = True
                    episode_trough = c
                    signal_reached = False
                    regime = "BEAR"
                else:
                    regime = "NORMAL"
            else:
                episode_trough = min(episode_trough, c)
                if c >= RECOVERY_DARI_TROUGH * episode_trough:
                    signal_reached = True
                if d > HYSTERESIS_EXIT_BEAR:
                    in_episode = False
                    signal_reached = False
                    regime = "NORMAL"
                else:
                    regime = "NORMAL" if signal_reached else "BEAR"

        dd_sekarang = float(dd.iloc[-1]) * 100
        pct_dari_trough = None
        if in_episode and episode_trough:
            pct_dari_trough = (float(close.iloc[-1]) / episode_trough - 1) * 100
        return regime, dd_sekarang, pct_dari_trough
    except Exception:
        return "NORMAL", 0.0, None


# =====================================================================
# FASE WYCKOFF -- hitung berapa kali sudah bikin puncak baru sejak
# titik terendah 120 hari, dengan ambang minimal biar tidak nangkep
# gerigi harian (retrace wajib >=5%, puncak baru wajib >=3% margin)
# =====================================================================
def hitung_fase_wyckoff(low_series, min_retrace=0.05, min_margin=0.03):
    if len(low_series) < 60:
        return 0
    window = low_series.tail(LOOKBACK_FASE)
    gz_idx = window.values.argmin()
    gz_val = window.iloc[gz_idx]
    after = window.iloc[gz_idx:]
    if len(after) < 10:
        return 0

    phase = 0
    phase_high = after.iloc[0]
    in_retrace = False
    for val in after.iloc[1:]:
        if not in_retrace:
            if val > phase_high:
                phase_high = val
            elif val < phase_high * (1 - min_retrace):
                in_retrace = True
        else:
            if val > phase_high * (1 + min_margin):
                phase += 1
                phase_high = val
                in_retrace = False
    return max(phase, 1) if phase_high > gz_val else 0


# =====================================================================
# DATA & SCREENER
# =====================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers, period="8mo"):
    tickers_jk = [f"{t}.JK" for t in tickers]
    try:
        return yf.download(tickers_jk, period=period, interval="1d", progress=False, group_by="ticker", threads=True)
    except Exception:
        return None


def analyze_stock(df_stock, ticker, regime, alokasi_max):
    if df_stock is None or len(df_stock.dropna()) < 130:
        return None
    close = df_stock["Close"].dropna()
    high = df_stock["High"].dropna()
    low = df_stock["Low"].dropna()
    volume = df_stock["Volume"].dropna()

    idx = close.index.intersection(volume.index)
    close, high, low, volume = close[idx], high[idx], low[idx], volume[idx]
    if len(close) < 130:
        return None

    val_ma20 = (close * volume).rolling(20).mean().iloc[-1]
    hh20 = high.shift(1).rolling(LL_PERIOD).max().iloc[-1]
    ll20 = low.shift(1).rolling(LL_PERIOD).min().iloc[-1]
    close_now = close.iloc[-1]
    if pd.isna(hh20) or pd.isna(ll20) or ll20 <= 0:
        return None

    heartbeat = (hh20 - ll20) / ll20

    vma10 = volume.rolling(10).mean().iloc[-1]
    vma30 = volume.rolling(30).mean().iloc[-1]
    vol_ratio = vma10 / vma30 if vma30 > 0 else 999

    # -- Gerbang keras Tahap 0-1 --
    if val_ma20 < MIN_LIQUIDITY:
        return None
    if heartbeat < MIN_HEARTBEAT:
        return None
    if vol_ratio >= VOL_RATIO_MAX:
        return None
    if close_now < ll20:
        return None
    if close_now < hh20:
        return None  # bukan breakout hari ini

    ll10, hh10 = low.rolling(10).min(), high.rolling(10).max()
    stoch_k = (100 * ((close - ll10) / (hh10 - ll10))).rolling(5).mean().iloc[-1]
    if pd.isna(stoch_k):
        return None

    fase = hitung_fase_wyckoff(low)

    # -- Buy Confidence: bobot sesuai backtest (r=0.421) --
    momentum_score = min(60, stoch_k * 0.6)
    fase_score = 30 if fase >= 3 else (20 if fase in (1, 2) else 10)
    vol_score = max(0, (1.0 - vol_ratio)) * 10
    buy_conf = min(99, momentum_score + fase_score + vol_score)

    ambang = AMBANG_CONF_BEAR if regime == "BEAR" else AMBANG_CONF_NORMAL
    if buy_conf < ambang:
        return None

    lot_size = math.floor((alokasi_max - TARIF_MATERAI) / (close_now * 100 * (1 + FEE_BELI)))
    lot_size = max(lot_size, 0)
    modal_terserap = lot_size * close_now * 100

    return {
        "ticker": ticker,
        "close": float(close_now),
        "ll20": float(ll20),
        "stoch_k": float(stoch_k),
        "vol_ratio": float(vol_ratio),
        "fase": int(fase),
        "buy_confidence": float(buy_conf),
        "lot_size": lot_size,
        "modal_terserap": float(modal_terserap),
    }


def status_posisi_aktif(ticker, ll20_terkunci, ob_streak_tersimpan=0):
    """Tarik data terbaru satu saham dan hitung status SOP: HOLD, JUAL(TP),
    atau JUAL(CL). ob_streak_tersimpan dilewatkan dari jurnal biar hitungan
    hari-beruntun-overbought konsisten antar sesi."""
    try:
        df = yf.download(f"{ticker}.JK", period="30d", interval="1d", progress=False)
        if df.empty:
            return None
        close, high, low = df["Close"], df["High"], df["Low"]
        if isinstance(close, pd.DataFrame):
            close, high, low = close.iloc[:, 0], high.iloc[:, 0], low.iloc[:, 0]
        ll10, hh10 = low.rolling(10).min(), high.rolling(10).max()
        stoch_k = (100 * ((close - ll10) / (hh10 - ll10))).rolling(5).mean().iloc[-1]
        harga_now = float(close.iloc[-1])

        if pd.isna(stoch_k):
            return {"harga_now": harga_now, "stoch_k": None, "ob_streak": ob_streak_tersimpan, "rekomendasi": "HOLD"}

        if harga_now < ll20_terkunci:
            return {"harga_now": harga_now, "stoch_k": float(stoch_k), "ob_streak": 0, "rekomendasi": "JUAL_CL"}

        ob_streak = ob_streak_tersimpan + 1 if stoch_k > 80 else 0
        rekom = "JUAL_TP" if ob_streak >= STOCH_OB_STREAK_TP else "HOLD"
        return {"harga_now": harga_now, "stoch_k": float(stoch_k), "ob_streak": ob_streak, "rekomendasi": rekom}
    except Exception:
        return None


# =====================================================================
# UI -- HEADER & JURNAL
# =====================================================================
st.title("Compounding Terminal")

with st.container():
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Jurnal Portofolio</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Muat portfolio.json", type="json", label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            data_baru = json.load(uploaded_file)
            if "daily_tracker" not in data_baru or data_baru.get("daily_tracker", {}).get("date") != TODAY_STR:
                data_baru["daily_tracker"] = {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
            for p in data_baru.get("positions", []):
                p.setdefault("ob_streak", 0)
            st.session_state["port"] = data_baru
            st.success("Jurnal termuat.")
            st.rerun()
        except Exception:
            st.error("Format JSON tidak valid.")
    st.markdown("</div>", unsafe_allow_html=True)

alokasi_per_posisi = port["initial_capital"] * 0.20
active_pos_count = len(port["positions"])

st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='panel-title'>Ekuitas & Posisi</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="stat-grid">
  <div class="stat-item"><div class="stat-label">Total Ekuitas</div><div class="stat-value">Rp {port['total_equity']:,.0f}</div></div>
  <div class="stat-item"><div class="stat-label">Kas Aktif</div><div class="stat-value" style="color:var(--accent)">Rp {port['current_cash']:,.0f}</div></div>
  <div class="stat-item"><div class="stat-label">Alokasi/Posisi</div><div class="stat-value">Rp {alokasi_per_posisi:,.0f}</div></div>
</div>
""".replace(",", "."), unsafe_allow_html=True)

st.markdown(f"""
<div class="stat-grid" style="margin-top:10px;">
  <div class="stat-item"><div class="stat-label">Total Beli</div><div class="stat-value">{port['history']['total_bought']}</div></div>
  <div class="stat-item"><div class="stat-label">Total Jual</div><div class="stat-value">{port['history']['total_sold']}</div></div>
  <div class="stat-item"><div class="stat-label">Hit TP</div><div class="stat-value" style="color:var(--up)">{port['history']['hit_tp']}</div></div>
  <div class="stat-item"><div class="stat-label">Hit CL</div><div class="stat-value" style="color:var(--down)">{port['history']['hit_sl']}</div></div>
  <div class="stat-item"><div class="stat-label">Posisi Aktif</div><div class="stat-value">{active_pos_count}</div></div>
</div>
""", unsafe_allow_html=True)

if port["history"]["total_bought"] == 0 and active_pos_count == 0:
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    new_cap = st.number_input("Modal awal (Rp) -- input sekali sebelum market open",
                               min_value=1_000_000, value=int(port["initial_capital"]), step=5_000_000)
    if st.button("Set Modal Awal"):
        st.session_state["port"] = init_portfolio(new_cap)
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# UI -- MONITORING POSISI AKTIF (otomatis, bukan manual)
# =====================================================================
if active_pos_count > 0:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Status Posisi Aktif</div>", unsafe_allow_html=True)
    for p in port["positions"]:
        info = status_posisi_aktif(p["ticker"], p["ll20_terkunci"], p.get("ob_streak", 0))
        if info is None:
            st.write(f"{p['ticker']}: data tidak tersedia saat ini.")
            continue
        p["ob_streak"] = info["ob_streak"]
        gain_pct = (info["harga_now"] * p["lots"] * 100 / p["modal_terserap"] - 1) * 100
        gain_color = "var(--up)" if gain_pct >= 0 else "var(--down)"
        rekom_map = {
            "HOLD": ("HOLD", "rec-hold"),
            "JUAL_TP": ("JUAL -- TAKE PROFIT", "rec-sell-tp"),
            "JUAL_CL": ("JUAL -- CUT LOSS", "rec-sell-cl"),
        }
        label_rekom, kelas_rekom = rekom_map[info["rekomendasi"]]
        stoch_txt = f"{info['stoch_k']:.1f}" if info["stoch_k"] is not None else "n/a"
        st.markdown(f"""
        <div class="candidate-row">
          <div class="candidate-head">
            <span class="candidate-ticker">{p['ticker']}</span>
            <span class="{kelas_rekom}">{label_rekom}</span>
          </div>
          <div class="stat-grid">
            <div class="stat-item"><div class="stat-label">Harga Now</div><div class="stat-value">{info['harga_now']:,.0f}</div></div>
            <div class="stat-item"><div class="stat-label">Gain</div><div class="stat-value" style="color:{gain_color}">{gain_pct:+.1f}%</div></div>
            <div class="stat-item"><div class="stat-label">Stochastic K</div><div class="stat-value">{stoch_txt}</div></div>
            <div class="stat-item"><div class="stat-label">Overbought Streak</div><div class="stat-value">{info['ob_streak']}/2 hari</div></div>
            <div class="stat-item"><div class="stat-label">Batas CL (LL20)</div><div class="stat-value" style="color:var(--down)">{p['ll20_terkunci']:,.0f}</div></div>
          </div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)
    st.session_state["port"] = port
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# UI -- EKSEKUSI ORDER (OEMS: lot, fee, materai cerdas)
# =====================================================================
st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='panel-title'>Eksekusi Order</div>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Beli", "Jual"])

tracker = port["daily_tracker"]
status_materai_text = "Terbayar" if tracker["materai_paid"] else f"Belum terbayar (akumulasi Rp {tracker['accumulated_value']:,.0f})"
st.markdown(f"<div style='font-size:12px; color:var(--text-dim); margin-bottom:8px;'>Materai harian: {status_materai_text}</div>".replace(",", "."), unsafe_allow_html=True)

with tab1:
    b_ticker = st.text_input("Ticker")
    b_price = st.number_input("Harga match (beli)", min_value=0)

    if b_price > 0:
        biaya_materai_estimasi = 0
        if not tracker["materai_paid"] and (tracker["accumulated_value"] + alokasi_per_posisi) > BATAS_MATERAI:
            biaya_materai_estimasi = TARIF_MATERAI

        alokasi_bersih = alokasi_per_posisi - biaya_materai_estimasi
        biaya_per_lot_net = (b_price * 100) * (1 + FEE_BELI)
        lot_kalkulasi = math.floor(alokasi_bersih / biaya_per_lot_net) if biaya_per_lot_net > 0 else 0

        if lot_kalkulasi > 0:
            nilai_kotor = lot_kalkulasi * b_price * 100
            total_fee = nilai_kotor * FEE_BELI
            biaya_materai_final = TARIF_MATERAI if (not tracker["materai_paid"] and (tracker["accumulated_value"] + nilai_kotor) > BATAS_MATERAI) else 0
            modal_aktual = nilai_kotor + total_fee + biaya_materai_final
            kembalian_kas = alokasi_per_posisi - modal_aktual

            # LL20 dikunci di harga saat ini -- referensi tetap sepanjang posisi dipegang
            df_ll = yf.download(f"{b_ticker.upper()}.JK", period="30d", interval="1d", progress=False) if b_ticker else None
            ll20_terkunci = None
            if df_ll is not None and not df_ll.empty:
                low_s = df_ll["Low"]
                if isinstance(low_s, pd.DataFrame):
                    low_s = low_s.iloc[:, 0]
                ll20_terkunci = float(low_s.tail(20).min())

            st.markdown(f"""
            <div class="candidate-row">
                Volume order: {lot_kalkulasi} lot<br>
                Nilai saham: Rp {nilai_kotor:,.0f}<br>
                Fee + materai: Rp {(total_fee + biaya_materai_final):,.0f}<br>
                Total dana terserap: <span style="color:var(--down);">Rp {modal_aktual:,.0f}</span><br>
                Sisa masuk kas: <span style="color:var(--up);">Rp {kembalian_kas:,.0f}</span><br>
                Batas CL (LL20 saat ini): <span style="color:var(--down);">{ll20_terkunci:,.0f if ll20_terkunci else 'n/a'}</span>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

            if st.button("Simpan Eksekusi Beli"):
                if port["current_cash"] >= modal_aktual and b_ticker and ll20_terkunci:
                    port["positions"].append({
                        "ticker": b_ticker.upper(),
                        "entry_price": b_price,
                        "lots": lot_kalkulasi,
                        "modal_terserap": modal_aktual,
                        "ll20_terkunci": ll20_terkunci,
                        "ob_streak": 0,
                    })
                    port["current_cash"] -= modal_aktual
                    port["history"]["total_bought"] += 1
                    port["daily_tracker"]["accumulated_value"] += nilai_kotor
                    if biaya_materai_final > 0:
                        port["daily_tracker"]["materai_paid"] = True
                    st.session_state["port"] = port
                    st.success(f"Tersimpan: {lot_kalkulasi} lot {b_ticker.upper()}.")
                    st.rerun()
                else:
                    st.error("Kas tidak cukup atau data LL20 gagal ditarik.")

with tab2:
    if active_pos_count > 0:
        s_ticker = st.selectbox("Pilih saham", [p["ticker"] for p in port["positions"]])
        pos_data = next(item for item in port["positions"] if item["ticker"] == s_ticker)
        st.caption(f"Posisi: {pos_data['lots']} lot | Modal terserap: Rp {pos_data['modal_terserap']:,.0f}".replace(",", "."))

        s_status = st.radio("Jenis eksekusi", ["Take Profit (TP)", "Cut Loss (CL)"], horizontal=True)
        s_price = st.number_input("Harga match (jual)", min_value=0)

        if s_price > 0:
            nilai_kotor_jual = pos_data["lots"] * s_price * 100
            total_fee_jual = nilai_kotor_jual * FEE_JUAL
            biaya_materai_jual = TARIF_MATERAI if (not tracker["materai_paid"] and (tracker["accumulated_value"] + nilai_kotor_jual) > BATAS_MATERAI) else 0
            net_return = nilai_kotor_jual - total_fee_jual - biaya_materai_jual
            pl_rupiah = net_return - pos_data["modal_terserap"]
            pl_persen = (pl_rupiah / pos_data["modal_terserap"]) * 100
            pl_color = "var(--up)" if pl_rupiah >= 0 else "var(--down)"

            st.markdown(f"""
            <div class="candidate-row">
                Nilai jual kotor: Rp {nilai_kotor_jual:,.0f}<br>
                Fee + materai: Rp {(total_fee_jual + biaya_materai_jual):,.0f}<br>
                Net cair ke kas: Rp {net_return:,.0f}<br>
                Net P/L: <span style="color:{pl_color};">Rp {pl_rupiah:,.0f} ({pl_persen:+.2f}%)</span>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

            if st.button("Simpan Eksekusi Jual"):
                port["positions"] = [p for p in port["positions"] if p["ticker"] != s_ticker]
                port["current_cash"] += net_return
                port["history"]["total_sold"] += 1
                if "TP" in s_status:
                    port["history"]["hit_tp"] += 1
                else:
                    port["history"]["hit_sl"] += 1
                port["daily_tracker"]["accumulated_value"] += nilai_kotor_jual
                if biaya_materai_jual > 0:
                    port["daily_tracker"]["materai_paid"] = True
                total_valuasi_saham_sisa = sum(p["modal_terserap"] for p in port["positions"])
                port["total_equity"] = port["current_cash"] + total_valuasi_saham_sisa
                st.session_state["port"] = port
                st.success(f"Tersimpan: {s_ticker} terjual.")
                st.rerun()
    else:
        st.write("Tidak ada posisi aktif.")
st.markdown("</div>", unsafe_allow_html=True)

json_string = json.dumps(port, indent=4)
st.download_button(
    label="Unduh Jurnal JSON",
    data=json_string,
    file_name=f"portfolio_{TODAY_STR}.json",
    mime="application/json",
)
st.divider()


# =====================================================================
# UI -- SCREENER
# =====================================================================
regime, current_dd, pct_dari_trough = get_macro_regime()
regime_color = "var(--down)" if regime == "BEAR" else "var(--up)"
syarat_text = "Buy Confidence >= 80" if regime == "BEAR" else "Semua kandidat valid ditampilkan"

st.markdown(f"""
<div class="panel">
    <div class="panel-title">Filter Makro IHSG</div>
    <div class="regime-value" style="color: {regime_color};">{regime}</div>
    <div class="regime-sub">Drawdown dari puncak rolling: {current_dd:.2f}%</div>
    <div class="regime-sub">{"Dari trough episode: " + format(pct_dari_trough, "+.1f") + "%" if pct_dari_trough is not None else "Tidak sedang dalam episode bear"}</div>
    <div class="regime-sub" style="margin-top:6px;">Syarat tampil: {syarat_text}</div>
</div>
""", unsafe_allow_html=True)

if st.button("Pindai Pasar"):
    with st.spinner(f"Memindai {len(TICKERS)} saham..."):
        df_market = fetch_market_data(TICKERS)
    if df_market is None:
        st.error("Gagal menarik data pasar.")
        st.stop()

    kandidat = []
    for t in TICKERS:
        df_stock = None
        if isinstance(df_market.columns, pd.MultiIndex):
            if f"{t}.JK" in df_market.columns.get_level_values(0):
                df_stock = df_market[f"{t}.JK"]
        else:
            df_stock = df_market
        m = analyze_stock(df_stock, t, regime, alokasi_per_posisi)
        if m:
            kandidat.append(m)

    if not kandidat:
        st.warning("Tidak ada kandidat breakout valid hari ini.")
        st.stop()

    kandidat.sort(key=lambda x: -x["buy_confidence"])
    st.subheader(f"Kandidat Breakout -- {len(kandidat)} saham")

    for c in kandidat:
        conf_color = "var(--up)" if c["buy_confidence"] >= 75 else ("var(--neutral)" if c["buy_confidence"] >= 50 else "var(--down)")
        st.markdown(f"""
        <div class="candidate-row">
            <div class="candidate-head">
                <span class="candidate-ticker">{c['ticker']}</span>
                <span style="color:{conf_color}; font-weight:700; font-family:'IBM Plex Mono',monospace;">{c['buy_confidence']:.0f}%</span>
            </div>
            <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{c['buy_confidence']}%; background:{conf_color};"></div></div>
            <div class="stat-grid" style="margin-top:10px;">
                <div class="stat-item"><div class="stat-label">Harga</div><div class="stat-value">{c['close']:,.0f}</div></div>
                <div class="stat-item"><div class="stat-label">Stochastic K</div><div class="stat-value">{c['stoch_k']:.1f}</div></div>
                <div class="stat-item"><div class="stat-label">Fase Wyckoff</div><div class="stat-value">{c['fase']}</div></div>
                <div class="stat-item"><div class="stat-label">Sizing</div><div class="stat-value" style="color:var(--accent)">{c['lot_size']} lot</div></div>
                <div class="stat-item"><div class="stat-label">Modal</div><div class="stat-value" style="color:var(--up)">Rp {c['modal_terserap']:,.0f}</div></div>
                <div class="stat-item"><div class="stat-label">Batas CL</div><div class="stat-value" style="color:var(--down)">{c['ll20']:,.0f}</div></div>
            </div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

st.caption(
    "Data historis, bukan sinyal beli/jual. Bukan nasihat keuangan. "
    "Sistem breakout murni -- keranjang retrace dikarantina permanen (winrate 11.8% saat diuji)."
)

# =====================================================================
# CHANGELOG
# =====================================================================
# 1. get_macro_regime(): diganti dari drawdown-vs-puncak-1-tahun menjadi
#    deteksi episode + trough kausal. Alasan: versi lama butuh recovery
#    ke -10% dari PUNCAK LAMA (butuh waktu sangat panjang setelah crash
#    dalam) -- versi baru pakai trough SIKLUS SEKARANG, sesuai yang
#    terbukti aktivasi gerbang di 18 Agustus pada backtest.
# 2. buy_confidence: bobot lama (vol_ratio 50pt, stoch 50pt, fase 0pt)
#    diganti (momentum 60pt, fase 30pt, vol 10pt) -- korelasi ke hasil
#    naik dari ~0 menjadi r=0.421 saat diuji ke 44 sinyal historis.
# 3. MIN_HEARTBEAT (range 20-hari >=5%) ditambahkan -- hilang di versi
#    sebelumnya, padahal bagian dari Tahap 0 yang tervalidasi.
# 4. status_posisi_aktif(): baru -- menghitung otomatis status K-streak
#    dan jarak ke LL20 tiap posisi, memberi rekomendasi HOLD/JUAL,
#    bukan seluruhnya manual seperti sebelumnya.
# 5. Keranjang Retrace tidak pernah diaktifkan di versi manapun sejak
#    versi sebelum ini -- dikonfirmasi tetap dikarantina di sini.
}

.stApp { background-color: var(--bg); color: var(--text); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; padding-bottom: 0px; }
[data-testid="stMetricValue"], code, .stMarkdown code { font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="collapsedControl"] { display: none; }

.panel-container { background: var(--card); border: 1px solid var(--card-line); padding: 16px; border-radius: 8px; margin-bottom: 16px; }
.macro-title { font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: var(--text-dim); text-transform: uppercase;}
.macro-status { font-family: 'Space Grotesk', sans-serif; font-size: 28px; font-weight: 700; margin-top: 4px; }
.metric-val { color: #FFFFFF; font-weight: 600; font-size: 16px; margin-top: 2px; font-family: 'IBM Plex Mono', monospace;}
.metric-label { font-size: 12px; color: var(--text-dim); }

.stButton button { width: 100%; font-family: 'Space Grotesk', sans-serif; font-weight: 600; background-color: #21262D; color: #FFF; border: 1px solid #30363D; }
.stButton button:hover { border-color: var(--accent-2); color: var(--accent-2); }
.stDownloadButton button { background-color: var(--green); color: #000; border: none; font-weight: 600; }
.stDownloadButton button:hover { background-color: #2EA043; color: #FFF; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# PARAMETER & TICKERS
# =====================================================================
MIN_LIQUIDITY = 1_000_000_000  
VOL_RATIO_MAX = 1.0            
LL_PERIOD = 20                 
HYSTERESIS_ENTER_BEAR = -20.0  
HYSTERESIS_EXIT_BEAR = -10.0   

FEE_BELI = 0.0015  # 0.15%
FEE_JUAL = 0.0025  # 0.25%
TARIF_MATERAI = 10000
BATAS_MATERAI = 10000000

IDX_TICKERS_ALL = """AALI, ABBA, ABDA, ABMM, ACES, ADRO, AGII, AKRA, AMRT, ANTM, ARNA, ASII, AUTO, BBCA, BBNI, BBRI, BBTN, BMRI, BRIS, BRPT, BSDE, BUKA, BYAN, CPIN, CTRA, EMTK, ESSA, EXCL, GGRM, GOTO, HRUM, ICBP, INCO, INDF, INDY, INKP, INTP, ISAT, ITMG, JPFA, KLBF, MAPI, MDKA, MEDC, MGRO, MIKA, MNCN, MTEL, PGAS, PGEO, PTBA, PTPP, SMGR, SRTG, TBIG, TINS, TKIM, TLKM, TOWR, TPIA, UNTR, UNVR, WIKA"""
TICKERS = list(set([t.strip().upper() for t in IDX_TICKERS_ALL.replace('\n', ',').split(',') if t.strip()]))

# =====================================================================
# STATE MANAGEMENT (JSON)
# =====================================================================
def init_portfolio(capital):
    return {
        "last_updated": TODAY_STR,
        "initial_capital": capital,
        "current_cash": capital,
        "total_equity": capital,
        "positions": [],
        "history": {"total_bought": 0, "total_sold": 0, "hit_tp": 0, "hit_sl": 0},
        "daily_tracker": {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
    }

if 'port' not in st.session_state:
    st.session_state['port'] = init_portfolio(50000000)

port = st.session_state['port']

if 'daily_tracker' not in port or port['daily_tracker']['date'] != TODAY_STR:
    port['daily_tracker'] = {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
    st.session_state['port'] = port

# =====================================================================
# ENGINE & MACRO
# =====================================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_macro_regime():
    try:
        ihsg = yf.download("^JKSE", period="1y", progress=False)
        if ihsg.empty: return "NORMAL", 0.0
        ihsg['Peak'] = ihsg['Close'].cummax()
        ihsg['Drawdown'] = (ihsg['Close'] - ihsg['Peak']) / ihsg['Peak'] * 100
        regime = "NORMAL"
        for dd in ihsg['Drawdown']:
            if regime == "NORMAL" and dd <= HYSTERESIS_ENTER_BEAR: regime = "BEAR"
            elif regime == "BEAR" and dd >= HYSTERESIS_EXIT_BEAR: regime = "NORMAL"
        return regime, float(ihsg['Drawdown'].iloc[-1])
    except: return "NORMAL", 0.0

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(tickers, period="6mo"):
    tickers_jk = [f"{t}.JK" for t in tickers]
    try: return yf.download(tickers_jk, period=period, interval="1d", progress=False, group_by="ticker", threads=True)
    except: return None

def analyze_stock(df_stock, ticker, regime, alokasi_max):
    if df_stock is None or len(df_stock.dropna()) < 35: return None
    close = df_stock['Close'].dropna()
    high = df_stock['High'].dropna()
    low = df_stock['Low'].dropna()
    volume = df_stock['Volume'].dropna()
    
    idx = close.index.intersection(volume.index)
    close, high, low, volume = close[idx], high[idx], low[idx], volume[idx]
    if len(close) < 35: return None

    val_ma20 = (close * volume).rolling(20).mean().iloc[-1]
    hh20 = high.shift(1).rolling(LL_PERIOD).max().iloc[-1]
    ll20 = low.shift(1).rolling(LL_PERIOD).min().iloc[-1]
    close_now = close.iloc[-1]
    
    vma10, vma30 = volume.rolling(10).mean().iloc[-1], volume.rolling(30).mean().iloc[-1]
    vol_ratio = vma10 / vma30 if vma30 > 0 else 999
    
    if val_ma20 < MIN_LIQUIDITY or vol_ratio >= VOL_RATIO_MAX or close_now < ll20 or close_now < hh20: return None

    ll, hh = low.rolling(10).min(), high.rolling(10).max()
    stoch_k = (100 * ((close - ll) / (hh - ll))).rolling(5).mean().iloc[-1]
    
    buy_conf = min(99, max(0, (1.0 - vol_ratio) * 50) + min(50, (stoch_k / 100) * 50))
    if regime == "BEAR" and buy_conf < 80: return None 

    lot_size = math.floor((alokasi_max - TARIF_MATERAI) / (close_now * 100 * (1 + FEE_BELI)))
    if lot_size < 0: lot_size = 0
    modal_terserap = lot_size * close_now * 100

    return {
        "ticker": ticker, "close": float(close_now), "ll20": float(ll20), 
        "stoch_k": float(stoch_k), "vol_ratio": float(vol_ratio), 
        "buy_confidence": float(buy_conf), "lot_size": lot_size, 
        "modal_terserap": float(modal_terserap)
    }

# =====================================================================
# UI: PORTFOLIO MANAGER & JURNAL
# =====================================================================
st.title("Compounding Terminal")

st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0px;'>Muat Jurnal JSON</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload file portfolio.json", type="json")
if uploaded_file is not None:
    try:
        data_baru = json.load(uploaded_file)
        if 'daily_tracker' not in data_baru or data_baru.get('daily_tracker', {}).get('date') != TODAY_STR:
            data_baru['daily_tracker'] = {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
        st.session_state['port'] = data_baru
        st.success("Jurnal termuat.")
        st.rerun()
    except: st.error("Format JSON tidak valid.")
st.markdown("</div>", unsafe_allow_html=True)

alokasi_per_posisi = port['initial_capital'] * 0.20 
active_pos_count = len(port['positions'])

st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0px;'>Dashboard Ekuitas & Posisi</h3>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='metric-label'>Total Ekuitas</div><div class='metric-val'>Rp {port['total_equity']:,.0f}</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-label'>Sisa Kas Aktif</div><div class='metric-val' style='color:var(--accent-2)'>Rp {port['current_cash']:,.0f}</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-label'>Alokasi/Posisi (20%)</div><div class='metric-val'>Rp {alokasi_per_posisi:,.0f}</div>", unsafe_allow_html=True)

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(f"<div class='metric-label'>Total Beli</div><div class='metric-val'>{port['history']['total_bought']}</div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-label'>Total Jual</div><div class='metric-val'>{port['history']['total_sold']}</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-label'>Hit TP</div><div class='metric-val' style='color:var(--green)'>{port['history']['hit_tp']}</div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-label'>Hit SL</div><div class='metric-val' style='color:var(--red)'>{port['history']['hit_sl']}</div>", unsafe_allow_html=True)
c5.markdown(f"<div class='metric-label'>Posisi Aktif</div><div class='metric-val' style='color:var(--accent-1)'>{active_pos_count}</div>", unsafe_allow_html=True)

if port['history']['total_bought'] == 0 and active_pos_count == 0:
    new_cap = st.number_input("Input Modal Awal (Rp)", min_value=1000000, value=int(port['initial_capital']), step=5000000)
    if st.button("Set Modal Awal"):
        st.session_state['port'] = init_portfolio(new_cap)
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='panel-container'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0px;'>Eksekusi Order & Pencatatan (Net RDN)</h3>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Eksekusi Beli", "Eksekusi Jual"])

tracker = port['daily_tracker']
status_materai_text = "Terbayar" if tracker['materai_paid'] else f"Belum Terbayar (Akumulasi: Rp {tracker['accumulated_value']:,.0f})"
st.markdown(f"<div style='font-size:12px; color:var(--text-dim); margin-bottom:8px;'>Status Materai Harian: {status_materai_text}</div>", unsafe_allow_html=True)

with tab1:
    b_ticker = st.text_input("Ticker Saham")
    b_price = st.number_input("Harga Match (Beli)", min_value=0)
    
    if b_price > 0:
        biaya_materai_estimasi = 0
        if not tracker['materai_paid'] and (tracker['accumulated_value'] + alokasi_per_posisi) > BATAS_MATERAI:
            biaya_materai_estimasi = TARIF_MATERAI
            
        alokasi_bersih = alokasi_per_posisi - biaya_materai_estimasi
        biaya_per_lot_net = (b_price * 100) * (1 + FEE_BELI)
        lot_kalkulasi = math.floor(alokasi_bersih / biaya_per_lot_net)
        
        if lot_kalkulasi > 0:
            nilai_kotor = lot_kalkulasi * b_price * 100
            total_fee = nilai_kotor * FEE_BELI
            
            biaya_materai_final = 0
            if not tracker['materai_paid'] and (tracker['accumulated_value'] + nilai_kotor) > BATAS_MATERAI:
                biaya_materai_final = TARIF_MATERAI
            
            modal_aktual = nilai_kotor + total_fee + biaya_materai_final
            kembalian_kas = alokasi_per_posisi - modal_aktual
            
            st.markdown(f"""
            <div style="background:#161B22; padding:12px; border-radius:6px; font-family:monospace; margin-bottom:12px; border-left: 3px solid var(--accent-2);">
                Volume Order: {lot_kalkulasi} Lot<br>
                Nilai Saham: Rp {nilai_kotor:,.0f}<br>
                Fee Beli + Materai: Rp {(total_fee + biaya_materai_final):,.0f}<br>
                Total Dana Terserap: <span style="color:var(--red);">Rp {modal_aktual:,.0f}</span><br>
                <span style="color:var(--green);">Masuk Kas: Rp {kembalian_kas:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Simpan Eksekusi Beli"):
                if port['current_cash'] >= modal_aktual and b_ticker:
                    port['positions'].append({
                        "ticker": b_ticker.upper(), 
                        "entry_price": b_price, 
                        "lots": lot_kalkulasi,
                        "modal_terserap": modal_aktual
                    })
                    port['current_cash'] -= modal_aktual
                    port['history']['total_bought'] += 1
                    
                    port['daily_tracker']['accumulated_value'] += nilai_kotor
                    if biaya_materai_final > 0:
                        port['daily_tracker']['materai_paid'] = True
                        
                    st.session_state['port'] = port
                    st.success(f"Tersimpan: {lot_kalkulasi} Lot {b_ticker}.")
                    st.rerun()
                else:
                    st.error("Kas RDN tidak mencukupi.")

with tab2:
    if active_pos_count > 0:
        s_ticker = st.selectbox("Pilih Saham", [p['ticker'] for p in port['positions']])
        pos_data = next(item for item in port['positions'] if item["ticker"] == s_ticker)
        
        st.info(f"Posisi: {pos_data['lots']} Lot | Modal Terserap: Rp {pos_data['modal_terserap']:,.0f}")
        
        s_status = st.radio("Jenis Eksekusi:", ["Take Profit (TP)", "Stop Loss (SL)"])
        s_price = st.number_input("Harga Match (Jual)", min_value=0)
        
        if s_price > 0:
            nilai_kotor_jual = pos_data['lots'] * s_price * 100
            total_fee_jual = nilai_kotor_jual * FEE_JUAL
            
            biaya_materai_jual = 0
            if not tracker['materai_paid'] and (tracker['accumulated_value'] + nilai_kotor_jual) > BATAS_MATERAI:
                biaya_materai_jual = TARIF_MATERAI
            
            net_return = nilai_kotor_jual - total_fee_jual - biaya_materai_jual
            pl_rupiah = net_return - pos_data['modal_terserap']
            pl_persen = (pl_rupiah / pos_data['modal_terserap']) * 100
            pl_color = "var(--green)" if pl_rupiah > 0 else "var(--red)"
            
            st.markdown(f"""
            <div style="background:#161B22; padding:12px; border-radius:6px; font-family:monospace; margin-bottom:12px; border-left: 3px solid {pl_color};">
                Nilai Jual Kotor: Rp {nilai_kotor_jual:,.0f}<br>
                Fee Jual + Materai: Rp {(total_fee_jual + biaya_materai_jual):,.0f}<br>
                Net Cair ke Kas: Rp {net_return:,.0f}<br>
                Net P/L: <span style="color:{pl_color};">Rp {pl_rupiah:,.0f} ({pl_persen:.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Simpan Eksekusi Jual"):
                port['positions'] = [p for p in port['positions'] if p['ticker'] != s_ticker]
                port['current_cash'] += net_return
                port['history']['total_sold'] += 1
                
                if "TP" in s_status: port['history']['hit_tp'] += 1
                else: port['history']['hit_sl'] += 1
                
                port['daily_tracker']['accumulated_value'] += nilai_kotor_jual
                if biaya_materai_jual > 0:
                    port['daily_tracker']['materai_paid'] = True
                
                total_valuasi_saham_sisa = sum([p['modal_terserap'] for p in port['positions']])
                port['total_equity'] = port['current_cash'] + total_valuasi_saham_sisa
                
                st.session_state['port'] = port
                st.success(f"Tersimpan: {s_ticker} terjual.")
                st.rerun()
    else: st.write("Tidak ada posisi aktif.")
st.markdown("</div>", unsafe_allow_html=True)

json_string = json.dumps(port, indent=4)
st.download_button(
    label="Download Jurnal JSON",
    data=json_string,
    file_name=f"portfolio_{TODAY_STR}.json",
    mime="application/json"
)
st.divider()

# =====================================================================
# RADAR SCREENER
# =====================================================================
regime, current_dd = get_macro_regime()
regime_color = "var(--bear)" if regime == "BEAR" else "var(--bull)"

st.markdown(f"""
<div class="macro-panel">
    <div class="macro-title">Filter Makro IHSG</div>
    <div class="macro-status" style="color: {regime_color};">{regime} REGIME</div>
    <div style="color: var(--text-dim); font-size: 13px; font-family: 'IBM Plex Mono', monospace; margin-top: 8px;">
        Drawdown: {current_dd:.2f}% | Syarat Tampil: {'<b>≥ 80%</b>' if regime == 'BEAR' else '<b>Semua Valid</b>'}
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Pindai Pasar (Breakout)"):
    with st.spinner(f"Memproses {len(TICKERS)} saham..."):
        df_market = fetch_market_data(TICKERS)
    if df_market is None: st.error("Gagal menarik data."); st.stop()

    k_breakout = []
    for t in TICKERS:
        df_stock = df_market[f"{t}.JK"] if isinstance(df_market.columns, pd.MultiIndex) and f"{t}.JK" in df_market.columns.get_level_values(0) else df_market if not isinstance(df_market.columns, pd.MultiIndex) else None
        m = analyze_stock(df_stock, t, regime, alokasi_per_posisi)
        if m: k_breakout.append(m)

    if not k_breakout:
        st.warning("Tidak ada eksekusi valid hari ini.")
        st.stop()

    k_breakout.sort(key=lambda x: x["buy_confidence"], reverse=True)
    st.subheader(f"Sinyal Eksekusi Valid ({len(k_breakout)} Saham)")
    
    for c in k_breakout:
        st.markdown(f"""
        <div class="panel-container" style="border-left: 4px solid var(--accent-2);">
            <div class="header-row"><span class="stock-title">{c['ticker']}</span></div>
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; font-family: monospace;">
                    <span>Confidence Momentum</span><span style="color:var(--green); font-weight:700;">{c['buy_confidence']:.0f}%</span>
                </div>
                <div style="background: #21262D; border-radius: 4px; height: 6px;"><div style="width: {c['buy_confidence']}%; background: var(--green); height: 100%; border-radius: 4px;"></div></div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; font-family: monospace; font-size: 12px; color: var(--text-dim); background:#161B22; padding:12px; border-radius:6px;">
                <div>Harga Terakhir <div style="color:#FFF; font-weight:600; font-size:14px;">{c['close']:,.0f}</div></div>
                <div>Sizing <div style="color:var(--accent-2); font-weight:600; font-size:14px;">{c['lot_size']} Lot</div></div>
                <div>Estimasi Modal <div style="color:var(--green); font-weight:600; font-size:14px;">Rp {c['modal_terserap']:,.0f}</div></div>
                <div>Cut Loss (LL20) <div style="color:var(--red); font-weight:600; font-size:14px;">{c['ll20']:,.0f}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

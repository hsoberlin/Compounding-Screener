"""
COMPOUNDING SCREENER & OEMS PORTFOLIO MANAGER
================================================================
Sistem penyaringan saham berbasis momentum absolut & OEMS JSON:
1. OEMS Logic: Pembulatan Lot, Net Fee (Beli 0.15%, Jual 0.25%).
2. Materai Cerdas: Potong Rp 10.000 maks 1x/hari jika Trx > 10 Jt.
3. Jurnal Portofolio (Upload/Download JSON State)
4. Filter Makro: Deteksi IHSG (Bear Regime Hysteresis -20% / -10%)
5. Sistem Breakout Murni (Harga > HH20, Vol Ratio < 1.0)
"""

import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import timedelta, timezone, datetime

st.set_page_config(page_title="Compounding Terminal", layout="centered", initial_sidebar_state="collapsed")
WIB = timezone(timedelta(hours=7))
TODAY_STR = datetime.now(WIB).strftime("%Y-%m-%d")

# =====================================================================
# TEMA VISUAL UI (INSTITUTIONAL DARK MODE)
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --bg: #060709; --card: #0D1117; --card-line: #30363D;
  --accent-1: #FF7B72; --accent-2: #58A6FF; 
  --text: #C9D1D9; --text-dim: #8B949E;
  --green: #3FB950; --red: #F85149;
  --bear: #E5534B; --bull: #2EA043;
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

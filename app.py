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
STARTING_DATE = "2026-09-07"   # hari pertama Compounding Terminal berjalan (tetap, jangan diubah)

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
  padding: 12px 14px;
  margin-bottom: 10px;
}
.panel-title {
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 8px;
}

.app-title {
  font-family: 'Inter', sans-serif;
  font-size: 26px;
  font-weight: 700;
  background: linear-gradient(90deg, #4A7DFF 0%, #26A65B 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 2px;
}
.app-subtitle {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
  letter-spacing: 0.5px;
  margin-bottom: 14px;
}

.regime-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px;
  font-weight: 600;
}
.regime-sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 3px;
}

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: 6px; }
.stat-item { background: #0F1114; border: 1px solid var(--line); border-radius: 4px; padding: 6px 8px; }
.stat-label { font-size: 9.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.4px; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 13.5px; font-weight: 600; margin-top: 1px; }

.candidate-row {
  background: #0F1114; border: 1px solid var(--line); border-radius: 5px;
  padding: 10px 12px; margin-bottom: 6px;
}
.candidate-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.candidate-ticker { font-family: 'Inter', sans-serif; font-size: 15px; font-weight: 700; }
.conf-bar-bg { background: #22252B; border-radius: 3px; height: 4px; overflow: hidden; margin-top: 4px; }
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

IDX_TICKERS_ALL = """AADI, AALI, ABDA, ABMM, ACES, ACRO, ACST, ADCP, ADES, ADHI, ADMF, ADMG,
ADMR, ADRO, AGAR, AGII, AGRO, AGRS, AHAP, AISA, AKKU, AKPI, AKRA, AKSI,
ALDO, ALII, ALKA, ALMI, AMAG, AMAN, AMAR, AMFG, AMIN, AMMN, AMMS, AMOR,
AMRT, ANDI, ANJT, ANTM, APEX, APIC, APII, APLI, APLN, ARCI, AREA, ARGO,
ARII, ARKO, ARNA, ARTA, ARTO, ASDM, ASGR, ASHA, ASII, ASJT, ASLC, ASLI,
ASPI, ASPR, ASRI, ASRM, ASSA, ATAP, ATIC, ATLA, AUTO, AVIA, AWAN, AXIO,
AYAM, BABP, BABY, BACA, BACH, BAIK, BAJA, BALI, BANK, BATR, BAUT, BAYU,
BBCA, BBHI, BBKP, BBLD, BBMD, BBNI, BBRI, BBRM, BBSI, BBSS, BBTN, BBYB,
BCAP, BCIC, BDKR, BDMN, BEBS, BEEF, BEER, BEKS, BELI, BELL, BESS, BEST,
BFIN, BGTG, BHAT, BHIT, BIKE, BINA, BINO, BIPI, BIPP, BIRD, BISI, BJBR,
BJTM, BKDP, BKSL, BKSW, BLES, BLOG, BLTA, BLTZ, BLUE, BMAS, BMHS, BMRI,
BMSR, BMTR, BNBA, BNBR, BNGA, BNII, BNLI, BOAT, BOBA, BOGA, BOLA, BOLT,
BPFI, BPII, BPTR, BRAM, BREN, BRIS, BRMS, BRNA, BRPT, BSBK, BSDE, BSIM,
BSML, BSSR, BSWD, BTEK, BTON, BTPN, BTPS, BUAH, BUDI, BUKA, BUKK, BULL,
BUMI, BUVA, BVIC, BWPT, BYAN, CAKK, CAMP, CARE, CARS, CASA, CASH, CASS,
CBDK, CBPE, CBRE, CBUT, CCSI, CDIA, CEKA, CENT, CFIN, CGAS, CHEK, CHIP,
CINT, CITA, CITY, CLAY, CLEO, CLPI, CMNP, CMNT, CMPP, CMRY, CNKO, CNMA,
COAL, COCO, COIN, CPIN, CPRO, CRAB, CRSN, CSAP, CSIS, CSRA, CTBN, CTRA,
CUAN, CYBR, DAAZ, DADA, DART, DATA, DAYA, DCII, DEPO, DEWA, DEWI, DGIK,
DGNS, DGWG, DILD, DIVA, DKFT, DKHH, DLTA, DMAS, DMMX, DMND, DNAR, DNET,
DOID, DOOH, DOSS, DPUM, DRMA, DSFI, DSNG, DSSA, DUTI, DVLA, DWGL, DYAN,
EAST, ECII, EDGE, EKAD, ELIT, ELPI, ELSA, ELTY, EMAS, EMDE, EMMI, EMTK,
ENAK, ENRG, EPAC, EPMT, ERAA, ERAL, ERTX, ESSA, ESTA, ESTI, EURO, EXCL,
FAPA, FAST, FASW, FILM, FIRE, FISH, FITT, FMII, FOLK, FORE, FORU, FPNI,
FUJI, FUTR, FWCT, GDST, GDYR, GEMS, GGRM, GGRP, GHON, GIAA, GJTL, GLOB,
GLVA, GMFI, GMTD, GOLD, GOLF, GOOD, GOTO, GPRA, GPSO, GRIA, GRPM, GSMF,
GTBO, GTRA, GTSI, GULA, GUNA, GWSA, GZCO, HAIS, HAJJ, HALO, HATM, HBAT,
HDFA, HEAL, HELI, HERO, HEXA, HILL, HITS, HMSP, HOKI, HOMI, HOPE, HRME,
HRTA, HRUM, HUMI, HYGN, IATA, IBOS, IBST, ICBP, IDPR, IFII, IFSH, IGAR,
IKAI, IKBI, IKPM, IMAS, IMJS, IMPC, INAF, INCO, INDF, INDO, INDR, INDS,
INDY, INET, INKP, INOV, INPC, INPP, INPS, INRU, INTA, INTD, INTP, IOTF,
IPAC, IPCC, IPCM, IPOL, IPTV, IRRA, IRSX, ISAT, ISSP, ITIC, ITMA, ITMG,
JARR, JATI, JAWA, JECC, JECX, JELI, JGLE, JIHD, JKON, JMAS, JPFA, JRPT,
JSMR, JSPT, JTPE, KAEF, KAQI, KBAG, KBLI, KBLM, KBLV, KDSI, KDTN, KEEN,
KEJU, KETR, KIAS, KIJA, KING, KINO, KKGI, KLAS, KLBF, KMDS, KMTR, KOBX,
KOCI, KOKA, KONI, KOTA, KPIG, KRAS, KREN, KSIX, LABS, LAND, LAPD, LEAD,
LFLO, LIFE, LINK, LION, LIVE, LMPI, LPCK, LPGI, LPIN, LPKR, LPLI, LPPF,
LPPS, LSIP, LTLS, LUCY, MAHA, MAIN, MAPA, MAPB, MAPI, MARK, MASB, MAXI,
MAYA, MBAP, MBMA, MBSS, MCAS, MCOL, MCOR, MDIA, MDIY, MDKA, MDKI, MDLA,
MDLN, MDRN, MEDC, MEGA, MEJA, MERK, MFMI, MGLV, MGNA, MGRO, MHKI, MICE,
MIDI, MIKA, MINA, MINE, MKAP, MKPI, MKTR, MLBI, MLIA, MLPL, MLPT, MMIX,
MMLP, MNCN, MOLI, MORA, MPIX, MPMX, MPPA, MPRO, MPXL, MRAT, MREI, MSIN,
MSJA, MSKY, MSTI, MTDL, MTEL, MTFN, MTLA, MTMH, MTPS, MTWI, MUTU, MYOH,
MYOR, MYTX, NAIK, NASA, NATO, NAYZ, NCKL, NELY, NEST, NETV, NFCX, NICE,
NICK, NICL, NIKL, NINE, NIRO, NISP, NOBU, NPGF, NRCA, NSSS, NTBK, NZIA,
OASA, OBAT, OBMD, OKAS, OMED, OMRE, PACK, PADA, PADI, PALM, PAMG, PANI,
PANR, PANS, PART, PBID, PBRX, PBSA, PDES, PDPP, PEGE, PEHA, PEVE, PGAS,
PGEO, PGJO, PGUN, PIPA, PJAA, PJHB, PKPK, PLIN, PMJS, PMUI, PNBN, PNBS,
PNGO, PNIN, PNLF, PNSE, POLA, POLI, POLU, PORT, POWR, PPGL, PPRE, PPRO,
PRAY, PRDA, PRDL, PRIM, PSAB, PSAT, PSDN, PSGO, PSKT, PSSI, PTBA, PTMP,
PTMR, PTPP, PTPS, PTPW, PTRO, PTSN, PTSP, PUDP, PURA, PWON, PYFA, PZZA,
RAAM, RAJA, RALS, RANC, RANS, RATU, RBMS, RDTX, REAL, RELF, RELI, RGAS,
RIGS, RISE, RLCO, RMKE, RMKO, ROCK, RODA, RONY, ROTI, RSCH, SAFE, SAGE,
SAME, SAMF, SAPX, SATU, SCCO, SCMA, SCNP, SDMU, SDPC, SDRA, SFAN, SGER,
SGRO, SHID, SHIP, SIDO, SILO, SIMP, SINI, SIPD, SKBM, SKLT, SKRN, SLIS,
SMAR, SMBR, SMCB, SMDM, SMDR, SMGA, SMGR, SMIL, SMKL, SMLE, SMMA, SMMT,
SMRA, SMSM, SOCI, SOFA, SOHO, SOLA, SONA, SOSS, SOTS, SPMA, SPTO, SQMI,
SRAJ, SRSN, SRTG, SSIA, SSMS, SSTM, STAA, STAR, STRK, STTP, SULI, SUNI,
SUPA, SUPR, SURE, SURI, SWID, TALF, TAMA, TAMU, TAPG, TARA, TBIG, TBLA,
TCID, TCPI, TEBE, TFAS, TFCO, TGKA, TGUK, TIFA, TINS, TIRA, TIRT, TKIM,
TLDN, TLKM, TMAS, TMPO, TOBA, TOSK, TOTL, TOTO, TOWR, TPIA, TPMA, TRGU,
TRIM, TRIN, TRIS, TRJA, TRON, TRST, TRUE, TRUK, TRUS, TSPC, TUGU, TYRE,
UANG, UCID, UDNG, UFOE, ULTJ, UNIC, UNIQ, UNSP, UNTD, UNTR, UNVR, URBN,
UVCR, VAST, VERN, VICI, VICO, VINS, VISI, VIVA, VKTR, VOKS, VRNA, VTNY,
WAPO, WBSA, WEGE, WEHA, WGSH, WICO, WIFI, WIIM, WIKA, WINE, WINS, WIRG,
WMPP, WMUU, WOMF, WOOD, WSBP, WTON, XSPI, YELO, YOII, YPAS, YULE, YUPI,
ZATA, ZINC, ZONE, ZYRX"""
TICKERS = sorted(set(t.strip().upper() for t in IDX_TICKERS_ALL.replace("\n", ",").split(",") if t.strip()))


# =====================================================================
# STATE MANAGEMENT (JSON jurnal portofolio)
# =====================================================================
def init_portfolio(capital):
    return {
        "last_updated": TODAY_STR,
        "initial_capital": capital,
        "total_modal_masuk": capital,   # kumulatif: modal awal + semua suntikan berikutnya
        "current_cash": capital,
        "total_equity": capital,
        "positions": [],
        "history": {"total_bought": 0, "total_sold": 0, "hit_tp": 0, "hit_sl": 0},
        "log_transaksi": [],            # riwayat bertanggal, dipakai buat rekap siklus 10 hari
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
# KALENDER HARI BURSA -- dipakai buat rekap siklus 10 hari, ditarik
# dari kalender IHSG (^JKSE) sehingga otomatis melewati akhir pekan
# dan hari libur bursa.
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_kalender_bursa():
    try:
        df = yf.download("^JKSE", start=STARTING_DATE, progress=False)
        if df.empty:
            return []
        return [d.strftime("%Y-%m-%d") for d in df.index]
    except Exception:
        return []


def hari_ke(tanggal_str, kalender):
    """Urutan hari bursa (1-based) untuk sebuah tanggal transaksi.
    Kalau tanggalnya bukan hari bursa (jarang terjadi), dibulatkan ke
    hari bursa berikutnya yang terdekat."""
    for i, d in enumerate(kalender):
        if d >= tanggal_str:
            return i + 1
    return len(kalender) if kalender else 1


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


def status_posisi_aktif(ticker, ll20_terkunci, ob_streak_tersimpan=0, last_check_date=None):
    """Tarik data terbaru satu saham dan hitung status SOP: HOLD, JUAL(TP),
    atau JUAL(CL). ob_streak_tersimpan dilewatkan dari jurnal biar hitungan
    hari-beruntun-overbought konsisten antar sesi. last_check_date mencegah
    streak nambah berkali-kali kalau app di-render ulang di hari yang sama
    (Streamlit rerun tiap ada interaksi) -- streak cuma boleh berubah
    SEKALI per hari kalender."""
    sudah_dicek_hari_ini = last_check_date == TODAY_STR
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
            ob_streak_final = 0 if not sudah_dicek_hari_ini else ob_streak_tersimpan
            return {"harga_now": harga_now, "stoch_k": float(stoch_k), "ob_streak": ob_streak_final, "rekomendasi": "JUAL_CL"}

        if sudah_dicek_hari_ini:
            ob_streak = ob_streak_tersimpan  # sudah dievaluasi hari ini, jangan nambah lagi
        else:
            ob_streak = ob_streak_tersimpan + 1 if stoch_k > 80 else 0
        rekom = "JUAL_TP" if ob_streak >= STOCH_OB_STREAK_TP else "HOLD"
        return {"harga_now": harga_now, "stoch_k": float(stoch_k), "ob_streak": ob_streak, "rekomendasi": rekom}
    except Exception:
        return None


# =====================================================================
# UI -- HEADER & JURNAL
# =====================================================================
st.markdown("""
<div class="app-title">Compounding Terminal</div>
<div class="app-subtitle">BREAKOUT-ONLY // REGIME-AWARE // v2</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Jurnal Portofolio</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Muat portfolio.json", type="json", label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            data_baru = json.load(uploaded_file)
            if "daily_tracker" not in data_baru or data_baru.get("daily_tracker", {}).get("date") != TODAY_STR:
                data_baru["daily_tracker"] = {"date": TODAY_STR, "accumulated_value": 0, "materai_paid": False}
            data_baru.setdefault("total_modal_masuk", data_baru.get("initial_capital", data_baru.get("current_cash", 0)))
            data_baru.setdefault("log_transaksi", [])
            for p in data_baru.get("positions", []):
                p.setdefault("ob_streak", 0)
                p.setdefault("asal", "compounding")
                p.setdefault("last_ob_check", None)
            st.session_state["port"] = data_baru
            st.success("Jurnal termuat.")
            st.rerun()
        except Exception:
            st.error("Format JSON tidak valid.")
    st.markdown("</div>", unsafe_allow_html=True)

alokasi_per_posisi = port["initial_capital"] * 0.20
active_pos_count = len(port["positions"])
total_modal_masuk = port.get("total_modal_masuk", port["initial_capital"])
return_pct = (port["total_equity"] / total_modal_masuk - 1) * 100 if total_modal_masuk > 0 else 0.0

st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='panel-title'>Ekuitas & Posisi</div>", unsafe_allow_html=True)
st.markdown(f"""
<div class="stat-grid">
  <div class="stat-item"><div class="stat-label">Total Ekuitas</div><div class="stat-value">Rp {port['total_equity']:,.0f}</div></div>
  <div class="stat-item"><div class="stat-label">Kas Aktif</div><div class="stat-value" style="color:var(--accent)">Rp {port['current_cash']:,.0f}</div></div>
  <div class="stat-item"><div class="stat-label">Total Modal Masuk</div><div class="stat-value">Rp {total_modal_masuk:,.0f}</div></div>
  <div class="stat-item"><div class="stat-label">Return</div><div class="stat-value" style="color:{'var(--up)' if return_pct>=0 else 'var(--down)'}">{return_pct:+.2f}%</div></div>
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
# UI -- TAMBAH MODAL (suntikan baru -- masuk ke current_cash DAN
# total_modal_masuk, dipakai buat transfer dari akun lain yang tidak
# lewat penjualan posisi warisan di app ini, mis. transfer cash murni)
# =====================================================================
with st.expander("Tambah Modal"):
    st.caption("Suntikan modal baru dari luar app ini (mis. transfer cash dari akun lain). Ini menambah Total Modal Masuk sekaligus Kas Aktif -- beda dari 'Sesuaikan Kas' yang cuma menimpa angka tanpa mengubah Total Modal Masuk.")
    tambah_nominal = st.number_input("Nominal tambahan (Rp)", min_value=0, value=0, step=100_000)
    if st.button("Tambah Modal Sekarang"):
        if tambah_nominal > 0:
            port["current_cash"] += tambah_nominal
            port["total_modal_masuk"] = port.get("total_modal_masuk", port["initial_capital"]) + tambah_nominal
            total_valuasi_saham = sum(p["modal_terserap"] for p in port["positions"])
            port["total_equity"] = port["current_cash"] + total_valuasi_saham
            st.session_state["port"] = port
            st.success(f"Modal bertambah Rp{tambah_nominal:,.0f}. Total Modal Masuk sekarang Rp{port['total_modal_masuk']:,.0f}".replace(",", "."))
            st.rerun()


# =====================================================================
# UI -- DAFTARKAN POSISI WARISAN (saham yang sudah dipegang sebelum
# app ini dipakai -- proceeds jualnya nanti dihitung sebagai modal
# masuk baru, bukan P&L trading Compounding Screener)
# =====================================================================
with st.expander("Daftarkan Posisi Warisan"):
    st.caption("Buat saham yang sudah kamu pegang sebelum mulai pakai app ini (bukan dibeli lewat form Beli). Begitu nanti terjual, hasilnya masuk sebagai Modal Masuk baru, bukan P&L trading.")
    w_ticker = st.text_input("Ticker", key="w_ticker")
    w_lots = st.number_input("Jumlah lot", min_value=0, step=1, key="w_lots")
    w_harga = st.number_input("Harga beli rata-rata", min_value=0.0, step=1.0, key="w_harga")
    w_ll20 = st.number_input("Batas CL (LL20 manual, opsional -- isi 0 jika belum tahu)", min_value=0.0, step=1.0, key="w_ll20")
    if st.button("Daftarkan Posisi Ini"):
        if w_ticker and w_lots > 0 and w_harga > 0:
            modal_terserap_warisan = w_lots * w_harga * 100
            port["positions"].append({
                "ticker": w_ticker.upper(),
                "entry_price": w_harga,
                "lots": int(w_lots),
                "modal_terserap": modal_terserap_warisan,
                "ll20_terkunci": float(w_ll20) if w_ll20 > 0 else 0,
                "ob_streak": 0,
                "asal": "warisan",
            })
            st.session_state["port"] = port
            st.success(f"{w_ticker.upper()} terdaftar sebagai posisi warisan.")
            st.rerun()


# =====================================================================
# UI -- SESUAIKAN KAS MANUAL (buat sinkron dgn broker: transfer antar
# akun, hasil jual saham lama yang tidak lewat app ini, dsb -- bukan
# transaksi, murni menyamakan angka current_cash dengan kas riil)
# =====================================================================
with st.expander("Sesuaikan Kas Manual"):
    st.caption("Pakai ini kalau kas riil di broker beda dari yang tercatat di sini -- misalnya setelah transfer antar akun atau menjual saham yang tidak lewat app ini. Ini tidak membuat catatan transaksi, cuma menimpa angka kas.")
    kas_baru = st.number_input("Kas riil sekarang (Rp)", min_value=0, value=int(port["current_cash"]), step=100_000)
    if st.button("Update Kas"):
        selisih = kas_baru - port["current_cash"]
        port["current_cash"] = kas_baru
        total_valuasi_saham = sum(p["modal_terserap"] for p in port["positions"])
        port["total_equity"] = port["current_cash"] + total_valuasi_saham
        st.session_state["port"] = port
        st.success(f"Kas diperbarui. Selisih dari sebelumnya: Rp{selisih:,.0f}".replace(",", "."))
        st.rerun()


# =====================================================================
# UI -- MONITORING POSISI AKTIF (otomatis, bukan manual)
# =====================================================================
total_unrealized = 0.0
if active_pos_count > 0:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Status Posisi Aktif</div>", unsafe_allow_html=True)
    total_unrealized = 0.0
    for p in port["positions"]:
        info = status_posisi_aktif(p["ticker"], p["ll20_terkunci"], p.get("ob_streak", 0), p.get("last_ob_check"))
        if info is None:
            st.write(f"{p['ticker']}: data tidak tersedia saat ini.")
            continue
        p["ob_streak"] = info["ob_streak"]
        p["last_ob_check"] = TODAY_STR
        nilai_now = info["harga_now"] * p["lots"] * 100
        gain_pct = (nilai_now / p["modal_terserap"] - 1) * 100
        total_unrealized += (nilai_now - p["modal_terserap"])
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
    b_lot_manual = st.number_input(
        "Lot manual (isi kalau lot fill beda dari saran otomatis -- kosongkan/0 untuk pakai saran)",
        min_value=0, value=0, step=1,
    )

    if b_price > 0:
        biaya_materai_estimasi = 0
        if not tracker["materai_paid"] and (tracker["accumulated_value"] + alokasi_per_posisi) > BATAS_MATERAI:
            biaya_materai_estimasi = TARIF_MATERAI

        alokasi_bersih = alokasi_per_posisi - biaya_materai_estimasi
        biaya_per_lot_net = (b_price * 100) * (1 + FEE_BELI)
        lot_saran = math.floor(alokasi_bersih / biaya_per_lot_net) if biaya_per_lot_net > 0 else 0
        lot_kalkulasi = b_lot_manual if b_lot_manual > 0 else lot_saran
        pakai_manual = b_lot_manual > 0

        if lot_kalkulasi > 0:
            nilai_kotor = lot_kalkulasi * b_price * 100
            total_fee = nilai_kotor * FEE_BELI
            biaya_materai_final = TARIF_MATERAI if (not tracker["materai_paid"] and (tracker["accumulated_value"] + nilai_kotor) > BATAS_MATERAI) else 0
            modal_aktual = nilai_kotor + total_fee + biaya_materai_final
            kembalian_kas = (modal_aktual if pakai_manual else alokasi_per_posisi) - modal_aktual
            if pakai_manual:
                kembalian_kas = 0  # lot manual: dana terserap = modal_aktual persis, tidak ada "sisa alokasi"

            # LL20 dikunci di harga saat ini -- referensi tetap sepanjang posisi dipegang
            df_ll = yf.download(f"{b_ticker.upper()}.JK", period="30d", interval="1d", progress=False) if b_ticker else None
            ll20_terkunci = None
            if df_ll is not None and not df_ll.empty:
                low_s = df_ll["Low"]
                if isinstance(low_s, pd.DataFrame):
                    low_s = low_s.iloc[:, 0]
                ll20_terkunci = float(low_s.tail(20).min())

            ll20_display = f"{ll20_terkunci:,.0f}" if ll20_terkunci is not None else "n/a"

            label_lot = f"{lot_kalkulasi} lot (input manual)" if pakai_manual else f"{lot_kalkulasi} lot (saran otomatis)"
            baris_sisa = "" if pakai_manual else f"Sisa masuk kas: <span style=\"color:var(--up);\">Rp {kembalian_kas:,.0f}</span><br>"
            st.markdown(f"""
            <div class="candidate-row">
                Volume order: {label_lot}<br>
                Nilai saham: Rp {nilai_kotor:,.0f}<br>
                Fee + materai: Rp {(total_fee + biaya_materai_final):,.0f}<br>
                Total dana terserap: <span style="color:var(--down);">Rp {modal_aktual:,.0f}</span><br>
                {baris_sisa}Batas CL (LL20 saat ini): <span style="color:var(--down);">{ll20_display}</span>
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
                        "asal": "compounding",
                    })
                    port["log_transaksi"].append({
                        "tanggal": TODAY_STR, "aksi": "BELI", "ticker": b_ticker.upper(),
                        "lot": lot_kalkulasi, "harga": b_price, "pl_rupiah": None,
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
        asal_label = "Warisan (bukan hasil Compounding)" if pos_data.get("asal", "compounding") == "warisan" else "Compounding"
        st.caption(f"Posisi: {pos_data['lots']} lot | Modal terserap: Rp {pos_data['modal_terserap']:,.0f} | Asal: {asal_label}".replace(",", "."))

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
            is_warisan = pos_data.get("asal", "compounding") == "warisan"
            catatan_warisan = "<br><span style=\"color:var(--neutral);\">Posisi warisan -- proceeds ini akan masuk Total Modal Masuk, bukan dihitung P/L trading.</span>" if is_warisan else ""

            st.markdown(f"""
            <div class="candidate-row">
                Nilai jual kotor: Rp {nilai_kotor_jual:,.0f}<br>
                Fee + materai: Rp {(total_fee_jual + biaya_materai_jual):,.0f}<br>
                Net cair ke kas: Rp {net_return:,.0f}<br>
                Net P/L: <span style="color:{pl_color};">Rp {pl_rupiah:,.0f} ({pl_persen:+.2f}%)</span>{catatan_warisan}
            </div>
            """.replace(",", "."), unsafe_allow_html=True)

            if st.button("Simpan Eksekusi Jual"):
                port["positions"] = [p for p in port["positions"] if p["ticker"] != s_ticker]
                port["current_cash"] += net_return
                if is_warisan:
                    port["total_modal_masuk"] = port.get("total_modal_masuk", port["initial_capital"]) + net_return
                port["history"]["total_sold"] += 1
                jenis_log = "JUAL_TP" if "TP" in s_status else "JUAL_CL"
                if "TP" in s_status:
                    port["history"]["hit_tp"] += 1
                else:
                    port["history"]["hit_sl"] += 1
                port["log_transaksi"].append({
                    "tanggal": TODAY_STR, "aksi": jenis_log, "ticker": s_ticker,
                    "lot": pos_data["lots"], "harga": s_price, "pl_rupiah": round(pl_rupiah),
                })
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
# UI -- REKAP SIKLUS 10 HARI BURSA
# =====================================================================
kalender_bursa = get_kalender_bursa()
log = port.get("log_transaksi", [])

if kalender_bursa and log:
    hari_sekarang = hari_ke(TODAY_STR, kalender_bursa)
    siklus_sekarang = (hari_sekarang - 1) // 10 + 1
    hari_dalam_siklus = (hari_sekarang - 1) % 10 + 1

    rekap = {}
    for entry in log:
        hk = hari_ke(entry["tanggal"], kalender_bursa)
        sk = (hk - 1) // 10 + 1
        r = rekap.setdefault(sk, {"beli": 0, "jual": 0, "tp": 0, "cl": 0, "realized": 0.0})
        if entry["aksi"] == "BELI":
            r["beli"] += 1
        elif entry["aksi"] == "JUAL_TP":
            r["jual"] += 1; r["tp"] += 1; r["realized"] += entry.get("pl_rupiah") or 0
        elif entry["aksi"] == "JUAL_CL":
            r["jual"] += 1; r["cl"] += 1; r["realized"] += entry.get("pl_rupiah") or 0

    total_realized = sum(r["realized"] for r in rekap.values())
    total_beli = sum(r["beli"] for r in rekap.values())
    total_jual = sum(r["jual"] for r in rekap.values())
    total_tp = sum(r["tp"] for r in rekap.values())
    total_cl = sum(r["cl"] for r in rekap.values())
    return_total = ((total_realized + total_unrealized) / total_modal_masuk * 100) if total_modal_masuk > 0 else 0

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown(f"<div class='panel-title'>Rekap Siklus 10 Hari Bursa -- Sejak {STARTING_DATE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='regime-sub' style='margin-bottom:8px;'>Hari bursa ke-{hari_sekarang} (siklus {siklus_sekarang}, hari {hari_dalam_siklus}/10)</div>", unsafe_allow_html=True)

    for sk in sorted(rekap.keys()):
        r = rekap[sk]
        tanda = " (berjalan)" if sk == siklus_sekarang else ""
        warna_r = "var(--up)" if r["realized"] >= 0 else "var(--down)"
        st.markdown(f"""
        <div class="stat-grid" style="margin-bottom:6px;">
          <div class="stat-item"><div class="stat-label">Siklus {sk}{tanda}</div><div class="stat-value">Hari {(sk-1)*10+1}-{sk*10}</div></div>
          <div class="stat-item"><div class="stat-label">Dibeli</div><div class="stat-value">{r['beli']}</div></div>
          <div class="stat-item"><div class="stat-label">Dijual</div><div class="stat-value">{r['jual']}</div></div>
          <div class="stat-item"><div class="stat-label">TP</div><div class="stat-value" style="color:var(--up)">{r['tp']}</div></div>
          <div class="stat-item"><div class="stat-label">CL</div><div class="stat-value" style="color:var(--down)">{r['cl']}</div></div>
          <div class="stat-item"><div class="stat-label">Realized</div><div class="stat-value" style="color:{warna_r}">Rp {r['realized']:,.0f}</div></div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-grid" style="margin-top:6px; border-top:1px solid var(--line); padding-top:8px;">
      <div class="stat-item"><div class="stat-label">Total Dibeli</div><div class="stat-value">{total_beli}</div></div>
      <div class="stat-item"><div class="stat-label">Total Dijual</div><div class="stat-value">{total_jual}</div></div>
      <div class="stat-item"><div class="stat-label">Total TP</div><div class="stat-value" style="color:var(--up)">{total_tp}</div></div>
      <div class="stat-item"><div class="stat-label">Total CL</div><div class="stat-value" style="color:var(--down)">{total_cl}</div></div>
      <div class="stat-item"><div class="stat-label">Realized</div><div class="stat-value" style="color:{'var(--up)' if total_realized>=0 else 'var(--down)'}">Rp {total_realized:,.0f}</div></div>
      <div class="stat-item"><div class="stat-label">Unrealized</div><div class="stat-value" style="color:{'var(--up)' if total_unrealized>=0 else 'var(--down)'}">Rp {total_unrealized:,.0f}</div></div>
    </div>
    <div class="stat-grid" style="margin-top:6px;">
      <div class="stat-item" style="grid-column: span 2;"><div class="stat-label">Return Total Sejak Hari 1</div><div class="stat-value" style="font-size:16px; color:{'var(--up)' if return_total>=0 else 'var(--down)'}">{return_total:+.2f}%</div></div>
    </div>
    """.replace(",", "."), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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

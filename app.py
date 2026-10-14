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

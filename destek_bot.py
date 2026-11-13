"""BIST destek yaklaşım botu - 1. parça: yatay destek bölgeleri (günlük grafik).

bildirim.py'ye dokunmaz, ondan bağımsız çalışır.

Kullanım (şimdilik sadece seviye yazdırma):
    python destek_bot.py seviye ASELS BIMAS INVES TRALT

Kurallar (kullanıcıyla netleşenler):
- Günlük grafikte dip ve tepelerin fitil uçları aday seviyedir.
- Birbirine yakın adaylar (en alttakinin %2 üstüne kadar) tek bölgede birleşir.
- En az 2 temas gerekir ve bunların en az 2'si dip olmalı (BIMAS'ta kullanıcının seçtikleriyle gürültü bu kuralla ayrıldı).
- Sadece fiyatın altında kalan (ya da fiyatı içine alan) bölgeler destek sayılır.
"""
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

PIVOT_N = 4          # bir dip/tepe sayılması için solunda ve sağında bakılan mum sayısı
BOLGE_TOL = 0.02     # bölge genişliği: en alttaki adayın en fazla %2 üstüne kadar birleştirilir
EN_AZ_TEMAS = 2      # bir bölgenin destek sayılması için en az temas
EN_AZ_DIP = 2        # bunların en az kaçı dip (fitil dibi) olmalı; tepeler bölgeyi güçlendirir ama tek başına yetmez
GUN_SAYISI = 250     # son kaç günlük mum incelenir
MAX_UZAKLIK = 30.0   # yazdırırken fiyattan en fazla %30 uzaktaki seviyeler gösterilir


def veri_cek_gunluk(sembol):
    """~1 yıllık günlük mum. Alınamazsa None."""
    df = None
    for deneme in range(3):
        try:
            df = yf.Ticker(sembol + ".IS").history(period="1y", interval="1d", auto_adjust=False)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.5 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return None if df.empty else df


def guncel_fiyat(sembol, df):
    """Gün içi (15 dk) verinin son kapanışı; alınamazsa günlük verinin son kapanışı.
    (Günlük verinin son mumu Yahoo'da bazen güncellenmiyor.)"""
    try:
        g = yf.Ticker(sembol + ".IS").history(period="5d", interval="15m", auto_adjust=False)
        if g is not None and not g.empty:
            return float(g["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return float(df["Close"].iloc[-1])


def pivotlar(df, n=PIVOT_N):
    """Dip (fitil dibi) ve tepe (fitil ucu) noktalarını bulur.
    Dönüş: [(fiyat, mum_sirasi, "dip" ya da "tepe"), ...]
    Son n mum onaylanmadığı için pivot sayılmaz."""
    alt = df["Low"].to_numpy(dtype=float)
    ust = df["High"].to_numpy(dtype=float)
    sonuc = []
    for i in range(n, len(df) - n):
        pencere_alt = alt[i - n:i + n + 1]
        pencere_ust = ust[i - n:i + n + 1]
        if alt[i] == pencere_alt.min():
            sonuc.append((float(alt[i]), i, "dip"))
        if ust[i] == pencere_ust.max():
            sonuc.append((float(ust[i]), i, "tepe"))
    return sonuc


def bolgelere_ayir(noktalar, tol=BOLGE_TOL):
    """Fiyata göre sıralı noktaları, en alttakinin %tol üstüne kadar tek bölgede toplar."""
    noktalar = sorted(noktalar, key=lambda x: x[0])
    bolgeler, mevcut = [], []
    for nokta in noktalar:
        if mevcut and nokta[0] > mevcut[0][0] * (1 + tol):
            bolgeler.append(mevcut)
            mevcut = []
        mevcut.append(nokta)
    if mevcut:
        bolgeler.append(mevcut)
    return bolgeler


def yatay_destekler(df, fiyat=None):
    """Fiyatın altındaki (ya da fiyatı içine alan) yatay destek bölgeleri, fiyata yakından uzağa.
    Her bölge: seviye (ortanca), alt, ust, temas, dip, tepe, son_temas_gun, mesafe (%)."""
    if df is None or len(df) < 40:
        return []
    df = df.tail(GUN_SAYISI)
    if fiyat is None:
        fiyat = float(df["Close"].iloc[-1])
    son_sira = len(df) - 1
    sonuc = []
    for b in bolgelere_ayir(pivotlar(df)):
        if len(b) < EN_AZ_TEMAS or sum(1 for x in b if x[2] == "dip") < EN_AZ_DIP:
            continue
        fiyatlar = [x[0] for x in b]
        alt, ust = min(fiyatlar), max(fiyatlar)
        if alt > fiyat:            # bölge fiyatın üstünde: destek değil
            continue
        mesafe = 0.0 if ust >= fiyat else (fiyat / ust - 1) * 100
        sonuc.append(dict(
            seviye=float(np.median(fiyatlar)), alt=alt, ust=ust, temas=len(b),
            dip=sum(1 for x in b if x[2] == "dip"), tepe=sum(1 for x in b if x[2] == "tepe"),
            son_temas_gun=son_sira - max(x[1] for x in b), mesafe=mesafe))
    return sorted(sonuc, key=lambda r: r["mesafe"])


def seviye_yazdir(sembol):
    df = veri_cek_gunluk(sembol)
    if df is None:
        print(f"{sembol}: veri alınamadı")
        return
    fiyat = guncel_fiyat(sembol, df)
    print(f"\n=== {sembol}  güncel fiyat {fiyat:.2f}  (günlük verinin son kapanışı {df['Close'].iloc[-1]:.2f}, {len(df)} mum) ===")
    liste = [r for r in yatay_destekler(df, fiyat) if r["mesafe"] <= MAX_UZAKLIK]
    if not liste:
        print("  yatay destek bölgesi bulunamadı")
    for r in liste:
        print(f"  seviye {r['seviye']:.2f}  (bölge {r['alt']:.2f}-{r['ust']:.2f})  "
              f"temas {r['temas']} (dip {r['dip']}, tepe {r['tepe']})  "
              f"son temas {r['son_temas_gun']} gün önce  fiyata uzaklık %{r['mesafe']:.1f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "seviye":
        for s in sys.argv[2:]:
            seviye_yazdir(s.strip().upper())
    else:
        print("Kullanım: python destek_bot.py seviye ASELS BIMAS")

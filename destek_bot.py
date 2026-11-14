"""BIST destek yaklaşım botu - 1. parça: yatay destek bölgeleri (günlük grafik).

bildirim.py'ye dokunmaz, ondan bağımsız çalışır.

Kullanım (şimdilik sadece seviye yazdırma):
    python destek_bot.py seviye ASELS BIMAS INVES TRALT

Kurallar (kullanıcıyla netleşenler):
- Günlük grafikte dip ve tepelerin fitil uçları aday seviyedir.
- Birbirine yakın adaylar (en alttakinin %2 üstüne kadar) tek bölgede birleşir.
- En az 2 temas gerekir ve bunların en az 2'si dip olmalı (BIMAS'ta kullanıcının seçtikleriyle gürültü bu kuralla ayrıldı).
- Birbirine %4'ten yakın iki seviyeden zayıf olan elenir.
- Yükselen trend: iki dibin fitil uçlarından geçen, kapanışla kırılmamış, en az 3 temaslı çizgiler.
- Genel yön aşağıysa (en yüksek kapanıştan inen, kırılmamış düşen trend çizgisi varsa) yükselen trend çizgileri bildirilmez.
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
MIN_ARALIK = 0.04    # iki seviye arası bundan yakınsa zayıf olan elenir (kullanıcının seviyeleri hep %4,8'den uzak)
TREND_PIVOT_N = 2        # trend çizgisi bağlantı noktaları için dip sayılma penceresi
TREND_TOL = 0.01         # fitil dibi çizginin %1'i içindeyse temas sayılır
TREND_KIRILIM = 0.01     # kapanış çizginin %1 altına inerse çizgi kırılmış sayılır
TREND_EN_AZ_TEMAS = 3    # çizgi için en az temas
TREND_MIN_GUN = 8        # çizgiyi belirleyen iki dip arası en az bu kadar mum
DUSEN_MIN_GUN = 30       # düşen çizginin çıktığı zirve en az bu kadar mum eski olmalı (genel yön sayılması için)
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
    # birbirine çok yakın seviyelerden zayıf olanı ele (güçlü = temas, sonra dip sayısı fazla olan)
    kalan = []
    for r in sorted(sonuc, key=lambda r: (r["temas"], r["dip"]), reverse=True):
        if any(abs(r["seviye"] / k["seviye"] - 1) < MIN_ARALIK for k in kalan):
            continue
        kalan.append(r)
    return sorted(kalan, key=lambda r: r["mesafe"])


def yukselen_trendler(df, fiyat=None):
    """Hâlâ geçerli (kırılmamış) yükselen trend çizgileri, fiyata yakından uzağa.
    Çizgi iki dibin fitil uçlarından geçer; en az TREND_EN_AZ_TEMAS temas aranır;
    kapanış çizginin %TREND_KIRILIM altına inmişse çizgi elenir (fitil delmesi kırılma sayılmaz)."""
    if df is None or len(df) < 40:
        return []
    df = df.tail(GUN_SAYISI)
    alt = df["Low"].to_numpy(dtype=float)
    kap = df["Close"].to_numpy(dtype=float)
    son = len(df) - 1
    if fiyat is None:
        fiyat = float(kap[-1])
    n = TREND_PIVOT_N
    dipler = [i for i in range(n, len(df) - n) if alt[i] == alt[i - n:i + n + 1].min()]
    bulunanlar = []
    for a in range(len(dipler)):
        for b in range(a + 1, len(dipler)):
            i, j = dipler[a], dipler[b]
            if j - i < TREND_MIN_GUN or alt[j] <= alt[i]:
                continue
            egim = (alt[j] - alt[i]) / (j - i)
            cizgi = alt[i] + egim * (np.arange(i, son + 1) - i)
            if (kap[i:] < cizgi * (1 - TREND_KIRILIM)).any():      # kapanışla kırılmış
                continue
            if (alt[i:] < cizgi * 0.85).any():                     # fitil aşırı derin delmiş (kırılma kararı kapanışa göre)
                continue
            yakin = np.flatnonzero(np.abs(alt[i:] / cizgi - 1) <= TREND_TOL) + i
            temaslar = []
            for t in yakin:
                if not temaslar or t - temaslar[-1] >= 3:
                    temaslar.append(int(t))
            if len(temaslar) < TREND_EN_AZ_TEMAS:
                continue
            deger = float(cizgi[-1])
            if fiyat < deger * (1 - TREND_KIRILIM):                # güncel fiyat çizginin altında
                continue
            bulunanlar.append(dict(
                deger=deger, baslangic=df.index[i].strftime("%Y-%m-%d"), uzunluk=son - i,
                temas=len(temaslar), son_temas_gun=son - temaslar[-1],
                mesafe=max(0.0, (fiyat / deger - 1) * 100)))
    # bugünkü değeri birbirine %1,5'ten yakın olan çizgilerden en çok temaslıyı tut
    kalan = []
    for r in sorted(bulunanlar, key=lambda r: (r["temas"], r["uzunluk"]), reverse=True):
        if any(abs(r["deger"] / k["deger"] - 1) < 0.015 for k in kalan):
            continue
        kalan.append(r)
    return sorted(kalan, key=lambda r: r["mesafe"])


def dusen_trend(df, fiyat=None):
    """Genel yön aşağı mı? En yüksek kapanıştan başlayan, kapanışla kırılmamış, en az 3 temaslı
    düşen trend çizgisi varsa ve fiyat hâlâ çizginin altındaysa o çizgiyi döndürür, yoksa None."""
    if df is None or len(df) < 60:
        return None
    df = df.tail(GUN_SAYISI)
    kap = df["Close"].to_numpy(dtype=float)
    ust = df["High"].to_numpy(dtype=float)
    son = len(df) - 1
    if fiyat is None:
        fiyat = float(kap[-1])
    h = int(np.argmax(kap))                               # zirve: en yüksek kapanış
    if son - h < DUSEN_MIN_GUN:
        return None
    n = TREND_PIVOT_N
    en_iyi = None
    for j in range(max(h + TREND_MIN_GUN, n), son - n + 1):
        if kap[j] >= kap[h] or kap[j] != kap[j - n:j + n + 1].max():
            continue
        egim = (kap[j] - kap[h]) / (j - h)
        cizgi = kap[h] + egim * (np.arange(h, son + 1) - h)
        if (kap[h:] > cizgi * (1 + TREND_KIRILIM)).any():   # kapanışla kırılmış
            continue
        if (ust[h:] > cizgi * 1.04).any():                  # fitil çok yukarı delmiş
            continue
        yakin = np.flatnonzero(np.abs(kap[h:] / cizgi - 1) <= TREND_TOL) + h
        temaslar = []
        for t in yakin:
            if not temaslar or t - temaslar[-1] >= 3:
                temaslar.append(int(t))
        if len(temaslar) < TREND_EN_AZ_TEMAS:
            continue
        deger = float(cizgi[-1])
        if fiyat > deger * (1 + TREND_KIRILIM):              # fiyat şu an çizginin üstüne çıkmış
            continue
        aday = dict(deger=deger, baslangic=df.index[h].strftime("%Y-%m-%d"), temas=len(temaslar),
                    mesafe=max(0.0, (deger / fiyat - 1) * 100))
        if en_iyi is None or aday["temas"] > en_iyi["temas"]:
            en_iyi = aday
    return en_iyi


def ayrinti_yazdir(sembol, merkez):
    """Hata ayıklama: merkez fiyatın %6 çevresindeki tüm dip/tepe noktalarını tarihleriyle yazdırır."""
    df = veri_cek_gunluk(sembol)
    if df is None:
        print(f"{sembol}: veri alınamadı")
        return
    df = df.tail(GUN_SAYISI)
    print(f"\n=== {sembol}: {merkez} çevresindeki dip/tepe noktaları (±%6) ===")
    bulundu = False
    for fiyat, sira, tur in pivotlar(df):
        if abs(fiyat / merkez - 1) <= 0.06:
            print(f"  {df.index[sira].strftime('%Y-%m-%d')}  {tur:5s}  {fiyat:.2f}")
            bulundu = True
    if not bulundu:
        print("  bu çevrede hiç dip/tepe noktası yok")


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
    trendler = [r for r in yukselen_trendler(df, fiyat) if r["mesafe"] <= MAX_UZAKLIK]
    dt = dusen_trend(df, fiyat)
    if dt:
        print(f"  genel yön: AŞAĞI  (düşen trend çizgisi bugün {dt['deger']:.2f}, başlangıç {dt['baslangic']}, "
              f"temas {dt['temas']}; fiyat çizginin %{dt['mesafe']:.1f} altında)")
        print("  -- yükselen trend çizgileri (ELENDİ: genel yön aşağı, bildirilmez) --")
    else:
        print("  genel yön: aşağı değil (kırılmamış düşen trend çizgisi bulunamadı)")
        print("  -- yükselen trend çizgileri --")
    if not trendler:
        print("  geçerli yükselen trend çizgisi bulunamadı")
    for r in trendler:
        print(f"  çizgi bugün {r['deger']:.2f}  başlangıç {r['baslangic']} ({r['uzunluk']} gün)  "
              f"temas {r['temas']}  son temas {r['son_temas_gun']} gün önce  fiyata uzaklık %{r['mesafe']:.1f}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "seviye":
        for s in sys.argv[2:]:
            s = s.strip().upper()
            if ":" in s:                      # örn. GUBRF:263.5  -> o fiyatın çevresindeki noktalar
                kod, fiyat_metni = s.split(":", 1)
                ayrinti_yazdir(kod, float(fiyat_metni.replace(",", ".")))
            else:
                seviye_yazdir(s)
    else:
        print("Kullanım: python destek_bot.py seviye ASELS BIMAS")

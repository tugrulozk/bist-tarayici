"""BIST destek yaklaşım botu - 1. parça: yatay destek bölgeleri (günlük grafik).

bildirim.py'ye dokunmaz, ondan bağımsız çalışır.

Kullanım:
    python destek_bot.py seviye ASELS BIMAS INVES TRALT     (test: seviyeleri yazdırır)
    python destek_bot.py mesaj-test                        (Telegram bağlantısını dener, tek mesaj yollar)
    python destek_bot.py tara                              (canlı: tüm hisseleri tarar, takibe alınacakları Telegram'a yollar)
    Ortam değişkenleri: DESTEK_TELEGRAM_TOKEN, DESTEK_TELEGRAM_SOHBET (zorunlu), HISSELER (boşlukla; boşsa hisseler.txt),
    ZORLA=true (seans saati dışında da çalıştır).

Kurallar (kullanıcıyla netleşenler):
- Günlük grafikte dip ve tepelerin fitil uçları aday seviyedir.
- Birbirine yakın adaylar (en alttakinin %2 üstüne kadar) tek bölgede birleşir.
- En az 2 temas gerekir ve bunların en az 2'si dip olmalı (BIMAS'ta kullanıcının seçtikleriyle gürültü bu kuralla ayrıldı).
- Birbirine %4'ten yakın iki seviyeden zayıf olan elenir.
- Yükselen trend: iki dibin fitil uçlarından geçen, kapanışla kırılmamış, en az 3 temaslı çizgiler.
- Genel yön aşağıysa (en yüksek kapanıştan inen, kırılmamış düşen trend çizgisi varsa) yükselen trend çizgileri bildirilmez.
- EMA200 (günlük): fiyat yukarıdan gelip EMA200'e dokunur ya da %2 yaklaşırsa tek başına bildirim konusu; diğer bildirimlerde uzaklığı not olarak yazılır.
- Sadece fiyatın altında kalan (ya da fiyatı içine alan) bölgeler destek sayılır.
- Hacim düğümü (son 120 gün): en çok hacmin işlem gördüğü fiyat bölgeleri; fiyatın altındaysa destek adayı. Saatlik veriden hesaplanır (yoksa günlük veriden; günlükte gün içi dağılım tahmindir).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

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
EMA_UZUNLUK = 200        # üstel hareketli ortalama periyodu (günlük)
EMA_YAKLASMA = 2.0       # fiyat EMA'nın üstünde ve en fazla bu kadar % uzaktaysa "yaklaşıyor"
EMA_YUKARIDAN_GUN = 10   # "yukarıdan geliyor" kontrolü için bakılan mum sayısı
EMA_YUKARIDAN_FARK = 2.0 # son mumların en yüksek kapanışı fiyattan en az bu kadar % yukarıdaysa "yukarıdan geliyor"
SMC_IC_BOYUT = 5         # LuxAlgo SMC iç yapı pivot uzunluğu (sabit 5)
SMC_SWING_BOYUT = 50     # LuxAlgo SMC swing pivot uzunluğu (kullanıcının ayarı 50)
SMC_ATR_UZUNLUK = 200    # Order Block Filter = ATR(200)
SMC_OB_SAYISI = 5        # gösterilen iç order block sayısı (kullanıcının ayarı 5, boğa+ayı toplam)
NW_BANT = 8.0            # Nadaraya-Watson Bandwidth (kullanıcının ayarı 8)
NW_CARPAN = 3.0          # Mult (kullanıcının ayarı 3)
NW_MUM = 500             # gösterge en fazla son 500 muma bakar
NW_YAKIN = 1.5           # fiyat alt bandın en fazla bu kadar % üstündeyse "yaklaşıyor"
NW_SINYAL_MUM = 3        # ▲ işareti son bu kadar mum içindeyse "sinyal" (son mumda işaret oluşmaz, bir sonraki mumu bekler)
GUN_SAYISI = 250     # son kaç günlük mum incelenir
MAX_UZAKLIK = 30.0   # yazdırırken fiyattan en fazla %30 uzaktaki seviyeler gösterilir
HACIM_GUN = 120          # hacim profili için son kaç günlük mum (kısa vade için)
HACIM_DILIM = 0.5        # fiyat dilimi genişliği (%): fiyat aralığı bu oranda dilimlere bölünür
HACIM_EN_AZ_KAT = 2.0    # düğüm sayılması için dilimin hacmi, ortalama dilim hacminin en az bu kadar katı olmalı (sahte veride gürültü 1,7-2,2 kata çıktı; gerçek veride gözleyip ayarlanacak)
HACIM_BOLGE_ORAN = 0.7   # düğüm bölgesi: tepe dilimin hacminin en az %70'i olan komşu dilimler (her yana en fazla 3 dilim)
HACIM_MAX_DUGUM = 3      # en fazla kaç düğüm gösterilir (fiyata en yakın olanlar)
ATR_UZUNLUK = 14         # günlük ATR uzunluğu (yaklaşma mesafesi için oynaklık ölçüsü)
ATR_KAT = 1.0            # yaklaşma eşiği = ATR% x bu kat (hisse oynaksa eşik geniş, sakinse dar)
YAKIN_MIN = 1.0          # yaklaşma eşiği en az %1
YAKIN_MAX = 4.0          # yaklaşma eşiği en çok %4
KESISIM_TOL = 0.003      # iki bölge arası bu orandan yakınsa (%0,3) kesişiyor sayılır


def veri_cek_gunluk(sembol):
    """~1 yıllık günlük mum. Alınamazsa None."""
    df = None
    for deneme in range(3):
        try:
            df = yf.Ticker(sembol + ".IS").history(period="3y", interval="1d", auto_adjust=False)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.5 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return None if df.empty else df


def gun_ici_bilgi(sembol, df):
    """(güncel fiyat, bugünün en düşüğü, bugünün en yükseği). Gün içi (15 dk) veriden alınır;
    alınamazsa günlük verinin son mumundan. (Günlük verinin son mumu Yahoo'da bazen güncellenmiyor.)"""
    try:
        g = yf.Ticker(sembol + ".IS").history(period="5d", interval="15m", auto_adjust=False)
        if g is not None and not g.empty:
            g = g.dropna(subset=["Close"])
            bugun = g[g.index.date == g.index[-1].date()]
            return float(g["Close"].iloc[-1]), float(bugun["Low"].min()), float(bugun["High"].max())
    except Exception:
        pass
    return float(df["Close"].iloc[-1]), float(df["Low"].iloc[-1]), float(df["High"].iloc[-1])


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


def hacim_dugumleri(df, fiyat=None, hepsi=False, mumlar=None):
    """Hacim profili: son HACIM_GUN günde en çok hacmin işlem gördüğü fiyat bölgeleri (düğümler).
    Her günün hacmi, o günün en düşük-en yüksek aralığına eşit dağıtılır (gün içi dağılım bilinmediği için tahmin).
    Yalnız fiyatın altındaki (ya da fiyatı içine alan) düğümler döner, fiyata yakından uzağa.
    Her düğüm: merkez, alt, ust, kat (ortalama dilim hacminin kaç katı), pay (toplam hacmin %'si), mesafe (%).
    mumlar: profilin hesaplanacağı mum tablosu (Low/High/Volume); verilmezse df'nin son HACIM_GUN günü kullanılır.
    hepsi=True: karşılaştırma için; fiyatın üstündekiler ve zayıflar da dahil, en güçlü 5 düğüm (kata göre) döner."""
    if df is None or len(df) < 40:
        return []
    d = mumlar if mumlar is not None else df.tail(HACIM_GUN)
    alt_f = d["Low"].to_numpy(dtype=float)
    ust_f = d["High"].to_numpy(dtype=float)
    hac = d["Volume"].to_numpy(dtype=float)
    lo, hi = float(alt_f.min()), float(ust_f.max())
    if fiyat is None:
        fiyat = float(d["Close"].iloc[-1])
    if lo <= 0 or hi <= lo or not np.isfinite(hac).any() or hac.sum() <= 0:
        return []
    oran = 1 + HACIM_DILIM / 100
    n = int(np.ceil(np.log(hi / lo) / np.log(oran)))
    if n < 10:
        return []
    kenar = lo * oran ** np.arange(n + 1)
    kenar[-1] = max(kenar[-1], hi)
    profil = np.zeros(n)
    for dusuk, yuksek, v in zip(alt_f, ust_f, hac):
        if not (v > 0) or not np.isfinite(v):
            continue
        if yuksek <= dusuk:                      # tek fiyatlı mum: hacim tek dilime
            i = min(max(int(np.searchsorted(kenar, dusuk, side="right")) - 1, 0), n - 1)
            profil[i] += v
            continue
        a = np.maximum(kenar[:-1], dusuk)
        u = np.minimum(kenar[1:], yuksek)
        kes = np.clip(u - a, 0, None)
        if kes.sum() > 0:
            profil += v * kes / kes.sum()
    toplam = profil.sum()
    duz = np.convolve(np.pad(profil, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")   # hafif düzeltme
    ort = duz.mean()
    adaylar = []
    for i in range(n):
        sol = duz[i - 1] if i > 0 else -1.0
        sag = duz[i + 1] if i < n - 1 else -1.0
        if duz[i] > sol and duz[i] >= sag and duz[i] >= (1.0 if hepsi else HACIM_EN_AZ_KAT) * ort:
            adaylar.append(i)
    sonuc = []
    for i in sorted(adaylar, key=lambda i: duz[i], reverse=True):     # güçlüden zayıfa
        l = r = i
        while l > 0 and i - l < 3 and duz[l - 1] >= HACIM_BOLGE_ORAN * duz[i]:
            l -= 1
        while r < n - 1 and r - i < 3 and duz[r + 1] >= HACIM_BOLGE_ORAN * duz[i]:
            r += 1
        alt, ust = float(kenar[l]), float(kenar[r + 1])
        merkez = float(np.sqrt(kenar[i] * kenar[i + 1]))
        if alt > fiyat and not hepsi:                                  # fiyatın üstünde: destek değil
            continue
        if any(abs(merkez / k["merkez"] - 1) < MIN_ARALIK for k in sonuc):
            continue
        sonuc.append(dict(merkez=merkez, alt=alt, ust=ust, kat=float(duz[i] / ort),
                          pay=float(profil[l:r + 1].sum() / toplam * 100),
                          mesafe=0.0 if ust >= fiyat else (fiyat / ust - 1) * 100))
    if hepsi:
        return sorted(sonuc, key=lambda x: x["kat"], reverse=True)[:5]
    sonuc = [x for x in sonuc if x["mesafe"] <= MAX_UZAKLIK]
    return sorted(sonuc, key=lambda x: x["mesafe"])[:HACIM_MAX_DUGUM]


def hacim_mumlari(df, ham):
    """Hacim profili için mum tablosu: saatlik veri varsa son HACIM_GUN günün saatlik mumları (gün içi dağılım daha doğru),
    yoksa günlük mumlar. (mum tablosu, kaynak adı) döner."""
    saatlik = ham[0] if ham is not None else None
    if saatlik is not None and len(saatlik) > 0 and "Volume" in saatlik.columns:
        tarih = np.array(saatlik.index.date)
        gunler = np.unique(tarih)[-HACIM_GUN:]
        h = saatlik[np.isin(tarih, gunler)]
        if len(h) >= HACIM_GUN * 4 and float(h["Volume"].sum()) > 0:
            return h, "saatlik"
    return df.tail(HACIM_GUN), "günlük"


def atr_yuzde(df, n=ATR_UZUNLUK):
    """Günlük ATR (Wilder) / son kapanış, yüzde olarak. Veri kısaysa None."""
    if df is None or len(df) < n + 2:
        return None
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    atr = tr[:n].mean()
    for t in tr[n:]:
        atr = (atr * (n - 1) + t) / n
    return float(atr / c[-1] * 100)


def yakinlik_esigi(df):
    """(yaklaşma eşiği %, ATR %). Eşik = ATR% x ATR_KAT, YAKIN_MIN ile YAKIN_MAX arasına sıkıştırılır.
    ATR hesaplanamazsa eşik %2 (eski sabit değer)."""
    a = atr_yuzde(df)
    if a is None:
        return 2.0, None
    return float(min(max(ATR_KAT * a, YAKIN_MIN), YAKIN_MAX)), a


def kesisim_bolgeleri(parcalar, fiyat, esik):
    """parcalar: [(arac_adi, alt, ust)]. Fiyatın altındaki ya da fiyatı içine alan parçalardan, birbirine
    değen/çok yakın olanları tek bölgede toplar. Her bölge: alt, ust, araclar (benzersiz araç adları, sayısı = puan),
    mesafe (% , fiyat bölgenin içindeyse 0), yakin (mesafe <= eşik). Fiyata yakından uzağa sıralı."""
    p = sorted([x for x in parcalar if x[1] <= fiyat], key=lambda x: x[1])
    gruplar = []
    for ad, alt, ust in p:
        if gruplar and alt <= gruplar[-1]["ust"] * (1 + KESISIM_TOL):
            g = gruplar[-1]
            g["ust"] = max(g["ust"], ust)
            if ad not in g["araclar"]:
                g["araclar"].append(ad)
        else:
            gruplar.append(dict(alt=alt, ust=ust, araclar=[ad]))
    for g in gruplar:
        g["mesafe"] = 0.0 if g["ust"] >= fiyat else (fiyat / g["ust"] - 1) * 100
        g["yakin"] = g["mesafe"] <= esik
    return sorted(gruplar, key=lambda g: g["mesafe"])


def ozet_yazdir(df, fiyat, yatay, trendler, dt, eb, dugumler, bloklar, nb):
    """Tüm destek araçlarını tek tabloda toplar: kesişim puanı (kaç farklı araç aynı bölgede) ve ATR'ye göre yakınlık."""
    esik, atr = yakinlik_esigi(df)
    atr_txt = f"ATR{ATR_UZUNLUK} %{atr:.1f}" if atr is not None else "ATR hesaplanamadı"
    print(f"  -- ÖZET: kesişen destekler ({atr_txt} -> yaklaşma eşiği %{esik:.1f}) --")
    parca = []
    for r in yatay:
        parca.append(("Yatay destek", r["alt"], r["ust"]))
    if not dt:                                           # genel yön aşağıyken yükselen trend desteği sayılmaz
        for r in trendler:
            parca.append(("Yükselen trend", r["deger"] * 0.995, r["deger"] * 1.005))
    if eb:
        parca.append(("EMA200", eb["deger"] * 0.995, eb["deger"] * 1.005))
    for x in dugumler:
        parca.append(("Hacim düğümü", x["alt"], x["ust"]))
    for b in (bloklar or []):
        if b["tur"] == "boga":
            parca.append(("SMC mavi blok", b["alt"], b["ust"]))
    if nb:
        parca.append(("NW alt bant", nb["alt"] * 0.995, nb["alt"] * 1.005))
    gruplar = [g for g in kesisim_bolgeleri(parca, fiyat, esik) if g["mesafe"] <= MAX_UZAKLIK]
    if not gruplar:
        print("  fiyatın altında ya da içinde destek bölgesi yok")
    for g in gruplar[:6]:
        durum = "YAKIN" if g["yakin"] else "uzak"
        konum = "fiyat bölgenin İÇİNDE" if g["mesafe"] == 0 else f"fiyata uzaklık %{g['mesafe']:.1f}"
        print(f"  bölge {g['alt']:.2f}-{g['ust']:.2f}  {len(g['araclar'])} araç ({', '.join(g['araclar'])})  {konum}  [{durum}]")


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


def ema_hesapla(kapanis, uzunluk=EMA_UZUNLUK):
    """Üstel hareketli ortalama, TradingView'daki gibi: ilk değer ilk 'uzunluk' kapanışın ortalaması,
    sonra EMA = alfa*kapanış + (1-alfa)*önceki EMA, alfa = 2/(uzunluk+1)."""
    k = np.asarray(kapanis, dtype=float)
    sonuc = np.full(len(k), np.nan)
    if len(k) < uzunluk:
        return sonuc
    alfa = 2 / (uzunluk + 1)
    onceki = float(k[:uzunluk].mean())
    sonuc[uzunluk - 1] = onceki
    for i in range(uzunluk, len(k)):
        onceki = alfa * k[i] + (1 - alfa) * onceki
        sonuc[i] = onceki
    return sonuc


def ema200_bilgi(df, fiyat=None, bugun_alt=None):
    """EMA200'ün değeri, fiyatın ona % farkı (+ ise üstünde), eğimi ve durumu.
    durum: "dokundu"    -> bugünün en düşüğü EMA'ya değdi/indi, dün kapanış EMA'nın üstündeydi, fiyat EMA'dan en fazla %1 aşağıda
           "yaklaşıyor" -> fiyat EMA'nın üstünde, en fazla %EMA_YAKLASMA uzakta ve yukarıdan düşerek geliyor
           "yok"        -> ikisi de değil
    Son mumun kapanışı yerine güncel fiyat kullanılır. Veri yetersizse None."""
    if df is None or len(df) < EMA_UZUNLUK + 20:
        return None
    kap = df["Close"].to_numpy(dtype=float).copy()
    if fiyat is None:
        fiyat = float(kap[-1])
    if bugun_alt is None:
        bugun_alt = float(df["Low"].iloc[-1])
    kap[-1] = fiyat
    ema = ema_hesapla(kap)
    deger = float(ema[-1])
    mesafe = (fiyat / deger - 1) * 100
    onceki_zirve = float(kap[-1 - EMA_YUKARIDAN_GUN:-1].max())
    yukaridan = onceki_zirve >= fiyat * (1 + EMA_YUKARIDAN_FARK / 100)
    dun_ustunde = kap[-2] > ema[-2]
    if bugun_alt <= deger * 1.001 and dun_ustunde and fiyat >= deger * 0.99:
        durum = "dokundu"
    elif 0 <= mesafe <= EMA_YAKLASMA and yukaridan:
        durum = "yaklaşıyor"
    else:
        durum = "yok"
    return dict(deger=deger, mesafe=mesafe, durum=durum, yukaridan=yukaridan,
                egim="yükseliyor" if deger > float(ema[-6]) else "düşüyor")


def saatlikten_4saatlige(h):
    """Saatlik muma dizisini 4 saatlik mumlara çevirir. BIST için mum başlangıçları 10:00, 14:00 ve
    (varsa) 18:00 kapanış seansı mumu. Dönüş: Open/High/Low/Close/Volume sütunlu, mum başlangıç zamanı indeksli tablo."""
    h = h.dropna(subset=["Close"]).copy()
    if h.empty:
        return None
    idx = h.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Europe/Istanbul")
    saat = idx.hour
    blok = np.where(saat < 14, 10, np.where(saat < 18, 14, 18))
    h["bar"] = idx.tz_localize(None).normalize() + pd.to_timedelta(blok, unit="h")
    g = h.groupby("bar").agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                              Close=("Close", "last"), Volume=("Volume", "sum"))
    return g


def bes_dakikadan_4saatlige(m5):
    """5 dakikalık mumlardan TradingView'daki gibi 4 saatlik mumlar: 09:40-13:40, 13:40-17:40, 17:40-kapanış.
    (BIST'te TradingView 4 saatlik mumlar seans başına göre hizalanıyor görünüyor; ekran görüntülerindeki
    çok küçük son mum bunu düşündürüyor.)"""
    m5 = m5.dropna(subset=["Close"]).copy()
    if m5.empty:
        return None
    idx = m5.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Europe/Istanbul")
    dk = idx.hour * 60 + idx.minute
    baslangic = np.where(dk < 13 * 60 + 40, 9 * 60 + 40, np.where(dk < 17 * 60 + 40, 13 * 60 + 40, 17 * 60 + 40))
    m5["bar"] = idx.tz_localize(None).normalize() + pd.to_timedelta(baslangic, unit="m")
    return m5.groupby("bar").agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                                  Close=("Close", "last"), Volume=("Volume", "sum"))


def _yf_gecmis(sembol, donem, aralik):
    for deneme in range(3):
        try:
            x = yf.Ticker(sembol + ".IS").history(period=donem, interval=aralik, auto_adjust=False)
        except Exception:
            x = None
        if x is not None and not x.empty:
            return x
        time.sleep(1.5 * (deneme + 1))
    return None


def _yf_saatlik(sembol):
    """~700 günlük saatlik veri. 'period=730d' bazı hisselerde Yahoo'nun 730 gün sınırına takıldığı için
    başlangıç tarihi açıkça verilir."""
    for gun in (700, 365):
        baslangic = (pd.Timestamp.now() - pd.Timedelta(days=gun)).strftime("%Y-%m-%d")
        for deneme in range(2):
            try:
                x = yf.Ticker(sembol + ".IS").history(start=baslangic, interval="1h", auto_adjust=False)
            except Exception:
                x = None
            if x is not None and not x.empty:
                return x
            time.sleep(1.5)
    return None


def ham_veriler(sembol):
    """(saatlik veri, 5 dakikalık veri) - biri alınamazsa None olur."""
    return _yf_saatlik(sembol), _yf_gecmis(sembol, "60d", "5m")


def _birlestir(eski, yeni):
    """Eski (saatlikten) ve yeni (5 dakikalıktan) mum tablolarını birleştirir; attrs['bes_dk_mum'] = yeni mum sayısı."""
    if yeni is not None and eski is not None:
        sinir = yeni.index[0].normalize()
        sonuc = pd.concat([eski[eski.index < sinir], yeni])
        sonuc.attrs["bes_dk_mum"] = len(yeni)
        return sonuc
    sonuc = yeni if yeni is not None else eski
    if sonuc is not None:
        sonuc.attrs["bes_dk_mum"] = len(yeni) if yeni is not None else 0
    return sonuc


def _4saat_ham(sembol, ham=None):
    """4 saatlik mumlar. Son ~2 ay 5 dakikalık veriden (TradingView'a uygun 09:40/13:40/17:40 mumları),
    daha eskisi saatlik veriden (10:00/14:00 yaklaşık mumlar). Alınamazsa None."""
    saatlik, bes_dk = ham if ham is not None else ham_veriler(sembol)
    eski = saatlikten_4saatlige(saatlik) if saatlik is not None else None
    yeni = bes_dakikadan_4saatlige(bes_dk) if bes_dk is not None else None
    return _birlestir(eski, yeni)


def saatlikten_3mum(h):
    """Saatlik mumlardan TradingView benzeri günde 3 mum (09:40, 13:40, 17:40) tahmini. Saatlik veri 13:40 ve
    17:40'ı ayıramadığı için o kapanışlar ilgili saatlik mumun içinde 2/3 oranında ara değerle (açılış + 2/3*(kapanış-açılış))
    tahmin edilir. Günün son mumunun kapanışı daha sonra resmi günlük kapanışla düzeltilir.
    Amaç: Nadaraya-Watson gibi mum SAYISINA duyarlı göstergelerde eski dönemi de günde 3 mumla temsil etmek."""
    h = h.dropna(subset=["Close"]).copy()
    if h.empty:
        return None
    idx = h.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Europe/Istanbul")
    h["gun"] = idx.tz_localize(None).normalize()
    h["saat"] = idx.hour
    satirlar = []
    for gun, g in h.groupby("gun"):
        g = g.set_index("saat")
        g = g[~g.index.duplicated(keep="last")]
        def al(saatler):
            return g[g.index.isin(saatler)]
        a, b, c = al([10, 11, 12, 13]), al([14, 15, 16, 17]), al([17, 18])
        if a.empty:
            continue
        def ara_deger(saat, onceki):
            if saat in g.index:
                return float(g.loc[saat, "Open"] + (2 / 3) * (g.loc[saat, "Close"] - g.loc[saat, "Open"]))
            return onceki
        kapanis_a = ara_deger(13, float(a["Close"].iloc[-1]))
        satirlar.append((gun + pd.Timedelta(hours=9, minutes=40), float(a["Open"].iloc[0]), float(a["High"].max()),
                         float(a["Low"].min()), kapanis_a, float(a["Volume"].sum())))
        if b.empty:
            continue
        kapanis_b = ara_deger(17, float(b["Close"].iloc[-1]))
        satirlar.append((gun + pd.Timedelta(hours=13, minutes=40), kapanis_a, float(b["High"].max()),
                         float(b["Low"].min()), kapanis_b, float(b["Volume"].sum())))
        if c.empty:
            continue
        satirlar.append((gun + pd.Timedelta(hours=17, minutes=40), kapanis_b, float(c["High"].max()),
                         float(c["Low"].min()), float(c["Close"].iloc[-1]), float(c["Volume"].sum()) / 3))
    if not satirlar:
        return None
    d = pd.DataFrame(satirlar, columns=["bar", "Open", "High", "Low", "Close", "Volume"]).set_index("bar")
    d["High"] = d[["High", "Open", "Close"]].max(axis=1)
    d["Low"] = d[["Low", "Open", "Close"]].min(axis=1)
    return d.sort_index()


def nw_serisi(ham, gunluk=None):
    """Nadaraya-Watson için mum tablosu: eski dönem saatlikten günde 3 mum, son ~2 ay 5 dakikalıktan; kapanışlar resmi
    günlük kapanışla düzeltilir. Alınamazsa None."""
    saatlik, bes_dk = ham
    eski = saatlikten_3mum(saatlik) if saatlik is not None else None
    yeni = bes_dakikadan_4saatlige(bes_dk) if bes_dk is not None else None
    return gun_kapanisi_duzelt(_birlestir(eski, yeni), gunluk)


def gun_kapanisi_duzelt(df4, gunluk):
    """Her günün son 4 saatlik mumunun kapanışını resmi günlük kapanışla değiştirir
    (Yahoo'nun saatlik/5 dk verisinde kapanış seansı fiyatı eksik). Seans devam eden günün son mumuna dokunmaz."""
    if df4 is None or gunluk is None or len(gunluk) == 0:
        return df4
    d = df4.copy()
    d.attrs = dict(df4.attrs)
    gi = gunluk.index
    if getattr(gi, "tz", None) is not None:
        gi = gi.tz_localize(None)
    gk = pd.Series(gunluk["Close"].to_numpy(dtype=float), index=gi.normalize())
    simdi = pd.Timestamp.now(tz="Europe/Istanbul")
    bugun = simdi.tz_localize(None).normalize()
    kapanis_sonrasi = simdi.hour * 60 + simdi.minute >= 18 * 60 + 15
    gunler = d.index.normalize()
    son_mum = ~pd.Series(gunler).duplicated(keep="last").to_numpy()
    kap_sut, yuk_sut, dus_sut = d.columns.get_loc("Close"), d.columns.get_loc("High"), d.columns.get_loc("Low")
    for pos in np.flatnonzero(son_mum):
        gun = gunler[pos]
        if gun in gk.index and (gun < bugun or kapanis_sonrasi):
            c = float(gk[gun])
            d.iloc[pos, kap_sut] = c
            d.iloc[pos, yuk_sut] = max(float(d.iloc[pos, yuk_sut]), c)
            d.iloc[pos, dus_sut] = min(float(d.iloc[pos, dus_sut]), c)
    return d


def veri_cek_4saat(sembol, gunluk=None, ham=None):
    """4 saatlik mumlar (ayrıntı için _4saat_ham). gunluk verilirse her günün son mumunun kapanışı resmi kapanışa çekilir."""
    return gun_kapanisi_duzelt(_4saat_ham(sembol, ham), gunluk)


def nw_zarf(kapanis, h=NW_BANT, carpan=NW_CARPAN, adet=NW_MUM):
    """LuxAlgo Nadaraya-Watson Envelope'un 'Repainting Smoothing' AÇIK halinin Python karşılığı:
    son 'adet' kapanışın tamamı üzerinden Gauss çekirdekli düzeltme, bant = düzeltme +/- (ort. mutlak hata * carpan).
    k[0] = en son mum. ▲ = kapanış alt bandın altına inmiş (bir önceki kapanış bandın üstündeyken),
    ▼ = kapanış üst bandın üstüne çıkmış. Göstergedeki gibi bir mumun bandı olarak bir sonraki (daha yeni) mumun
    düzeltme değeri kullanılır; bu yüzden en son mumda işaret oluşmaz.
    Not: Pine kodundan hatırlanan mantığın yeniden yazımıdır; gerçek grafikle karşılaştırılarak doğrulanmalıdır."""
    k = np.asarray(kapanis, dtype=float)[::-1][:adet]
    n = len(k)
    if n < 30:
        return None
    idx = np.arange(n)
    w = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * h * h))
    y = (w @ k) / w.sum(axis=1)
    mae = float(np.abs(k - y).sum() / min(499, n - 1) * carpan)
    sinyaller = []
    for i in range(1, n - 1):
        ust_esik, alt_esik = y[i - 1] + mae, y[i - 1] - mae
        if k[i] > ust_esik and k[i + 1] < ust_esik:
            sinyaller.append((i, "▼"))
        if k[i] < alt_esik and k[i + 1] > alt_esik:
            sinyaller.append((i, "▲"))
    return dict(y=y, ust=y + mae, alt=y - mae, mae=mae, sinyaller=sinyaller, k=k)


def nw_bilgi(df4, fiyat=None):
    """Son mumdaki bant değerleri ve durum.
    durum: "sinyal"     -> son NW_SINYAL_MUM mumda ▲ işareti oluşmuş
           "değdi"      -> son 2 mumun düşüğü kendi alt bandına değmiş/inmiş ya da fiyat alt bandın altında
           "yaklaşıyor" -> fiyat alt bandın en fazla %NW_YAKIN üstünde
           "yok"        -> hiçbiri"""
    z = nw_zarf(df4["Close"].to_numpy(dtype=float))
    if z is None:
        return None
    if fiyat is None:
        fiyat = float(z["k"][0])
    alt0, ust0 = float(z["alt"][0]), float(z["ust"][0])
    mesafe = (fiyat / alt0 - 1) * 100
    dusukler = df4["Low"].to_numpy(dtype=float)[::-1]
    dokundu = any(dusukler[i] <= z["alt"][i] for i in (0, 1) if i < len(dusukler))
    son_yukari = min((i for i, t in z["sinyaller"] if t == "▲"), default=None)
    if son_yukari is not None and son_yukari <= NW_SINYAL_MUM:
        durum = "sinyal"
    elif dokundu or fiyat <= alt0:
        durum = "değdi"
    elif mesafe <= NW_YAKIN:
        durum = "yaklaşıyor"
    else:
        durum = "yok"
    return dict(alt=alt0, ust=ust0, mesafe=mesafe, durum=durum, son_yukari=son_yukari, zarf=z)


def smc_order_bloklar(df4, ic=SMC_IC_BOYUT, swing=SMC_SWING_BOYUT, atr_uz=SMC_ATR_UZUNLUK, adet=SMC_OB_SAYISI, kayit=None):
    """LuxAlgo 'Smart Money Concepts' göstergesinin iç (internal) order block mantığının Python karşılığı.
    Ayarlar kullanıcının ekran görüntüsündeki gibi: iç order block açık (5), swing order block kapalı,
    filtre ATR, mitigation High/Low. Dönüş: en yeni 'adet' aktif blok, en yeniden eskiye:
    [{tur: 'boga'|'ayi', ust, alt, mum: zaman}, ...]. Boğa order block = grafikteki mavi bölge.
    kayit: bir liste verilirse yapı kırılımları ve blok silinmeleri oraya metin olarak eklenir (hata ayıklama).
    Not: Pine kodundan hatırlanan mantığın yeniden yazımıdır; gerçek grafikle karşılaştırılarak doğrulanmalıdır."""
    h = df4["High"].to_numpy(dtype=float)
    l = df4["Low"].to_numpy(dtype=float)
    c = df4["Close"].to_numpy(dtype=float)
    n = len(df4)
    if n < max(swing, ic) + 10:
        return []
    # ATR (Wilder/RMA) ve yüksek oynaklık mumları: onların high/low'u ters çevrilir
    onceki_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - onceki_c), np.abs(l - onceki_c)))
    tr[0] = h[0] - l[0]
    atr = np.full(n, np.nan)
    if n >= atr_uz:
        atr[atr_uz - 1] = tr[:atr_uz].mean()
        for i in range(atr_uz, n):
            atr[i] = (atr[i - 1] * (atr_uz - 1) + tr[i]) / atr_uz
    yuksek_vol = np.where(np.isnan(atr), False, (h - l) >= 2 * atr)
    ph = np.where(yuksek_vol, l, h)
    pl = np.where(yuksek_vol, h, l)

    def bacak(boyut):
        """Her mum için mevcut bacak: 1 = yükselen (boğa), 0 = düşen (ayı). Pine'daki leg() fonksiyonu."""
        b = np.zeros(n, dtype=int)
        for i in range(n):
            if i >= boyut:
                yeni_tepe = h[i - boyut] > h[i - boyut + 1:i + 1].max()
                yeni_dip = l[i - boyut] < l[i - boyut + 1:i + 1].min()
                if yeni_tepe:
                    b[i] = 0
                elif yeni_dip:
                    b[i] = 1
                else:
                    b[i] = b[i - 1]
            else:
                b[i] = 0
        return b

    ic_bacak, sw_bacak = bacak(ic), bacak(swing)
    nan = float("nan")
    ic_tepe = dict(seviye=nan, sira=-1, kirildi=False)
    ic_dip = dict(seviye=nan, sira=-1, kirildi=False)
    sw_tepe, sw_dip = nan, nan
    bloklar = []
    egilim = 0                      # iç yapı yönü: 1 yukarı, -1 aşağı, 0 henüz yok
    for i in range(n):
        # 1) swing pivotları (yalnızca "iç seviye swing seviyesiyle aynı mı" kontrolü için)
        if i > 0:
            if sw_bacak[i] != sw_bacak[i - 1] and i >= swing:
                if sw_bacak[i] == 1:
                    sw_dip = l[i - swing]
                else:
                    sw_tepe = h[i - swing]
            # 2) iç pivotlar
            if ic_bacak[i] != ic_bacak[i - 1] and i >= ic:
                if ic_bacak[i] == 1:
                    ic_dip = dict(seviye=l[i - ic], sira=i - ic, kirildi=False)
                else:
                    ic_tepe = dict(seviye=h[i - ic], sira=i - ic, kirildi=False)
            # 3) yapı kırılımı -> order block
            ek_tepe = np.isnan(sw_tepe) or ic_tepe["seviye"] != sw_tepe
            if (not np.isnan(ic_tepe["seviye"]) and not ic_tepe["kirildi"] and ek_tepe
                    and c[i] > ic_tepe["seviye"] and c[i - 1] <= ic_tepe["seviye"]):
                ic_tepe["kirildi"] = True
                etiket = "CHoCH" if egilim == -1 else "BOS"
                egilim = 1
                dilim = pl[ic_tepe["sira"]:i]
                if len(dilim):
                    k = ic_tepe["sira"] + int(np.argmin(dilim))
                    bloklar.insert(0, dict(tur="boga", ust=float(ph[k]), alt=float(pl[k]), mum=df4.index[k]))
                    if kayit is not None:
                        kayit.append(f"{df4.index[i]:%Y-%m-%d %H:%M}  YUKARI {etiket}: kapanış {c[i]:.2f} > iç tepe {ic_tepe['seviye']:.2f} "
                                     f"(pivot {df4.index[ic_tepe['sira']]:%m-%d %H:%M})  -> BOĞA blok {pl[k]:.2f}-{ph[k]:.2f} "
                                     f"(mum {df4.index[k]:%m-%d %H:%M}{', yüksek oynaklık mumu' if yuksek_vol[k] else ''})")
            ek_dip = np.isnan(sw_dip) or ic_dip["seviye"] != sw_dip
            if (not np.isnan(ic_dip["seviye"]) and not ic_dip["kirildi"] and ek_dip
                    and c[i] < ic_dip["seviye"] and c[i - 1] >= ic_dip["seviye"]):
                ic_dip["kirildi"] = True
                etiket = "CHoCH" if egilim == 1 else "BOS"
                egilim = -1
                dilim = ph[ic_dip["sira"]:i]
                if len(dilim):
                    k = ic_dip["sira"] + int(np.argmax(dilim))
                    bloklar.insert(0, dict(tur="ayi", ust=float(ph[k]), alt=float(pl[k]), mum=df4.index[k]))
                    if kayit is not None:
                        kayit.append(f"{df4.index[i]:%Y-%m-%d %H:%M}  AŞAĞI {etiket}: kapanış {c[i]:.2f} < iç dip {ic_dip['seviye']:.2f} "
                                     f"(pivot {df4.index[ic_dip['sira']]:%m-%d %H:%M})  -> AYI blok {pl[k]:.2f}-{ph[k]:.2f} "
                                     f"(mum {df4.index[k]:%m-%d %H:%M}{', yüksek oynaklık mumu' if yuksek_vol[k] else ''})")
        # 4) mitigation (High/Low): boğa blok low altına inerse, ayı blok high üstüne çıkarsa silinir
        kalanlar = []
        for b in bloklar:
            silindi = (b["tur"] == "ayi" and h[i] > b["ust"]) or (b["tur"] == "boga" and l[i] < b["alt"])
            if silindi and kayit is not None:
                kayit.append(f"{df4.index[i]:%Y-%m-%d %H:%M}  {'BOĞA' if b['tur'] == 'boga' else 'AYI'} blok {b['alt']:.2f}-{b['ust']:.2f} SİLİNDİ "
                             f"(mum düşüğü {l[i]:.2f}, yüksek {h[i]:.2f})")
            if not silindi:
                kalanlar.append(b)
        bloklar = kalanlar
    return bloklar[:adet]


def smc_kayit_yazdir(sembol, son=30):
    """Hata ayıklama: iç yapı kırılımlarının ve order block oluşum/silinme olaylarının son kayıtları."""
    df4 = veri_cek_4saat(sembol)
    if df4 is None:
        print(f"{sembol}: 4 saatlik veri alınamadı")
        return
    kayit = []
    smc_order_bloklar(df4, kayit=kayit)
    print(f"\n=== {sembol}: SMC olay kaydı  ({len(df4)} 4 saatlik mum, {df4.index[0]:%Y-%m-%d} ... {df4.index[-1]:%Y-%m-%d %H:%M}) ===")
    for satir in kayit[-son:]:
        print("  " + satir)


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
    fiyat, bugun_alt, bugun_ust = gun_ici_bilgi(sembol, df)
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
    eb = ema200_bilgi(df, fiyat, bugun_alt)
    print(f"  -- EMA{EMA_UZUNLUK} (günlük) --")
    if eb:
        yon = "üstünde" if eb["mesafe"] >= 0 else "altında"
        print(f"  EMA{EMA_UZUNLUK} {eb['deger']:.2f}  fiyat EMA'nın %{abs(eb['mesafe']):.1f} {yon}  EMA {eb['egim']}  "
              f"yukarıdan geliyor: {'evet' if eb['yukaridan'] else 'hayır'}  DURUM: {eb['durum']}")
        if len(df) < 500:
            print("  (not: geçmiş veri kısa, EMA200 TradingView'dakinden küçük farklı çıkabilir)")
    else:
        print(f"  EMA{EMA_UZUNLUK} hesaplanamadı (en az ~220 günlük veri gerekir)")
    ham = ham_veriler(sembol)
    hmum, hkaynak = hacim_mumlari(df, ham)
    print(f"  -- hacim düğümleri ({hkaynak} veriden, son {HACIM_GUN} gün) --")
    dugumler = hacim_dugumleri(df, fiyat, mumlar=hmum)
    if not dugumler:
        print("  fiyatın altında belirgin hacim düğümü yok")
    for x in dugumler:
        print(f"  düğüm {x['merkez']:.2f}  (bölge {x['alt']:.2f}-{x['ust']:.2f})  ortalama dilimin {x['kat']:.1f} katı  "
              f"toplam hacmin %{x['pay']:.1f}'i  fiyata uzaklık %{x['mesafe']:.1f}")
    print("  (karşılaştırma için filtresiz en güçlü 5 düğüm, fiyatın üstündekiler dahil:)")
    for x in hacim_dugumleri(df, fiyat, hepsi=True, mumlar=hmum):
        if x["alt"] > fiyat:
            konum = f"fiyatın %{(x['alt'] / fiyat - 1) * 100:.1f} üstünde"
        elif x["ust"] >= fiyat:
            konum = "fiyatın içinde"
        else:
            konum = f"fiyatın %{(fiyat / x['ust'] - 1) * 100:.1f} altında"
        print(f"    {x['merkez']:.2f}  (bölge {x['alt']:.2f}-{x['ust']:.2f})  ortalama dilimin {x['kat']:.1f} katı  {konum}")
    df4 = veri_cek_4saat(sembol, df, ham)
    if df4 is None or len(df4) < 100:
        print("  -- 4 saatlik veri alınamadı --")
        ozet_yazdir(df, fiyat, liste, trendler, dt, eb, dugumler, [], None)
        return
    print(f"  -- 4 saatlik mumlar (TradingView ile karşılaştırmak için son 6; toplam {len(df4)} mum, "
          f"bunların {df4.attrs.get('bes_dk_mum', 0)} tanesi 5 dakikalık veriden) --")
    for zaman, m in df4.tail(6).iterrows():
        print(f"  {zaman.strftime('%Y-%m-%d %H:%M')}  A {m['Open']:.2f}  Y {m['High']:.2f}  D {m['Low']:.2f}  K {m['Close']:.2f}")
    print(f"  -- SMC iç order block (4 saatlik, aktif son {SMC_OB_SAYISI} blok; boğa = mavi bölge) --")
    bloklar = smc_order_bloklar(df4)
    if not bloklar:
        print("  aktif order block yok")
    for b in bloklar:
        if b["tur"] == "boga":
            if b["alt"] <= fiyat <= b["ust"]:
                durum = "fiyat BÖLGENİN İÇİNDE"
            elif fiyat > b["ust"]:
                durum = f"fiyat bölgenin %{(fiyat / b['ust'] - 1) * 100:.1f} üstünde"
            else:
                durum = "fiyat bölgenin altında"
        else:
            durum = f"fiyat bölgenin %{abs(b['alt'] / fiyat - 1) * 100:.1f} altında" if fiyat < b["alt"] else "fiyat bölgenin içinde/üstünde"
        ad = "BOĞA (mavi)" if b["tur"] == "boga" else "ayı (kırmızı)"
        print(f"  {ad}  bölge {b['alt']:.2f}-{b['ust']:.2f}  mum {b['mum'].strftime('%Y-%m-%d %H:%M')}  {durum}")
    dfn = nw_serisi(ham, df)
    if dfn is None or len(dfn) < 100:
        dfn = df4
    nb = nw_bilgi(dfn, fiyat)
    print(f"  -- Nadaraya-Watson (4 saatlik, Bandwidth {NW_BANT:g}, Mult {NW_CARPAN:g}, kapanış, repainting AÇIK) --")
    if nb is None:
        print("  hesaplanamadı (veri yetersiz)")
    else:
        print(f"  son mumda alt bant {nb['alt']:.2f}  üst bant {nb['ust']:.2f}  fiyat alt bandın %{nb['mesafe']:.1f} üstünde  DURUM: {nb['durum']}  "
              f"({len(nb['zarf']['k'])} mum kullanıldı)")
        z = nb["zarf"]
        tarihler = dfn.index[::-1]
        isaretler = sorted(z["sinyaller"], key=lambda x: -x[0])[-14:]
        print("  son işaretler (eskiden yeniye): " + ", ".join(f"{t} {tarihler[i]:%m-%d %H:%M}" for i, t in isaretler))
    ozet_yazdir(df, fiyat, liste, trendler, dt, eb, dugumler, bloklar, nb)


# ===================== CANLI MOD (tara) =====================
MAX_ADAY = 40            # bir turda en fazla kaç hisse ayrıntılı analize girer
MAX_MESAJ = 30           # bir turda en fazla kaç Telegram mesajı (yalnız taşkın koruması; gerçek tetiklemeyi sınırlamaz)
ADAY_PAYI = 1.5          # hızlı elemede eşiğe eklenen pay (%), aday kaçırmamak için
CEVRIM_SN = 300          # iki tarama turu arası hedef süre (saniye)
ROT_ADET = 25            # her turda kaç hissenin 4 saatlik seviyeleri (SMC, NW) yenilenir
TEKRAR_BAKMA_SN = 1800   # ayrıntılı analizde tetiklemeyen hisseye bu süre bir daha bakılmaz
DURUM_DOSYASI = os.path.join("durum_destek", "durum.json")
RSI_UZUNLUK = 14
MFI_UZUNLUK = 14
BB_UZUNLUK = 20
BB_CARPAN = 2.0
MFI_KULLAN = True        # AÇIK (kullanıcı kararı): ASELS'te TradingView ile ~8 puan fark çıktı (bot 56,9, TradingView 65,0); hacim verisi incelenecek, mesajlarda 'yaklaşık' diye işaretlenir.
RSI_ASIRI = 30.0
MFI_ASIRI = 20.0
UYUMSUZLUK_PENCERE = 60  # uyumsuzluk için bakılan son mum sayısı (4 saatlik)
UYUMSUZLUK_YAKIN = 12    # son dip bu kadar mum içinde olmalı


def _simdi():
    return pd.Timestamp.now(tz="Europe/Istanbul")


def _t(x, n=2):
    """Türkçe sayı biçimi: 395.75 -> 395,75"""
    return f"{x:.{n}f}".replace(".", ",")


def hisse_listesi():
    """hisseler.txt: boşluk, virgül ya da satırlarla ayrılmış kodlar; '#' ile başlayan satır yorum; '.IS' eki atılır."""
    yol = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hisseler.txt")
    kodlar = []
    with open(yol, encoding="utf-8") as f:
        for satir in f:
            satir = satir.split("#", 1)[0]
            for k in satir.replace(",", " ").split():
                k = k.strip().upper()
                if k.endswith(".IS"):
                    k = k[:-3]
                if k and k not in kodlar:
                    kodlar.append(k)
    return kodlar


def seans_icinde(simdi):
    """Hafta içi 09:55-18:20 (İstanbul saati). Resmî tatilleri bilmez; tatilde veri değişmediği için en kötü ihtimalle boşa çalışır."""
    dk = simdi.hour * 60 + simdi.minute
    return simdi.weekday() < 5 and 9 * 60 + 55 <= dk <= 18 * 60 + 20


def seans_oncesi(simdi):
    """Hafta içi 08:30-09:55: 4 saatlik seviyeleri önceden hazırlama vakti."""
    dk = simdi.hour * 60 + simdi.minute
    return simdi.weekday() < 5 and 8 * 60 + 30 <= dk < 9 * 60 + 55


def telegram_gonder(metin):
    """Mesajı yollar. Başarılıysa True. Token hiçbir yere yazdırılmaz."""
    token = os.environ.get("DESTEK_TELEGRAM_TOKEN", "").strip()
    sohbet = os.environ.get("DESTEK_TELEGRAM_SOHBET", "").strip()
    if not token or not sohbet:
        print("Telegram bilgileri (secret) tanımlı değil")
        return False
    veri = json.dumps({"chat_id": sohbet, "text": metin, "disable_web_page_preview": True}).encode("utf-8")
    istek = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=veri,
                                   headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(istek, timeout=20) as yanit:
            return yanit.status == 200
    except urllib.error.HTTPError as e:
        try:
            govde = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            govde = ""
        print(f"Telegram hatası: HTTP {e.code} {govde}")
    except Exception as e:
        print(f"Telegram hatası: {type(e).__name__}")
    return False


def durum_oku(bugun):
    try:
        with open(DURUM_DOSYASI, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("gun") == bugun:
            return d
    except Exception:
        pass
    return {"gun": bugun, "gonderilen": []}


DORT_DOSYASI = os.path.join("durum_destek", "dort.json")


def dort_oku(semboller):
    """Önceki çalışmadan kalan 4 saatlik seviyeler (ikinci iş anında hazır başlasın diye). Okunamazsa boş."""
    try:
        with open(DORT_DOSYASI, encoding="utf-8") as f:
            d = json.load(f)
        return {k: v for k, v in d.items() if k in semboller and isinstance(v, dict) and "bloklar" in v}
    except Exception:
        return {}


def dort_yaz(dort):
    os.makedirs(os.path.dirname(DORT_DOSYASI), exist_ok=True)
    with open(DORT_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(dort, f, ensure_ascii=False)


def durum_yaz(d):
    os.makedirs(os.path.dirname(DURUM_DOSYASI), exist_ok=True)
    with open(DURUM_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def gunluk_toplu(semboller, parca=80):
    """1. aşama için günlük veriyi toplu çeker: {sembol: DataFrame}. Alınamayanlar sonuca girmez."""
    sonuc = {}
    for i in range(0, len(semboller), parca):
        grup = semboller[i:i + parca]
        veri = None
        for deneme in range(2):
            try:
                veri = yf.download(" ".join(x + ".IS" for x in grup), period="3y", interval="1d",
                                   auto_adjust=False, group_by="ticker", progress=False, threads=True)
            except Exception:
                veri = None
            if veri is not None and not veri.empty:
                break
            time.sleep(2)
        if veri is None or veri.empty:
            continue
        for x in grup:
            try:
                if isinstance(veri.columns, pd.MultiIndex):
                    if (x + ".IS") not in veri.columns.get_level_values(0):
                        continue
                    d = veri[x + ".IS"]
                else:
                    d = veri
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= 60:
                    sonuc[x] = d
            except Exception:
                pass
    return sonuc


def anlik_fiyatlar(semboller, parca=60):
    """5 dakikalık veriden toplu: {sembol: (son fiyat, bugünün en düşüğü)}. Alınamayan hisse sonuca girmez."""
    sonuc = {}
    for i in range(0, len(semboller), parca):
        grup = semboller[i:i + parca]
        veri = None
        for deneme in range(2):
            try:
                veri = yf.download(" ".join(x + ".IS" for x in grup), period="1d", interval="5m",
                                   auto_adjust=False, group_by="ticker", progress=False, threads=True)
            except Exception:
                veri = None
            if veri is not None and not veri.empty:
                break
            time.sleep(1.5)
        if veri is None or veri.empty:
            continue
        for x in grup:
            try:
                if isinstance(veri.columns, pd.MultiIndex):
                    if (x + ".IS") not in veri.columns.get_level_values(0):
                        continue
                    d = veri[x + ".IS"]
                else:
                    d = veri
                d = d[["Low", "Close"]].dropna()
                if len(d):
                    sonuc[x] = (float(d["Close"].iloc[-1]), float(d["Low"].min()))
            except Exception:
                pass
    return sonuc


# ---------- teyit göstergeleri (4 saatlik) ----------
def rsi_serisi(c, n=RSI_UZUNLUK):
    """Wilder RSI (TradingView ta.rsi ile aynı mantık). Veri kısaysa None."""
    c = np.asarray(c, dtype=float)
    if len(c) < n + 2:
        return None
    d = np.diff(c)
    yuk = np.where(d > 0, d, 0.0)
    asa = np.where(d < 0, -d, 0.0)
    ru, rd = yuk[:n].mean(), asa[:n].mean()
    cikti = np.full(len(c), np.nan)
    cikti[n] = 100.0 if rd == 0 else 100 - 100 / (1 + ru / rd)
    for i in range(n, len(d)):
        ru = (ru * (n - 1) + yuk[i]) / n
        rd = (rd * (n - 1) + asa[i]) / n
        cikti[i + 1] = 100.0 if rd == 0 else 100 - 100 / (1 + ru / rd)
    return cikti


def mfi_serisi(h, l, c, v, n=MFI_UZUNLUK):
    """Money Flow Index (tipik fiyat = (H+L+C)/3). Veri kısaysa None."""
    h, l, c, v = (np.asarray(x, dtype=float) for x in (h, l, c, v))
    if len(c) < n + 2:
        return None
    tp = (h + l + c) / 3
    mf = tp * v
    poz = np.zeros(len(c))
    neg = np.zeros(len(c))
    for i in range(1, len(c)):
        if tp[i] > tp[i - 1]:
            poz[i] = mf[i]
        elif tp[i] < tp[i - 1]:
            neg[i] = mf[i]
    cikti = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        p, q = poz[i - n + 1:i + 1].sum(), neg[i - n + 1:i + 1].sum()
        if p == 0 and q == 0:
            cikti[i] = 50.0
        else:
            cikti[i] = 100.0 if q == 0 else 100 - 100 / (1 + p / q)
    return cikti


def bollinger_alt(c, n=BB_UZUNLUK, k=BB_CARPAN):
    """Son mumdaki Bollinger alt bandı (ortalama - k x popülasyon std, TradingView gibi). Veri kısaysa None."""
    c = np.asarray(c, dtype=float)
    if len(c) < n:
        return None
    son = c[-n:]
    return float(son.mean() - k * son.std())


def pozitif_uyumsuzluk(dusuk, gosterge, n=3):
    """Fiyat daha düşük dip yaparken gösterge daha yüksek dip yapmış mı? Son iki onaylı dip (solunda/sağında n mum)
    karşılaştırılır; ikinci dip son UYUMSUZLUK_YAKIN mum içinde olmalı, gösterge 50'nin altında kalmalı."""
    dusuk = np.asarray(dusuk, dtype=float)
    gosterge = np.asarray(gosterge, dtype=float)
    uz = len(dusuk)
    if gosterge is None or uz < 2 * n + 5:
        return False
    bas = max(n, uz - UYUMSUZLUK_PENCERE)
    dipler = [i for i in range(bas, uz - n)
              if dusuk[i] == dusuk[i - n:i + n + 1].min() and not np.isnan(gosterge[i])]
    if len(dipler) < 2:
        return False
    i1, i2 = dipler[-2], dipler[-1]
    if i2 < uz - 1 - UYUMSUZLUK_YAKIN:
        return False
    return bool(dusuk[i2] < dusuk[i1] * 0.999 and gosterge[i2] > gosterge[i1] + 1.0 and gosterge[i2] < 50)


def mum_kapandi(zaman, simdi):
    """BIST 4 saatlik mumları 09:40 / 13:40 / 17:40'ta başlar; kapanışları 13:40, 17:40, 18:10."""
    if zaman.date() < simdi.date():
        return True
    bas = zaman.hour * 60 + zaman.minute
    kapanis = 13 * 60 + 40 if bas < 13 * 60 + 40 else (17 * 60 + 40 if bas < 17 * 60 + 40 else 18 * 60 + 10)
    return simdi.hour * 60 + simdi.minute >= kapanis


def yutan_boga(df4, simdi):
    """Son mum ya da bir öncekisi yutan boğa mumu mu? Dönüş: açıklama metni ya da None.
    Yutan boğa: önceki mum kırmızı, bu mum yeşil, açılışı önceki kapanışın altında/eşit, kapanışı önceki açılışın üstünde/eşit."""
    o = df4["Open"].to_numpy(dtype=float)
    c = df4["Close"].to_numpy(dtype=float)
    if len(o) < 3:
        return None
    def f(i):
        return c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] <= c[i - 1] and c[i] >= o[i - 1]
    if f(len(o) - 1):
        kapali = mum_kapandi(df4.index[-1], simdi)
        return "son 4 saatlik mum" + ("" if kapali else " (henüz kapanmadı)")
    if f(len(o) - 2):
        return "bir önceki 4 saatlik mum"
    return None


def teyit_hesapla(df4, simdi, nb=None):
    """4 saatlik mumlarla teyitler. Dönüş: dict(satirlar=[...yalnız var olanlar...], bilgi='RSI x | MFI y'), veri yoksa None."""
    if df4 is None or len(df4) < 60:
        return None
    h, l, c, o = (df4[k].to_numpy(dtype=float) for k in ("High", "Low", "Close", "Open"))
    v = df4["Volume"].to_numpy(dtype=float)
    rsi = rsi_serisi(c)
    mfi = mfi_serisi(h, l, c, v) if MFI_KULLAN and np.nansum(v) > 0 else None
    satirlar = []
    yb = yutan_boga(df4, simdi)
    if yb:
        satirlar.append(f"Yutan boğa mumu ({yb})")
    if rsi is not None and pozitif_uyumsuzluk(l, rsi):
        satirlar.append("RSI pozitif uyumsuzluk (fiyat daha düşük dip, RSI daha yüksek dip)")
    if mfi is not None and pozitif_uyumsuzluk(l, mfi):
        satirlar.append("MFI pozitif uyumsuzluk (fiyat daha düşük dip, MFI daha yüksek dip; MFI yaklaşık)")
    rsi_son = float(rsi[-1]) if rsi is not None and not np.isnan(rsi[-1]) else None
    mfi_son = float(mfi[-1]) if mfi is not None and not np.isnan(mfi[-1]) else None
    if rsi_son is not None and rsi_son < RSI_ASIRI:
        satirlar.append(f"RSI aşırı satımda ({_t(rsi_son, 1)})")
    if mfi_son is not None and mfi_son < MFI_ASIRI:
        satirlar.append(f"MFI aşırı satımda ({_t(mfi_son, 1)}; MFI yaklaşık)")
    bb = bollinger_alt(c)
    if bb is not None:
        if c[-1] < bb:
            satirlar.append("Bollinger alt bandın dışında")
        elif l[-1] <= bb:
            satirlar.append("Bollinger alt banda değdi")
    if nb and nb.get("durum") == "sinyal":
        satirlar.append("Nadaraya-Watson ▲ sinyali (son 3 mumda)")
    parcalar = []
    if rsi_son is not None:
        parcalar.append(f"RSI {_t(rsi_son, 1)}")
    if mfi_son is not None:
        parcalar.append(f"MFI {_t(mfi_son, 1)} (yaklaşık)")
    return dict(satirlar=satirlar, bilgi=" | ".join(parcalar) if parcalar else "")


def dort_saat_seviye(sembol, df):
    """Bir hissenin 4 saatlik seviyeleri (SMC mavi bloklar, NW alt bant). Sırayla (rotasyonla) yenilenir. Veri yoksa None."""
    ham = ham_veriler(sembol)
    df4 = veri_cek_4saat(sembol, df, ham)
    if df4 is None or len(df4) < 100:
        return None
    fiyat = float(df["Close"].iloc[-1])
    bloklar = [dict(alt=float(b["alt"]), ust=float(b["ust"])) for b in smc_order_bloklar(df4) if b["tur"] == "boga"]
    dfn = nw_serisi(ham, df)
    if dfn is None or len(dfn) < 100:
        dfn = df4
    nb = nw_bilgi(dfn, fiyat)
    return dict(bloklar=bloklar, nb_alt=float(nb["alt"]) if nb else None, nb_durum=nb["durum"] if nb else None)


def aday_puani(df, fiyat=None, bugun_alt=None, d4=None):
    """Hızlı eleme (ucuz, bellekteki verilerle): (puan, aday_mi). Günlük araçlar (yatay destek, yükselen trend, EMA200)
    ve varsa 4 saatlik seviyeler (SMC mavi blok, NW alt bant) güncel fiyata yakın mı bakar."""
    if fiyat is None:
        fiyat = float(df["Close"].iloc[-1])
    if bugun_alt is None:
        bugun_alt = float(df["Low"].iloc[-1])
    esik, _ = yakinlik_esigi(df)
    esik += ADAY_PAYI
    parca = [("Yatay destek", r["alt"], r["ust"]) for r in yatay_destekler(df, fiyat)]
    if not dusen_trend(df, fiyat):
        parca += [("Yükselen trend", r["deger"] * 0.995, r["deger"] * 1.005) for r in yukselen_trendler(df, fiyat)]
    eb = ema200_bilgi(df, fiyat, bugun_alt)
    if eb:
        parca.append(("EMA200", eb["deger"] * 0.995, eb["deger"] * 1.005))
    nw_sinyal = False
    if d4:
        parca += [("SMC mavi blok", b["alt"], b["ust"]) for b in d4["bloklar"]]
        if d4.get("nb_alt"):
            parca.append(("NW alt bant", d4["nb_alt"] * 0.995, d4["nb_alt"] * 1.005))
        nw_sinyal = d4.get("nb_durum") == "sinyal"
    yakin = [g for g in kesisim_bolgeleri(parca, fiyat, esik) if g["yakin"]]
    ema_tetik = bool(eb) and eb["durum"] != "yok"
    puan = (max(len(g["araclar"]) for g in yakin) if yakin else 0) + (1 if ema_tetik else 0) + (1 if nw_sinyal else 0)
    return puan, bool(yakin) or ema_tetik or nw_sinyal


IKON = {"Yatay destek": "🧱", "Yükselen trend": "📐", "EMA200": "〰️", "NW alt bant": "🌊",
        "SMC mavi blok": "🟦", "Hacim düğümü": "📦"}
SIMGE_ACIKLAMA = "Simgeler: 〰️ EMA200 · 🌊 NW alt bant · 🟦 SMC mavi blok · 🧱 yatay destek · 📐 trend · 📦 hacim"


def bagimsiz_ikonlar(a):
    """Doğrudan bildirim tetikleyen üç kriterin (EMA200, NW, SMC) simgeleri: başlıkta hızlı tarama için."""
    k = []
    if a.get("ema_tetik"):
        k.append(IKON["EMA200"])
    if a.get("nw_tetik"):
        k.append(IKON["NW alt bant"])
    if a.get("smc_tetik"):
        k.append(IKON["SMC mavi blok"])
    return "".join(k)


def _mesafe(fiyat, alt, ust):
    return 0.0 if alt <= fiyat <= ust else (fiyat / ust - 1) * 100


def _konum(m):
    return "fiyat bölgenin içinde" if m == 0 else f"fiyat bölgenin %{_t(m, 1)} üstünde"


def en_yogun_kesisim(parcalar):
    """parcalar: [(arac_adi, alt, ust)]. Farklı araçların en çok üst üste bindiği fiyat aralığını bulur (zincirleme birleştirme yok).
    Dönüş: (araç sayısı, alt, ust, araç adları) ya da parça yoksa None. Eşitlikte fiyata yakın (yüksek) aralık seçilir."""
    if not parcalar:
        return None
    sinir = sorted({x for p in parcalar for x in (p[1], p[2])})
    aralik = []
    for lo, hi in zip(sinir[:-1], sinir[1:]):
        adlar = tuple(sorted({p[0] for p in parcalar if p[1] <= lo and p[2] >= hi}))
        if adlar:
            aralik.append([lo, hi, adlar])
    if not aralik:                                           # hepsi tek noktaysa
        p = parcalar[0]
        return (1, p[1], p[2], [p[0]])
    birlesik = []
    for lo, hi, adlar in aralik:                             # bitişik, aynı araç kümesi olan aralıkları birleştir
        if birlesik and birlesik[-1][2] == adlar and abs(birlesik[-1][1] - lo) < 1e-9:
            birlesik[-1][1] = hi
        else:
            birlesik.append([lo, hi, adlar])
    en = max(birlesik, key=lambda x: (len(x[2]), x[0]))
    return (len(en[2]), en[0], en[1], list(en[2]))


DESTEK_ALTI_SIRA = ["Yatay destek", "SMC mavi blok", "Hacim düğümü", "EMA200", "Yükselen trend", "NW alt bant"]


def destek_alti(yakin_parca):
    """Yaklaşılan yerler arasından 'bunun altına inerse destek bozulur' seviyesi: yatay destek varsa onun alt kenarı,
    yoksa sırayla SMC bloğu, hacim düğümü, EMA200, trend, NW. Aynı araçtan birden çok varsa fiyata en yakın olan."""
    for ad in DESTEK_ALTI_SIRA:
        adaylar = [p for p in yakin_parca if p[0] == ad]
        if adaylar:
            return min(adaylar, key=lambda p: p[3])[1]
    return None


def analiz(sembol, df, fiyat=None, bugun_alt=None):
    """2. aşama: tüm araçlarla ayrıntılı analiz. Dönüş: mesaj için gereken her şey (sözlük)."""
    if fiyat is None or bugun_alt is None:
        fiyat, bugun_alt, _ = gun_ici_bilgi(sembol, df)
    esik, atr = yakinlik_esigi(df)
    yatay = [r for r in yatay_destekler(df, fiyat) if r["mesafe"] <= MAX_UZAKLIK]
    dt = dusen_trend(df, fiyat)
    trendler = [] if dt else [r for r in yukselen_trendler(df, fiyat) if r["mesafe"] <= MAX_UZAKLIK]
    eb = ema200_bilgi(df, fiyat, bugun_alt)
    ham = ham_veriler(sembol)
    hmum, _ = hacim_mumlari(df, ham)
    dugumler = hacim_dugumleri(df, fiyat, mumlar=hmum)
    df4 = veri_cek_4saat(sembol, df, ham)
    bloklar = smc_order_bloklar(df4) if df4 is not None and len(df4) >= 100 else []
    dfn = nw_serisi(ham, df)
    if dfn is None or len(dfn) < 100:
        dfn = df4
    nb = nw_bilgi(dfn, fiyat) if dfn is not None and len(dfn) >= 100 else None

    parcalar = []      # (ad, alt, ust, mesafe, metin); yalnız fiyatın altında/içinde olanlar
    def ekle(ad, alt, ust, mesafe, metin):
        if alt <= fiyat:
            parcalar.append((ad, alt, ust, mesafe, metin))
    for r in yatay:
        m = _mesafe(fiyat, r["alt"], r["ust"])
        ekle("Yatay destek", r["alt"], r["ust"], m,
             f"Yatay destek {_t(r['alt'])}-{_t(r['ust'])} ({r['temas']} temas): {_konum(m)}")
    for r in trendler:
        ekle("Yükselen trend", r["deger"] * 0.995, r["deger"] * 1.005, r["mesafe"],
             f"Yükselen trend çizgisi {_t(r['deger'])} ({r['temas']} temas, {r['uzunluk']} gün): fiyat çizginin %{_t(r['mesafe'], 1)} üstünde")
    if eb:
        if eb["durum"] == "dokundu":
            d_metin = "bugün dokundu"
        elif eb["durum"] == "yaklaşıyor":
            d_metin = "yukarıdan yaklaşıyor"
        else:
            d_metin = None
        yon = "üstünde" if eb["mesafe"] >= 0 else "altında"
        m_txt = f"EMA200 (günlük) {_t(eb['deger'])}: " + (d_metin + ", " if d_metin else "") + f"fiyat %{_t(abs(eb['mesafe']), 1)} {yon}"
        ekle("EMA200", eb["deger"] * 0.995, eb["deger"] * 1.005, max(eb["mesafe"], 0.0), m_txt)
    for x in dugumler:
        m = _mesafe(fiyat, x["alt"], x["ust"])
        ekle("Hacim düğümü", x["alt"], x["ust"], m,
             f"Hacim düğümü {_t(x['alt'])}-{_t(x['ust'])} (ortalamanın {_t(x['kat'], 1)} katı): {_konum(m)}")
    for b in bloklar:
        if b["tur"] == "boga":
            m = _mesafe(fiyat, b["alt"], b["ust"])
            ekle("SMC mavi blok", b["alt"], b["ust"], m,
                 f"SMC mavi order block {_t(b['alt'])}-{_t(b['ust'])} (4 saatlik): {_konum(m)}")
    if nb:
        ek = {"sinyal": " - son mumlarda ▲ sinyali", "değdi": " - alt banda değdi", "yaklaşıyor": " - yaklaşıyor"}.get(nb["durum"], "")
        yon = "üstünde" if nb["mesafe"] >= 0 else "altında"
        ekle("NW alt bant", nb["alt"] * 0.995, nb["alt"] * 1.005, max(nb["mesafe"], 0.0),
             f"Nadaraya-Watson alt bant {_t(nb['alt'])} (4 saatlik): fiyat %{_t(abs(nb['mesafe']), 1)} {yon}{ek}")

    gruplar = kesisim_bolgeleri([(p[0], p[1], p[2]) for p in parcalar], fiyat, esik)
    yakin_gruplar = [g for g in gruplar if g["yakin"]]
    yakin_parca = sorted([p for p in parcalar if p[3] <= esik], key=lambda p: p[3])
    ema_tetik = bool(eb) and eb["durum"] in ("dokundu", "yaklaşıyor")
    nw_tetik = bool(nb) and nb["durum"] in ("sinyal", "değdi", "yaklaşıyor")
    teyit = teyit_hesapla(df4, _simdi(), nb)
    smc_tetik = any(p[0] == "SMC mavi blok" for p in yakin_parca)
    kes = en_yogun_kesisim([(p[0], p[1], p[2]) for p in yakin_parca])
    puan = (kes[0] if kes else 0) + (1 if ema_tetik else 0) + (1 if nw_tetik else 0) + (len(teyit["satirlar"]) if teyit else 0)
    return dict(sembol=sembol, fiyat=fiyat, esik=esik, atr=atr, dt=dt, eb=eb, nb=nb, parcalar=parcalar,
                yakin_parca=yakin_parca, yakin_gruplar=yakin_gruplar, gruplar=gruplar,
                tetik=bool(yakin_parca) or ema_tetik or nw_tetik, puan=puan,
                nw_tetik=nw_tetik, ema_tetik=ema_tetik, smc_tetik=smc_tetik, teyit=teyit,
                kesisim=kes, destek_alti=destek_alti(yakin_parca))


def mesaj_olustur(a):
    """Onaylanan biçim: hisse başına tek mesaj; yaklaşılan tüm yerler, destek altı, EMA200 notu, genel yön."""
    ik = bagimsiz_ikonlar(a)
    satirlar = [f"📍 {a['sembol']} – takibe al (puan {a['puan']})" + (f" {ik}" if ik else ""), f"Fiyat: {_t(a['fiyat'])}", "", "Yaklaştığı yerler:"]
    gosterilen = list(a["yakin_parca"])
    for p in a["parcalar"]:                     # eşik dışında ama tetikleyen göstergeler (EMA200 / NW durumu) de yazılsın
        if p not in gosterilen and ((p[0] == "EMA200" and a["ema_tetik"]) or (p[0] == "NW alt bant" and a["nw_tetik"])):
            gosterilen.append(p)
    for p in gosterilen:
        satirlar.append(f"{IKON.get(p[0], '•')} {p[4]}")
    if not gosterilen:
        satirlar.append("• (yok)")
    satirlar.append("")
    k = a.get("kesisim")
    if k and k[0] >= 2:
        satirlar.append(f"Kesişim: {k[0]} araç aynı yerde ({_t(k[1])}-{_t(k[2])}: {', '.join(k[3])})")
    if a.get("destek_alti") is not None:
        satirlar.append(f"Destek altı: {_t(a['destek_alti'])}")
    t = a.get("teyit")
    if t:
        satirlar.append("Teyitler (4 saatlik):" + (" yok" if not t["satirlar"] else ""))
        for x in t["satirlar"]:
            satirlar.append("✅ " + x)
        satirlar.append(t["bilgi"])
    if a["eb"] and not any(p[0] == "EMA200" for p in gosterilen):
        yon = "üstünde" if a["eb"]["mesafe"] >= 0 else "altında"
        satirlar.append(f"EMA200: {_t(a['eb']['deger'])} (fiyat %{_t(abs(a['eb']['mesafe']), 1)} {yon})")
    if a["dt"]:
        satirlar.append(f"Genel yön: AŞAĞI (düşen çizgi bugün {_t(a['dt']['deger'])}, fiyat %{_t(a['dt']['mesafe'], 1)} altında)")
    else:
        satirlar.append("Genel yön: aşağı değil")
    atr_txt = f", ATR %{_t(a['atr'], 1)}" if a["atr"] is not None else ""
    satirlar.append(f"(yaklaşma eşiği %{_t(a['esik'], 1)}{atr_txt})")
    return "\n".join(satirlar)


def tara_calistir():
    """Canlı döngü: sabah günlük veriyi alır, 4 saatlik seviyeleri sırayla hazırlar, sonra seans boyunca her ~5 dakikada
    güncel fiyatları çekip seviyelere yaklaşanları ayrıntılı inceler ve Telegram'a yollar. ZORLA=true: tek tur, seans kontrolü yok."""
    zorla = os.environ.get("ZORLA", "").strip().lower() in ("1", "true", "evet")
    simdi = _simdi()
    if not zorla and not (seans_icinde(simdi) or seans_oncesi(simdi)):
        print(f"Seans dışı ({simdi:%Y-%m-%d %H:%M}), çıkılıyor.")
        return
    if not os.environ.get("DESTEK_TELEGRAM_TOKEN", "").strip() or not os.environ.get("DESTEK_TELEGRAM_SOHBET", "").strip():
        print("HATA: DESTEK_TELEGRAM_TOKEN / DESTEK_TELEGRAM_SOHBET secret'ları tanımlı değil.")
        sys.exit(1)
    istenen = [x.strip().upper() for x in os.environ.get("HISSELER", "").replace(",", " ").split() if x.strip()]
    semboller = istenen or hisse_listesi()
    sure_limit = float(os.environ.get("DONGU_SAAT", "5.2")) * 3600
    t_basla = time.time()
    print(f"{simdi:%Y-%m-%d %H:%M} başladı: {len(semboller)} hisse, " + ("tek tur (ZORLA)" if zorla else "döngü modu"))
    gunluk = gunluk_toplu(semboller)
    print(f"günlük veri: {len(gunluk)}/{len(semboller)} hisse ({time.time() - t_basla:.0f} sn)")
    if not gunluk:
        print("HATA: hiçbir hissenin günlük verisi alınamadı.")
        sys.exit(1)
    kuyruk = [s for s in semboller if s in gunluk]
    dort = dort_oku(set(kuyruk))
    if dort:
        print(f"önceki çalışmadan {len(dort)} hissenin 4 saatlik seviyesi devralındı")
    rot = 0
    durum = durum_oku(f"{simdi:%Y-%m-%d}")
    bakildi = {}
    tur = 0
    while True:
        t_tur = time.time()
        simdi = _simdi()
        seans = zorla or seans_icinde(simdi)
        if not seans and not seans_oncesi(simdi):
            print("Seans bitti, döngü kapanıyor.")
            break
        if time.time() - t_basla > sure_limit:
            print("Süre sınırı doldu, döngü kapanıyor (sonraki çalışma devralır).")
            break
        tur += 1
        try:
            # 1) 4 saatlik seviyeleri sırayla yenile (zorla modunda hepsi tek turda, en fazla 60 hisse)
            if not seans and (rot >= len(kuyruk) or len(dort) >= 0.9 * len(kuyruk)):   # hazırlık bitti: seansı bekle
                print(f"[{simdi:%H:%M}] hazırlık tamam: 4 saatlik seviyeler {len(dort)}/{len(kuyruk)}, seans bekleniyor")
                time.sleep(120)
                continue
            adet = len(kuyruk) if zorla and len(kuyruk) <= 60 else ROT_ADET
            for _ in range(adet):
                s = kuyruk[rot % len(kuyruk)]
                rot += 1
                try:
                    r = dort_saat_seviye(s, gunluk[s])
                    if r:
                        dort[s] = r
                except Exception as e:
                    print(f"  {s}: 4 saatlik seviye hatası ({type(e).__name__})")
            if not seans:                                    # seans öncesi: yalnız hazırlık
                print(f"[{simdi:%H:%M}] hazırlık: 4 saatlik seviyeler {len(dort)}/{len(kuyruk)}")
                time.sleep(2)
                continue
            # 2) güncel fiyatlar ve hızlı eleme
            fiyatlar = anlik_fiyatlar(kuyruk)
            adaylar = []
            for s in kuyruk:
                if s in durum["gonderilen"] or s not in fiyatlar:
                    continue
                if time.time() - bakildi.get(s, 0) < TEKRAR_BAKMA_SN:
                    continue
                f, alt_bugun = fiyatlar[s]
                try:
                    puan, aday = aday_puani(gunluk[s], f, alt_bugun, dort.get(s))
                except Exception as e:
                    print(f"  {s}: hızlı eleme hatası ({type(e).__name__})")
                    continue
                if aday:
                    adaylar.append((puan, s))
            adaylar.sort(reverse=True)
            n_aday = len(adaylar)
            # 3) ayrıntılı analiz
            sonuclar = []
            for _, s in adaylar[:MAX_ADAY]:
                f, alt_bugun = fiyatlar[s]
                bakildi[s] = time.time()
                try:
                    a = analiz(s, gunluk[s], f, alt_bugun)
                except Exception as e:
                    print(f"  {s}: ayrıntılı analiz hatası ({type(e).__name__}: {str(e)[:80]})")
                    continue
                if a["tetik"]:
                    sonuclar.append(a)
            sonuclar.sort(key=lambda a: -a["puan"])
            # 4) gönder
            gonderilen = 0
            gonderilecek = sonuclar[:MAX_MESAJ]
            if len(gonderilecek) >= 3:                       # çok mesaj gelecekse önce tek satırlık sıralı özet
                ozet = [f"📋 Tur özeti ({simdi:%H:%M}): {len(gonderilecek)} hisse, en güçlüden başlayarak"]
                for i, a in enumerate(gonderilecek, 1):
                    ik = bagimsiz_ikonlar(a)
                    ozet.append(f"{i}. {a['sembol']} (puan {a['puan']})" + (f" {ik}" if ik else ""))
                ozet.append(SIMGE_ACIKLAMA)
                telegram_gonder("\n".join(ozet))
                time.sleep(0.6)
            for a in gonderilecek:
                if telegram_gonder(mesaj_olustur(a)):
                    durum["gonderilen"].append(a["sembol"])
                    gonderilen += 1
                    durum_yaz(durum)
                    time.sleep(0.6)
                else:
                    bakildi.pop(a["sembol"], None)           # gönderilemedi: sonraki turda yeniden denensin
            if tur % 5 == 0:
                dort_yaz(dort)
            print(f"[{simdi:%H:%M}] tur {tur}: fiyat {len(fiyatlar)}/{len(kuyruk)}, aday {n_aday}, tetik {len(sonuclar)}, "
                  f"mesaj {gonderilen}, 4s seviye {len(dort)}/{len(kuyruk)} ({time.time() - t_tur:.0f} sn)")
            if len(sonuclar) > MAX_MESAJ:
                print(f"  uyarı: {len(sonuclar)} tetikleme var, bu turda {MAX_MESAJ} tanesi gönderildi, kalanı sonraki turda.")
        except Exception as e:
            print(f"tur hatası ({type(e).__name__}: {str(e)[:100]})")
        if zorla:
            break
        gecen = time.time() - t_tur
        time.sleep(max(20, CEVRIM_SN - gecen))
    durum_yaz(durum)
    dort_yaz(dort)
    print(f"Bitti ({time.time() - t_basla:.0f} sn).")


def mesaj_testi():
    simdi = pd.Timestamp.now(tz="Europe/Istanbul")
    if telegram_gonder(f"✅ Destek botu bağlantı testi başarılı ({simdi:%Y-%m-%d %H:%M})."):
        print("Test mesajı gönderildi.")
    else:
        print("HATA: test mesajı gönderilemedi (yukarıdaki satıra bak).")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "seviye":
        for s in sys.argv[2:]:
            s = s.strip().upper()
            if ":" in s:                      # GUBRF:263.5 -> o fiyatın çevresindeki noktalar; TRALT:SMC -> SMC olay kaydı
                kod, ek = s.split(":", 1)
                if ek == "SMC":
                    smc_kayit_yazdir(kod)
                else:
                    ayrinti_yazdir(kod, float(ek.replace(",", ".")))
            else:
                seviye_yazdir(s)
    elif len(sys.argv) >= 2 and sys.argv[1] == "mesaj-test":
        mesaj_testi()
    elif len(sys.argv) >= 2 and sys.argv[1] == "tara":
        tara_calistir()
    else:
        print("Kullanım: python destek_bot.py seviye ASELS BIMAS | mesaj-test | tara")

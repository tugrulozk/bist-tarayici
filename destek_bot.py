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
- EMA200 (günlük): fiyat yukarıdan gelip EMA200'e dokunur ya da %2 yaklaşırsa tek başına bildirim konusu; diğer bildirimlerde uzaklığı not olarak yazılır.
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
EMA_UZUNLUK = 200        # üstel hareketli ortalama periyodu (günlük)
EMA_YAKLASMA = 2.0       # fiyat EMA'nın üstünde ve en fazla bu kadar % uzaktaysa "yaklaşıyor"
EMA_YUKARIDAN_GUN = 10   # "yukarıdan geliyor" kontrolü için bakılan mum sayısı
EMA_YUKARIDAN_FARK = 2.0 # son mumların en yüksek kapanışı fiyattan en az bu kadar % yukarıdaysa "yukarıdan geliyor"
SMC_IC_BOYUT = 5         # LuxAlgo SMC iç yapı pivot uzunluğu (sabit 5)
SMC_SWING_BOYUT = 50     # LuxAlgo SMC swing pivot uzunluğu (kullanıcının ayarı 50)
SMC_ATR_UZUNLUK = 200    # Order Block Filter = ATR(200)
SMC_OB_SAYISI = 5        # gösterilen iç order block sayısı (kullanıcının ayarı 5, boğa+ayı toplam)
GUN_SAYISI = 250     # son kaç günlük mum incelenir
MAX_UZAKLIK = 30.0   # yazdırırken fiyattan en fazla %30 uzaktaki seviyeler gösterilir


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


def veri_cek_4saat(sembol):
    """~2 yıllık saatlik veriden üretilmiş 4 saatlik mumlar. Alınamazsa None."""
    h = None
    for deneme in range(3):
        try:
            h = yf.Ticker(sembol + ".IS").history(period="730d", interval="1h", auto_adjust=False)
        except Exception:
            h = None
        if h is not None and not h.empty:
            break
        time.sleep(1.5 * (deneme + 1))
    if h is None or h.empty:
        return None
    return saatlikten_4saatlige(h)


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
    df4 = veri_cek_4saat(sembol)
    if df4 is None or len(df4) < 100:
        print("  -- 4 saatlik veri alınamadı --")
        return
    print("  -- 4 saatlik mumlar (TradingView ile karşılaştırmak için son 6) --")
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
    else:
        print("Kullanım: python destek_bot.py seviye ASELS BIMAS")

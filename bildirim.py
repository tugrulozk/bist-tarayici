"""BIST düşen trend kırılımı + aşırı satış tepkisi bildirimi (Telegram).

app.py'den bağımsız çalışır, app.py'ye dokunmaz. GitHub Actions tarafından
seans saatlerinde 15 dakikada bir çalıştırılır.

Sabit kurallar:
- Saatlik mum (15 dakikalık veriden tam saatlere birleştirilir), aktif mum dahil
- Düşen trend çizgisi kırılımı (🚀): en az 3 temas, eğim %3, temas yakınlığı %0,5
- Kırılım: kapanmış mum çizginin en az %0,2 üstünde kapanmış ya da aktif mum %0,5 üstünde, ve fiyat hâlâ çizgi üstünde
- Dönüş teyidi: 6 şart (yeşil mum, hacim, dip korunması, fiyat KAMA üstünde, MACD ve Wave Trend yukarı kesişimi son 5 mumda);
  🚀 = en az 3/6 ve hacim dahil, 🔸 = hacim yok ama diğer 5 teyidin hepsi var (5/6)
- Fiyatın hâlâ altında kaldığı başka geçerli çizgi varsa mesaja uyarı satırı eklenir
- Aynı mesajdaki hisseler en çok teyide göre sıralanır, ilki ⭐ ile işaretlenir
- Aşırı satış tepkisi (🔻): app.py'deki beş göstergeli tarama, ekran görüntüsündeki ayarlarla
  (en az 3 kriter, günlük en az %6 düşüş, en az 2 dönüş teyidi, tabana oturmuşlar hariç)
- Tüm sinyaller tek mesajda, ayrı başlıklar altında gelir; hiçbir grupta yeni hisse yoksa mesaj gelmez
- Aynı hisse aynı gün içinde, her grup için yalnızca bir kez bildirilir

Çalışma modları (MOD ortam değişkeni):
- bildirim : normal çalışma (seans saati kontrolü + tekrar bildirmeme)
- dongu    : tek çalışma içinde DONGU_BITIS'e kadar her DONGU_ARA dakikada bir "bildirim" taraması
- deneme   : seans saati ve tekrar kontrolü yok, o an eşleşen herkesi bildirir
- baglanti : sadece Telegram'a deneme mesajı yollar
- rapor    : gün sonu raporu; bugün bildirilen sinyallerin sonradan 15 dakikalık mumlarla değerlendirmesi
             (en yüksek / en düşük / kapanış, stopa mı hedefe mi önce değdi), akşam seans bittikten sonra çalışır
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

TR = ZoneInfo("Europe/Istanbul")
TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

PERIYOT, ARALIK, MUM_DAKIKA = "60d", "15m", 60
MIN_MUM = 270      # app.py ile aynı: bundan az mumu olan hisse atlanır (NW 50 + MAE 200 + 20)
# Alti teyit: 🟢 yesil mum, 📊 hacim, 🔒 dip korunmus, 📈 fiyat KAMA'nin ustunde,
#             🔵 MACD (mavi cizgi kirmiziyi yukari kesti), 🌊 Wave Trend (yesil cizgi kirmiziyi yukari kesti)
TEYIT_TOPLAM = 6
MIN_TEYIT = 2          # 🚀 icin 6 teyitten en az 2'si VE hacim sartlarin icinde
MIN_TEYIT_HACIMSIZ = 5  # 🔸 (hacimsiz kirilim) icin: hacim disindaki 5 teyidin hepsi sart
TEYIT_SON_N = 5        # MACD ve Wave Trend kesisimi icin bakilan son mum sayisi (bolge sarti yok)
KIRILIM_KAPANMIS = 0.2  # KAPANMIS mum cizginin en az % kac ustunde kapanmali
KIRILIM_AKTIF = 0.5     # AKTIF (son, kapanmamis) mum cizginin en az % kac ustunde olmali
KIRILIM_PENCERE = 3     # kirilim, son bu kadar saatlik mumda (aktif mum dahil) aranir
KAMA_N, KAMA_HIZLI, KAMA_YAVAS = 21, 2, 30   # TradingView'deki KAMA ayari
UST_CIZGI_YAKIN = 1.5   # fiyatin ustundeki gecerli cizgi en fazla bu % uzaktaysa mesaja uyari yazilir
FIB_GERI = 40          # Fibonacci tepesi: dipten onceki en fazla bu kadar mum icindeki en yuksek nokta
# app.py'deki varsayılan trend ayarları (kaydırıcılar varsayılan değerde)
P = dict(trend_temas=3, trend_egim=3.0, trend_tol=0.5, trend_yakin=1.5)
DURUM_DOSYASI = Path("durum") / "bildirilen.json"

# ---- Bildirimde hangi bölümler açık (kod duruyor, True yapınca geri gelir) ----
ASIRI_GOSTER = False    # 🔻 aşırı satış tepkisi: kapalı (elle Streamlit uygulamasından taranıyor)
ERKEN_GOSTER = False    # ⏳ kırılım adayı: çizgiye yakın, henüz kırılmamış, MFI ve Wave Trend son mumlarda al vermiş
ERKEN_SON_N = 5         # adaylıkta MFI / Wave Trend sinyali için bakılan son mum sayısı
ERKEN_MFI_SEVIYE = 20.0 # MFI bu seviyeyi aşağıdan yukarı kesmiş olmalı (aşırı satımdan çıkış)
ERKEN_YAKIN = 1.0       # fiyat çizginin en fazla bu % altında olmalı
LIKIDITE_MIN = 0.0      # gunluk islem tutari (TL) bunun altindaki hisselerin trend sinyalleri (🚀🔸⏳) gosterilmez; 0 = filtre kapali
LIKIDITE_GUN = 20       # islem tutari: son bu kadar tam gunun medyani (bugun haric)
LIKIDITE = {}           # sembol -> medyan gunluk islem tutari (TL); analiz() doldurur
DIRENC_GOSTER = False   # 4 saatlik düşen çizgi direnci: kapalı (hedef olarak yalnızca ilk Fibonacci hedefi)
RETEST_GOSTER = True    # 🔁 retest: bildirilen kırılımdan sonraki saatlik mumlarda fiyat çizgiye geri dönüp üstünde tutunursa kısa bildirim
RETEST_YAKIN = 0.3      # retest: mumun dibi çizginin en fazla bu % üstüne kadar inmiş (ya da altına iğne atmış) olmalı
RETEST_STOP = 0.5       # retest: kapanış çizginin bu % altına inerse kırılım bozuldu sayılır, izleme biter
RETEST_DOSYASI = Path("durum") / "retest.json"
GUNLUK_GOSTER = True    # 📅 günlük grafikte düşen çizgi bilgisi (🚀/🔸 mesajlarına bir satır): kırıldı / şu kadar yukarıda / yok
GUNLUK_ONBELLEK_SN = 1800   # aynı hissenin günlük sonucu bu kadar saniye saklanır (her 5 dakikada yeniden çekmemek için)
GUNLUK_ONBELLEK = {}    # sembol -> (zaman, sonuç)
SIKISMA_GOSTER = True   # 🗜 kırılım öncesi mum boyu sıkışması: mesaja bilgi satırı (eleme yapmaz)
SIKISMA_KISA = 5        # kırılım mumundan önceki bu kadar mumun ortalama boyu (yüksek-düşük)
SIKISMA_UZUN = 50       # ...bu kadar mumun ortalama boyuyla kıyaslanır (kısa dönem dahil)
SIKISMA_ESIK = 0.70     # oran bunun altındaysa "sıkışma var"
RETEST_IZLE = {}        # sembol -> kaydedilen çizgi bilgisi; main() doldurur, analiz() bakar
RETEST_SONUC = {}       # sembol -> None / "bozuldu" / dict(cizgi, fiyat, dip); analiz() doldurur

# ---- Aşırı satış tarama ayarları (app.py'deki ekran görüntüsüyle aynı değerler) ----
ASIRI_EN_AZ_KRITER = 3   # 5 göstergeden en az kaç puan (✅ = 1, 🟡 yakın = 1)
ASIRI_MIN_DUSUS = 6.0    # gün içinde en az % kaç düşmüş (önceki kapanışa ya da bugünkü açılışa göre)
ASIRI_EN_AZ_TEYIT = 2    # 3 dönüş teyidinden (🟢 🔒 📊) en az kaçı
TABAN_ESIGI = -9.5       # günlük değişim bunun altındaysa tabana oturmuş sayılır, hariç tutulur
STOP_ATR = 1.5           # örnek stop: fiyat - 1,5 x ATR
# ---- Giriş sınırı ve 4 saatlik direnç (mesajdaki ek bilgiler)
GIRIS_MIN_RG = 1.5   # "en fazla bu fiyata kadar gir" sınırı için istenen en düşük risk/getiri (kazanç ≥ risk x bu)
DIRENC_PERIYOT = "6mo"  # 4 saatlik düşen çizgi direnci için saatlik mum geçmişi (yalnızca sinyal veren hisselere çekilir)
# app.py'deki "Gelişmiş ayarlar" varsayılanları
PA = dict(yakin_say=True, son_n=2, macd_seviye=0.0, macd_yakinlik=0.3, mfi_os=20.0,
          stoch_os=20.0, wt_yakin=4.0, nw_h=8.0, nw_carpan=3.0)

SEMBOL_KALIBI = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{3,6}$")
YOK_SAY = {"USD", "TRY", "EUR", "TTM", "EPS", "FKO", "POINT", "SEMBOL", "FIYAT", "HACIM", "PYS",
           "FINANS", "ENERJI", "SAT"}


# ---------------------------------------------------- app.py'den birebir alınanlar
def liste_coz(metin):
    """Kutuya yapistirilan metinden hisse kodlarini ayiklar.
    Hem 'AKBNK GARAN THYAO' gibi kod listesini hem de kod + sirket adi iceren
    tablo kopyalarini (her satirin ilk kelimesi) anlar."""
    sonuc = set()
    for satir in metin.replace(",", " ").replace(";", " ").splitlines():
        parcalar = [p.upper().replace(".IS", "").split(":")[-1] for p in satir.split()]
        if not parcalar:
            continue
        hepsi_kod = all(SEMBOL_KALIBI.match(p) and p not in YOK_SAY for p in parcalar)
        for p in (parcalar if hepsi_kod else parcalar[:1]):
            if SEMBOL_KALIBI.match(p) and p not in YOK_SAY:
                sonuc.add(p)
    return sorted(sonuc)


def gunluk_degisim(df):
    gunler = np.array(df.index.date)
    onceki = df[gunler != gunler[-1]]
    if onceki.empty:
        return np.nan
    return (df["Close"].iloc[-1] / onceki["Close"].iloc[-1] - 1) * 100


def trend_kirilimi(df, p, ust_liste=None):
    """Dusen trend cizgisi kirilimi.
    Son ~200 mumdaki tepelerden (pivot) dusen cizgiler dener. Cizgi gecerli sayilir:
    - en az `trend_temas` AYRI tepe cizgiye degmis (mum ucu cizgiye %tol icinde),
    - cizgi gercekten egimli (baslangic-bitis arasi en az `trend_egim` %),
    - kirilimdan once hicbir kapanis cizginin (tolerans) ustune cikmamis, mum uclari da cok tasmamis.
    Durum: 2 = son mumlarda kapanisla cizginin TOLERANS USTUNE cikti (kirilim),
           1 = cizginin hemen altinda ya da uzerinde ama toleransin icinde (yakin), None = yok."""
    n = len(df)
    pencere = min(n, 200)
    d = df.iloc[-pencere:]
    hi = d["High"].to_numpy(dtype=float)
    cl = d["Close"].to_numpy(dtype=float)
    vol = d["Volume"].to_numpy(dtype=float)
    k = 3
    tol = p["trend_tol"] / 100.0
    m = pencere - 1
    sn = KIRILIM_PENCERE  # kirilim son `sn` mum icinde aranir (aktif mum dahil)
    son_bar = m - sn  # kirilim bolgesinden onceki son mum
    roll = d["High"].rolling(2 * k + 1, center=True).max().to_numpy()
    piv = [i for i in range(k, pencere - k) if hi[i] == roll[i] and i <= son_bar]
    en_iyi = None
    ustte = []   # fiyatin hala altinda kaldigi gecerli cizgilerin son degerleri (uyari icin)
    for a in range(len(piv)):
        for b in range(a + 1, len(piv)):
            i, j = piv[a], piv[b]
            if j - i < 10 or hi[j] >= hi[i]:
                continue
            egim = (hi[j] - hi[i]) / (j - i)
            cizgi = hi[i] + egim * (np.arange(i, m + 1) - i)
            if cizgi[-1] <= 0:
                continue
            # cizgi yeterince egimli olmali (yatay sikisma cizgisi dusen trend degildir)
            if (cizgi[0] - cizgi[-1]) / cizgi[0] * 100 < p["trend_egim"]:
                continue
            on_uz = son_bar - i + 1
            cizgi_on = cizgi[:on_uz]
            # kirilimdan once kapanis cizgiyi gecmemis, mum ucu da cok tasmamis olmali
            if (cl[i:son_bar + 1] > cizgi_on * (1 + tol)).any():
                continue
            if (hi[i:son_bar + 1] > cizgi_on * (1 + 3 * tol)).any():
                continue
            # temas: cizgiye degen AYRI tepeler (aralarinda en az 4 mum)
            temas_idx = [q for q in piv if i <= q <= son_bar
                         and abs(hi[q] - cizgi[q - i]) <= tol * cizgi[q - i]]
            temas, onceki = 0, None
            for q in temas_idx:
                if onceki is None or q - onceki >= 4:
                    temas += 1
                onceki = q
            if temas < p["trend_temas"]:
                continue
            cizgi_son = float(cizgi[-1])
            son_kap = float(cl[m])
            if son_kap < cizgi_son:
                ustte.append(cizgi_son)
            # kapanmis mum (sondan bir onceki) %0,2, aktif mum (son) %0,5 ustunde olmali; fiyat hala cizgi ustunde
            # penceredeki mumlara bak: en eski kirilim mumu (geri = kac mum once; 0 = aktif mum)
            kgeri = None
            for geri in range(sn - 1, -1, -1):
                esik = KIRILIM_AKTIF if geri == 0 else KIRILIM_KAPANMIS
                if cl[m - geri] > cizgi[-1 - geri] * (1 + esik / 100.0):
                    kgeri = geri
                    break
            if kgeri is not None and son_kap > cizgi_son:
                durum = 2  # kirilim: kapanmis mum cizgiyi %0,2 ya da aktif mum %0,5 asti ve fiyat hala cizginin ustunde
            elif son_kap <= cizgi_son * (1 + tol) and son_kap >= cizgi_son * (1 - p["trend_yakin"] / 100.0):
                durum = 1
            else:
                continue
            anahtar = (durum, temas, j)
            if en_iyi is None or anahtar > en_iyi[0]:
                ort_hacim = vol[max(0, m - 20):m].mean() if m > 0 else np.nan
                hacim_x = vol[m] / ort_hacim if ort_hacim and ort_hacim > 0 else np.nan
                lo = d["Low"].to_numpy(dtype=float)
                dip_i = i + int(np.argmin(lo[i:m + 1]))
                dip = float(lo[dip_i])
                # Fibonacci icin SON dusus bacagi: dipten onceki FIB_GERI mum icindeki en yuksek nokta
                tepe_fib = float(hi[max(i, dip_i - FIB_GERI):dip_i + 1].max())
                # kirilim mumu: kapanmis mum (sondan bir onceki) sartı sagladiysa o, yoksa aktif (son) mum
                kgeri = kgeri if durum == 2 else None
                en_iyi = (anahtar, dict(durum=durum, temas=temas, cizgi=cizgi_son,
                                        kirilim=(son_kap / cizgi_son - 1) * 100, hacim=hacim_x,
                                        tepe=tepe_fib, dip=dip,
                                        nokta_i=(d.index[i], float(hi[i])), nokta_j=(d.index[j], float(hi[j])),
                                        kirilim_geri=kgeri,
                                        kirilim_zaman=None if kgeri is None else d.index[m - kgeri]))
    if ust_liste is not None:        # 4 saatlik direnç için: fiyatın altında kalan geçerli çizgilerin son değerleri
        ust_liste.extend(ustte)
    if en_iyi is None:
        return None
    sonuc = en_iyi[1]
    yakin = [c for c in ustte if (c / float(cl[m]) - 1) * 100 <= UST_CIZGI_YAKIN]
    sonuc["ust_cizgi"] = min(yakin) if yakin else None
    return sonuc


def donus_teyidi(df):
    """Dusen hissede tepki basladi mi? 3 basit kontrol (son kapanmis mum):
    yesil mum, hacim son 20 mum ortalamasinin ustunde, onceki mumun dibi kirilmamis."""
    yesil = bool(df["Close"].iloc[-1] > df["Open"].iloc[-1])
    hacim = bool(df["Volume"].iloc[-1] > df["Volume"].iloc[-21:-1].mean())
    dip = bool(df["Low"].iloc[-1] >= df["Low"].iloc[-2])
    isaret = ("🟢" if yesil else "") + ("📊" if hacim else "") + ("🔒" if dip else "")
    return int(yesil) + int(hacim) + int(dip), isaret


def kama_hesapla(kapanis, n=KAMA_N, hizli=KAMA_HIZLI, yavas=KAMA_YAVAS):
    """Kaufman uyarlanabilir hareketli ortalama (TradingView ile ayni formul)."""
    c = np.asarray(kapanis, dtype=float)
    sonuc = np.full(len(c), np.nan)
    if len(c) <= n:
        return sonuc
    hizli_sc, yavas_sc = 2.0 / (hizli + 1), 2.0 / (yavas + 1)
    fark = np.abs(np.diff(c, prepend=c[0]))
    kama = c[n - 1]
    sonuc[n - 1] = kama
    for t in range(n, len(c)):
        gurultu = fark[t - n + 1:t + 1].sum()
        er = abs(c[t] - c[t - n]) / gurultu if gurultu > 0 else 0.0
        sc = (er * (hizli_sc - yavas_sc) + yavas_sc) ** 2
        kama = kama + sc * (c[t] - kama)
        sonuc[t] = kama
    return sonuc


def kama_kesisimi(df, sn=2):
    """(kesti_mi, ustunde_mi, kama_son). Kesti = son `sn` mumdan birinde kapanis KAMA'yi
    alttan uste kesmis ve fiyat su an KAMA'nin ustunde. Ustunde = kesisim eski olsa da su an ustunde."""
    c = df["Close"].to_numpy(dtype=float)
    k = kama_hesapla(c)
    if np.isnan(k[-1]) or np.isnan(k[-1 - sn]):
        return False, False, float("nan")
    ustunde = bool(c[-1] > k[-1])
    kesti = False
    if ustunde:
        for t in range(len(c) - sn, len(c)):
            if c[t - 1] <= k[t - 1] and c[t] > k[t]:
                kesti = True
    return kesti, ustunde, float(k[-1])


# ------------------------------------------------- aşırı satış göstergeleri (app.py'den birebir)
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def dema(s, n):
    e = ema(s, n)
    return 2 * e - ema(e, n)


def rsi_hesapla(kapanis, periyot=14):
    fark = kapanis.diff()
    kazanc = fark.clip(lower=0)
    kayip = -fark.clip(upper=0)
    ort_k = kazanc.ewm(alpha=1 / periyot, adjust=False, min_periods=periyot).mean()
    ort_z = kayip.ewm(alpha=1 / periyot, adjust=False, min_periods=periyot).mean()
    rs = ort_k / ort_z.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(ort_z != 0, 100.0)
    return rsi.where(ort_z.notna())


def mfi_hesapla(df, n=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    akis = tp * df["Volume"]
    poz = akis.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = akis.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    oran = poz / neg.replace(0, np.nan)
    mfi = 100 - 100 / (1 + oran)
    return mfi.where(neg != 0, 100.0)


def wave_trend(df, n1=10, n2=21):
    ap = (df["High"] + df["Low"] + df["Close"]) / 3
    esa = ema(ap, n1)
    d = ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, n2)
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2


def stoch_rsi(kapanis, rsi_n=14, stoch_n=14, k_n=3, d_n=3):
    rsi = rsi_hesapla(kapanis, rsi_n)
    alt = rsi.rolling(stoch_n).min()
    ust = rsi.rolling(stoch_n).max()
    st_ = (rsi - alt) / (ust - alt).replace(0, np.nan) * 100
    k = st_.rolling(k_n).mean()
    d = k.rolling(d_n).mean()
    return k, d


def nadaraya_tv(kapanis, h=8.0, carpan=3.0, n_max=500):
    """TradingView'deki LuxAlgo 'repainting' moduna yaklasik benzetme (app.py ile ayni)."""
    x = kapanis.to_numpy(dtype=float)[-n_max:]
    n = len(x)
    idx = np.arange(n)
    w = np.exp(-((idx[:, None] - idx[None, :]) ** 2) / (2 * h * h))
    y = (w @ x) / w.sum(axis=1)
    sae = np.abs(x - y).sum() / max(n - 1, 1) * carpan
    nw = pd.Series(np.nan, index=kapanis.index)
    nw.iloc[-n:] = y
    return nw, nw - sae, nw + sae


def yukari_kesti(a, b, son_n):
    k = (a.shift(1) <= b.shift(1)) & (a > b)
    return bool(k.iloc[-son_n:].any())


def atr_hesapla(df, n=14):
    onceki = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - onceki).abs(),
                    (df["Low"] - onceki).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bugunku_mumlar(df):
    gunler = np.array(df.index.date)
    return df[gunler == gunler[-1]]


def acilistan_dusus(df):
    """Bugunku seansin acilis fiyatindan son kapanisa degisim (%)."""
    return (df["Close"].iloc[-1] / bugunku_mumlar(df)["Open"].iloc[0] - 1) * 100


def kriter_macd(df, p):
    macd = dema(df["Close"], 12) - dema(df["Close"], 26)
    sinyal = ema(macd, 9)
    hist = macd - sinyal
    altinda = bool(macd.iloc[-1] < p["macd_seviye"])
    kesti = yukari_kesti(macd, sinyal, p["son_n"])
    h1, h0 = hist.iloc[-1], hist.iloc[-2]
    yakin = bool(h1 < 0 and h1 > h0 and abs(h1) <= p["macd_yakinlik"] * hist.abs().iloc[-50:].max())
    return 2 if (altinda and kesti) else 1 if (altinda and yakin) else 0


def kriter_mfi(df, p):
    v = mfi_hesapla(df).iloc[-1]
    return 2 if v <= p["mfi_os"] else 1 if v <= p["mfi_os"] + 8 else 0


def kriter_wt(df, p):
    wt1, wt2 = wave_trend(df)
    kesti = yukari_kesti(wt1, wt2, p["son_n"])
    altta = bool(wt1.iloc[-1] < 0)
    yaklasiyor = bool(wt1.iloc[-1] < wt2.iloc[-1] and wt1.iloc[-1] > wt1.iloc[-2]
                      and (wt2.iloc[-1] - wt1.iloc[-1]) <= p["wt_yakin"] and altta)
    return 2 if (kesti and altta) else 1 if yaklasiyor else 0


def kriter_stoch(df, p):
    k, d = stoch_rsi(df["Close"])
    kesti = yukari_kesti(k, d, p["son_n"])
    asiri = bool(k.iloc[-(p["son_n"] + 1):].min() <= p["stoch_os"])
    yakin_asiri = bool(k.iloc[-1] <= p["stoch_os"] + 10)
    return 2 if (asiri and kesti) else 1 if (asiri or (yakin_asiri and kesti)) else 0


def kriter_nw(df, p):
    """TradingView tarzi (repaint): kapanis alt bandin altina yeni inmisse = yesil ucgen."""
    nw, alt, ust = nadaraya_tv(df["Close"], p["nw_h"], p["nw_carpan"])
    kap = df["Close"]
    alt_alti = kap < alt
    ilk = alt_alti & ~alt_alti.shift(1, fill_value=False)
    ucgen = bool(ilk.iloc[-p["son_n"]:].any())
    konum = (kap.iloc[-1] - alt.iloc[-1]) / (ust.iloc[-1] - alt.iloc[-1]) * 100
    yakin = bool(alt_alti.iloc[-1]) or konum <= 10
    return 2 if ucgen else 1 if yakin else 0


ASIRI_KRITERLER = (("MACD", kriter_macd), ("MFI", kriter_mfi), ("Wave Trend", kriter_wt),
                   ("Stoch RSI", kriter_stoch), ("Nadaraya", kriter_nw))
ISARET = {2: "✅", 1: "🟡", 0: "—"}


def asiri_satis(df, sembol):
    """Aşırı satış tepkisi: app.py'deki seçim kuralının aynısı.
    (puan >= ASIRI_EN_AZ_KRITER) ve (günlük ya da açılıştan düşüş >= ASIRI_MIN_DUSUS) ve
    (tabana oturmamış) ve (dönüş teyidi >= ASIRI_EN_AZ_TEYIT). Bulursa sonuç sözlüğü, yoksa None.
    Ucuz kontroller önce yapılır; beş gösterge yalnızca düşüş ve teyit şartı geçen hisselere hesaplanır."""
    degisim = gunluk_degisim(df)
    acilis = float(acilistan_dusus(df))
    dusus = ((not np.isnan(degisim)) and degisim <= -ASIRI_MIN_DUSUS) or acilis <= -ASIRI_MIN_DUSUS
    if not dusus:
        return None
    if (not np.isnan(degisim)) and degisim <= TABAN_ESIGI:
        return None
    teyit_n, teyit_isaret = donus_teyidi(df)
    if teyit_n < ASIRI_EN_AZ_TEYIT:
        return None
    durumlar = [f(df, PA) for _, f in ASIRI_KRITERLER]
    tam = sum(1 for d in durumlar if d == 2)
    yakin = sum(1 for d in durumlar if d == 1)
    puan = tam + (yakin if PA["yakin_say"] else 0)
    if puan < ASIRI_EN_AZ_KRITER:
        return None
    fiyat = float(df["Close"].iloc[-1])
    atr = float(atr_hesapla(df).iloc[-1])
    stop = fiyat - STOP_ATR * atr
    nw_orta = float(nadaraya_tv(df["Close"], PA["nw_h"], PA["nw_carpan"])[0].iloc[-1])
    hedef = nw_orta if nw_orta > fiyat else None
    risk = fiyat - stop
    rg = (hedef - fiyat) / risk if (hedef is not None and risk > 0) else None
    return dict(
        sembol=sembol, fiyat=fiyat,
        gunluk=None if np.isnan(degisim) else float(degisim), acilis=acilis,
        puan=int(puan), durumlar=durumlar, teyit=int(teyit_n), isaret=teyit_isaret,
        stop=stop, stop_uzak=(fiyat - stop) / fiyat * 100,
        hedef=hedef, hedef_uzak=None if hedef is None else (hedef / fiyat - 1) * 100, rg=rg,
    )


# ------------------------------------------------------------------------ yardımcı
def hisse_listesi():
    icerik = (Path(__file__).parent / "hisseler.txt").read_text(encoding="utf-8")
    return liste_coz(icerik)


def veri_cek(sembol):
    """app.py'deki veri_cek ile aynı işlem; Yahoo geçici hata verirse 3 kez dener."""
    df = None
    for deneme in range(3):
        try:
            df = yf.Ticker(sembol + ".IS").history(period=PERIYOT, interval=ARALIK, auto_adjust=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.5 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Istanbul")
    df = df.resample(f"{MUM_DAKIKA}min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "Close"])
    return None if df.empty else df


STOP_ALTI = 0.5                 # stop: kirilan cizginin % kac altinda
FIB_ORANLARI = (0.382, 0.5, 0.618)


def stop_hedef(fiyat, cizgi, tepe, dip):
    """Stop = kirilan cizginin biraz alti. Hedefler = son dusus bacaginin (dipten onceki
    FIB_GERI mum icindeki tepe -> dip) Fibonacci duzeltme seviyeleri; fiyatin zaten
    gectigi seviyeler listeye alinmaz."""
    stop = cizgi * (1 - STOP_ALTI / 100.0)
    hedefler = []
    if tepe > dip:
        for r in FIB_ORANLARI:
            seviye = dip + r * (tepe - dip)
            if seviye > fiyat:
                hedefler.append((r, seviye))
    return dict(stop=stop, stop_uzak=(fiyat - stop) / fiyat * 100, hedefler=hedefler)


def gunluk_islem_tutari(df, bugun):
    """Son LIKIDITE_GUN tam gunun medyan gunluk islem tutari (fiyat x hacim, TL). Veri azsa nan."""
    tutar = (df["Close"] * df["Volume"]).groupby(df.index.date).sum()
    tutar = tutar[np.array([g != bugun for g in tutar.index])].iloc[-LIKIDITE_GUN:]
    return float(tutar.median()) if len(tutar) >= 5 else float("nan")


def trend_sinyali(df, sembol):
    """Düşen trend sinyali: 🚀 güçlü kırılım ya da ⏳ kırılım adayı. Yoksa None."""
    tr = trend_kirilimi(df, P)
    if not tr:
        return None
    if tr["durum"] == 2:
        return guclu_kirilim(df, sembol, tr)
    if ERKEN_GOSTER and tr["durum"] == 1:
        return kirilim_adayi(df, sembol, tr)
    return None


def sikisma_orani(df, geri):
    """Kırılım mumundan hemen önceki SIKISMA_KISA mumun ortalama boyu / önceki SIKISMA_UZUN mumun ortalama boyu.
    geri: kırılım mumu kaç mum önce (0 = aktif mum). Veri yetmezse ya da boy 0 ise None."""
    son = len(df) - 1 - int(geri or 0)          # kırılım mumunun sırası
    uzun = (df["High"] - df["Low"]).iloc[max(0, son - SIKISMA_UZUN):son]
    if len(uzun) < SIKISMA_UZUN // 2:
        return None
    kisa = uzun.iloc[-SIKISMA_KISA:]
    if uzun.mean() <= 0:
        return None
    return float(kisa.mean() / uzun.mean())


def guclu_kirilim(df, sembol, tr):
    """Kırılım olmuş. Hacim + en az MIN_TEYIT teyit varsa 🚀 (guclu); hacimsiz ama 5/6 varsa 🔸 (hacimsiz);
    teyit yetmiyorsa kademe "elenen" (bildirilmez, gün sonu raporunda ayrıca ölçülür)."""
    teyit_n, isaret = donus_teyidi(df)
    kama_kesti, kama_ustunde, kama_son = kama_kesisimi(df)
    if kama_ustunde:                 # fiyat su an KAMA'nin ustundeyse teyit sayilir
        teyit_n += 1
        isaret += "📈"
    if macd_teyidi(df):              # son 5 mumda MACD yukari kesisim
        teyit_n += 1
        isaret += "🔵"
    if wt_teyidi(df):                # son 5 mumda Wave Trend yukari kesisim
        teyit_n += 1
        isaret += "🌊"
    if "📊" in isaret and teyit_n >= MIN_TEYIT:
        kademe = "guclu"
    elif "📊" not in isaret and teyit_n >= MIN_TEYIT_HACIMSIZ:
        kademe = "hacimsiz"
    else:
        kademe = "elenen"            # kirilim var ama teyit yetmedi: bildirilmez, sadece rapor icin kaydedilir
    degisim = gunluk_degisim(df)
    return dict(
        sembol=sembol, fiyat=float(df["Close"].iloc[-1]),
        gunluk=None if np.isnan(degisim) else float(degisim),
        cizgi=float(tr["cizgi"]), temas=int(tr["temas"]), kirilim=float(tr["kirilim"]),
        hacim=None if np.isnan(tr["hacim"]) else float(tr["hacim"]),
        teyit=int(teyit_n), isaret=isaret, kademe=kademe,
        eksik=eksik_sartlar(isaret),
        kama=None if np.isnan(kama_son) else kama_son,
        kama_ustunde=bool(kama_ustunde), kama_yeni=bool(kama_kesti),
        ust_cizgi=tr.get("ust_cizgi"), son_mum=df.index[-1],
        sikisma=sikisma_orani(df, tr.get("kirilim_geri")) if SIKISMA_GOSTER else None,
        nokta_i=tr.get("nokta_i"), nokta_j=tr.get("nokta_j"),
        kirilim_geri=tr.get("kirilim_geri"), kirilim_zaman=tr.get("kirilim_zaman"),
        **stop_hedef(float(df["Close"].iloc[-1]), float(tr["cizgi"]), tr["tepe"], tr["dip"]),
    )


def macd_teyidi(df, n=None):
    """Son n mumda MACD (mavi) sinyal cizgisini (kirmizi) asagidan yukari kesti mi? Bolge sarti yok."""
    n = n or TEYIT_SON_N
    macd = dema(df["Close"], 12) - dema(df["Close"], 26)   # app.py ile ayni MACD
    return yukari_kesti(macd, ema(macd, 9), n)


def wt_teyidi(df, n=None):
    """Son n mumda Wave Trend yesil cizgisi (wt1) kirmiziyi (wt2) yukari kesti mi? Bolge sarti yok."""
    n = n or TEYIT_SON_N
    wt1, wt2 = wave_trend(df)
    return yukari_kesti(wt1, wt2, n)


def mfi_yukari_kesti(df, n=None):
    """Son n mumda MFI, ERKEN_MFI_SEVIYE'yi aşağıdan yukarı kesti mi? Kaç mum önce olduğunu döner (0 = son mum), yoksa None."""
    n = n or ERKEN_SON_N
    mfi = mfi_hesapla(df).to_numpy(dtype=float)
    for geri in range(n):
        t = len(mfi) - 1 - geri
        if t >= 1 and mfi[t - 1] < ERKEN_MFI_SEVIYE <= mfi[t]:
            return geri
    return None


def wt_al_verdi(df, n=None):
    """Son n mumda Wave Trend al verdi mi (wt1, wt2'yi sıfırın altında aşağıdan yukarı kesti)? Kaç mum önce, yoksa None."""
    n = n or ERKEN_SON_N
    wt1, wt2 = wave_trend(df)
    a, b = wt1.to_numpy(dtype=float), wt2.to_numpy(dtype=float)
    for geri in range(n):
        t = len(a) - 1 - geri
        if t >= 1 and a[t - 1] <= b[t - 1] and a[t] > b[t] and a[t] < 0:
            return geri
    return None


def kirilim_adayi(df, sembol, tr):
    """⏳ Henüz kırılmamış ama çizgiye çok yakın; son birkaç mumda MFI ve Wave Trend birlikte al vermiş."""
    if tr["kirilim"] < -ERKEN_YAKIN:
        return None
    mfi_geri = mfi_yukari_kesti(df)
    if mfi_geri is None:
        return None
    wt_geri = wt_al_verdi(df)
    if wt_geri is None:
        return None
    fiyat = float(df["Close"].iloc[-1])
    sh = stop_hedef(fiyat, float(tr["cizgi"]), tr["tepe"], tr["dip"])
    # aday henüz çizginin altında; stop, son birkaç mumun dibinin biraz altı
    stop = float(df["Low"].iloc[-ERKEN_SON_N:].min()) * (1 - STOP_ALTI / 100.0)
    sh["stop"], sh["stop_uzak"] = stop, (fiyat - stop) / fiyat * 100
    degisim = gunluk_degisim(df)
    return dict(
        sembol=sembol, fiyat=fiyat, gunluk=None if np.isnan(degisim) else float(degisim),
        cizgi=float(tr["cizgi"]), temas=int(tr["temas"]), kirilim=float(tr["kirilim"]),
        hacim=None, teyit=2, isaret="", kademe="erken", eksik=[], ust_cizgi=None,
        mfi_geri=int(mfi_geri), wt_geri=int(wt_geri),
        nokta_i=tr.get("nokta_i"), nokta_j=tr.get("nokta_j"), **sh,
    )


def veri_cek_uzun(sembol):
    """4 saatlik düşen çizgi için ~6 aylık saatlik mum. Alınamazsa None."""
    df = None
    for deneme in range(2):
        try:
            df = yf.Ticker(sembol + ".IS").history(period=DIRENC_PERIYOT, interval="1h", auto_adjust=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.0 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Istanbul")
    return df


def veri_cek_gunluk(sembol):
    """Günlük düşen çizgi için ~1 yıllık günlük mum. Alınamazsa None."""
    df = None
    for deneme in range(2):
        try:
            df = yf.Ticker(sembol + ".IS").history(period="1y", interval="1d", auto_adjust=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.0 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return None if df.empty else df


def gunluk_durum(df, fiyat):
    """Günlük grafikte (saatlikle aynı çizgi kuralı) düşen trend çizgisi durumu:
    kirildi (yuzde = çizgi üstü %, geri = kaç gün önce, 0 = bugün), yakin (çizgiye yapışık),
    altinda (yuzde = çizgi fiyatın kaç % yukarısında), yok. Veri azsa None."""
    if df is None or len(df) < 60:
        return None
    ust = []
    tr = trend_kirilimi(df, P, ust_liste=ust)
    if tr and tr["durum"] == 2:
        return dict(tur="kirildi", yuzde=float(tr["kirilim"]), geri=int(tr.get("kirilim_geri") or 0))
    ustunde = [c for c in ust if c > fiyat]
    if ustunde:
        return dict(tur="altinda", yuzde=(min(ustunde) / fiyat - 1) * 100)
    if tr and tr["durum"] == 1:
        return dict(tur="yakin", yuzde=float(tr["kirilim"]))
    return dict(tur="yok")


def gunluk_bilgi(sembol, fiyat):
    """gunluk_durum + kısa süreli önbellek (döngü modunda aynı hisse her taramada yeniden çekilmesin)."""
    simdi = time.time()
    eski = GUNLUK_ONBELLEK.get(sembol)
    if eski and simdi - eski[0] < GUNLUK_ONBELLEK_SN:
        return eski[1]
    sonuc = gunluk_durum(veri_cek_gunluk(sembol), fiyat)
    GUNLUK_ONBELLEK[sembol] = (simdi, sonuc)
    return sonuc


def dort_saatlik(df):
    """Saatlik mumlardan 4 saatlik mumlar üretir (seans: 10-14 ve 14-18)."""
    bas = df.index.normalize() + pd.to_timedelta(np.where(df.index.hour >= 14, 14, 10), unit="h")
    g = df.groupby(bas)
    return pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                         "Close": g["Close"].last(), "Volume": g["Volume"].sum()})


def direnc_4s(df, fiyat):
    """Fiyatın üstündeki en yakın GEÇERLİ düşen trend çizgisi (4 saatlik grafikte, saatlik grafikteki
    çizgi bulma kuralının aynısıyla); yoksa None. Henüz kırılmamış, fiyatın üstünde kalan çizgiler aranır."""
    d4 = dort_saatlik(df)
    if len(d4) < 40:
        return None
    ust = []
    trend_kirilimi(d4, P, ust_liste=ust)
    ustunde = [c for c in ust if c > fiyat]
    return min(ustunde) if ustunde else None


def ilk_hedef(r):
    if "hedefler" in r:
        return r["hedefler"][0][1] if r["hedefler"] else None
    return r.get("hedef")


def etkin_hedef(fiyat, hedef, direnc):
    """Fiyatın üstündeki hedef ve direncin yakın olanı. Dönüş: (hedef, dirence_gore_mi)."""
    adaylar = [(h, ad) for h, ad in ((hedef, "hedef"), (direnc, "direnc")) if h is not None and h > fiyat]
    if not adaylar:
        return None, False
    h, ad = min(adaylar)
    return h, ad == "direnc"


def hedef_bilgisi(r):
    return etkin_hedef(r["fiyat"], ilk_hedef(r), r.get("direnc"))


def rg_hesapla(fiyat, stop, hedef):
    if hedef is None or hedef <= fiyat or fiyat <= stop:
        return None
    return (hedef - fiyat) / (fiyat - stop)


def giris_satiri(fiyat, stop, hedef, dirence=False):
    """'En fazla bu fiyata kadar gir' sınırı: risk/getiri GIRIS_MIN_RG'nin altına düşmesin.
    Hedef olarak ilk hedef ya da (daha yakınsa) 4 saatlik direnç alınır."""
    rg = rg_hesapla(fiyat, stop, hedef)
    if rg is None:
        return ""
    sinir = (hedef + GIRIS_MIN_RG * stop) / (1 + GIRIS_MIN_RG)
    ek = " · hedef olarak 4s direnç alındı" if dirence else ""
    if rg < GIRIS_MIN_RG:
        return (f"⚠️ R/G şu an {sayi(rg, 1)} ({sayi(GIRIS_MIN_RG, 1)} altında) · "
                f"giriş sınırı {sayi(sinir)} geçilmiş{ek}\n")
    return (f"Giriş sınırı {sayi(sinir)} (R/G ≥ {sayi(GIRIS_MIN_RG, 1)} için) · "
            f"şu an R/G {sayi(rg, 1)}{ek}\n")


def risk_getiri_satiri(r):
    hedef, dirence = hedef_bilgisi(r)
    return giris_satiri(r["fiyat"], r["stop"], hedef, dirence)


def direnc_satiri(r):
    if "direnc" not in r:
        return ""
    if r["direnc"] is None:
        return "4s düşen çizgi direnci: geçerli çizgi yok\n"
    uzak = (r["direnc"] / r["fiyat"] - 1) * 100
    return f"4s düşen çizgi direnci {sayi(r['direnc'])} (+%{sayi(uzak, 1)})\n"


def analiz(sembol, bugun):
    """(durum, trend_sonucu, asiri_sonucu) döner.
    durum: veri_yok / kisa / bugun_yok / yok / bulundu (iki taramadan en az biri sinyal verdi)"""
    df = veri_cek(sembol)
    if df is None:
        return "veri_yok", None, None
    if len(df) < MIN_MUM:
        return "kisa", None, None
    if df.index[-1].date() != bugun:
        return "bugun_yok", None, None   # tatil günü ya da bugün hiç işlem görmemiş hisse
    LIKIDITE[sembol] = gunluk_islem_tutari(df, bugun)
    if sembol in RETEST_IZLE:            # bu hissenin bugün bildirilmiş kırılımı var: retest oldu mu
        try:
            RETEST_SONUC[sembol] = retest_kontrol(df, RETEST_IZLE[sembol])
        except Exception as e:
            print(f"Retest kontrolü hata verdi: {sembol} ({type(e).__name__}: {e})")
    tr_r = trend_sinyali(df, sembol)
    if tr_r and LIKIDITE_MIN > 0 and LIKIDITE[sembol] < LIKIDITE_MIN:   # nan ise filtre uygulanmaz
        tr_r = None
    if GUNLUK_GOSTER and tr_r and tr_r.get("kademe") in ("guclu", "hacimsiz"):
        try:
            tr_r["gunluk_cizgi"] = gunluk_bilgi(sembol, float(df["Close"].iloc[-1]))
        except Exception as e:
            print(f"Günlük çizgi hesabı hata verdi: {sembol} ({type(e).__name__}: {e})")
    try:
        as_r = asiri_satis(df, sembol) if ASIRI_GOSTER else None
    except Exception as e:               # tek hissenin hesap hatası taramanın tamamını durdurmasın
        print(f"Aşırı satış hesabı hata verdi: {sembol} ({type(e).__name__}: {e})")
        as_r = None
    if DIRENC_GOSTER and (tr_r or as_r):
        try:
            uzun = veri_cek_uzun(sembol)
            direnc = direnc_4s(uzun if uzun is not None else df, float(df["Close"].iloc[-1]))
            for r in (tr_r, as_r):
                if r:
                    r["direnc"] = direnc
        except Exception as e:
            print(f"4 saatlik direnç hesabı hata verdi: {sembol} ({type(e).__name__}: {e})")
    return ("bulundu" if (tr_r or as_r) else "yok"), tr_r, as_r


def sayi(x, n=2, isaretli=False):
    s = f"{x:+.{n}f}" if isaretli else f"{x:.{n}f}"
    return s.replace(".", ",")


def eksik_sartlar(isaret):
    adlar = (("🟢", "yeşil mum"), ("📊", "hacim"), ("🔒", "dip korunması"), ("📈", "KAMA üstü"),
             ("🔵", "MACD kesişimi"), ("🌊", "Wave Trend kesişimi"))
    return [ad for sim, ad in adlar if sim not in isaret]


def kama_satiri(r):
    if r.get("kama") is None:
        return ""
    if r.get("kama_ustunde"):
        ek = " · fiyat üstünde" + (" (yeni kesişim)" if r.get("kama_yeni") else "")
    else:
        ek = " · fiyat altında"
    return f"KAMA {sayi(r['kama'])}{ek}\n"


def gunluk_satiri(r):
    g = r.get("gunluk_cizgi")
    if not g:
        return ""
    if g["tur"] == "kirildi":
        zaman = "bugün" if g["geri"] == 0 else f"{g['geri']} gün önce"
        return f"📅 Günlük: düşen çizgi kırıldı ({zaman}, çizgi üstü %{sayi(g['yuzde'], 1, True)})\n"
    if g["tur"] == "altinda":
        return f"📅 Günlük: düşen çizgi %{sayi(g['yuzde'], 1)} yukarıda\n"
    if g["tur"] == "yakin":
        return f"📅 Günlük: düşen çizgiye yapışık (%{sayi(g['yuzde'], 1, True)})\n"
    return "📅 Günlük: geçerli düşen çizgi yok\n"


def sikisma_satiri(r):
    o = r.get("sikisma")
    if o is None:
        return ""
    if o < SIKISMA_ESIK:
        return f"🗜 Sıkışma: var · kırılım öncesi mum boyu normalin %{o * 100:.0f} düzeyinde\n"
    return f"🗜 Sıkışma: yok · mum boyu normalin %{o * 100:.0f} düzeyinde\n"


def stop_hedef_satirlari(r):
    satir = f"Stop {sayi(r['stop'])} (-%{sayi(r['stop_uzak'], 1)})\n"
    if r["hedefler"]:
        satir += "Hedef " + " · ".join(sayi(sv) for _, sv in r["hedefler"]) + "\n"
    else:
        satir += "Hedef: fib seviyeleri geçilmiş\n"
    if r.get("ust_cizgi"):
        satir += f"⚠️ Üstte hâlâ geçerli çizgi var: {sayi(r['ust_cizgi'])}\n"
    return satir


def siralama_anahtari(r):
    """En cok teyit once; sonra ustte cizgi uyarisi olmayan; sonra hacim, temas."""
    return (-r["teyit"], r.get("ust_cizgi") is not None, -(r["hacim"] or 0.0), -r["temas"], r["sembol"])


def cizgi_nokta_satiri(r):
    a, b = r.get("nokta_i"), r.get("nokta_j")
    if not a or not b:
        return ""
    return f"Çizgi noktaları: {a[0]:%d.%m %H:%M} ({sayi(a[1])}) → {b[0]:%d.%m %H:%M} ({sayi(b[1])})\n"


def kirilim_mum_satiri(r):
    if r.get("kirilim_geri") is None or r.get("kirilim_zaman") is None:
        return ""
    g = r["kirilim_geri"]
    durum = "aktif, kapanmadı" if g == 0 else f"kapanmış, {g} mum önce"
    return f"Kırılım mumu: {r['kirilim_zaman']:%H:%M} ({durum})\n"


def blok_trend(r, yildiz):
    gun = f"gün {sayi(r['gunluk'], 1, True)}%" if r["gunluk"] is not None else "gün ?"
    hac = f" · hacim {sayi(r['hacim'], 1)}x" if r["hacim"] is not None else ""
    return (
        f"{yildiz}{r['sembol']}  {sayi(r['fiyat'])} TL  ({gun})\n"
        f"Çizgi {sayi(r['cizgi'])} · {r['temas']} temas · çizgi üstü %{sayi(r['kirilim'], 1, True)}{hac}\n"
        f"{kirilim_mum_satiri(r)}"
        f"Teyit {r['teyit']}/{TEYIT_TOPLAM} {r['isaret']}"
        f"{'  · eksik: ' + ', '.join(r['eksik']) if r['eksik'] else ''}\n"
        f"{kama_satiri(r)}"
        f"{gunluk_satiri(r)}"
        f"{sikisma_satiri(r)}"
        f"{stop_hedef_satirlari(r)}"
        f"{risk_getiri_satiri(r)}"
        f"{direnc_satiri(r)}"
        f"https://www.tradingview.com/chart/?symbol=BIST%3A{r['sembol']}\n"
    )


def tl_kisa(x):
    return f"{x / 1e6:.1f}".replace(".", ",") + " mn"


def likidite_ozeti(sinyaller):
    """Deneme modunda: 582 hissenin gunluk islem tutari dagilimi (esigi veriden secmek icin)."""
    v = {s: x for s, x in LIKIDITE.items() if x == x}
    if not v:
        return "💧 Likidite verisi hesaplanamadı."
    sirali = sorted(v.values())
    yuzde = lambda q: sirali[min(len(sirali) - 1, int(q * len(sirali)))]
    satirlar = [f"💧 Günlük işlem tutarı dağılımı (son {LIKIDITE_GUN} günün medyanı, TL) · {len(v)} hisse",
                "Yüzdelikler: " + " · ".join(f"%{int(q * 100)}: {tl_kisa(yuzde(q))}" for q in (0.1, 0.25, 0.5, 0.75, 0.9)),
                ""]
    kenar = [0, 1e6, 2.5e6, 5e6, 10e6, 25e6, 50e6, float("inf")]
    for alt, ust in zip(kenar[:-1], kenar[1:]):
        grup = sorted((x, s) for s, x in v.items() if alt <= x < ust)
        if not grup:
            continue
        ad = f"{tl_kisa(alt)} - {tl_kisa(ust)}" if ust != float("inf") else f"{tl_kisa(alt)} üstü"
        if alt == 0:
            ad = f"{tl_kisa(ust)} altı"
        orta = len(grup) // 2
        ornek = ", ".join(s for _, s in grup[max(0, orta - 1):orta + 2])
        satirlar.append(f"{ad}: {len(grup)} hisse (örn. {ornek})")
    if sinyaller:
        satirlar += ["", "Şu an sinyal verenler: " + " · ".join(
            f"{r['sembol']} {tl_kisa(v[r['sembol']]) if r['sembol'] in v else '?'}" for r in sinyaller)]
    satirlar += ["", "Eşik seçmek için: \"şu banttan altı bana sığ\" demeniz yeter (LIKIDITE_MIN)."]
    return "\n".join(satirlar)


def mum_oncesi(geri):
    return "şu anki mum" if geri == 0 else f"{geri} mum önce"


def blok_aday(r, yildiz):
    gun = f"gün {sayi(r['gunluk'], 1, True)}%" if r["gunluk"] is not None else "gün ?"
    return (
        f"{yildiz}{r['sembol']}  {sayi(r['fiyat'])} TL  ({gun})\n"
        f"Düşen çizgi {sayi(r['cizgi'])} (fiyat %{sayi(-r['kirilim'], 1)} altında) · {r['temas']} temas\n"
        f"MFI yukarı kesti ({mum_oncesi(r['mfi_geri'])}) · Wave Trend al ({mum_oncesi(r['wt_geri'])})\n"
        f"🔔 Midas alarmı: {sayi(r['cizgi'])} üstü\n"
        f"{stop_hedef_satirlari(r)}"
        f"{risk_getiri_satiri(r)}"
        f"https://www.tradingview.com/chart/?symbol=BIST%3A{r['sembol']}\n"
    )


def blok_asiri(r, yildiz):
    gun = f"gün {sayi(r['gunluk'], 1, True)}%" if r["gunluk"] is not None else "gün ?"
    kriterler = " · ".join(f"{ad} {ISARET[d]}" for (ad, _), d in zip(ASIRI_KRITERLER, r["durumlar"]))
    if r["hedef"] is not None:
        hedef = f"Hedef {sayi(r['hedef'])} (+%{sayi(r['hedef_uzak'], 1)})"
    else:
        hedef = "Hedef yok (fiyat Nadaraya orta çizgisinin üstünde)"
    return (
        f"{yildiz}{r['sembol']}  {sayi(r['fiyat'])} TL  ({gun} · açılıştan {sayi(r['acilis'], 1, True)}%)\n"
        f"Puan {r['puan']}/5 · {kriterler}\n"
        f"Teyit {r['teyit']}/3 {r['isaret']}\n"
        f"Stop {sayi(r['stop'])} (-%{sayi(r['stop_uzak'], 1)}) · {hedef}\n"
        f"{risk_getiri_satiri(r)}"
        f"{direnc_satiri(r)}"
        f"https://www.tradingview.com/chart/?symbol=BIST%3A{r['sembol']}\n"
    )


def bolum(tur, ad, liste, blok_fonksiyonu):
    """Bir başlık altındaki hisseleri (sembol, metin) çiftleri olarak hazırlar; ilki ⭐ (birden çoksa)."""
    bloklar = []
    for sira, r in enumerate(liste):
        yildiz = "⭐ " if (sira == 0 and len(liste) > 1) else ""
        bloklar.append((r["sembol"], blok_fonksiyonu(r, yildiz)))
    return (tur, ad, bloklar)


def mesaj_parcalari(bolumler, simdi, onek="", ek_satir="", baslik=None, alt=None):
    """Tüm başlıkları tek mesajda toplar; çok uzarsa parçalara böler.
    Dönüş: [(metin, [(tur, sembol), ...]), ...]  (her parçada hangi hisselerin gittiği bilinsin diye)."""
    ust = baslik if baslik is not None else f"{onek}📡 BIST tarayıcı · {simdi:%H:%M}"
    if ek_satir:
        ust += "\n" + ek_satir
    if alt is None:
        alt = ("⚠️ Veri ~15 dk gecikmeli. Aktif mumda sinyal, mum kapanınca kaybolabilir. "
               "Girmeden önce grafiği ve Midas'taki güncel fiyatı kontrol et.")
    parcalar, mevcut, anahtarlar = [], ust + "\n", []
    for tur, ad, bloklar in bolumler:
        mevcut += f"\n{ad} · {len(bloklar)} hisse\n\n"
        for sembol, metin in bloklar:
            if len(mevcut) + len(metin) > 3500:
                parcalar.append((mevcut.rstrip(), anahtarlar))
                mevcut, anahtarlar = f"{ad} (devam)\n\n", []
            mevcut += metin + "\n"
            anahtarlar.append((tur, sembol))
    parcalar.append(((mevcut + alt).strip(), anahtarlar))
    return parcalar


def gonder(bolumler, simdi, onek="", ek_satir=""):
    """Mesajı yollar. Dönüş: (başarıyla giden (tur, sembol) kümesi, hata_var_mı)."""
    gitti, hata = set(), False
    for metin, anahtarlar in mesaj_parcalari(bolumler, simdi, onek, ek_satir):
        if telegram_gonder(metin):
            gitti |= set(anahtarlar)
        else:
            hata = True
    return gitti, hata


# ------------------------------------------------------- sinyal kaydı ve gün sonu raporu
GECMIS_SINIR = 600   # geçmişte en fazla bu kadar değerlendirilmiş sinyal tutulur


def kayit_yap(tur, r, simdi):
    """Gönderilen bir sinyalin sonradan değerlendirilmesi için gereken bilgiler."""
    hedef, _ = hedef_bilgisi(r)      # mesajdaki hedef: ilk hedef ya da daha yakınsa 4s direnç
    if tur in ("g", "h", "x", "e"):
        ek = dict(ust_cizgi=r.get("ust_cizgi") is not None, teyit=r["teyit"], sikisma=r.get("sikisma"))
        if tur in ("g", "h", "x"):
            ek["tmax"] = TEYIT_TOPLAM      # teyit kaç üzerinden (eski 4'lü kayıtlarla karışmasın)
        if tur == "x":
            ek["isaret"] = r["isaret"]
        if tur == "e":
            ek["cizgi"] = r["cizgi"]
    else:
        ek = dict(puan=r["puan"], teyit=r["teyit"])
    rg = rg_hesapla(r["fiyat"], r["stop"], hedef)
    if rg is not None:
        ek["rg_ok"] = bool(rg >= GIRIS_MIN_RG)
    return dict(tur=tur, sembol=r["sembol"], saat=f"{simdi:%H:%M}", fiyat=r["fiyat"],
                stop=r["stop"], hedef=hedef, **ek)


def kayit_dosyasi():
    return DURUM_DOSYASI.parent / "sinyaller.json"


def gecmis_dosyasi():
    return DURUM_DOSYASI.parent / "gecmis.json"


def kayit_oku(bugun_str):
    try:
        d = json.loads(kayit_dosyasi().read_text(encoding="utf-8"))
        if d.get("tarih") == bugun_str:
            return d.get("kayitlar", [])
    except Exception:
        pass
    return []


def kayit_ekle(bugun_str, yeni):
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    kayit_dosyasi().write_text(
        json.dumps({"tarih": bugun_str, "kayitlar": kayit_oku(bugun_str) + yeni}, ensure_ascii=False),
        encoding="utf-8")


def gecmis_oku():
    try:
        return json.loads(gecmis_dosyasi().read_text(encoding="utf-8"))
    except Exception:
        return []


def gecmis_yaz(bugun_str, yeni):
    """Bugünün değerlendirilmiş sinyallerini geçmişe ekler (aynı gün iki kez çalışırsa üstüne yazar)."""
    liste = [k for k in gecmis_oku() if k.get("tarih") != bugun_str] + yeni
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    gecmis_dosyasi().write_text(json.dumps(liste[-GECMIS_SINIR:], ensure_ascii=False), encoding="utf-8")


def veri_cek_15dk(sembol):
    """Gün sonu değerlendirmesi için ham 15 dakikalık mumlar (birleştirme yok)."""
    df = None
    for deneme in range(3):
        try:
            df = yf.Ticker(sembol + ".IS").history(period="5d", interval="15m", auto_adjust=True)
        except Exception:
            df = None
        if df is not None and not df.empty:
            break
        time.sleep(1.5 * (deneme + 1))
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Istanbul")
    return df


def degerlendir(k, df15, tarih):
    """Sinyalden SONRAKİ 15 dakikalık mumlara bakar (sinyal saatini bir sonraki 15 dk sınırına yuvarlar).
    sonuc: hedef / stop / belirsiz (aynı mumda ikisi de) / yok (ikisine de değmedi) / bos (mum verisi yok).
    en yüksek / en düşük / kapanış, sinyal fiyatına göre yüzdedir ve sinyalden sonra günün sonuna kadardır."""
    h, m = (int(x) for x in k["saat"].split(":"))
    dak = -(-(h * 60 + m) // 15) * 15
    bas = pd.Timestamp(tarih.year, tarih.month, tarih.day, tz=TR) + pd.Timedelta(minutes=dak)
    gunluk = np.array(df15.index.date) == tarih
    yol = df15[(df15.index >= bas) & gunluk]
    if yol.empty:
        return dict(sonuc="bos")
    sonuc, olay = "yok", None
    for zaman, satir in yol.iterrows():
        stop_deydi = bool(satir["Low"] <= k["stop"])
        hedef_deydi = k["hedef"] is not None and bool(satir["High"] >= k["hedef"])
        if stop_deydi and hedef_deydi:
            sonuc, olay = "belirsiz", zaman
            break
        if stop_deydi:
            sonuc, olay = "stop", zaman
            break
        if hedef_deydi:
            sonuc, olay = "hedef", zaman
            break
    giris = k["fiyat"]
    ek = {}
    if k.get("cizgi"):
        asan = yol[yol["High"] >= k["cizgi"]]
        ek = dict(kirdi=not asan.empty, kirdi_saat=None if asan.empty else f"{asan.index[0]:%H:%M}")
    return dict(
        **ek, sonuc=sonuc, olay=None if olay is None else f"{olay:%H:%M}",
        yuk=(float(yol["High"].max()) / giris - 1) * 100,
        dus=(float(yol["Low"].min()) / giris - 1) * 100,
        kap=(float(yol["Close"].iloc[-1]) / giris - 1) * 100,
    )


def sonuc_metni(k, d):
    hedef_ad = "ilk hedefe" if k["tur"] in ("g", "h", "x", "e") else "hedefe"
    if d["sonuc"] == "hedef":
        return f"🎯 {d['olay']} mumunda {hedef_ad} ({sayi(k['hedef'])}) değdi"
    if d["sonuc"] == "stop":
        return f"🛑 {d['olay']} mumunda stopa ({sayi(k['stop'])}) değdi"
    if d["sonuc"] == "belirsiz":
        return f"❓ {d['olay']} mumunda hem stop hem hedef var, sıra belli değil"
    if k["hedef"] is None:
        return "⏳ stopa değmedi (hedef tanımlı değildi)"
    return "⏳ stopa da hedefe de değmedi"


def rapor_blogu(k, d):
    baslik = f"{k['sembol']}  {k['saat']} · {sayi(k['fiyat'])} TL"
    if k["tur"] == "x":
        baslik += f" · teyit {k.get('teyit')}/{TEYIT_TOPLAM} {k.get('isaret', '')}"
        if d["sonuc"] == "bos":
            return baslik + "\nmum verisi alınamadı\n"
        return (baslik + "\n"
                f"en yüksek %{sayi(d['yuk'], 1, True)} · en düşük %{sayi(d['dus'], 1, True)} · "
                f"kapanış %{sayi(d['kap'], 1, True)}\n")
    if d["sonuc"] == "bos":
        return baslik + "\n" + "Sinyalden sonra mum verisi alınamadı\n"
    cizgi = ""
    if "kirdi" in d:
        cizgi = (f"Çizgiye ({sayi(k['cizgi'])}) {d['kirdi_saat']} mumunda ulaştı\n" if d["kirdi"]
                 else f"Çizgiye ({sayi(k['cizgi'])}) ulaşamadı\n")
    return (baslik + "\n"
            f"en yüksek %{sayi(d['yuk'], 1, True)} · en düşük %{sayi(d['dus'], 1, True)} · "
            f"kapanış %{sayi(d['kap'], 1, True)}\n"
            f"{cizgi}"
            f"{sonuc_metni(k, d)}\n")


def ozet_satiri(etiket, grup):
    """Değerlendirilmiş (mum verisi olan) sinyallerin kısa özeti."""
    grup = [k for k in grup if k.get("sonuc") != "bos"]
    if not grup:
        return None
    n = len(grup)
    say = lambda ad: sum(1 for k in grup if k["sonuc"] == ad)
    ort = lambda alan: sum(k[alan] for k in grup) / n
    kd = [k for k in grup if "kirdi" in k]
    kirdi = f" · çizgiye ulaşan {sum(1 for k in kd if k['kirdi'])}/{len(kd)}" if kd else ""
    return (f"{etiket} {n} sinyal · 🎯 {say('hedef')} · 🛑 {say('stop')} · ❓ {say('belirsiz')} · "
            f"⏳ {say('yok')}{kirdi}\n   ort. en yüksek %{sayi(ort('yuk'), 1, True)} · "
            f"ort. en düşük %{sayi(ort('dus'), 1, True)} · ort. kapanış %{sayi(ort('kap'), 1, True)}")


def genel_toplam(gecmis):
    gunler = {k["tarih"] for k in gecmis}
    satirlar = [f"📈 Genel toplam ({len(gunler)} işlem günü)"]
    for tur, simge in (("g", "🚀"), ("h", "🔸"), ("e", "⏳"), ("a", "🔻"), ("x", "🚫 elenen")):
        satir = ozet_satiri(simge, [k for k in gecmis if k["tur"] == tur])
        if satir:
            satirlar.append(satir)
    # teyit sayisina gore (yalnizca 6'li sistemdeki 🚀 ve 🔸 kayitlari)
    for tur, simge in (("g", "🚀"), ("h", "🔸")):
        for n in range(MIN_TEYIT_HACIMSIZ if tur == "h" else MIN_TEYIT, TEYIT_TOPLAM + 1):
            grup = [k for k in gecmis if k["tur"] == tur and k.get("tmax") == TEYIT_TOPLAM and k.get("teyit") == n]
            satir = ozet_satiri(f"{simge} {n}/{TEYIT_TOPLAM} teyitli", grup)
            if satir:
                satirlar.append(satir)
    trend = [k for k in gecmis if k["tur"] in ("g", "h", "e") and "ust_cizgi" in k]
    for ad, secim in (("⚠️ üstte çizgi uyarılı 🚀🔸⏳", True), ("üstte çizgi uyarısız 🚀🔸⏳", False)):
        satir = ozet_satiri(ad, [k for k in trend if k["ust_cizgi"] == secim])
        if satir:
            satirlar.append(satir)
    rg_var = [k for k in gecmis if "rg_ok" in k and k["tur"] != "x"]
    for ad, secim in ((f"R/G ≥ {sayi(GIRIS_MIN_RG, 1)} olanlar", True), (f"⚠️ R/G < {sayi(GIRIS_MIN_RG, 1)} olanlar", False)):
        satir = ozet_satiri(ad, [k for k in rg_var if k["rg_ok"] == secim])
        if satir:
            satirlar.append(satir)
    if sum(1 for k in gecmis if k.get("sonuc") != "bos") < 30:
        satirlar.append("Örnek sayısı az; kesin yorum için erken.")
    return "\n".join(satirlar)


def rapor_calistir(simdi, bugun):
    kayitlar = kayit_oku(str(bugun))
    if not kayitlar:
        print("Bugün bildirilmiş sinyal kaydı yok, rapor gönderilmedi.")
        return
    semboller = sorted({k["sembol"] for k in kayitlar})
    with ThreadPoolExecutor(max_workers=4) as havuz:
        veriler = dict(zip(semboller, havuz.map(veri_cek_15dk, semboller)))
    sonuclar = []
    for k in sorted(kayitlar, key=lambda x: x["saat"]):
        df = veriler.get(k["sembol"])
        d = degerlendir(k, df, bugun) if df is not None else dict(sonuc="bos")
        sonuclar.append((k, d))

    gecmise = []
    for k, d in sonuclar:
        kayit = dict(tarih=str(bugun), tur=k["tur"], sembol=k["sembol"], saat=k["saat"],
                     sonuc=d["sonuc"], teyit=k.get("teyit"))
        for alan in ("ust_cizgi", "puan", "rg_ok", "tmax"):
            if alan in k:
                kayit[alan] = k[alan]
        if "kirdi" in d:
            kayit["kirdi"] = d["kirdi"]
        for alan in ("yuk", "dus", "kap"):
            if alan in d:
                kayit[alan] = round(d[alan], 2)
        gecmise.append(kayit)
    gecmis_yaz(str(bugun), gecmise)

    adlar = (("g", "🚀 Düşen trend kırılımı"), ("h", "🔸 Hacimsiz kırılım"), ("e", "⏳ Kırılım adayı"),
             ("a", "🔻 Aşırı satış tepkisi"), ("x", "🚫 Elenen kırılımlar (teyit yetmedi, bildirilmedi)"))
    bolumler = []
    for tur, ad in adlar:
        bloklar = [(k["sembol"], rapor_blogu(k, d)) for k, d in sonuclar if k["tur"] == tur]
        if bloklar:
            bolumler.append((tur, ad, bloklar))
    degerli = [d for k, d in sonuclar if d["sonuc"] != "bos" and k["tur"] != "x"]
    say = lambda ad: sum(1 for d in degerli if d["sonuc"] == ad)
    ozet = (f"Bugün {sum(1 for k, _ in sonuclar if k['tur'] != 'x')} sinyal "
            f"(+ {sum(1 for k, _ in sonuclar if k['tur'] == 'x')} elenen kırılım): 🎯 {say('hedef')} hedefe · 🛑 {say('stop')} stopa · "
            f"❓ {say('belirsiz')} belirsiz · ⏳ {say('yok')} hiçbirine değmedi")
    alt = (genel_toplam(gecmis_oku()) + "\n\n"
           "ℹ️ Değerlendirme, sinyalden sonraki ilk 15 dk sınırından günün sonuna kadarki 15 dakikalık "
           "mumlarla yapıldı. Sinyal fiyatı gecikmeli veriden geldiği için gerçek girişin biraz farklı "
           "olabilir. Aynı mumda hem stop hem hedef varsa sıra bilinemez (❓).")
    ok = True
    for metin, _ in mesaj_parcalari(bolumler, simdi, ek_satir=ozet, alt=alt,
                                    baslik=f"📊 Gün sonu raporu · {bugun:%d.%m.%Y}"):
        ok = telegram_gonder(metin) and ok
    if not ok:
        print("Rapor mesajı gönderilemedi.")
        sys.exit(1)


def telegram_gonder(metin):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    veri = urllib.parse.urlencode(
        {"chat_id": CHAT_ID, "text": metin, "disable_web_page_preview": "true"}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=veri), timeout=30) as yanit:
            return bool(json.load(yanit).get("ok"))
    except urllib.error.HTTPError as e:
        try:
            aciklama = json.load(e).get("description", "")
        except Exception:
            aciklama = ""
        print(f"Telegram hatası: HTTP {e.code} {aciklama}")
    except Exception as e:
        print(f"Telegram hatası: {type(e).__name__}")
    return False


def durum_oku(bugun_str):
    """(guclu, hacimsiz, erken, asiri, elenen) bildirilen / kaydedilen hisse kümelerini döner."""
    try:
        d = json.loads(DURUM_DOSYASI.read_text(encoding="utf-8"))
        if d.get("tarih") == bugun_str:
            return (set(d.get("hisseler", [])), set(d.get("hacimsizler", [])),
                    set(d.get("erkenler", [])), set(d.get("asirilar", [])), set(d.get("elenenler", [])))
    except Exception:
        pass
    return set(), set(), set(), set(), set()


def durum_yaz(bugun_str, guclu, hacimsiz, erken, asiri, elenen=()):
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    DURUM_DOSYASI.write_text(
        json.dumps({"tarih": bugun_str, "hisseler": sorted(guclu), "hacimsizler": sorted(hacimsiz),
                    "erkenler": sorted(erken), "asirilar": sorted(asiri), "elenenler": sorted(elenen)},
                   ensure_ascii=False),
        encoding="utf-8")


# ------------------------------------------------------- retest (kırılım sonrası çizgiye geri dönüş)
def retest_oku(bugun_str):
    """(izle, bitti): bugün bildirilen kırılımların çizgi bilgileri ve izlemesi bitenler."""
    try:
        d = json.loads(RETEST_DOSYASI.read_text(encoding="utf-8"))
        if d.get("tarih") == bugun_str:
            return d.get("izle", {}), set(d.get("bitti", []))
    except Exception:
        pass
    return {}, set()


def retest_yaz(bugun_str, izle, bitti):
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    RETEST_DOSYASI.write_text(
        json.dumps({"tarih": bugun_str, "izle": izle, "bitti": sorted(bitti)}, ensure_ascii=False),
        encoding="utf-8")


def retest_kaydet(bugun_str, liste):
    """Bildirimi giden 🚀/🔸 kırılımların çizgisini (iki tepe noktası) ve bildirim mumunu kaydeder."""
    izle, bitti = retest_oku(bugun_str)
    for r in liste:
        a, b, m = r.get("nokta_i"), r.get("nokta_j"), r.get("son_mum")
        if not a or not b or m is None:
            continue
        izle.setdefault(r["sembol"], dict(i=[a[0].isoformat(), float(a[1])],
                                          j=[b[0].isoformat(), float(b[1])], mum=m.isoformat()))
    retest_yaz(bugun_str, izle, bitti)


def retest_kontrol(df, k):
    """Kaydedilen çizgi bugünün saatlik mumlarına uzatılır. Bildirim mumundan SONRAKİ mumlarda:
    kapanış çizginin RETEST_STOP altına indiyse "bozuldu"; son iki mumdan birinin dibi çizgiye
    RETEST_YAKIN içinde inmiş ve kapanış çizgi üstündeyse retest sözlüğü; yoksa None."""
    try:
        ts_i, ts_j, ts_m = pd.Timestamp(k["i"][0]), pd.Timestamp(k["j"][0]), pd.Timestamp(k["mum"])
        pi, pj = float(k["i"][1]), float(k["j"][1])
        pos_i, pos_j = df.index.get_loc(ts_i), df.index.get_loc(ts_j)
    except (KeyError, ValueError, TypeError):
        return None                      # çizginin tepe mumları veride yoksa güvenli tarafta kal
    if pos_j <= pos_i:
        return None
    egim = (pj - pi) / (pos_j - pos_i)
    cl = df["Close"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    sonraki = [p for p in range(len(df)) if df.index[p] > ts_m]
    retest = None
    for p in sonraki:
        cizgi = pi + egim * (p - pos_i)
        if cl[p] < cizgi * (1 - RETEST_STOP / 100.0):
            return "bozuldu"
        if p >= len(df) - 2 and lo[p] <= cizgi * (1 + RETEST_YAKIN / 100.0) and cl[p] > cizgi:
            retest = dict(cizgi=float(cizgi), fiyat=float(cl[p]), dip=float(lo[p]))
    return retest


def retest_isle(simdi, bugun):
    """Taramadan sonra: bozulanları izlemeden çıkarır, retest olanlara tek mesajla haber verir."""
    if not RETEST_IZLE:
        return
    izle, bitti = retest_oku(str(bugun))
    bloklar, gidecek = [], []
    for s in sorted(RETEST_IZLE):
        sonuc = RETEST_SONUC.get(s)
        if sonuc == "bozuldu":
            bitti.add(s)
        elif isinstance(sonuc, dict):
            bloklar.append(f"{s}  {sayi(sonuc['fiyat'])} TL · çizgi {sayi(sonuc['cizgi'])} · dip {sayi(sonuc['dip'])}\n"
                           f"https://www.tradingview.com/chart/?symbol=BIST%3A{s}\n")
            gidecek.append(s)
    print("Retest:", gidecek, "| Bozulan:", sorted(s for s in RETEST_IZLE if RETEST_SONUC.get(s) == "bozuldu"))
    if bloklar:
        metin = f"🔁 Retest ({simdi:%H:%M}) · kırılım sonrası çizgiye dönüp üstünde tutundu\n\n" + "\n".join(bloklar)
        if telegram_gonder(metin):
            bitti |= set(gidecek)
        else:
            print("Retest mesajı gönderilemedi; sonraki taramada tekrar denenecek.")
    retest_yaz(str(bugun), izle, bitti)


def seans_icinde(simdi):
    dakika = simdi.hour * 60 + simdi.minute
    return simdi.weekday() < 5 and 10 * 60 <= dakika <= 18 * 60 + 10


def main(mod=None):
    mod = (mod or os.environ.get("MOD") or "bildirim").strip()
    simdi = dt.datetime.now(TR)
    bugun = simdi.date()
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)

    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil (GitHub Secrets).")
        sys.exit(1)

    if mod == "baglanti":
        ok = telegram_gonder("✅ BIST tarayıcı botu bağlandı. Bildirimler bu sohbete gelecek.")
        print("Bağlantı mesajı gönderildi." if ok else "Bağlantı mesajı GÖNDERİLEMEDİ.")
        sys.exit(0 if ok else 1)

    if mod == "rapor":
        rapor_calistir(simdi, bugun)
        return

    if mod == "bildirim" and not seans_icinde(simdi):
        print(f"Seans dışı ({simdi:%d.%m.%Y %H:%M}), tarama yapılmadı.")
        return

    RETEST_IZLE.clear()
    RETEST_SONUC.clear()
    if mod == "bildirim" and RETEST_GOSTER:
        izle_, bitti_ = retest_oku(str(bugun))
        RETEST_IZLE.update({s: k for s, k in izle_.items() if s not in bitti_})
    semboller = hisse_listesi()
    print(f"{len(semboller)} hisse taranacak ({simdi:%d.%m.%Y %H:%M}, mod: {mod}).")
    sayac = {"veri_yok": 0, "kisa": 0, "bugun_yok": 0, "yok": 0, "bulundu": 0}
    trend_liste, asiri_liste = [], []
    with ThreadPoolExecutor(max_workers=6) as havuz:
        for durum, tr_r, as_r in havuz.map(lambda s: analiz(s, bugun), semboller):
            sayac[durum] += 1
            if tr_r:
                trend_liste.append(tr_r)
            if as_r:
                asiri_liste.append(as_r)
    print("Sayım:", sayac)

    if sayac["veri_yok"] > 0.5 * len(semboller):
        print("Hisselerin yarısından fazlasının verisi alınamadı (Yahoo erişim sorunu olabilir).")
        sys.exit(1)

    if mod == "bildirim" and RETEST_GOSTER:
        retest_isle(simdi, bugun)
    trend_liste.sort(key=siralama_anahtari)
    elenenler = [r for r in trend_liste if r["kademe"] == "elenen"]     # sadece rapor icin kaydedilir
    trend_liste = [r for r in trend_liste if r["kademe"] != "elenen"]
    # aşırı satış sırası: önce puan (çoktan aza), sonra daha çok düşen
    asiri_liste.sort(key=lambda r: (-r["puan"], -r["teyit"], r["gunluk"] if r["gunluk"] is not None else 0.0, r["sembol"]))
    ozet = (f"Taranan: {len(semboller)} · veri alınamayan: {sayac['veri_yok']} · "
            f"bugün işlem görmeyen: {sayac['bugun_yok']}")

    gucluler = [r for r in trend_liste if r["kademe"] == "guclu"]
    hacimsizler = [r for r in trend_liste if r["kademe"] == "hacimsiz"]
    erkenler = [r for r in trend_liste if r["kademe"] == "erken"] if ERKEN_GOSTER else []
    erkenler.sort(key=lambda r: (abs(r["kirilim"]), -r["temas"], r["sembol"]))   # çizgiye en yakın önce
    BASLIK_G, BASLIK_E, BASLIK_A = "🚀 Düşen trend kırılımı", "⏳ Kırılım adayı (çizgiye yakın, henüz kırılmadı)", "🔻 Aşırı satış tepkisi"
    BASLIK_H = "🔸 Hacimsiz kırılım (çizgi kırıldı, hacim henüz teyit etmedi)"

    def bolumleri_kur(g, h, e, a):
        sonuc = []
        if g:
            sonuc.append(bolum("g", BASLIK_G, g, blok_trend))
        if h:
            sonuc.append(bolum("h", BASLIK_H, h, blok_trend))
        if e:
            sonuc.append(bolum("e", BASLIK_E, e, blok_aday))
        if a:
            sonuc.append(bolum("a", BASLIK_A, a, blok_asiri))
        return sonuc

    if mod == "deneme":
        telegram_gonder(likidite_ozeti(trend_liste))
        if not (trend_liste or asiri_liste):
            telegram_gonder(f"🧪 DENEME ({simdi:%H:%M}): şu an şartları sağlayan hisse yok.\n{ozet}")
            return
        gonder(bolumleri_kur(gucluler, hacimsizler, erkenler, asiri_liste), simdi, "🧪 DENEME · ", ozet)
        return

    g_gitti, h_gitti, e_gitti, a_gitti, x_gitti = durum_oku(str(bugun))
    yeni_g = [r for r in gucluler if r["sembol"] not in g_gitti]
    # hacimsiz kirilim: bugun ne 🚀 ne 🔸 olarak bildirilmis olmali (hacim gelince 🚀 ayrica gelir)
    yeni_h = [r for r in hacimsizler if r["sembol"] not in h_gitti and r["sembol"] not in g_gitti]
    # erken sinyal: ne erken ne guclu ne hacimsiz olarak bugun bildirilmis olmali
    yeni_e = [r for r in erkenler if r["sembol"] not in e_gitti and r["sembol"] not in g_gitti
              and r["sembol"] not in h_gitti]
    yeni_a = [r for r in asiri_liste if r["sembol"] not in a_gitti]
    print("Güçlü:", [r["sembol"] for r in gucluler], "| Yeni:", [r["sembol"] for r in yeni_g])
    print("Hacimsiz:", [r["sembol"] for r in hacimsizler], "| Yeni:", [r["sembol"] for r in yeni_h])
    # elenen kirilim: bugun ilk kez elendiyse ve bildirilmemisse sessizce kaydedilir (gun sonu raporu icin)
    yeni_x = [r for r in elenenler if r["sembol"] not in x_gitti and r["sembol"] not in g_gitti
              and r["sembol"] not in h_gitti]
    print("Elenen kırılım:", [r["sembol"] for r in elenenler], "| Yeni kayıt:", [r["sembol"] for r in yeni_x])
    print("Erken:", [r["sembol"] for r in erkenler], "| Yeni:", [r["sembol"] for r in yeni_e])
    print("Aşırı satış:", [r["sembol"] for r in asiri_liste], "| Yeni:", [r["sembol"] for r in yeni_a])
    if yeni_x and not (yeni_g or yeni_h or yeni_e or yeni_a):
        x_gitti |= {r["sembol"] for r in yeni_x}
        durum_yaz(str(bugun), g_gitti, h_gitti, e_gitti, a_gitti, x_gitti)
        kayit_ekle(str(bugun), [kayit_yap("x", r, simdi) for r in yeni_x])
    if not (yeni_g or yeni_h or yeni_e or yeni_a):
        return

    gitti, hata = gonder(bolumleri_kur(yeni_g, yeni_h, yeni_e, yeni_a), simdi)
    g_gitti |= {s for t, s in gitti if t == "g"}
    h_gitti |= {s for t, s in gitti if t == "h"}
    e_gitti |= {s for t, s in gitti if t == "e"}
    a_gitti |= {s for t, s in gitti if t == "a"}
    x_gitti |= {r["sembol"] for r in yeni_x}
    durum_yaz(str(bugun), g_gitti, h_gitti, e_gitti, a_gitti, x_gitti)
    if RETEST_GOSTER:
        retest_kaydet(str(bugun), [r for tur, liste in (("g", yeni_g), ("h", yeni_h))
                                   for r in liste if (tur, r["sembol"]) in gitti])
    yeni_kayitlar = [kayit_yap(tur, r, simdi)
                     for tur, liste in (("g", yeni_g), ("h", yeni_h), ("e", yeni_e), ("a", yeni_a))
                     for r in liste if (tur, r["sembol"]) in gitti]
    yeni_kayitlar += [kayit_yap("x", r, simdi) for r in yeni_x]
    if yeni_kayitlar:
        kayit_ekle(str(bugun), yeni_kayitlar)
    if hata:
        print("Bazı mesajlar gönderilemedi; o hisseler kayda eklenmedi, sonraki taramada tekrar denenecek.")
        sys.exit(1)


def dongu_calistir():
    """MOD=dongu: tek bir GitHub çalışması içinde, DONGU_BITIS (TR saati, "SS:DD") olana kadar
    her DONGU_ARA dakikada bir normal taramayı çalıştırır. GitHub zamanlayıcısının geç kalması
    ya da çalışma atlaması sorununu aşmak için; durum dosyaları aynı çalışma içinde diskte durur."""
    if not TOKEN or not CHAT_ID:
        print("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID tanımlı değil (GitHub Secrets).")
        sys.exit(1)
    ara = max(1, int(os.environ.get("DONGU_ARA") or 5))
    sa, dk = (int(x) for x in (os.environ.get("DONGU_BITIS") or "18:10").split(":"))
    print(f"Döngü başladı: her {ara} dakikada bir tarama, bitiş {sa:02d}:{dk:02d} (TR).")
    while True:
        simdi = dt.datetime.now(TR)
        bitis = simdi.replace(hour=sa, minute=dk, second=0, microsecond=0)
        if simdi > bitis:
            break
        if seans_icinde(simdi):
            try:
                main("bildirim")
            except SystemExit as e:        # tek taramanın hatası döngüyü durdurmasın
                print(f"Tarama hata koduyla bitti ({e.code}); döngü devam ediyor.")
            except Exception as e:
                print(f"Tarama beklenmedik hata verdi ({type(e).__name__}: {e}); döngü devam ediyor.")
        simdi = dt.datetime.now(TR)
        sonraki = simdi.replace(second=0, microsecond=0) + dt.timedelta(minutes=ara - simdi.minute % ara)
        if sonraki > bitis:
            break
        time.sleep(max(0.0, (sonraki - dt.datetime.now(TR)).total_seconds()))
    print("Döngü bitti.")


if __name__ == "__main__":
    if (os.environ.get("MOD") or "").strip() == "dongu":
        dongu_calistir()
    else:
        main()

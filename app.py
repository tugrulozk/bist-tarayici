import datetime as dt
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Listeyi uygulama icinden de duzenleyebilirsin ("Hisse listesi" bolumu).
VARSAYILAN_HISSELER = """ASELS TUPRS GARAN KCHOL ENKAI BIMAS THYAO AKBNK ISCTR VAKBN HALKB YKBNK FROTO EREGL CCOLA
ASTOR TCELL TTKOM SAHOL ISDMR TRALT GUBRF ENJSA TOASO ENERY AHGAZ SISE TURSG AEFES OYAKC SASA BRSAN TAVHL
TRGYO MGROS AKSEN TKFEN MPARK AGHOL PGSUS EKGYO RGYAS AYGAZ TABGD ARCLK TRMET RYSAS PETKM CVKMD ANSGR
KRDMD GLRMK ISMEN DOHOL ECILC ANHYT ALARK BRYAT AKSA OTKAR TTRAK CIMSA AKCNS SOKM ULKER GLYHO MAVI DOAS
TSKB GRSEL AKFYE TRENJ ECZYT CWENE TCKRC ALBRK KCAER PAHOL HEKTS ENTRA EGEEN FENER OBAMS ALTNY LMKDC
BTCIM KORDS GWIND SNGYO CANTE EGGUB ZOREN TUKAS ODAS BERA KARSN BINHO EUREN VESTL KATMR"""

ARALIK_AYARLARI = {
    # aralik: (veri periyodu, mum suresi dakika)
    "1h": ("180d", 60),
    "30m": ("60d", 30),
    "15m": ("60d", 15),
}

KRITERLER = ["MACD", "MFI", "Wave Trend", "Stoch RSI", "Nadaraya"]
IDEAL_ARALIK = (-8.0, -4.0)  # gunluk degisim icin "ideal" bolge (%)
TABAN_ESIGI = -9.5  # BIST'te alt limit %10; bunun altindakiler tabana oturmus sayilir
NW_PENCERE = 50    # Nadaraya-Watson cekirdek uzunlugu (mum)
MAE_PENCERE = 200  # bant genisligi icin ortalama hata penceresi (mum)


SEMBOL_KALIBI = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{3,6}$")  # en az bir harf icermeli
YOK_SAY = {"USD", "TRY", "EUR", "TTM", "EPS", "FKO", "POINT", "SEMBOL", "FIYAT", "HACIM", "PYS",
           "FINANS", "ENERJI", "SAT"}


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


def varsayilan_liste():
    """Depodaki hisseler.txt dosyasi varsa onu, yoksa koddaki listeyi kullanir."""
    try:
        icerik = (Path(__file__).parent / "hisseler.txt").read_text(encoding="utf-8")
        kodlar = liste_coz(icerik)
        if kodlar:
            return " ".join(kodlar)
    except Exception:
        pass
    return VARSAYILAN_HISSELER


# ----------------------------------------------------------------- gostergeler
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


def nadaraya_watson(kapanis, h=8.0, carpan=3.0):
    """Gecmisi degistirmeyen (sadece onceki mumlara bakan) versiyon."""
    x = kapanis.to_numpy(dtype=float)
    w = np.exp(-(np.arange(NW_PENCERE) ** 2) / (2 * h * h))
    nw = np.convolve(x, w, mode="full")[: len(x)] / w.sum()
    nw[: NW_PENCERE - 1] = np.nan
    nw = pd.Series(nw, index=kapanis.index)
    mae = (kapanis - nw).abs().rolling(MAE_PENCERE).mean() * carpan
    return nw, nw - mae, nw + mae


def yukari_kesti(a, b, son_n):
    k = (a.shift(1) <= b.shift(1)) & (a > b)
    return bool(k.iloc[-son_n:].any())


# ------------------------------------------------------------------- kriterler
def kriter_macd(df, p):
    macd = dema(df["Close"], 12) - dema(df["Close"], 26)
    sinyal = ema(macd, 9)
    hist = macd - sinyal
    altinda = bool(macd.iloc[-1] < p["macd_seviye"])
    kesti = yukari_kesti(macd, sinyal, p["son_n"])
    h1, h0 = hist.iloc[-1], hist.iloc[-2]
    yakin = bool(h1 < 0 and h1 > h0 and abs(h1) <= p["macd_yakinlik"] * hist.abs().iloc[-50:].max())
    durum = 2 if (altinda and kesti) else 1 if (altinda and yakin) else 0
    return durum, f"{macd.iloc[-1]:.2f}"


def kriter_mfi(df, p):
    v = mfi_hesapla(df).iloc[-1]
    durum = 2 if v <= p["mfi_os"] else 1 if v <= p["mfi_os"] + 8 else 0
    return durum, f"{v:.0f}"


def kriter_wt(df, p):
    wt1, wt2 = wave_trend(df)
    kesti = yukari_kesti(wt1, wt2, p["son_n"])
    altta = bool(wt1.iloc[-1] < 0)
    yaklasiyor = bool(wt1.iloc[-1] < wt2.iloc[-1] and wt1.iloc[-1] > wt1.iloc[-2]
                      and (wt2.iloc[-1] - wt1.iloc[-1]) <= p["wt_yakin"] and altta)
    durum = 2 if (kesti and altta) else 1 if yaklasiyor else 0
    return durum, f"{wt1.iloc[-1]:.0f}"


def kriter_stoch(df, p):
    k, d = stoch_rsi(df["Close"])
    kesti = yukari_kesti(k, d, p["son_n"])
    asiri = bool(k.iloc[-(p["son_n"] + 1):].min() <= p["stoch_os"])
    yakin_asiri = bool(k.iloc[-1] <= p["stoch_os"] + 10)
    durum = 2 if (asiri and kesti) else 1 if (asiri or (yakin_asiri and kesti)) else 0
    return durum, f"{k.iloc[-1]:.0f}"


def kriter_nw(df, p):
    nw, alt, ust = nadaraya_watson(df["Close"], p["nw_h"], p["nw_carpan"])
    n = p["son_n"] + 1
    degdi = bool((df["Low"].iloc[-n:] <= alt.iloc[-n:]).any())
    donus = bool(df["Close"].iloc[-1] > df["Close"].iloc[-2])
    kapanis = df["Close"].iloc[-1]
    yakin = bool((kapanis - alt.iloc[-1]) <= 0.2 * (nw.iloc[-1] - alt.iloc[-1]))
    durum = 2 if (degdi and donus) else 1 if (degdi or yakin) else 0
    konum = (kapanis - alt.iloc[-1]) / (ust.iloc[-1] - alt.iloc[-1]) * 100
    return durum, f"%{konum:.0f}"


KRITER_FONKSIYONLARI = [kriter_macd, kriter_mfi, kriter_wt, kriter_stoch, kriter_nw]
ISARET = {2: "✅", 1: "🟡", 0: "—"}


# ------------------------------------------------------------------------ veri
@st.cache_data(ttl=600, show_spinner=False)
def veri_cek(sembol, periyot, aralik):
    try:
        df = yf.Ticker(sembol + ".IS").history(period=periyot, interval=aralik, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.empty:
        return None
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Istanbul")
    return df


def gunluk_degisim(df):
    gunler = np.array(df.index.date)
    onceki = df[gunler != gunler[-1]]
    if onceki.empty:
        return np.nan
    return (df["Close"].iloc[-1] / onceki["Close"].iloc[-1] - 1) * 100


def bugunku_mumlar(df):
    gunler = np.array(df.index.date)
    return df[gunler == gunler[-1]]


def acilistan_dusus(df):
    """Bugunku seansin acilis fiyatindan son kapanisa degisim (%)."""
    return (df["Close"].iloc[-1] / bugunku_mumlar(df)["Open"].iloc[0] - 1) * 100


def dusus_maskesi(tablo, min_dusus, acilis_say, taban_haric):
    maske = tablo["Günlük %"] <= -min_dusus
    if acilis_say:
        maske = maske | (tablo["Açılıştan %"] <= -min_dusus)
    if taban_haric:
        maske = maske & ~(tablo["Günlük %"] <= TABAN_ESIGI)
    return maske


def analiz_et(df, sembol, mum_dakika, p, simdi=None):
    df = df.copy()
    if p["olusan_mumu_cikar"] and len(df) > 0:
        if simdi is None:
            simdi = pd.Timestamp.now(tz="Europe/Istanbul")
        bas = df.index[-1]
        bitis = bas + pd.Timedelta(minutes=mum_dakika)
        # Seans 18:00'de biter; ondan sonra son mum tamamlanmis sayilir.
        if bitis > simdi and bas.date() == simdi.date() and simdi.hour < 18:
            df = df.iloc[:-1]

    if len(df) < NW_PENCERE + MAE_PENCERE + 20:
        return None

    durumlar, metinler = [], []
    for f in KRITER_FONKSIYONLARI:
        d, t = f(df, p)
        durumlar.append(d)
        metinler.append(t)

    tam = sum(1 for d in durumlar if d == 2)
    yakin = sum(1 for d in durumlar if d == 1)
    puan = tam + (yakin if p["yakin_say"] else 0)

    degisim = gunluk_degisim(df)
    acilis = acilistan_dusus(df)
    bugun = bugunku_mumlar(df)
    tutar = float((bugun["Close"] * bugun["Volume"]).sum() / 1e6)
    if not np.isnan(degisim) and degisim <= TABAN_ESIGI:
        not_ = "⛔ taban"
    elif IDEAL_ARALIK[0] <= degisim <= IDEAL_ARALIK[1]:
        not_ = "⭐"
    else:
        not_ = ""
    satir = {
        "Hisse": sembol,
        "Fiyat": round(float(df["Close"].iloc[-1]), 2),
        "Günlük %": round(float(degisim), 2) if not np.isnan(degisim) else np.nan,
        "Açılıştan %": round(float(acilis), 2),
        "Tutar (mn TL)": round(tutar, 1),
        "Puan": puan,
        "Not": not_,
    }
    for ad, d, t in zip(KRITERLER, durumlar, metinler):
        satir[ad] = f"{ISARET[d]} {t}"
    satir["Son mum (TR)"] = df.index[-1].strftime("%d.%m %H:%M")
    return satir


# --------------------------------------------------------------------- arayuz
def main():
    st.set_page_config(page_title="BIST Saatlik Tarayıcı", page_icon="📈", layout="wide")
    st.title("📈 BIST Saatlik Tarayıcı")
    st.caption(
        "Veriler ücretsiz kaynaktan gelir ve gecikmelidir (yaklaşık 15 dk). "
        "Bu araç sadece aday listesi üretir, yatırım tavsiyesi değildir. "
        "İşleme girmeden önce grafikte teyit et ve güncel fiyatı aracı kurum ekranından kontrol et."
    )

    with st.sidebar:
        st.header("Ayarlar")
        aralik = st.selectbox("Zaman dilimi", list(ARALIK_AYARLARI.keys()), index=0)
        en_az = st.slider("En az kaç kriter sağlansın?", 1, 5, 2)
        min_dusus = st.slider(
            "Gün içinde en az % kaç düşmüş olsun?", 0.0, 9.0, 4.0, 0.5,
            help="Hisse, önceki kapanışa göre (açılıştaki boşluk dahil) ya da bugünkü açılış "
                 "fiyatına göre bu kadar düşmüş olmalı. ⭐ = -%4 ile -%8 arası. "
                 "0 yaparsan filtre fiilen kapanır.",
        )
        taban_haric = st.checkbox(
            "Tabana oturmuşları hariç tut", value=True,
            help="Günlük değişimi yaklaşık -%9,5 ve altında olan hisseler tabana oturmuş sayılır "
                 "(BIST'te alt limit %10).",
        )
        with st.expander("Hisse listesi"):
            metin = st.text_area("Kodları yapıştır (boşluk, virgül veya alt alta)", varsayilan_liste(), height=180)
        with st.expander("Gelişmiş ayarlar"):
            acilis_say = st.checkbox("Açılış fiyatına göre düşüşü de say", value=True,
                                     help="Açıksa hisse sadece önceki kapanışa göre değil, bugünkü "
                                          "açılış fiyatına göre düşüşe de bakılır.")
            min_tutar = st.slider("Bugünkü minimum işlem tutarı (milyon TL)", 0, 100, 0,
                                  help="Düşük hacimli hisseleri elemek için. 0 = filtre kapalı.")
            yakin_say = st.checkbox("🟡 'yakın' durumları da say", value=True)
            olusan_mumu_cikar = st.checkbox("Kapanmamış mumu hariç tut", value=True)
            son_n = st.slider("Kesişim/değme son kaç mumda aransın?", 1, 5, 4)
            macd_seviye = st.number_input("MACD DEMA seviyesi", value=0.0, step=0.1)
            macd_yakinlik = st.slider("MACD kesişime yakınlık", 0.1, 0.8, 0.3, 0.05)
            mfi_os = st.slider("MFI aşırı satış", 10, 40, 20)
            stoch_os = st.slider("Stoch RSI aşırı satış", 10, 40, 20)
            wt_yakin = st.slider("Wave Trend kesişime yakınlık", 1.0, 10.0, 4.0, 0.5)
            nw_h = st.slider("Nadaraya bant genişliği (h)", 3.0, 15.0, 8.0, 0.5)
            nw_carpan = st.slider("Nadaraya bant çarpanı", 1.0, 5.0, 3.0, 0.5)

    p = dict(yakin_say=yakin_say, olusan_mumu_cikar=olusan_mumu_cikar, son_n=int(son_n),
             macd_seviye=float(macd_seviye), macd_yakinlik=float(macd_yakinlik),
             mfi_os=float(mfi_os), stoch_os=float(stoch_os), wt_yakin=float(wt_yakin),
             nw_h=float(nw_h), nw_carpan=float(nw_carpan))

    semboller = liste_coz(metin)
    st.write(f"**{len(semboller)}** hisse taranacak.")
    if not st.button("🔍 Tara", type="primary"):
        st.info("Soldaki ayarları kontrol edip **Tara** düğmesine bas.")
        return

    periyot, mum_dakika = ARALIK_AYARLARI[aralik]
    ilerleme = st.progress(0.0, text="Veriler çekiliyor...")
    veriler = {}

    def getir(s):
        return s, veri_cek(s, periyot, aralik)

    with ThreadPoolExecutor(max_workers=8) as havuz:
        for i, (s, df) in enumerate(havuz.map(getir, semboller), start=1):
            veriler[s] = df
            ilerleme.progress(i / len(semboller), text=f"Veriler çekiliyor... {i}/{len(semboller)}")
    ilerleme.empty()

    sonuclar, verisiz = [], []
    for s in semboller:
        df = veriler.get(s)
        r = analiz_et(df, s, mum_dakika, p) if df is not None else None
        if r is None:
            verisiz.append(s)
        else:
            sonuclar.append(r)

    st.caption(f"Tarama zamanı: {dt.datetime.now().strftime('%d.%m.%Y %H:%M')} (sunucu saati)")
    if not sonuclar:
        st.error("Hiçbir hissenin verisi alınamadı. Birkaç dakika sonra tekrar dene.")
        return

    tablo = pd.DataFrame(sonuclar)
    sinyal = tablo[(tablo["Puan"] >= en_az) & dusus_maskesi(tablo, min_dusus, acilis_say, taban_haric)
                  & (tablo["Tutar (mn TL)"] >= min_tutar)]
    sinyal = sinyal.sort_values(["Puan", "Günlük %"], ascending=[False, True])

    st.subheader(f"Aday hisseler ({len(sinyal)})")
    st.caption("✅ = kriter sağlandı   🟡 = kriter yakın   — = sağlanmadı. "
               "Yanındaki sayı göstergenin değeridir (Nadaraya'da %0 alt bant, %100 üst bant).")
    if sinyal.empty:
        st.warning("Şu an şartları sağlayan hisse yok. 'En az kriter' sayısını ya da düşüş eşiğini gevşetmeyi dene.")
    else:
        st.dataframe(sinyal, hide_index=True)
        st.download_button("CSV olarak indir", sinyal.to_csv(index=False).encode("utf-8"),
                           "tarama_sonucu.csv", "text/csv")

    with st.expander(f"Taranan tüm hisseler ({len(tablo)})"):
        st.dataframe(tablo.sort_values("Puan", ascending=False), hide_index=True)
    if verisiz:
        st.caption("Verisi alınamayan veya yetersiz olan hisseler: " + ", ".join(verisiz))


main()

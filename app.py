import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Listeyi uygulama icinden de duzenleyebilirsin (kenar cubugundaki kutu).
VARSAYILAN_HISSELER = """AKBNK ALARK ARCLK ASELS ASTOR BIMAS BRSAN CCOLA CIMSA DOHOL
EGEEN EKGYO ENJSA ENKAI EREGL FROTO GARAN GUBRF HALKB HEKTS ISCTR KCHOL KONTR
KRDMD MAVI MGROS ODAS OYAKC PETKM PGSUS SAHOL SASA SISE SOKM TAVHL TCELL THYAO
TKFEN TOASO TSKB TTKOM TUPRS ULKER VAKBN VESTL YKBNK"""

ARALIK_AYARLARI = {
    # aralik: (veri periyodu, mum suresi dakika)
    "1h": ("60d", 60),
    "30m": ("30d", 30),
    "15m": ("30d", 15),
}

MOD_KESISIM = "RSI eşiği yukarı kesti (dönüş sinyali)"
MOD_ALTINDA = "RSI eşiğin altında (aşırı satış)"


def rsi_hesapla(kapanis: pd.Series, periyot: int = 14) -> pd.Series:
    fark = kapanis.diff()
    kazanc = fark.clip(lower=0)
    kayip = -fark.clip(upper=0)
    ort_kazanc = kazanc.ewm(alpha=1 / periyot, adjust=False, min_periods=periyot).mean()
    ort_kayip = kayip.ewm(alpha=1 / periyot, adjust=False, min_periods=periyot).mean()
    rs = ort_kazanc / ort_kayip.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(ort_kayip != 0, 100.0)
    return rsi.where(ort_kayip.notna())


@st.cache_data(ttl=600, show_spinner=False)
def veri_cek(sembol: str, periyot: str, aralik: str):
    try:
        df = yf.Ticker(sembol + ".IS").history(
            period=periyot, interval=aralik, auto_adjust=True
        )
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


def analiz_et(
    df: pd.DataFrame,
    sembol: str,
    mum_dakika: int,
    rsi_periyot: int,
    esik: float,
    mod: str,
    geriye_bakis: int,
    ema_filtre: bool,
    ema_periyot: int,
    hacim_periyot: int,
    min_goreceli_hacim: float,
    olusan_mumu_cikar: bool,
    simdi=None,
):
    df = df.copy()

    if olusan_mumu_cikar and len(df) > 0:
        if simdi is None:
            simdi = pd.Timestamp.now(tz="Europe/Istanbul")
        son_baslangic = df.index[-1]
        bitis = son_baslangic + pd.Timedelta(minutes=mum_dakika)
        # Seans 18:00'de biter; ondan sonra son mum tamamlanmis sayilir.
        if bitis > simdi and son_baslangic.date() == simdi.date() and simdi.hour < 18:
            df = df.iloc[:-1]

    gerekli = max(rsi_periyot, ema_periyot, hacim_periyot) + geriye_bakis + 5
    if len(df) < gerekli:
        return None

    df["rsi"] = rsi_hesapla(df["Close"], rsi_periyot)
    df["ema"] = df["Close"].ewm(span=ema_periyot, adjust=False).mean()
    ort_hacim = df["Volume"].rolling(hacim_periyot).mean().shift(1)
    df["gh"] = df["Volume"] / ort_hacim.replace(0, np.nan)

    son = df.iloc[-1]
    if pd.isna(son["rsi"]):
        return None

    if mod == MOD_KESISIM:
        kesis = (df["rsi"].shift(1) < esik) & (df["rsi"] >= esik)
        rsi_sinyal = bool(kesis.iloc[-geriye_bakis:].any())
    else:
        rsi_sinyal = bool(son["rsi"] < esik)

    ema_tamam = (not ema_filtre) or bool(son["Close"] > son["ema"])

    gh_son = df["gh"].iloc[-geriye_bakis:].max()
    gh_son = 0.0 if pd.isna(gh_son) else float(gh_son)
    hacim_tamam = gh_son >= min_goreceli_hacim

    return {
        "Hisse": sembol,
        "Fiyat": round(float(son["Close"]), 2),
        "RSI": round(float(son["rsi"]), 1),
        "EMA'ya uzaklık %": round(float((son["Close"] / son["ema"] - 1) * 100), 2),
        "Göreceli hacim": round(gh_son, 2),
        "Son mum (TR saati)": df.index[-1].strftime("%d.%m %H:%M"),
        "_gecti": rsi_sinyal and ema_tamam and hacim_tamam,
    }


def main():
    st.set_page_config(page_title="BIST Saatlik Tarayıcı", page_icon="📈", layout="wide")
    st.title("📈 BIST Saatlik Tarayıcı")
    st.caption(
        "Veriler ücretsiz kaynaktan gelir ve gecikmelidir (yaklaşık 15 dk). "
        "Bu araç sadece aday listesi üretir, yatırım tavsiyesi değildir. "
        "İşleme girmeden önce güncel fiyatı aracı kurum ekranından kontrol et."
    )

    with st.sidebar:
        st.header("Ayarlar")
        aralik = st.selectbox("Zaman dilimi", list(ARALIK_AYARLARI.keys()), index=0)
        mod = st.radio("Sinyal türü", [MOD_KESISIM, MOD_ALTINDA])
        rsi_periyot = st.number_input("RSI periyodu", 2, 50, 14)
        esik = st.slider("RSI eşiği", 10, 50, 30)
        geriye_bakis = 1
        if mod == MOD_KESISIM:
            geriye_bakis = st.slider(
                "Kesişim son kaç mumda olsun?", 1, 5, 1,
                help="1 = sadece son kapanan mumda. 3 = son 3 mumdan birinde.",
            )
        st.divider()
        ema_filtre = st.checkbox("Fiyat EMA üstünde olsun (trend filtresi)", value=True)
        ema_periyot = st.number_input("EMA periyodu", 5, 200, 50)
        st.divider()
        min_gh = st.slider(
            "Minimum göreceli hacim", 0.0, 3.0, 1.0, 0.1,
            help="Mum hacmi / önceki ortalama hacim. 0 yaparsan hacim filtresi kapanır.",
        )
        hacim_periyot = st.number_input("Ortalama hacim için mum sayısı", 5, 100, 20)
        st.divider()
        olusan_mumu_cikar = st.checkbox(
            "Henüz kapanmamış mumu hariç tut", value=True,
            help="Açıksa sadece kapanmış mumlara göre sinyal aranır.",
        )
        metin = st.text_area("Taranacak hisseler (boşluk veya virgülle ayır)",
                             VARSAYILAN_HISSELER, height=200)

    semboller = sorted({s.strip().upper().replace(".IS", "")
                        for s in metin.replace(",", " ").split() if s.strip()})

    st.write(f"**{len(semboller)}** hisse taranacak.")
    if not st.button("🔍 Tara", type="primary"):
        st.info("Soldaki ayarları kontrol edip **Tara** düğmesine bas.")
        return

    periyot, mum_dakika = ARALIK_AYARLARI[aralik]
    ilerleme = st.progress(0.0, text="Veriler çekiliyor...")
    veriler = {}

    def getir(s):
        return s, veri_cek(s, periyot, aralik)

    with ThreadPoolExecutor(max_workers=5) as havuz:
        for i, (s, df) in enumerate(havuz.map(getir, semboller), start=1):
            veriler[s] = df
            ilerleme.progress(i / len(semboller), text=f"Veriler çekiliyor... {i}/{len(semboller)}")
    ilerleme.empty()

    sonuclar, verisiz = [], []
    for s in semboller:
        df = veriler.get(s)
        if df is None:
            verisiz.append(s)
            continue
        r = analiz_et(df, s, mum_dakika, int(rsi_periyot), float(esik), mod,
                      int(geriye_bakis), ema_filtre, int(ema_periyot),
                      int(hacim_periyot), float(min_gh), olusan_mumu_cikar)
        if r is None:
            verisiz.append(s)
        else:
            sonuclar.append(r)

    st.caption(f"Tarama zamanı: {dt.datetime.now().strftime('%d.%m.%Y %H:%M')} (sunucu saati)")

    if not sonuclar:
        st.error("Hiçbir hissenin verisi alınamadı. Birkaç dakika sonra tekrar dene.")
        return

    tablo = pd.DataFrame(sonuclar)
    gecenler = tablo[tablo["_gecti"]].drop(columns="_gecti").sort_values("RSI")

    st.subheader(f"Sinyal veren hisseler ({len(gecenler)})")
    if gecenler.empty:
        st.warning("Şu an tüm şartları sağlayan hisse yok. Filtreleri gevşetmeyi deneyebilirsin.")
    else:
        st.dataframe(gecenler, hide_index=True)
        st.download_button("CSV olarak indir", gecenler.to_csv(index=False).encode("utf-8"),
                           "tarama_sonucu.csv", "text/csv")

    with st.expander(f"Taranan tüm hisseler ({len(tablo)})"):
        st.dataframe(tablo.drop(columns="_gecti").sort_values("RSI"), hide_index=True)
    if verisiz:
        st.caption("Verisi alınamayan veya yetersiz olan hisseler: " + ", ".join(verisiz))


main()

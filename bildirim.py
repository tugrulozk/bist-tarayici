"""BIST düşen trend kırılımı bildirimi (Telegram).

app.py'den bağımsız çalışır, app.py'ye dokunmaz. GitHub Actions tarafından
seans saatlerinde yarım saatte bir çalıştırılır.

Sabit kurallar:
- Saatlik mum (15 dakikalık veriden tam saatlere birleştirilir), aktif mum dahil
- Düşen trend çizgisi kırılımı (🚀): en az 3 temas, eğim %3, temas yakınlığı %0,5
- Dönüş teyidi 3/3 (yeşil mum + hacim + önceki mumun dibi kırılmamış)
- Aynı hisse aynı gün içinde yalnızca bir kez bildirilir

Çalışma modları (MOD ortam değişkeni):
- bildirim : normal çalışma (seans saati kontrolü + tekrar bildirmeme)
- deneme   : seans saati ve tekrar kontrolü yok, o an eşleşen herkesi bildirir
- baglanti : sadece Telegram'a deneme mesajı yollar
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
MIN_TEYIT = 3
# app.py'deki varsayılan trend ayarları (kaydırıcılar varsayılan değerde)
P = dict(trend_temas=3, trend_egim=3.0, trend_tol=0.5, trend_yakin=1.5)
DURUM_DOSYASI = Path("durum") / "bildirilen.json"

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


def trend_kirilimi(df, p):
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
    sn = 2  # kirilim "son 2 mum" icinde aranir (aktif mum dahil)
    son_bar = m - sn  # kirilim bolgesinden onceki son mum
    roll = d["High"].rolling(2 * k + 1, center=True).max().to_numpy()
    piv = [i for i in range(k, pencere - k) if hi[i] == roll[i] and i <= son_bar]
    en_iyi = None
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
            kirdi = cl[son_bar + 1:] > cizgi[on_uz:] * (1 + tol)
            if kirdi.any() and son_kap > cizgi_son:
                durum = 2  # son 2 mumdan biri cizgiyi kapanisla kirdi ve fiyat hala cizginin ustunde
            elif son_kap <= cizgi_son * (1 + tol) and son_kap >= cizgi_son * (1 - p["trend_yakin"] / 100.0):
                durum = 1
            else:
                continue
            anahtar = (durum, temas, j)
            if en_iyi is None or anahtar > en_iyi[0]:
                ort_hacim = vol[max(0, m - 20):m].mean() if m > 0 else np.nan
                hacim_x = vol[m] / ort_hacim if ort_hacim and ort_hacim > 0 else np.nan
                en_iyi = (anahtar, dict(durum=durum, temas=temas, cizgi=cizgi_son,
                                        kirilim=(son_kap / cizgi_son - 1) * 100, hacim=hacim_x))
    return None if en_iyi is None else en_iyi[1]


def donus_teyidi(df):
    """Dusen hissede tepki basladi mi? 3 basit kontrol (son kapanmis mum):
    yesil mum, hacim son 20 mum ortalamasinin ustunde, onceki mumun dibi kirilmamis."""
    yesil = bool(df["Close"].iloc[-1] > df["Open"].iloc[-1])
    hacim = bool(df["Volume"].iloc[-1] > df["Volume"].iloc[-21:-1].mean())
    dip = bool(df["Low"].iloc[-1] >= df["Low"].iloc[-2])
    isaret = ("🟢" if yesil else "") + ("📊" if hacim else "") + ("🔒" if dip else "")
    return int(yesil) + int(hacim) + int(dip), isaret


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


def analiz(sembol, bugun):
    """(durum, sonuç) döner. durum: veri_yok / kisa / bugun_yok / yok / bulundu"""
    df = veri_cek(sembol)
    if df is None:
        return "veri_yok", None
    if len(df) < MIN_MUM:
        return "kisa", None
    if df.index[-1].date() != bugun:
        return "bugun_yok", None   # tatil günü ya da bugün hiç işlem görmemiş hisse
    tr = trend_kirilimi(df, P)
    if not tr or tr["durum"] != 2:
        return "yok", None
    teyit_n, isaret = donus_teyidi(df)
    if teyit_n < MIN_TEYIT:
        return "yok", None
    degisim = gunluk_degisim(df)
    return "bulundu", dict(
        sembol=sembol, fiyat=float(df["Close"].iloc[-1]),
        gunluk=None if np.isnan(degisim) else float(degisim),
        cizgi=float(tr["cizgi"]), temas=int(tr["temas"]), kirilim=float(tr["kirilim"]),
        hacim=None if np.isnan(tr["hacim"]) else float(tr["hacim"]),
        teyit=int(teyit_n), isaret=isaret,
    )


def sayi(x, n=2, isaretli=False):
    s = f"{x:+.{n}f}" if isaretli else f"{x:.{n}f}"
    return s.replace(".", ",")


def mesaj_parcalari(yeniler, simdi, baslik, ek_satir=""):
    ust = [f"{baslik} · {simdi:%H:%M}", f"Yeni: {len(yeniler)} hisse"]
    if ek_satir:
        ust.append(ek_satir)
    ust.append("")
    bloklar = []
    for r in yeniler:
        gun = f"gün {sayi(r['gunluk'], 1, True)}%" if r["gunluk"] is not None else "gün ?"
        hac = f" · hacim {sayi(r['hacim'], 1)}x" if r["hacim"] is not None else ""
        bloklar.append(
            f"{r['sembol']}  {sayi(r['fiyat'])} TL  ({gun})\n"
            f"Çizgi {sayi(r['cizgi'])} · {r['temas']} temas · çizgi üstü %{sayi(r['kirilim'], 1, True)}{hac}\n"
            f"Teyit {r['teyit']}/3 {r['isaret']}\n"
            f"https://www.tradingview.com/chart/?symbol=BIST%3A{r['sembol']}\n"
        )
    alt = ("⚠️ Veri ~15 dk gecikmeli. Aktif mumda sinyal, mum kapanınca kaybolabilir. "
           "Girmeden önce grafiği ve Midas'taki güncel fiyatı kontrol et.")
    parcalar, mevcut = [], "\n".join(ust) + "\n"
    for b in bloklar:
        if len(mevcut) + len(b) > 3500:
            parcalar.append(mevcut.rstrip())
            mevcut = ""
        mevcut += b + "\n"
    parcalar.append((mevcut + alt).strip())
    return parcalar


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
    try:
        d = json.loads(DURUM_DOSYASI.read_text(encoding="utf-8"))
        if d.get("tarih") == bugun_str:
            return set(d.get("hisseler", []))
    except Exception:
        pass
    return set()


def durum_yaz(bugun_str, hisseler):
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    DURUM_DOSYASI.write_text(
        json.dumps({"tarih": bugun_str, "hisseler": sorted(hisseler)}, ensure_ascii=False),
        encoding="utf-8")


def seans_icinde(simdi):
    dakika = simdi.hour * 60 + simdi.minute
    return simdi.weekday() < 5 and 10 * 60 <= dakika <= 18 * 60 + 10


def main():
    mod = (os.environ.get("MOD") or "bildirim").strip()
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

    if mod == "bildirim" and not seans_icinde(simdi):
        print(f"Seans dışı ({simdi:%d.%m.%Y %H:%M}), tarama yapılmadı.")
        return

    semboller = hisse_listesi()
    print(f"{len(semboller)} hisse taranacak ({simdi:%d.%m.%Y %H:%M}, mod: {mod}).")
    sayac = {"veri_yok": 0, "kisa": 0, "bugun_yok": 0, "yok": 0, "bulundu": 0}
    sonuclar = []
    with ThreadPoolExecutor(max_workers=6) as havuz:
        for durum, r in havuz.map(lambda s: analiz(s, bugun), semboller):
            sayac[durum] += 1
            if r:
                sonuclar.append(r)
    print("Sayım:", sayac)

    if sayac["veri_yok"] > 0.5 * len(semboller):
        print("Hisselerin yarısından fazlasının verisi alınamadı (Yahoo erişim sorunu olabilir).")
        sys.exit(1)

    sonuclar.sort(key=lambda r: (-r["temas"], r["sembol"]))
    ozet = (f"Taranan: {len(semboller)} · veri alınamayan: {sayac['veri_yok']} · "
            f"bugün işlem görmeyen: {sayac['bugun_yok']}")

    if mod == "deneme":
        if not sonuclar:
            telegram_gonder(f"🧪 DENEME ({simdi:%H:%M}): şu an şartları sağlayan hisse yok.\n{ozet}")
            return
        for parca in mesaj_parcalari(sonuclar, simdi, "🧪 DENEME · 🚀 Düşen trend kırılımı", ozet):
            telegram_gonder(parca)
        return

    gonderilmis = durum_oku(str(bugun))
    yeniler = [r for r in sonuclar if r["sembol"] not in gonderilmis]
    print("Bulunan:", [r["sembol"] for r in sonuclar], "| Yeni:", [r["sembol"] for r in yeniler])
    if not yeniler:
        return
    hepsi_gitti = True
    for parca in mesaj_parcalari(yeniler, simdi, "🚀 Düşen trend kırılımı"):
        hepsi_gitti = telegram_gonder(parca) and hepsi_gitti
    if hepsi_gitti:
        durum_yaz(str(bugun), gonderilmis | {r["sembol"] for r in yeniler})
    else:
        print("Mesaj gönderilemedi; hisseler kayda eklenmedi, sonraki taramada tekrar denenecek.")
        sys.exit(1)


if __name__ == "__main__":
    main()

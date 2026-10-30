"""BIST düşen trend kırılımı bildirimi (Telegram).

app.py'den bağımsız çalışır, app.py'ye dokunmaz. GitHub Actions tarafından
seans saatlerinde 15 dakikada bir çalıştırılır.

Sabit kurallar:
- Saatlik mum (15 dakikalık veriden tam saatlere birleştirilir), aktif mum dahil
- Düşen trend çizgisi kırılımı (🚀): en az 3 temas, eğim %3, temas yakınlığı %0,5
- Dönüş teyidi: 4 şart (yeşil mum, hacim, dip korunması, fiyat KAMA üstünde); 🚀 = en az 3 ve hacim dahil, ⏳ = en az 2
- Fiyatın hâlâ altında kaldığı başka geçerli çizgi varsa mesaja uyarı satırı eklenir
- Aynı mesajdaki hisseler en çok teyide göre sıralanır, ilki ⭐ ile işaretlenir
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
# Dort teyit: 🟢 yesil mum, 📊 hacim, 🔒 dip korunmus, 📈 fiyat KAMA'nin ustunde
MIN_TEYIT = 3          # guclu sinyal (🚀): 4 teyitten en az 3'u VE hacim sartlarin icinde
ERKEN_TEYIT = 3        # erken sinyal (⏳): hacim disindaki uc teyidin (🟢🔒📈) hepsi sart, sadece hacim eksik
KAMA_N, KAMA_HIZLI, KAMA_YAVAS = 21, 2, 30   # TradingView'deki KAMA ayari
UST_CIZGI_YAKIN = 1.5   # fiyatin ustundeki gecerli cizgi en fazla bu % uzaktaysa mesaja uyari yazilir
FIB_GERI = 40          # Fibonacci tepesi: dipten onceki en fazla bu kadar mum icindeki en yuksek nokta
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
                lo = d["Low"].to_numpy(dtype=float)
                dip_i = i + int(np.argmin(lo[i:m + 1]))
                dip = float(lo[dip_i])
                # Fibonacci icin SON dusus bacagi: dipten onceki FIB_GERI mum icindeki en yuksek nokta
                tepe_fib = float(hi[max(i, dip_i - FIB_GERI):dip_i + 1].max())
                en_iyi = (anahtar, dict(durum=durum, temas=temas, cizgi=cizgi_son,
                                        kirilim=(son_kap / cizgi_son - 1) * 100, hacim=hacim_x,
                                        tepe=tepe_fib, dip=dip))
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
    kama_kesti, kama_ustunde, kama_son = kama_kesisimi(df)
    if kama_ustunde:                 # fiyat su an KAMA'nin ustundeyse teyit sayilir
        teyit_n += 1
        isaret += "📈"
    hacim_var = "📊" in isaret
    if teyit_n < ERKEN_TEYIT:
        return "yok", None
    degisim = gunluk_degisim(df)
    return "bulundu", dict(
        sembol=sembol, fiyat=float(df["Close"].iloc[-1]),
        gunluk=None if np.isnan(degisim) else float(degisim),
        cizgi=float(tr["cizgi"]), temas=int(tr["temas"]), kirilim=float(tr["kirilim"]),
        hacim=None if np.isnan(tr["hacim"]) else float(tr["hacim"]),
        teyit=int(teyit_n), isaret=isaret,
        kademe="guclu" if (teyit_n >= MIN_TEYIT and hacim_var) else "erken",
        eksik=eksik_sartlar(isaret),
        kama=None if np.isnan(kama_son) else kama_son,
        kama_ustunde=bool(kama_ustunde), kama_yeni=bool(kama_kesti),
        ust_cizgi=tr.get("ust_cizgi"),
        **stop_hedef(float(df["Close"].iloc[-1]), float(tr["cizgi"]), tr["tepe"], tr["dip"]),
    )


def sayi(x, n=2, isaretli=False):
    s = f"{x:+.{n}f}" if isaretli else f"{x:.{n}f}"
    return s.replace(".", ",")


def eksik_sartlar(isaret):
    adlar = (("🟢", "yeşil mum"), ("📊", "hacim"), ("🔒", "dip korunması"), ("📈", "KAMA üstü"))
    return [ad for sim, ad in adlar if sim not in isaret]


def kama_satiri(r):
    if r.get("kama") is None:
        return ""
    if r.get("kama_ustunde"):
        ek = " · fiyat üstünde" + (" (yeni kesişim)" if r.get("kama_yeni") else "")
    else:
        ek = " · fiyat altında"
    return f"KAMA {sayi(r['kama'])}{ek}\n"


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


def mesaj_parcalari(yeniler, simdi, baslik, ek_satir=""):
    ust = [f"{baslik} · {simdi:%H:%M}", f"Yeni: {len(yeniler)} hisse"]
    if ek_satir:
        ust.append(ek_satir)
    ust.append("")
    bloklar = []
    for sira, r in enumerate(yeniler):
        yildiz = "⭐ " if (sira == 0 and len(yeniler) > 1) else ""
        gun = f"gün {sayi(r['gunluk'], 1, True)}%" if r["gunluk"] is not None else "gün ?"
        hac = f" · hacim {sayi(r['hacim'], 1)}x" if r["hacim"] is not None else ""
        bloklar.append(
            f"{yildiz}{r['sembol']}  {sayi(r['fiyat'])} TL  ({gun})\n"
            f"Çizgi {sayi(r['cizgi'])} · {r['temas']} temas · çizgi üstü %{sayi(r['kirilim'], 1, True)}{hac}\n"
            f"Teyit {r['teyit']}/4 {r['isaret']}"
            f"{'  · eksik: ' + ', '.join(r['eksik']) if r['eksik'] else ''}\n"
            f"{kama_satiri(r)}"
            f"{stop_hedef_satirlari(r)}"
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
    """(guclu_bildirilen, erken_bildirilen) kume cifti doner."""
    try:
        d = json.loads(DURUM_DOSYASI.read_text(encoding="utf-8"))
        if d.get("tarih") == bugun_str:
            return set(d.get("hisseler", [])), set(d.get("erkenler", []))
    except Exception:
        pass
    return set(), set()


def durum_yaz(bugun_str, guclu, erken):
    DURUM_DOSYASI.parent.mkdir(exist_ok=True)
    DURUM_DOSYASI.write_text(
        json.dumps({"tarih": bugun_str, "hisseler": sorted(guclu), "erkenler": sorted(erken)},
                   ensure_ascii=False),
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

    sonuclar.sort(key=siralama_anahtari)
    ozet = (f"Taranan: {len(semboller)} · veri alınamayan: {sayac['veri_yok']} · "
            f"bugün işlem görmeyen: {sayac['bugun_yok']}")

    gucluler = [r for r in sonuclar if r["kademe"] == "guclu"]
    erkenler = [r for r in sonuclar if r["kademe"] == "erken"]
    BASLIK_G, BASLIK_E = "🚀 Düşen trend kırılımı", "⏳ Erken sinyal"

    if mod == "deneme":
        if not sonuclar:
            telegram_gonder(f"🧪 DENEME ({simdi:%H:%M}): şu an şartları sağlayan hisse yok.\n{ozet}")
            return
        if gucluler:
            for parca in mesaj_parcalari(gucluler, simdi, "🧪 DENEME · " + BASLIK_G, ozet):
                telegram_gonder(parca)
        if erkenler:
            for parca in mesaj_parcalari(erkenler, simdi, "🧪 DENEME · " + BASLIK_E, "" if gucluler else ozet):
                telegram_gonder(parca)
        return

    g_gitti, e_gitti = durum_oku(str(bugun))
    yeni_g = [r for r in gucluler if r["sembol"] not in g_gitti]
    # erken sinyal: ne erken ne guclu olarak bugun bildirilmis olmali
    yeni_e = [r for r in erkenler if r["sembol"] not in e_gitti and r["sembol"] not in g_gitti]
    print("Güçlü:", [r["sembol"] for r in gucluler], "| Yeni:", [r["sembol"] for r in yeni_g])
    print("Erken:", [r["sembol"] for r in erkenler], "| Yeni:", [r["sembol"] for r in yeni_e])
    if not yeni_g and not yeni_e:
        return

    hata = False
    if yeni_g:
        ok = True
        for parca in mesaj_parcalari(yeni_g, simdi, BASLIK_G):
            ok = telegram_gonder(parca) and ok
        if ok:
            g_gitti = g_gitti | {r["sembol"] for r in yeni_g}
        else:
            hata = True
    if yeni_e:
        ok = True
        for parca in mesaj_parcalari(yeni_e, simdi, BASLIK_E):
            ok = telegram_gonder(parca) and ok
        if ok:
            e_gitti = e_gitti | {r["sembol"] for r in yeni_e}
        else:
            hata = True
    durum_yaz(str(bugun), g_gitti, e_gitti)
    if hata:
        print("Bazı mesajlar gönderilemedi; o hisseler kayda eklenmedi, sonraki taramada tekrar denenecek.")
        sys.exit(1)


if __name__ == "__main__":
    main()

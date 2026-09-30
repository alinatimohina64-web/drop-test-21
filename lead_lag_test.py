#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
«BTC ПОШЁЛ — АЛЬТ ЕЩЁ НЕТ»: успевает ли альт догнать биткоин так, чтобы на этом заработать после комиссий?

Данные: 5-минутки Binance (спот) с 2021 года: BTC и 8 альтов (ETH, SOL, XRP, DOGE, ADA, LINK, AVAX, LTC).
Кэш — data_btc/ (как у btc_lab.py). Разработка 2021–2023, ПРОВЕРКА 2024–2026.

Сигнал (на закрытии 5m-свечи, всё по прошлому):
  BTC за последние 15 минут (3 свечи 5m) сдвинулся на >= 0.6% в одну сторону,
  а альт за те же 15 минут прошёл в ту же сторону меньше половины хода BTC — «отстал».
Вход в альт в сторону BTC по открытию следующей 5m-свечи. Выход по времени: через 15, 30 или 60 минут.
Одна сделка на альт за раз. Комиссия 0.11% за круг (рынок), для справки — 0.04% (лимитки).

Сравнение (контроль): те же моменты, но альт НЕ отстал (уже прошёл >= хода BTC). Если догоняние реально,
у отставших ход дальше в сторону BTC должен быть больше, чем у не отставших.

ВАРИАНТ 2 «часовой импульс» (объявлен до прогона, переменные окружения): LL_LOOK=12 LL_MOVE=1.0
LL_HOLDS=12,24,48 — BTC за 1 час прошёл >= 1%, альт отстал, выход через 1, 2 или 4 часа. Критерий тот же.

КРИТЕРИЙ (объявлен ДО прогона). Идея работает, если хотя бы для одного из трёх сроков выхода:
  1) на проверке 2024–2026 средняя сделка после комиссии 0.11% > 0 и t >= 2;
  2) на разработке 2021–2023 средняя сделка после комиссии тоже > 0;
  3) в плюсе не меньше 2 из 3 лет проверки;
  4) отставшие альты дальше идут в сторону BTC сильнее, чем не отставшие (на проверке).
Результат в % на сделку и в $ при позиции 3 000 $ (маржа 300 × 10). Отчёт: lead_lag_report.txt
"""
import csv
import io
import math
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np

REPORT = "lead_lag_report.txt"
DATA_DIR = "data_btc"
ALTS = ["ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "LTC"]
M5 = 300000
LOOK = int(os.environ.get("LL_LOOK") or 3)             # свечей 5m назад: 3 = 15 минут, 12 = 1 час
MOVE = float(os.environ.get("LL_MOVE") or 0.6) / 100    # ход BTC, %
LAG = 0.5                                               # альт прошёл меньше половины хода BTC
HOLDS = [(int(h), "%d мин" % (int(h) * 5)) for h in (os.environ.get("LL_HOLDS") or "3,6,12").split(",")]
FEE_T, FEE_M = 0.11, 0.04
NOTIONAL = 3000.0
START = datetime(2021, 1, 1, tzinfo=timezone.utc)
TEST_FROM = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TEST_YEARS = [2024, 2025, 2026]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def _get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * a)
        except Exception:                                        # noqa: BLE001
            time.sleep(2 + 3 * a)
    return None


def _cached(name, url, keep=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    blob = _get(url)
    if blob is not None and keep:
        with open(p, "wb") as f:
            f.write(blob)
    return blob


def _rows(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    with z.open(z.namelist()[0]) as f:
        return list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))


def load_5m(sym):
    base = "https://data.binance.vision/data/spot/%s/klines/" + sym + "USDT/5m/"
    now = datetime.now(timezone.utc)
    bars = {}
    y, m = START.year, START.month
    while (y, m) <= (now.year, now.month):
        tag = "%04d-%02d" % (y, m)
        cur = (y, m) == (now.year, now.month)
        # у BTC старые месяцы уже лежат в кэше btc_lab под именем 5m-YYYY-MM.zip
        name = ("5m-%s.zip" % tag) if sym == "BTC" else ("%s-5m-%s.zip" % (sym, tag))
        blob = None if cur else _cached(name, base % "monthly" + "%sUSDT-5m-%s.zip" % (sym, tag))
        blobs = [blob] if blob is not None else []
        if blob is None:
            d = datetime(y, m, 1, tzinfo=timezone.utc)
            while d.month == m and d.date() < now.date():
                ds = d.strftime("%Y-%m-%d")
                nm = ("5m-%s.zip" % ds) if sym == "BTC" else ("%s-5m-%s.zip" % (sym, ds))
                b = _cached(nm, base % "daily" + "%sUSDT-5m-%s.zip" % (sym, ds),
                            keep=d.date() < (now - timedelta(days=1)).date())
                if b is not None:
                    blobs.append(b)
                d += timedelta(days=1)
        for b in blobs:
            for r in _rows(b):
                if not r or not r[0].isdigit():
                    continue
                t = int(r[0])
                if t > 10 ** 14:
                    t //= 1000
                bars[t] = (float(r[1]), float(r[4]))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    ts = np.array(sorted(bars), dtype=np.int64)
    o = np.array([bars[t][0] for t in ts])
    c = np.array([bars[t][1] for t in ts])
    say("  %s: %d свечей 5m" % (sym, len(ts)))
    return ts, o, c


def mean_t(v):
    n = len(v)
    if n < 3:
        return (sum(v) / n if n else 0.0), 0.0
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, (mu / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def main():
    t0 = time.time()
    say("«BTC ПОШЁЛ — АЛЬТ ЕЩЁ НЕТ» · %s · BTC за %d мин >= %.1f%%, альт прошёл < %.0f%% хода BTC · 5m Binance с 2021"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), LOOK * 5, MOVE * 100, LAG * 100))
    bt, bo, bc = load_5m("BTC")
    bidx = {int(t): i for i, t in enumerate(bt)}
    res = {h: {"lag": [], "nolag": []} for h, _ in HOLDS}     # (T, ход в сторону BTC, %)
    for sym in ALTS:
        at, ao, ac = load_5m(sym)
        n_sig = 0
        busy = {h: 0 for h, _ in HOLDS}
        for i in range(LOOK, len(at) - max(h for h, _ in HOLDS) - 2):
            T = int(at[i])
            j = bidx.get(T)
            if j is None or j < LOOK or int(bt[j - LOOK]) != T - LOOK * M5 or int(at[i - LOOK]) != T - LOOK * M5:
                continue
            br = bc[j] / bc[j - LOOK] - 1
            if abs(br) < MOVE:
                continue
            d = 1 if br > 0 else -1
            ar = ac[i] / ac[i - LOOK] - 1
            lag = d * ar < LAG * abs(br)
            nolag = d * ar >= abs(br)
            if not (lag or nolag):
                continue
            if int(at[i + 1]) != T + M5:
                continue
            p0 = ao[i + 1]
            for h, _ in HOLDS:
                if T < busy[h]:
                    continue
                if int(at[i + h]) != T + h * M5:
                    continue
                px = ao[i + 1 + h] if i + 1 + h < len(ao) else ac[i + h]
                mv = d * (px / p0 - 1) * 100
                res[h]["lag" if lag else "nolag"].append((T, mv))
                busy[h] = T + (h + 1) * M5
            n_sig += lag
        say("  %s: сигналов «отстал» %d · %.0f с" % (sym, n_sig, time.time() - t0))

    ok_any = False
    for h, nm in HOLDS:
        say("")
        say("=" * 110)
        say("ВЫХОД ЧЕРЕЗ %s (ход в сторону BTC, %% на сделку; $ — позиция 3 000 $)" % nm)
        for grp, title in (("lag", "альт отстал (сигнал)"), ("nolag", "альт не отстал (контроль)")):
            for per, cond in (("разработка 2021–2023", lambda T: T < TEST_FROM), ("проверка 2024–2026", lambda T: T >= TEST_FROM)):
                v = [x[1] for x in res[h][grp] if cond(x[0])]
                g, tg = mean_t(v)
                nt, tt = mean_t([x - FEE_T for x in v])
                nm_, _ = mean_t([x - FEE_M for x in v])
                say("  %-26s %-22s сделок %6d · ход %+.3f%% · после 0.11%%: %+.3f%% (t = %+.1f) = %+.2f $ · после 0.04%%: %+.3f%%"
                    % (title, per, len(v), g, nt, tt, nt / 100 * NOTIONAL, nm_))
        dev = [x[1] - FEE_T for x in res[h]["lag"] if x[0] < TEST_FROM]
        tst = [x[1] - FEE_T for x in res[h]["lag"] if x[0] >= TEST_FROM]
        m_d, _ = mean_t(dev)
        m_t, t_t = mean_t(tst)
        yrs = sum(1 for y in TEST_YEARS if sum(x[1] - FEE_T for x in res[h]["lag"] if year_of(x[0]) == y) > 0)
        g_lag, _ = mean_t([x[1] for x in res[h]["lag"] if x[0] >= TEST_FROM])
        g_no, _ = mean_t([x[1] for x in res[h]["nolag"] if x[0] >= TEST_FROM])
        say("  по годам проверки: " + " · ".join(
            "%d: %+.0f $ (%d)" % (y, sum(x[1] - FEE_T for x in res[h]["lag"] if year_of(x[0]) == y) / 100 * NOTIONAL,
                                 sum(1 for x in res[h]["lag"] if year_of(x[0]) == y)) for y in TEST_YEARS))
        c1, c2, c3, c4 = m_t > 0 and t_t >= 2, m_d > 0, yrs >= 2, g_lag > g_no
        ok = c1 and c2 and c3 and c4
        ok_any |= ok
        say("  критерии: проверка >0 и t>=2 [%s] · разработка >0 [%s] · 2 из 3 лет [%s] · отставшие догоняют сильнее [%s] → %s"
            % tuple(["да" if x else "нет" for x in (c1, c2, c3, c4)] + ["РАБОТАЕТ" if ok else "не работает"]))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): %s" % ("ИДЕЯ РАБОТАЕТ хотя бы для одного срока" if ok_any else "идея не работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

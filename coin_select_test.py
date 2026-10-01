#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОТБОР МОНЕТ И ЦЕЛИ ПОД МОНЕТУ — подбор на 2022–2026, ПРОВЕРКА на независимом 2019–2021.

Сигналы — проверенный набор TT2: график 30m, старший ТФ 4ч, только по дневному тренду BTC, фильтр MRC
(дневной канал, упрощённая схема, уровень 2). Выход: стоп 20 × ATR 30m, цель 3 / 4 / 5 / 6 × ATR.
Вход по открытию 15m после сигнала, одна позиция на монету, внутри 15m-свечи «и цель, и стоп» — стоп.
Позиция 1000 $ (маржа 100 × 10), комиссия 0.11% за круг, фандинг 0.01% за 8 ч против позиции.
Данные — Binance спот 15m (кэш data_btc/), одинаковые для обоих периодов.

ОТБОР (только по 2022-01…2026-09):
  А) готовый список, объявленный 01.10.2026 по прошлому тесту: ADA, ALGO, DOT, LINK, NEAR, UNI, XLM — цель 5 ATR;
  Б) подбор под монету: для каждой монеты — цель (3/4/5/6), давшая лучший итог в $ (не меньше 30 сделок);
     монета берётся в работу, если этот итог в плюсе.
ПРОВЕРКА (2019-01…2021-12 — этих лет при отборе не было):
  отобранные монеты с отобранной целью против остальных монет (с их же «лучшей» целью) и против
  случайных входов в отобранных монетах с теми же выходами.

КРИТЕРИЙ (объявлен ДО прогона), отдельно для А и Б:
  1) отобранные монеты на 2019–2021 в сумме в плюсе;
  2) их средняя сделка лучше, чем у остальных монет;
  3) их средняя сделка лучше случайного входа;
  4) в плюсе не меньше 2 из 3 лет (2019, 2020, 2021), где есть сделки.
Лежит рядом с compare_test.py и coin_wf_test.py. Отчёт: coin_select_report.txt
"""
import bisect
import csv
import io
import math
import os
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np

import compare_test as c
import coin_wf_test as cw

REPORT = "coin_select_report.txt"
DATA_DIR = "data_btc"
COINS = ["ETH", "SOL", "XRP", "DOGE", "LINK", "ADA", "AVAX", "BNB", "LTC", "UNI", "FIL", "DOT", "NEAR",
         "XLM", "BCH", "AAVE", "ATOM", "ETC", "ALGO", "CRV", "ONE"]
LIST_A = ["ADA", "ALGO", "DOT", "LINK", "NEAR", "UNI", "XLM"]
M15, M30 = 900000, 1800000
D1, H4 = c.D1, c.H4
TPS = [3.0, 4.0, 5.0, 6.0]
SL = 20.0
NOTIONAL = 1000.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
LOAD_FROM = datetime(2018, 1, 1, tzinfo=timezone.utc)
VAL = (int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
       int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
SEL_FROM = VAL[1]
VAL_YEARS = [2019, 2020, 2021]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


# ---------------- данные Binance 15m ----------------
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


def load_15m(sym):
    base = "https://data.binance.vision/data/spot/%s/klines/" + sym + "USDT/15m/"
    now = datetime.now(timezone.utc)
    bars = {}
    y, m = LOAD_FROM.year, LOAD_FROM.month
    while (y, m) <= (now.year, now.month):
        tag = "%04d-%02d" % (y, m)
        cur = (y, m) == (now.year, now.month)
        blob = None if cur else _cached("%s-15m-%s.zip" % (sym, tag), base % "monthly" + "%sUSDT-15m-%s.zip" % (sym, tag))
        blobs = [blob] if blob is not None else []
        if blob is None and (cur or (y, m) >= (now.year, now.month - 1)):
            d = datetime(y, m, 1, tzinfo=timezone.utc)
            while d.month == m and d.date() < now.date():
                ds = d.strftime("%Y-%m-%d")
                b = _cached("%s-15m-%s.zip" % (sym, ds), base % "daily" + "%sUSDT-15m-%s.zip" % (sym, ds),
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
                bars[t] = [t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
        m += 1
        if m == 13:
            y, m = y + 1, 1
    out = [bars[k] for k in sorted(bars)]
    say("  %s: 15m %d свечей%s" % (sym, len(out), (" с " + datetime.fromtimestamp(out[0][0] / 1000, tz=timezone.utc)
                                                     .strftime("%Y-%m-%d")) if out else ""))
    return out


def supersmoother(src, n):
    a1 = math.exp(-math.sqrt(2) * math.pi / n)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / n)
    c3 = -a1 * a1
    c1 = 1 - b1 - c3
    out = []
    for i, v in enumerate(src):
        s1 = out[i - 1] if i >= 1 else v
        s2 = out[i - 2] if i >= 2 else s1
        out.append(c1 * v + b1 * s1 + c3 * s2)
    return out


def trade(t15, h, l, cl, k0, d, p0, tp, sl):
    if d == 1:
        ht = np.nonzero(h[k0:] >= p0 * (1 + tp))[0]
        hs = np.nonzero(l[k0:] <= p0 * (1 - sl))[0]
    else:
        ht = np.nonzero(l[k0:] <= p0 * (1 - tp))[0]
        hs = np.nonzero(h[k0:] >= p0 * (1 + sl))[0]
    it = int(ht[0]) if len(ht) else None
    is_ = int(hs[0]) if len(hs) else None
    if is_ is not None and (it is None or is_ <= it):
        k, move = k0 + is_, -sl
    elif it is not None:
        k, move = k0 + it, tp
    else:
        k, move = None, d * (cl[-1] / p0 - 1)
    t_end = int(t15[k]) + M15 if k is not None else int(t15[-1])
    hours = (t_end - int(t15[k0])) / 3600000
    return NOTIONAL * move - NOTIONAL * FEE - NOTIONAL * FUND_8H * hours / 8, (t_end if k is not None else None)


def main():
    t0 = time.time()
    say("ОТБОР МОНЕТ И ЦЕЛИ (стоп 20 ATR) · подбор 2022–2026 → проверка 2019–2021 · %s · Binance 15m"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = load_15m("BTC")
    bd1 = c.agg(btc, D1)
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - D1) - 1
        return btc_tr[i] if i >= 0 else 0

    rnd = random.Random(19)
    res = {}            # (монета, цель) -> [(T, $)]
    rnd_res = {}        # (монета, цель) -> [$] — случайные входы на проверке
    for sym in COINS:
        b15 = load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        b30 = c.agg(b15, M30)
        b30t = [x[0] for x in b30]
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        sigs = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < VAL[0] or atrp[i] <= 0 or btc_at(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - D1) - 1
            if jd < 250:
                continue
            thr = (math.pi * rng[jd] + math.pi * 2.415 * rng[jd]) / 2
            px = b30[i][4]
            if (d == 1 and px >= mean[jd] + thr) or (d == -1 and px <= mean[jd] - thr):
                continue
            sigs.append((T, d, atrp[i] / 100))
        for tp in TPS:
            busy = 0
            out = []
            per_year = {}
            for T, d, a in sigs:
                if T < busy:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                pnl, te = trade(t15, h, l, cl, k0, d, o[k0], tp * a, SL * a)
                out.append((T, pnl))
                busy = te if te is not None else 10 ** 15
                if T < VAL[1]:
                    per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
            res[(sym, tp)] = out
            rr = []
            for y, cnt in per_year.items():                    # случайные входы на проверке
                y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                y1 = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                y0 = max(y0, int(t15[0]) + 260 * D1)
                got = tries = 0
                while got < cnt and tries < cnt * 20 and y1 > y0:
                    tries += 1
                    T = rnd.randrange(y0, y1)
                    d = btc_at(T)
                    j = bisect.bisect_right(b30t, T - M30) - 1
                    if d == 0 or j < 20 or atrp[j] <= 0:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    rr.append(trade(t15, h, l, cl, k0, d, o[k0], tp * atrp[j] / 100, SL * atrp[j] / 100)[0])
                    got += 1
            rnd_res[(sym, tp)] = rr
        say("  %s: сигналов %d · %.0f с" % (sym, len(sigs), time.time() - t0))

    have = sorted({k[0] for k in res})

    def sel_part(sym, tp):
        return [p for T, p in res.get((sym, tp), []) if T >= SEL_FROM]

    def val_part(sym, tp):
        return [(T, p) for T, p in res.get((sym, tp), []) if VAL[0] <= T < VAL[1]]

    say("")
    say("=" * 110)
    say("ПОДБОР (2022–2026): итог в $ по целям 3 / 4 / 5 / 6 ATR (стоп 20 ATR)")
    best = {}
    for sym in have:
        tots = []
        for tp in TPS:
            v = sel_part(sym, tp)
            tots.append((sum(v), len(v), tp))
        ok = [x for x in tots if x[1] >= 30]
        b = max(ok) if ok else None
        best[sym] = b
        say("  %-5s %s → лучшая цель %s" % (sym, " · ".join("%g: %+.0f (%d)" % (tp, s_, n_) for s_, n_, tp in tots),
                                          ("%g ATR, %+.0f $" % (b[2], b[0])) if b else "—"))
    sel_B = {s: best[s][2] for s in have if best[s] and best[s][0] > 0}
    sel_A = {s: 5.0 for s in LIST_A if s in have}
    rest_B = {s: (best[s][2] if best[s] else 5.0) for s in have if s not in sel_B}
    rest_A = {s: 5.0 for s in have if s not in sel_A}

    def summary(sel, name):
        v = []
        for s, tp in sel.items():
            v += val_part(s, tp)
        n = len(v)
        tot = sum(p for T, p in v)
        yrs = {}
        for T, p in v:
            yrs.setdefault(year_of(T), []).append(p)
        rv = []
        for s, tp in sel.items():
            rv += rnd_res.get((s, tp), [])
        return n, tot, yrs, rv

    for label, sel, rest in (("А — готовый список (цель 5 ATR)", sel_A, rest_A),
                             ("Б — подбор монеты и цели по 2022–2026", sel_B, rest_B)):
        say("")
        say("=" * 110)
        say("ОТБОР %s: %s" % (label, ", ".join("%s(%g)" % (s, tp) for s, tp in sel.items()) or "—"))
        n, tot, yrs, rv = summary(sel, label)
        n2, tot2, _, _ = summary(rest, "rest")
        a_sel = tot / max(1, n)
        a_rest = tot2 / max(1, n2)
        a_rnd = sum(rv) / max(1, len(rv))
        say("  ПРОВЕРКА 2019–2021: отобранные — сделок %d · итог %+.0f $ · %+.2f $/сд | остальные — сделок %d · %+.2f $/сд | "
            "случайный вход в отобранных — %+.2f $/сд (%d)" % (n, tot, a_sel, n2, a_rest, a_rnd, len(rv)))
        say("  по годам (отобранные): " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(p), len(p)) for y, p in sorted(yrs.items())))
        per = []
        for s, tp in sel.items():
            vv = val_part(s, tp)
            per.append("%s %+.0f (%d)" % (s, sum(p for T, p in vv), len(vv)))
        say("  по монетам (проверка): " + ", ".join(per))
        y_ok = sum(1 for y in VAL_YEARS if y in yrs and sum(yrs[y]) > 0)
        y_have = sum(1 for y in VAL_YEARS if y in yrs)
        c1 = tot > 0
        c2 = a_sel > a_rest
        c3 = a_sel > a_rnd
        c4 = y_have > 0 and y_ok >= min(2, y_have)
        say("  критерий: итог+ [%s] · лучше остальных монет [%s] · лучше случайного входа [%s] · 2 из 3 лет [%s, %d из %d] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет", "да" if c4 else "нет", y_ok, y_have,
               "ОТБОР РАБОТАЕТ" if (c1 and c2 and c3 and c4) else "отбор не работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

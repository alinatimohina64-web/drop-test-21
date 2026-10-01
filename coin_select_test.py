#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОТБОР МОНЕТ И ЦЕЛИ ПОД МОНЕТУ + ВАРИАНТЫ ФИЛЬТРА MRC — подбор на 2022–2026, ПРОВЕРКА на независимом 2019–2021.

Сигналы — TT2: график 30m, старший ТФ 4ч, только по дневному тренду BTC. Фильтр MRC (дневной канал,
SuperSmoother 200, прошлый закрытый день) — 4 варианта, все объявлены до прогона:
  none    — без MRC (база для сравнения);
  simple  — упрощённая, уровень 2 (сейчас в индикаторе): порог = середина между π·1·размах и π·2.415·размах;
  simple3 — упрощённая, уровень 3: порог = внешняя граница π·2.415·размах;
  dev3    — как у разработчика, уровень 3: порог = внешняя граница + 1.5 размаха.
Пороги — те же, что в mrc_filter_test.py. Варианты можно сузить: MRC_VARIANTS=none,simple,...

Выходы: стоп 20 × ATR 30m, цель 3 / 4 / 5 / 6 × ATR (для отбора монет) + проверенная пара цель 5 / стоп 3.5 × ATR
(только для проверки фильтра). Вход по открытию 15m после сигнала, одна позиция на монету, внутри 15m-свечи
«и цель, и стоп» — стоп. Позиция 1000 $ (маржа 100 × 10), комиссия 0.11% за круг, фандинг 0.01% за 8 ч
против позиции. Данные — Binance спот 15m (кэш data_btc/). Сигналы начинаются, когда у монеты есть 250 дневок
(канал MRC), поэтому молодые монеты (SOL, DOT, NEAR, UNI, AVAX, FIL, AAVE, CRV) на проверке — только с ~середины 2021.

ОТБОР (только по 2022-01…2026-09), для каждого варианта MRC отдельно:
  А) готовый список, объявленный 01.10.2026 по прошлому тесту: ADA, ALGO, DOT, LINK, NEAR, UNI, XLM — цель 5 ATR;
  Б) подбор под монету: цель (3/4/5/6), давшая лучший итог в $ (не меньше 30 сделок); монета берётся, если итог в плюсе.
КРИТЕРИЙ ОТБОРА (объявлен ДО прогона), отдельно для А и Б:
  1) отобранные монеты на 2019–2021 в сумме в плюсе; 2) их средняя сделка лучше, чем у остальных монет;
  3) лучше случайного входа; 4) в плюсе не меньше 2 из 3 лет (2019, 2020, 2021), где есть сделки.

ПРОВЕРКА ФИЛЬТРА MRC НА НЕЗАВИСИМОМ 2019–2021 (все монеты вместе, выходы 5/20 и 5/3.5).
КРИТЕРИЙ (объявлен ДО прогона):
  а) вариант «держится», если его средняя сделка лучше, чем без MRC, в целом и не меньше чем в 2 из 3 лет
     (годы, где есть сделки), при ОБОИХ выходах;
  б) уровень 3 (упрощённый или разработчика) «лучше уровня 2», если его итог в $ больше, чем у simple,
     при обоих выходах и на обоих периодах (2019–2021 и 2022–2026).
Для справки печатается то же по 2022–2026 (там фильтр уже подбирался — это не проверка).

СТАРШИЙ ТФ ПОД МОНЕТУ (4ч или 1Д), для каждого варианта MRC, выходы 5/3.5 и 5/20:
  для каждой монеты по 2022–2026 выбирается старший ТФ с большим итогом в $ (у обоих не меньше 30 сделок,
  иначе остаётся 4ч). На 2019–2021 сравниваются сделки с ВЫБРАННЫМ ТФ и с НЕВЫБРАННЫМ (у тех же монет).
КРИТЕРИЙ (объявлен ДО прогона): выбор ТФ под монету работает, если при ОБОИХ выходах на 2019–2021
  1) средняя сделка выбранного ТФ лучше невыбранного; 2) лучше в 2 из 3 лет, где есть сделки;
  3) выбранный ТФ лучше у большинства монет (по итогу в $); 4) итог выбранных лучше, чем у «все монеты на 4ч».
  Решение — по варианту MRC «упрощённая ур. 2» (как в индикаторе), остальные варианты — для справки.
Остальные разделы (отбор монет, фильтр MRC) — на старшем ТФ 4ч, как в проверенном наборе.
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
EXITS = [(tp, SL) for tp in TPS] + [(5.0, 3.5)]
MRC_EXITS = [(5.0, 20.0), (5.0, 3.5)]
VARIANTS = [("none", "без MRC"), ("simple", "упрощённая ур. 2 (сейчас)"),
            ("simple3", "упрощённая ур. 3"), ("dev3", "разработчика ур. 3")]
if os.environ.get("MRC_VARIANTS"):
    _keep = os.environ["MRC_VARIANTS"].split(",")
    VARIANTS = [v for v in VARIANTS if v[0] in _keep or v[0] == "none"]
VNAME = dict(VARIANTS)
NOTIONAL = 1000.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
LOAD_FROM = datetime(2018, 1, 1, tzinfo=timezone.utc)
VAL = (int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000),
       int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
SEL_FROM = VAL[1]
VAL_YEARS = [2019, 2020, 2021]
SEL_YEARS = [2022, 2023, 2024, 2025, 2026]
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


def hot(var, d, px, mean, rng):
    """True — сигнал убирается фильтром (цена слишком далеко в сторону сделки)."""
    if var == "none":
        return False
    outer = math.pi * 2.415 * rng
    if var == "simple":
        thr = (math.pi * 1.0 * rng + outer) / 2
    elif var == "simple3":
        thr = outer
    elif var == "dev3":
        thr = outer + 1.5 * rng
    else:
        raise ValueError(var)
    return px >= mean + thr if d == 1 else px <= mean - thr


def trade(t15, h, l, cl, k0, d, p0, tp, sl):
    """Цель/стоп в долях. Ищем первое касание окнами растущего размера (быстро, правила те же:
    в свече, где задеты оба, — стоп). Возвращает ($, R, время выхода или None, если открыта)."""
    n = len(h)
    tgt = p0 * (1 + d * tp)
    stp = p0 * (1 - d * sl)
    k, step, kk, move = k0, 4096, None, None
    while k < n:
        e = min(n, k + step)
        if d == 1:
            mt, ms = h[k:e] >= tgt, l[k:e] <= stp
        else:
            mt, ms = l[k:e] <= tgt, h[k:e] >= stp
        m = mt | ms
        if m.any():
            j = int(np.argmax(m))
            kk = k + j
            move = -sl if ms[j] else tp
            break
        k, step = e, step * 2
    if kk is None:
        move = d * (cl[-1] / p0 - 1)
    t_end = int(t15[kk]) + M15 if kk is not None else int(t15[-1])
    hours = (t_end - int(t15[k0])) / 3600000
    pnl = NOTIONAL * move - NOTIONAL * FEE - NOTIONAL * FUND_8H * hours / 8
    return pnl, pnl / (NOTIONAL * sl), (t_end if kk is not None else None)


def main():
    t0 = time.time()
    say("ОТБОР МОНЕТ И ЦЕЛИ + ВАРИАНТЫ MRC · подбор 2022–2026 → проверка 2019–2021 · %s · Binance 15m"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    say("Варианты MRC: " + ", ".join("%s — %s" % v for v in VARIANTS))
    btc = load_15m("BTC")
    bd1 = c.agg(btc, D1)
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - D1) - 1
        return btc_tr[i] if i >= 0 else 0

    res = {}            # (вариант, монета, цель, стоп) -> [(T, $, R)]
    rnd_res = {}        # (вариант, монета, цель, стоп) -> [(T, $, R)] — случайные входы на проверке
    have = []
    for sym in COINS:
        b15 = load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        have.append(sym)
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
        cd = [x[4] for x in d1]
        du = [a > b_ for a, b_ in zip(c.ema(cd, 9), c.ema(cd, 21))]
        htfs = (("4h", [x[0] for x in h4a], hu, H4), ("1d", dt, du, D1))
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        counts = []
        for htf, ht, hup, hms in htfs:
          raw = []
          for T, d, i in cw.ind_entries_tf(b30, M30, ht, hup, hms):
            if T < VAL[0] or atrp[i] <= 0 or btc_at(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - D1) - 1
            if jd < 250:
                continue
            raw.append((T, d, atrp[i] / 100, b30[i][4], mean[jd], rng[jd]))
          for var, _ in VARIANTS:
            sigs = [(T, d, a) for T, d, a, px, mn, rg in raw if not hot(var, d, px, mn, rg)]
            counts.append("%s/%s %d" % (htf, var, len(sigs)))
            for tp, sl in (EXITS if htf == "4h" else MRC_EXITS):
                busy = 0
                out = []
                per_year = {}
                for T, d, a in sigs:
                    if T < busy:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    pnl, r, te = trade(t15, h, l, cl, k0, d, o[k0], tp * a, sl * a)
                    out.append((T, pnl, r))
                    busy = te if te is not None else 10 ** 15
                    per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
                key = (var, sym, tp, sl)
                res[(htf,) + key] = out
                if htf != "4h":
                    continue                                    # случайные входы нужны только для 4ч
                rnd = random.Random("%s-%s-%g-%g" % key)
                rr = []
                for y, cnt in per_year.items():                 # случайные входы, столько же по годам
                    y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                    y1 = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                    y0 = max(y0, int(t15[0]) + 260 * D1)
                    y1 = min(y1, int(t15[-1]) - 30 * D1)
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
                        pnl, r, _ = trade(t15, h, l, cl, k0, d, o[k0], tp * atrp[j] / 100, sl * atrp[j] / 100)
                        rr.append((T, pnl, r))
                        got += 1
                rnd_res[("4h",) + key] = rr
        say("  %s: сигналов %s · %.0f с" % (sym, ", ".join(counts), time.time() - t0))

    def part(src, var, sym, tp, sl, period, htf="4h"):
        a, b = period
        return [x for x in src.get((htf, var, sym, tp, sl), []) if a <= x[0] < b]

    SELP = (SEL_FROM, 10 ** 15)

    # ======================= ОТБОР МОНЕТ — по каждому варианту MRC =======================
    for var, vname in VARIANTS:
        say("")
        say("#" * 110)
        say("ВАРИАНТ MRC: %s" % vname)
        say("=" * 110)
        say("ПОДБОР (2022–2026): итог в $ по целям 3 / 4 / 5 / 6 ATR (стоп 20 ATR)")
        best = {}
        for sym in have:
            tots = []
            for tp in TPS:
                v = part(res, var, sym, tp, SL, SELP)
                tots.append((sum(x[1] for x in v), len(v), tp))
            ok = [x for x in tots if x[1] >= 30]
            b = max(ok) if ok else None
            best[sym] = b
            say("  %-5s %s → лучшая цель %s" % (sym, " · ".join("%g: %+.0f (%d)" % (tp, s_, n_) for s_, n_, tp in tots),
                                              ("%g ATR, %+.0f $" % (b[2], b[0])) if b else "—"))
        sel_B = {s: best[s][2] for s in have if best[s] and best[s][0] > 0}
        sel_A = {s: 5.0 for s in LIST_A if s in have}
        rest_B = {s: (best[s][2] if best[s] else 5.0) for s in have if s not in sel_B}
        rest_A = {s: 5.0 for s in have if s not in sel_A}

        def summary(sel):
            v, rv = [], []
            for s, tp in sel.items():
                v += part(res, var, s, tp, SL, VAL)
                rv += part(rnd_res, var, s, tp, SL, VAL)
            yrs = {}
            for x in v:
                yrs.setdefault(year_of(x[0]), []).append(x[1])
            return len(v), sum(x[1] for x in v), yrs, [x[1] for x in rv]

        for label, sel, rest in (("А — готовый список (цель 5 ATR)", sel_A, rest_A),
                                 ("Б — подбор монеты и цели по 2022–2026", sel_B, rest_B)):
            say("")
            say("ОТБОР %s: %s" % (label, ", ".join("%s(%g)" % (s, tp) for s, tp in sel.items()) or "—"))
            n, tot, yrs, rv = summary(sel)
            n2, tot2, _, _ = summary(rest)
            a_sel = tot / max(1, n)
            a_rest = tot2 / max(1, n2)
            a_rnd = sum(rv) / max(1, len(rv))
            say("  ПРОВЕРКА 2019–2021: отобранные — сделок %d · итог %+.0f $ · %+.2f $/сд | остальные — сделок %d · %+.2f $/сд | "
                "случайный вход в отобранных — %+.2f $/сд (%d)" % (n, tot, a_sel, n2, a_rest, a_rnd, len(rv)))
            say("  по годам (отобранные): " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(p), len(p)) for y, p in sorted(yrs.items())))
            per = []
            for s, tp in sel.items():
                vv = part(res, var, s, tp, SL, VAL)
                per.append("%s %+.0f (%d)" % (s, sum(x[1] for x in vv), len(vv)))
            say("  по монетам (проверка): " + ", ".join(per))
            y_ok = sum(1 for y in VAL_YEARS if y in yrs and sum(yrs[y]) > 0)
            y_have = sum(1 for y in VAL_YEARS if y in yrs)
            c1, c2, c3 = tot > 0, a_sel > a_rest, a_sel > a_rnd
            c4 = y_have > 0 and y_ok >= min(2, y_have)
            say("  критерий: итог+ [%s] · лучше остальных монет [%s] · лучше случайного входа [%s] · 2 из 3 лет [%s, %d из %d] → %s"
                % ("да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет", "да" if c4 else "нет", y_ok, y_have,
                   "ОТБОР РАБОТАЕТ" if (c1 and c2 and c3 and c4) else "отбор не работает"))

    # ======================= ПРОВЕРКА ФИЛЬТРА MRC (все монеты вместе) =======================
    def pool(src, var, tp, sl, period):
        v = []
        for s in have:
            v += part(src, var, s, tp, sl, period)
        return v

    def stats(v, years):
        yrs = {}
        for x in v:
            yrs.setdefault(year_of(x[0]), []).append(x)
        n = len(v)
        return {"n": n, "tot": sum(x[1] for x in v), "avg": sum(x[1] for x in v) / max(1, n),
                "R": sum(x[2] for x in v) / max(1, n),
                "yrs": {y: (len(yrs[y]), sum(x[1] for x in yrs[y]), sum(x[1] for x in yrs[y]) / len(yrs[y]))
                        for y in years if y in yrs}}

    say("")
    say("#" * 110)
    say("ФИЛЬТР MRC — ВСЕ МОНЕТЫ ВМЕСТЕ (позиция 1000 $; одна позиция на монету)")
    holds = {v: True for v, _ in VARIANTS if v != "none"}
    totals = {}
    for pname, period, years, is_val in (("ПРОВЕРКА 2019–2021 (независимые годы)", VAL, VAL_YEARS, True),
                                          ("справка 2022–2026 (на этих годах фильтр выбирали)", SELP, SEL_YEARS, False)):
        say("")
        say("=" * 110)
        say(pname)
        for tp, sl in MRC_EXITS:
            say("  выход: цель %g / стоп %g × ATR" % (tp, sl))
            st = {v: stats(pool(res, v, tp, sl, period), years) for v, _ in VARIANTS}
            rs = {v: stats(pool(rnd_res, v, tp, sl, period), years) for v, _ in VARIANTS}
            base = st["none"]
            for v, vn in VARIANTS:
                s = st[v]
                totals[(v, tp, sl, is_val)] = s["tot"]
                line = "    %-27s сделок %4d · итог %+6.0f $ · %+6.2f $/сд (%+.3f R) · случ. вход %+6.2f $/сд · по годам: %s" % (
                    vn, s["n"], s["tot"], s["avg"], s["R"], rs[v]["avg"],
                    " · ".join("%d: %+.0f $ (%d, %+.1f/сд)" % (y, s["yrs"][y][1], s["yrs"][y][0], s["yrs"][y][2])
                               for y in years if y in s["yrs"]))
                if v != "none" and is_val:
                    yb = [y for y in years if y in s["yrs"] and y in base["yrs"]]
                    yw = sum(1 for y in yb if s["yrs"][y][2] > base["yrs"][y][2])
                    ok = s["avg"] > base["avg"] and yw >= min(2, len(yb)) and len(yb) > 0
                    holds[v] = holds[v] and ok
                    line += "  → лучше без MRC: %s (лет %d из %d)" % ("да" if ok else "нет", yw, len(yb))
                say(line)

    say("")
    say("=" * 110)
    say("ИТОГ ПО КРИТЕРИЮ ФИЛЬТРА (объявлен до прогона)")
    for v, vn in VARIANTS:
        if v == "none":
            continue
        say("  а) %-27s на 2019–2021 при обоих выходах: %s" % (vn, "ДЕРЖИТСЯ" if holds[v] else "не держится"))
    if "simple" in VNAME:
        for v in ("simple3", "dev3"):
            if v not in VNAME:
                continue
            cells = []
            better = True
            for tp, sl in MRC_EXITS:
                for is_val in (True, False):
                    a, b = totals[(v, tp, sl, is_val)], totals[("simple", tp, sl, is_val)]
                    better = better and a > b
                    cells.append("%g/%g %s: %+.0f против %+.0f" % (tp, sl, "19–21" if is_val else "22–26", a, b))
            say("  б) %-27s против ур. 2: %s → %s" % (VNAME[v], " · ".join(cells),
                                                     "ЛУЧШЕ ур. 2" if better else "не лучше ур. 2"))
    # ======================= СТАРШИЙ ТФ ПОД МОНЕТУ =======================
    say("")
    say("#" * 110)
    say("СТАРШИЙ ТФ ПОД МОНЕТУ: выбор 4ч/1Д по 2022–2026 → проверка 2019–2021")
    htf_verdict = {}
    for var, vname in VARIANTS:
        say("")
        say("=" * 110)
        say("ВАРИАНТ MRC: %s" % vname)
        ok_all = True
        for tp, sl in MRC_EXITS:
            say("  выход: цель %g / стоп %g × ATR" % (tp, sl))
            chosen, notch, all4 = [], [], []
            coins_better = coins_have = 0
            picks = []
            for sym in have:
                s4 = part(res, var, sym, tp, sl, SELP, "4h")
                s1 = part(res, var, sym, tp, sl, SELP, "1d")
                t4, t1 = sum(x[1] for x in s4), sum(x[1] for x in s1)
                pick = "1d" if (len(s4) >= 30 and len(s1) >= 30 and t1 > t4) else "4h"
                other = "4h" if pick == "1d" else "1d"
                vp = part(res, var, sym, tp, sl, VAL, pick)
                vo = part(res, var, sym, tp, sl, VAL, other)
                chosen += vp
                notch += vo
                all4 += part(res, var, sym, tp, sl, VAL, "4h")
                if vp or vo:
                    coins_have += 1
                    if sum(x[1] for x in vp) > sum(x[1] for x in vo):
                        coins_better += 1
                picks.append("%s %s (подбор 4ч %+.0f / 1Д %+.0f; проверка %+.0f против %+.0f)" % (
                    sym, "1Д" if pick == "1d" else "4ч", t4, t1, sum(x[1] for x in vp), sum(x[1] for x in vo)))
            say("    по монетам: " + "; ".join(picks))
            sc, sn, sa = stats(chosen, VAL_YEARS), stats(notch, VAL_YEARS), stats(all4, VAL_YEARS)
            for nm, st_ in (("выбранный ТФ", sc), ("невыбранный ТФ", sn), ("все на 4ч", sa)):
                say("    %-15s сделок %4d · итог %+6.0f $ · %+6.2f $/сд (%+.3f R) · по годам: %s" % (
                    nm, st_["n"], st_["tot"], st_["avg"], st_["R"],
                    " · ".join("%d: %+.0f $ (%d, %+.1f/сд)" % (y, st_["yrs"][y][1], st_["yrs"][y][0], st_["yrs"][y][2])
                               for y in VAL_YEARS if y in st_["yrs"])))
            yb = [y for y in VAL_YEARS if y in sc["yrs"] and y in sn["yrs"]]
            yw = sum(1 for y in yb if sc["yrs"][y][2] > sn["yrs"][y][2])
            k1 = sc["avg"] > sn["avg"]
            k2 = len(yb) > 0 and yw >= min(2, len(yb))
            k3 = coins_better * 2 > coins_have
            k4 = sc["tot"] > sa["tot"]
            ok = k1 and k2 and k3 and k4
            ok_all = ok_all and ok
            say("    критерий: лучше невыбранного [%s] · по годам [%s, %d из %d] · у большинства монет [%s, %d из %d] · "
                "лучше «все на 4ч» [%s] → %s" % ("да" if k1 else "нет", "да" if k2 else "нет", yw, len(yb),
                                                "да" if k3 else "нет", coins_better, coins_have, "да" if k4 else "нет",
                                                "да" if ok else "нет"))
        htf_verdict[var] = ok_all
        say("  → выбор ТФ под монету (%s): %s" % (vname, "РАБОТАЕТ" if ok_all else "не работает"))
    if "simple" in htf_verdict:
        say("")
        say("РЕШЕНИЕ по старшему ТФ под монету (MRC упрощённая ур. 2): %s"
            % ("РАБОТАЕТ — можно выбирать 4ч/1Д по монете" if htf_verdict["simple"] else "НЕ РАБОТАЕТ — оставить 4ч для всех"))

    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

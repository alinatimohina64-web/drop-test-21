#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОНТРТРЕНДОВАЯ ЛОНГ-СЕТКА: какие настройки и какой депозит убирают долгие «инвесты» (как 500 дней у XRP в 2019–2020).

Сетка как в Cryptorg (вкладка «Усреднение»): первый ордер F по рынку, страховочный ордер k = F·M^k (СО1 = 2F при M=2),
шаг k-го СО = шаг · (множитель шага)^(k−1), цены СО отсчитываются от цены входа нарастающим итогом (как в предпросмотре
бота: −0.89%, −1.78% …). Количество монет каждого ордера фиксируется по цене входа (как в боте). Тейк 0.8% от средней,
стопа нет, после тейка новый цикл сразу. Кросс-маржа, депозит = полная сетка (номинал F·(M^(N+1)−1)/(M−1)).
Ликвидация: депозит + заработанное − убыток позиции дошёл до 0.5% позиции → монета выбывает с итогом −депозит.
Исполнение — Binance 15m, с 2019-01-01. В свече, где набрался ордер, тейк не считается (худший порядок).
Комиссия 0.04% с каждой стороны, фандинг 0.005% / 8 ч на набранный объём (GRID_FUND меняет).

ВАРИАНТЫ (объявлены до прогона, без подбора; первый ордер 10 $, 9 СО, объём ×2 — кроме V8):
  V1 сейчас:              шаг 0.89%, множитель шага 1    → ширина  8%, депозит 10 230 $
  V2 шире:                шаг 1.33%, множитель 1          → ширина 12%, депозит 10 230 $
  V3 шире:                шаг 1.78%, множитель 1          → ширина 16%, депозит 10 230 $
  V4 шире:                шаг 2.22%, множитель 1          → ширина 20%, депозит 10 230 $
  V5 шире:                шаг 3.33%, множитель 1          → ширина 30%, депозит 10 230 $
  V6 растущий шаг:        шаг 0.89%, множитель шага 1.15  → ширина ≈15%, депозит 10 230 $
  V7 растущий шаг:        шаг 0.89%, множитель шага 1.3   → ширина ≈29%, депозит 10 230 $
  V8 больше депозит:      10 СО, шаг 1.6%, множитель 1    → ширина 16%, депозит 20 470 $
«Инвест» = цикл от входа до тейка. Считаем: самый долгий цикл, сколько циклов дольше 30 / 90 / 180 дней, сколько
всего дней провели в циклах дольше 30 дней.

КРИТЕРИЙ (объявлен ДО прогона). Вариант лучше текущего (V1), если на ≥ 2 из 3 наборов монет одновременно:
  1) медиана по монетам самого долгого цикла короче, чем у V1, хотя бы на 30%;
  2) итог набора в % от депозита не хуже V1 больше чем на 20% (итог ≥ V1 − 0.2·|V1|);
  3) ликвидаций не больше, чем у V1;
  4) в обеих половинах периода (до / после 2023-01-01) итог не хуже V1 больше чем на 20%.
Если проходят несколько — рекомендуем самый близкий к текущему (наименьшая ширина): меньше перемен — меньше подгонки.
XRP — справочно (одна монета). Контроль: те же варианты на перемешанных 15m-свечах каждой монеты — справочно,
показывает, что часть эффекта ширины — простая арифметика сетки, а не свойство рынка.
Лежит рядом с grid_regime_test.py, coin_select_test.py. Отчёт: grid_settings_report.txt
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import grid_regime_test as gr

REPORT = "grid_settings_report.txt"
D1 = cs.D1
H8 = 8 * 3600 * 1000
TP = 0.008
FEE = 0.0004
FUND8 = float(os.environ.get("GRID_FUND", "0.005")) / 100
MMR = 0.005
START, HALF = gr.START, gr.HALF
SETS = gr.SETS
# имя, первый ордер, СО, множитель объёма, шаг %, множитель шага
VARS = [("V1", 10.0, 9, 2.0, 0.89, 1.0), ("V2", 10.0, 9, 2.0, 1.33, 1.0), ("V3", 10.0, 9, 2.0, 1.78, 1.0),
        ("V4", 10.0, 9, 2.0, 2.22, 1.0), ("V5", 10.0, 9, 2.0, 3.33, 1.0), ("V6", 10.0, 9, 2.0, 0.89, 1.15),
        ("V7", 10.0, 9, 2.0, 0.89, 1.3), ("V8", 10.0, 10, 2.0, 1.6, 1.0)]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def grid(v):
    _, F, N, M, st_, sm = v
    sizes = [F * M ** k for k in range(N + 1)]
    off, s, cur = [0.0], 0.0, st_
    for k in range(N):
        s += cur
        off.append(s / 100.0)
        cur *= sm
    return sizes, off, sum(sizes)


def vdesc(v):
    sizes, off, dep = grid(v)
    return "%s: первый %.0f $, %d СО ×%g, шаг %.2f%% ×%g → ширина %.1f%%, депозит %.0f $" % (
        v[0], v[1], v[2], v[3], v[4], v[5], off[-1] * 100, dep)


def sim(t, o, h, l, c, v):
    sizes, off, DEPO = grid(v)
    N = len(sizes) - 1
    n = len(t)
    i = int(np.searchsorted(t, START))
    real, peak, mdd = 0.0, 0.0, 0.0
    cyc, liq, end_mtm = 0, None, 0.0
    durs = []                      # (дней, t начала, t конца или None)
    half = [0.0, 0.0]
    while i < n and liq is None:
        p0 = o[i]
        lev = [p0 * (1 - x) for x in off]
        qty = sizes[0] / p0
        cost = qty * p0
        fees = cost * FEE
        fund, tl, k, j, ts = 0.0, t[i], 1, i, t[i]
        while True:
            tp = cost / qty * (1 + TP)
            L = lev[k] if k <= N else -1.0
            m = gr.first_hit(l, h, j, L, tp, n)
            if m > j:
                lowseg = float(l[j:m].min())
                fnow = fund + cost * (t[m - 1] - tl) / H8 * FUND8
                p_liq = (cost + fees + fnow - real - DEPO) / (qty * (1 - MMR))
                if lowseg <= p_liq:
                    liq = t[j + int(np.argmax(l[j:m] <= p_liq))]
                    break
                mdd = max(mdd, peak - (real + qty * lowseg - cost - fees - fnow))
            if m >= n:
                fund += cost * (t[n - 1] - tl) / H8 * FUND8
                end_mtm = qty * c[n - 1] * (1 - FEE) - cost - fees - fund
                durs.append(((t[n - 1] - ts) / D1, ts, None))
                i = n
                break
            filled = False
            while k <= N and l[m] <= lev[k]:
                fund += cost * (t[m] - tl) / H8 * FUND8
                tl = t[m]
                q = sizes[k] / p0
                qty += q
                cost += q * lev[k]
                fees += q * lev[k] * FEE
                k += 1
                filled = True
            if filled:
                p_liq = (cost + fees + fund - real - DEPO) / (qty * (1 - MMR))
                if l[m] <= p_liq:
                    liq = t[m]
                    break
                mdd = max(mdd, peak - (real + qty * l[m] - cost - fees - fund))
                j = m + 1
                if j >= n:
                    end_mtm = qty * c[n - 1] * (1 - FEE) - cost - fees - fund
                    durs.append(((t[n - 1] - ts) / D1, ts, None))
                    i = n
                    break
                continue
            fund += cost * (t[m] - tl) / H8 * FUND8
            pnl = qty * tp * (1 - FEE) - cost - fees - fund
            real += pnl
            half[0 if t[m] < HALF else 1] += pnl
            cyc += 1
            durs.append(((t[m] - ts) / D1, ts, t[m]))
            peak = max(peak, real)
            i = m + 1
            break
    if liq is not None:
        half[0 if liq < HALF else 1] += -DEPO - real
        mdd = max(mdd, peak + DEPO)
        durs.append(((liq - ts) / D1, ts, None))
        real, end_mtm = -DEPO, 0.0
    half[1] += end_mtm
    d = [x[0] for x in durs]
    return {"tot": real + end_mtm, "pct": 100 * (real + end_mtm) / DEPO, "dep": DEPO, "mdd": mdd, "cyc": cyc,
            "liq": liq, "maxd": max(d) if d else 0.0, "n30": sum(1 for x in d if x > 30),
            "n90": sum(1 for x in d if x > 90), "n180": sum(1 for x in d if x > 180),
            "d30": sum(x for x in d if x > 30), "long": sorted([x for x in durs if x[0] > 60], key=lambda z: z[1]),
            "half": half, "half_pct": [100 * x / DEPO for x in half], "mtm": end_mtm}


def work(sym):
    t0 = time.time()
    try:
        bars = cs.load_15m(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(bars) < 2000:
        return sym, None, "мало данных"
    a = np.array(bars, dtype=float)
    t = a[:, 0].astype(np.int64)
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    if t[-1] < START + 200 * D1:
        return sym, None, "мало данных после 2019"
    i0 = int(np.searchsorted(t, START))
    res = {"years": (t[-1] - t[i0]) / (365.25 * D1)}
    for v in VARS:
        res[v[0]] = sim(t, o, h, l, c, v)
    rng = np.random.default_rng(sum(ord(ch) * 131 ** k for k, ch in enumerate(sym)) % (2 ** 32))
    seg = slice(i0, len(t))
    oo, hh, ll, cc = o[seg], h[seg], l[seg], c[seg]
    p = rng.permutation(len(oo))
    rc_, rh_, rl_ = (cc / oo)[p], (hh / oo)[p], (ll / oo)[p]
    no = np.empty(len(oo))
    no[0] = oo[0]
    no[1:] = oo[0] * np.cumprod(rc_[:-1])
    tt = t[seg]
    for v in VARS:
        res["R" + v[0]] = sim(tt, no, no * rh_, no * rl_, no * rc_, v)
    return sym, res, "%.0f с" % (time.time() - t0)


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def ds(t):
    return gr.dstr(t) if t else "НЕ ВЫШЕЛ"


def main():
    t_start = time.time()
    say("НАСТРОЙКИ СЕТКИ И ДОЛГИЕ ИНВЕСТЫ · %s · Binance 15m · с 2019-01-01 · фандинг %.4f%% / 8 ч"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), FUND8 * 100))
    for v in VARS:
        say("  " + vdesc(v))
    allc = []
    for _, cl in SETS:
        allc += [x for x in cl if x not in allc]
    got = {}
    with mp.get_context("fork").Pool(max(1, os.cpu_count() or 2)) as pool:
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    names = [v[0] for v in VARS]

    say()
    say("=" * 110)
    if "XRP" in got:
        r = got["XRP"]
        say("XRP ПОДРОБНО (справочно — одна монета)")
        for nm in names:
            x = r[nm]
            say("  %s итог %+8.0f $ (%+5.0f%% депо, %+5.0f $ в мес) · половины %+7.0f / %+7.0f · циклов %5d · самый долгий "
                "%4.0f дн · >30д %2d · >90д %2d · >180д %d · дней в инвестах >30д %4.0f · просадка %6.0f $%s"
                % (nm, x["tot"], x["pct"], x["tot"] / (r["years"] * 12), x["half"][0], x["half"][1], x["cyc"], x["maxd"],
                   x["n30"], x["n90"], x["n180"], x["d30"], x["mdd"],
                   (" · ЛИКВИДАЦИЯ " + gr.dstr(x["liq"])) if x["liq"] else ""))
        say("  долгие циклы XRP (> 60 дней):")
        for nm in names:
            say("    %s: %s" % (nm, "; ".join("%s → %s (%.0f дн)" % (gr.dstr(a[1]), ds(a[2]), a[0])
                                              for a in r[nm]["long"]) or "нет"))
        say("  случайные цены XRP: " + " · ".join("%s самый долгий %.0f дн, итог %+.0f%%"
                                                  % (nm, r["R" + nm]["maxd"], r["R" + nm]["pct"]) for nm in names))

    say()
    say("=" * 110)
    say("ПО НАБОРАМ (итог в % от депозита — сумма по монетам; самый долгий цикл — медиана по монетам)")
    passed = {nm: 0 for nm in names[1:]}
    detail = {nm: [] for nm in names[1:]}
    for sn, cl in SETS:
        rs = [got[x] for x in cl if x in got]
        if not rs:
            continue
        say("  НАБОР «%s» (%d монет)" % (sn, len(rs)))
        S = {}
        for nm in names:
            X = [r[nm] for r in rs]
            S[nm] = {"pct": sum(x["pct"] for x in X), "h0": sum(x["half_pct"][0] for x in X),
                     "h1": sum(x["half_pct"][1] for x in X), "maxd": med([x["maxd"] for x in X]),
                     "liq": sum(1 for x in X if x["liq"]), "mdd": med([100 * x["mdd"] / x["dep"] for x in X])}
            RX = [r["R" + nm] for r in rs]
            say("    %s итог %+7.0f%% (половины %+6.0f / %+6.0f) · в плюсе %2d/%d · самый долгий цикл: медиана %4.0f, "
                "макс %4.0f дн · >90д %3d · >180д %3d · просадка медиана %3.0f%% депо · ЛИКВИДАЦИЙ %d · "
                "случайные цены: итог %+6.0f%%, долгий %4.0f дн"
                % (nm, S[nm]["pct"], S[nm]["h0"], S[nm]["h1"], sum(1 for x in X if x["tot"] > 0), len(X), S[nm]["maxd"],
                   max(x["maxd"] for x in X), sum(x["n90"] for x in X), sum(x["n180"] for x in X), S[nm]["mdd"],
                   S[nm]["liq"], sum(x["pct"] for x in RX), med([x["maxd"] for x in RX])))
        b = S["V1"]

        def nb(x, y):
            return x >= y - 0.2 * abs(y)
        for nm in names[1:]:
            s = S[nm]
            c1 = s["maxd"] <= 0.7 * b["maxd"]
            c2 = nb(s["pct"], b["pct"])
            c3 = s["liq"] <= b["liq"]
            c4 = nb(s["h0"], b["h0"]) and nb(s["h1"], b["h1"])
            ok = c1 and c2 and c3 and c4
            passed[nm] += ok
            detail[nm].append(ok)
            say("      %s против V1: долгий −30%% [%s] · итог [%s] · ликвидации [%s] · половины [%s] → %s"
                % (nm, "да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет", "да" if c4 else "нет",
                   "ПРОШЁЛ" if ok else "нет"))

    say()
    say("=" * 110)
    say("ПО МОНЕТАМ: итог % депо / самый долгий цикл, дней")
    say("  %-7s " % "" + " | ".join("%11s" % nm for nm in names))
    for _, cl in SETS:
        for x in cl:
            if x in got:
                r = got[x]
                say("  %-7s " % x + " | ".join("%+5.0f%%/%4.0f%s" % (r[nm]["pct"], r[nm]["maxd"], "L" if r[nm]["liq"] else " ")
                                              for nm in names))
    say("  (L — ликвидация)")

    say()
    say("=" * 110)
    ok = [nm for nm in names[1:] if passed[nm] >= 2]
    for nm in names[1:]:
        say("КРИТЕРИЙ %s: прошёл на %d из 3 наборов %s → %s"
            % (nm, passed[nm], ["да" if q else "нет" for q in detail[nm]], "ПРИНЯТ" if passed[nm] >= 2 else "нет"))
    if ok:
        w = min(ok, key=lambda nm: grid(VARS[names.index(nm)])[1][-1])
        say("→ ИТОГ: лучше текущей — %s; рекомендуется самая близкая к текущей: %s" % (", ".join(ok), vdesc(VARS[names.index(w)])))
    else:
        say("→ ИТОГ: ни один вариант не убирает долгие инвесты без потери дохода — оставляем V1")
    say()
    say("Время %.0f мин" % ((time.time() - t_start) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

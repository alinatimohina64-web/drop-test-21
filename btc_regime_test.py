#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТОРГОВАТЬ ТОЛЬКО КОГДА BTC ВЫШЕ SMA200: помогает ли режим BTC пунктиру и простому EMA200 на альтах?
РЕЖИМ BTC (как в роботе): ВЧЕРАШНЕЕ дневное закрытие BTCUSDT выше дневной SMA200 (без заглядывания вперёд).
Данные, издержки, исполнение — как в прошлых тестах (функции из trend_trail_test.py / c_confirm_test.py).

ВАРИАНТЫ (объявлены до прогона, без подбора), ТФ 4h и 2h, периоды «весь» и «с 2022»:
  C    пунктир + EMA200 монеты (база)
  CB   C, но НОВЫЕ входы только когда BTC выше SMA200 (открытая сделка доводится по пунктиру)      ← гипотеза 1
  CBX  справочно: CB + в позиции только пока BTC выше SMA200 (ушёл ниже — выход по рынку)
  M    держать монету, пока close > EMA200 этого ТФ (база)
  MB   держать, пока close > EMA200 И BTC выше SMA200                                               ← гипотеза 2
  BB   справочно: держать монету, пока BTC выше SMA200 (только режим BTC)
  B&H  купить и держать

КОНТРОЛЬ «сдвинутый режим»: CB и MB с режимом BTC, циклически сдвинутым на 12 разных сроков (та же доля и длина
  «выключений», но в другие даты). Если настоящий режим не лучше сдвинутого — польза случайна.

КРИТЕРИЙ (объявлен ДО прогона), отдельно для каждой гипотезы (CB против C, MB против M) и каждого ТФ.
  ФИЛЬТР BTC ПОЛЕЗЕН, если на НОВОМ наборе в ОБОИХ периодах одновременно:
   1) MAR с фильтром выше, чем без него, у >= 2/3 монет;
   2) медианная MaxDD меньше, а медианный CAGR не хуже больше чем на 2 п.п.;
   3) медианный MAR с настоящим режимом выше, чем со сдвинутым, для >= 10 из 12 сдвигов;
  И на СТАРОМ наборе медианный MAR с фильтром не ниже, чем без него, в обоих периодах.
  BTC входит в старый набор как есть (для него фильтр = собственная SMA200).
Отчёт: btc_regime_report.txt
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

import c_confirm_test as cc
import trend_trail_test as tt

REPORT = "btc_regime_report.txt"
NEW, OLD = cc.NEW, cc.OLD
TFS = [("4h", 240), ("2h", 120)]
VARS = [("C", "пунктир + EMA200"), ("CB", "C, вход при BTC>SMA200"), ("CBX", "CB + выход при BTC<SMA"),
        ("M", "держать > EMA200"), ("MB", "M и BTC>SMA200"), ("BB", "держать при BTC>SMA200")]
HYP = [("CB", "C", "ГИПОТЕЗА 1: пунктир, входы только при BTC>SMA200"), ("MB", "M", "ГИПОТЕЗА 2: EMA200 + BTC>SMA200")]
PERIODS = cc.PERIODS
N_SHIFT = 12
D1 = tt.D1
G = {}
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def btc_regime():
    base, _ = tt.load("BTC")
    d = tt.resample(base, 1440)
    c = d[:, 4]
    sma = np.r_[np.full(199, np.nan), np.convolve(c, np.ones(200) / 200, "valid")]
    with np.errstate(invalid="ignore"):
        up = c > sma
    dk = (d[:, 0] // D1).astype(np.int64)
    d0 = int(np.searchsorted(dk, tt.START // D1))
    L = len(dk) - d0
    shifts = [int(L * (k + 1) / (N_SHIFT + 1)) for k in range(N_SHIFT)]
    return dk, up, d0, shifts


def regime_for(t, shift=0):
    dk, up, d0, _ = G["btc"]
    want = (t // D1).astype(np.int64) - 1
    j = np.clip(np.searchsorted(dk, want, side="right") - 1, 0, len(dk) - 1)
    ok = dk[j] <= want
    if shift:
        L = len(dk) - d0
        j = np.where(j >= d0, d0 + ((j - d0 - shift) % L), j)
    return up[j] & ok


def pos_all(st, above, btc):
    F = np.zeros(len(above), bool)
    cb = tt.positions(st, above & btc, F)
    return {"C": tt.positions(st, above, F), "CB": cb, "CBX": cb * btc, "M": above.astype(float),
            "MB": (above & btc).astype(float), "BB": btc.astype(float)}


def work(sym):
    t_s = time.time()
    try:
        base, _ = tt.load(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(base) < 20000:
        return sym, None, "мало данных"
    t_first = max(tt.START, int(base[0, 0]) + tt.WARM_D * D1)
    res = {"tf": {}}
    for tfn, mins in TFS:
        a = tt.resample(base, mins)
        t, o, h, l, c = (a[:, k] for k in range(5))
        bar_h = mins / 60
        above = c > tt.ema(c, 200)
        st = tt.supertrend_dir(h, l, c, *tt.MAIN)
        btc = regime_for(t)
        pos = pos_all(st, above, btc)
        shp = []
        for s in G["btc"][3]:
            bs = regime_for(t, s)
            shp.append({"CB": tt.positions(st, above & bs, np.zeros(len(c), bool)), "MB": (above & bs).astype(float)})
        R = {}
        for pn, p0, half, _, _ in PERIODS:
            i0 = int(np.searchsorted(t, max(t_first, p0 or 0)))
            bh = cc.run_p(t, o, h, l, c, np.ones(len(c)), i0, half, bar_h, funding=False)
            R[(pn, "bh")] = {k: bh[k] for k in ("ret", "cagr", "mdd", "mar")}
            for vn, _ in VARS:
                r = cc.run_p(t, o, h, l, c, pos[vn], i0, half, bar_h)
                x = {k: r[k] for k in ("ret", "cagr", "mdd", "mar")}
                x.update(tt.trade_stats(tt.trades(t, o, h, l, r["eff"], r["retv"], bar_h)))
                x["expo"] = float(np.mean(r["eff"][i0:] != 0))
                R[(pn, vn)] = x
            for vn in ("CB", "MB"):
                R[(pn, vn, "sh")] = [cc.run_p(t, o, h, l, c, sp[vn], i0, half, bar_h)["mar"] for sp in shp]
        res["tf"][tfn] = R
    return sym, res, "%.0f с" % (time.time() - t_s)


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def main():
    t0 = time.time()
    G["btc"] = btc_regime()
    dk, up, d0, shifts = G["btc"]
    say("РЕЖИМ BTC > SMA200 · %s · SuperTrend ATR %d × %.1f · издержки %.3f%% на сторону%s" % (
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), tt.MAIN[0], tt.MAIN[1], tt.FEE * 100,
        " · !!! СИНТЕТИКА" if tt.SYNTH else ""))
    y22 = int(np.searchsorted(dk, cc.Y22 // D1))
    say("BTC выше SMA200: %.0f%% дней с 2018 · %.0f%% дней с 2022 · сдвиги режима, дней: %s" % (
        100 * up[d0:].mean(), 100 * up[y22:].mean(), " ".join(map(str, shifts))))
    allc = NEW + [x for x in OLD if x not in NEW]
    got = {}
    with mp.get_context("fork").Pool(max(1, min(len(allc), os.cpu_count() or 2))) as pool:
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    sets = [("НОВЫЙ", [s for s in NEW if s in got]), ("СТАРЫЙ", [s for s in OLD if s in got])]

    say()
    say("=" * 118)
    say("ЧАСТЬ 1 — СВОДКА (медианы по монетам). «>B&H» — у скольких MAR выше B&H")
    for sn, coins in sets:
        say("  НАБОР %s (%d монет)" % (sn, len(coins)))
        say("  %-4s %-8s %-30s %5s %8s %6s %6s %6s %7s %6s %6s" % ("ТФ", "период", "вариант", "PF", "CAGR", "MaxDD",
                                                              "MAR", "сделок", "в рынке", "в +", ">B&H"))
        for tfn, _ in TFS:
            for pn, *_ in PERIODS:
                B = [got[s]["tf"][tfn][(pn, "bh")] for s in coins]
                say("  %-4s %-8s %-30s %5s %+7.1f%% %5.1f%% %6.2f %6s %6s %3d/%-2d" % (
                    tfn, pn, "B&H", "", 100 * med([b["cagr"] for b in B]), 100 * med([b["mdd"] for b in B]),
                    med([b["mar"] for b in B]), "", "100%", sum(b["ret"] > 0 for b in B), len(coins)))
                for vn, vl in VARS:
                    X = [got[s]["tf"][tfn][(pn, vn)] for s in coins]
                    say("  %-4s %-8s %-4s %-25s %5.2f %+7.1f%% %5.1f%% %6.2f %6.0f %5.0f%% %3d/%-2d %3d/%-2d" % (
                        tfn, pn, vn, vl, med([x["pf"] for x in X]), 100 * med([x["cagr"] for x in X]),
                        100 * med([x["mdd"] for x in X]), med([x["mar"] for x in X]), med([x["n"] for x in X]),
                        100 * med([x["expo"] for x in X]), sum(x["ret"] > 0 for x in X), len(coins),
                        sum(x["mar"] > b["mar"] for x, b in zip(X, B)), len(coins)))
                say("  " + "-" * 112)

    say()
    say("=" * 118)
    say("ЧАСТЬ 2 — ПО МОНЕТАМ, 4h (итог % · MaxDD %): C | CB | M | MB | BB | B&H")
    for sn, coins in sets:
        say("  НАБОР %s" % sn)
        for s in coins:
            R = got[s]["tf"]["4h"]
            say("  %-5s " % s + "   ‖   ".join("%s: " % pn + " | ".join("%+6.0f %3.0f" % (
                100 * R[(pn, v)]["ret"], 100 * R[(pn, v)]["mdd"]) for v in ("C", "CB", "M", "MB", "BB", "bh"))
                for pn, *_ in PERIODS))

    say()
    say("=" * 118)
    say("ИТОГ ПО КРИТЕРИЮ")
    nn = len(sets[0][1])
    need = int(np.ceil(nn * 2 / 3))
    for hv, base, title in HYP:
        say("  " + title)
        for tfn, _ in TFS:
            okall = True
            parts = []
            for pn, *_ in PERIODS:
                cn = sets[0][1]
                X = [got[s]["tf"][tfn][(pn, hv)] for s in cn]
                Y = [got[s]["tf"][tfn][(pn, base)] for s in cn]
                c1n = sum(x["mar"] > y["mar"] for x, y in zip(X, Y))
                c1 = c1n >= need
                ddx, ddy = med([x["mdd"] for x in X]), med([y["mdd"] for y in Y])
                cgx, cgy = med([x["cagr"] for x in X]), med([y["cagr"] for y in Y])
                c2 = ddx < ddy and cgx >= cgy - 0.02
                real = med([x["mar"] for x in X])
                shm = [med([got[s]["tf"][tfn][(pn, hv, "sh")][k] for s in cn]) for k in range(N_SHIFT)]
                beat = sum(real > v for v in shm)
                c3 = beat >= 10
                co = sets[1][1]
                mo_x = med([got[s]["tf"][tfn][(pn, hv)]["mar"] for s in co])
                mo_y = med([got[s]["tf"][tfn][(pn, base)]["mar"] for s in co])
                c4 = mo_x >= mo_y
                okall &= c1 and c2 and c3 and c4
                parts.append("%s: MAR↑ %d/%d [%s] · DD %.0f%%→%.0f%%, CAGR %+.1f%%→%+.1f%% [%s] · сдвиги %d/12 [%s] · старый MAR %.2f→%.2f [%s]" % (
                    pn, c1n, nn, "да" if c1 else "нет", 100 * ddy, 100 * ddx, 100 * cgy, 100 * cgx, "да" if c2 else "нет",
                    beat, "да" if c3 else "нет", mo_y, mo_x, "да" if c4 else "нет"))
            say("    %s → %s" % (tfn, "ФИЛЬТР BTC ПОЛЕЗЕН" if okall else "не доказан"))
            for p in parts:
                say("       " + p)
    say("Время %.1f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

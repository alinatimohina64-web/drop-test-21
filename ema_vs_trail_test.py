#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ДАЁТ ЛИ ПУНКТИР ХОТЬ ЧТО-ТО СВЕРХ ПРОСТОГО ПРАВИЛА EMA200?
Сравнение на тех же 24 монетах, тех же издержках и исполнении (функции из trend_trail_test.py / c_confirm_test.py).

ВАРИАНТЫ (объявлены до прогона, без подбора):
  C   пунктир: лонг по SuperTrend 10x3, вход только если close > EMA200 (как в прошлом тесте)
  M   только EMA200 этого же ТФ: в лонге, пока close > EMA200; вне рынка, пока ниже
  MD  только дневная SMA200 (классика «выше 200-дневной»): в лонге, пока ВЧЕРАШНИЙ дневной close > SMA200
  B&H купить и держать (справочно)
ТФ 4h и 2h. Периоды: «весь» и «с 2022».

КРИТЕРИЙ (объявлен ДО прогона), отдельно для 4h и 2h:
  ПУНКТИР ПОЛЕЗЕН, если на НОВОМ наборе в ОБОИХ периодах MAR(C) > MAR(M) у >= 2/3 монет
  И медианный MAR(C) > медианного MAR(M) в обоих периодах на ОБОИХ наборах.
  Иначе — пунктир ничего не добавляет к простому фильтру.
  Справочно: простое правило (M или MD) «лучше B&H», если MAR > MAR B&H у >= 2/3 монет нового набора в обоих периодах.
Отчёт: ema_vs_trail_report.txt
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

import c_confirm_test as cc
import trend_trail_test as tt

REPORT = "ema_vs_trail_report.txt"
NEW, OLD = cc.NEW, cc.OLD
TFS = [("4h", 240), ("2h", 120)]
VARS = [("C", "пунктир + EMA200"), ("M", "только EMA200 ТФ"), ("MD", "только SMA200 дневная")]
PERIODS = cc.PERIODS
D1 = tt.D1
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def work(sym):
    t_s = time.time()
    try:
        base, _ = tt.load(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(base) < 20000:
        return sym, None, "мало данных"
    t_first = max(tt.START, int(base[0, 0]) + tt.WARM_D * D1)
    daily = tt.resample(base, 1440)
    dc = daily[:, 4]
    dsma = np.r_[np.full(199, np.nan), np.convolve(dc, np.ones(200) / 200, "valid")]
    with np.errstate(invalid="ignore"):
        d_above = dc > dsma                                      # nan -> False
    dk = (daily[:, 0] // D1).astype(np.int64)
    res = {"tf": {}}
    for tfn, mins in TFS:
        a = tt.resample(base, mins)
        t, o, h, l, c = (a[:, k] for k in range(5))
        bar_h = mins / 60
        above = c > tt.ema(c, 200)
        want = (t // D1).astype(np.int64) - 1                    # вчерашний день
        j = np.clip(np.searchsorted(dk, want, side="right") - 1, 0, len(dk) - 1)
        mdp = (d_above[j] & (dk[j] <= want)).astype(float)
        st = tt.supertrend_dir(h, l, c, *tt.MAIN)
        pos = {"C": tt.positions(st, above, np.zeros(len(c), bool)), "M": above.astype(float), "MD": mdp}
        R = {}
        for pn, p0, half, _, _ in PERIODS:
            i0 = int(np.searchsorted(t, max(t_first, p0 or 0)))
            bh = cc.run_p(t, o, h, l, c, np.ones(len(c)), i0, half, bar_h, funding=False)
            R[(pn, "bh")] = {k: bh[k] for k in ("ret", "cagr", "mdd", "mar")}
            for vn, _ in VARS:
                r = cc.run_p(t, o, h, l, c, pos[vn], i0, half, bar_h)
                x = {k: r[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2")}
                x.update(tt.trade_stats(tt.trades(t, o, h, l, r["eff"], r["retv"], bar_h)))
                x["expo"] = float(np.mean(r["eff"][i0:] != 0))
                R[(pn, vn)] = x
        res["tf"][tfn] = R
    return sym, res, "%.0f с" % (time.time() - t_s)


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def main():
    t0 = time.time()
    say("ПУНКТИР vs EMA200 · %s · SuperTrend ATR %d × %.1f · издержки %.3f%% на сторону%s" % (
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), tt.MAIN[0], tt.MAIN[1], tt.FEE * 100,
        " · !!! СИНТЕТИКА" if tt.SYNTH else ""))
    allc = NEW + [x for x in OLD if x not in NEW]
    got = {}
    with mp.get_context("fork").Pool(max(1, min(len(allc), os.cpu_count() or 2))) as pool:
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    sets = [("НОВЫЙ", [s for s in NEW if s in got]), ("СТАРЫЙ", [s for s in OLD if s in got])]
    M = {}
    say()
    say("=" * 118)
    say("ЧАСТЬ 1 — СВОДКА (медианы по монетам). «C>M» — у скольких монет MAR пунктира выше MAR простого EMA200; "
        "«>B&H» — у скольких MAR варианта выше B&H")
    for sn, coins in sets:
        say("  НАБОР %s (%d монет)" % (sn, len(coins)))
        say("  %-4s %-8s %-24s %5s %8s %6s %6s %6s %7s %6s %7s %6s" % ("ТФ", "период", "вариант", "PF", "CAGR", "MaxDD",
            "MAR", "сделок", "в рынке", "в +", ">B&H", "C>M"))
        for tfn, _ in TFS:
            for pn, *_ in PERIODS:
                B = [got[s]["tf"][tfn][(pn, "bh")] for s in coins]
                say("  %-4s %-8s %-24s %5s %+7.1f%% %5.1f%% %6.2f %6s %6s %3d/%-2d" % (
                    tfn, pn, "B&H", "", 100 * med([b["cagr"] for b in B]), 100 * med([b["mdd"] for b in B]),
                    med([b["mar"] for b in B]), "", "100%", sum(b["ret"] > 0 for b in B), len(coins)))
                cm = sum(got[s]["tf"][tfn][(pn, "C")]["mar"] > got[s]["tf"][tfn][(pn, "M")]["mar"] for s in coins)
                M[(sn, tfn, pn, "C>M")] = cm
                for vn, vl in VARS:
                    X = [got[s]["tf"][tfn][(pn, vn)] for s in coins]
                    M[(sn, tfn, pn, vn)] = med([x["mar"] for x in X])
                    M[(sn, tfn, pn, vn, "bh")] = sum(x["mar"] > b["mar"] for x, b in zip(X, B))
                    say("  %-4s %-8s %-3s %-20s %5.2f %+7.1f%% %5.1f%% %6.2f %6.0f %5.0f%% %3d/%-2d %3d/%-2d %s" % (
                        tfn, pn, vn, vl, med([x["pf"] for x in X]), 100 * med([x["cagr"] for x in X]),
                        100 * med([x["mdd"] for x in X]), M[(sn, tfn, pn, vn)], med([x["n"] for x in X]),
                        100 * med([x["expo"] for x in X]), sum(x["ret"] > 0 for x in X), len(coins),
                        M[(sn, tfn, pn, vn, "bh")], len(coins), ("%3d/%d" % (cm, len(coins))) if vn == "C" else ""))
                say("  " + "-" * 112)

    say()
    say("=" * 118)
    say("ЧАСТЬ 2 — ПО МОНЕТАМ, 4h (итог % · MaxDD %): C | M | MD | B&H")
    for sn, coins in sets:
        say("  НАБОР %s" % sn)
        for s in coins:
            cells = []
            for pn, *_ in PERIODS:
                R = got[s]["tf"]["4h"]
                cells.append("%s: " % pn + " | ".join("%+6.0f %3.0f" % (100 * R[(pn, v)]["ret"], 100 * R[(pn, v)]["mdd"])
                                                      for v in ("C", "M", "MD", "bh")))
            say("  %-5s " % s + "   ‖   ".join(cells))

    say()
    say("=" * 118)
    nn = len(sets[0][1])
    need = int(np.ceil(nn * 2 / 3))
    for tfn, _ in TFS:
        a = all(M[("НОВЫЙ", tfn, p, "C>M")] >= need for p, *_ in PERIODS)
        b = all(M[(s, tfn, p, "C")] > M[(s, tfn, p, "M")] for s, _ in sets for p, *_ in PERIODS)
        say("→ %s · пунктир лучше EMA200 у >= %d/%d новых монет в обоих периодах [%s: %s] · медиана выше везде [%s] → %s" % (
            tfn, need, nn, "да" if a else "нет", " / ".join("%s %d" % (p, M[("НОВЫЙ", tfn, p, "C>M")]) for p, *_ in PERIODS),
            "да" if b else "нет", "ПУНКТИР ПОЛЕЗЕН" if a and b else "пунктир ничего не добавляет к EMA200"))
        for v in ("M", "MD"):
            ok = all(M[("НОВЫЙ", tfn, p, v, "bh")] >= need for p, *_ in PERIODS)
            say("  справочно %s лучше B&H по MAR у новых монет: %s → %s" % (
                v, " / ".join("%s %d/%d" % (p, M[("НОВЫЙ", tfn, p, v, "bh")], nn) for p, *_ in PERIODS),
                "да" if ok else "нет"))
    say("Время %.1f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

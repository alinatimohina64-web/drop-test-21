#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МЕДВЕЖЬЯ ФАЗА: КОГДА BTC НИЖЕ SMA200 — ТОЛЬКО ШОРТЫ. Работает ли зеркальное правило?

Сигналы — TT2 (30m, ст. ТФ 4ч, без MRC), только ШОРТ, только по тренду BTC 1Д вниз и только когда BTC вчера
закрылся НИЖЕ своей дневной SMA200. Одна позиция на монету. Binance 15m, комиссия 0.11%, фандинг 0.01% / 8 ч
(против позиции — в медвежке шорты часто получают фандинг, тест считает худший случай), в спорной свече — стоп.
Три набора: основной (21), новый (20), третий (40).

ВАРИАНТЫ (объявлены до прогона) — 12 = выход × подвижность:
  выход:  цель 5 / стоп 20 ATR (как в лонгах) · цель 3 / стоп 10 · цель 3 / стоп 5 · цель 6 / стоп 5
  подвижность (ATR 30m): любая · ≥ 1.0% · ≥ 1.5%   (при стопе 20 ATR это «стоп ≥ 20% / 30%»)
Случайный шорт: те же монеты, случайные моменты из тех же периодов (BTC ниже SMA200, тренд вниз), тот же
выход и фильтр подвижности.

КРИТЕРИЙ (объявлен ДО прогона). Вариант проходит, если на ВСЕХ ТРЁХ наборах средний R в плюсе в ОБЕИХ половинах
периода И лучше случайного шорта. Шорт-правило принимается, только если прошёл хотя бы один вариант И в плюсе
(по среднему R на всех трёх наборах) не меньше 2/3 вариантов — чтобы не взять один случайно удачный из 12.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: sma200_short_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

c, cw = cs.c, cs.cw
REPORT = "sma200_short_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
EXITS = [(5.0, 20.0), (3.0, 10.0), (3.0, 5.0), (6.0, 5.0)]
VOLS = [0.0, 1.0, 1.5]
VARS = [(tp, sl, v) for tp, sl in EXITS for v in VOLS]
RND_N = 1500
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(x):
    tp, sl, v = x
    return "цель %g / стоп %g · ATR %s" % (tp, sl, ("≥ %.1f%%" % v) if v else "любой")


def main():
    t0 = time.time()
    say("ШОРТЫ НИЖЕ SMA200 · %s · Binance 15m" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    ok = {x: True for x in VARS}
    pos_all = {x: True for x in VARS}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res = {x: [] for x in VARS}
        rnd = {x: [] for x in VARS}
        first_t, end_t = None, 0
        for sym in coins:
            b15 = cs.load_15m(sym)
            if len(b15) < 20000:
                say("  %s: мало данных — пропуск" % sym)
                continue
            t = np.array([x[0] for x in b15], dtype=np.int64)
            o = np.array([x[1] for x in b15])
            h = np.array([x[2] for x in b15])
            l = np.array([x[3] for x in b15])
            cl = np.array([x[4] for x in b15])
            end_t = max(end_t, int(t[-1]))
            b30 = c.agg(b15, M30)
            b30t = [x[0] for x in b30]
            atrp = c.atr_pct(b30)
            h4a = c.agg(b15, H4)
            c4 = [x[4] for x in h4a]
            hu = [a_ > b_ for a_, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
            d1 = c.agg(b15, D1)
            dt = [x[0] for x in d1]
            raw = []
            for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
                if d != -1 or T < st.START or atrp[i] <= 0 or btc.trend(T) != -1 or btc.above(T):
                    continue
                if bisect.bisect_right(dt, T - D1) - 1 < 250:
                    continue
                raw.append((T, atrp[i]))
            for x in VARS:
                tp, sl, v = x
                busy = 0
                for T, ap in raw:
                    if T < busy or ap < v:
                        continue
                    k0 = int(np.searchsorted(t, T, side="left"))
                    if k0 >= len(t) - 1:
                        continue
                    a = ap / 100
                    pnl, r, te = cs.trade(t, h, l, cl, k0, -1, o[k0], tp * a, sl * a)
                    res[x].append((T, r))
                    first_t = T if first_t is None else min(first_t, T)
                    busy = te if te is not None else 10 ** 15
            y0 = max(st.START, int(t[0]) + 260 * D1)
            y1 = int(t[-1]) - 5 * D1
            rg = random.Random("short-%s" % sym)
            got = tries = 0
            while got < RND_N and tries < RND_N * 40 and y1 > y0:
                tries += 1
                T = rg.randrange(y0, y1)
                if btc.trend(T) != -1 or btc.above(T):
                    continue
                j = bisect.bisect_right(b30t, T - M30) - 1
                if j < 20 or atrp[j] <= 0:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                got += 1
                ap = atrp[j]
                for x in VARS:
                    tp, sl, v = x
                    if ap >= v:
                        rnd[x].append(cs.trade(t, h, l, cl, k0, -1, o[k0], tp * ap / 100, sl * ap / 100)[1])
            say("  [%s] %s: шорт-сигналов ниже SMA200 %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
        if first_t is None:
            say("  нет сделок")
            continue
        mid = (first_t + end_t) // 2
        say("")
        say("  половины: до / после %s" % st.dstr(mid))
        for x in VARS:
            rs = [q[1] for q in res[x]]
            r1 = [q[1] for q in res[x] if q[0] < mid]
            r2 = [q[1] for q in res[x] if q[0] >= mid]
            a, a1, a2 = (np.mean(rs) if rs else 0), (np.mean(r1) if r1 else 0), (np.mean(r2) if r2 else 0)
            ra = np.mean(rnd[x]) if rnd[x] else 0
            yrs = {}
            for q in res[x]:
                yrs[cs.year_of(q[0])] = yrs.get(cs.year_of(q[0]), 0) + q[1]
            good = len(rs) > 0 and a1 > 0 and a2 > 0 and a > ra
            ok[x] = ok[x] and good
            pos_all[x] = pos_all[x] and a > 0
            say("    %-30s сделок %4d · WR %4.1f%% · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+6.1f · "
                "случайный шорт %+.3f → %s"
                % (vname(x), len(rs), 100 * sum(1 for q in rs if q > 0) / max(1, len(rs)), a, a1, a2, sum(rs), ra,
                   "да" if good else "нет"))
            say("        по годам (сумма R): " + " · ".join("%d: %+.1f" % (y, s_) for y, s_ in sorted(yrs.items())))
    say("")
    say("=" * 110)
    passed = [x for x in VARS if ok[x]]
    npos = sum(1 for x in VARS if pos_all[x])
    say("Прошли все три набора: %s" % (", ".join(vname(x) for x in passed) if passed else "ни один"))
    say("В плюсе на всех трёх наборах: %d из %d вариантов (нужно не меньше %d)" % (npos, len(VARS), (2 * len(VARS) + 2) // 3))
    accept = bool(passed) and npos * 3 >= 2 * len(VARS)
    say("→ ИТОГ: %s" % ("ШОРТ-ПРАВИЛО ДЛЯ МЕДВЕЖКИ ЕСТЬ" if accept else "шорт-правила ниже SMA200 нет — в медвежке стоим в стороне"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

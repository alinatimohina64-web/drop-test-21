#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОСЛЕДНЯЯ ПРОВЕРКА ШОРТОВ: «настоящая» медвежка + фильтр MRC для шортов + подвижность — то, что ещё не стояло вместе.

Сигнал — шорт TT2 (30m, ст. ТФ 4ч вниз, тренд BTC 1Д вниз), только когда BTC ниже SMA200 И SMA200 за 30 дней пошла
вниз («наклон»). Одна позиция на монету. Binance 15m, комиссия 0.11%, фандинг 0.01% / 8 ч против позиции,
в спорной свече — стоп. Три набора монет: основной (21), новый (20), третий (40).

ВАРИАНТЫ (18, объявлены до прогона):
  MRC для шортов (не шортить, когда монета уже у нижней границы дневного канала — перепродана):
      нет · уровень 2 (упрощённая схема) · уровень 3 (внешняя граница);
  подвижность (ATR 30m): любая · ≥ 1.0% · ≥ 1.5%;
  выход: цель 5 / стоп 20 ATR · цель 3 / стоп 10 ATR (как в эксперименте «🐻 рискованный шорт»).
Случайный шорт: те же монеты, случайные моменты той же медвежки, тот же выход и фильтр подвижности.

КРИТЕРИЙ (объявлен ДО прогона). Вариант проходит, если на ВСЕХ ТРЁХ наборах средний R в плюсе в ОБЕИХ половинах
периода И лучше случайного шорта. Правило принимается, только если прошёл хотя бы один вариант И в плюсе (по среднему
R на всех трёх наборах) не меньше 2/3 вариантов. Если не принято — тема шортов закрыта без оговорок.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: bear_final_short_report.txt
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
REPORT = "bear_final_short_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
EXITS = [(5.0, 20.0), (3.0, 10.0)]
VOLS = [0.0, 1.0, 1.5]
MRCS = ["none", "simple", "simple3"]
MRC_NAME = {"none": "MRC нет", "simple": "MRC ур.2", "simple3": "MRC ур.3"}
VARS = [(tp, sl, m, v) for tp, sl in EXITS for m in MRCS for v in VOLS]
RND_N = 1500
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(x):
    tp, sl, m, v = x
    return "цель %g/стоп %g · %s · ATR %s" % (tp, sl, MRC_NAME[m], ("≥ %.1f%%" % v) if v else "любой")


def main():
    t0 = time.time()
    say("ПОСЛЕДНЯЯ ПРОВЕРКА ШОРТОВ: наклон SMA200 + MRC + подвижность · %s · Binance 15m" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    s50, s200 = c.sma(btc.cl, 50), c.sma(btc.cl, 200)

    def bear(T, rg):
        i = btc.idx(T)
        if i < 230 or s200[i] is None or s50[i] is None or s200[i - 30] is None:
            return False
        if not btc.cl[i] < s200[i] or btc.trend(T) != -1:
            return False
        return s200[i] < s200[i - 30] if rg == "НАКЛОН" else s50[i] < s200[i]
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
            mean = cs.supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
            rngd = cs.supersmoother(c.trs(d1), 200)
            raw = []
            for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
                if d != -1 or T < st.START or atrp[i] <= 0 or not bear(T, "НАКЛОН"):
                    continue
                jd = bisect.bisect_right(dt, T - D1) - 1
                if jd < 250:
                    continue
                raw.append((T, atrp[i], b30[i][4], mean[jd], rngd[jd]))
            for x in VARS:
                tp, sl, m, v = x
                busy = 0
                for T, ap, px, mn, rg_ in raw:
                    if T < busy or ap < v or cs.hot(m, -1, px, mn, rg_):
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
            rg = random.Random("bear-final-%s" % sym)
            got = tries = 0
            while got < RND_N and tries < RND_N * 40 and y1 > y0:
                tries += 1
                T = rg.randrange(y0, y1)
                if not bear(T, "НАКЛОН"):
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
                    tp, sl, m, v = x
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
    say("→ ИТОГ: %s" % ("ШОРТ-ПРАВИЛО НАЙДЕНО" if accept else "шорты не подтвердились — тема закрыта"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200: КАКАЯ ЦЕЛЬ ЛУЧШЕ — 3, 4, 5, 6, 7 или 8 × ATR 30m (стоп 20 × ATR).

Сигналы — как в sma200_vol_test.py: TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC вчера
закрылся выше SMA200, стоп 20 × ATR не ближе ПОРОГА (20% и 30%), одна позиция на монету (очередь по выходу
цели 4 — текущей). Все цели считаются на ОДНИХ И ТЕХ ЖЕ входах, разница — только от цели.
Цель и стоп в ATR подстраиваются под подвижность каждой монеты сами. Binance 15m, комиссия 0.11%, фандинг 0.01% / 8 ч.
Три набора: основной (21), новый (20), третий (40, из топа).

КРИТЕРИЙ (объявлен ДО прогона): цель X заменяет 4, если для порога 30% на ВСЕХ ТРЁХ наборах средний R на сделку
у X выше, чем у 4, в ОБЕИХ половинах периода. Если таких несколько — берётся та, у которой сумма R по трём
наборам больше. Иначе остаётся 4.
Справочно: порог 20%; WR; сколько дней сделка держит деньги (медиана и доля дольше 30 дней) — дальняя цель
дольше занимает слот из 3 позиций.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: sma200_target_report.txt
"""
import bisect
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

c, cw = cs.c, cs.cw
REPORT = "sma200_target_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
SL = 20.0
TPS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
BASE_TP = 4.0
THRS = [20.0, 30.0]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def run_set(set_name, coins, btc):
    t0 = time.time()
    res = {(thr, tp): [] for thr in THRS for tp in TPS}     # (T, R, дней)
    end_t = 0
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
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a_ > b_ for a_, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        raw = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < st.START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                continue
            if bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            raw.append((T, d, atrp[i] / 100))
        for thr in THRS:
            busy = 0
            for T, d, a in raw:
                if T < busy or SL * a * 100 < thr:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                te_base = None
                for tp in TPS:
                    pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], tp * a, SL * a)
                    days = ((te if te is not None else int(t[-1])) - T) / D1
                    res[(thr, tp)].append((T, r, days))
                    if tp == BASE_TP:
                        te_base = te
                busy = te_base if te_base is not None else 10 ** 15
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return res, end_t


def main():
    t0 = time.time()
    say("ПРАВИЛО SMA200: ЦЕЛЬ 3–8 × ATR (стоп 20 × ATR) · %s · Binance 15m"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    better = {tp: True for tp in TPS if tp != BASE_TP}
    sumR = {tp: 0.0 for tp in TPS}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res, end_t = run_set(set_name, coins, btc)
        for thr in THRS:
            base = res[(thr, BASE_TP)]
            if not base:
                continue
            mid = (base[0][0] + end_t) // 2
            b1 = np.mean([x[1] for x in base if x[0] < mid])
            b2 = np.mean([x[1] for x in base if x[0] >= mid])
            say("")
            say("  ПОРОГ СТОПА %.0f%% (сделок %d)" % (thr, len(base)))
            for tp in TPS:
                v = res[(thr, tp)]
                rs = np.array([x[1] for x in v])
                r1 = np.mean([x[1] for x in v if x[0] < mid])
                r2 = np.mean([x[1] for x in v if x[0] >= mid])
                dd = sorted(x[2] for x in v)
                mark = ""
                if tp != BASE_TP:
                    good = r1 > b1 and r2 > b2
                    mark = " → лучше цели 4 в обеих половинах: %s" % ("да" if good else "нет")
                    if thr == THRS[-1]:
                        better[tp] = better[tp] and good
                if thr == THRS[-1]:
                    sumR[tp] += rs.sum()
                say("    цель %g: средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f · WR %4.1f%% · держит деньги: "
                    "медиана %.1f дн, дольше 30 дн %.0f%%%s"
                    % (tp, rs.mean(), r1, r2, rs.sum(), 100 * (rs > 0).mean(), dd[len(dd) // 2],
                       100 * sum(1 for x in dd if x > 30) / max(1, len(dd)), mark))
    say("")
    say("=" * 110)
    ok = [tp for tp, g in better.items() if g]
    for tp in TPS:
        if tp != BASE_TP:
            say("цель %g: %s (сумма R по трём наборам при пороге 30%%: %+.1f против %+.1f у цели 4)"
                % (tp, "лучше цели 4 на всех наборах" if better[tp] else "не лучше", sumR[tp], sumR[BASE_TP]))
    if ok:
        best = max(ok, key=lambda x: sumR[x])
        say("→ РЕШЕНИЕ: цель %g × ATR вместо 4" % best)
    else:
        say("→ РЕШЕНИЕ: оставить цель 4 × ATR")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

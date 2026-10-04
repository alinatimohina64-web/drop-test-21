#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200: ПОРОГ СТОПА 20 / 25 / 30% — «золотая середина» 25%?

Сигналы и сделки — как в нашем правиле: TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC вчера
закрылся выше SMA200, цель 5 / стоп 20 × ATR (стоп не дальше −80%), сигнал берётся, только если стоп ≥ ПОРОГА.
Binance 15m, комиссия 0.11%, фандинг 0.01% / 8 ч, в спорной свече — стоп. Три набора: основной, новый, третий.

ЧАСТЬ 1 — качество сделки: средний R по наборам и половинам периода (одна позиция на монету).
ЧАСТЬ 2 — как в жизни: все 81 монета на одном счёте, сделки по очереди, до 3 и до 5 позиций; R в год, 2-я половина,
          худший год, просадка в R.

КРИТЕРИЙ (объявлен ДО прогона): порог 25% становится ОСНОВНЫМ вместо 30%, если
  1) в портфеле (81 монета, до 5 позиций) сумма R за весь период И за 2-ю половину у 25% не меньше, чем у 30%;
  2) средний R сделки у 25% выше, чем у 20%, в обеих половинах на ВСЕХ ТРЁХ наборах (качество лучше, чем у 20%).
Иначе основным остаётся 30% (20% — запасной, как сейчас).
Лежит рядом с sma200_test.py, sma200_vol_test.py, sma200_check_test.py, coin_select_test.py. Отчёт: sma200_thr25_report.txt
"""
import bisect
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_check_test as ck
import sma200_test as st
import sma200_vol_test as sv

c, cw = cs.c, cs.cw
REPORT = "sma200_thr25_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
TP, SL, MAXSTOP = 5.0, 20.0, 80.0
THRS = [20.0, 25.0, 30.0]
CAPS = [3, 5]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    say("ПОРОГ СТОПА 20 / 25 / 30%% · %s · Binance 15m · цель %g / стоп %g ATR"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), TP, SL))
    btc = st.Btc()
    seq = {}                       # (набор, порог) -> [(T, R)]   одна позиция на монету
    pool = {thr: [] for thr in THRS}   # все сигналы для портфеля
    first_t, end_t = None, 0
    sets = (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS))
    for set_name, names in sets:
        for thr in THRS:
            seq[(set_name, thr)] = []
        for sym in names:
            b15 = cs.load_15m(sym)
            if len(b15) < 20000:
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
            sigs = []
            for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
                if T < st.START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                    continue
                if bisect.bisect_right(dt, T - D1) - 1 < 250:
                    continue
                a = atrp[i] / 100
                if SL * a * 100 < THRS[0]:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                slp = min(SL * a, MAXSTOP / 100)
                pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], TP * a, slp)
                sigs.append((T, pnl, r, te, SL * a * 100))
                first_t = T if first_t is None else min(first_t, T)
            for thr in THRS:
                busy = 0
                for T, pnl, r, te, sp in sigs:
                    if sp < thr:
                        continue
                    pool[thr].append((T, pnl, r, te, sym, 1, 0.0, 0.0))
                    if T < busy:
                        continue
                    seq[(set_name, thr)].append((T, r))
                    busy = te if te is not None else 10 ** 15
            say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(sigs), time.time() - t0))
    mid = (first_t + end_t) // 2
    years = (end_t - first_t) / (365.25 * D1)
    say("")
    say("=" * 110)
    say("ЧАСТЬ 1 — КАЧЕСТВО СДЕЛКИ (одна позиция на монету) · половины до / после %s" % st.dstr(mid))
    q = {}
    for set_name, _ in sets:
        say("  НАБОР «%s»" % set_name)
        for thr in THRS:
            v = seq[(set_name, thr)]
            rs = [x[1] for x in v]
            r1 = [x[1] for x in v if x[0] < mid]
            r2 = [x[1] for x in v if x[0] >= mid]
            q[(set_name, thr)] = (np.mean(r1) if r1 else 0, np.mean(r2) if r2 else 0)
            say("    стоп ≥ %2.0f%%: сделок %4d · WR %4.1f%% · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f"
                % (thr, len(rs), 100 * sum(1 for x in rs if x > 0) / max(1, len(rs)), np.mean(rs) if rs else 0,
                   q[(set_name, thr)][0], q[(set_name, thr)][1], sum(rs)))
    say("")
    say("=" * 110)
    say("ЧАСТЬ 2 — ПОРТФЕЛЬ: 81 монета, %.1f лет, R в год (при риске 50 $ ≈ × 50 $)" % years)
    port = {}
    for cap in CAPS:
        for thr in THRS:
            tr = ck.capped(pool[thr], cap)
            rs = [x[2] for x in tr]
            yrs = {}
            for x in tr:
                yrs[cs.year_of(x[0])] = yrs.get(cs.year_of(x[0]), 0) + x[2]
            eq = pk = dd = 0.0
            for x in sorted(tr, key=lambda x: x[3] if x[3] is not None else 10 ** 15):
                eq += x[2]
                pk = max(pk, eq)
                dd = min(dd, eq - pk)
            h2 = sum(x[2] for x in tr if x[0] >= mid)
            port[(cap, thr)] = (sum(rs), h2)
            say("  до %d позиций · стоп ≥ %2.0f%%: сделок %4d (%.0f в год) · сумма R %+6.1f (%+.1f R в год ≈ %+.0f $) · 2-я половина %+.1f R · "
                "худший год %+.1f R · просадка %.1f R" % (cap, thr, len(tr), len(tr) / years, sum(rs), sum(rs) / years,
                                                         sum(rs) / years * 50, h2, min(yrs.values()) if yrs else 0, dd))
    say("")
    say("=" * 110)
    lo, md, hi = THRS
    k1 = port[(5, md)][0] >= port[(5, hi)][0] and port[(5, md)][1] >= port[(5, hi)][1]
    k2 = all(q[(s, md)][0] > q[(s, lo)][0] and q[(s, md)][1] > q[(s, lo)][1] for s, _ in sets)
    say("КРИТЕРИЙ: портфель 25%% не хуже 30%% [%s: %+.1f / %+.1f против %+.1f / %+.1f] · качество 25%% лучше 20%% на всех наборах [%s]"
        % ("да" if k1 else "нет", port[(5, md)][0], port[(5, md)][1], port[(5, hi)][0], port[(5, hi)][1],
           "да" if k2 else "нет"))
    say("→ ИТОГ: %s" % ("ПОРОГ 25% — ОСНОВНОЙ" if (k1 and k2) else "основным остаётся 30% (20% — запасной)"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

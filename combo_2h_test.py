#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОМБИНАЦИЯ: текущее правило (A) + подтверждённая стратегия 2ч/1Д 6/7 (B) на отдельных слотах.

A — 30m / 4ч, 5/20 ATR, ⭐, 5 слотов, риск 50 $.
B — 2ч / 1Д, цель 6 / стоп 7 ATR(2ч), без ⭐, правило SMA200, свои 5 слотов, риск RB $ (15 или 25).
Подтверждение B (confirm_2h_test.py, 24.11.2026): перевес над случайным +0.07…+0.12 R на 4 независимых наборах,
контроль чистый; одна B по доходу/просадке хуже A (2.35 против 3.30 на четвёртом наборе).

КРИТЕРИЙ (объявлен ДО прогона). Комбинация A + B(RB) принимается, если на ВСЕХ ЧЕТЫРЁХ наборах:
  1) итог суммы в $ больше итога A;
  2) доход/просадка суммы (по $ кривой закрытых сделок) не хуже, чем у A.
Медиана по 50 случайным порядкам равных сигналов. Справочно: корреляция месячных $ A и B, максимум
одновременных позиций A+B.

Запуск: python combo_2h_test.py (кэш data_btc/). Отчёт: combo_2h_report.txt
"""
import time
from datetime import datetime, timezone

import sma200_test as st
import tf_test as tt
import tf2h_grid_test as g
import confirm_2h_test as cf

cs = st.cs
REPORT = "combo_2h_report.txt"
g.VARS = [g.BASE, ("g", g.H2, g.D1, 6.0, 7.0, 0.0)]
RA = 50.0
RBS = (15.0, 25.0)
SCEN = 50
SETS = [("основной", list(st.MAIN_COINS)), ("новый", list(st.NEW_COINS)), ("третий", tt.THIRD),
        ("четвёртый", cf.FOURTH)]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def collect(coins, btc):
    A, B, end_t = [], [], 0
    for sym in coins:
        try:
            b15 = cs.load_15m(sym)
        except Exception:                                        # noqa: BLE001
            continue
        if len(b15) < 20000:
            continue
        res, e = g.prep(sym, b15, btc)
        end_t = max(end_t, e)
        A.extend(res[0][0])
        B.extend(res[1][0])
    return A, B, end_t


def events(taken, risk, end_t):
    return [((s["te"] if s["te"] is not None else end_t), s["r"] * risk, s["T"]) for s in taken]


def stats(ev):
    ev = sorted(ev)
    cum = peak = dd = 0.0
    for _, x, _ in ev:
        cum += x
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return cum, dd


def monthly(ev):
    m = {}
    for t, x, _ in ev:
        k = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m")
        m[k] = m.get(k, 0.0) + x
    return m


def corr(a, b):
    ks = sorted(set(a) | set(b))
    if len(ks) < 3:
        return float("nan")
    x = [a.get(k, 0.0) for k in ks]
    y = [b.get(k, 0.0) for k in ks]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((v - mx) ** 2 for v in x) ** 0.5
    sy = sum((v - my) ** 2 for v in y) ** 0.5
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def maxpos(ta, tb, end_t):
    ev = []
    for s in ta + tb:
        ev.append((s["T"], 1))
        ev.append(((s["te"] if s["te"] is not None else end_t), -1))
    cur = best = 0
    for _, d in sorted(ev):
        cur += d
        best = max(best, cur)
    return best


def main():
    t0 = time.time()
    say("КОМБИНАЦИЯ A (30m/4ч ⭐, риск 50 $) + B (2ч/1Д 6/7, риск 15 / 25 $) · %s"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    verdict = {rb: True for rb in RBS}
    for name, coins in SETS:
        A, B, end_t = collect(coins, btc)
        if not A or not B:
            say("  %s: нет сигналов" % name)
            for rb in RBS:
                verdict[rb] = False
            continue
        rows = {"A": [], **{rb: [] for rb in RBS}}
        cors, mps = [], []
        for sd in range(SCEN):
            ta = tt.portfolio(A, sd)
            tb = tt.portfolio(B, 1000 + sd)
            ea = events(ta, RA, end_t)
            rows["A"].append(stats(ea))
            for rb in RBS:
                rows[rb].append(stats(ea + events(tb, rb, end_t)))
            if sd == 0:
                cors.append(corr(monthly(ea), monthly(events(tb, 1.0, end_t))))
                mps.append(maxpos(ta, tb, end_t))
        say("")
        say("=" * 100)
        say("НАБОР «%s» · корреляция месячных результатов A и B %.2f · позиций одновременно до %d"
            % (name, cors[0], mps[0]))
        a_tot = tt.med([r[0] for r in rows["A"]])
        a_dd = tt.med([r[1] for r in rows["A"]])
        a_ratio = a_tot / -a_dd if a_dd < 0 else float("inf")
        say("  только A              итог %+7.0f $ · просадка %+6.0f $ · доход/просадка %.2f" % (a_tot, a_dd, a_ratio))
        for rb in RBS:
            tot = tt.med([r[0] for r in rows[rb]])
            dd = tt.med([r[1] for r in rows[rb]])
            ratio = tot / -dd if dd < 0 else float("inf")
            ok = tot > a_tot and ratio >= a_ratio
            verdict[rb] = verdict[rb] and ok
            say("  A + B (риск B %2.0f $)  итог %+7.0f $ · просадка %+6.0f $ · доход/просадка %.2f → %s"
                % (rb, tot, dd, ratio, "лучше A" if ok else "не лучше"))
    say("")
    say("=" * 100)
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона)")
    for rb in RBS:
        say("  A + B с риском B %.0f $: %s" % (rb, "ПРИНЯТЬ — лучше A на всех 4 наборах" if verdict[rb]
                                              else "не принимать"))
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

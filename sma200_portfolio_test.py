#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200 КАК В ЖИЗНИ: ОБЩИЙ СЧЁТ, ЛИМИТ ПОЗИЦИЙ, СДЕЛКИ ПО ОЧЕРЕДИ.

Вопросы: какая цель даёт больше денег В ГОД, когда слотов мало (длинная цель дольше держит слот)?
Сколько сделок в год получится? Даёт ли больше монет больше сделок и денег?

Сигналы — как в sma200_vol_test.py: TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC вчера закрылся
выше SMA200, стоп 20 × ATR не ближе ПОРОГА (20% / 30%). Цели: 4, 5, 7 × ATR. Binance 15m, комиссия 0.11%,
фандинг 0.01% / 8 ч. Все сигналы всех монет идут по времени; вход берётся, если монета свободна и открытых позиций
меньше ЛИМИТА (3 / 5). Позиция от риска: 1 R = стоп. Результат в R; при риске 50 $ умножить на 50.
Вселенные: «21» (основной набор) и «81» (основной + новый + третий).

КРИТЕРИЙ (объявлен ДО прогона): цель 7 (победитель sma200_target_test.py) идёт в живую торговлю, если на вселенной
81 монета при лимите 3 и пороге 30% сумма R за весь период И сумма R во второй половине у неё не меньше, чем у цели 4,
и обе в плюсе (иначе «лучше» значит лишь «меньше в минусе»).
Иначе — та из целей 5 / 4, что проходит то же условие против 4 (5 против 4), иначе остаётся 4.
Справочно: лимит 5, порог 20%, вселенная 21; сделок в год, R в год, худший год, просадка в R (по закрытым).
Лежит рядом с sma200_test.py, sma200_vol_test.py, sma200_check_test.py, coin_select_test.py. Отчёт: sma200_portfolio_report.txt
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
REPORT = "sma200_portfolio_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
SL = 20.0
TPS = [4.0, 5.0, 7.0]
THRS = [20.0, 30.0]
CAPS = [3, 5]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    say("ПРАВИЛО SMA200 ПОРТФЕЛЕМ · %s · Binance 15m" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    main_set = set(st.MAIN_COINS)
    allsig = {(thr, tp): [] for thr in THRS for tp in TPS}     # (T, $, R, t_выхода, монета, ...)
    end_t = 0
    first_t = None
    for sym in st.MAIN_COINS + st.NEW_COINS + sv.THIRD_COINS:
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
        n = 0
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
            n += 1
            first_t = T if first_t is None else min(first_t, T)
            for tp in TPS:
                pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], tp * a, SL * a)
                for thr in THRS:
                    if SL * a * 100 >= thr:
                        allsig[(thr, tp)].append((T, pnl, r, te, sym, d, 0.0, 0.0))
        say("  %s: сигналов по правилу %d · %.0f с" % (sym, n, time.time() - t0))

    years = (end_t - first_t) / (365.25 * D1)
    mid = (first_t + end_t) // 2
    say("")
    say("Период %.1f лет (%s … %s)" % (years, st.dstr(first_t), st.dstr(end_t)))
    totals = {}
    for uni_name, uni in (("81 монета", None), ("21 монета", main_set)):
        for thr in THRS:
            for cap in CAPS:
                say("")
                say("=" * 110)
                say("ВСЕЛЕННАЯ %s · ПОРОГ СТОПА %.0f%% · ДО %d ПОЗИЦИЙ" % (uni_name, thr, cap))
                for tp in TPS:
                    src = [x for x in allsig[(thr, tp)] if uni is None or x[4] in uni]
                    tr = ck.capped(src, cap)
                    rs = [x[2] for x in tr]
                    tot = sum(rs)
                    h2 = sum(x[2] for x in tr if x[0] >= mid)
                    yrs = {}
                    for x in tr:
                        yrs[cs.year_of(x[0])] = yrs.get(cs.year_of(x[0]), 0) + x[2]
                    eq = pk = dd = 0.0
                    for x in sorted(tr, key=lambda x: x[3] if x[3] is not None else 10 ** 15):
                        eq += x[2]
                        pk = max(pk, eq)
                        dd = min(dd, eq - pk)
                    totals[(uni_name, thr, cap, tp)] = (tot, h2)
                    say("  цель %g: сделок %4d (%.0f в год) · WR %4.1f%% · сумма R %+6.1f (%+.1f R в год; при риске 50 $ ≈ %+.0f $ в год) · "
                        "2-я половина %+.1f R · худший год %+.1f R · просадка %.1f R"
                        % (tp, len(tr), len(tr) / years, 100 * sum(1 for r in rs if r > 0) / max(1, len(rs)), tot,
                           tot / years, tot / years * 50, h2, min(yrs.values()) if yrs else 0, dd))
                    say("      по годам: " + " · ".join("%d: %+.1f" % (y, v) for y, v in sorted(yrs.items())))
    say("")
    say("=" * 110)
    th = THRS[-1]                                           # строгий порог (30%)
    base = totals[("81 монета", th, 3, 4.0)]
    choice = 4.0
    for tp in (7.0, 5.0):
        x = totals[("81 монета", th, 3, tp)]
        ok = x[0] >= base[0] and x[1] >= base[1] and x[0] > 0 and x[1] > 0
        say("КРИТЕРИЙ: цель %g против 4 (81 монета, порог 30%%, до 3 позиций): сумма R %+.1f против %+.1f, 2-я половина %+.1f против %+.1f → %s"
            % (tp, x[0], base[0], x[1], base[1], "да" if ok else "нет"))
        if ok:
            choice = tp
            break
    say("→ РЕШЕНИЕ: цель %g × ATR" % choice)
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

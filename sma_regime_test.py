#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
РЕЖИМ РЫНКА ПО SMA50 / SMA100 / SMA200 BTC — и дают ли что-то сделки «выше SMA50, но ниже SMA200».

Сигналы и сделки — как в нашем правиле: TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), стоп 20 × ATR не ближе
порога (20% / 30%), цель 5 × ATR, одна позиция на монету (фильтр режима — ДО очереди). Binance 15m,
комиссия 0.11%, фандинг 0.01% / 8 ч. Режим — по вчерашнему дневному закрытию BTC (без подглядывания).
Три набора: основной (21), новый (20), третий (40).

ВАРИАНТЫ РЕЖИМА:
  R200  выше SMA200 (как сейчас)        R100  выше SMA100        R50  выше SMA50
  X50   выше SMA50, но НЕ выше SMA200 (добавочные сделки, если взять «выше 50 ИЛИ выше 200»)
  U     выше SMA50 ИЛИ выше SMA200 (объединение)
Случайный вход для X50: те же монеты, в случайные моменты из тех же периодов (выше 50, ниже 200), с тем же выходом.

КРИТЕРИИ (объявлены ДО прогона), для порога 30% на ВСЕХ ТРЁХ наборах:
  1) SMA50 вместо SMA200 — если средний R у R50 не ниже, чем у R200, в обеих половинах периода И сумма R больше;
  2) добавить сделки X50 (торговать «выше 50 или выше 200») — если средний R у X50 в плюсе в обеих половинах
     И лучше случайного входа в те же периоды.
Справочно: порог 20%, R100, U; портфель на 81 монете до 5 позиций — R в год у R200, R50 и U.
Лежит рядом с sma200_test.py, sma200_vol_test.py, sma200_check_test.py, coin_select_test.py. Отчёт: sma_regime_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_check_test as ck
import sma200_test as st
import sma200_vol_test as sv

c, cw = cs.c, cs.cw
REPORT = "sma_regime_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
TP, SL = 5.0, 20.0
THRS = [20.0, 30.0]
VARS = ["R200", "R100", "R50", "X50", "U"]
RND_N = 1200
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    say("РЕЖИМ ПО SMA50 / 100 / 200 · %s · Binance 15m · цель 5 / стоп 20 ATR"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    sm = {n: c.sma(btc.cl, n) for n in (50, 100, 200)}

    def up(T, n):
        i = btc.idx(T)
        return i >= 0 and sm[n][i] is not None and btc.cl[i] > sm[n][i]

    def regime(T, v):
        a50, a200 = up(T, 50), up(T, 200)
        if v == "R200":
            return a200
        if v == "R100":
            return up(T, 100)
        if v == "R50":
            return a50
        if v == "X50":
            return a50 and not a200
        return a50 or a200

    crit1 = crit2 = True
    pool = {(thr, v): [] for thr in THRS for v in VARS}     # для портфеля (все монеты)
    end_all, first_all = 0, None
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res = {(thr, v): [] for thr in THRS for v in VARS}  # (T, R)
        rnd = {thr: [] for thr in THRS}                       # случайный вход в периоды X50
        end_t, first_t = 0, None
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
                if T < st.START or atrp[i] <= 0 or btc.trend(T) != d:
                    continue
                if bisect.bisect_right(dt, T - D1) - 1 < 250:
                    continue
                raw.append((T, d, atrp[i] / 100))
            for thr in THRS:
                for v in VARS:
                    busy = 0
                    for T, d, a in raw:
                        if T < busy or SL * a * 100 < thr or not regime(T, v):
                            continue
                        k0 = int(np.searchsorted(t, T, side="left"))
                        if k0 >= len(t) - 1:
                            continue
                        pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], TP * a, SL * a)
                        res[(thr, v)].append((T, r))
                        pool[(thr, v)].append((T, pnl, r, te, sym, d, 0.0, 0.0))
                        first_t = T if first_t is None else min(first_t, T)
                        busy = te if te is not None else 10 ** 15
            y0 = max(st.START, int(t[0]) + 260 * D1)
            y1 = int(t[-1]) - 5 * D1
            rg = random.Random("x50-%s" % sym)
            got = tries = 0
            while got < RND_N and tries < RND_N * 40 and y1 > y0:
                tries += 1
                T = rg.randrange(y0, y1)
                d = btc.trend(T)
                if d == 0 or not regime(T, "X50"):
                    continue
                j = bisect.bisect_right(b30t, T - M30) - 1
                if j < 20 or atrp[j] <= 0:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                a = atrp[j] / 100
                pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], TP * a, SL * a)
                got += 1
                for thr in THRS:
                    if SL * a * 100 >= thr:
                        rnd[thr].append(r)
            say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
        end_all = max(end_all, end_t)
        first_all = first_t if first_all is None else min(first_all, first_t or first_all)
        mid = (first_t + end_t) // 2
        for thr in THRS:
            say("")
            say("  ПОРОГ СТОПА %.0f%%" % thr)
            st_ = {}
            for v in VARS:
                rs = [x[1] for x in res[(thr, v)]]
                r1 = [x[1] for x in res[(thr, v)] if x[0] < mid]
                r2 = [x[1] for x in res[(thr, v)] if x[0] >= mid]
                st_[v] = (np.mean(rs) if rs else 0, np.mean(r1) if r1 else 0, np.mean(r2) if r2 else 0, sum(rs), len(rs))
                extra = ""
                if v == "X50":
                    extra = " · случайный вход в те же периоды %+.3f R (%d)" % (np.mean(rnd[thr]) if rnd[thr] else 0, len(rnd[thr]))
                say("    %-5s сделок %4d · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f%s"
                    % (v, st_[v][4], st_[v][0], st_[v][1], st_[v][2], st_[v][3], extra))
            if thr == THRS[-1]:
                b, f = st_["R200"], st_["R50"]
                k1 = f[1] >= b[1] and f[2] >= b[2] and f[3] > b[3]
                x = st_["X50"]
                rr = np.mean(rnd[thr]) if rnd[thr] else 0
                k2 = x[4] > 0 and x[1] > 0 and x[2] > 0 and x[0] > rr
                crit1 = crit1 and k1
                crit2 = crit2 and k2
                say("    критерий 1 (SMA50 вместо 200): %s · критерий 2 (добавить X50): %s"
                    % ("да" if k1 else "нет", "да" if k2 else "нет"))
    years = (end_all - first_all) / (365.25 * D1)
    say("")
    say("=" * 110)
    say("ПОРТФЕЛЬ: 81 монета, до 5 позиций, %.1f лет (справочно)" % years)
    for thr in THRS:
        for v in ("R200", "R50", "U"):
            tr = ck.capped(pool[(thr, v)], 5)
            rs = [x[2] for x in tr]
            mid = (first_all + end_all) // 2
            say("  порог %.0f%% · %-4s сделок %4d (%.0f в год) · сумма R %+.1f (%+.1f R в год ≈ %+.0f $ в год при риске 50 $) · 2-я половина %+.1f R"
                % (thr, v, len(tr), len(tr) / years, sum(rs), sum(rs) / years, sum(rs) / years * 50,
                   sum(x[2] for x in tr if x[0] >= mid)))
    say("")
    say("=" * 110)
    say("ИТОГ: SMA50 вместо SMA200 — %s · добавить сделки «выше SMA50, ниже SMA200» — %s"
        % ("ДА" if crit1 else "нет", "ДА" if crit2 else "нет"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200 + ТОЛЬКО ПОДВИЖНЫЕ СДЕЛКИ: брать сигнал, только если стоп 20 × ATR 30m не ближе X% от входа.

Откуда гипотеза (sma200_exit_test.py): сделки со стопом > 30% дали +37.6 / +28.1 $ на сделку (основной / новый набор),
а со стопом ≤ 30% — около нуля. Находка сделана на тех же данных, поэтому здесь — отдельная проверка с критерием
до прогона, в R (прибыль в долях риска), с позицией от риска и контролем случайным входом.

Вход и выход — как R2 (sma200_test.py): сигнал TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC
вчера закрылся выше SMA200, цель 4 / стоп 20 × ATR 30m, одна позиция на монету, Binance 15m, комиссия 0.11%,
фандинг 0.01% / 8 ч. Фильтр по стопу применяется ДО очереди «одна позиция на монету».
Пороги: 0 (все), 20%, 30%. Монеты: основной (21), новый (20) и ТРЕТИЙ набор — монеты из топ-50 Bybit по объёму,
которых не было ни в одном тесте (список ниже; монеты без истории на Binance пропускаются сами). Третий набор
выбран по сегодняшнему топу — это монеты, которые «выжили» и выросли, поэтому он немного приукрашен.
Случайный вход: те же монеты, в случайные моменты при BTC выше SMA200 по тренду BTC 1Д, с тем же порогом стопа.

РАЗМЕР ПОЗИЦИИ ОТ РИСКА: на каждую сделку риск 20 $ (2% депозита 1000 $): позиция = 20 $ / стоп%.
Пример: стоп 40% → позиция 50 $; стоп 10% → позиция 200 $. Итог такой торговли = сумма R × 20 $.

КРИТЕРИЙ (объявлен ДО прогона), порог 30% принимается, если на ВСЕХ ТРЁХ наборах:
  1) средний R сделок с порогом 30% выше, чем у всех сигналов (порог 0), в ОБЕИХ половинах периода;
  2) средний R выше, чем у случайного входа с тем же порогом;
  3) сумма R с порогом 30% в плюсе в обеих половинах периода.
Справочно — то же для порога 20%, и деньги при риске 20 $ на сделку: до 3 позиций одновременно.

ФИЛЬТР MRC ВМЕСТЕ С ПРАВИЛОМ (все сигналы, без порога стопа): без MRC / упрощённая ур. 2 / упрощённая ур. 3 /
разработчика ур. 3 — средний R, сумма R и сколько дней сделка держит деньги (медиана, доля сделок дольше 7 дней).
КРИТЕРИЙ MRC (объявлен ДО прогона): вариант принимается, если на ВСЕХ ТРЁХ наборах средний R выше, чем без MRC,
в обеих половинах периода.
Лежит рядом с sma200_test.py, sma200_check_test.py, coin_select_test.py. Отчёт: sma200_vol_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_check_test as ck
import sma200_test as st

c, cw = cs.c, cs.cw
REPORT = "sma200_vol_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
TP, SL = 4.0, 20.0
THRS = [0.0, 20.0, 30.0]
RISK = 20.0
RND_N = 1500
THIRD_COINS = ["ENA", "TAO", "PEPE", "WIF", "BONK", "ONDO", "ZEC", "RENDER", "JUP", "PYTH", "ORDI", "POL", "VET",
               "QNT", "GRT", "EGLD", "THETA", "XTZ", "EOS", "RUNE", "LDO", "ENS", "PENDLE", "JASMY", "FLOKI", "CAKE",
               "COMP", "SNX", "DYDX", "STRK", "ZRO", "NOT", "BOME", "ETHFI", "JTO", "KAVA", "CFX", "ARKM", "PEOPLE", "ACH"]
MRCS = ["none", "simple", "simple3", "dev3"]
MRC_NAME = {"none": "без MRC", "simple": "MRC упр. ур.2", "simple3": "MRC упр. ур.3", "dev3": "MRC разраб. ур.3"}
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def run_set(set_name, coins, btc):
    t0 = time.time()
    res = {x: [] for x in THRS}      # (T, $, R, t_выхода, монета, напр, цена, риск $ при позиции 1000)
    resm = {m: [] for m in MRCS}     # все сигналы, варианты MRC
    rnd = {x: [] for x in THRS}      # (T, R)
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
            if T < st.START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                continue
            jd = bisect.bisect_right(dt, T - D1) - 1
            if jd < 250:
                continue
            raw.append((T, d, atrp[i] / 100, b30[i][4], mean[jd], rngd[jd]))
        for m in MRCS:
            busy = 0
            for T, d, a, px, mn, rg_ in raw:
                if T < busy or cs.hot(m, d, px, mn, rg_):
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], TP * a, SL * a)
                resm[m].append((T, pnl, r, te))
                busy = te if te is not None else 10 ** 15
        for thr in THRS:
            busy = 0
            for T, d, a, _px, _mn, _rg in raw:
                if T < busy or SL * a * 100 < thr:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                pnl, r, te = cs.trade(t, h, l, cl, k0, d, o[k0], TP * a, SL * a)
                res[thr].append((T, pnl, r, te, sym, d, float(o[k0]), 1000.0 * SL * a))
                busy = te if te is not None else 10 ** 15
        y0 = max(st.START, int(t[0]) + 260 * D1)
        y1 = int(t[-1]) - 5 * D1
        rg = random.Random("vol-%s" % sym)
        tries = got = 0
        while got < RND_N and tries < RND_N * 30 and y1 > y0:
            tries += 1
            T = rg.randrange(y0, y1)
            d = btc.trend(T)
            if d == 0 or not btc.above(T):
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
                    rnd[thr].append((T, r))
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return res, rnd, resm, end_t


def main():
    t0 = time.time()
    say("SMA200 + ТОЛЬКО ПОДВИЖНЫЕ СДЕЛКИ · %s · Binance 15m · риск на сделку %.0f $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), RISK))
    btc = st.Btc()
    ok_all = True
    mrc_ok = {m: True for m in MRCS if m != "none"}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res, rnd, resm, end_t = run_set(set_name, coins, btc)
        if not res[0.0]:
            say("  нет сделок — набор пропущен")
            continue
        t_first = min(x[0] for x in res[0.0])
        mid = (t_first + end_t) // 2
        stat = {}
        say("")
        for thr in THRS:
            v = res[thr]
            n = len(v)
            rs = [x[2] for x in v]
            r1 = [x[2] for x in v if x[0] < mid]
            r2 = [x[2] for x in v if x[0] >= mid]
            rr = [x[1] for x in rnd[thr]]
            yrs = {}
            for x in v:
                yrs.setdefault(cs.year_of(x[0]), []).append(x[2])
            stat[thr] = {"avg": np.mean(rs) if rs else 0, "a1": np.mean(r1) if r1 else 0, "a2": np.mean(r2) if r2 else 0,
                         "s1": sum(r1), "s2": sum(r2), "rnd": np.mean(rr) if rr else 0}
            s = stat[thr]
            say("  стоп ≥ %2.0f%%: сделок %4d · WR %4.1f%% · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f "
                "(%+.1f / %+.1f) · случайный вход %+.3f R (%d) · при риске %.0f $: %+.0f $"
                % (thr, n, sum(1 for x in rs if x > 0) / max(1, n) * 100, s["avg"], s["a1"], s["a2"], sum(rs),
                   s["s1"], s["s2"], s["rnd"], len(rr), RISK, sum(rs) * RISK))
            say("      по годам (сумма R): " + " · ".join("%d: %+.1f (%d)" % (y, sum(z), len(z)) for y, z in sorted(yrs.items())))
            # деньги: позиция от риска, до 3 позиций одновременно, просадка по закрытым
            tr = ck.capped(v, 3)
            eq = peak = dd = 0.0
            for x in sorted(tr, key=lambda x: x[3] if x[3] is not None else 10 ** 15):
                eq += x[2] * RISK
                peak = max(peak, eq)
                dd = min(dd, eq - peak)
            pos = [RISK / (x[7] / 1000.0) for x in v]
            say("      до 3 позиций, риск %.0f $: сделок %d · итог %+.0f $ · просадка по закрытым %+.0f $ · позиция в среднем %.0f $ "
                "(от %.0f до %.0f $)" % (RISK, len(tr), sum(x[2] for x in tr) * RISK, dd,
                                        np.mean(pos) if pos else 0, min(pos) if pos else 0, max(pos) if pos else 0))
        a, b = stat[30.0], stat[0.0]
        k1 = a["a1"] > b["a1"] and a["a2"] > b["a2"]
        k2 = a["avg"] > a["rnd"]
        k3 = a["s1"] > 0 and a["s2"] > 0
        ok = k1 and k2 and k3
        ok_all = ok_all and ok
        say("")
        say("  ФИЛЬТР MRC ВМЕСТЕ С ПРАВИЛОМ (все сигналы):")
        base = None
        for m in MRCS:
            v = resm[m]
            rs = [x[2] for x in v]
            r1 = [x[2] for x in v if x[0] < mid]
            r2 = [x[2] for x in v if x[0] >= mid]
            hold = sorted(((x[3] if x[3] is not None else end_t) - x[0]) / D1 for x in v)
            a1 = np.mean(r1) if r1 else 0
            a2 = np.mean(r2) if r2 else 0
            if m == "none":
                base = (a1, a2)
                mark = ""
            else:
                good = a1 > base[0] and a2 > base[1]
                mrc_ok[m] = mrc_ok[m] and good
                mark = " → лучше без MRC в обеих половинах: %s" % ("да" if good else "нет")
            say("    %-17s сделок %4d · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f · держит деньги: медиана "
                "%.1f дн, дольше 7 дн %.0f%%, дольше 30 дн %.0f%%%s"
                % (MRC_NAME[m], len(v), np.mean(rs) if rs else 0, a1, a2, sum(rs), hold[len(hold) // 2] if hold else 0,
                   100 * sum(1 for x in hold if x > 7) / max(1, len(hold)),
                   100 * sum(1 for x in hold if x > 30) / max(1, len(hold)), mark))
        say("  критерий (порог 30%%): лучше всех сигналов в обеих половинах [%s] · лучше случайного входа [%s, %+.3f против %+.3f] "
            "· обе половины в плюсе [%s] → %s" % ("да" if k1 else "нет", "да" if k2 else "нет", a["avg"], a["rnd"],
                                                  "да" if k3 else "нет", "да" if ok else "нет"))
    say("")
    say("=" * 110)
    say("ИТОГ: %s" % ("ПОРОГ 30% ПРИНЯТ — брать только сигналы со стопом ≥ 30%, позиция от риска"
                      if ok_all else "порог 30% не принят — брать все сигналы"))
    for m, ok in mrc_ok.items():
        say("%s вместе с правилом: %s" % (MRC_NAME[m], "ПРИНЯТ (лучше на всех наборах)" if ok else "не принят"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

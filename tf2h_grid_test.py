#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СЕТКА ЦЕЛЬ/СТОП НА 2ч ГРАФИКЕ, СТАРШИЙ ТФ 1Д — проверка (идея: подобрать цель/стоп под 2ч / 1Д).

Сигнал: TT2 на графике 2ч, старший ТФ 1Д (EMA 9/21), по тренду BTC 1Д, только когда BTC вчера закрылся выше SMA200.
Порога ⭐ НЕТ (он задан в % стопа и при стопе 2–10 ATR 2ч убил бы все сигналы); стоп не дальше 80%.
Риск 50 $ на сделку, одна позиция на монету, BTC не торгуем. Исполнение — cs.trade на 15m (спорная свеча — стоп,
комиссия 0.11% за круг, фандинг 0.01% за 8 ч). R = результат / фактический стоп (комиссия входит).

ВАРИАНТЫ (объявлены ДО прогона): цель 2/3/4/5/6/7 × стоп 2/4/6/7/10 ATR (2ч) — 30 вариантов.
БАЗА — текущее правило: 30m / 4ч, 5/20 ATR, только ⭐ (стоп ≥ 30%).
НАБОРЫ: основной 21, новый 20, третий 40 (как в tf_test.py).

КРИТЕРИЙ (объявлен ДО прогона). Вариант лучше базы, если на ВСЕХ ТРЁХ наборах:
  1) портфель 5 слотов (медиана 50 случайных порядков): итог в R больше базы, просадка в R не больше чем в 1.2 раза хуже;
  2) средний R сделки больше среднего R случайного входа с тем же выходом;
  3) обе половины периода в плюсе (R портфеля);
  4) контроль: на перемешанных ценах (основной набор) перевес над случайным входом не больше +0.02 R.
Вариантов 30 — требование «все три набора + контроль» обязательно; отдельно печатаются таблицы цель × стоп,
чтобы видеть, сплошная ли область хороших клеток (интересно) или отдельные пятна (шум).

Запуск: python tf2h_grid_test.py (кэш data_btc/). Переменные: START_YEAR (2019), RND_N (100), SCEN (50).
Отчёт: tf2h_grid_report.txt
"""
import bisect
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import sma200_test as st
import tf_test as tt

cs, c, cw = st.cs, st.c, st.cw
REPORT = "tf2h_grid_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
H2 = 4 * M30
SL_CAP = 0.80
START = int(datetime(int(os.environ.get("START_YEAR") or 2019), 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
RND_N = int(os.environ.get("RND_N") or 100)
SCEN = int(os.environ.get("SCEN") or 50)
TPS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
SLS = (2.0, 4.0, 6.0, 7.0, 10.0)
BASE = ("base", M30, H4, 5.0, 20.0, 0.30)
VARS = [BASE] + [("g", H2, D1, tp, sl, 0.0) for tp in TPS for sl in SLS]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(v):
    if v[0] == "base":
        return "БАЗА 30m/4ч 5/20 ⭐"
    return "2ч/1Д цель %g / стоп %g" % (v[3], v[4])


def fmt(x):
    return "—" if x != x else "%+.3f" % x


def prep(sym, b15, btc):
    t15 = np.array([x[0] for x in b15], dtype=np.int64)
    o = np.array([x[1] for x in b15])
    h = np.array([x[2] for x in b15])
    l = np.array([x[3] for x in b15])
    cl = np.array([x[4] for x in b15])
    dt = [x[0] for x in c.agg(b15, D1)]
    end_t = int(t15[-1])
    charts, highs = {}, {}
    for tf in (M30, H2):
        bars = c.agg(b15, tf)
        charts[tf] = (bars, [x[0] for x in bars], c.atr_pct(bars))
    for htf in (H4, D1):
        hb = c.agg(b15, htf)
        hc = [x[4] for x in hb]
        highs[htf] = ([x[0] for x in hb], [a > b_ for a, b_ in zip(c.ema(hc, 9), c.ema(hc, 21))])
    raw = {}
    for tf, htf in ((M30, H4), (H2, D1)):
        bars, bt, atrp = charts[tf]
        ht, hu = highs[htf]
        lst = []
        for T, d, i in cw.ind_entries_tf(bars, tf, ht, hu, htf):
            if T < START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                continue
            if bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            lst.append((T, d, atrp[i] / 100))
        raw[(tf, htf)] = lst

    def one(T, d, a, tp, sl):
        k0 = int(np.searchsorted(t15, T, side="left"))
        if k0 >= len(t15) - 1:
            return None
        return cs.trade(t15, h, l, cl, k0, d, float(o[k0]), tp * a, min(sl * a, SL_CAP))

    res = {}
    for vi, (_, tf, htf, tp, sl, mins) in enumerate(VARS):
        sigs, busy = [], 0
        for T, d, a in raw[(tf, htf)]:
            if T < busy or sl * a < mins:
                continue
            x = one(T, d, a, tp, sl)
            if x is None:
                continue
            pnl, r, te = x
            sigs.append({"sym": sym, "T": T, "r": r, "te": te})
            busy = te if te is not None else 10 ** 15
        bars, bt, atrp = charts[tf]
        rnd, tries = [], 0
        rg = random.Random("g2h-%s-%d" % (sym, vi))
        y0 = max(START, int(t15[0]) + 260 * D1)
        y1 = end_t - 5 * D1
        while len(rnd) < RND_N and tries < RND_N * 30 and y1 > y0:
            tries += 1
            T = rg.randrange(y0, y1)
            d = btc.trend(T)
            if d == 0 or not btc.above(T):
                continue
            j = bisect.bisect_right(bt, T - tf) - 1
            if j < 20 or atrp[j] <= 0:
                continue
            a = atrp[j] / 100
            if sl * a < mins or bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            x = one(T, d, a, tp, sl)
            if x is not None:
                rnd.append(x[1])
        res[vi] = (sigs, rnd)
    return res, end_t


def run_set(name, coins, btc, shuffled=False):
    t0 = time.time()
    allres = {vi: ([], []) for vi in range(len(VARS))}
    end_t, n = 0, 0
    for sym in coins:
        try:
            b15 = cs.load_15m(sym)
        except Exception as e:                                   # noqa: BLE001
            say("  [%s] %s: нет данных (%s)" % (name, sym, e))
            continue
        if len(b15) < 20000:
            continue
        if shuffled:
            b15 = tt.shuffle_days(b15, "coin-%s" % sym)
        res, e = prep(sym, b15, btc)
        end_t = max(end_t, e)
        n += 1
        for vi, (sg, rn) in res.items():
            allres[vi][0].extend(sg)
            allres[vi][1].extend(rn)
        say("  [%s] %s: база %d, сетка 2ч %d сигналов · %.0f с" % (
            name, sym, len(res[0][0]), len(res[1][0]), time.time() - t0))
    out = {}
    for vi in range(len(VARS)):
        sigs, rnd = allres[vi]
        if not sigs:
            out[vi] = None
            continue
        mid = (min(s["T"] for s in sigs) + end_t) // 2
        runs = []
        for sd in range(SCEN):
            tk = tt.portfolio(sigs, sd)
            tot, dd = tt.curve(tk, end_t)
            runs.append((tot, dd, sum(s["r"] for s in tk if s["T"] < mid),
                         sum(s["r"] for s in tk if s["T"] >= mid), len(tk)))
        hold = [((s["te"] if s["te"] is not None else end_t) - s["T"]) / D1 for s in sigs]
        out[vi] = {"n": len(sigs), "avg": tt.mean([s["r"] for s in sigs]), "rnd": tt.mean(rnd),
                   "wr": 100 * sum(1 for s in sigs if s["r"] > 0) / len(sigs),
                   "tot": tt.med([r[0] for r in runs]), "dd": tt.med([r[1] for r in runs]),
                   "h1": tt.med([r[2] for r in runs]), "h2": tt.med([r[3] for r in runs]),
                   "taken": tt.med([r[4] for r in runs]), "hold": tt.med(hold)}
    say("  [%s] монет %d" % (name, n))
    return out


def show(name, res):
    say("")
    say("=" * 110)
    say("НАБОР «%s»" % name)
    for vi, v in enumerate(VARS):
        r = res.get(vi)
        if not r:
            say("  %-26s нет сигналов" % vname(v))
            continue
        say("  %-26s сделок %5d · WR %3.0f%% · R %s (случ. %s) · удерж. %.1f дн · портфель: взято %d, итог %+.1f R ≈ %+.0f $, "
            "просадка %+.1f R, половины %+.1f / %+.1f" % (
                vname(v), r["n"], r["wr"], fmt(r["avg"]), fmt(r["rnd"]), r["hold"], r["taken"], r["tot"], r["tot"] * 50,
                r["dd"], r["h1"], r["h2"]))
    say("")
    say("  Таблица: итог портфеля в R (база %+.1f R); строки — цель, столбцы — стоп" % (res[0]["tot"] if res.get(0) else 0))
    say("  цель\\стоп " + "".join("%9g" % s for s in SLS))
    for tp in TPS:
        cells = []
        for sl in SLS:
            vi = VARS.index(("g", H2, D1, tp, sl, 0.0))
            r = res.get(vi)
            cells.append("%9s" % ("—" if not r else "%+.1f" % r["tot"]))
        say("  %9g " % tp + "".join(cells))
    say("  Таблица: средний R сделки минус случайный вход")
    say("  цель\\стоп " + "".join("%9g" % s for s in SLS))
    for tp in TPS:
        cells = []
        for sl in SLS:
            vi = VARS.index(("g", H2, D1, tp, sl, 0.0))
            r = res.get(vi)
            cells.append("%9s" % ("—" if not r else "%+.3f" % (r["avg"] - r["rnd"])))
        say("  %9g " % tp + "".join(cells))


def main():
    t0 = time.time()
    say("СЕТКА ЦЕЛЬ/СТОП · 2ч / 1Д · %s · Binance 15m · риск 50 $, 5 слотов"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    sets = [("основной", list(st.MAIN_COINS)), ("новый", list(st.NEW_COINS)), ("третий", tt.THIRD)]
    real = {}
    for nm, coins in sets:
        real[nm] = run_set(nm, coins, btc)
        show(nm, real[nm])
    say("")
    say("#" * 110)
    say("КОНТРОЛЬ: перемешанные цены (сутки переставлены; BTC тоже), основной набор")
    btc_sh = tt.BtcFrom(tt.shuffle_days(cs.load_15m("BTC"), "coin-BTC"))
    ctrl = run_set("перемешанные", list(st.MAIN_COINS), btc_sh, shuffled=True)
    show("перемешанные (контроль)", ctrl)

    say("")
    say("=" * 110)
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона)")
    passed = []
    for vi in range(1, len(VARS)):
        ok, marks = True, []
        for nm, _ in sets:
            r, b = real[nm].get(vi), real[nm].get(0)
            if not r or not b:
                ok = False
                marks.append("%s: нет" % nm)
                continue
            k1 = r["tot"] > b["tot"] and -r["dd"] <= 1.2 * -b["dd"]
            k2 = r["avg"] > r["rnd"]
            k3 = r["h1"] > 0 and r["h2"] > 0
            ok = ok and k1 and k2 and k3
            marks.append("%s %s%s%s" % (nm, "П" if k1 else "п", "С" if k2 else "с", "Г" if k3 else "г"))
        cr = ctrl.get(vi)
        k4 = cr is None or cr["avg"] - cr["rnd"] <= 0.02
        ok = ok and k4
        if ok:
            passed.append(vname(VARS[vi]))
        say("  %-26s %s · контроль %s → %s" % (vname(VARS[vi]), " · ".join(marks),
                                              fmt(cr["avg"] - cr["rnd"]) if cr else "—", "ЛУЧШЕ БАЗЫ" if ok else "нет"))
    say("  (П — портфель лучше базы при допустимой просадке, С — лучше случайного входа, Г — обе половины в плюсе;"
        " строчная буква — не выполнено)")
    say("")
    say("ИТОГ: лучше базы %d из %d%s" % (len(passed), len(VARS) - 1, (" — " + "; ".join(passed)) if passed else ""))
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

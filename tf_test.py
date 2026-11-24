#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТАЙМФРЕЙМ ГРАФИКА И СТАРШИЙ ТФ ДЛЯ ПРАВИЛА SMA200 — проверка (идея: 2ч график + старший 1Д лучше по индикатору).

Правило, кроме проверяемого: сигнал TT2 по тренду BTC 1Д, только когда BTC вчера закрылся выше SMA200,
только ⭐ (стоп не ближе 30%), стоп не дальше 80%, риск 50 $ на сделку, одна позиция на монету, BTC не торгуем.
Исполнение — cs.trade на 15m (спорная свеча — стоп, комиссия 0.11% за круг, фандинг 0.01% за 8 ч).

ВАРИАНТЫ (объявлены ДО прогона):
  график 30m / 1h / 2h × старший ТФ 4ч / 1Д (EMA 9/21 старшего ТФ, как в индикаторе) — 6 сочетаний;
  цель/стоп в ATR графика масштабируются на корень из длины свечи, чтобы расстояния в % были сопоставимы:
  30m — 5/20, 1h — 3.54/14.1, 2h — 2.5/10;
  7-й вариант «как в индикаторе»: 2h / 1Д с прежними 5/20 × ATR 2h.
  БАЗА — 30m / 4ч, 5/20 (как сейчас).
НАБОРЫ МОНЕТ: основной 21, новый 20, третий 40 (монеты с историей, которых нет в первых двух).

КРИТЕРИЙ (объявлен ДО прогона). Вариант лучше базы, если на ВСЕХ ТРЁХ наборах:
  1) портфель 5 слотов (медиана 50 случайных порядков равных сигналов): итог в R больше базы,
     а максимальная просадка в R не больше чем в 1.2 раза хуже;
  2) средний R сделки (одна позиция на монету) больше среднего R случайного входа с тем же выходом и ТФ;
  3) обе половины периода в плюсе (по R портфеля);
  4) контроль: на ПЕРЕМЕШАННЫХ ценах (основной набор, дни переставлены случайно — тренды разрушены,
     форма свечей сохранена) у варианта нет перевеса над случайным входом больше +0.02 R; иначе результат
     считается артефактом и вариант не принимается.
Требование «все три набора» — защита от одного удачного варианта из семи.

Запуск: python tf_test.py (данные из кэша data_btc/). Переменные: START_YEAR (2019), RND_N (200), SCEN (50).
Отчёт: tf_report.txt
"""
import bisect
import math
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import sma200_test as st

cs, c, cw = st.cs, st.c, st.cw
REPORT = "tf_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
H1, H2 = 2 * M30, 4 * M30
SL_CAP, STAR = 0.80, 0.30
SLOTS = 5
START = int(datetime(int(os.environ.get("START_YEAR") or 2019), 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
RND_N = int(os.environ.get("RND_N") or 200)
SCEN = int(os.environ.get("SCEN") or 50)
TFN = {M30: "30m", H1: "1h", H2: "2h", H4: "4ч", D1: "1Д"}


def k_for(tf):
    s = math.sqrt(tf / M30)
    return 5.0 / s, 20.0 / s


VARS = []
for _tf in (M30, H1, H2):
    for _htf in (H4, D1):
        _tp, _sl = k_for(_tf)
        VARS.append((_tf, _htf, _tp, _sl))
VARS.append((H2, D1, 5.0, 20.0))
BASE = 0

THIRD = ["MATIC", "FTM", "EOS", "XTZ", "VET", "THETA", "EGLD", "FLOW", "KAVA", "RUNE", "CHZ", "ENJ", "GRT", "SNX",
         "COMP", "MKR", "YFI", "SUSHI", "1INCH", "BAT", "ZEC", "DASH", "NEO", "IOTA", "ZIL", "QTUM", "ICX", "ONT",
         "KSM", "CELO", "LRC", "ANKR", "HOT", "LDO", "APE", "MINA", "ENS", "DYDX", "SSV", "ROSE"]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(v):
    tf, htf, tp, sl = v
    return "%s / %s, %g/%g ATR" % (TFN[tf], TFN[htf], round(tp, 2), round(sl, 1))


def fmt(x):
    return "—" if x != x else "%+.3f" % x


class BtcFrom:
    def __init__(self, b15):
        bd1 = c.agg(b15, D1)
        self.t = [x[0] for x in bd1]
        self.tr = c.trend_arr(bd1)
        self.cl = [x[4] for x in bd1]
        self.sma = c.sma(self.cl, 200)

    def idx(self, T):
        return bisect.bisect_right(self.t, T - D1) - 1

    def trend(self, T):
        i = self.idx(T)
        return self.tr[i] if i >= 0 else 0

    def above(self, T):
        i = self.idx(T)
        return i >= 0 and self.sma[i] is not None and self.cl[i] > self.sma[i]


def shuffle_days(b15, seed):
    """Перемешанные цены: 15m-свечи разбиты на сутки, сутки переставлены, цены пересобраны от первой цены.
    Форма свечей внутри дня сохранена, тренды и связи между днями разрушены."""
    n = len(b15) // 96
    if n < 300:
        return b15
    blocks = []
    for k in range(n):
        part = b15[k * 96:(k + 1) * 96]
        base = part[0][1]
        if base <= 0:
            return b15
        blocks.append([(x[1] / base, x[2] / base, x[3] / base, x[4] / base, x[5]) for x in part])
    rg = random.Random(seed)
    rg.shuffle(blocks)
    out, p = [], float(b15[0][1])
    for k, blk in enumerate(blocks):
        for j, (ro, rh, rl, rc, v) in enumerate(blk):
            t = b15[k * 96 + j][0]
            out.append([t, p * ro, p * rh, p * rl, p * rc, v])
        p = p * blk[-1][3]
    return out


def prep(sym, b15, btc):
    t15 = np.array([x[0] for x in b15], dtype=np.int64)
    o = np.array([x[1] for x in b15])
    h = np.array([x[2] for x in b15])
    l = np.array([x[3] for x in b15])
    cl = np.array([x[4] for x in b15])
    d1 = c.agg(b15, D1)
    dt = [x[0] for x in d1]
    end_t = int(t15[-1])
    charts = {}
    for tf in (M30, H1, H2):
        bars = c.agg(b15, tf)
        charts[tf] = (bars, [x[0] for x in bars], c.atr_pct(bars))
    highs = {}
    for htf in (H4, D1):
        hb = c.agg(b15, htf)
        hc = [x[4] for x in hb]
        highs[htf] = ([x[0] for x in hb], [a > b_ for a, b_ in zip(c.ema(hc, 9), c.ema(hc, 21))])
    res = {}
    for vi, (tf, htf, tp, sl) in enumerate(VARS):
        bars, bt, atrp = charts[tf]
        ht, hu = highs[htf]
        sigs, busy = [], 0
        for T, d, i in cw.ind_entries_tf(bars, tf, ht, hu, htf):
            if T < START or T < busy or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                continue
            a = atrp[i] / 100
            if sl * a < STAR or bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                continue
            pnl, r, te = cs.trade(t15, h, l, cl, k0, d, float(o[k0]), tp * a, min(sl * a, SL_CAP))
            sigs.append({"sym": sym, "T": T, "r": r, "te": te})
            busy = te if te is not None else 10 ** 15
        rnd = []
        rg = random.Random("tf-%s-%d" % (sym, vi))
        y0 = max(START, int(t15[0]) + 260 * D1)
        y1 = end_t - 5 * D1
        tries = 0
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
            if sl * a < STAR or bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                continue
            pnl, r, te = cs.trade(t15, h, l, cl, k0, d, float(o[k0]), tp * a, min(sl * a, SL_CAP))
            rnd.append(r)
        res[vi] = (sigs, rnd)
    return res, end_t


def portfolio(sigs, seed):
    rg = random.Random(seed)
    el = sorted(sigs, key=lambda s: (s["T"], rg.random()))
    open_, taken = [], []
    for s in el:
        open_ = [x for x in open_ if x[0] is None or x[0] > s["T"]]
        if len(open_) >= SLOTS or any(x[1] == s["sym"] for x in open_):
            continue
        open_.append((s["te"], s["sym"]))
        taken.append(s)
    return taken


def curve(taken, end_t):
    ev = sorted(((s["te"] if s["te"] is not None else end_t), s["r"]) for s in taken)
    cum = peak = dd = 0.0
    for _, r in ev:
        cum += r
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return cum, dd


def med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else float("nan")


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def run_set(name, coins, btc, shuffled=False):
    t0 = time.time()
    allres = {vi: ([], []) for vi in range(len(VARS))}
    end_t, n_coins = 0, 0
    for k, sym in enumerate(coins):
        try:
            b15 = cs.load_15m(sym)
        except Exception as e:                                   # noqa: BLE001
            say("  [%s] %s: нет данных (%s)" % (name, sym, e))
            continue
        if len(b15) < 20000:
            say("  [%s] %s: мало данных — пропуск" % (name, sym))
            continue
        if shuffled:
            b15 = shuffle_days(b15, "coin-%s" % sym)
        res, e = prep(sym, b15, btc)
        end_t = max(end_t, e)
        n_coins += 1
        for vi, (sg, rn) in res.items():
            allres[vi][0].extend(sg)
            allres[vi][1].extend(rn)
        say("  [%s] %s: сигналов по вариантам %s · %.0f с" % (
            name, sym, "/".join(str(len(res[vi][0])) for vi in range(len(VARS))), time.time() - t0))
    out = {}
    for vi in range(len(VARS)):
        sigs, rnd = allres[vi]
        if not sigs:
            out[vi] = None
            continue
        mid = (min(s["T"] for s in sigs) + end_t) // 2
        runs = []
        for sd in range(SCEN):
            tk = portfolio(sigs, sd)
            tot, dd = curve(tk, end_t)
            h1 = sum(s["r"] for s in tk if s["T"] < mid)
            h2 = sum(s["r"] for s in tk if s["T"] >= mid)
            runs.append((tot, dd, h1, h2, len(tk)))
        tk0 = portfolio(sigs, 0)
        yrs = {}
        for s in tk0:
            yrs.setdefault(datetime.fromtimestamp(s["T"] / 1000, tz=timezone.utc).year, []).append(s["r"])
        hold = [((s["te"] if s["te"] is not None else end_t) - s["T"]) / D1 for s in sigs]
        out[vi] = {"n": len(sigs), "avg": mean([s["r"] for s in sigs]), "rnd": mean(rnd), "nrnd": len(rnd),
                   "wr": 100 * sum(1 for s in sigs if s["r"] > 0) / len(sigs),
                   "tot": med([r[0] for r in runs]), "dd": med([r[1] for r in runs]),
                   "h1": med([r[2] for r in runs]), "h2": med([r[3] for r in runs]), "taken": med([r[4] for r in runs]),
                   "hold": med(hold), "yrs": yrs}
    say("  [%s] монет с данными %d" % (name, n_coins))
    return out


def show(name, res):
    say("")
    say("=" * 110)
    say("НАБОР «%s»" % name)
    for vi, v in enumerate(VARS):
        r = res.get(vi)
        if not r:
            say("  %-30s нет сигналов" % vname(v))
            continue
        say("  %-30s%s сделок %5d · WR %3.0f%% · средний R %s (случайный %s) · медиана удержания %.1f дн"
            % (vname(v), " ← база" if vi == BASE else "", r["n"], r["wr"], fmt(r["avg"]), fmt(r["rnd"]), r["hold"]))
        say("      портфель %d слотов: взято %d · итог %+.1f R ≈ %+.0f $ · просадка %+.1f R · половины %+.1f / %+.1f R · по годам: %s"
            % (SLOTS, r["taken"], r["tot"], r["tot"] * 50, r["dd"], r["h1"], r["h2"],
               " · ".join("%d: %+.1f (%d)" % (y, sum(x), len(x)) for y, x in sorted(r["yrs"].items()))))


def main():
    t0 = time.time()
    say("ТАЙМФРЕЙМ ГРАФИКА И СТАРШИЙ ТФ · %s · Binance 15m · ⭐ (стоп ≥ 30%%), риск 50 $, %d слотов"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), SLOTS))
    btc_real = st.Btc()
    sets = [("основной", list(st.MAIN_COINS)), ("новый", list(st.NEW_COINS)), ("третий", THIRD)]
    real = {}
    for nm, coins in sets:
        real[nm] = run_set(nm, coins, btc_real)
        show(nm, real[nm])

    say("")
    say("#" * 110)
    say("КОНТРОЛЬ: перемешанные цены (сутки переставлены; BTC тоже), основной набор")
    btc_sh = BtcFrom(shuffle_days(cs.load_15m("BTC"), "coin-BTC"))
    ctrl = run_set("перемешанные", list(st.MAIN_COINS), btc_sh, shuffled=True)
    show("перемешанные (контроль)", ctrl)

    say("")
    say("=" * 110)
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона)")
    passed = []
    for vi, v in enumerate(VARS):
        if vi == BASE:
            continue
        marks, ok = [], True
        for nm, _ in sets:
            r, b = real[nm].get(vi), real[nm].get(BASE)
            if not r or not b:
                marks.append("%s: нет данных" % nm)
                ok = False
                continue
            k1 = r["tot"] > b["tot"] and -r["dd"] <= 1.2 * -b["dd"]
            k2 = r["avg"] > r["rnd"]
            k3 = r["h1"] > 0 and r["h2"] > 0
            ok = ok and k1 and k2 and k3
            marks.append("%s: портф. %+.1f/%+.1f R, просадка %+.1f/%+.1f [%s] · против случ. [%s] · половины [%s]" % (
                nm, r["tot"], b["tot"], r["dd"], b["dd"], "да" if k1 else "нет", "да" if k2 else "нет",
                "да" if k3 else "нет"))
        c4r = ctrl.get(vi)
        k4 = c4r is None or (c4r["avg"] - c4r["rnd"]) <= 0.02
        ok = ok and k4
        marks.append("контроль: перевес над случайным %s [%s]" % (
            fmt(c4r["avg"] - c4r["rnd"]) if c4r else "—", "ок" if k4 else "АРТЕФАКТ"))
        say("  %s → %s" % (vname(v), "ЛУЧШЕ БАЗЫ" if ok else "не лучше"))
        for m in marks:
            say("      " + m)
        if ok:
            passed.append(vname(v))
    b4 = ctrl.get(BASE)
    say("  справочно, база на перемешанных: перевес над случайным %s" % (fmt(b4["avg"] - b4["rnd"]) if b4 else "—"))
    say("")
    say("ИТОГ: лучше базы %d из %d%s" % (len(passed), len(VARS) - 1, (" — " + "; ".join(passed)) if passed else ""))
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

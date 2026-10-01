#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО «ТОРГОВАТЬ ТОЛЬКО КОГДА BTC ВЫШЕ SMA200» — проверка находки из regime_test.py.

Откуда гипотеза: в regime_test.py выше SMA200 широкий стоп без MRC (4/20) был лучшим и на подборе
(2019–2024, +32 564 $), и на проверке (2024–2026, +10 381 $), а ниже SMA200 на проверке проиграли все
настройки. Правило найдено на 21 основной монете — поэтому главная проверка на ДРУГИХ монетах.

Сигналы — TT2: график 30m, старший ТФ 4ч, только по дневному тренду BTC (как в coin_select_test.py).
«BTC выше SMA200» = вчерашнее дневное закрытие BTC выше SMA200 по дневкам (известно в момент входа).
Вход по открытию 15m после сигнала, одна позиция на монету в каждом правиле, внутри 15m-свечи
«и цель, и стоп» — стоп. Позиция 1000 $ (маржа 100 × 10), комиссия 0.11% за круг, фандинг 0.01% за 8 ч.
Данные — Binance спот 15m (кэш data_btc/). Сигналы — когда у монеты есть 250 дневок (как в прошлых тестах).

ПРАВИЛА (без всякого подбора, объявлены до прогона):
  R0  неизменный набор: MRC упр. 2, цель 5 / стоп 3.5 ATR, всегда;
  R1  без MRC, цель 4 / стоп 20 ATR, всегда;
  R2  без MRC, цель 4 / стоп 20 ATR, только когда BTC выше SMA200   ← проверяемая находка;
  R3  неизменный набор, только когда BTC выше SMA200 (помогает ли SMA200 и обычному набору).
  Б   для сравнения — выбор одной настройки на все монеты по последним 12 месяцам (как в regime_test.py).

НАБОРЫ МОНЕТ:
  основной (21, на нём правило нашли): ETH SOL XRP DOGE LINK ADA AVAX BNB LTC UNI FIL DOT NEAR XLM BCH AAVE ATOM ETC ALGO CRV ONE
  новый (независимый, им раньше перепроверяли MRC): SUI APT ARB OP INJ TIA SEI WLD TRX HBAR ICP FET STX IMX GALA SAND MANA AXS SHIB TON

ДЕНЬГИ КРОСС-МАРЖИ (для каждого правила): кривая счёта по дневным закрытиям с учётом НЕзакрытых позиций,
максимальная просадка в $ и когда, сколько позиций открыто одновременно, сумма их стопов в $.

КРИТЕРИЙ «R2 — НАХОДКА» (объявлен ДО прогона). На НОВОМ наборе монет:
  1) итог R2 в плюсе;
  2) средняя сделка R2 лучше случайного входа с тем же выходом, тоже только выше SMA200;
  3) итог R2 больше итога R1 (фильтр SMA200 добавляет деньги, а не просто режет сделки);
  4) обе половины периода в плюсе и в плюсе больше половины лет (годы, где не меньше 30 сделок).
  И на ОСНОВНОМ наборе: итог R2 в плюсе и больше итога R1.
  Справочно (не входит в критерий): R2 против Б — если R2 не хуже, подбор настроек не нужен.
Лежит рядом с coin_select_test.py, compare_test.py, coin_wf_test.py. Отчёт: sma200_report.txt
"""
import bisect
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs

c, cw = cs.c, cs.cw
REPORT = "sma200_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
NOTIONAL = cs.NOTIONAL
MAIN_COINS = list(cs.COINS)
NEW_COINS = ["SUI", "APT", "ARB", "OP", "INJ", "TIA", "SEI", "WLD", "TRX", "HBAR", "ICP", "FET", "STX", "IMX",
             "GALA", "SAND", "MANA", "AXS", "SHIB", "TON"]
MRCS = ["none", "simple", "simple3", "dev3"]
MRC_NAME = {"none": "без MRC", "simple": "MRC упр.2", "simple3": "MRC упр.3", "dev3": "MRC разр.3"}
EXITS = [(tp, sl) for sl in (3.5, 20.0) for tp in (3.0, 4.0, 5.0, 6.0)]
COMBOS = [(m, tp, sl) for m in MRCS for tp, sl in EXITS]
RULES = [("R0", "неизменный (MRC упр.2 5/3.5), всегда", ("simple", 5.0, 3.5), False),
         ("R1", "без MRC 4/20, всегда", ("none", 4.0, 20.0), False),
         ("R2", "без MRC 4/20, только BTC выше SMA200", ("none", 4.0, 20.0), True),
         ("R3", "неизменный, только BTC выше SMA200", ("simple", 5.0, 3.5), True)]
RND_N = 1500
START = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def ms(y, m, d=1):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def dstr(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def cname(cb):
    m, tp, sl = cb
    return "%s %g/%g" % (MRC_NAME[m], tp, sl)


class Btc:
    def __init__(self):
        b15 = cs.load_15m("BTC")
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


def run_set(set_name, coins, btc):
    """Считает все последовательности сделок для набора монет. Сделка: (T, $, R, t_выхода|None, монета, напр., цена, риск $)."""
    t0 = time.time()
    seq = {}        # (комбо, только_выше) -> монета -> [сделки]
    rnd = {}        # (цель, стоп) -> монета -> [сделки], у каждой флаг «выше SMA200» в конце
    days = {}       # монета -> (времена дней, закрытия)
    have = []
    end_t = 0
    for sym in coins:
        b15 = cs.load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        have.append(sym)
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        end_t = max(end_t, int(t15[-1]))
        b30 = c.agg(b15, M30)
        b30t = [x[0] for x in b30]
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        days[sym] = (dt, [x[4] for x in d1])
        mean = cs.supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = cs.supersmoother(c.trs(d1), 200)
        raw = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < START or atrp[i] <= 0 or btc.trend(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - D1) - 1
            if jd < 250:
                continue
            raw.append((T, d, atrp[i] / 100, b30[i][4], mean[jd], rng[jd]))

        def run(sigs, tp, sl):
            busy, out = 0, []
            for T, d, a in sigs:
                if T < busy:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * a, sl * a)
                out.append((T, pnl, r, te, sym, d, float(o[k0]), NOTIONAL * sl * a))
                busy = te if te is not None else 10 ** 15
            return out

        for m in MRCS:
            sigs = [(T, d, a) for T, d, a, px, mn, rg in raw if not cs.hot(m, d, px, mn, rg)]
            for tp, sl in EXITS:
                seq.setdefault(((m, tp, sl), False), {})[sym] = run(sigs, tp, sl)
        for name, _, cb, only_up in RULES:
            if only_up:
                m, tp, sl = cb
                sigs = [(T, d, a) for T, d, a, px, mn, rg in raw
                        if not cs.hot(m, d, px, mn, rg) and btc.above(T)]
                seq.setdefault((cb, True), {})[sym] = run(sigs, tp, sl)
        y0 = max(START, int(t15[0]) + 260 * D1)
        y1 = int(t15[-1]) - 5 * D1
        for tp, sl in ((4.0, 20.0), (5.0, 3.5)):
            rr, tries = [], 0
            rg_ = random.Random("%s-%g-%g" % (sym, tp, sl))
            while len(rr) < RND_N and tries < RND_N * 20 and y1 > y0:
                tries += 1
                T = rg_.randrange(y0, y1)
                d = btc.trend(T)
                j = bisect.bisect_right(b30t, T - M30) - 1
                if d == 0 or j < 20 or atrp[j] <= 0:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * atrp[j] / 100, sl * atrp[j] / 100)
                rr.append((T, pnl, r, te, btc.above(T)))
            rnd.setdefault((tp, sl), {})[sym] = rr
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return seq, rnd, days, have, end_t


def regime_B(seq, have, end_t):
    """Б: одна настройка на все монеты по закрытым сделкам последних 12 месяцев, пересчёт раз в квартал."""
    keys = {k: {s: [x[0] for x in v] for s, v in d.items()} for k, d in seq.items() if not k[1]}

    def part(cb, sym, a, b):
        lst = seq[(cb, False)].get(sym, [])
        kk = keys[(cb, False)].get(sym, [])
        return lst[bisect.bisect_left(kk, a):bisect.bisect_left(kk, b)]

    out, picks = [], []
    y, m = 2020, 1
    while ms(y, m) < end_t:
        ny, nm = (y, m + 3) if m < 10 else (y + 1, 1)
        qs, qe = ms(y, m), min(ms(ny, nm), end_t + 1)
        w0 = qs - 365 * D1
        best = None
        for cb in COMBOS:
            v = [x for s in have for x in part(cb, s, w0, qs) if x[3] is not None and x[3] < qs]
            if len(v) >= 50:
                tot = sum(x[1] for x in v)
                if best is None or tot > best[0]:
                    best = (tot, cb)
        if best and best[0] > 0:
            for s in have:
                out += part(best[1], s, qs, qe)
            picks.append("%d-Q%d %s" % (y, (m - 1) // 3 + 1, cname(best[1])))
        else:
            picks.append("%d-Q%d вне рынка" % (y, (m - 1) // 3 + 1))
        y, m = ny, nm
    return out, picks


def money(trades, days, end_t):
    """Кривая счёта по дневным закрытиям: закрытые сделки + незакрытые по цене закрытия дня."""
    real, mtm, cnt, risk = {}, {}, {}, {}
    for T, pnl, r, te, sym, d, p0, rk in trades:
        dt, dc = days[sym]
        i0 = max(0, bisect.bisect_right(dt, T) - 1)
        i1 = (bisect.bisect_right(dt, te) - 1) if te is not None else len(dt) - 1
        for i in range(i0, max(i0, i1)):
            k = dt[i] + D1
            mtm[k] = mtm.get(k, 0.0) + d * (dc[i] / p0 - 1) * NOTIONAL - NOTIONAL * cs.FEE
            cnt[k] = cnt.get(k, 0) + 1
            risk[k] = risk.get(k, 0.0) + rk
        k = dt[min(i1, len(dt) - 1)] + D1
        real[k] = real.get(k, 0.0) + pnl
    ks = sorted(set(real) | set(mtm))
    cum = peak = 0.0
    dd, dd_t, pk_t, dd_from = 0.0, None, None, None
    for k in ks:
        cum += real.get(k, 0.0)
        eq = cum + mtm.get(k, 0.0)
        if eq > peak or pk_t is None:
            peak, pk_t = max(peak, eq), k
        if eq - peak < dd:
            dd, dd_t, dd_from = eq - peak, k, pk_t
    mc = max(cnt.items(), key=lambda x: x[1]) if cnt else (None, 0)
    mr = max(risk.items(), key=lambda x: x[1]) if risk else (None, 0)
    return {"dd": dd, "dd_t": dd_t, "dd_from": dd_from, "maxpos": mc[1], "maxpos_t": mc[0],
            "maxrisk": mr[1], "maxrisk_t": mr[0], "final": cum}


def describe(label, trades, days, end_t, rnd_avg=None):
    trades = sorted(trades)
    n = len(trades)
    tot = sum(x[1] for x in trades)
    avg = tot / max(1, n)
    wr = sum(1 for x in trades if x[1] > 0) / max(1, n) * 100
    yrs = {}
    for x in trades:
        yrs.setdefault(cs.year_of(x[0]), []).append(x[1])
    if trades:
        mid = (trades[0][0] + end_t) // 2
        h1 = sum(x[1] for x in trades if x[0] < mid)
        h2 = sum(x[1] for x in trades if x[0] >= mid)
    else:
        mid, h1, h2 = 0, 0, 0
    mo = money(trades, days, end_t) if trades else None
    say("  %-40s сделок %5d · WR %4.1f%% · итог %+7.0f $ · %+6.2f $/сд%s" % (
        label, n, wr, tot, avg, (" · случ. вход %+.2f $/сд" % rnd_avg) if rnd_avg is not None else ""))
    if trades:
        say("      половины (до/после %s): %+.0f / %+.0f $ · по годам: %s" % (
            dstr(mid), h1, h2, " · ".join("%d: %+.0f $ (%d)" % (y, sum(v), len(v)) for y, v in sorted(yrs.items()))))
        say("      деньги: макс. просадка счёта %+.0f $ (%s → %s) · позиций одновременно до %d (%s) · сумма их стопов до %.0f $ (%s)"
            % (mo["dd"], dstr(mo["dd_from"]) if mo["dd_from"] else "-", dstr(mo["dd_t"]) if mo["dd_t"] else "-",
               mo["maxpos"], dstr(mo["maxpos_t"]) if mo["maxpos_t"] else "-", mo["maxrisk"],
               dstr(mo["maxrisk_t"]) if mo["maxrisk_t"] else "-"))
        if mo["dd"] < 0:
            say("      итог / просадка = %.2f · депозит, чтобы просадка была не больше 50%%: ≈ %.0f $"
                % (tot / -mo["dd"], -mo["dd"] * 2))
    y_ok = sum(1 for v in yrs.values() if len(v) >= 30 and sum(v) > 0)
    y_n = sum(1 for v in yrs.values() if len(v) >= 30)
    return {"n": n, "tot": tot, "avg": avg, "h1": h1, "h2": h2, "y_ok": y_ok, "y_n": y_n}


def report_set(set_name, coins, btc):
    say("")
    say("#" * 110)
    say("НАБОР «%s»: %s" % (set_name, " ".join(coins)))
    seq, rnd, days, have, end_t = run_set(set_name, coins, btc)
    say("  монет с данными: %d" % len(have))
    res = {}
    for name, label, cb, only_up in RULES:
        trades = [x for s in have for x in seq[(cb, only_up)].get(s, [])]
        tp, sl = cb[1], cb[2]
        rr = [x for s in have for x in rnd[(tp, sl)].get(s, []) if (x[4] or not only_up)]
        ravg = sum(x[1] for x in rr) / max(1, len(rr))
        res[name] = describe("%s %s" % (name, label), trades, days, end_t, ravg)
        res[name]["ravg"] = ravg
    bt, picks = regime_B(seq, have, end_t)
    res["B"] = describe("Б выбор по 12 месяцам", bt, days, end_t)
    say("      выбор Б по кварталам: " + " · ".join(picks))
    return res


def main():
    t0 = time.time()
    say("ПРАВИЛО SMA200 BTC · %s · Binance 15m · позиция 1000 $ (100 × 10)"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = Btc()
    sets = [("основной", MAIN_COINS), ("новый", NEW_COINS)]
    if os.environ.get("SMA_ONLY_NEW"):
        sets = sets[1:]
    out = {nm: report_set(nm, cl, btc) for nm, cl in sets}

    say("")
    say("=" * 110)
    say("КРИТЕРИЙ «R2 — НАХОДКА» (объявлен до прогона)")
    ok_all = True
    if "новый" in out:
        r = out["новый"]
        k1 = r["R2"]["tot"] > 0
        k2 = r["R2"]["avg"] > r["R2"]["ravg"]
        k3 = r["R2"]["tot"] > r["R1"]["tot"]
        k4 = r["R2"]["h1"] > 0 and r["R2"]["h2"] > 0 and r["R2"]["y_n"] > 0 and r["R2"]["y_ok"] * 2 > r["R2"]["y_n"]
        ok_all = k1 and k2 and k3 and k4
        say("  новый набор: итог+ [%s] · лучше случайного входа выше SMA200 [%s, %+.2f против %+.2f] · лучше R1 [%s, %+.0f против %+.0f] · "
            "половины и годы [%s, лет %d из %d]"
            % ("да" if k1 else "нет", "да" if k2 else "нет", r["R2"]["avg"], r["R2"]["ravg"], "да" if k3 else "нет",
               r["R2"]["tot"], r["R1"]["tot"], "да" if k4 else "нет", r["R2"]["y_ok"], r["R2"]["y_n"]))
    else:
        ok_all = False
    if "основной" in out:
        r = out["основной"]
        m1 = r["R2"]["tot"] > 0 and r["R2"]["tot"] > r["R1"]["tot"]
        ok_all = ok_all and m1
        say("  основной набор: итог R2 в плюсе и больше R1 [%s, %+.0f против %+.0f]"
            % ("да" if m1 else "нет", r["R2"]["tot"], r["R1"]["tot"]))
    say("  → %s" % ("R2 — НАХОДКА: правило «4/20 без MRC только выше SMA200» подтвердилось на независимых монетах"
                    if ok_all else "R2 не подтвердилось"))
    for nm, r in out.items():
        say("  справочно, %s набор: R2 %+.0f $ против Б %+.0f $ → %s" % (
            nm, r["R2"]["tot"], r["B"]["tot"], "подбор не нужен, хватает правила" if r["R2"]["tot"] >= r["B"]["tot"]
            else "Б больше"))
        say("  справочно, %s набор: SMA200 и неизменному набору — R3 %+.0f $ против R0 %+.0f $" % (
            nm, r["R3"]["tot"], r["R0"]["tot"]))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

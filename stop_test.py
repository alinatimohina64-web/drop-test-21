#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ ШИРИНЫ СТОПА для ручной торговли по индикатору (Trand-Test-2 на 1h + фильтр BTC 1Д).

Входы, данные и исполнение — как вариант 3 в compare_test.py (лежит рядом, функции берутся
оттуда). Цель всегда 2.5 × ATR 1ч. Меняется только стоп:
  S52 — 5.2 × ATR (как сейчас в индикаторе)
  S40 — 4.0 × ATR
  S30 — 3.0 × ATR
  S20 — 2.0 × ATR
ОДИНАКОВЫЙ РИСК: на стопе всегда теряется 20 $ (2% депозита 1000 $) плюс комиссии.
Позиция = 20 $ / ширина стопа — чем уже стоп, тем больше позиция и тем больше выигрыш
по цели (и тем больше комиссия). Результат считается в $ и в R (R = 20 $).

КРИТЕРИЙ (объявлен ДО прогона). Вариант лучше текущего стопа (S52), если в сценариях
«одна сделка за раз, 30% сигналов пропущено»:
  1) медиана итога в $ выше в ОБЕИХ половинах года;
  2) медианная просадка не больше, чем у S52.
Вариантов 4 — других ширин стопа не перебираем.

Отчёт: stop_report.txt
"""
import bisect
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c

RISK = 20.0
TP_MULT = 2.5
VARIANTS = [("S52", 5.2), ("S40", 4.0), ("S30", 3.0), ("S20", 2.0)]
REPORT = "stop_report.txt"
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def sim(b5, k0, d, atrp, sl_mult):
    tp, sl = TP_MULT * atrp / 100, sl_mult * atrp / 100
    notional = RISK / sl
    p0 = b5[k0][1]
    qty = notional / p0
    tgt, stp = p0 * (1 + d * tp), p0 * (1 - d * sl)
    fund = 0.0

    def done(px, k, maker=False):
        return (d * (qty * px - notional) - notional * c.F_TAKER - qty * px * (c.F_MAKER if maker else c.F_TAKER)
                + fund, b5[k][0] + c.M5)

    for k in range(k0, len(b5)):
        t, o, h, l, cl = b5[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        if d * o <= d * stp:
            return done(o, k)
        if d * adv <= d * stp:
            return done(stp, k)
        if d * fav >= d * tgt:
            return done(o if d * o > d * tgt else tgt, k, True)
        if (t + c.M5) % (8 * c.H1) == 0:
            fund -= d * c.FUND * qty * cl
    return done(b5[-1][4], len(b5) - 1)


def main():
    t0 = time.time()
    coins = [x for x in c.FIRST_RUN if x != "BTC"]
    try:
        coins += [x for x in c.top_coins(50) if x not in coins and x != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    say("ТЕСТ ШИРИНЫ СТОПА · %s · монет %d · %d дней · риск на стопе %.0f $, цель %.1f × ATR 1ч"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), c.DAYS, RISK, TP_MULT))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(c.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_ind(bar_open):
        p = bar_open - bar_open % c.D1 - c.D1
        i = bisect.bisect_left(btc_t, p)
        return btc_tr[i] if i < len(btc_t) and btc_t[i] == p else 0

    trades = {v: [] for v, _ in VARIANTS}
    t_min = t_max = None
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(h4) < 400 or len(d1) < 260:
            continue
        t5 = [x[0] for x in b5]
        h1 = c.agg(b5, c.H1)
        atr1 = c.atr_pct(h1)
        h4a = c.agg(b5, c.H4)
        cl4 = [x[4] for x in h4a]
        h4a_up = [a > b_ for a, b_ in zip(c.ema(cl4, 9), c.ema(cl4, 21))]
        ie = [(T, d, i) for T, d, i in c.ind_entries(h1, [x[0] for x in h4a], h4a_up) if btc_ind(h1[i][0]) == d]
        for T, d, i in ie:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1 or atr1[i] <= 0:
                continue
            for v, m in VARIANTS:
                pnl, te = sim(b5, k0, d, atr1[i], m)
                trades[v].append((T, pnl, te))
        t_min = h1[0][0] if t_min is None else min(t_min, h1[0][0])
        t_max = h1[-1][0] if t_max is None else max(t_max, h1[-1][0])
    if t_min is None:
        say("Нет данных.")
        return
    mid = (t_min + t_max) / 2

    say("")
    say("Все сигналы (без «одна за раз»):")
    for v, m in VARIANTS:
        p = [x[1] for x in trades[v]]
        h = [[x[1] for x in trades[v] if (x[0] < mid) == (k == 1)] for k in (1, 2)]
        say("  %s стоп %.1f ATR · сделок %5d · винрейт %4.1f%% · средняя %+5.2f $ (%+.3f R) · половины %+.3f / %+.3f R"
            % (v, m, len(p), 100 * sum(1 for x in p if x > 0) / len(p), sum(p) / len(p), sum(p) / len(p) / RISK,
               sum(h[0]) / max(len(h[0]), 1) / RISK, sum(h[1]) / max(len(h[1]), 1) / RISK))
    say("")
    say("ОДНА СДЕЛКА ЗА РАЗ, 30%% пропущено, %d сценариев, депозит 1000 $, риск 20 $ на сделку:" % c.SCEN)
    res = {}
    for v, m in VARIANTS:
        r = c.scenarios(trades[v], mid)
        res[v] = r
        say("  %s стоп %.1f ATR · сделок ~%d · итог медиана %+6.0f $ (10%% худших %+5.0f) · в плюсе %3.0f%% · "
            "половины %+5.0f / %+5.0f $ · просадка медиана %4.0f $ (10%% худших %4.0f)"
            % (v, m, c.med(r["n"]), c.med(r["tot"]), r["tot"][c.SCEN // 10],
               100 * sum(1 for x in r["tot"] if x > 0) / c.SCEN, c.med(r["h1"]), c.med(r["h2"]),
               c.med(r["dd"]), r["dd"][c.SCEN * 9 // 10]))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): лучше S52, если медиана выше в обеих половинах и просадка не больше")
    b = res["S52"]
    for v, m in VARIANTS[1:]:
        r = res[v]
        c1 = c.med(r["h1"]) > c.med(b["h1"]) and c.med(r["h2"]) > c.med(b["h2"])
        c2 = c.med(r["dd"]) <= c.med(b["dd"])
        say("  %s стоп %.1f ATR: половины [%s] · просадка [%s] → %s"
            % (v, m, "да" if c1 else "нет", "да" if c2 else "нет", "ЛУЧШЕ" if (c1 and c2) else "не лучше"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

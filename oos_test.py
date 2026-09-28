#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА ТРЕНДОВОГО ВЫХОДА (D3) НА НЕЗАВИСИМОМ ГОДЕ.

Все прошлые тесты индикатора шли на последних 365 днях. Здесь грузятся 2 года; главный —
ПРЕДЫДУЩИЙ год (его не видел ни один тест). Последний год — для справки.

Входы: сигналы Trand-Test-2 на 1h (старший ТФ 4ч), только по дневному тренду BTC — как база
в ideas_test.py. Исполнение, комиссии, фандинг, сценарии — как в compare_test.py.
  БАЗА — цель 2.5 × ATR 1ч, стоп 5.2 × ATR, риск на стопе 20 $.
  D3   — стоп 3 × ATR, цели нет, трейлинг 3 × ATR от лучшей цены после входа, риск 14 $
         (риск уменьшен, чтобы просадка была сравнима с базой — решено ДО этого прогона).
  Для справки: D3 с риском 20 $.

КРИТЕРИЙ (объявлен ДО прогона), на ПРЕДЫДУЩЕМ году. D3 (риск 14 $) лучше базы, если в
сценариях «одна сделка за раз, 30% пропущено»:
  1) медиана итога выше в ОБЕИХ половинах предыдущего года;
  2) медианная просадка не больше, чем у базы.

Лежит рядом с compare_test.py и ideas_test.py. Переменная DAYS должна быть 730
(в workflow так и стоит). Отчёт: oos_report.txt
"""
import bisect
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c
import ideas_test as it

REPORT = "oos_report.txt"
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def sim_fixed(b5, k0, d, atrp, risk, tp_mult=2.5, sl_mult=5.2):
    tp, sl = tp_mult * atrp / 100, sl_mult * atrp / 100
    notional = risk / sl
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


def sim_trail(b5, k0, d, atrp, risk, mult=3.0):
    dist = mult * atrp / 100
    notional = risk / dist
    p0 = b5[k0][1]
    qty = notional / p0
    best, fund = p0, 0.0
    for k in range(k0, len(b5)):
        t, o, h, l, cl = b5[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        stop = best * (1 - d * dist)
        if d * o <= d * stop or d * adv <= d * stop:
            px = o if d * o <= d * stop else stop
            return (d * (qty * px - notional) - notional * c.F_TAKER - qty * px * c.F_TAKER + fund,
                    b5[k][0] + c.M5)
        if d * fav > d * best:
            best = fav
        if (t + c.M5) % (8 * c.H1) == 0:
            fund -= d * c.FUND * qty * cl
    px = b5[-1][4]
    return (d * (qty * px - notional) - notional * c.F_TAKER - qty * px * c.F_TAKER + fund, b5[-1][0] + c.M5)


def main():
    t0 = time.time()
    if c.DAYS < 600:
        say("! DAYS=%d — для проверки нужен 2-летний период, поставьте DAYS=730" % c.DAYS)
    coins = [x for x in c.FIRST_RUN if x != "BTC"]
    try:
        coins += [x for x in c.top_coins(50) if x not in coins and x != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    now = int(time.time() * 1000)
    split = now - 365 * c.D1
    say("ПРОВЕРКА D3 НА НЕЗАВИСИМОМ ГОДЕ · %s · монет %d · %d дней"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), c.DAYS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(c.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_ind(bar_open):
        p = bar_open - bar_open % c.D1 - c.D1
        i = bisect.bisect_left(btc_t, p)
        return btc_tr[i] if i < len(btc_t) and btc_t[i] == p else 0

    V = {"БАЗА": "цель 2.5 / стоп 5.2 ATR, риск 20 $", "D3": "трейлинг 3 ATR, риск 14 $",
         "D3r20": "трейлинг 3 ATR, риск 20 $ (справка)"}
    tr = {y: {k: [] for k in V} for y in ("prev", "last")}
    t_first = None
    used = 0
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(d1) < 260:
            continue
        used += 1
        t5 = [x[0] for x in b5]
        h1 = c.agg(b5, c.H1)
        atr1 = c.atr_pct(h1)
        h4a = c.agg(b5, c.H4)
        c4 = [x[4] for x in h4a]
        up4 = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        sig = [(T, d, i) for T, d, i in it.ind_entries_htf(h1, [x[0] for x in h4a], up4, c.H4)
               if btc_ind(h1[i][0]) == d]
        t_first = h1[0][0] if t_first is None else min(t_first, h1[0][0])
        for T, d, i in sig:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1 or atr1[i] <= 0:
                continue
            y = "prev" if T < split else "last"
            x = sim_fixed(b5, k0, d, atr1[i], 20.0)
            tr[y]["БАЗА"].append((T, x[0], x[1]))
            x = sim_trail(b5, k0, d, atr1[i], 14.0)
            tr[y]["D3"].append((T, x[0], x[1]))
            x = sim_trail(b5, k0, d, atr1[i], 20.0)
            tr[y]["D3r20"].append((T, x[0], x[1]))
    say("монет с данными: %d" % used)
    if t_first is None:
        say("Нет данных.")
        return

    res = {}
    for y, (a, b) in (("prev", (t_first, split)), ("last", (split, now))):
        mid = (a + b) / 2
        title = "ПРЕДЫДУЩИЙ ГОД (независимый)" if y == "prev" else "ПОСЛЕДНИЙ ГОД (справка, его видели тесты)"
        say("")
        say("=" * 110)
        say("%s: %s … %s" % (title, datetime.fromtimestamp(a / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                             datetime.fromtimestamp(b / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
        for k, nm in V.items():
            p = [x[1] for x in tr[y][k]]
            if not p:
                say("  %-6s нет сделок" % k)
                continue
            r = c.scenarios(tr[y][k], mid)
            res[(y, k)] = r
            say("  %-6s %-38s сигналов %5d · винрейт %4.1f%% · средняя %+5.2f $" %
                (k, nm, len(p), 100 * sum(1 for x in p if x > 0) / len(p), sum(p) / len(p)))
            say("         одна за раз: сделок ~%d · итог медиана %+6.0f $ (10%% худших %+5.0f) · в плюсе %3.0f%% · "
                "половины %+5.0f / %+5.0f $ · просадка медиана %4.0f $ (10%% худших %4.0f)"
                % (c.med(r["n"]), c.med(r["tot"]), r["tot"][c.SCEN // 10],
                   100 * sum(1 for x in r["tot"] if x > 0) / c.SCEN, c.med(r["h1"]), c.med(r["h2"]),
                   c.med(r["dd"]), r["dd"][c.SCEN * 9 // 10]))
    say("")
    say("ВЕРДИКТ (объявлен до прогона, по ПРЕДЫДУЩЕМУ году):")
    b, d3 = res.get(("prev", "БАЗА")), res.get(("prev", "D3"))
    if b and d3:
        c1 = c.med(d3["h1"]) > c.med(b["h1"]) and c.med(d3["h2"]) > c.med(b["h2"])
        c2 = c.med(d3["dd"]) <= c.med(b["dd"])
        say("  D3 (риск 14 $) против базы: половины [%s] · просадка [%s] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "D3 ПОДТВЕРЖДЁН" if (c1 and c2) else "не подтверждён"))
    else:
        say("  не хватает данных за предыдущий год")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА ФИЛЬТРА BTC 1Д ДЛЯ БОТА НА НЕЗАВИСИМОМ ГОДЕ.

Фильтр «входы бота только по дневному тренду BTC» принят по compare_test.py на последних
365 днях. Здесь — ПРЕДЫДУЩИЙ год (его не видел ни один тест фильтра). Входы робота 🟢,
сетка GHOST B, исполнение, сценарии — ровно как варианты 1 и 2 в compare_test.py.

КРИТЕРИЙ (объявлен ДО прогона), по ПРЕДЫДУЩЕМУ году. Фильтр подтверждён, если «бот + BTC 1Д»
лучше «бота как есть» в сценариях «одна сделка за раз, 30% пропущено»:
  1) медиана итога выше в ОБЕИХ половинах предыдущего года;
  2) медианная просадка не больше.
Последний год — для справки (должен повторить compare_test).

Лежит рядом с compare_test.py. DAYS=730 (в workflow так и стоит). Отчёт: bot_oos_report.txt
"""
import bisect
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c

REPORT = "bot_oos_report.txt"
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    if c.DAYS < 600:
        say("! DAYS=%d — нужен 2-летний период, поставьте DAYS=730" % c.DAYS)
    coins = [x for x in c.FIRST_RUN if x != "BTC"]
    try:
        coins += [x for x in c.top_coins(50) if x not in coins and x != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    now = int(time.time() * 1000)
    split = now - 365 * c.D1
    say("ПРОВЕРКА ФИЛЬТРА BTC 1Д ДЛЯ БОТА НА НЕЗАВИСИМОМ ГОДЕ · %s · монет %d · %d дней"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), c.DAYS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(c.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at_close(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    V = {"БОТ": "бот как есть", "БОТ+BTC1Д": "только по дневному тренду BTC"}
    tr = {y: {k: [] for k in V} for y in ("prev", "last")}
    t_first, used = None, 0
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(h4) < 400 or len(d1) < 260:
            continue
        used += 1
        t5 = [x[0] for x in b5]
        h1 = c.agg(b5, c.H1)
        atr1 = c.atr_pct(h1)
        t_first = h1[0][0] if t_first is None else min(t_first, h1[0][0])
        for T, d, i in c.bot_entries(h1, h4, d1):
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1:
                continue
            pnl, te = c.sim_grid(b5, k0, d, max(atr1[i], c.MIN_STEP))
            y = "prev" if T < split else "last"
            tr[y]["БОТ"].append((T, pnl, te))
            if btc_at_close(T) == d:
                tr[y]["БОТ+BTC1Д"].append((T, pnl, te))
    say("монет с данными: %d" % used)
    if t_first is None:
        say("Нет данных.")
        return

    res = {}
    for y, (a, b) in (("prev", (t_first, split)), ("last", (split, now))):
        mid = (a + b) / 2
        title = "ПРЕДЫДУЩИЙ ГОД (независимый)" if y == "prev" else "ПОСЛЕДНИЙ ГОД (справка)"
        say("")
        say("=" * 110)
        say("%s: %s … %s" % (title, datetime.fromtimestamp(a / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                             datetime.fromtimestamp(b / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
        for k, nm in V.items():
            p = [x[1] for x in tr[y][k]]
            if not p:
                say("  %-10s нет сделок" % k)
                continue
            r = c.scenarios(tr[y][k], mid)
            res[(y, k)] = r
            say("  %-10s %-32s входов %5d · винрейт %4.1f%% · средняя %+5.2f $"
                % (k, nm, len(p), 100 * sum(1 for x in p if x > 0) / len(p), sum(p) / len(p)))
            say("             одна за раз: сделок ~%d · итог медиана %+6.0f $ (10%% худших %+5.0f) · в плюсе %3.0f%% · "
                "половины %+5.0f / %+5.0f $ · просадка медиана %4.0f $ (10%% худших %4.0f)"
                % (c.med(r["n"]), c.med(r["tot"]), r["tot"][c.SCEN // 10],
                   100 * sum(1 for x in r["tot"] if x > 0) / c.SCEN, c.med(r["h1"]), c.med(r["h2"]),
                   c.med(r["dd"]), r["dd"][c.SCEN * 9 // 10]))
    say("")
    say("ВЕРДИКТ (объявлен до прогона, по ПРЕДЫДУЩЕМУ году):")
    b, f = res.get(("prev", "БОТ")), res.get(("prev", "БОТ+BTC1Д"))
    if b and f:
        c1 = c.med(f["h1"]) > c.med(b["h1"]) and c.med(f["h2"]) > c.med(b["h2"])
        c2 = c.med(f["dd"]) <= c.med(b["dd"])
        say("  БОТ+BTC1Д против БОТ: половины [%s] · просадка [%s] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "ФИЛЬТР ПОДТВЕРЖДЁН" if (c1 and c2) else "не подтверждён"))
    else:
        say("  не хватает данных за предыдущий год")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

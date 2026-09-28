#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОТБОР МОНЕТ ПО ПРОШЛОМУ ГОДУ: торговать только 10 лучших — работает ли это?

Два года данных (DAYS=730). По ПРЕДЫДУЩЕМУ году считается итог каждой монеты, выбираются
10 лучших (не меньше 20 сделок за год). Потом смотрим, как эти 10 торговались в ПОСЛЕДНЕМ
году — по сравнению со всеми монетами и с 10 худшими.

Стратегии (ровно как в прошлых тестах):
  ИНДИКАТОР — сигналы Trand-Test-2 на 1h (старший ТФ 4ч) по дневному тренду BTC, цель 2.5 ATR,
              стоп 5.2 ATR, риск 20 $ (база oos_test.py);
  БОТ       — входы робота 🟢 по дневному тренду BTC, сетка GHOST B (compare_test.py, вариант 2).

КРИТЕРИЙ (объявлен ДО прогона), для каждой стратегии: отбор работает, если в ПОСЛЕДНЕМ году
у топ-10
  1) средняя сделка лучше, чем у всех монет вместе;
  2) медиана итога «одна сделка за раз, 30% пропущено» лучше, чем у всех монет вместе.
Для справки — ранговая корреляция итогов монет в двух годах (0 — связи нет, 1 — полная).

Лежит рядом с compare_test.py, ideas_test.py, oos_test.py. Отчёт: select_report.txt
"""
import bisect
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c
import ideas_test as it
import oos_test as ot

REPORT = "select_report.txt"
TOP = 10
MIN_N = 20
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def spearman(a, b):
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(x):
        o = sorted(range(n), key=lambda i: x[i])
        r = [0] * n
        for k, i in enumerate(o):
            r[i] = k
        return r
    ra, rb = ranks(a), ranks(b)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def main():
    t0 = time.time()
    coins = [x for x in c.FIRST_RUN if x != "BTC"]
    try:
        coins += [x for x in c.top_coins(50) if x not in coins and x != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    now = int(time.time() * 1000)
    split = now - 365 * c.D1
    say("ОТБОР МОНЕТ ПО ПРОШЛОМУ ГОДУ · %s · монет %d · %d дней"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), c.DAYS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(c.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_ind(bar_open):
        p = bar_open - bar_open % c.D1 - c.D1
        i = bisect.bisect_left(btc_t, p)
        return btc_tr[i] if i < len(btc_t) and btc_t[i] == p else 0

    def btc_at_close(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    per = {"ИНДИКАТОР": {}, "БОТ": {}}          # стратегия -> монета -> {"prev": [...], "last": [...]}
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(h4) < 400 or len(d1) < 260 or b5[0][0] > split - 300 * c.D1:
            continue                          # нужна история и за предыдущий год
        t5 = [x[0] for x in b5]
        h1 = c.agg(b5, c.H1)
        atr1 = c.atr_pct(h1)
        h4a = c.agg(b5, c.H4)
        c4 = [x[4] for x in h4a]
        up4 = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        ind = {"prev": [], "last": []}
        for T, d, i in it.ind_entries_htf(h1, [x[0] for x in h4a], up4, c.H4):
            if btc_ind(h1[i][0]) != d:
                continue
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1 or atr1[i] <= 0:
                continue
            pnl, te = ot.sim_fixed(b5, k0, d, atr1[i], 20.0)
            ind["prev" if T < split else "last"].append((T, pnl, te))
        bot = {"prev": [], "last": []}
        for T, d, i in c.bot_entries(h1, h4, d1):
            if btc_at_close(T) != d:
                continue
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1:
                continue
            pnl, te = c.sim_grid(b5, k0, d, max(atr1[i], c.MIN_STEP))
            bot["prev" if T < split else "last"].append((T, pnl, te))
        per["ИНДИКАТОР"][sym] = ind
        per["БОТ"][sym] = bot

    mid_last = (split + now) / 2
    for strat, pc in per.items():
        ok = {s: v for s, v in pc.items() if len(v["prev"]) >= MIN_N and len(v["last"]) >= MIN_N}
        say("")
        say("=" * 110)
        say("%s · монет с историей за оба года (>= %d сделок в каждом): %d" % (strat, MIN_N, len(ok)))
        if len(ok) < 2 * TOP:
            say("  мало монет для отбора")
            continue
        tot_prev = {s: sum(x[1] for x in v["prev"]) for s, v in ok.items()}
        tot_last = {s: sum(x[1] for x in v["last"]) for s, v in ok.items()}
        order = sorted(ok, key=lambda s: -tot_prev[s])
        top, bottom = order[:TOP], order[-TOP:]
        say("  Топ-10 по предыдущему году: " + ", ".join("%s %+.0f→%+.0f" % (s, tot_prev[s], tot_last[s]) for s in top))
        say("  Худшие 10:                 " + ", ".join("%s %+.0f→%+.0f" % (s, tot_prev[s], tot_last[s]) for s in bottom))
        say("  (число слева — итог монеты в предыдущем году, справа — в последнем)")
        rho = spearman([tot_prev[s] for s in ok], [tot_last[s] for s in ok])
        say("  Ранговая корреляция итогов монет между годами: %+.2f" % rho)
        groups = {"все монеты": list(ok), "топ-10": top, "худшие 10": bottom}
        res = {}
        for g, lst in groups.items():
            trades = [x for s in lst for x in ok[s]["last"]]
            p = [x[1] for x in trades]
            r = c.scenarios(trades, mid_last)
            res[g] = (sum(p) / len(p), r)
            say("  ПОСЛЕДНИЙ ГОД, %-10s сделок %5d · средняя %+5.2f $ · одна за раз: медиана %+5.0f $, "
                "в плюсе %3.0f%%, просадка %4.0f $"
                % (g, len(p), sum(p) / len(p), c.med(r["tot"]), 100 * sum(1 for x in r["tot"] if x > 0) / c.SCEN,
                   c.med(r["dd"])))
        a_all, r_all = res["все монеты"]
        a_top, r_top = res["топ-10"]
        c1 = a_top > a_all
        c2 = c.med(r_top["tot"]) > c.med(r_all["tot"])
        say("  ВЕРДИКТ: средняя сделка топ-10 лучше [%s] · итог «одна за раз» лучше [%s] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "ОТБОР РАБОТАЕТ" if (c1 and c2) else "отбор не работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

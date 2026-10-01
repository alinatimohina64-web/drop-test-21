#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СИГНАЛЫ TT2 ПО КАЖДОЙ МОНЕТЕ ОТДЕЛЬНО: без стопа и с широким стопом (20 × ATR). Какие монеты застревают.

Сигналы — проверенный набор: график 30m, старший ТФ 4ч, только по дневному тренду BTC, фильтр MRC
(дневной канал, упрощённая схема, уровень 2). Вход по открытию 15m после сигнала. Каждая монета —
отдельно (как будто на своём счёте), одна позиция за раз; пока сделка открыта, новые сигналы этой
монеты пропускаются. Позиция 1000 $ (маржа 100 × 10). Комиссия 0.11% за круг, фандинг 0.01% за 8 ч
против позиции. Внутри 15m-свечи, где задеты и цель, и стоп, — стоп.

Варианты (объявлены до прогона):
  1) цель 5 × ATR, без стопа        2) цель 5 × ATR, стоп 20 × ATR        3) цель 6 × ATR, стоп 20 × ATR
Сделка без стопа, не дошедшая до цели к концу данных, считается по текущей цене (плавающий результат).

Контроль: случайные входы (столько же в каждой монете и году, в сторону дневного тренда BTC, те же
выходы).

КРИТЕРИЙ (объявлен ДО прогона), для каждого варианта, сумма по монетам:
  1) итог в плюсе в 3 из 4 лет 2023–2026 и в обеих половинах 2022–2026 (по году входа);
  2) средняя сделка лучше случайного входа.
По монетам: «не застревает» — ни одна сделка не висела дольше 30 дней и нет открытых старше 30 дней.
Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py, mrc_filter_test.py.
Монеты: 21 «старая» (по умолчанию) или свой список: переменная WS_COINS=SOL,XRP,...  Отчёт: widestop_report.txt
"""
import bisect
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw
from mrc_filter_test import supersmoother, hot_flag

REPORT = "widestop_report.txt"
M15, M30 = 900000, 1800000
NOTIONAL = 1000.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
VARIANTS = [(5.0, None), (5.0, 20.0), (6.0, 20.0)]
TEST_YEARS = [2023, 2024, 2025, 2026]
STUCK_DAYS = 30
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def trade(t15, h, l, cl, k0, d, p0, tp, sl):
    """tp, sl — доли от входа (sl=None — без стопа). Возвращает ($, время выхода или None, часы, MAE, итог)."""
    if d == 1:
        ht = np.nonzero(h[k0:] >= p0 * (1 + tp))[0]
        hs = np.nonzero(l[k0:] <= p0 * (1 - sl))[0] if sl else np.array([], dtype=int)
    else:
        ht = np.nonzero(l[k0:] <= p0 * (1 - tp))[0]
        hs = np.nonzero(h[k0:] >= p0 * (1 + sl))[0] if sl else np.array([], dtype=int)
    it = int(ht[0]) if len(ht) else None
    is_ = int(hs[0]) if len(hs) else None
    if is_ is not None and (it is None or is_ <= it):
        k, move, how = k0 + is_, -sl, "стоп"
    elif it is not None:
        k, move, how = k0 + it, tp, "цель"
    else:
        k, move, how = None, d * (cl[-1] / p0 - 1), "открыта"
    end = (k + 1) if k is not None else len(h)
    mae = ((p0 - l[k0:end].min()) / p0) if d == 1 else ((h[k0:end].max() - p0) / p0)
    t_end = int(t15[k]) + M15 if k is not None else int(t15[-1])
    hours = (t_end - int(t15[k0])) / 3600000
    pnl = NOTIONAL * move - NOTIONAL * FEE - NOTIONAL * FUND_8H * hours / 8
    return pnl, (t_end if k is not None else None), hours, max(0.0, mae), how


def main():
    t0 = time.time()
    env = os.environ.get("WS_COINS")
    coins = [x.strip().upper() for x in env.split(",")] if env else list(mt.MONEY)
    say("TT2 ПО МОНЕТАМ: БЕЗ СТОПА И СО СТОПОМ 20 ATR · %s · монет %d · позиция %.0f $ (100 × 10)"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), NOTIONAL))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    rnd = random.Random(11)
    res = {v: {"sig": [], "rnd": []} for v in VARIANTS}       # (T, монета, $, часы, MAE, как, открыта?)
    now = 0
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 20000 or len(d1) < 400:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        now = max(now, int(t15[-1]))
        b30 = c.agg(b15, M30)
        b30t = [x[0] for x in b30]
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        dt = [x[0] for x in d1]
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        sigs = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, c.H4):
            if T < mt.START or atrp[i] <= 0 or btc_at(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - c.D1) - 1
            if jd < 250 or hot_flag("simple", d, b30[i][4], mean[jd], rng[jd]):
                continue
            sigs.append((T, d, atrp[i] / 100))
        for v in VARIANTS:
            tpm, slm = v
            busy = 0
            per_year = {}
            for T, d, a in sigs:
                if T < busy:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                pnl, te, hours, mae, how = trade(t15, h, l, cl, k0, d, o[k0], tpm * a, slm * a if slm else None)
                res[v]["sig"].append((T, sym, pnl, hours, mae, how))
                busy = te if te is not None else 10 ** 15
                per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
            for y, cnt in per_year.items():
                y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                y1 = min(int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), int(t15[-1]) - c.D1)
                got = tries = 0
                while got < cnt and tries < cnt * 20 and y1 > y0:
                    tries += 1
                    T = rnd.randrange(y0, y1)
                    d = btc_at(T)
                    j = bisect.bisect_right(b30t, T - M30) - 1
                    if d == 0 or j < 20 or atrp[j] <= 0:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    a = atrp[j] / 100
                    r = trade(t15, h, l, cl, k0, d, o[k0], tpm * a, slm * a if slm else None)
                    res[v]["rnd"].append((T, sym, r[0], r[2], r[3], r[4]))
                    got += 1
        say("  %s: сигналов после фильтров %d · %.0f с" % (sym, len(sigs), time.time() - t0))

    mid = mt.START + (now - mt.START) // 2
    for v in VARIANTS:
        tpm, slm = v
        tr = res[v]["sig"]
        rv = res[v]["rnd"]
        say("")
        say("=" * 110)
        say("ЦЕЛЬ %g × ATR · %s" % (tpm, "БЕЗ СТОПА" if slm is None else "СТОП %g × ATR" % slm))
        n = len(tr)
        if not n:
            say("  сделок нет")
            continue
        tot = sum(x[2] for x in tr)
        n_t = sum(1 for x in tr if x[5] == "цель")
        n_s = sum(1 for x in tr if x[5] == "стоп")
        n_o = sum(1 for x in tr if x[5] == "открыта")
        hold = sorted(x[3] / 24 for x in tr if x[5] != "открыта")
        say("  сделок %d · цель %d (%.1f%%) · стоп %d · открыты на конец %d · итог %+.0f $ (%+.2f $ на сделку) · "
            "случайный вход: %+.2f $ на сделку"
            % (n, n_t, 100 * n_t / n, n_s, n_o, tot, tot / n, sum(x[2] for x in rv) / max(1, len(rv))))
        if hold:
            say("  время в сделке: медиана %.1f дн · 90%% быстрее %.1f дн · дольше всего %.0f дн"
                % (hold[len(hold) // 2], hold[int(len(hold) * 0.9)], hold[-1]))
        maes = sorted(x[4] for x in tr)
        say("  просадка внутри сделки: медиана %.1f%% · 10%% худших > %.1f%% · худшая %.1f%% (%.0f $ на позиции 1000 $)"
            % (100 * maes[len(maes) // 2], 100 * maes[int(len(maes) * 0.9)], 100 * maes[-1], NOTIONAL * maes[-1]))
        yrs = {}
        for x in tr:
            yrs.setdefault(year_of(x[0]), []).append(x[2])
        say("  по году входа: " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(p), len(p)) for y, p in sorted(yrs.items())))
        h1_ = sum(x[2] for x in tr if x[0] < mid)
        h2_ = sum(x[2] for x in tr if x[0] >= mid)
        say("  половины: %+.0f $ / %+.0f $" % (h1_, h2_))
        say("  ПО МОНЕТАМ (сделок · цель · стоп · открыто · дольше всего · худшая просадка · итог):")
        rows = []
        for sym in sorted({x[1] for x in tr}):
            ps = [x for x in tr if x[1] == sym]
            long_ = max(x[3] for x in ps) / 24
            stuck_open = any(x[5] == "открыта" and x[3] / 24 > STUCK_DAYS for x in ps)
            ok = long_ <= STUCK_DAYS and not stuck_open
            rows.append((ok, sum(x[2] for x in ps), sym, len(ps), sum(1 for x in ps if x[5] == "цель"),
                         sum(1 for x in ps if x[5] == "стоп"), sum(1 for x in ps if x[5] == "открыта"), long_,
                         max(x[4] for x in ps)))
        for ok, tot_c, sym, n_, nt, ns, no, long_, worst in sorted(rows, key=lambda r: (-r[0], -r[1])):
            say("    %-5s %3d · %3d · %2d · %d · %5.0f дн · %5.1f%% · %+6.0f $ %s"
                % (sym, n_, nt, ns, no, long_, 100 * worst, tot_c, "✔ не застревает" if ok else ""))
        c1 = sum(1 for y in TEST_YEARS if sum(yrs.get(y, [0.0])) > 0) >= 3 and h1_ > 0 and h2_ > 0
        c2 = tot / n > sum(x[2] for x in rv) / max(1, len(rv))
        say("  критерии: 3 из 4 лет и обе половины в плюсе [%s] · лучше случайного входа [%s] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "РАБОТАЕТ" if (c1 and c2) else "не работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

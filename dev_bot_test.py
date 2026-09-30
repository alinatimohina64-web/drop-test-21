#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТРЕНДОВЫЙ БОТ С НАСТРОЙКАМИ РАЗРАБОТЧИКА + НАШИ ФИЛЬТРЫ — проверка на 21 монете, 2022–2026.

Сетка (видео разработчика 30.09.2026): 4 ордера по 100 $ (депозит 1100), доливки ПО ТРЕНДУ на +6%, +10.8%,
+14.64% от первого входа (шаг 6%, динамический шаг 0.8), тейк 10% от средней, трейлинг 50% после первой
доливки. Стоп — два варианта (оба объявлены заранее): «без стопа» (как у разработчика; позиция закрывается
по рынку через 30 дней, если ни тейк, ни трейлинг не сработали — ручное ведение тест не моделирует)
и «стоп 9% от первого входа».
Исполнение на 15m, спорная свеча — в худшую сторону (как в исправленных тестах), комиссия 0.055%
тейкер / 0.02% мейкер на тейке, фандинг 0.01% за 8 ч против позиции. Одна позиция на монету за раз.

Входы (два варианта): «1ч» — разворот 1Ч по тренду 4Ч/1Д (робот 🟢), «4ч» — разворот 4Ч по тренду 1Д.
Фильтры для обоих: вход только по дневному тренду BTC и НЕ выше границы MRC (канал по дневкам,
упрощённая схема, уровень 2) для лонга / не ниже — для шорта.

Контроль: случайный вход — столько же входов в каждой монете и году, в случайные моменты, в сторону
дневного тренда BTC, с теми же выходами.

КРИТЕРИИ (объявлены ДО прогона), для каждого варианта (вход × стоп):
  1) сумма $ в плюсе в 3 из 4 лет 2023–2026 и в обеих половинах 2022–2026;
  2) средняя сделка лучше случайного входа;
  3) отбор монет: 10 лучших монет по году N в году N+1 лучше всех монет (средняя сделка) в 3 из 4
     переходов 2022→23 … 2025→26 — только тогда «найти десяток прибыльных монет» имеет смысл.
Лежит рядом с compare_test.py, multi_test.py (данные — из их кэша). Отчёт: dev_bot_report.txt
"""
import bisect
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c
import multi_test as mt

REPORT = "dev_bot_report.txt"
M15 = 900000
VOL = 100.0
LV = [0.0, 0.06, 0.06 + 0.048, 0.06 + 0.048 + 0.0384]
TP = 0.10
KEEP = 0.5
F_T, F_M = 0.055 / 100, 0.02 / 100
FUND = 0.01 / 100
MAXB = 30 * 96
TEST_YEARS = [2023, 2024, 2025, 2026]
TOPN = 10
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def supersmoother(src, n):
    a1 = math.exp(-math.sqrt(2) * math.pi / n)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi / n)
    c3 = -a1 * a1
    c1 = 1 - b1 - c3
    out = []
    for i, v in enumerate(src):
        s1 = out[i - 1] if i >= 1 else v
        s2 = out[i - 2] if i >= 2 else s1
        out.append(c1 * v + b1 * s1 + c3 * s2)
    return out


def sim_dev(t, o, h, l, cl, k0, d, stop):
    """Сетка разработчика; возвращает ($, время выхода). stop — доля от первого входа или None."""
    p0 = o[k0]
    prices = [p0 * (1 + d * x) for x in LV]
    qty, cost, n = VOL / p0, VOL, 1
    fees, fund = VOL * F_T, 0.0
    sl_px = p0 * (1 - d * stop) if stop else None
    peak = None
    k1 = min(len(o), k0 + MAXB)

    def done(px, k, maker=False):
        return d * (qty * px - cost) - fees - qty * px * (F_M if maker else F_T) + fund, t[k] + M15

    for k in range(k0, k1):
        ok_, fav, adv = o[k], (h[k] if d == 1 else l[k]), (l[k] if d == 1 else h[k])
        avg = cost / qty
        st = sl_px
        if n >= 2:
            trl = avg + KEEP * (peak - avg)
            if st is None or d * trl > d * st:
                st = trl
        if st is not None:
            if d * ok_ <= d * st:
                return done(ok_, k)
            if d * adv <= d * st:
                return done(st, k)
        while n < 4 and d * fav >= d * prices[n]:
            px = ok_ if d * ok_ > d * prices[n] else prices[n]
            qty += VOL / px
            cost += VOL
            fees += VOL * F_T
            n += 1
            if peak is None or d * px > d * peak:
                peak = px
        avg = cost / qty
        tp_px = avg * (1 + d * TP)
        if d * fav >= d * tp_px:
            return done(ok_ if d * ok_ > d * tp_px else tp_px, k, True)
        if n >= 2:
            if d * fav > d * peak:
                peak = fav
            trl = avg + KEEP * (peak - avg)
            if d * adv <= d * trl:                     # спорная свеча — в худшую сторону
                return done(trl, k)
        if (t[k] + M15) % (8 * c.H1) == 0:
            fund -= d * FUND * qty * cl[k]
    return done(cl[k1 - 1], k1 - 1)


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("ТРЕНДОВЫЙ БОТ РАЗРАБОТЧИКА + ФИЛЬТРЫ · %s · монет %d · 4 × %.0f $, шаг 6/4.8/3.84%%, тейк 10%%, трейлинг 50%%"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), VOL))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    rnd = random.Random(2026)
    variants = [(e, s) for e in ("1ч", "4ч") for s in (None, 0.09)]
    res = {v: [] for v in variants}                    # (T, монета, $)
    rand = {v: [] for v in variants}
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 50000 or len(d1) < 400:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t = [x[0] for x in b15]
        o = [x[1] for x in b15]
        h = [x[2] for x in b15]
        l = [x[3] for x in b15]
        cl = [x[4] for x in b15]
        h1 = c.agg(b15, c.H1)
        h4a = c.agg(b15, c.H4)
        dt = [x[0] for x in d1]
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)

        def mrc_ok(T, d, px):
            jd = bisect.bisect_right(dt, T - c.D1) - 1
            if jd < 250:
                return False
            thr = (math.pi * rng[jd] + math.pi * 2.415 * rng[jd]) / 2
            return not ((d == 1 and px >= mean[jd] + thr) or (d == -1 and px <= mean[jd] - thr))

        entries = {"1ч": [(T, d, h1[i][4]) for T, d, i in c.bot_entries(h1, h4a, d1)],
                   "4ч": [(T, d, h4a[i][4]) for T, d, i in mt.bot_entries_4h(h4a, d1)]}
        for (en, st) in variants:
            busy = 0
            per_year = {}
            for T, d, px in entries[en]:
                if T < mt.START or T < busy or btc_at(T) != d or not mrc_ok(T, d, px):
                    continue
                k0 = bisect.bisect_left(t, T)
                if k0 >= len(t) - 1:
                    continue
                pnl, te = sim_dev(t, o, h, l, cl, k0, d, st)
                res[(en, st)].append((T, sym, pnl))
                busy = te
                per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
            for y, cnt in per_year.items():                  # случайные входы
                y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
                y1 = min(int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), t[-1] - 31 * c.D1)
                got = tries = 0
                while got < cnt and tries < cnt * 30 and y1 > y0:
                    tries += 1
                    T = rnd.randrange(y0, y1)
                    d = btc_at(T)
                    if d == 0:
                        continue
                    k0 = bisect.bisect_left(t, T)
                    if k0 >= len(t) - 1:
                        continue
                    rand[(en, st)].append((T, sym, sim_dev(t, o, h, l, cl, k0, d, st)[0]))
                    got += 1
        say("  %s: сделок %s · %.0f с" % (sym, " / ".join(str(sum(1 for x in res[v] if x[1] == sym)) for v in variants),
                                           time.time() - t0))

    now = int(time.time() * 1000)
    mid = mt.START + (now - mt.START) // 2
    for v in variants:
        en, st = v
        tr = res[v]
        say("")
        say("=" * 110)
        say("ВХОД %s · %s" % (en, "без стопа (выход по рынку через 30 дней)" if st is None else "стоп 9% от первого входа"))
        n = len(tr)
        if not n:
            say("  сделок нет")
            continue
        tot = sum(x[2] for x in tr)
        wr = 100 * sum(1 for x in tr if x[2] > 0) / n
        rv = [x[2] for x in rand[v]]
        say("  сделок %d · WR %.1f%% · средняя %+.2f $ · итог %+.0f $ · случайный вход: средняя %+.2f $ (%d)"
            % (n, wr, tot / n, tot, sum(rv) / max(1, len(rv)), len(rv)))
        yrs = {}
        for T, s_, p in tr:
            yrs.setdefault(year_of(T), []).append(p)
        say("  по годам: " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(p), len(p)) for y, p in sorted(yrs.items())))
        h1_ = sum(x[2] for x in tr if x[0] < mid)
        h2_ = sum(x[2] for x in tr if x[0] >= mid)
        say("  половины: %+.0f $ / %+.0f $" % (h1_, h2_))
        per_coin = {}
        for T, s_, p in tr:
            per_coin[s_] = per_coin.get(s_, 0.0) + p
        say("  по монетам: " + ", ".join("%s %+.0f" % (k, x) for k, x in sorted(per_coin.items(), key=lambda kv: -kv[1])))
        c1 = sum(1 for y in TEST_YEARS if sum(yrs.get(y, [0.0])) > 0) >= 3 and h1_ > 0 and h2_ > 0
        c2 = tot / n > sum(rv) / max(1, len(rv))
        # отбор монет: топ-10 по году N → год N+1
        wins = 0
        lines = []
        for y0 in (2022, 2023, 2024, 2025):
            prev = {}
            for T, s_, p in tr:
                if year_of(T) == y0:
                    prev[s_] = prev.get(s_, 0.0) + p
            top = [k for k, _ in sorted(prev.items(), key=lambda kv: -kv[1])[:TOPN]]
            nxt = [x for x in tr if year_of(x[0]) == y0 + 1]
            a_all = sum(x[2] for x in nxt) / max(1, len(nxt))
            sel = [x for x in nxt if x[1] in top]
            a_top = sum(x[2] for x in sel) / max(1, len(sel))
            ok = len(sel) >= 10 and a_top > a_all
            wins += ok
            lines.append("%d→%d: топ-10 %+.2f $ против всех %+.2f $ %s" % (y0, y0 + 1, a_top, a_all, "✔" if ok else "✖"))
        c3 = wins >= 3
        say("  отбор монет: " + " · ".join(lines))
        say("  критерии: 3 из 4 лет и обе половины в плюсе [%s] · лучше случайного входа [%s] · отбор 10 монет "
            "работает [%s, %d из 4] → %s"
            % ("да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет", wins,
               "СТРАТЕГИЯ РАБОТАЕТ" if (c1 and c2) else "стратегия не работает"))
        if c1 and c2 and c3:
            best = sorted(((k, sum(x[2] for x in tr if x[1] == k and year_of(x[0]) == 2026)) for k in per_coin),
                          key=lambda kv: -kv[1])[:TOPN]
            say("  монеты для торговли (лучшие по 2026): " + ", ".join("%s %+.0f" % kv for kv in best))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

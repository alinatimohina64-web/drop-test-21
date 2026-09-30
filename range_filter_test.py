#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФИЛЬТР «ЦЕЛЬ ВНУТРИ ОБЫЧНОГО ДНЯ» (как в индикаторе v4.13) — проверка на 21 монете, 2022–2026.

Сигналы: индикатор TT2, график 30m, старший ТФ 4ч, фильтр BTC 1Д «требовать» (как в coin_wf_test).
Вход по открытию 15m после сигнала, цель и стоп от входа в × ATR графика, спорная свеча — стоп,
комиссия 0.11% за круг, не дольше 30 дней. Две пары цели/стопа: 2.5/2.5 и 5/3.5 × ATR.

Фильтр: у сигнала «цель внутри», если цель не дальше линии «обычного дня» на момент сигнала:
  лонг: цель <= минимум текущего UTC-дня (до сигнала) + обычный размах;
  шорт: цель >= максимум дня − обычный размах;
  обычный размах = медиана (макс − мин) / закрытие дневок монеты за прошлые 90 дней.

КРИТЕРИЙ (объявлен ДО прогона). Фильтр работает, если для ОБЕИХ пар (сумма по всем монетам):
  1) сделки «цель внутри» в среднем лучше всех сделок (R на сделку) в обеих половинах 2022–2026
     и в 3 из 4 лет 2023–2026;
  2) разница «внутри» против «за линией» статистически заметна: t >= 2.
Результат — в R и в $ при риске 20 $. Отчёт: range_filter_report.txt
Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py (данные — из их кэша).
"""
import bisect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw

REPORT = "range_filter_report.txt"
M15, M30 = 900000, 1800000
FEE = 0.11 / 100
MAX_BARS = 30 * 96
RISK = 20.0
PAIRS = [(2.5, 2.5), (5.0, 3.5)]
TEST_FROM = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TEST_YEARS = [2023, 2024, 2025, 2026]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def exit_R(o, h, l, cl, k0, d, tp, sl):
    k1 = min(len(o), k0 + MAX_BARS)
    p0 = o[k0]
    tgt, stp = p0 * (1 + d * tp), p0 * (1 - d * sl)
    for k in range(k0, k1):
        fav, adv = (h[k], l[k]) if d == 1 else (l[k], h[k])
        if d * adv <= d * stp:
            return -1.0 - FEE / sl
        if d * fav >= d * tgt:
            return tp / sl - FEE / sl
    return d * (cl[k1 - 1] - p0) / p0 / sl - FEE / sl


def mean_t(v):
    n = len(v)
    if n < 3:
        return (sum(v) / n if n else 0.0), 0.0, n
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, (mu / (sd / math.sqrt(n)) if sd > 0 else 0.0), n


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("ФИЛЬТР «ЦЕЛЬ ВНУТРИ ОБЫЧНОГО ДНЯ» · %s · монет %d · 30m / старший 4ч / BTC 1Д · риск %.0f $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), RISK))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    res = {p: [] for p in PAIRS}                      # (T, внутри?, R)
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 50000 or len(d1) < 260:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        b30 = c.agg(b15, M30)
        atr30 = c.atr_pct(b30)
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        dd = c.agg(b15, c.D1)
        dt = [x[0] for x in dd]
        rng = [(x[2] - x[3]) / x[4] for x in dd]
        med = [None] * len(dd)
        for j in range(90, len(dd)):
            s = sorted(rng[j - 90:j])
            med[j] = s[45]
        n_in = n_all = 0
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, c.H4):
            if T < mt.START or atr30[i] <= 0 or btc_at(T) != d:
                continue
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                continue
            day0 = T - T % c.D1 if T % c.D1 else T - c.D1          # сигнал ровно в 00:00 — это конец прошлого дня
            jd = bisect.bisect_left(dt, day0)
            if jd >= len(dd) or dt[jd] != day0 or med[jd] is None:
                continue
            ka = int(np.searchsorted(t15, day0, side="left"))
            if ka >= k0:
                continue
            dhi, dlo = h[ka:k0].max(), l[ka:k0].min()
            px = cl[k0 - 1]
            m_abs = med[jd] * px
            a = atr30[i] / 100
            for tp, sl in PAIRS:
                tgt = px * (1 + d * tp * a)
                ok = tgt <= dlo + m_abs if d == 1 else tgt >= dhi - m_abs
                r = exit_R(o, h, l, cl, k0, d, tp * a, sl * a)
                res[(tp, sl)].append((T, ok, r))
                if (tp, sl) == PAIRS[0]:
                    n_all += 1
                    n_in += ok
        say("  %s: сигналов %d · цель внутри (2.5/2.5) %.0f%% · %.0f с"
            % (sym, n_all, 100 * n_in / max(1, n_all), time.time() - t0))

    now = int(time.time() * 1000)
    mid = TEST_FROM + (now - TEST_FROM) // 2
    verdicts = []
    for tp, sl in PAIRS:
        v = res[(tp, sl)]
        say("")
        say("=" * 110)
        say("ПАРА цель %g / стоп %g × ATR (проверка 2022–2026; R — в рисках сделки, $ — при риске %.0f $)" % (tp, sl, RISK))

        def line(nm, arr):
            mu, t, n = mean_t(arr)
            wr = 100 * sum(1 for x in arr if x > 0) / max(1, n)
            return "    %-20s сделок %5d · WR %4.1f%% · %+.3f R/сд · итог %+7.0f $" % (nm, n, wr, mu, sum(arr) * RISK)

        test = [x for x in v if x[0] >= TEST_FROM]
        rows = [("все сделки", [x[2] for x in test]), ("цель внутри", [x[2] for x in test if x[1]]),
                ("цель за линией", [x[2] for x in test if not x[1]])]
        for nm, arr in rows:
            say(line(nm, arr))
        better_y = 0
        for y in [2022] + TEST_YEARS:
            a_all = [x[2] for x in test if year_of(x[0]) == y]
            a_in = [x[2] for x in test if year_of(x[0]) == y and x[1]]
            m_all, _, _ = mean_t(a_all)
            m_in, _, _ = mean_t(a_in)
            if y in TEST_YEARS:
                better_y += m_in > m_all
            say("    %d: все %+.3f R (%d) · внутри %+.3f R (%d) · %s" % (y, m_all, len(a_all), m_in, len(a_in),
                                                                     "лучше" if m_in > m_all else "не лучше"))
        halves = []
        for nm, cond in (("1-я половина", lambda T: T < mid), ("2-я половина", lambda T: T >= mid)):
            a_all = [x[2] for x in test if cond(x[0])]
            a_in = [x[2] for x in test if cond(x[0]) and x[1]]
            m_all, _, _ = mean_t(a_all)
            m_in, _, _ = mean_t(a_in)
            halves.append(m_in > m_all)
            say("    %s: все %+.3f R · внутри %+.3f R" % (nm, m_all, m_in))
        a_in = [x[2] for x in test if x[1]]
        a_out = [x[2] for x in test if not x[1]]
        m1, _, n1 = mean_t(a_in)
        m2, _, n2 = mean_t(a_out)
        v1 = sum((x - m1) ** 2 for x in a_in) / max(1, n1 - 1)
        v2 = sum((x - m2) ** 2 for x in a_out) / max(1, n2 - 1)
        t_diff = (m1 - m2) / math.sqrt(v1 / max(1, n1) + v2 / max(1, n2)) if n1 > 2 and n2 > 2 else 0.0
        ok = all(halves) and better_y >= 3 and t_diff >= 2
        verdicts.append(ok)
        say("    разница «внутри» − «за линией»: %+.3f R · t = %+.1f" % (m1 - m2, t_diff))
        say("    критерий: обе половины [%s] · 3 из 4 лет [%s, %d] · t >= 2 [%s] → %s"
            % ("да" if all(halves) else "нет", "да" if better_y >= 3 else "нет", better_y,
               "да" if t_diff >= 2 else "нет", "ФИЛЬТР РАБОТАЕТ" if ok else "не работает"))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): %s" % ("ФИЛЬТР РАБОТАЕТ (для обеих пар)" if all(verdicts)
                                             else "фильтр не работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОНТРОЛЬ НАХОДКИ SMA200: «СДВИНУТЫЙ РЕЖИМ».

Вопрос: правило R2 (TT2 без MRC, цель 4 / стоп 20 ATR, входы только когда BTC вчера закрылся выше SMA200)
зарабатывает потому, что SMA200 действительно отмечает хорошие периоды, — или любое «вкл/выкл» с такой же
долей и длиной периодов дало бы столько же?

Как проверяем: на НАСТОЯЩИХ данных берём настоящую последовательность дней «выше / ниже SMA200» (с 2019 года)
и сдвигаем её по кругу на случайное число дней (не меньше 180 и не больше длины минус 180) — 200 раз.
Длины и доля периодов сохраняются, меняется только КОГДА они приходятся. Для каждого сдвига считается
итог R2 в $ (одна позиция на монету, всё остальное как в sma200_test.py: 30m, ст. ТФ 4ч, тренд BTC 1Д,
15m Binance, позиция 1000 $, в спорной свече — стоп, 250 дневок у монеты).

КРИТЕРИЙ (объявлен ДО прогона): на КАЖДОМ наборе монет (основной и новый) доля сдвигов, давших итог не меньше
настоящего, — не больше 5% (p ≤ 0.05). Тогда SMA200 ловит хорошие периоды неслучайно.
Справочно печатается то же для неизменного набора (MRC упр. 2, 5/3.5) — R3.
Лежит рядом с sma200_test.py и coin_select_test.py. Отчёт: sma200_control_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st

c, cw = cs.c, cs.cw
REPORT = "sma200_control_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
N_SHIFT = 200
RULES = [("R2 без MRC 4/20", "none", 4.0, 20.0), ("R3 MRC упр.2 5/3.5", "simple", 5.0, 3.5)]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    say("КОНТРОЛЬ SMA200 «СДВИНУТЫЙ РЕЖИМ» · %s · %d сдвигов · Binance 15m · позиция 1000 $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), N_SHIFT))
    btc = st.Btc()
    i0 = bisect.bisect_left(btc.t, st.START) - 1
    i0 = max(i0, 200)
    mask = [btc.sma[i] is not None and btc.cl[i] > btc.sma[i] for i in range(len(btc.t))]
    base = mask[i0:]
    L = len(base)
    say("дней в разметке: %d · выше SMA200: %.0f%%" % (L, 100 * sum(base) / L))
    rnd = random.Random(2026)
    shifts = [0] + [rnd.randrange(180, L - 180) for _ in range(N_SHIFT)]

    def above_fn(k):
        def f(T):
            i = btc.idx(T)
            if i < i0:
                return False
            return base[(i - i0 + k) % L]
        return f

    verdict = {}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS)):
        say("")
        say("#" * 100)
        say("НАБОР «%s»" % set_name)
        per = {r[0]: [] for r in RULES}       # правило -> по монетам [(T, d, pnl, te)] для всех сигналов
        for sym in coins:
            b15 = cs.load_15m(sym)
            if len(b15) < 20000:
                say("  %s: мало данных — пропуск" % sym)
                continue
            t15 = np.array([x[0] for x in b15], dtype=np.int64)
            o = np.array([x[1] for x in b15])
            h = np.array([x[2] for x in b15])
            l = np.array([x[3] for x in b15])
            cl = np.array([x[4] for x in b15])
            b30 = c.agg(b15, M30)
            atrp = c.atr_pct(b30)
            h4a = c.agg(b15, H4)
            c4 = [x[4] for x in h4a]
            hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
            d1 = c.agg(b15, D1)
            dt = [x[0] for x in d1]
            mean = cs.supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
            rng = cs.supersmoother(c.trs(d1), 200)
            raw = []
            for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
                if T < st.START or atrp[i] <= 0 or btc.trend(T) != d:
                    continue
                jd = bisect.bisect_right(dt, T - D1) - 1
                if jd < 250:
                    continue
                raw.append((T, d, atrp[i] / 100, b30[i][4], mean[jd], rng[jd]))
            for name, m, tp, sl in RULES:
                out = []
                for T, d, a, px, mn, rg in raw:
                    if cs.hot(m, d, px, mn, rg):
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * a, sl * a)
                    out.append((T, pnl, te))
                per[name].append(out)
            say("  %s: сигналов %d · %.0f с" % (sym, len(raw), time.time() - t0))

        def total(name, f):
            s = 0.0
            n = 0
            for out in per[name]:
                busy = 0
                for T, pnl, te in out:
                    if T < busy or not f(T):
                        continue
                    s += pnl
                    n += 1
                    busy = te if te is not None else 10 ** 15
            return s, n

        for name, m, tp, sl in RULES:
            real, n_real = total(name, above_fn(0))
            sh = [total(name, above_fn(k))[0] for k in shifts[1:]]
            ge = sum(1 for v in sh if v >= real)
            p = (ge + 1) / (len(sh) + 1)
            sh_sorted = sorted(sh)
            say("  %s: настоящий режим %+.0f $ (%d сд) · сдвинутые: медиана %+.0f $, 95%% из них ниже %+.0f $, "
                "лучший %+.0f $ · не хуже настоящего %d из %d → p = %.3f"
                % (name, real, n_real, sh_sorted[len(sh) // 2], sh_sorted[int(len(sh) * 0.95)], sh_sorted[-1],
                   ge, len(sh), p))
            if name.startswith("R2"):
                verdict[set_name] = p <= 0.05

    say("")
    say("=" * 100)
    say("КРИТЕРИЙ (объявлен до прогона): p ≤ 0.05 на обоих наборах — " + " · ".join(
        "%s: %s" % (k, "да" if v else "нет") for k, v in verdict.items()))
    say("→ %s" % ("SMA200 ловит хорошие периоды НЕСЛУЧАЙНО — находка подтверждена контролем"
                  if verdict and all(verdict.values()) else
                  "контроль НЕ пройден — сдвинутый режим часто даёт столько же, находку считать сомнительной"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

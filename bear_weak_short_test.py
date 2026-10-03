#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МЕДВЕЖКА: ШОРТ САМЫХ СЛАБЫХ МОНЕТ ОТНОСИТЕЛЬНО BTC (относительная сила) — без сигнала TT2.

Логика: в медвежке слабые монеты падают быстрее рынка и хуже отскакивают. Раз в H дней (по закрытым дневкам,
без подглядывания) берём K = 5 монет набора с самым слабым результатом относительно BTC за L дней
(доходность монеты − доходность BTC) и шортим их на H дней. Новый отбор — после выхода (без наложения).
Режим (объявлен до прогона): «ниже» — BTC вчера закрылся ниже дневной SMA200; «наклон» — ещё и SMA200 за 30 дней
пошла вниз. Вход — по открытию 15m после закрытия дня; выход — по времени (открытие через H дней) или по стопу
(m × дневной ATR монеты от входа, в спорной свече — стоп). Комиссия 0.11% за круг, фандинг 0.01% / 8 ч против
позиции (худший случай). Три набора монет (основной 21, новый 20, третий 40) — отбор внутри своего набора.

ВАРИАНТЫ (16, объявлены до прогона): L 14 / 30 дней × H 3 / 7 дней × стоп нет / 3 × ATR дня × режим «ниже» / «наклон».
КОНТРОЛЬ: в те же моменты — K случайных монет того же набора (20 повторов), те же выходы. Если «слабые» не лучше
случайных, относительная сила не работает.

РЕЗУЛЬТАТ: средняя доходность сделки в % (и в R для вариантов со стопом, R = доходность / стоп).
КРИТЕРИЙ (объявлен ДО прогона): вариант проходит, если на ВСЕХ ТРЁХ наборах средняя сделка в плюсе в ОБЕИХ половинах
периода И лучше случайных монет. Правило принимается, только если прошёл хотя бы один вариант И в плюсе (по средней
сделке на всех трёх наборах) не меньше 2/3 вариантов.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: bear_weak_short_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

c = cs.c
REPORT = "bear_weak_short_report.txt"
M15, D1 = cs.M15, cs.D1
K = 5
LOOKS = [14, 30]
HOLDS = [3, 7]
STOPS = [None, 3.0]
REGS = ["ниже", "наклон"]
VARS = [(lb, hd, sp, rg) for lb in LOOKS for hd in HOLDS for sp in STOPS for rg in REGS]
N_RND = 20
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(x):
    lb, hd, sp, rg = x
    return "L %2d · H %d · стоп %s · %s" % (lb, hd, ("%g ATRд" % sp) if sp else "нет", rg)


class Coin:
    def __init__(self, b15):
        self.t = np.array([x[0] for x in b15], dtype=np.int64)
        self.o = np.array([x[1] for x in b15])
        self.h = np.array([x[2] for x in b15])
        self.l = np.array([x[3] for x in b15])
        d1 = c.agg(b15, D1)
        self.dt = [x[0] for x in d1]
        self.dc = [x[4] for x in d1]
        self.atr = c.atr_pct(d1)            # дневной ATR, %

    def day(self, day_open):
        j = bisect.bisect_left(self.dt, day_open)
        return j if j < len(self.dt) and self.dt[j] == day_open else None

    def short(self, T, hold, stop_pct):
        """Шорт с T на hold дней. -> (доходность в долях, R или None) или None, если нет данных."""
        k0 = int(np.searchsorted(self.t, T, side="left"))
        kE = int(np.searchsorted(self.t, T + hold * D1, side="left"))
        if k0 >= len(self.t) - 1 or kE >= len(self.t) or kE <= k0:
            return None
        e = self.o[k0]
        px = self.o[kE]
        k_end = kE
        if stop_pct:
            lvl = e * (1 + stop_pct / 100)
            hit = np.nonzero(self.h[k0:kE] >= lvl)[0]
            if len(hit):
                k_end = k0 + int(hit[0])
                px = max(lvl, self.o[k_end]) if k_end > k0 else lvl
        hours = (self.t[k_end] - self.t[k0]) / 3600000
        ret = (e - px) / e - FEE - FUND_8H * hours / 8
        return ret, (ret / (stop_pct / 100) if stop_pct else None)


def main():
    t0 = time.time()
    say("МЕДВЕЖКА: ШОРТ СЛАБЫХ МОНЕТ ОТНОСИТЕЛЬНО BTC · %s · K = %d · Binance 15m"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), K))
    btc = st.Btc()
    s200 = btc.sma
    btc_dt, btc_dc = btc.t, btc.cl

    def regime_ok(i, rg):
        if i < 230 or s200[i] is None or s200[i - 30] is None:
            return False
        if not btc_dc[i] < s200[i]:
            return False
        return True if rg == "ниже" else s200[i] < s200[i - 30]

    ok = {x: True for x in VARS}
    pos_all = {x: True for x in VARS}
    for set_name, names in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        coins = {}
        for sym in names:
            b15 = cs.load_15m(sym)
            if len(b15) < 20000:
                continue
            coins[sym] = Coin(b15)
        say("  монет с данными: %d · %.0f с" % (len(coins), time.time() - t0))
        res = {x: [] for x in VARS}      # (T, доходность, R)
        rnd = {x: [] for x in VARS}
        rg_ = random.Random("weak-%s" % set_name)
        for x in VARS:
            lb, hd, sp, rg = x
            i = 0
            next_ok = 0
            while i < len(btc_dt):
                T = btc_dt[i] + D1                       # после закрытия дня i
                if T < st.START or T < next_ok or not regime_ok(i, rg) or i < lb:
                    i += 1
                    continue
                b_ret = btc_dc[i] / btc_dc[i - lb] - 1
                cand = []
                for sym, cn in coins.items():
                    j = cn.day(btc_dt[i])
                    if j is None or j < lb + 30 or cn.atr[j] <= 0:
                        continue
                    cand.append((cn.dc[j] / cn.dc[j - lb] - 1 - b_ret, sym, cn.atr[j]))
                if len(cand) < K * 2:
                    i += 1
                    continue
                cand.sort()
                for rs, sym, atr in cand[:K]:
                    r = coins[sym].short(T, hd, sp * atr if sp else None)
                    if r:
                        res[x].append((T, r[0], r[1]))
                for _ in range(N_RND):
                    for rs, sym, atr in rg_.sample(cand, K):
                        r = coins[sym].short(T, hd, sp * atr if sp else None)
                        if r:
                            rnd[x].append(r[0])
                next_ok = T + hd * D1
                i += 1
        first_t = min((q[0] for v in res.values() for q in v), default=None)
        if first_t is None:
            say("  сделок нет")
            continue
        end_t = max(q[0] for v in res.values() for q in v)
        mid = (first_t + end_t) // 2
        say("  половины: до / после %s" % st.dstr(mid))
        for x in VARS:
            v = res[x]
            rs = [q[1] for q in v]
            r1 = [q[1] for q in v if q[0] < mid]
            r2 = [q[1] for q in v if q[0] >= mid]
            a, a1, a2 = (np.mean(rs) if rs else 0), (np.mean(r1) if r1 else 0), (np.mean(r2) if r2 else 0)
            ra = np.mean(rnd[x]) if rnd[x] else 0
            rr = [q[2] for q in v if q[2] is not None]
            yrs = {}
            for q in v:
                yrs[cs.year_of(q[0])] = yrs.get(cs.year_of(q[0]), 0) + q[1] * 100
            good = len(rs) >= 30 and a1 > 0 and a2 > 0 and a > ra
            ok[x] = ok[x] and good
            pos_all[x] = pos_all[x] and a > 0
            say("    %-34s сделок %4d · WR %4.1f%% · сделка %+.2f%% (половины %+.2f / %+.2f)%s · случайные %+.2f%% → %s"
                % (vname(x), len(rs), 100 * sum(1 for q in rs if q > 0) / max(1, len(rs)), 100 * a, 100 * a1, 100 * a2,
                   (" · %+.3f R" % np.mean(rr)) if rr else "", 100 * ra, "да" if good else "нет"))
            say("        по годам (сумма %% на сделку): " + " · ".join("%d: %+.0f" % (y, s_) for y, s_ in sorted(yrs.items())))
    say("")
    say("=" * 110)
    passed = [x for x in VARS if ok[x]]
    npos = sum(1 for x in VARS if pos_all[x])
    say("Прошли все три набора: %s" % (", ".join(vname(x) for x in passed) if passed else "ни один"))
    say("В плюсе на всех трёх наборах: %d из %d (нужно не меньше %d)" % (npos, len(VARS), (2 * len(VARS) + 2) // 3))
    accept = bool(passed) and npos * 3 >= 2 * len(VARS)
    say("→ ИТОГ: %s" % ("ШОРТ СЛАБЫХ МОНЕТ В МЕДВЕЖКЕ РАБОТАЕТ" if accept else "шорт слабых монет в медвежке не подтвердился"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

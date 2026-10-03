#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МЕДВЕЖКА: ШОРТ ОТ ВЕРХНЕЙ ГРАНИЦЫ ДНЕВНОГО КАНАЛА MRC С ЦЕЛЬЮ В СЕРЕДИНУ (идея из сделок разработчика, LINK).

Логика: в медвежке (BTC вчера закрылся НИЖЕ дневной SMA200) продаём выносы вверх — когда монета на отскоке
доходит до верхней границы своего дневного канала MRC. Канал как в индикаторе: SuperSmoother 200 по дневкам,
по прошлому закрытому дню; середина M, единица ширины U = π × размах.
Вход: 30m-свеча закрылась ВЫШЕ уровня L (а предыдущая — не выше), шорт по открытию следующей 15m-свечи.
Одна позиция на монету. Binance 15m, комиссия 0.11%, фандинг 0.01% / 8 ч против позиции, в спорной свече — стоп.
Три набора: основной (21), новый (20), третий (40). У монеты не меньше 250 дневок.

ВАРИАНТЫ (12, объявлены до прогона):
  уровень входа L: M + 1·U (внутренняя граница) · M + 1.71·U (середина зоны, «ур. 2») · M + 2.415·U (внешняя)
  стоп: вход + 1·U · вход + 2·U
  цель: середина канала M · половина пути до M
Случайный шорт (контроль «граница важна?»): случайные моменты тех же медвежьих периодов, те же монеты; цель и стоп
в % берутся от случайной НАСТОЯЩЕЙ сделки того же варианта.

КРИТЕРИЙ (объявлен ДО прогона): вариант проходит, если на ВСЕХ ТРЁХ наборах средний R в плюсе в ОБЕИХ половинах
периода И лучше случайного шорта. Правило принимается, только если прошёл хотя бы один вариант И в плюсе (средний R
на всех трёх наборах) не меньше 2/3 вариантов.
Справочно: то же правило вне медвежки (BTC выше SMA200) — чтобы понять, медвежка ли тут важна.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: bear_mrc_short_report.txt
"""
import bisect
import math
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

c = cs.c
REPORT = "bear_mrc_short_report.txt"
M15, M30, D1 = cs.M15, cs.M30, cs.D1
LEVELS = [1.0, 1.71, 2.415]
STOPS = [1.0, 2.0]
TARGETS = ["середина", "половина пути"]
VARS = [(lv, sp, tg) for lv in LEVELS for sp in STOPS for tg in TARGETS]
RND_PER = 3          # случайных шортов на одну настоящую сделку
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(x):
    lv, sp, tg = x
    return "вход M+%.2gU · стоп +%gU · цель %s" % (lv, sp, tg)


def main():
    t0 = time.time()
    say("МЕДВЕЖКА: ШОРТ ОТ ВЕРХНЕЙ ГРАНИЦЫ ДНЕВНОГО MRC · %s · Binance 15m"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    ok = {x: True for x in VARS}
    pos_all = {x: True for x in VARS}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res = {(x, bear): [] for x in VARS for bear in (True, False)}   # (T, R)
        rnd = {x: [] for x in VARS}
        first_t, end_t = None, 0
        for sym in coins:
            b15 = cs.load_15m(sym)
            if len(b15) < 20000:
                say("  %s: мало данных — пропуск" % sym)
                continue
            t = np.array([x[0] for x in b15], dtype=np.int64)
            o = np.array([x[1] for x in b15])
            h = np.array([x[2] for x in b15])
            l = np.array([x[3] for x in b15])
            cl = np.array([x[4] for x in b15])
            end_t = max(end_t, int(t[-1]))
            b30 = c.agg(b15, M30)
            d1 = c.agg(b15, D1)
            dt = [x[0] for x in d1]
            mean = cs.supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
            rngd = cs.supersmoother(c.trs(d1), 200)
            # входы по каждому уровню: (T, M, U, бычий/медвежий режим)
            ents = {lv: [] for lv in LEVELS}
            prev_above = {lv: True for lv in LEVELS}
            for i in range(1, len(b30)):
                T = b30[i][0] + M30
                if T < st.START:
                    continue
                jd = bisect.bisect_right(dt, T - D1) - 1
                if jd < 250:
                    continue
                M, U = mean[jd], math.pi * rngd[jd]
                if U <= 0:
                    continue
                px = b30[i][4]
                for lv in LEVELS:
                    above = px > M + lv * U
                    if above and not prev_above[lv]:
                        ents[lv].append((T, M, U, not btc.above(T)))
                    prev_above[lv] = above
            bear_times = []
            for x in VARS:
                lv, sp, tg = x
                busy = {True: 0, False: 0}
                for T, M, U, bear in ents[lv]:
                    if T < busy[bear]:
                        continue
                    k0 = int(np.searchsorted(t, T, side="left"))
                    if k0 >= len(t) - 1:
                        continue
                    e = o[k0]
                    tgt = M if tg == "середина" else e - 0.5 * (e - M)
                    if tgt >= e:
                        continue
                    tpf, slf = (e - tgt) / e, sp * U / e
                    pnl, r, te = cs.trade(t, h, l, cl, k0, -1, e, tpf, slf)
                    res[(x, bear)].append((T, r, tpf, slf))
                    if bear:
                        first_t = T if first_t is None else min(first_t, T)
                    busy[bear] = te if te is not None else 10 ** 15
            # случайные шорты в медвежьи моменты, цель/стоп в % — от настоящих сделок
            rg = random.Random("bear-%s" % sym)
            y0 = max(st.START, int(t[0]) + 260 * D1)
            y1 = int(t[-1]) - 5 * D1
            for x in VARS:
                own = res[(x, True)]
                if not own or y1 <= y0:
                    continue
                need = RND_PER * sum(1 for q in res[(x, True)])
                got = tries = 0
                while got < need and tries < need * 40:
                    tries += 1
                    T = rg.randrange(y0, y1)
                    if btc.above(T):
                        continue
                    k0 = int(np.searchsorted(t, T, side="left"))
                    if k0 >= len(t) - 1:
                        continue
                    q = own[rg.randrange(len(own))]
                    rnd[x].append(cs.trade(t, h, l, cl, k0, -1, o[k0], q[2], q[3])[1])
                    got += 1
            say("  [%s] %s: касаний (ур.1/2/3) %s · %.0f с" % (
                set_name, sym, "/".join(str(sum(1 for z in ents[lv] if z[3])) for lv in LEVELS), time.time() - t0))
        if first_t is None:
            say("  нет сделок в медвежке")
            continue
        mid = (first_t + end_t) // 2
        say("")
        say("  МЕДВЕЖКА (BTC ниже SMA200) · половины до / после %s" % st.dstr(mid))
        for x in VARS:
            v = res[(x, True)]
            rs = [q[1] for q in v]
            r1 = [q[1] for q in v if q[0] < mid]
            r2 = [q[1] for q in v if q[0] >= mid]
            a, a1, a2 = (np.mean(rs) if rs else 0), (np.mean(r1) if r1 else 0), (np.mean(r2) if r2 else 0)
            ra = np.mean(rnd[x]) if rnd[x] else 0
            good = len(rs) >= 30 and a1 > 0 and a2 > 0 and a > ra
            ok[x] = ok[x] and good
            pos_all[x] = pos_all[x] and a > 0
            say("    %-46s сделок %4d · WR %4.1f%% · средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+6.1f · "
                "случайный %+.3f → %s" % (vname(x), len(rs), 100 * sum(1 for q in rs if q > 0) / max(1, len(rs)),
                                         a, a1, a2, sum(rs), ra, "да" if good else "нет"))
        say("")
        say("  СПРАВОЧНО — то же вне медвежки (BTC выше SMA200):")
        for x in VARS:
            rs = [q[1] for q in res[(x, False)]]
            say("    %-46s сделок %4d · средний R %+.3f · сумма R %+6.1f"
                % (vname(x), len(rs), np.mean(rs) if rs else 0, sum(rs)))
    say("")
    say("=" * 110)
    passed = [x for x in VARS if ok[x]]
    npos = sum(1 for x in VARS if pos_all[x])
    say("Прошли все три набора: %s" % (", ".join(vname(x) for x in passed) if passed else "ни один"))
    say("В плюсе на всех трёх наборах: %d из %d (нужно не меньше %d)" % (npos, len(VARS), (2 * len(VARS) + 2) // 3))
    accept = bool(passed) and npos * 3 >= 2 * len(VARS)
    say("→ ИТОГ: %s" % ("ШОРТ ОТ ГРАНИЦЫ MRC В МЕДВЕЖКЕ РАБОТАЕТ" if accept else "шорт от границы MRC в медвежке не подтвердился"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

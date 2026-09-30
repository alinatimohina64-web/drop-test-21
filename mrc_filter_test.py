#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ФИЛЬТР MRC (как в индикаторе v4.16): не брать лонг, когда цена высоко в дневном канале MRC, и шорт — когда низко.
Проверка на 21 монете, 2022–2026.

Сигналы: индикатор TT2, график 15m и 30m, старший ТФ 4ч, фильтр BTC 1Д «требовать».
Вход по открытию 15m после сигнала, цель и стоп в × ATR графика, спорная свеча — стоп, комиссия 0.11%,
не дольше 30 дней. Пары цели/стопа: 2.5/2.5 и 5/3.5 × ATR.

Канал MRC — по ДНЕВКАМ монеты (SuperSmoother 200 от hlc3 и от истинного диапазона), берётся прошлый
закрытый день (как в индикаторе: request.security с [1]). Цена — закрытие сигнальной свечи.
Две схемы, обе объявлены до прогона:
  А «упрощённая, уровень ≥ 2» (нашёл Владимир на XRP): лонг «горячий», если цена выше середины между
    внутренней (π·1·размах) и внешней (π·2.415·размах) границей; шорт — зеркально;
  Б «как у разработчика, уровень ≥ 1»: лонг «горячий», если цена выше «внешняя граница − 2 размаха»
    (вошла в светлую зону); шорт — зеркально.
Фильтр убирает «горячие» сигналы (упрощение: откат кулдауна после убранного сигнала не пересчитывается).

КРИТЕРИЙ (объявлен ДО прогона), для каждой схемы, в каждой из 4 ячеек (ТФ 15m/30m × пара 2.5/2.5, 5/3.5):
  1) оставшиеся сделки в среднем лучше всех сделок (R на сделку) в обеих половинах 2022–2026
     и в 3 из 4 лет 2023–2026;
  2) разница «оставшиеся» − «горячие» статистически заметна: t >= 2.
Схема РАБОТАЕТ, если критерий выполнен не меньше чем в 3 ячейках из 4.
Результат в R и в $ при позиции 3 000 $ (маржа 300 × 10). Отчёт: mrc_filter_report.txt
СРАВНЕНИЕ СТАРШЕГО ТФ 4ч ПРОТИВ 1Д (переменная MRC_HTF_COMPARE=1, объявлено до прогона): график 30m,
  схема А (уровень 2), обе пары; сравниваются сделки ПОСЛЕ фильтра. «1Д лучше 4ч», если в обеих парах
  средняя сделка 1Д выше в обеих половинах 2022–2026 и в 3 из 4 лет 2023–2026 (для молодых монет —
  в большинстве лет со сделками). Запускать на обоих наборах монет (основном и MRC_COINS) — вывод
  принимается, только если оба прогона согласны.
ПОВТОРНАЯ ПРОВЕРКА на других монетах (режим подтверждения, задаётся переменными окружения):
  MRC_COINS=SUI,APT,...  MRC_TFS=30m  MRC_SCHEMES=simple  — тогда схема подтверждена, если критерий выполнен
  во ВСЕХ ячейках, а условие по годам — «лучше в большинстве лет, где есть сделки» (у молодых монет меньше лет).
Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py (данные — из их кэша).
"""
import bisect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import os

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw

REPORT = "mrc_filter_report.txt"
M15, M30 = 900000, 1800000
FEE = 0.11 / 100
MAX_BARS = 30 * 96
NOTIONAL = 3000.0                 # маржа 300 $ × плечо 10, как в индикаторе
PAIRS = [(2.5, 2.5), (5.0, 3.5)]
cw.EXITS = PAIRS                      # быстрый расчёт выходов (numpy) из coin_wf_test — те же правила
TEST_FROM = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TEST_YEARS = [2023, 2024, 2025, 2026]
M15 = 900000
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


SCHEMES = [("А упрощённая, ур. ≥ 2", "simple"), ("Б как у разработчика, ур. ≥ 1", "dev")]
TFS = [("15m", M15), ("30m", M30)]
CONFIRM = bool(os.environ.get("MRC_COINS"))
if os.environ.get("MRC_TFS"):
    TFS = [x for x in TFS if x[0] in os.environ["MRC_TFS"].split(",")]
if os.environ.get("MRC_SCHEMES"):
    SCHEMES = [x for x in SCHEMES if x[1] in os.environ["MRC_SCHEMES"].split(",")]


def hot_flag(scheme, d, px, mean, rng):
    outer = math.pi * 2.415 * rng
    if scheme == "simple":
        thr = (math.pi * 1.0 * rng + outer) / 2          # середина между внутренней и внешней границей
    else:
        thr = outer - 2.0 * rng                          # вход в светлую зону у внешней границы
    return px >= mean + thr if d == 1 else px <= mean - thr


def compare_htf():
    """Сравнение старшего ТФ индикатора 4ч и 1Д на отфильтрованных сделках (схема А, 30m)."""
    t0 = time.time()
    coins = [x.strip().upper() for x in os.environ["MRC_COINS"].split(",")] if CONFIRM else list(mt.MONEY)
    say("СТАРШИЙ ТФ 4ч ПРОТИВ 1Д · %s · монет %d · 30m · MRC схема А (ур. 2) · BTC 1Д · позиция %.0f $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), NOTIONAL))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    res = {(hk, p): [] for hk in ("4ч", "1Д") for p in PAIRS}
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < (20000 if CONFIRM else 50000) or len(d1) < 400:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        cd = [x[4] for x in d1]
        dt = [x[0] for x in d1]
        htfs = {"4ч": ([x[0] for x in h4a], [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))], c.H4),
                "1Д": (dt, [a > b_ for a, b_ in zip(c.ema(cd, 9), c.ema(cd, 21))], c.D1)}
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        b = c.agg(b15, M30)
        atrp = c.atr_pct(b)
        for hk, (ht, hu, hms) in htfs.items():
            for T, d, i in cw.ind_entries_tf(b, M30, ht, hu, hms):
                if T < mt.START or atrp[i] <= 0 or btc_at(T) != d:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                jd = bisect.bisect_right(dt, T - c.D1) - 1
                if jd < 250 or hot_flag("simple", d, b[i][4], mean[jd], rng[jd]):
                    continue
                a = atrp[i] / 100
                rs = cw.exits_R(o, h, l, cl, k0, d, atrp[i])
                for (tp, sl), r in zip(PAIRS, rs):
                    res[(hk, (tp, sl))].append((T, r, r * sl * a * NOTIONAL))
        say("  %s: готово · %.0f с" % (sym, time.time() - t0))
    now = int(time.time() * 1000)
    mid = TEST_FROM + (now - TEST_FROM) // 2
    ok_all = True
    for tp, sl in PAIRS:
        say("")
        say("  цель %g / стоп %g × ATR (сделки после фильтра MRC, проверка 2022–2026)" % (tp, sl))
        for hk in ("4ч", "1Д"):
            v = [x for x in res[(hk, (tp, sl))] if x[0] >= TEST_FROM]
            mu, t, n = mean_t([x[1] for x in v])
            wr = 100 * sum(1 for x in v if x[1] > 0) / max(1, n)
            say("    старший %s: сделок %5d · WR %4.1f%% · %+.3f R/сд · %+.2f $/сд · итог %+8.0f $"
                % (hk, n, wr, mu, sum(x[2] for x in v) / max(1, n), sum(x[2] for x in v)))
        a4 = [x for x in res[("4ч", (tp, sl))] if x[0] >= TEST_FROM]
        a1 = [x for x in res[("1Д", (tp, sl))] if x[0] >= TEST_FROM]
        yl, better, n_y = [], 0, 0
        for y in TEST_YEARS:
            v4 = [x[1] for x in a4 if year_of(x[0]) == y]
            v1 = [x[1] for x in a1 if year_of(x[0]) == y]
            if len(v4) >= 50 and len(v1) >= 50:
                n_y += 1
            m4, _, _ = mean_t(v4)
            m1, _, _ = mean_t(v1)
            better += m1 > m4
            yl.append("%d: 4ч %+.3f / 1Д %+.3f" % (y, m4, m1))
        halves = []
        for cond in (lambda T: T < mid, lambda T: T >= mid):
            m4, _, _ = mean_t([x[1] for x in a4 if cond(x[0])])
            m1, _, _ = mean_t([x[1] for x in a1 if cond(x[0])])
            halves.append(m1 > m4)
        need = (n_y // 2 + 1) if CONFIRM else 3
        ok = all(halves) and better >= need
        ok_all &= ok
        say("    по годам (R/сд): " + " · ".join(yl))
        say("    1Д лучше: обе половины [%s] · лет %d (нужно %d) → %s" % ("да" if all(halves) else "нет", better, need,
                                                                         "да" if ok else "нет"))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): %s" % ("старший 1Д ЛУЧШЕ 4ч на этом наборе монет" if ok_all
                                             else "старший 1Д не лучше 4ч на этом наборе монет"))
    say("  (вывод принимается, только если прогоны на обоих наборах монет согласны)")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


def main():
    t0 = time.time()
    coins = [x.strip().upper() for x in os.environ["MRC_COINS"].split(",")] if CONFIRM else list(mt.MONEY)
    if CONFIRM:
        say("РЕЖИМ ПОДТВЕРЖДЕНИЯ на других монетах: все ячейки должны пройти, годы — большинство лет со сделками")
    say("ФИЛЬТР MRC (дневной канал) · %s · монет %d · 15m и 30m / старший 4ч / BTC 1Д · позиция %.0f $ (300 × 10)"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), NOTIONAL))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    res = {(tf, p, sc): [] for tf, _ in TFS for p in PAIRS for _, sc in SCHEMES}   # (T, горячий?, R, $)
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < (20000 if CONFIRM else 50000) or len(d1) < 400:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        dt = [x[0] for x in d1]
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        cnt = {sc: [0, 0] for _, sc in SCHEMES}
        for tfn, tfms in TFS:
            b = b15 if tfms == M15 else c.agg(b15, tfms)
            atrp = c.atr_pct(b)
            for T, d, i in cw.ind_entries_tf(b, tfms, [x[0] for x in h4a], hu, c.H4):
                if T < mt.START or atrp[i] <= 0 or btc_at(T) != d:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                jd = bisect.bisect_right(dt, T - c.D1) - 1        # прошлый закрытый день
                if jd < 250:
                    continue
                px = b[i][4]
                a = atrp[i] / 100
                rs = cw.exits_R(o, h, l, cl, k0, d, atrp[i])
                for _, sc in SCHEMES:
                    hot = hot_flag(sc, d, px, mean[jd], rng[jd])
                    if tfn == TFS[-1][0]:
                        cnt[sc][0] += 1
                        cnt[sc][1] += hot
                    for (tp, sl), r in zip(PAIRS, rs):
                        res[(tfn, (tp, sl), sc)].append((T, hot, r, r * sl * a * NOTIONAL))
        say("  %s: доля «горячих» сигналов (%s): %s · %.0f с"
            % (sym, TFS[-1][0], " · ".join("%s %.0f%%" % (scn[0], 100 * cnt[sc][1] / max(1, cnt[sc][0])) for scn, sc in SCHEMES),
               time.time() - t0))

    now = int(time.time() * 1000)
    mid = TEST_FROM + (now - TEST_FROM) // 2
    verdict = {}
    for scn, sc in SCHEMES:
        say("")
        say("#" * 110)
        say("СХЕМА %s" % scn)
        passed = 0
        for tfn, _ in TFS:
            for tp, sl in PAIRS:
                v = [x for x in res[(tfn, (tp, sl), sc)] if x[0] >= TEST_FROM]
                say("")
                say("  %s · цель %g / стоп %g × ATR" % (tfn, tp, sl))

                def line(nm, sub):
                    arr = [x[2] for x in sub]
                    mu, t, n = mean_t(arr)
                    usd = [x[3] for x in sub]
                    wr = 100 * sum(1 for x in arr if x > 0) / max(1, n)
                    return "    %-22s сделок %5d · WR %4.1f%% · %+.3f R/сд · %+.2f $/сд · итог %+8.0f $" % (
                        nm, n, wr, mu, sum(usd) / max(1, n), sum(usd))
                for nm, sub in (("все сделки", v), ("после фильтра", [x for x in v if not x[1]]),
                                ("убранные («горячие»)", [x for x in v if x[1]])):
                    say(line(nm, sub))
                better_y = 0
                n_y = 0
                yl = []
                for y in TEST_YEARS:
                    if sum(1 for x in v if year_of(x[0]) == y) >= 50:
                        n_y += 1
                    m_all, _, _ = mean_t([x[2] for x in v if year_of(x[0]) == y])
                    m_k, _, _ = mean_t([x[2] for x in v if year_of(x[0]) == y and not x[1]])
                    better_y += m_k > m_all
                    yl.append("%d: %+.3f/%+.3f" % (y, m_all, m_k))
                halves = []
                for cond in (lambda T: T < mid, lambda T: T >= mid):
                    m_all, _, _ = mean_t([x[2] for x in v if cond(x[0])])
                    m_k, _, _ = mean_t([x[2] for x in v if cond(x[0]) and not x[1]])
                    halves.append(m_k > m_all)
                a_k = [x[2] for x in v if not x[1]]
                a_h = [x[2] for x in v if x[1]]
                m1, _, n1 = mean_t(a_k)
                m2, _, n2 = mean_t(a_h)
                v1 = sum((x - m1) ** 2 for x in a_k) / max(1, n1 - 1)
                v2 = sum((x - m2) ** 2 for x in a_h) / max(1, n2 - 1)
                t_diff = (m1 - m2) / math.sqrt(v1 / max(1, n1) + v2 / max(1, n2)) if n1 > 2 and n2 > 2 else 0.0
                need_y = (n_y // 2 + 1) if CONFIRM else 3
                ok = all(halves) and better_y >= need_y and t_diff >= 2
                passed += ok
                say("    по годам (все / после фильтра, R): " + " · ".join(yl))
                say("    критерий: обе половины [%s] · лет лучше: %d (нужно %d) [%s] · t >= 2 [%s, %+.1f] → %s"
                    % ("да" if all(halves) else "нет", better_y, need_y, "да" if better_y >= need_y else "нет",
                       "да" if t_diff >= 2 else "нет", t_diff, "да" if ok else "нет"))
        n_cells = len(TFS) * len(PAIRS)
        need = n_cells if CONFIRM else 3
        verdict[scn] = passed >= need
        say("")
        say("  СХЕМА %s: критерий выполнен в %d из %d ячеек (нужно %d) → %s"
            % (scn, passed, n_cells, need, "РАБОТАЕТ" if passed >= need else "не работает"))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): " + " · ".join("%s — %s" % (k, "РАБОТАЕТ" if v else "не работает")
                                                     for k, v in verdict.items()))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    if os.environ.get("MRC_HTF_COMPARE") == "1":
        compare_htf()
    else:
        main()

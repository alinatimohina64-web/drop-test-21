#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА АВТОПОДБОРА ЦЕЛИ И СТОПА (как в индикаторе Trand-Test-2 v4.10) — walk-forward, 2022–2026.

Как торгует человек по индикатору:
  всё зафиксировано — график 30m, старший ТФ 4ч, фильтр BTC 1Д «требовать» (сигнал только
  по дневному тренду BTC); раз в неделю для каждой монеты смотрит строку «За 60 дн.»:
  автоподбор цели и стопа по последним 60 дням —
    цель 2.0…6.0 × ATR, стоп 1.5…4.0 × ATR, шаг 0.5 (54 пары);
    пара допускается, если цель >= стопа, винрейт >= 55%, сделок >= 20 и итог пары в плюсе;
    из допущенных выбирается пара с лучшим СРЕДНИМ результатом вместе с соседними парами
    (±1 шаг по цели и по стопу) — как в индикаторе, чтобы не брать случайный пик.
  Нашлась пара — следующие 7 дней монета торгуется этой парой. Не нашлась — монета пропускается.
  В подборе участвуют только сделки, ЗАКРЫВШИЕСЯ до начала недели (без заглядывания вперёд).

Сделка: сигнал индикатора (основной + откат, как в прошлых тестах, сверено с таблицей TradingView),
вход по открытию 15-минутки после закрытия сигнальной свечи, цель/стоп от входа в × ATR(14) графика,
внутри 15m-свечи, где задеты и цель, и стоп, — стоп. Комиссия 0.11% за круг. Не больше 30 дней.
Результат в R (R = риск на стопе) и в $ при фиксированном риске 20 $ на сделку.

СРАВНЕНИЕ:
  А — автоподбор (как выше);
  Б — одна постоянная пара 2.5 / 2.5 × ATR на всех монетах, без подбора и без пропусков;
  В — случайные входы: для каждой сделки А — вход в случайный момент той же недели той же монеты,
      в сторону дневного тренда BTC, с той же подобранной парой (среднее по 5 прогонам).
      Показывает, добавляют ли что-то сами сигналы индикатора.

КРИТЕРИЙ (объявлен ДО прогона). Автоподбор работает, если одновременно:
  1) итог А в плюсе в 3 из 4 лет 2023, 2024, 2025, 2026 (2026 — неполный);
  2) А лучше Б и лучше В по сумме за весь период;
  3) А лучше Б и лучше В в обеих половинах периода.

Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py (данные 15m — из кэша).
Нужен numpy. Отчёт: autofit_report.txt
"""
import bisect
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw

REPORT = "autofit_report.txt"
M15, M30 = 900000, 1800000
TPS = np.round(np.arange(2.0, 6.0001, 0.5), 2)
SLS = np.round(np.arange(1.5, 4.0001, 0.5), 2)
NT, NS = len(TPS), len(SLS)
NP = NT * NS
PAIR_TP = np.repeat(TPS, NS)
PAIR_SL = np.tile(SLS, NT)
FIX_K = int(np.where((PAIR_TP == 2.5) & (PAIR_SL == 2.5))[0][0])
FEE = 0.11 / 100
MAX_BARS = 30 * 96
RISK = 20.0
LOOK = 60 * c.D1
WEEK = 7 * c.D1
MIN_N = 20
MIN_WR = 55.0
RATIO = 1.0
N_RAND = 5
TEST_YEARS = [2023, 2024, 2025, 2026]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def exits_all(t15, o, h, l, cl, k0, d, atrp):
    """Для всех пар: R (с комиссией), выигрыш (цель раньше стопа), время выхода."""
    k1 = min(len(o), k0 + MAX_BARS)
    p0 = o[k0]
    if d == 1:
        fav = np.maximum.accumulate((h[k0:k1] - p0) / p0)
        adv = np.maximum.accumulate((p0 - l[k0:k1]) / p0)
    else:
        fav = np.maximum.accumulate((p0 - l[k0:k1]) / p0)
        adv = np.maximum.accumulate((h[k0:k1] - p0) / p0)
    n = len(fav)
    last = d * (cl[k1 - 1] - p0) / p0
    a = atrp / 100
    it_arr = np.searchsorted(fav, TPS * a, side="left")
    is_arr = np.searchsorted(adv, SLS * a, side="left")
    R = np.empty(NP)
    W = np.zeros(NP, dtype=bool)
    X = np.empty(NP, dtype=np.int64)
    for ti in range(NT):
        it = it_arr[ti]
        for si in range(NS):
            k = ti * NS + si
            is_ = is_arr[si]
            slx = SLS[si] * a
            if is_ < n and is_ <= it:
                r, w, x = -1.0, False, is_
            elif it < n:
                r, w, x = TPS[ti] / SLS[si], True, it
            else:
                r = last / slx
                w, x = r > 0, n - 1
            R[k] = r - FEE / slx
            W[k] = w
            X[k] = t15[k0 + x] + M15
    return R, W, X


def select(Tw, T, R, WIN, X):
    """Автоподбор как в индикаторе v4.10 на 60 днях до Tw. Возвращает индекс пары или None."""
    rows = (T >= Tw - LOOK) & (T < Tw)
    if not rows.any():
        return None
    m = X[rows] < Tw                                  # только сделки, закрытые до начала недели
    n = m.sum(0)
    w = (WIN[rows] & m).sum(0)
    s = np.where(m, R[rows], 0.0).sum(0)
    valid = n >= MIN_N
    wr = w / np.maximum(n, 1) * 100
    cand = valid & (PAIR_TP >= RATIO * PAIR_SL - 1e-9) & (wr >= MIN_WR) & (s > 0)
    if not cand.any():
        return None
    sc = np.where(valid, s, np.nan).reshape(NT, NS)
    best, best_v = None, None
    for k in np.nonzero(cand)[0]:
        ti, si = divmod(int(k), NS)
        blk = sc[max(0, ti - 1):ti + 2, max(0, si - 1):si + 2]
        v = np.nanmean(blk)
        if best is None or v > best_v:
            best, best_v = int(k), v
    return best


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("АВТОПОДБОР ЦЕЛИ И СТОПА (как в индикаторе v4.10), WALK-FORWARD · %s · монет %d · пар %d"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), NP))
    say("  30m · старший 4ч · BTC 1Д требовать · подбор по 60 дням раз в неделю · цель >= стопа · "
        "WR >= %.0f%% · сделок >= %d · сглаживание по соседям · риск %.0f $" % (MIN_WR, MIN_N, RISK))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_trend = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_trend[i] if i >= 0 else 0

    now = int(time.time() * 1000)
    weeks = list(range(mt.START + LOOK, now - WEEK, WEEK))
    mid_w = weeks[len(weeks) // 2]
    rnd = random.Random(11)

    # итоги: ключ (вариант, год) и (вариант, половина) -> [сумма R, сделок, выигрышей]
    st = defaultdict(lambda: [0.0, 0, 0])
    per_coin = defaultdict(lambda: [0.0, 0, 0.0, 0])      # А: R, n · Б: R, n
    pair_use = defaultdict(int)
    cover = [0, 0]                                        # недель с парой / всего монето-недель
    now_pick = {}

    def add(v, T, r, w):
        for key in ((v, year_of(T)), (v, "h1" if T < mid_w else "h2"), (v, "all")):
            a = st[key]
            a[0] += r
            a[1] += 1
            a[2] += 1 if w else 0

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
        t30 = [x[0] for x in b30]
        atr30 = c.atr_pct(b30)
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        ht = [x[0] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        Ts, Rs, Ws, Xs = [], [], [], []
        for T, d, i in cw.ind_entries_tf(b30, M30, ht, hu, c.H4):
            if T < mt.START or atr30[i] <= 0 or btc_at(T) != d:
                continue
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                continue
            R, W, X = exits_all(t15, o, h, l, cl, k0, d, atr30[i])
            Ts.append(T)
            Rs.append(R)
            Ws.append(W)
            Xs.append(X)
        if not Ts:
            say("  %s: нет сигналов" % sym)
            continue
        T = np.array(Ts, dtype=np.int64)
        R = np.vstack(Rs)
        WIN = np.vstack(Ws)
        X = np.vstack(Xs)
        n_a = 0
        for Wk in weeks:
            cover[1] += 1
            rows_w = np.nonzero((T >= Wk) & (T < Wk + WEEK))[0]
            for r_ in rows_w:                              # Б — постоянная пара на всех сигналах
                add("B", int(T[r_]), R[r_, FIX_K], WIN[r_, FIX_K])
                per_coin[sym][2] += R[r_, FIX_K]
                per_coin[sym][3] += 1
            k = select(Wk, T, R, WIN, X)
            if k is None:
                continue
            cover[0] += 1
            if len(rows_w) == 0:
                continue
            pair_use[k] += len(rows_w)
            for r_ in rows_w:                              # А — автоподбор
                add("A", int(T[r_]), R[r_, k], WIN[r_, k])
                per_coin[sym][0] += R[r_, k]
                per_coin[sym][1] += 1
                n_a += 1
            # В — случайные входы той же недели, та же пара, по тренду BTC 1Д
            kw0 = int(np.searchsorted(t15, Wk, side="left"))
            kw1 = int(np.searchsorted(t15, Wk + WEEK, side="left"))
            if kw1 - kw0 < 10:
                continue
            for _ in range(len(rows_w)):
                acc_r = acc_w = 0.0
                got = 0
                tries = 0
                while got < N_RAND and tries < N_RAND * 20:
                    tries += 1
                    kk = rnd.randrange(kw0, kw1)
                    Tr = int(t15[kk])
                    dr = btc_at(Tr)
                    j = bisect.bisect_right(t30, Tr - M30) - 1
                    if dr == 0 or j < 20 or atr30[j] <= 0 or kk >= len(t15) - 1:
                        continue
                    Rr, Wr, _ = exits_all(t15, o, h, l, cl, kk, dr, atr30[j])
                    acc_r += Rr[k]
                    acc_w += 1 if Wr[k] else 0
                    got += 1
                if got:
                    add("C", Wk, acc_r / got, acc_w / got >= 0.5)
        now_pick[sym] = select(now, T, R, WIN, X)
        say("  %s: сигналов (BTC 1Д за) %d · сделок А %d · %.0f с" % (sym, len(T), n_a, time.time() - t0))

    names = {"A": "А автоподбор", "B": "Б пара 2.5/2.5", "C": "В случайный вход"}

    def line(v, key):
        a = st.get((v, key))
        if not a or a[1] == 0:
            return "%-18s нет сделок" % names[v]
        return "%-18s сделок %5d · WR %4.1f%% · %+.3f R/сд · итог %+7.0f $" % (
            names[v], a[1], 100 * a[2] / a[1], a[0] / a[1], a[0] * RISK)

    say("")
    say("=" * 110)
    say("ПОКРЫТИЕ: монето-недель с найденной парой %d из %d (%.0f%%)"
        % (cover[0], cover[1], 100 * cover[0] / max(1, cover[1])))
    say("")
    say("ПО ГОДАМ (риск %.0f $ на сделку; для В — WR средний по случайным прогонам):" % RISK)
    for y in [2022] + TEST_YEARS:
        say("  %d" % y)
        for v in ("A", "B", "C"):
            say("    " + line(v, y))
    say("")
    say("ПО ПОЛОВИНАМ ПЕРИОДА:")
    for hk, hn in (("h1", "1-я половина"), ("h2", "2-я половина")):
        say("  " + hn)
        for v in ("A", "B", "C"):
            say("    " + line(v, hk))
    say("")
    say("ВЕСЬ ПЕРИОД:")
    for v in ("A", "B", "C"):
        say("    " + line(v, "all"))

    say("")
    say("ПО МОНЕТАМ (А / Б, $):")
    say("  " + ", ".join("%s %+.0f (%d сд) / %+.0f" % (s_, v[0] * RISK, v[1], v[2] * RISK)
                         for s_, v in sorted(per_coin.items(), key=lambda kv: -kv[1][0])))
    say("")
    say("КАКИЕ ПАРЫ ВЫБИРАЛИСЬ (цель/стоп × ATR: сделок):")
    say("  " + ", ".join("%g/%g: %d" % (PAIR_TP[k], PAIR_SL[k], n) for k, n in
                         sorted(pair_use.items(), key=lambda kv: -kv[1])[:15]))

    def tot(v, key):
        a = st.get((v, key))
        return a[0] if a else 0.0

    years_ok = sum(1 for y in TEST_YEARS if tot("A", y) > 0)
    c1 = years_ok >= 3
    c2 = tot("A", "all") > tot("B", "all") and tot("A", "all") > tot("C", "all")
    c3 = all(tot("A", hk) > tot("B", hk) and tot("A", hk) > tot("C", hk) for hk in ("h1", "h2"))
    say("")
    say("ВЕРДИКТ (объявлен до прогона):")
    say("  1) А в плюсе в 3 из 4 лет 2023–2026: %d из 4 → %s" % (years_ok, "да" if c1 else "нет"))
    say("  2) А лучше Б и В за весь период → %s" % ("да" if c2 else "нет"))
    say("  3) А лучше Б и В в обеих половинах → %s" % ("да" if c3 else "нет"))
    say("  → %s" % ("АВТОПОДБОР РАБОТАЕТ" if (c1 and c2 and c3) else "автоподбор не работает"))
    say("")
    say("Пары на сегодня (строка «За 60 дн.»; торговать — только если автоподбор работает):")
    for s_, k in now_pick.items():
        say("  %-5s %s" % (s_, "нет пары — пропустить" if k is None else "цель %g / стоп %g × ATR" % (PAIR_TP[k], PAIR_SL[k])))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

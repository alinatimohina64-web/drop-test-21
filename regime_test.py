#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОДСТРОЙКА ПОД ТЕКУЩИЙ РЫНОК: (1) выбор настроек по последним 12 месяцам с пересчётом раз в квартал;
(2) свои настройки для каждой фазы рынка BTC. Всё без подглядывания: выбор делается только по сделкам,
ЗАКРЫТЫМ до момента выбора, фаза определяется по данным BTC, известным в момент входа.

Сигналы — TT2: график 30m, старший ТФ 4ч, только по дневному тренду BTC (как в coin_select_test.py).
Варианты настроек (32 = 4 × 8), объявлены до прогона:
  фильтр MRC: без MRC / упрощённая ур. 2 / упрощённая ур. 3 / разработчика ур. 3;
  выход: стоп 3.5 или 20 × ATR, цель 3 / 4 / 5 / 6 × ATR.
«Неизменный набор» (база) = индикатор по умолчанию: MRC упрощённая ур. 2, цель 5 / стоп 3.5, все монеты.
Вход по открытию 15m после сигнала, одна позиция на монету в каждой настройке, внутри 15m-свечи
«и цель, и стоп» — стоп. Позиция 1000 $ (100 × 10), комиссия 0.11% за круг, фандинг 0.01% за 8 ч.
Данные — Binance спот 15m (кэш data_btc/, общий с coin_select_test.py). Сигналы с 2019 года.
Упрощение: сделки квартала — те, что ОТКРЫТЫ в квартале по выбранной настройке (переходящая с прошлого
квартала сделка другой настройки отдельно не учитывается).

ЧАСТЬ 1. СКОЛЬЗЯЩЕЕ ОКНО 12 МЕСЯЦЕВ, кварталы с 2020-Q1 по 2026-Q3.
  А «под монету»: для каждой монеты — настройка с лучшим итогом в $ за 12 месяцев (не меньше 10 закрытых
     сделок); монета торгуется в следующем квартале, только если этот итог в плюсе.
  Б «под рынок»: одна настройка для всех монет — лучшая по итогу всех монет за 12 месяцев (не меньше
     50 сделок); если её итог не в плюсе — квартал пропускается (сидим вне рынка).
  КРИТЕРИЙ (объявлен ДО прогона), для А и Б отдельно:
    1) итог за все кварталы в плюсе;
    2) средняя сделка лучше, чем у неизменного набора за те же кварталы;
    3) средняя сделка лучше случайного входа с теми же выходами в тех же монетах и кварталах;
    4) в плюсе обе половины (2020–2022 и 2023–2026) и не меньше половины торговых кварталов.

ЧАСТЬ 2. СВОИ НАСТРОЙКИ ДЛЯ ФАЗЫ РЫНКА. Подбор — до халвинга 20.04.2024 (2019 … 2024-04),
  проверка — после (2024-04 … 2026-09). Две разметки, обе известны в момент входа:
    «SMA200»: дневное закрытие BTC вчера выше / ниже своей SMA200;
    «халвинг» (как в индикаторе BTC_ЦИКЛ): дней от халвинга < 518 «рост», 518–889 «вершина и спад»,
      ≥ 890 «дно и поздняя фаза».
  Для каждой фазы — одна настройка для всех монет, лучшая по итогу на подборе (не меньше 100 сделок,
  иначе берётся лучшая без деления на фазы).
  КРИТЕРИЙ (объявлен ДО прогона), для каждой разметки: на проверке итог «по фазам»
    1) в плюсе; 2) больше, чем у неизменного набора; 3) больше, чем у одной лучшей настройки подбора
    без деления на фазы.
Лежит рядом с coin_select_test.py, compare_test.py, coin_wf_test.py. Отчёт: regime_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs

c, cw = cs.c, cs.cw
REPORT = "regime_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
MRCS = ["none", "simple", "simple3", "dev3"]
MRC_NAME = {"none": "без MRC", "simple": "MRC упр.2", "simple3": "MRC упр.3", "dev3": "MRC разр.3"}
EXITS = [(tp, sl) for sl in (3.5, 20.0) for tp in (3.0, 4.0, 5.0, 6.0)]
COMBOS = [(m, tp, sl) for m in MRCS for tp, sl in EXITS]
DEFAULT = ("simple", 5.0, 3.5)
RND_N = 2000


def ms(y, m, d=1):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


START = ms(2019, 1, 1)
SPLIT = ms(2024, 4, 20)
Q_FIRST = (2020, 1)
HALVINGS = [ms(2016, 7, 9), ms(2020, 5, 11), ms(2024, 4, 20)]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def cname(cb):
    m, tp, sl = cb
    return "%s %g/%g" % (MRC_NAME[m], tp, sl)


def halving_phase(T):
    last = max(h for h in HALVINGS if h <= T)
    days = (T - last) / D1
    return "рост" if days < 518 else "вершина и спад" if days < 890 else "дно и поздняя"


def main():
    t0 = time.time()
    say("ПОДСТРОЙКА ПОД РЫНОК · окно 12 мес + фазы BTC · %s · Binance 15m · позиция 1000 $"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = cs.load_15m("BTC")
    bd1 = c.agg(btc, D1)
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)
    btc_c = [x[4] for x in bd1]
    btc_sma = c.sma(btc_c, 200)

    def day_idx(T):
        return bisect.bisect_right(btc_t, T - D1) - 1

    def btc_at(T):
        i = day_idx(T)
        return btc_tr[i] if i >= 0 else 0

    def sma_phase(T):
        i = day_idx(T)
        if i < 0 or btc_sma[i] is None:
            return None
        return "выше SMA200" if btc_c[i] > btc_sma[i] else "ниже SMA200"

    res = {}        # (m, sym, tp, sl) -> [(T, $, R, t_выхода|None)], по T
    rpool = {}      # (sym, tp, sl) -> случайные входы [(T, $, R, t_выхода|None)], по T
    have = []
    data_end = 0
    for sym in cs.COINS:
        b15 = cs.load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        have.append(sym)
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        data_end = max(data_end, int(t15[-1]))
        b30 = c.agg(b15, M30)
        b30t = [x[0] for x in b30]
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
            if T < START or atrp[i] <= 0 or btc_at(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - D1) - 1
            if jd < 250:
                continue
            raw.append((T, d, atrp[i] / 100, b30[i][4], mean[jd], rng[jd]))
        for m in MRCS:
            sigs = [(T, d, a) for T, d, a, px, mn, rg in raw if not cs.hot(m, d, px, mn, rg)]
            for tp, sl in EXITS:
                busy, out = 0, []
                for T, d, a in sigs:
                    if T < busy:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * a, sl * a)
                    out.append((T, pnl, r, te))
                    busy = te if te is not None else 10 ** 15
                res[(m, sym, tp, sl)] = out
        y0 = max(START, int(t15[0]) + 260 * D1)
        y1 = int(t15[-1]) - 5 * D1
        for tp, sl in EXITS:
            rnd = random.Random("%s-%g-%g" % (sym, tp, sl))
            rr = []
            tries = 0
            while len(rr) < RND_N and tries < RND_N * 20 and y1 > y0:
                tries += 1
                T = rnd.randrange(y0, y1)
                d = btc_at(T)
                j = bisect.bisect_right(b30t, T - M30) - 1
                if d == 0 or j < 20 or atrp[j] <= 0:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * atrp[j] / 100, sl * atrp[j] / 100)
                rr.append((T, pnl, r, te))
            rr.sort()
            rpool[(sym, tp, sl)] = rr
        say("  %s: сигналов %d · %.0f с" % (sym, len(raw), time.time() - t0))

    tkeys = {k: [x[0] for x in v] for k, v in res.items()}
    rkeys = {k: [x[0] for x in v] for k, v in rpool.items()}

    def sl_(lst, keys, a, b):
        return lst[bisect.bisect_left(keys, a):bisect.bisect_left(keys, b)]

    def trades(cb, sym, a, b):
        m, tp, sl = cb
        k = (m, sym, tp, sl)
        return sl_(res.get(k, []), tkeys.get(k, []), a, b)

    def closed(cb, sym, a, b):
        return [x for x in trades(cb, sym, a, b) if x[3] is not None and x[3] < b]

    def rnd_trades(sym, tp, sl, a, b, n, seed):
        k = (sym, tp, sl)
        pool = sl_(rpool.get(k, []), rkeys.get(k, []), a, b)
        if not pool or n <= 0:
            return []
        return random.Random(seed).sample(pool, min(n, len(pool)))

    # ======================= ЧАСТЬ 1: скользящее окно =======================
    quarters = []
    y, m = Q_FIRST
    while ms(y, m) < data_end:
        ny, nm = (y, m + 3) if m < 10 else (y + 1, 1)
        quarters.append((y, (m - 1) // 3 + 1, ms(y, m), min(ms(ny, nm), data_end)))
        y, m = ny, nm
    say("")
    say("#" * 110)
    say("ЧАСТЬ 1. ВЫБОР НАСТРОЕК ПО ПОСЛЕДНИМ 12 МЕСЯЦАМ, ПЕРЕСЧЁТ РАЗ В КВАРТАЛ (%d кварталов)" % len(quarters))
    rows = {"A": [], "B": [], "base": []}
    for y, qn, qs, qe in quarters:
        w0 = qs - 365 * D1
        # А — под монету
        a_tr, a_rnd, a_pick = [], [], []
        for sym in have:
            best = None
            for cb in COMBOS:
                v = closed(cb, sym, w0, qs)
                if len(v) >= 10:
                    s = sum(x[1] for x in v)
                    if best is None or s > best[0]:
                        best = (s, cb)
            if best and best[0] > 0:
                cb = best[1]
                tq = trades(cb, sym, qs, qe)
                a_tr += tq
                a_rnd += rnd_trades(sym, cb[1], cb[2], qs, qe, len(tq), "A%s%d" % (sym, qs))
                a_pick.append("%s:%s" % (sym, cname(cb)))
        rows["A"].append((y, qn, a_tr, a_rnd, len(a_pick)))
        # Б — под рынок
        best = None
        for cb in COMBOS:
            v = []
            for sym in have:
                v += closed(cb, sym, w0, qs)
            if len(v) >= 50:
                s = sum(x[1] for x in v)
                if best is None or s > best[0]:
                    best = (s, cb)
        b_tr, b_rnd, b_name = [], [], "вне рынка"
        if best and best[0] > 0:
            cb = best[1]
            b_name = cname(cb)
            for sym in have:
                tq = trades(cb, sym, qs, qe)
                b_tr += tq
                b_rnd += rnd_trades(sym, cb[1], cb[2], qs, qe, len(tq), "B%s%d" % (sym, qs))
        rows["B"].append((y, qn, b_tr, b_rnd, b_name, best[0] if best else 0))
        base = []
        for sym in have:
            base += trades(DEFAULT, sym, qs, qe)
        rows["base"].append((y, qn, base))
        say("  %d-Q%d: А монет %2d · %+6.0f $ (%3d сд) | Б %-20s (за год %+6.0f $) → %+6.0f $ (%4d сд) | база %+6.0f $ (%4d сд)"
            % (y, qn, len(a_pick), sum(x[1] for x in a_tr), len(a_tr), b_name, best[0] if best else 0,
               sum(x[1] for x in b_tr), len(b_tr), sum(x[1] for x in base), len(base)))

    base_all = [x for r in rows["base"] for x in r[2]]
    base_avg = sum(x[1] for x in base_all) / max(1, len(base_all))
    say("")
    say("  неизменный набор (%s, все монеты): сделок %d · итог %+.0f $ · %+.2f $/сд"
        % (cname(DEFAULT), len(base_all), sum(x[1] for x in base_all), base_avg))
    for key, label in (("A", "А — под монету"), ("B", "Б — одна настройка под рынок")):
        tr = [x for r in rows[key] for x in r[2]]
        rn = [x for r in rows[key] for x in r[3]]
        tot = sum(x[1] for x in tr)
        avg = tot / max(1, len(tr))
        ravg = sum(x[1] for x in rn) / max(1, len(rn))
        h1 = sum(x[1] for r in rows[key] if r[0] <= 2022 for x in r[2])
        h2 = sum(x[1] for r in rows[key] if r[0] >= 2023 for x in r[2])
        qtr = [sum(x[1] for x in r[2]) for r in rows[key] if r[2]]
        qpos = sum(1 for v in qtr if v > 0)
        yrs = {}
        for x in tr:
            yrs[cs.year_of(x[0])] = yrs.get(cs.year_of(x[0]), 0) + x[1]
        say("")
        say("  %s: сделок %d · итог %+.0f $ · %+.2f $/сд · случайный вход %+.2f $/сд (%d) · половины %+.0f / %+.0f $ · "
            "кварталов в плюсе %d из %d" % (label, len(tr), tot, avg, ravg, len(rn), h1, h2, qpos, len(qtr)))
        say("    по годам: " + " · ".join("%d: %+.0f $" % (k, v) for k, v in sorted(yrs.items())))
        k1, k2, k3 = tot > 0, avg > base_avg, avg > ravg
        k4 = h1 > 0 and h2 > 0 and qtr and qpos * 2 >= len(qtr)
        say("    критерий: итог+ [%s] · лучше неизменного набора [%s] · лучше случайного входа [%s] · половины и кварталы [%s] → %s"
            % ("да" if k1 else "нет", "да" if k2 else "нет", "да" if k3 else "нет", "да" if k4 else "нет",
               "РАБОТАЕТ" if (k1 and k2 and k3 and k4) else "не работает"))

    # ======================= ЧАСТЬ 2: фазы =======================
    say("")
    say("#" * 110)
    say("ЧАСТЬ 2. СВОИ НАСТРОЙКИ ДЛЯ ФАЗЫ BTC · подбор 2019 … 20.04.2024 · проверка 20.04.2024 … конец данных")
    END = data_end + 1
    for lname, lab in (("SMA200", sma_phase), ("халвинг", halving_phase)):
        say("")
        say("=" * 110)
        say("РАЗМЕТКА «%s»" % lname)
        train = {}   # (cb, фаза) -> [$]
        test = {}
        for cb in COMBOS:
            for sym in have:
                for x in closed(cb, sym, START, SPLIT):
                    train.setdefault((cb, lab(x[0])), []).append(x[1])
                for x in trades(cb, sym, SPLIT, END):
                    test.setdefault((cb, lab(x[0])), []).append(x[1])
        phases = sorted(p for p in ({k[1] for k in train} | {k[1] for k in test}) if p is not None)
        tot_tr = {cb: sum(sum(train.get((cb, p), [])) for p in phases) for cb in COMBOS}
        single = max(COMBOS, key=lambda cb: tot_tr[cb])
        say("  одна лучшая настройка на подборе без фаз: %s (%+.0f $)" % (cname(single), tot_tr[single]))
        sum_ph = sum_def = sum_single = 0.0
        for p in phases:
            cand = [(sum(train[(cb, p)]), cb) for cb in COMBOS if len(train.get((cb, p), [])) >= 100]
            if cand:
                s_tr, cb = max(cand)
                note = ""
            else:
                cb, s_tr, note = single, sum(train.get((single, p), [])), " (мало сделок — без деления)"
            vt = test.get((cb, p), [])
            vd = test.get((DEFAULT, p), [])
            vs = test.get((single, p), [])
            sum_ph += sum(vt)
            sum_def += sum(vd)
            sum_single += sum(vs)
            say("  фаза «%s»: подбор → %s%s (%+.0f $, %d сд) | проверка: по фазе %+.0f $ (%d сд, %+.2f/сд) · "
                "неизменный %+.0f $ (%d) · одна лучшая %+.0f $ (%d)"
                % (p, cname(cb), note, s_tr, len(train.get((cb, p), [])), sum(vt), len(vt), sum(vt) / max(1, len(vt)),
                   sum(vd), len(vd), sum(vs), len(vs)))
            top = sorted(((sum(train.get((x, p), [])), x) for x in COMBOS if len(train.get((x, p), [])) >= 100),
                         reverse=True)[:3]
            say("      три лучших на подборе: " + " · ".join("%s %+.0f $" % (cname(x), s) for s, x in top)
                + " | на проверке у них: " + " · ".join("%+.0f $" % sum(test.get((x, p), [])) for s, x in top))
        k1, k2, k3 = sum_ph > 0, sum_ph > sum_def, sum_ph > sum_single
        say("  ИТОГ проверки: по фазам %+.0f $ · неизменный набор %+.0f $ · одна лучшая без фаз %+.0f $" % (sum_ph, sum_def, sum_single))
        say("  критерий: в плюсе [%s] · лучше неизменного [%s] · лучше одной лучшей без фаз [%s] → %s"
            % ("да" if k1 else "нет", "да" if k2 else "нет", "да" if k3 else "нет",
               "ФАЗЫ РАБОТАЮТ" if (k1 and k2 and k3) else "фазы не работают"))

    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТРЕНДОВЫЙ БОТ РАЗРАБОТЧИКА ПО СИГНАЛУ TT2 + ПРАВИЛО SMA200.

Бот — ровно как в dev_bot_test.py (функция sim_dev, исправленный трейлинг): 4 ордера по 100 $, доливки ПО ТРЕНДУ
на +6%, +10.8%, +14.64% от первого входа (шаг 6%, динамический 0.8), тейк 10% от средней, трейлинг 50% после
первой доливки; без стопа — выход по рынку через 30 дней (ручное ведение тест не моделирует) или стоп 9% от
первого входа. Исполнение 15m, спорная свеча — в худшую сторону, комиссии тейкер 0.055% / мейкер 0.02%,
фандинг 0.01% за 8 ч. Одна позиция на монету.

Вход — сигнал TT2 (30m, ст. ТФ 4ч, по дневному тренду BTC, без MRC), как в sma200_test.py; данные — Binance 15m
(кэш data_btc/). Сигналы — когда у монеты есть 250 дневок. Монеты: основной (21) и новый (20) наборы.

ВАРИАНТЫ (объявлены до прогона):
  A  выше SMA200, без стопа      B  выше SMA200, стоп 9%
  C  всегда, без стопа           D  всегда, стоп 9%            (C, D — для сравнения: что даёт SMA200)
Случайный вход: столько же входов в каждой монете и году, в случайные моменты по тренду BTC 1Д
(для A и B — только в дни, когда BTC выше SMA200), с теми же выходами.

КРИТЕРИЙ (объявлен ДО прогона): вариант A или B «работает», если на ОБОИХ наборах монет:
  1) итог в плюсе и обе половины периода в плюсе;
  2) средняя сделка лучше случайного входа;
  3) средняя сделка лучше того же варианта без SMA200 (A против C, B против D).
Справочно: доход на 100 $ первого ордера, просадка по закрытым сделкам, сколько сделок дошло до доливок.
Функция sim_dev скопирована из dev_bot_test.py без изменений. Лежит рядом с sma200_test.py, coin_select_test.py,
compare_test.py, coin_wf_test.py.
Отчёт: sma200_devbot_report.txt
"""
import bisect
import random
import time
from datetime import datetime, timezone

import coin_select_test as cs
import sma200_test as st

c, cw = cs.c, cs.cw
REPORT = "sma200_devbot_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
VARIANTS = [("A", True, None), ("B", True, 0.09), ("C", False, None), ("D", False, 0.09)]
VNAME = {"A": "выше SMA200, без стопа (30 дн)", "B": "выше SMA200, стоп 9%",
         "C": "всегда, без стопа (30 дн)", "D": "всегда, стоп 9%"}
# --- параметры бота и sim_dev — копия из dev_bot_test.py ---
M15 = 900000
VOL = 100.0
LV = [0.0, 0.06, 0.06 + 0.048, 0.06 + 0.048 + 0.0384]
TP = 0.10
KEEP = 0.5
F_T, F_M = 0.055 / 100, 0.02 / 100
FUND = 0.01 / 100
MAXB = 30 * 96
_out = []


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


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def run_set(set_name, coins, btc):
    t0 = time.time()
    res = {v[0]: [] for v in VARIANTS}       # (T, монета, $, t_выхода)
    rand = {v[0]: [] for v in VARIANTS}
    end_t = 0
    for sym in coins:
        b15 = cs.load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t = [x[0] for x in b15]
        o = [x[1] for x in b15]
        h = [x[2] for x in b15]
        l = [x[3] for x in b15]
        cl = [x[4] for x in b15]
        end_t = max(end_t, t[-1])
        b30 = c.agg(b15, M30)
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        raw = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < st.START or atrp[i] <= 0 or btc.trend(T) != d:
                continue
            if bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            raw.append((T, d))
        y_lo = max(st.START, t[0] + 260 * D1)
        y_hi = t[-1] - 31 * D1
        for name, only_up, stop in VARIANTS:
            busy = 0
            per_year = {}
            rnd = random.Random("%s-%s" % (sym, name))
            for T, d in raw:
                if T < busy or (only_up and not btc.above(T)):
                    continue
                k0 = bisect.bisect_left(t, T)
                if k0 >= len(t) - 1:
                    continue
                pnl, te = sim_dev(t, o, h, l, cl, k0, d, stop)
                res[name].append((T, sym, pnl, te))
                busy = te
                per_year[cs.year_of(T)] = per_year.get(cs.year_of(T), 0) + 1
            for y, cnt in per_year.items():
                a = max(y_lo, int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
                b = min(y_hi, int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000))
                got = tries = 0
                while got < cnt and tries < cnt * 40 and b > a:
                    tries += 1
                    T = rnd.randrange(a, b)
                    d = btc.trend(T)
                    if d == 0 or (only_up and not btc.above(T)):
                        continue
                    k0 = bisect.bisect_left(t, T)
                    if k0 >= len(t) - 1:
                        continue
                    pnl, te = sim_dev(t, o, h, l, cl, k0, d, stop)
                    rand[name].append((T, sym, pnl, te))
                    got += 1
        say("  [%s] %s: сигналов %d · сделок %s · %.0f с" % (
            set_name, sym, len(raw), " / ".join("%s %d" % (v[0], sum(1 for x in res[v[0]] if x[1] == sym))
                                                for v in VARIANTS), time.time() - t0))
    return res, rand, end_t


def stats(tr, end_t):
    n = len(tr)
    tot = sum(x[2] for x in tr)
    if not n:
        return {"n": 0, "tot": 0, "avg": 0, "h1": 0, "h2": 0, "dd": 0, "wr": 0, "yrs": {}}
    tr = sorted(tr)
    mid = (tr[0][0] + end_t) // 2
    eq = peak = dd = 0.0
    for x in sorted(tr, key=lambda x: x[3]):
        eq += x[2]
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    yrs = {}
    for x in tr:
        yrs.setdefault(cs.year_of(x[0]), []).append(x[2])
    return {"n": n, "tot": tot, "avg": tot / n, "h1": sum(x[2] for x in tr if x[0] < mid),
            "h2": sum(x[2] for x in tr if x[0] >= mid), "dd": dd, "wr": 100 * sum(1 for x in tr if x[2] > 0) / n,
            "yrs": yrs, "mid": mid}


def main():
    t0 = time.time()
    say("БОТ РАЗРАБОТЧИКА ПО СИГНАЛУ TT2 + SMA200 · %s · 4 × 100 $, шаг 6/4.8/3.84%%, тейк 10%%, трейлинг 50%%"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    verdict = {"A": True, "B": True}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res, rand, end_t = run_set(set_name, coins, btc)
        S = {k: stats(v, end_t) for k, v in res.items()}
        R = {k: (sum(x[2] for x in v) / len(v) if v else 0.0) for k, v in rand.items()}
        for name, _, _ in VARIANTS:
            s = S[name]
            say("")
            say("  %s %s: сделок %d · WR %.1f%% · итог %+.0f $ · %+.2f $/сд · случайный вход %+.2f $/сд (%d) · "
                "просадка по закрытым %+.0f $" % (name, VNAME[name], s["n"], s["wr"], s["tot"], s["avg"], R[name],
                                                  len(rand[name]), s["dd"]))
            if s["n"]:
                say("      половины (до/после %s): %+.0f / %+.0f $ · по годам: %s" % (
                    st.dstr(s["mid"]), s["h1"], s["h2"],
                    " · ".join("%d: %+.0f $ (%d)" % (y, sum(v), len(v)) for y, v in sorted(s["yrs"].items()))))
        for name, base in (("A", "C"), ("B", "D")):
            s = S[name]
            k1 = s["tot"] > 0 and s["h1"] > 0 and s["h2"] > 0
            k2 = s["avg"] > R[name]
            k3 = s["avg"] > S[base]["avg"]
            ok = k1 and k2 and k3
            verdict[name] = verdict[name] and ok
            say("  критерий %s: итог и половины+ [%s] · лучше случайного входа [%s] · лучше без SMA200 (%s) [%s, %+.2f против %+.2f] → %s"
                % (name, "да" if k1 else "нет", "да" if k2 else "нет", base, "да" if k3 else "нет",
                   s["avg"], S[base]["avg"], "да" if ok else "нет"))
    say("")
    say("=" * 110)
    for name in ("A", "B"):
        say("ИТОГ %s (%s): %s" % (name, VNAME[name], "РАБОТАЕТ на обоих наборах" if verdict[name] else "не работает"))
    say("Для сравнения: правило R2 (TT2, цель 4 / стоп 20 ATR) — около +12.5 $ на сделку при позиции 1000 $ "
        "(≈ +1.25 $ на 100 $ позиции); у бота первый ордер 100 $, а позиция растёт до 400 $ только при доливках.")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

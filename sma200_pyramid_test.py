#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПИРАМИДИНГ ПО ПРАВИЛУ SMA200: маленький пробный ордер на сигнале → большой ордер, когда цена пошла в нашу сторону.

Сигналы — как в sma200_vol_test.py: TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC вчера
закрылся выше SMA200, стоп 20 × ATR не ближе ПОРОГА (20% и 30% — оба варианта), одна позиция на монету
(очередь по выходу базового варианта). Все варианты считаются на ОДНИХ И ТЕХ ЖЕ входах. Binance 15m.
Три набора монет: основной (21), новый (20), третий (40, из топа).

РАЗМЕР: риск на сделку одинаковый — стоп всей позиции (если долив случился) стоит ровно 1 R.
  BASE  весь объём на сигнале; цель 4 ATR, стоп 20 ATR от входа (как сейчас).
  Пирамидинг: 10% объёма на сигнале (пробник), 90% — когда цена пройдёт +k × ATR в нашу сторону;
  общий стоп — тот же уровень (вход − 20 ATR). Если долива не было, пробник идёт к своей цели или стопу.
  PY1  долив на +1 ATR, цель — средняя цена + 4 ATR (как тейк от средней у бота);
  PY2  долив на +2 ATR, цель — средняя + 4 ATR;
  PY3  долив на +1 ATR, цель — та же цена, что у BASE (вход + 4 ATR).
Исполнение без подглядывания: если в свече долива задет и стоп — считается, что долив был, потом стоп (полный
убыток); цель проверяется только со следующей свечи после долива; гэп — по цене открытия.
Комиссия 0.11% за круг на каждый объём, фандинг 0.01% / 8 ч на открытый объём.

КРИТЕРИЙ (объявлен ДО прогона): вариант пирамидинга лучше BASE, если для порога 30% на ВСЕХ ТРЁХ наборах
средний R на сигнал выше, чем у BASE, в обеих половинах периода И сам в плюсе в обеих половинах. Условие «в плюсе»
обязательно: если оба варианта в минусе, «лучше» означает лишь «теряет медленнее» (при неудачном сигнале
пирамидинг теряет только пробник). Справочно — порог 20%, доля сделок с доливом,
WR, средняя прибыль и убыток в R.
Контроль: тест прогнан на случайных ценах (реалистичные свечи из минутного блуждания) — ложных «лучше» быть не должно.
Лежит рядом с sma200_test.py, sma200_vol_test.py, coin_select_test.py. Отчёт: sma200_pyramid_report.txt
"""
import bisect
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

c, cw = cs.c, cs.cw
REPORT = "sma200_pyramid_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
TP, SL = 4.0, 20.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
PROBE = 0.10
THRS = [20.0, 30.0]
VARS = [("BASE", "весь объём на сигнале", None, None), ("PY1", "долив +1 ATR, цель от средней", 1.0, "avg"),
        ("PY2", "долив +2 ATR, цель от средней", 2.0, "avg"), ("PY3", "долив +1 ATR, цель как у BASE", 1.0, "p0")]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def first_hit(h, l, k0, d, fav, adv):
    n = len(h)
    k, step = k0, 4096
    while k < n:
        e = min(n, k + step)
        if d == 1:
            mt, ms_ = h[k:e] >= fav, l[k:e] <= adv
        else:
            mt, ms_ = l[k:e] <= fav, h[k:e] >= adv
        m = mt | ms_
        if m.any():
            j = int(np.argmax(m))
            return k + j, ("sl" if ms_[j] else "tp")
        k, step = e, step * 2
    return None, None


def sim(t, o, h, l, cl, k0, d, a, var):
    """-> (R, долив был?, t_выхода | None). 1 R = стоп всей позиции."""
    _, _, kk, tgt_mode = var
    p0 = o[k0]
    stp = p0 * (1 - d * SL * a)

    def hours(k_end):
        te = t[k_end] + M15 if k_end is not None else t[-1]
        return (te - t[k0]) / 3600000, (te if k_end is not None else None)

    if kk is None:                                         # BASE: позиция P, стоп стоит 1 R
        P = 1.0 / (SL * a)
        tgt = p0 * (1 + d * TP * a)
        k, kind = first_hit(h, l, k0, d, tgt, stp)
        if k is None:
            hr, te = hours(None)
            move = d * (cl[-1] / p0 - 1)
        else:
            hr, te = hours(k)
            move = -SL * a if kind == "sl" else TP * a
        return P * move - P * FEE - P * FUND_8H * hr / 8, False, te
    pa_lvl = p0 * (1 + d * kk * a)
    # объём: стоп всей позиции = 1 R
    loss_unit = PROBE * SL * a + (1 - PROBE) * (SL + kk) * a / (1 + d * kk * a)
    P = 1.0 / loss_unit
    q1 = PROBE * P
    tgt1 = p0 * (1 + d * TP * a)                          # цель пробника, если долива не было
    k, kind = first_hit(h, l, k0, d, pa_lvl, stp)
    if k is None:
        hr, te = hours(None)
        return q1 * d * (cl[-1] / p0 - 1) - q1 * FEE - q1 * FUND_8H * hr / 8, False, te
    if kind == "sl":
        hr, te = hours(k)
        return -q1 * SL * a - q1 * FEE - q1 * FUND_8H * hr / 8, False, te
    pa = max(o[k], pa_lvl) if d == 1 else min(o[k], pa_lvl)  # гэп — по открытию (хуже)
    q2 = (1 - PROBE) * P
    # средняя цена по объёму в $: монет = q/p
    coins = q1 / p0 + q2 / pa
    avg = (q1 + q2) / coins
    tgt = avg * (1 + d * TP * a) if tgt_mode == "avg" else tgt1

    def pnl_at(px):
        return d * (coins * px - (q1 + q2))

    adv_k = l[k] if d == 1 else h[k]
    if (adv_k <= stp) if d == 1 else (adv_k >= stp):       # долив и стоп в одной свече — худшее
        hr, te = hours(k)
        return pnl_at(stp) - (q1 + q2) * FEE - (q1 + q2) * FUND_8H * hr / 8, True, te
    if k + 1 >= len(h):
        hr, te = hours(None)
        return pnl_at(cl[-1]) - (q1 + q2) * FEE, True, None
    k2, kind2 = first_hit(h, l, k + 1, d, tgt, stp)
    if k2 is None:
        hr, te = hours(None)
        px = cl[-1]
    else:
        hr, te = hours(k2)
        px = stp if kind2 == "sl" else tgt
    return pnl_at(px) - (q1 + q2) * FEE - (q1 + q2) * FUND_8H * hr / 8, True, te


def run_set(set_name, coins, btc):
    t0 = time.time()
    res = {(thr, v[0]): [] for thr in THRS for v in VARS}    # (T, R, долив)
    end_t = 0
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
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a_ > b_ for a_, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        raw = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < st.START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
                continue
            if bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            raw.append((T, d, atrp[i] / 100))
        for thr in THRS:
            busy = 0
            for T, d, a in raw:
                if T < busy or SL * a * 100 < thr:
                    continue
                k0 = int(np.searchsorted(t, T, side="left"))
                if k0 >= len(t) - 1:
                    continue
                te0 = None
                for v in VARS:
                    r, added, te = sim(t, o, h, l, cl, k0, d, a, v)
                    res[(thr, v[0])].append((T, r, added))
                    if v[0] == "BASE":
                        te0 = te
                busy = te0 if te0 is not None else 10 ** 15
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return res, end_t


def main():
    t0 = time.time()
    say("ПИРАМИДИНГ ПО ПРАВИЛУ SMA200 · %s · Binance 15m · пробник %.0f%%"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), PROBE * 100))
    btc = st.Btc()
    verdict = {v[0]: True for v in VARS if v[0] != "BASE"}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS), ("третий (топ-50)", sv.THIRD_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res, end_t = run_set(set_name, coins, btc)
        for thr in THRS:
            base = res[(thr, "BASE")]
            if not base:
                continue
            mid = (base[0][0] + end_t) // 2
            say("")
            say("  ПОРОГ СТОПА %.0f%% (сигналов %d)" % (thr, len(base)))
            b1 = np.mean([x[1] for x in base if x[0] < mid]) if base else 0
            b2 = np.mean([x[1] for x in base if x[0] >= mid]) if base else 0
            for name, label, _, _ in VARS:
                v = res[(thr, name)]
                rs = np.array([x[1] for x in v])
                r1 = np.mean([x[1] for x in v if x[0] < mid])
                r2 = np.mean([x[1] for x in v if x[0] >= mid])
                add = [x for x in v if x[2]]
                win = rs[rs > 0]
                los = rs[rs <= 0]
                mark = ""
                if name != "BASE":
                    good = r1 > b1 and r2 > b2 and r1 > 0 and r2 > 0
                    mark = " → лучше BASE и в плюсе в обеих половинах: %s" % ("да" if good else "нет")
                    if thr == THRS[-1]:                        # решение — по строгому порогу (30%)
                        verdict[name] = verdict[name] and good
                say("    %-4s %-30s средний R %+.3f (половины %+.3f / %+.3f) · сумма R %+.1f · WR %4.1f%% · прибыль %+.2f R / убыток %+.2f R"
                    " · долив в %.0f%% сделок%s" % (name, label, rs.mean(), r1, r2, rs.sum(), 100 * len(win) / max(1, len(rs)),
                                                  win.mean() if len(win) else 0, los.mean() if len(los) else 0,
                                                  100 * len(add) / max(1, len(v)), mark))
    say("")
    say("=" * 110)
    for name, ok in verdict.items():
        say("%s (%s): %s" % (name, dict((v[0], v[1]) for v in VARS)[name],
                             "ЛУЧШЕ обычного входа на всех трёх наборах" if ok else "не лучше обычного входа"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

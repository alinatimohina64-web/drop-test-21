#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ВХОД ПОСЛЕ ОТКАТА ВМЕСТО НЕМЕДЛЕННОГО ВХОДА ПО СИГНАЛУ — проверка на истории.

Правило сигнала — как у робота, без изменений: TT2 (30m, старший ТФ 4ч, по тренду BTC 1Д), только когда BTC вчера
закрылся выше SMA200, без MRC, цель 5 / стоп 20 × ATR (стоп не дальше −80%), только ⭐ (стоп не ближе 30%).
Монеты — 41 старая (те, что в работе). Сигналы — те же, что берёт базовое правило при «одной позиции на монету»
(последовательно по каждой монете по немедленному входу); все варианты считаются на ЭТОМ же наборе сигналов.
Движок исполнения — cs.trade (15m, спорная свеча — стоп, комиссия 0.11% за круг, фандинг 0.01% за 8 ч).

ВАРИАНТЫ (15, объявлены до прогона). База: рынок по открытию 15m после сигнала.
  Лимитная заявка на лучшую цену на 0.5 / 1 / 2 ATR (30m) от цены базового входа, живёт 1 час или 4 часа;
  не исполнилась — сигнал пропущен (0 R). Два способа выставить цель и стоп после исполнения:
    A — прежние уровни (цель и стоп остаются там же, где были бы при входе по сигналу: выигрыш на цели больше,
        стоп ближе — как предложено);
    B — цель 5 / стоп 20 × ATR заново от цены исполнения.
  Задержка: вход рынком через 1 / 2 / 4 часа после базового входа, уровни от цены нового входа.
Риск всегда 50 $: позиция = риск / фактический стоп, R = результат / фактический стоп.
Исполнение лимитки: сделка проходит через цену заявки (строго), при гэпе исполняем по открытию; в свече
исполнения цель не засчитывается, а стоп — да (худший вариант). Комиссия лимитки не снижена (консервативно).

КРИТЕРИЙ (объявлен ДО прогона). Вариант лучше немедленного входа, если ВСЕ:
  1) средний R на сигнал (пропущенный вход = 0 R) выше, чем у немедленного входа, и t-статистика парной
     разницы не меньше 2.5 (вариантов 15 — порог выше обычного 2; сигналы коррелированы, t завышена);
  2) прирост положителен в обеих половинах периода и в обеих группах монет (чётные/нечётные по алфавиту);
  3) прирост на реальных сигналах больше прироста того же варианта на случайных входах (на случайных входах
     перевеса быть не должно; если там прирост больше +0.02 R и больше половины прироста на сигналах —
     исполнение оптимистично, результат недействителен);
  4) исполнено не меньше 50% заявок.
ДОПОЛНИТЕЛЬНО печатается: насколько глубоко цена идёт против входа за 1ч / 4ч / 12ч после сигнала — у сигналов
и у случайных входов (в ATR 30m), чтобы проверить «после сигнала цена чаще идёт против».
ОГРАНИЧЕНИЯ: слоты и пересечения сделок не моделируются (R на сигнал); ожидающая заявка слот не занимает;
в реальности вход рынком тоже приходит позже закрытия свечи — варианты с задержкой показывают чувствительность.
Контрольный прогон на синтетических ценах не добавлен: базой служит случайный вход на реальных свечах.

Запуск: python pullback_entry_test.py (данные из кэша data_btc/). Переменные: START_YEAR (2019), RND_N (300),
COINS_LIMIT. Отчёт: pullback_entry_report.txt
"""
import bisect
import os
import random
import time
from datetime import datetime, timezone

import numpy as np

import sma200_test as st

cs, c, cw = st.cs, st.c, st.cw
REPORT = "pullback_entry_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
TP_X, SL_X, SL_CAP, STAR = 5.0, 20.0, 0.80, 0.30
NOTIONAL, FEE = cs.NOTIONAL, cs.FEE
START = int(datetime(int(os.environ.get("START_YEAR") or 2019), 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
RND_N = int(os.environ.get("RND_N") or 300)
DEPTHS = (0.5, 1.0, 2.0)
EXPIRY = (4, 16)          # баров по 15m: 1 час, 4 часа
DELAYS = (4, 8, 16)       # 1 час, 2 часа, 4 часа
WINDOWS = (4, 16, 48)     # окна для «хода против входа»: 1ч, 4ч, 12ч
WLAB = {4: "1ч", 8: "2ч", 16: "4ч", 48: "12ч"}

VARS = [("base",)]
for _anchor in ("A", "B"):
    for _depth in DEPTHS:
        for _exp in EXPIRY:
            VARS.append(("L", _anchor, _depth, _exp))
for _dl in DELAYS:
    VARS.append(("delay", _dl))
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def vname(v):
    if v[0] == "base":
        return "немедленный вход (как сейчас)"
    if v[0] == "delay":
        return "рынок через %s" % WLAB[v[1]]
    _, anchor, depth, exp = v
    return "лимит лучше на %g ATR, ждать %s, %s" % (
        depth, WLAB[exp], "уровни прежние" if anchor == "A" else "цель/стоп от входа")


def run_variant(var, A, k0, d, a, slf):
    """Возвращает (R, исполнено)."""
    t15, o, h, l, cl = A
    n = len(t15)
    p0 = float(o[k0])
    tp0, sl0 = TP_X * a, slf
    kind = var[0]
    if kind == "base":
        pnl, r, te = cs.trade(t15, h, l, cl, k0, d, p0, tp0, sl0)
        return r, True
    if kind == "delay":
        kd = k0 + var[1]
        if kd >= n - 1:
            return None, False
        pnl, r, te = cs.trade(t15, h, l, cl, kd, d, float(o[kd]), tp0, sl0)
        return r, True
    _, anchor, depth, exp = var
    lim = p0 * (1 - d * depth * a)
    e = min(n, k0 + exp)
    m = (l[k0:e] < lim) if d == 1 else (h[k0:e] > lim)
    if not m.any():
        return None, False
    kf = k0 + int(np.argmax(m))
    fill = min(lim, float(o[kf])) if d == 1 else max(lim, float(o[kf]))
    if anchor == "A":
        tgt_abs = p0 * (1 + d * tp0)
        stp_abs = p0 * (1 - d * sl0)
        tp_act = d * (tgt_abs - fill) / fill
        sl_act = d * (fill - stp_abs) / fill
    else:
        stp_abs = fill * (1 - d * sl0)
        tp_act, sl_act = tp0, sl0
    if tp_act <= 0 or sl_act <= 0:
        return None, False
    adv = float(l[kf]) if d == 1 else float(h[kf])
    if (d == 1 and adv <= stp_abs) or (d == -1 and adv >= stp_abs):
        pnl = NOTIONAL * (-sl_act) - NOTIONAL * FEE
        return pnl / (NOTIONAL * sl_act), True
    if kf + 1 >= n - 1:
        return None, False
    pnl, r, te = cs.trade(t15, h, l, cl, kf + 1, d, fill, tp_act, sl_act)
    return r, True


def mae(A, k0, d, a):
    t15, o, h, l, cl = A
    p0 = float(o[k0])
    res = []
    for w in WINDOWS:
        e = min(len(l), k0 + w)
        if e <= k0:
            res.append(None)
            continue
        ex = (p0 - float(l[k0:e].min())) / p0 if d == 1 else (float(h[k0:e].max()) - p0) / p0
        res.append(max(ex, 0.0) / a)
    return res


def make_rec(sym, T, d, a, slf, k0, A):
    res = []
    for var in VARS:
        r, ok = run_variant(var, A, k0, d, a, slf)
        res.append(r if ok else None)
    return {"sym": sym, "T": T, "res": res, "mae": mae(A, k0, d, a)}


def prep(sym, btc):
    b15 = cs.load_15m(sym)
    if len(b15) < 20000:
        return None
    t15 = np.array([x[0] for x in b15], dtype=np.int64)
    o = np.array([x[1] for x in b15])
    h = np.array([x[2] for x in b15])
    l = np.array([x[3] for x in b15])
    cl = np.array([x[4] for x in b15])
    A = (t15, o, h, l, cl)
    d1 = c.agg(b15, D1)
    dt = [x[0] for x in d1]
    b30 = c.agg(b15, M30)
    b30t = [x[0] for x in b30]
    atrp = c.atr_pct(b30)
    h4a = c.agg(b15, H4)
    c4 = [x[4] for x in h4a]
    hu = [x > y for x, y in zip(c.ema(c4, 9), c.ema(c4, 21))]
    cand = []
    for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
        if T < START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
            continue
        a = atrp[i] / 100
        if SL_X * a < STAR or bisect.bisect_right(dt, T - D1) - 1 < 250:
            continue
        k0 = int(np.searchsorted(t15, T, side="left"))
        if k0 >= len(t15) - 60:
            continue
        cand.append((T, d, a, min(SL_X * a, SL_CAP), k0))
    sig, busy = [], 0
    for T, d, a, slf, k0 in cand:
        if T < busy:
            continue
        pnl, r, te = cs.trade(t15, h, l, cl, k0, d, float(o[k0]), TP_X * a, slf)
        busy = te if te is not None else 10 ** 15
        sig.append(make_rec(sym, T, d, a, slf, k0, A))
    rnd = []
    y0 = max(START, int(t15[0]) + 260 * D1)
    y1 = int(t15[-1]) - 5 * D1
    rg = random.Random("pb-" + sym)
    tries = 0
    while len(rnd) < RND_N and tries < RND_N * 30 and y1 > y0:
        tries += 1
        T = rg.randrange(y0, y1)
        d = btc.trend(T)
        if d == 0 or not btc.above(T):
            continue
        j = bisect.bisect_right(b30t, T - M30) - 1
        if j < 20 or atrp[j] <= 0:
            continue
        a = atrp[j] / 100
        if SL_X * a < STAR or bisect.bisect_right(dt, T - D1) - 1 < 250:
            continue
        k0 = int(np.searchsorted(t15, T, side="left"))
        if k0 >= len(t15) - 60:
            continue
        rnd.append(make_rec(sym, T, d, a, min(SL_X * a, SL_CAP), k0, A))
    return {"sig": sig, "rnd": rnd, "end": int(t15[-1])}


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def fmt(x):
    return "—" if x != x else "%+.3f" % x


def vals(recs, vi):
    return [0.0 if r["res"][vi] is None else r["res"][vi] for r in recs]


def tstat(dd):
    n = len(dd)
    if n < 3:
        return float("nan")
    m = sum(dd) / n
    var = sum((v - m) ** 2 for v in dd) / (n - 1)
    return m / ((var / n) ** 0.5) if var > 0 else float("nan")


def main():
    t0 = time.time()
    say("ВХОД ПОСЛЕ ОТКАТА ВМЕСТО НЕМЕДЛЕННОГО · %s · Binance 15m · риск 50 $ на сигнал"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    coins = [s for s in dict.fromkeys(list(st.MAIN_COINS) + list(st.NEW_COINS)) if s != "BTC"]
    lim = int(os.environ.get("COINS_LIMIT") or 0)
    if lim:
        coins = coins[:lim]
    P = {}
    for sym in coins:
        p = prep(sym, btc)
        if p:
            P[sym] = p
            say("  %s: сигналов %d, случайных %d · %.0f с" % (sym, len(p["sig"]), len(p["rnd"]), time.time() - t0))
    allsig = [r for p in P.values() for r in p["sig"]]
    allrnd = [r for p in P.values() for r in p["rnd"]]
    if not allsig:
        say("Нет сигналов.")
        return
    end_t = max(p["end"] for p in P.values())
    mid = (min(r["T"] for r in allsig) + end_t) // 2
    grp = {sym: i % 2 for i, sym in enumerate(sorted(P))}

    say("")
    say("=" * 110)
    say("ХОД ЦЕНЫ ПРОТИВ ВХОДА ПОСЛЕ СИГНАЛА (в ATR 30m от цены входа; больше — глубже откат против позиции)")
    for wi, w in enumerate(WINDOWS):
        for name, recs in (("сигналы", allsig), ("случайные входы", allrnd)):
            v = sorted(r["mae"][wi] for r in recs if r["mae"][wi] is not None)
            if not v:
                continue
            say("  за %-4s %-16s n %5d · среднее %.2f · медиана %.2f · ≥1 ATR %3.0f%% · ≥2 ATR %3.0f%%" % (
                WLAB[w], name, len(v), mean(v), v[len(v) // 2],
                100 * sum(1 for x in v if x >= 1) / len(v), 100 * sum(1 for x in v if x >= 2) / len(v)))

    base_s, base_r = vals(allsig, 0), vals(allrnd, 0)
    say("")
    say("=" * 110)
    say("ВАРИАНТЫ ВХОДА (R на сигнал, пропущенный вход = 0). Сигналов %d, случайных входов %d; граница половин %s"
        % (len(allsig), len(allrnd), datetime.fromtimestamp(mid / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))
    say("  %-46s R на сигнал %s · WR %.0f%% · случайный вход %s" % (
        vname(VARS[0]), fmt(mean(base_s)), 100 * sum(1 for x in base_s if x > 0) / len(base_s), fmt(mean(base_r))))
    passed = []
    for vi in range(1, len(VARS)):
        vs = vals(allsig, vi)
        diffs = [x - y for x, y in zip(vs, base_s)]
        dm, t = mean(diffs), tstat(diffs)
        filled = [r["res"][vi] for r in allsig if r["res"][vi] is not None]
        fill = len(filled) / len(allsig)
        h1 = mean([x for x, r in zip(diffs, allsig) if r["T"] < mid])
        h2 = mean([x for x, r in zip(diffs, allsig) if r["T"] >= mid])
        g0 = mean([x for x, r in zip(diffs, allsig) if grp[r["sym"]] == 0])
        g1 = mean([x for x, r in zip(diffs, allsig) if grp[r["sym"]] == 1])
        rv = vals(allrnd, vi)
        rd = mean([x - y for x, y in zip(rv, base_r)])
        k1 = dm > 0 and t >= 2.5
        k2 = h1 > 0 and h2 > 0 and g0 > 0 and g1 > 0
        k3 = dm > rd
        k4 = fill >= 0.5
        sus = rd > 0.02 and rd > 0.5 * dm
        ok = k1 and k2 and k3 and k4 and not sus
        if ok:
            passed.append(vname(VARS[vi]))
        say("  %-46s исполнено %3.0f%% · R/исполн. %s · на сигнал %s · прирост %s (t %.1f) · половины %s/%s · группы %s/%s · "
            "случайные: прирост %s → %s%s" % (
                vname(VARS[vi]), 100 * fill, fmt(mean(filled)), fmt(mean(vs)), fmt(dm), t, fmt(h1), fmt(h2),
                fmt(g0), fmt(g1), fmt(rd), "ЛУЧШЕ" if ok else "не лучше",
                " ⚠ исполнение подозрительно оптимистично" if sus else ""))
    say("")
    say("=" * 110)
    say("ИТОГ: прошло критерий вариантов из %d: %d%s" % (len(VARS) - 1, len(passed),
                                                       (" — " + "; ".join(passed)) if passed else ""))
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

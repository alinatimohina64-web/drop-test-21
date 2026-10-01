#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200: КАК ВЫЙТИ ИЗ СДЕЛКИ С МАКСИМУМОМ И КАКИЕ СДЕЛКИ НЕ БРАТЬ.

Вход — как R2 в sma200_test.py: сигнал TT2 (30m, ст. ТФ 4ч, по тренду BTC 1Д, без MRC), только когда BTC вчера
закрылся выше SMA200; стоп 20 × ATR 30m; одна позиция на монету; данные Binance 15m; позиция 1000 $;
комиссия 0.11% за круг, фандинг 0.01% за 8 ч. Монеты: основной (21) и новый (20) наборы.

ВЫХОДЫ (объявлены до прогона):
  E0  цель 4 ATR — закрыть всё (как сейчас, проверено);
  E1  на 4 ATR не закрывать, включить трейлинг: закрыть, когда прибыль упадёт до 50% от максимальной
      (как trailing_stop_levels = 50 в GHOST);
  E2  то же, но сохранить 70% максимальной прибыли;
  E3  на 4 ATR закрыть половину, вторую половину вести трейлингом 50%.
  ВСЕ ВЫХОДЫ СЧИТАЮТСЯ НА ОДНИХ И ТЕХ ЖЕ ВХОДАХ (входы E0: одна позиция на монету по выходам E0), чтобы разница
  была только от выхода, а не от числа сделок (контроль на случайных ценах показал, что иначе вариант с более
  длинными сделками «выигрывает» просто тем, что сделок меньше).
  Трейлинг — без подглядывания: внутри 15m-свечи стоп проверяется по максимуму, достигнутому ДО этой свечи,
  новый максимум учитывается со следующей свечи; если свеча открылась ниже трейлинга — выход по открытию.
  До 4 ATR действует стоп 20 ATR. Сделка без выхода — закрывается по последней цене.

РАЗМЕР СТОПА (для E0): сделки по корзинам «стоп ≤ 15%», «15–30%», «> 30%» от цены входа (стоп = 20 × ATR%).

КРИТЕРИИ (объявлены ДО прогона):
  а) трейлинг-вариант (E1/E2/E3) лучше E0, если на ОБОИХ наборах: итог больше, чем у E0, и в ОБЕИХ половинах
     периода; средняя разница с E0 на тех же входах не меньше +2 $ на сделку и t ≥ 2. Запас +2 $ — погрешность
     симуляции трейлинга, измеренная на случайных ценах (на них трейлинг получал фору ≈ +0.5…3 $ на сделку);
  б) правило «не брать сделку, если стоп > 30%» принимается, если на ОБОИХ наборах сделки этой корзины в среднем
     в минусе и без них итог E0 больше.
Лежит рядом с sma200_test.py и coin_select_test.py. Отчёт: sma200_exit_report.txt
"""
import bisect
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st

c, cw = cs.c, cs.cw
REPORT = "sma200_exit_report.txt"
M15, M30, D1, H4 = cs.M15, cs.M30, cs.D1, cs.H4
NOT = 1000.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
TP, SL = 4.0, 20.0
EXITS = [("E0", "цель 4 ATR, закрыть всё"), ("E1", "с 4 ATR трейлинг 50%"),
         ("E2", "с 4 ATR трейлинг 70%"), ("E3", "половина на 4 ATR + трейлинг 50%")]
BUCKETS = [("стоп ≤ 15%", 0, 15), ("стоп 15–30%", 15, 30), ("стоп > 30%", 30, 1e9)]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def first_hit(h, l, k0, d, lvl_fav, lvl_adv):
    """Первая 15m-свеча с k0, где задета цель (fav) или стоп (adv); в одной свече — стоп. -> (k, 'tp'|'sl') или (None, None)."""
    n = len(h)
    k, step = k0, 4096
    while k < n:
        e = min(n, k + step)
        if d == 1:
            mt, ms_ = h[k:e] >= lvl_fav, l[k:e] <= lvl_adv
        else:
            mt, ms_ = l[k:e] <= lvl_fav, h[k:e] >= lvl_adv
        m = mt | ms_
        if m.any():
            j = int(np.argmax(m))
            return k + j, ("sl" if ms_[j] else "tp")
        k, step = e, step * 2
    return None, None


def trail_exit(t, o, h, l, cl, k_act, d, p0, sl_px, keep, peak0):
    """С свечи после активации: трейлинг keep от максимальной прибыли. -> (цена выхода, индекс | None)."""
    peak = peak0
    n = len(o)
    for k in range(k_act + 1, n):
        trl = p0 + keep * (peak - p0) if d == 1 else p0 - keep * (p0 - peak)
        stp = max(trl, sl_px) if d == 1 else min(trl, sl_px)
        if (o[k] <= stp) if d == 1 else (o[k] >= stp):
            return o[k], k
        if (l[k] <= stp) if d == 1 else (h[k] >= stp):
            return stp, k
        fav = h[k] if d == 1 else l[k]
        if (fav > peak) if d == 1 else (fav < peak):
            peak = fav
    return cl[-1], None


def sim(t, o, h, l, cl, k0, d, a, variant):
    """-> ($, t_выхода | None)"""
    p0 = o[k0]
    tgt = p0 * (1 + d * TP * a)
    sl_px = p0 * (1 - d * SL * a)
    k, kind = first_hit(h, l, k0, d, tgt, sl_px)

    def fund(k_end):
        te = t[k_end] + M15 if k_end is not None else t[-1]
        return NOT * FUND_8H * (te - t[k0]) / 3600000 / 8, (te if k_end is not None else None)

    if k is None:
        f, te = fund(None)
        return NOT * d * (cl[-1] / p0 - 1) - NOT * FEE - f, None
    if kind == "sl":
        f, te = fund(k)
        return -NOT * SL * a - NOT * FEE - f, te
    if variant == "E0":
        f, te = fund(k)
        return NOT * TP * a - NOT * FEE - f, te
    keep = 0.7 if variant == "E2" else 0.5
    peak0 = h[k] if d == 1 else l[k]
    px, kx = trail_exit(t, o, h, l, cl, k, d, p0, sl_px, keep, peak0)
    f, te = fund(kx)
    run = NOT * d * (px / p0 - 1)
    if variant == "E3":
        return 0.5 * NOT * TP * a + 0.5 * run - NOT * FEE - f, te
    return run - NOT * FEE - f, te


def run_set(set_name, coins, btc):
    t0 = time.time()
    res = {e[0]: [] for e in EXITS}      # (T, $, стоп %)
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
        busy = 0
        for T, d, a in raw:
            if T < busy:
                continue
            k0 = int(np.searchsorted(t, T, side="left"))
            if k0 >= len(t) - 1:
                continue
            te0 = None
            for name, _ in EXITS:
                pnl, te = sim(t, o, h, l, cl, k0, d, a, name)
                res[name].append((T, pnl, SL * a * 100))
                if name == "E0":
                    te0 = te
            busy = te0 if te0 is not None else 10 ** 15
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return res, end_t


def main():
    t0 = time.time()
    say("SMA200: ВЫХОДЫ И РАЗМЕР СТОПА · %s · Binance 15m · позиция 1000 $"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    tots = {}
    bucket_ok = True
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        res, end_t = run_set(set_name, coins, btc)
        t_first = min(x[0] for v in res.values() for x in v)
        mid = (t_first + end_t) // 2
        say("")
        for name, label in EXITS:
            v = res[name]
            n = len(v)
            tot = sum(x[1] for x in v)
            wins = [x[1] for x in v if x[1] > 0]
            loss = [x[1] for x in v if x[1] <= 0]
            h1 = sum(x[1] for x in v if x[0] < mid)
            h2 = sum(x[1] for x in v if x[0] >= mid)
            yrs = {}
            for x in v:
                yrs[cs.year_of(x[0])] = yrs.get(cs.year_of(x[0]), 0) + x[1]
            dif = np.array([x[1] - y[1] for x, y in zip(v, res["E0"])])
            dm = float(dif.mean()) if len(dif) else 0.0
            dt_ = dm / (float(dif.std()) / len(dif) ** 0.5) if len(dif) > 1 and dif.std() > 0 else 0.0
            tots[(set_name, name)] = (tot, h1, h2, dm, dt_)
            say("  %s %-32s сделок %4d · WR %4.1f%% · итог %+7.0f $ · %+6.2f $/сд · средняя прибыль %+.1f / убыток %+.1f $"
                % (name, label, n, len(wins) / max(1, n) * 100, tot, tot / max(1, n),
                   sum(wins) / max(1, len(wins)), sum(loss) / max(1, len(loss))))
            if name != "E0":
                say("      разница с E0 на тех же входах: %+.2f $ на сделку (t = %.1f)" % (dm, dt_))
            say("      половины (до/после %s): %+.0f / %+.0f $ · по годам: %s" % (
                st.dstr(mid), h1, h2, " · ".join("%d: %+.0f" % (y, s) for y, s in sorted(yrs.items()))))
        say("")
        say("  РАЗМЕР СТОПА (E0):")
        v = res["E0"]
        big = [x for x in v if x[2] > 30]
        for nm, lo, hi in BUCKETS:
            b = [x for x in v if lo < x[2] <= hi] if lo > 0 else [x for x in v if x[2] <= hi]
            say("    %-12s сделок %4d · WR %4.1f%% · итог %+7.0f $ · %+6.2f $/сд" % (
                nm, len(b), sum(1 for x in b if x[1] > 0) / max(1, len(b)) * 100, sum(x[1] for x in b),
                sum(x[1] for x in b) / max(1, len(b))))
        avg_big = sum(x[1] for x in big) / max(1, len(big))
        ok = len(big) > 0 and avg_big < 0 and sum(x[1] for x in big) < 0
        bucket_ok = bucket_ok and ok
        say("    стоп > 30%%: в среднем в минусе и без них итог больше — %s" % ("да" if ok else "нет"))

    say("")
    say("=" * 110)
    say("КРИТЕРИЙ а) трейлинг лучше фиксированной цели (итог и половины больше, разница ≥ +2 $/сд и t ≥ 2 — на обоих наборах):")
    best = None
    for name, label in EXITS[1:]:
        ok = True
        cells = []
        for set_name in ("основной", "новый"):
            a, b = tots[(set_name, name)], tots[(set_name, "E0")]
            ok = ok and a[0] > b[0] and a[1] > b[1] and a[2] > b[2] and a[3] >= 2.0 and a[4] >= 2.0
            cells.append("%s: %+.0f против %+.0f (половины %+.0f/%+.0f против %+.0f/%+.0f; разница %+.2f $/сд, t %.1f)"
                         % (set_name, a[0], b[0], a[1], a[2], b[1], b[2], a[3], a[4]))
        say("  %s %s — %s → %s" % (name, label, " · ".join(cells), "ЛУЧШЕ" if ok else "не лучше"))
        if ok and (best is None or sum(tots[(s, name)][0] for s in ("основной", "новый")) >
                   sum(tots[(s, best)][0] for s in ("основной", "новый"))):
            best = name
    say("  → выход: %s" % (("%s (%s)" % (best, dict(EXITS)[best])) if best else "оставить E0 — цель 4 ATR"))
    say("КРИТЕРИЙ б) не брать сделки со стопом > 30%%: %s" % ("ПРИНЯТО" if bucket_ok else "не принято"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СИГНАЛЫ TT2 БЕЗ СТОПА — держать до цели. Какие монеты «выгребают» сделку и выдерживает ли общий счёт.

Сигналы — проверенный набор индикатора: график 30m, старший ТФ 4ч, только по дневному тренду BTC,
фильтр MRC (канал по дневкам, упрощённая схема, уровень 2). Цель 5 × ATR 30m от входа. СТОПА НЕТ:
сделка закрывается только на цели. Вход по открытию 15m после сигнала, одна позиция на монету за раз.

Счёт: субаккаунт 1000 $, кросс-маржа, каждая сделка — маржа 100 $ × плечо 10 = позиция 1000 $.
Новая сделка открывается, только если свободной маржи хватает (баланс с учётом плавающего результата
минус маржа открытых позиций ≥ 100 $). Комиссия 0.11% за круг, фандинг 0.01% за 8 ч против позиции
(осторожная оценка). Ликвидация счёта — если капитал (с учётом худшей цены дня по всем открытым позициям)
опускается до поддерживающей маржи (0.5% от суммы позиций). Монеты — 21 «старая» монета (как в тестах),
2022–2026.

Контроль: случайные входы (столько же по каждой монете и году, в сторону дневного тренда BTC, те же
правила) — чтобы понять, дают ли что-то сами сигналы, или «без стопа» выгребает любой вход.

КРИТЕРИЙ (объявлен ДО прогона). Подход «без стопа» годится, если одновременно:
  1) счёт ни разу не ликвидирован и капитал ни разу не опускался ниже 500 $ (половина депозита);
  2) итог на конец периода (закрытые + плавающий результат незакрытых) в плюсе;
  3) сигналы лучше случайного входа (итог на сделку).
По монетам: «выдерживает», если все её сделки, кроме открытых в последние 30 дней, дошли до цели,
а худшая просадка внутри сделки меньше 50%.
Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py, mrc_filter_test.py. Отчёт: nostop_report.txt
"""
import bisect
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw
from mrc_filter_test import supersmoother, hot_flag

REPORT = "nostop_report.txt"
M15, M30 = 900000, 1800000
DEPOSIT, MARGIN, LEV = 1000.0, 100.0, 10.0
NOTIONAL = MARGIN * LEV
TP_ATR = 5.0
FEE = 0.11 / 100
FUND_8H = 0.01 / 100
MAINT = 0.005
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def ts(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def run_trade(t15, h, l, k0, d, p0, tgt):
    """Первый бар, где цель достигнута (или None). Возвращает (k_exit, MAE в долях)."""
    if d == 1:
        hit = np.nonzero(h[k0:] >= tgt)[0]
    else:
        hit = np.nonzero(l[k0:] <= tgt)[0]
    k1 = k0 + int(hit[0]) if len(hit) else None
    end = (k1 + 1) if k1 is not None else len(h)
    if d == 1:
        mae = (p0 - l[k0:end].min()) / p0
    else:
        mae = (h[k0:end].max() - p0) / p0
    return k1, max(0.0, mae)


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("TT2 БЕЗ СТОПА, ДО ЦЕЛИ · %s · монет %d · счёт %.0f $, маржа %.0f × %.0f на сделку, кросс"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), DEPOSIT, MARGIN, LEV))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)

    def btc_at(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    rnd = random.Random(7)
    now = None
    cand = {"sig": [], "rnd": []}           # (T, монета, d, p0, tgt, k0) — кандидаты по монетам
    arrays = {}
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 50000 or len(d1) < 400:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        dd = c.agg(b15, c.D1)
        arrays[sym] = (t15, o, h, l, cl, [x[0] for x in dd], [x[2] for x in dd], [x[3] for x in dd], [x[4] for x in dd])
        now = int(t15[-1]) if now is None else max(now, int(t15[-1]))
        b30 = c.agg(b15, M30)
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        dt = [x[0] for x in d1]
        mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in d1], 200)
        rng = supersmoother(c.trs(d1), 200)
        n_sig = 0
        per_year = {}
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, c.H4):
            if T < mt.START or atrp[i] <= 0 or btc_at(T) != d:
                continue
            jd = bisect.bisect_right(dt, T - c.D1) - 1
            if jd < 250 or hot_flag("simple", d, b30[i][4], mean[jd], rng[jd]):
                continue
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                continue
            p0 = o[k0]
            tgt = p0 * (1 + d * TP_ATR * atrp[i] / 100)
            cand["sig"].append((T, sym, d, p0, tgt, k0))
            per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
            n_sig += 1
        b30t = [x[0] for x in b30]
        for y, cnt in per_year.items():
            y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            y1 = min(int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), int(t15[-1]) - c.D1)
            for _ in range(cnt):
                T = rnd.randrange(y0, y1)
                d = btc_at(T)
                j = bisect.bisect_right(b30t, T - M30) - 1
                if d == 0 or j < 20 or atrp[j] <= 0:
                    continue
                k0 = int(np.searchsorted(t15, T, side="left"))
                if k0 >= len(t15) - 1:
                    continue
                p0 = o[k0]
                cand["rnd"].append((T, sym, d, p0, p0 * (1 + d * TP_ATR * atrp[j] / 100), k0))
        say("  %s: сигналов после фильтров %d · %.0f с" % (sym, n_sig, time.time() - t0))

    for mode, title in (("sig", "СИГНАЛЫ TT2"), ("rnd", "КОНТРОЛЬ — СЛУЧАЙНЫЕ ВХОДЫ")):
        say("")
        say("#" * 110)
        say(title)
        evs = sorted(cand[mode])
        busy = {}                                   # монета -> время выхода текущей позиции
        open_ = []                                  # открытые позиции для учёта маржи
        trades = []
        skipped_margin = 0
        realized = 0.0
        # события: проходим кандидатов по времени, закрываем позиции, у которых выход раньше
        for T, sym, d, p0, tgt, k0 in evs:
            t15, o, h, l, cl, dts, dhs, dls, dcs = arrays[sym]
            # закрыть позиции, вышедшие до T
            still = []
            for p in open_:
                if p["exit"] is not None and p["exit"] <= T:
                    realized += p["pnl"]
                else:
                    still.append(p)
            open_ = still
            if busy.get(sym, 0) > T:
                continue
            # свободная маржа: баланс + плавающий результат (по цене закрытия последней 15m) − маржа открытых
            flo = 0.0
            for p in open_:
                a = arrays[p["sym"]]
                kk = int(np.searchsorted(a[0], T, side="right")) - 1
                pxn = a[4][max(kk, 0)]
                flo += NOTIONAL * p["d"] * (pxn / p["p0"] - 1)
            free = DEPOSIT + realized + flo - MARGIN * len(open_)
            if free < MARGIN:
                skipped_margin += 1
                continue
            k1, mae = run_trade(t15, h, l, k0, d, p0, tgt)
            if k1 is not None:
                te = int(t15[k1]) + M15
                hours = (te - T) / 3600000
                pnl = NOTIONAL * abs(tgt / p0 - 1) - NOTIONAL * FEE - NOTIONAL * FUND_8H * hours / 8
            else:
                te = None
                hours = (now - T) / 3600000
                pnl = NOTIONAL * d * (cl[-1] / p0 - 1) - NOTIONAL * FEE - NOTIONAL * FUND_8H * hours / 8
            p = {"T": T, "sym": sym, "d": d, "p0": p0, "exit": te, "pnl": pnl, "mae": mae, "hours": hours}
            trades.append(p)
            open_.append(p)
            busy[sym] = te if te is not None else 10 ** 15
        # капитал по дням (с худшей ценой дня по каждой открытой позиции)
        day0 = mt.START - mt.START % c.D1
        days = list(range(day0, now, c.D1))
        eq_min, eq_min_day, max_open, liq_day = 10 ** 9, None, 0, None
        closed_by = {}
        for p in trades:
            if p["exit"] is not None:
                closed_by.setdefault(p["exit"] - p["exit"] % c.D1, []).append(p["pnl"])
        realized = 0.0
        for D in days:
            realized += sum(closed_by.get(D - c.D1, []))
            worst = 0.0
            n_open = 0
            notional_open = 0.0
            for p in trades:
                if p["T"] <= D + c.D1 and (p["exit"] is None or p["exit"] > D):
                    a = arrays[p["sym"]]
                    j = bisect.bisect_left(a[5], D)
                    if j >= len(a[5]) or a[5][j] != D:
                        continue
                    px = a[7][j] if p["d"] == 1 else a[6][j]
                    worst += NOTIONAL * p["d"] * (px / p["p0"] - 1)
                    n_open += 1
                    notional_open += NOTIONAL
            eq = DEPOSIT + realized + worst
            max_open = max(max_open, n_open)
            if eq < eq_min:
                eq_min, eq_min_day = eq, D
            if liq_day is None and n_open and eq <= MAINT * notional_open:
                liq_day = D
        n = len(trades)
        closed = [p for p in trades if p["exit"] is not None]
        open_end = [p for p in trades if p["exit"] is None]
        total = sum(p["pnl"] for p in trades)
        say("  сделок %d · дошли до цели %d · открыты на конец %d · пропущено из-за маржи %d · одновременно до %d позиций"
            % (n, len(closed), len(open_end), skipped_margin, max_open))
        if closed:
            hrs = sorted(p["hours"] for p in closed)
            say("  время до цели: медиана %.1f дн · 90%% сделок быстрее %.1f дн · самая долгая %.0f дн"
                % (hrs[len(hrs) // 2] / 24, hrs[int(len(hrs) * 0.9)] / 24, hrs[-1] / 24))
        maes = sorted(p["mae"] for p in trades)
        if maes:
            say("  просадка внутри сделки: медиана %.1f%% · 10%% худших > %.1f%% · худшая %.1f%% (= %.0f $ на позиции 1000 $)"
                % (100 * maes[len(maes) // 2], 100 * maes[int(len(maes) * 0.9)], 100 * maes[-1], NOTIONAL * maes[-1]))
        say("  итог: закрытые %+.0f $ · плавающий незакрытых %+.0f $ · всего %+.0f $ (%+.2f $ на сделку)"
            % (sum(p["pnl"] for p in closed), sum(p["pnl"] for p in open_end), total, total / max(1, n)))
        say("  капитал счёта: минимум %.0f $ (%s) · %s"
            % (eq_min, ts(eq_min_day) if eq_min_day else "—",
               "ЛИКВИДАЦИЯ %s" % ts(liq_day) if liq_day else "ликвидаций не было"))
        yrs = {}
        for p in trades:
            yrs.setdefault(year_of(p["T"]), []).append(p["pnl"])
        say("  по году входа: " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(v), len(v)) for y, v in sorted(yrs.items())))
        if mode == "sig":
            sig_res = (liq_day, eq_min, total, n)
            say("")
            say("  ПО МОНЕТАМ (сделок · дошли до цели · открыто · худшая просадка · дольше всего · итог):")
            rows = []
            for sym in sorted({p["sym"] for p in trades}):
                ps = [p for p in trades if p["sym"] == sym]
                cl_ = [p for p in ps if p["exit"] is not None]
                op_ = [p for p in ps if p["exit"] is None]
                old_open = [p for p in op_ if now - p["T"] > 30 * c.D1]
                worst = max(p["mae"] for p in ps)
                longest = max(p["hours"] for p in ps) / 24
                tot = sum(p["pnl"] for p in ps)
                ok = not old_open and worst < 0.5
                rows.append((ok, tot, sym, len(ps), len(cl_), len(op_), worst, longest))
            for ok, tot, sym, n_, nc, no, worst, longest in sorted(rows, key=lambda r: (-r[0], -r[1])):
                say("    %-5s %3d · %3d · %d · %5.1f%% · %4.0f дн · %+6.0f $ %s"
                    % (sym, n_, nc, no, 100 * worst, longest, tot, "✔ выдерживает" if ok else "✖"))
        else:
            rnd_res = (total, n)
    liq_day, eq_min, total, n = sig_res
    c1 = liq_day is None and eq_min >= DEPOSIT * 0.5
    c2 = total > 0
    c3 = total / max(1, n) > rnd_res[0] / max(1, rnd_res[1])
    say("")
    say("=" * 110)
    say("ВЕРДИКТ (объявлен до прогона): счёт без ликвидации и не ниже 500 $ [%s] · итог в плюсе [%s] · лучше "
        "случайного входа [%s] → %s"
        % ("да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет",
           "ПОДХОД «БЕЗ СТОПА» ГОДИТСЯ" if (c1 and c2 and c3) else "подход «без стопа» не годится"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

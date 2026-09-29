#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОМЕСЯЧНАЯ ПОДСТРОЙКА ИНДИКАТОРА ПОД МОНЕТУ (адаптация к текущему рынку).

Как это делал бы человек в TradingView: каждый месяц для каждой монеты выбирается лучшая
настройка Trand-Test-2 по последним 30 (вариант А) или 60 дням (вариант Б) и следующие 30 дней
торгуется этой настройкой. Потом снова подбор. С 2022 по 2026 год — около 55 месяцев.

Настройки, сигналы, сделки — ровно как в coin_wf_test.py (192 варианта: ТФ 30m/1h, старший
4ч/1Д, BTC 1ч/4ч/1Д, цель 3/4/5/6 × ATR, стоп 2/3/3.5/4 × ATR; вход по 15m после сигнала,
спорная свеча — стоп, комиссия 0.11%). Результат в R и в $ при риске 20 $ на сделку.
Для сравнения — одна общая настройка (30m, старший 1Д, BTC 1Д, цель 5, стоп 3.5).

ПРОГОН 2: два способа считать деньги.
  «Как в индикаторе» — фиксированная позиция 300 $ × плечо 10 = 3 000 $ на сделку (широкий стоп —
     больше потеря); подбор лучшей настройки по итогу в $ — как по таблице индикатора.
  «Фиксированный риск» — на стопе всегда 20 $ (как в прогоне 1).

КРИТЕРИЙ (объявлен ДО прогона), для каждого варианта и способа. Подстройка работает, если
  1) итог за всё время в плюсе и
  2) она лучше общей настройки в ОБЕИХ половинах периода (месяцы до середины и после).

Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py. Отчёт: adapt_report.txt
"""
import bisect
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw

REPORT = "adapt_report.txt"
BLOCK = 30 * c.D1
VARIANTS = [("30 дней", 1, 10), ("60 дней", 2, 15)]      # название, блоков на подбор, минимум сделок
NOTIONAL = 3000.0                                          # как в индикаторе: маржа 300 × плечо 10
MODES = [("КАК В ИНДИКАТОРЕ (позиция 300 × 10 = 3 000 $)", 2, NOTIONAL),
         ("ФИКСИРОВАННЫЙ РИСК 20 $ на стопе", 0, cw.RISK)]    # название, индекс в агрегате, множитель в $
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("ПОМЕСЯЧНАЯ ПОДСТРОЙКА ПОД МОНЕТУ · %s · монет %d · настроек %d · риск %.0f $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins),
           len(cw.TFS) * len(cw.HTFS) * len(cw.BTCS) * len(cw.EXITS), cw.RISK))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    b15b, h4b, d1b = data["BTC"]
    btc = {}
    for name, ms in cw.BTCS:
        bb = d1b if ms == c.D1 else c.agg(b15b, ms)
        btc[name] = ([x[0] for x in bb], c.trend_arr(bb), ms)

    def btc_tr(name, T):
        tt, tr, ms = btc[name]
        i = bisect.bisect_right(tt, T - ms) - 1
        return tr[i] if i >= 0 else 0

    agg = defaultdict(lambda: [0.0, 0, 0.0])   # (монета, tf, htf, btc, tp, sl, блок) -> [R, n, доля от позиции]
    max_block = 0
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
        h4a = c.agg(b15, c.H4)
        htf = {}
        for k, (bb, ms) in {"4ч": (h4a, c.H4), "1Д": (d1, c.D1)}.items():
            cc = [x[4] for x in bb]
            htf[k] = ([x[0] for x in bb], [a > b_ for a, b_ in zip(c.ema(cc, 9), c.ema(cc, 21))], ms)
        n_sig = 0
        for tfn, tfms in cw.TFS:
            b = c.agg(b15, tfms)
            atrp = c.atr_pct(b)
            for hk in cw.HTFS:
                ht, hu, hms = htf[hk]
                for T, d, i in cw.ind_entries_tf(b, tfms, ht, hu, hms):
                    if T < mt.START or atrp[i] <= 0:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    blk = (T - mt.START) // BLOCK
                    max_block = max(max_block, blk)
                    rs = cw.exits_R(o, h, l, cl, k0, d, atrp[i])
                    n_sig += 1
                    for bn, _ in cw.BTCS:
                        if btc_tr(bn, T) != d:
                            continue
                        for (tp, sl), r in zip(cw.EXITS, rs):
                            a = agg[(sym, tfn, hk, bn, tp, sl, blk)]
                            a[0] += r
                            a[1] += 1
                            a[2] += r * sl * atrp[i] / 100      # результат в долях позиции (R × ширина стопа)
        say("  %s: сигналов %d · %.0f с" % (sym, n_sig, time.time() - t0))

    settings = [(tfn, hk, bn, tp, sl) for tfn, _ in cw.TFS for hk in cw.HTFS for bn, _ in cw.BTCS
                for tp, sl in cw.EXITS]
    common = cw.COMMON
    n_blocks = max_block + 1
    mid_block = n_blocks // 2

    def report(mi, mult, vname, nb, min_n):
        """mi — что суммировать (0: R, 2: доля позиции), mult — перевод в $."""
        say("")
        say("=" * 118)
        say("ПОДБОР ПО ПОСЛЕДНИМ %s → торговля следующие 30 дней" % vname)
        per_block = []
        per_coin = defaultdict(lambda: [0.0, 0, 0.0, 0])
        for blk in range(nb, n_blocks):
            ra = na = rc = nc = 0.0
            for sym in coins:
                best, best_r = None, None
                for st in settings:
                    rr = nn = 0.0
                    for bb in range(blk - nb, blk):
                        v = agg.get((sym,) + st + (bb,))
                        if v:
                            rr += v[mi]
                            nn += v[1]
                    if nn >= min_n and (best_r is None or rr > best_r):
                        best, best_r = st, rr
                vc = agg.get((sym,) + common + (blk,))
                if vc:
                    rc += vc[mi]
                    nc += vc[1]
                    per_coin[sym][2] += vc[mi]
                if best is None:
                    continue
                v = agg.get((sym,) + best + (blk,))
                if v:
                    ra += v[mi]
                    na += v[1]
                    per_coin[sym][0] += v[mi]
            per_block.append((blk, ra, na, rc, nc))
        halves = ([b for b in per_block if b[0] < mid_block], [b for b in per_block if b[0] >= mid_block])
        tot_a, tot_c = sum(b[1] for b in per_block), sum(b[3] for b in per_block)
        n_a, n_c = sum(b[2] for b in per_block), sum(b[4] for b in per_block)
        say("  Месяцев проверки: %d · подстройка лучше общей в %d · в плюсе в %d"
            % (len(per_block), sum(1 for b in per_block if b[1] > b[3]), sum(1 for b in per_block if b[1] > 0)))
        say("  ИТОГО подстройка: %d сделок · %+.2f $/сделку · %+.0f $  |  общая: %d сделок · %+.2f $/сделку · %+.0f $"
            % (n_a, tot_a * mult / n_a if n_a else 0, tot_a * mult, n_c, tot_c * mult / n_c if n_c else 0, tot_c * mult))
        hs = []
        for k, hb in enumerate(halves):
            a, cc = sum(b[1] for b in hb), sum(b[3] for b in hb)
            hs.append(a > cc)
            say("  %d-я половина: подстройка %+.0f $ · общая %+.0f $" % (k + 1, a * mult, cc * mult))
        years = defaultdict(lambda: [0.0, 0.0])
        for blk, ra, na_, rc, nc_ in per_block:
            y = datetime.fromtimestamp((mt.START + blk * BLOCK) / 1000, tz=timezone.utc).year
            years[y][0] += ra
            years[y][1] += rc
        say("  По годам (подстройка / общая, $): " +
            " · ".join("%d: %+.0f / %+.0f" % (y, v[0] * mult, v[1] * mult) for y, v in sorted(years.items())))
        say("  По монетам (подстройка / общая, $): " +
            ", ".join("%s %+.0f/%+.0f" % (s_, v[0] * mult, v[2] * mult)
                      for s_, v in sorted(per_coin.items(), key=lambda kv: -kv[1][0])))
        ok = tot_a > 0 and all(hs)
        say("  ВЕРДИКТ (объявлен до прогона): итог в плюсе [%s] · лучше общей в обеих половинах [%s] → %s"
            % ("да" if tot_a > 0 else "нет", "да" if all(hs) else "нет",
               "ПОДСТРОЙКА РАБОТАЕТ" if ok else "подстройка не работает"))

    for mname, mi, mult in MODES:
        say("")
        say("#" * 118)
        say("СПОСОБ СЧЁТА: %s" % mname)
        for vname, nb, min_n in VARIANTS:
            report(mi, mult, vname, nb, min_n)
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

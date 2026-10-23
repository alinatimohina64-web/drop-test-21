#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ОСНОВНОЙ МЕТОД ИЗ ВИДЕО АВТОРА: Index inside % на 2h + глобальный режим рынка, фьючерсы лонг и шорт.
Из видео: «на медвежьем рынке ловим отскоки вверх до 90–95 — шорт; на бычьем — откаты вниз до 10 и ниже — лонг;
выход из лонга, когда индекс дошёл до 90–95». Глобальный режим у них — закрытый «стратегический» индикатор;
здесь он заменён проверенным режимом: ВЧЕРАШНИЙ дневной close BTC выше / ниже дневной SMA200.
Index % — наша реконструкция из 8 осцилляторов по их описанию (tt.index_pct), код оригинала закрыт.

ИСПОЛНЕНИЕ: решение по закрытию бара, сделка по открытию следующего; издержки и фандинг как в прошлых тестах.
  ВХОД — когда индекс ВПЕРВЫЕ заходит в зону (пересекает порог). Стоп проверяется по закрытию бара (выход по открытию
  следующего) — чуть хуже реального стоп-ордера. УСРЕДНЕНИЕ (доливки) НЕ моделируется.

ВАРИАНТЫ (объявлены до прогона, пороги 90/10 — из видео):
  V1  «как в видео»: BTC>SMA200 → лонг при входе индекса ≤ 10, выход при ≥ 90;
                      BTC<SMA200 → шорт при входе индекса ≥ 90, выход при ≤ 10; без стопа
  V2  V1 + стоп 3 × ATR(14) от входа
  V3  режим как V1, выход при возврате индекса к 50, стоп 3 × ATR
  V4  без режима: лонг при ≤ 10 и шорт при ≥ 90 всегда, выход при противоположной зоне, без стопа (справочно)
ТФ: 2h (главный — совет из видео), 1h и 4h — справочно.
МОНЕТЫ: все 62 из прошлых тестов — для ЭТОЙ гипотезы все они новые (входы по Index % мы не тестировали).

КОНТРОЛЬ «сдвиг»: те же позиции, циклически сдвинутые на 12 сроков (та же доля лонга/шорта и длина сделок).
КРИТЕРИЙ (объявлен ДО прогона): монета прошла, если PF >= 1.2, MAR > MAR B&H, обе половины (до/после 2022) > 0,
  лучше >= 10 из 12 сдвигов, сделок >= 20. МЕТОД ПОДТВЕРЖДЁН, если V1 или V2 на 2h прошли >= 2/3 монет
  И медианный PF >= 1.2 при ВСЕХ трёх порогах сетки (85/15, 90/10, 95/5) для этого варианта.
ПЛЕЧО — справочно: сделки V1/V2 2h с плечом x2…x100 на всю сумму, ликвидация при ходе против входа >= 1/L − 0.5%.
Отчёт: index_regime_report.txt
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

import trend_trail_test as tt

REPORT = "index_regime_report.txt"
SETS = [("первый", os.environ.get("S1", "BTC,ETH,BNB,XRP,ADA,SOL,DOGE,LTC,LINK,TRX,AVAX,DOT").split(",")),
        ("второй", os.environ.get("S2", "ETC,XLM,VET,BCH,ATOM,ALGO,ZEC,XTZ,NEAR,FIL,UNI,AAVE").split(",")),
        ("третий", os.environ.get("S3", "SHIB,HBAR,ICP,FET,INJ,GRT,STX,RUNE,EGLD,AXS,THETA,SAND,MANA,CHZ,KAVA,NEO,"
                                      "DASH,ZIL,QTUM,BAT,ICX,ENJ,CRV,COMP,SNX,1INCH,SUSHI,YFI,CAKE,ONE,IOTA,IOTX,"
                                      "ROSE,KSM,FLOW,GALA,LRC,ANKR").split(","))]
SETS = [(n, [x for x in c if x]) for n, c in SETS]
TFS = [("2h", 120), ("1h", 60), ("4h", 240)]
#        имя, режим, выход (opp/mid), стоп ATR, описание
VARS = [("V1", True, "opp", 0.0, "как в видео, без стопа"), ("V2", True, "opp", 3.0, "V1 + стоп 3 ATR"),
        ("V3", True, "mid", 3.0, "выход на 50 + стоп"), ("V4", False, "opp", 0.0, "без режима BTC")]
TH_MAIN = (90.0, 10.0)
GRID = [(85.0, 15.0), (90.0, 10.0), (95.0, 5.0)]
LEVS = [1, 2, 3, 5, 10, 20, 50, 100]
D1 = tt.D1
G = {}
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def btc_regime():
    base, _ = tt.load("BTC")
    d = tt.resample(base, 1440)
    c = d[:, 4]
    sma = np.r_[np.full(199, np.nan), np.convolve(c, np.ones(200) / 200, "valid")]
    with np.errstate(invalid="ignore"):
        up = c > sma
    return (d[:, 0] // D1).astype(np.int64), up


def regime_for(t):
    dk, up = G["btc"]
    want = (t // D1).astype(np.int64) - 1
    j = np.clip(np.searchsorted(dk, want, side="right") - 1, 0, len(dk) - 1)
    return up[j] & (dk[j] <= want)


def strat(idx, c, a, btc, use_reg, exit_mode, k, hi, lo):
    """Позиция после закрытия бара i: 1 лонг, -1 шорт, 0 вне рынка."""
    ix, cl, av, bt = idx.tolist(), c.tolist(), a.tolist(), btc.tolist()
    n = len(ix)
    p = [0.0] * n
    pos, stop = 0, 0.0
    for i in range(1, n):
        x, xp = ix[i], ix[i - 1]
        if pos == 1:
            if (x >= hi if exit_mode == "opp" else x >= 50) or (k and cl[i] < stop):
                pos = 0
        elif pos == -1:
            if (x <= lo if exit_mode == "opp" else x <= 50) or (k and cl[i] > stop):
                pos = 0
        if pos == 0 and av[i] == av[i]:
            okL = bt[i] if use_reg else True
            okS = (not bt[i]) if use_reg else True
            if okL and x <= lo < xp:
                pos, stop = 1, cl[i] - k * av[i]
            elif okS and x >= hi > xp:
                pos, stop = -1, cl[i] + k * av[i]
        p[i] = float(pos)
    return np.array(p)


def work(sym):
    t_s = time.time()
    try:
        base, _ = tt.load(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(base) < 20000:
        return sym, None, "мало данных (%d баров)" % len(base)
    t_first = max(tt.START, int(base[0, 0]) + tt.WARM_D * D1)
    res = {"tf": {}, "t0": t_first}
    for tfn, mins in TFS:
        a = tt.resample(base, mins)
        t, o, h, l, c, v = (a[:, k] for k in range(6))
        bar_h = mins / 60
        i0 = int(np.searchsorted(t, t_first))
        idx = tt.index_pct(o, h, l, c, v)
        av = tt.atr(h, l, c, 14)
        btc = regime_for(t)
        bh = tt.run(t, o, h, l, c, np.ones(len(c)), i0, bar_h, funding=False)
        R = {"bh": {k: bh[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2")}}
        for vn, use_reg, ex, k, _ in VARS:
            grid = GRID if (tfn == "2h" and vn in ("V1", "V2")) else [TH_MAIN]
            for hi, lo in grid:
                p = strat(idx, c, av, btc, use_reg, ex, k, hi, lo)
                r = tt.run(t, o, h, l, c, p, i0, bar_h)
                tr = tt.trades(t, o, h, l, r["eff"], r["retv"], bar_h)
                x = {q: r[q] for q in ("ret", "cagr", "mdd", "mar", "h1", "h2", "expo")}
                x.update(tt.trade_stats(tr))
                x["nshort"] = sum(1 for q in tr if q[1] < 0)
                if (hi, lo) == TH_MAIN:
                    pv = p[i0:]
                    sh = []
                    for s in range(1, tt.N_SHIFT + 1):
                        p2 = p.copy()
                        p2[i0:] = np.roll(pv, int(len(pv) * s / (tt.N_SHIFT + 1)))
                        sh.append(tt.run(t, o, h, l, c, p2, i0, bar_h)["mar"])
                    x["beat"] = int(sum(1 for s in sh if x["mar"] > s))
                    if tfn == "2h":
                        x["lev"] = {L: tt.lev_sim(tr, L) for L in LEVS}
                R[(vn, hi, lo)] = x
        res["tf"][tfn] = R
    return sym, res, "%.0f с, с %s" % (time.time() - t_s, tt.dstr(t_first))


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def main():
    t0 = time.time()
    G["btc"] = btc_regime()
    say("INDEX %% + РЕЖИМ BTC (метод из видео) · %s · фьючерсы лонг/шорт · издержки %.3f%% на сторону%s" % (
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), tt.FEE * 100, " · !!! СИНТЕТИКА" if tt.SYNTH else ""))
    allc = []
    for _, cl in SETS:
        allc += [x for x in cl if x not in allc]
    got = {}
    with mp.get_context("fork").Pool(max(1, min(len(allc), os.cpu_count() or 2))) as pool:
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    coins = [s for s in allc if s in got]
    need = int(np.ceil(len(coins) * 2 / 3))
    say("Монет загрузилось: %d · для принятия нужно >= %d" % (len(coins), need))

    say()
    say("=" * 118)
    say("ЧАСТЬ 1 — СВОДКА ПО ВСЕМ МОНЕТАМ (пороги 90/10). «по условиям»: 1 PF>=1.2 / 2 MAR>B&H / 3 обе половины>0 / "
        "4 сдвиги>=10 / 5 сделок>=20")
    say("  %-4s %-30s %7s %5s %5s %8s %6s %6s %7s %6s %6s %6s  %s" % ("ТФ", "вариант", "прошли", "PF", "WR", "CAGR",
        "MaxDD", "MAR", "MAR B&H", "сделок", "шортов", "в +", "по условиям"))
    npass_main = {}
    for tfn, _ in TFS:
        for vn, _, _, _, vl in VARS:
            X = [(got[s]["tf"][tfn][(vn,) + TH_MAIN], got[s]["tf"][tfn]["bh"]) for s in coins]
            cs = [tt.passed(x, b) for x, b in X]
            npass = sum(all(q) for q in cs)
            npass_main[(tfn, vn)] = npass
            say("  %-4s %s %-27s %3d/%-3d %5.2f %4.0f%% %+7.1f%% %5.1f%% %6.2f %7.2f %6.0f %6.0f %3d/%-3d %s" % (
                tfn, vn, vl, npass, len(coins), med([x["pf"] for x, _ in X]), 100 * med([x["wr"] for x, _ in X]),
                100 * med([x["cagr"] for x, _ in X]), 100 * med([x["mdd"] for x, _ in X]), med([x["mar"] for x, _ in X]),
                med([b["mar"] for _, b in X]), med([x["n"] for x, _ in X]), med([x["nshort"] for x, _ in X]),
                sum(x["ret"] > 0 for x, _ in X), len(coins), "/".join(str(sum(q[k] for q in cs)) for k in range(5))))
        say("  " + "-" * 112)

    say()
    say("=" * 118)
    say("ЧАСТЬ 2 — 2h ПО НАБОРАМ (медианы): PF · CAGR · MaxDD · в плюсе")
    for sn, cl in SETS:
        cc_ = [s for s in cl if s in got]
        if not cc_:
            continue
        cells = []
        for vn, *_ in VARS:
            X = [got[s]["tf"]["2h"][(vn,) + TH_MAIN] for s in cc_]
            cells.append("%s PF %.2f CAGR %+.1f%% DD %.0f%% +%d/%d" % (vn, med([x["pf"] for x in X]),
                         100 * med([x["cagr"] for x in X]), 100 * med([x["mdd"] for x in X]), sum(x["ret"] > 0 for x in X), len(X)))
        B = [got[s]["tf"]["2h"]["bh"] for s in cc_]
        say("  %-8s (%2d) B&H CAGR %+.1f%% DD %.0f%% │ %s" % (sn, len(cc_), 100 * med([b["cagr"] for b in B]),
                                                      100 * med([b["mdd"] for b in B]), " │ ".join(cells)))

    say()
    say("=" * 118)
    say("ЧАСТЬ 3 — УСТОЙЧИВОСТЬ К ПОРОГАМ, 2h: медианный PF · CAGR · сделок · прошли условия 1,2,3,5")
    gridok = {}
    for vn in ("V1", "V2"):
        cells = []
        ok = True
        for hi, lo in GRID:
            X = [(got[s]["tf"]["2h"][(vn, hi, lo)], got[s]["tf"]["2h"]["bh"]) for s in coins]
            pf = med([x["pf"] for x, _ in X])
            ok &= pf >= 1.2
            cells.append("%.0f/%.0f: PF %.2f CAGR %+.1f%% n %d прошли %d/%d" % (
                hi, lo, pf, 100 * med([x["cagr"] for x, _ in X]), med([x["n"] for x, _ in X]),
                sum(all(tt.passed(x, b, use_shift=False)) for x, b in X), len(coins)))
        gridok[vn] = ok
        say("  %s  " % vn + "  │  ".join(cells))

    say()
    say("=" * 118)
    say("ЧАСТЬ 4 — ПЛЕЧО, 2h (все деньги в каждой сделке): сколько монет обнулило счёт · медиана итога у выживших")
    for vn in ("V1", "V2"):
        cells = []
        for L in LEVS:
            lv = [got[s]["tf"]["2h"][(vn,) + TH_MAIN]["lev"][L] for s in coins]
            dead = sum(1 for e in lv if e[2] is not None)
            alive = [e[0] for e in lv if e[2] is None]
            cells.append("x%d: ликв %d/%d, итог %.2fx" % (L, dead, len(coins), med(alive) if alive else 0.0))
        say("  %s  " % vn + " │ ".join(cells))
    for s in ("BTC", "ETH", "SOL"):
        if s in got:
            lv = got[s]["tf"]["2h"][("V1",) + TH_MAIN]["lev"]
            say("  %-4s V1: " % s + " · ".join("x%d %s" % (L, ("ЛИКВ " + tt.dstr(lv[L][2])) if lv[L][2] else
                                                     "%.2fx dd%.0f%%" % (lv[L][0], 100 * lv[L][1])) for L in LEVS))

    say()
    say("=" * 118)
    say("ПО МОНЕТАМ, 2h (итог % · MaxDD % · PF · сделок): V1 | V2 | B&H")
    for s in coins:
        R = got[s]["tf"]["2h"]
        say("  %-6s " % s + " | ".join("%+7.0f %3.0f %4.2f %3d" % (100 * R[(v,) + TH_MAIN]["ret"], 100 * R[(v,) + TH_MAIN]["mdd"],
                                                              R[(v,) + TH_MAIN]["pf"], R[(v,) + TH_MAIN]["n"]) for v in ("V1", "V2"))
            + " | %+7.0f %3.0f" % (100 * R["bh"]["ret"], 100 * R["bh"]["mdd"]))

    say()
    say("=" * 118)
    for vn in ("V1", "V2"):
        a = npass_main[("2h", vn)] >= need
        say("→ %s 2h: прошли %d/%d (нужно %d) [%s] · PF >= 1.2 на всех порогах [%s] → %s" % (
            vn, npass_main[("2h", vn)], len(coins), need, "да" if a else "нет", "да" if gridok[vn] else "нет",
            "МЕТОД ПОДТВЕРЖДЁН" if a and gridok[vn] else "не подтверждён"))
    say("Время %.1f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

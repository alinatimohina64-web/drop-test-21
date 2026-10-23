#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА ВАРИАНТА C НА НОВЫХ ДАННЫХ (подтверждающий тест после trend_trail_test.py).
Вариант C = только лонг по пунктиру (SuperTrend ATR 10 x 3), вход лишь если close > EMA200 этого ТФ.
Параметры НЕ меняются — те же, что в первом тесте.

НАБОРЫ: «НОВЫЙ» — 12 монет, которых не было в первом тесте (главный); «СТАРЫЙ» — 12 монет первого теста (справочно).
ТФ: 4h (главный — лучший в первом тесте) и 2h (на нём торгуют на стриме).
ПЕРИОДЫ: «весь» (с 2018 или листинга, половины до/после 2022-01-01) и «с 2022» (половины 2022–2023 / с 2024-01-01).
Издержки, исполнение, фандинг — как в первом тесте (функции берутся из trend_trail_test.py).

ВАРИАНТЫ (объявлены до прогона):
  B  только лонг (справочно)
  C  лонг > EMA200 (проверяемый)
  G  C + Fear&Greed вчера < 50 (входить только при «страхе»)
  H  C + Fear&Greed вчера >= 50 (входить только при «жадности»)
  Fear & Greed — alternative.me, дневной, с 2018-02-01; берётся значение ПРОШЛОГО дня (без заглядывания вперёд).

КРИТЕРИЙ (объявлен ДО прогона), отдельно для 4h и 2h, на НОВОМ наборе:
  «весь»:  монета прошла, если PF >= 1.2, MAR > MAR B&H, обе половины > 0, лучше >= 10 из 12 сдвигов, сделок >= 20;
  «с 2022»: монета прошла, если PF >= 1.2, MAR > MAR B&H, обе половины > 0, сделок >= 10;
  C ПОДТВЕРЖДЁН, если в ОБОИХ периодах прошли >= 2/3 загрузившихся монет нового набора.
  Фильтр G/H ПОЛЕЗЕН, если он сам подтверждён И его медианный MAR выше, чем у C, в обоих периодах на обоих наборах.
Отчёт: c_confirm_report.txt
"""
import json
import multiprocessing as mp
import os
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

import trend_trail_test as tt

REPORT = "c_confirm_report.txt"
NEW = os.environ.get("NEW", "ETC,XLM,VET,BCH,ATOM,ALGO,ZEC,XTZ,NEAR,FIL,UNI,AAVE").split(",")
OLD = os.environ.get("OLD", "BTC,ETH,BNB,XRP,ADA,SOL,DOGE,LTC,LINK,TRX,AVAX,DOT").split(",")
TFS = [("4h", 240), ("2h", 120)]
VARS = [("B", "только лонг"), ("C", "лонг > EMA200"), ("G", "C + F&G < 50 (страх)"), ("H", "C + F&G >= 50 (жадность)")]
D1 = tt.D1
Y22 = tt.ms(2022, 1, 1)
Y24 = tt.ms(2024, 1, 1)
PERIODS = [("весь", None, Y22, 20, True), ("с 2022", Y22, Y24, 10, False)]
LEVS = [1, 2, 3, 5, 10]
G = {}
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def load_fng():
    if tt.SYNTH:
        rng = np.random.default_rng(7)
        days = np.arange(tt.ms(2018, 2, 1) // D1, tt.ms(2026, 10, 22) // D1)
        v = np.clip(50 + np.cumsum(rng.normal(0, 4, len(days))) % 60 - 30 + 20, 0, 100)
        return dict(zip(days.tolist(), v.tolist())), "синтетика"
    try:
        with urllib.request.urlopen("https://api.alternative.me/fng/?limit=0&format=json", timeout=60) as r:
            js = json.loads(r.read().decode())
        d = {int(x["timestamp"]) // 86400: float(x["value"]) for x in js["data"]}
        return d, "%d дней, %s → %s" % (len(d), tt.dstr(min(d) * D1), tt.dstr(max(d) * D1))
    except Exception as e:                                       # noqa: BLE001
        return {}, "НЕ ЗАГРУЗИЛСЯ: %s" % e


def run_p(t, o, h, l, c, p, i0, half, bar_h, funding=True):
    """Как tt.run, но с заданной границей половин."""
    p = p.copy()
    p[:i0] = 0
    eff = np.r_[0.0, p[:-1]]
    ret = np.r_[o[1:] / o[:-1] - 1, c[-1] / o[-1] - 1]
    chg = np.abs(np.diff(np.r_[0.0, eff]))
    fund = eff * tt.FUND8 * bar_h / 8 if funding else 0.0
    f = 1 + eff * ret - chg * tt.FEE - fund
    worst = np.minimum(eff * (l / o - 1), eff * (h / o - 1))
    fw = (1 + worst - chg * tt.FEE - fund)[i0:]
    E = np.cumprod(f[i0:])
    Ep = np.r_[1.0, E[:-1]]
    pk = np.maximum.accumulate(np.r_[1.0, E])[:-1]
    mdd = float(np.max(1 - np.minimum(Ep * fw, E) / pk))
    ih = max(i0, int(np.searchsorted(t, half)))
    yrs = (t[-1] - t[i0]) / (365.25 * D1)
    fin = float(E[-1])
    cagr = fin ** (1 / yrs) - 1 if fin > 0 else -1.0
    return {"ret": fin - 1, "cagr": cagr, "mdd": mdd, "mar": cagr / max(mdd, 1e-9),
            "h1": float(np.prod(f[i0:ih])) - 1, "h2": float(np.prod(f[ih:])) - 1, "eff": eff, "retv": ret}


def work(sym):
    t_s = time.time()
    try:
        base, fails = tt.load(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(base) < 20000:
        return sym, None, "мало данных (%d баров)" % len(base)
    t_first = max(tt.START, int(base[0, 0]) + tt.WARM_D * D1)
    fng = G["fng"]
    res = {"t0": t_first, "t1": int(base[-1, 0]), "tf": {}}
    for tfn, mins in TFS:
        a = tt.resample(base, mins)
        t, o, h, l, c = (a[:, k] for k in range(5))
        bar_h = mins / 60
        above = c > tt.ema(c, 200)
        fv = np.array([fng.get(int(x // D1) - 1, np.nan) for x in t])
        F = np.zeros(len(c), bool)
        filt = {"B": np.ones(len(c), bool), "C": above, "G": above & (fv < 50), "H": above & (fv >= 50)}
        d = tt.supertrend_dir(h, l, c, *tt.MAIN)
        R = {}
        for pn, p0, half, _, shift in PERIODS:
            i0 = int(np.searchsorted(t, max(t_first, p0 or 0)))
            bh = run_p(t, o, h, l, c, np.ones(len(c)), i0, half, bar_h, funding=False)
            R[(pn, "bh")] = {k: bh[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2")}
            for vn, _ in VARS:
                if vn in "GH" and not fng:
                    continue
                p = tt.positions(d, filt[vn], F)
                r = run_p(t, o, h, l, c, p, i0, half, bar_h)
                tr = [q for q in tt.trades(t, o, h, l, r["eff"], r["retv"], bar_h)]
                x = {k: r[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2")}
                x.update(tt.trade_stats(tr))
                if shift:
                    pv = p[i0:]
                    sh = []
                    for k in range(1, tt.N_SHIFT + 1):
                        p2 = p.copy()
                        p2[i0:] = np.roll(pv, int(len(pv) * k / (tt.N_SHIFT + 1)))
                        sh.append(run_p(t, o, h, l, c, p2, i0, half, bar_h)["mar"])
                    x["beat"] = int(sum(1 for s in sh if x["mar"] > s))
                if vn == "C":
                    x["lev"] = {L: tt.lev_sim(tr, L) for L in LEVS}
                R[(pn, vn)] = x
        res["tf"][tfn] = R
    return sym, res, "%.0f с, %d баров 15m, с %s" % (time.time() - t_s, len(base), tt.dstr(int(base[0, 0])))


def passed(x, bh, min_n, shift):
    return [x["pf"] >= 1.2, x["mar"] > bh["mar"], x["h1"] > 0 and x["h2"] > 0,
            (x.get("beat", 0) >= 10) if shift else True, x["n"] >= min_n]


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def main():
    t0 = time.time()
    G["fng"], fmsg = load_fng()
    say("ПРОВЕРКА ВАРИАНТА C · %s · SuperTrend ATR %d × %.1f · издержки %.3f%% на сторону%s" % (
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), tt.MAIN[0], tt.MAIN[1], tt.FEE * 100,
        " · !!! СИНТЕТИКА — цифры ничего не значат" if tt.SYNTH else ""))
    say("Fear & Greed: " + fmsg)
    allc = NEW + [x for x in OLD if x not in NEW]
    got = {}
    with mp.get_context("fork").Pool(max(1, min(len(allc), os.cpu_count() or 2))) as pool:
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    sets = [("НОВЫЙ (главный)", [s for s in NEW if s in got]), ("СТАРЫЙ (справочно)", [s for s in OLD if s in got])]
    vlist = [v for v in VARS if v[0] in "BC" or G["fng"]]
    verdict = {}
    medmar = {}

    say()
    say("=" * 118)
    say("ЧАСТЬ 1 — СВОДКА (медианы по монетам). «по условиям»: 1 PF>=1.2 / 2 MAR>B&H / 3 обе половины>0 / 4 сдвиги>=10 / 5 сделок")
    for sn, coins in sets:
        if not coins:
            continue
        need = int(np.ceil(len(coins) * 2 / 3))
        say("  НАБОР %s: %d монет, нужно >= %d" % (sn, len(coins), need))
        say("  %-4s %-8s %-24s %7s %5s %5s %8s %6s %6s %7s %6s %8s  %s" % ("ТФ", "период", "вариант", "прошли", "PF", "WR",
            "CAGR", "MaxDD", "MAR", "MAR B&H", "сделок", "половины", "по условиям"))
        for tfn, _ in TFS:
            for pn, _, _, min_n, shift in PERIODS:
                for vn, vl in vlist:
                    X = [(got[s]["tf"][tfn][(pn, vn)], got[s]["tf"][tfn][(pn, "bh")]) for s in coins]
                    cs = [passed(x, b, min_n, shift) for x, b in X]
                    npass = sum(all(q) for q in cs)
                    ok = npass >= need
                    verdict[(sn, tfn, pn, vn)] = ok
                    medmar[(sn, tfn, pn, vn)] = med([x["mar"] for x, _ in X])
                    say("  %-4s %-8s %s %-22s %3d/%-3d %5.2f %4.0f%% %+7.1f%% %5.1f%% %6.2f %7.2f %6.0f %3d/%-3d  %s%s" % (
                        tfn, pn, vn, vl, npass, len(coins), med([x["pf"] for x, _ in X]), 100 * med([x["wr"] for x, _ in X]),
                        100 * med([x["cagr"] for x, _ in X]), 100 * med([x["mdd"] for x, _ in X]), medmar[(sn, tfn, pn, vn)],
                        med([b["mar"] for _, b in X]), med([x["n"] for x, _ in X]), sum(q[2] for q in cs), len(coins),
                        "/".join(str(sum(q[k] for q in cs)) if (k != 3 or shift) else "-" for k in range(5)),
                        "  ← прошло" if ok else ""))
                say("  " + "-" * 112)

    say()
    say("=" * 118)
    say("ЧАСТЬ 2 — ПО МОНЕТАМ, вариант C (итог % · MaxDD % · PF · сделок) и Buy&Hold (итог % · MaxDD %)")
    for sn, coins in sets:
        say("  НАБОР %s" % sn)
        for s in coins:
            cells = []
            for tfn, _ in TFS:
                for pn, *_ in PERIODS:
                    x = got[s]["tf"][tfn][(pn, "C")]
                    b = got[s]["tf"][tfn][(pn, "bh")]
                    cells.append("%s %s: C %+6.0f %3.0f %4.2f %3d | B&H %+6.0f %3.0f" % (
                        tfn, pn, 100 * x["ret"], 100 * x["mdd"], x["pf"], x["n"], 100 * b["ret"], 100 * b["mdd"]))
            say("  %-5s " % s + "  ‖  ".join(cells))

    say()
    say("=" * 118)
    say("ЧАСТЬ 3 — ПЛЕЧО, вариант C 4h, весь период (все деньги в каждой сделке). «2.6x dd43%» — вырос в 2.6 раза, просадка 43%")
    for sn, coins in sets:
        nl = {L: 0 for L in LEVS}
        for s in coins:
            lv = got[s]["tf"]["4h"][("весь", "C")]["lev"]
            cells = []
            for L in LEVS:
                eq, dd, rt = lv[L]
                nl[L] += rt is not None
                cells.append("%16s" % (("ЛИКВ " + tt.dstr(rt)) if rt else "%.2fx dd%.0f%%" % (eq, 100 * dd)))
            say("  %-5s " % s + " ".join(cells) + "   (x" + "/x".join(str(L) for L in LEVS) + ")")
        say("  набор %s — ликвидировано монет: " % sn + " · ".join("x%d: %d/%d" % (L, nl[L], len(coins)) for L in LEVS))

    say()
    say("=" * 118)
    sn0 = sets[0][0]
    for tfn, _ in TFS:
        okC = verdict.get((sn0, tfn, "весь", "C")) and verdict.get((sn0, tfn, "с 2022", "C"))
        say("→ %s · вариант C: весь период [%s] · с 2022 [%s] → %s" % (
            tfn, "да" if verdict.get((sn0, tfn, "весь", "C")) else "нет",
            "да" if verdict.get((sn0, tfn, "с 2022", "C")) else "нет", "ПОДТВЕРЖДЁН" if okC else "НЕ подтверждён"))
        for vn in ("G", "H"):
            if not G["fng"]:
                continue
            own = verdict.get((sn0, tfn, "весь", vn)) and verdict.get((sn0, tfn, "с 2022", vn))
            better = all(medmar.get((s, tfn, p, vn), -9) > medmar.get((s, tfn, p, "C"), 9)
                         for s, _ in sets for p, *_ in PERIODS if (s, tfn, p, vn) in medmar)
            say("  фильтр %s: сам подтверждён [%s] · MAR выше C везде [%s] → %s" % (
                vn, "да" if own else "нет", "да" if better else "нет", "ПОЛЕЗЕН" if own and better else "не доказан"))
    say("Время %.1f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

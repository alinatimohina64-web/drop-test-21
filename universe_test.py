#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
РАСШИРЕНИЕ СПИСКА МОНЕТ (топ-50 → топ-100 / топ-150 / весь набор по обороту) ДЛЯ СИГНАЛОВ ⭐ — проверка на истории.

Правило — как у робота (проверенный набор), без изменений: сигнал TT2 (30m, старший ТФ 4ч, по тренду BTC 1Д),
только когда BTC вчера закрылся выше SMA200, без MRC, цель 5 / стоп 20 × ATR (стоп не дальше −80%),
только ⭐-сигналы (стоп 20 × ATR не ближе 30%). Риск 50 $ на сделку, до 5 позиций, одна на монету,
BTC не торгуем. Вход по открытию 15m после сигнала, в спорной свече — стоп, комиссия 0.11% за круг,
фандинг 0.01% за 8 ч (движок cs.trade — тот же, что в остальных тестах).

ЧТО МЕНЯЕМ: только список монет. РАНГ монеты = место по среднему дневному обороту за 30 закрытых дней
(доллары = объём × цена, ранг среди монет набора, BTC в рейтинге участвует, но не торгуется — как в роботе)
на момент сигнала, то есть без заглядывания вперёд. Сравниваем: топ-50 (как сейчас) / 100 / 150 / весь набор.

ИЗДЕРЖКИ НА ТОНКИХ МОНЕТАХ: для монет с рангом > 50 дополнительно вычитается 0.2% за круг (проскальзывание
и спред), в R это 0.002 / стоп%.

КРИТЕРИЙ (объявлен ДО прогона). Расширение до топ-N (N = 100, 150, весь набор) принимается, если ВСЕ:
  0) контроль: группа 1–50 сама в плюсе (средний R > 0) и лучше случайного входа — иначе тест несравним;
  1) новые монеты (ранг 51…N), одна позиция на монету: сделок не меньше 50 и средний R с издержками > 0;
  2) их средний R (без издержек) больше среднего R случайного входа в тех же рангах, при том же режиме и ⭐;
  3) средний R > 0 в обеих половинах периода (в каждой не меньше 15 сделок) и в обеих непересекающихся
     группах монет (чётные/нечётные по алфавиту, не меньше 15 сделок в каждой);
  4) портфель (5 слотов, медиана по 100 случайным порядкам равных сигналов, с издержками) на топ-N даёт больше
     суммарного R, чем на топ-50, а максимальная просадка в R не больше чем в 1.2 раза выше.
ОГРАНИЧЕНИЯ (честно): набор — монеты, которые есть на Binance спот; ранг считается внутри набора, а не по всему
рынку; монеты, которых давно нет, в набор попали только если у них есть исторические файлы; новые монеты без
250 дневок (как VAL) правилом не торгуются вовсе. Контрольный прогон на синтетических случайных ценах не
добавлен: движок не менялся, базой служит случайный вход на реальных свечах.

Запуск: python universe_test.py. Первый прогон долгий (скачивание истории ~100 монет), дальше — из кэша data_btc/.
Переменные: START_YEAR (2019), RND_N (300), COINS_LIMIT (для быстрой пробы). Отчёт: universe_report.txt
"""
import bisect
import os
import random
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np

import sma200_test as st

cs, c, cw = st.cs, st.c, st.cw
REPORT = "universe_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
TP_X, SL_X, SL_CAP, STAR = 5.0, 20.0, 0.80, 0.30
SLOTS, RISK = 5, 50.0
EXTRA_RT = 0.002
RND_N = int(os.environ.get("RND_N") or 300)
START = int(datetime(int(os.environ.get("START_YEAR") or 2019), 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
BANDS = [(1, 50), (51, 100), (101, 150), (151, 10 ** 6)]
CANDS = [100, 150, 10 ** 6]
SCEN = 100

EXTRA = ["MATIC", "FTM", "EOS", "XTZ", "VET", "THETA", "EGLD", "FLOW", "KAVA", "RUNE", "CHZ", "ENJ", "GRT", "SNX",
         "COMP", "MKR", "YFI", "SUSHI", "1INCH", "BAT", "ZEC", "DASH", "XMR", "NEO", "IOTA", "ZIL", "QTUM", "ICX",
         "ONT", "KSM", "CELO", "LRC", "ANKR", "HOT", "LDO", "APE", "GMX", "MINA", "RNDR", "ENS", "PEPE", "FLOKI",
         "BONK", "WIF", "JUP", "PYTH", "DYDX", "SSV", "BLUR", "JTO", "STRK", "ORDI", "ETHFI", "W", "ENA", "AR",
         "ROSE", "CFX", "PENDLE", "RPL", "WOO", "CKB", "IOTX", "GMT", "MASK", "API3", "CVX", "BAL", "ZRX", "STORJ",
         "SKL", "COTI", "JASMY", "AUDIO", "BAND", "CTSI", "DENT", "FXS", "GLM", "ID", "LQTY", "MAGIC", "OCEAN",
         "AGIX", "ACH", "AMP", "ARPA", "CELR", "CHR", "DGB", "DUSK", "FET", "FUN", "HIVE", "KNC", "LINA", "LSK",
         "NKN", "OGN", "PERP", "REEF", "RSR", "SXP", "TLM", "TRB", "UMA", "WAVES", "XEM", "YGG"]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def med(x):
    x = sorted(x)
    return x[len(x) // 2] if x else float("nan")


def has_data(sym):
    for tag in ("2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"):
        url = "https://data.binance.vision/data/spot/monthly/klines/%sUSDT/15m/%sUSDT-15m-%s.zip" % (sym, sym, tag)
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20):
                return True
        except Exception:                                       # noqa: BLE001
            continue
    return False


def prep(sym, btc):
    b15 = cs.load_15m(sym)
    if len(b15) < 20000:
        return None
    d1 = c.agg(b15, D1)
    dt = [x[0] for x in d1]
    dv = [x[5] * x[4] for x in d1]
    vol30, s = {}, 0.0
    for i, v in enumerate(dv):
        s += v
        if i >= 30:
            s -= dv[i - 30]
        if i >= 29:
            vol30[dt[i]] = s / 30
    out = {"vol30": vol30, "sig": [], "rnd": [], "end": int(b15[-1][0])}
    if sym == "BTC":
        return out
    t15 = np.array([x[0] for x in b15], dtype=np.int64)
    o = np.array([x[1] for x in b15])
    h = np.array([x[2] for x in b15])
    l = np.array([x[3] for x in b15])
    cl = np.array([x[4] for x in b15])
    b30 = c.agg(b15, M30)
    b30t = [x[0] for x in b30]
    atrp = c.atr_pct(b30)
    h4a = c.agg(b15, H4)
    c4 = [x[4] for x in h4a]
    hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]

    def one(T, d, a):
        slf = min(SL_X * a, SL_CAP)
        jd = bisect.bisect_right(dt, T - D1) - 1
        if jd < 250 or SL_X * a < STAR:
            return None
        k0 = int(np.searchsorted(t15, T, side="left"))
        if k0 >= len(t15) - 1:
            return None
        pnl, r, te = cs.trade(t15, h, l, cl, k0, d, float(o[k0]), TP_X * a, slf)
        return (T, d, r, te, slf, dt[jd])

    for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
        if T < START or atrp[i] <= 0 or btc.trend(T) != d or not btc.above(T):
            continue
        res = one(T, d, atrp[i] / 100)
        if res:
            out["sig"].append(res)
    y0 = max(START, int(t15[0]) + 260 * D1)
    y1 = int(t15[-1]) - 5 * D1
    rg = random.Random("rnd-" + sym)
    tries = 0
    while len(out["rnd"]) < RND_N and tries < RND_N * 30 and y1 > y0:
        tries += 1
        T = rg.randrange(y0, y1)
        d = btc.trend(T)
        if d == 0 or not btc.above(T):
            continue
        j = bisect.bisect_right(b30t, T - M30) - 1
        if j < 20 or atrp[j] <= 0:
            continue
        res = one(T, d, atrp[j] / 100)
        if res:
            out["rnd"].append(res)
    return out


def adj(s, extra):
    return s["r"] - (EXTRA_RT / s["slf"] if extra and s["rank"] > 50 else 0.0)


def avg(v, extra=False):
    return sum(adj(s, extra) for s in v) / len(v) if v else float("nan")


def fmt(x):
    return "—" if x != x else "%+.3f" % x


def one_per_coin(lst):
    by = {}
    for s in sorted(lst, key=lambda s: s["T"]):
        by.setdefault(s["sym"], []).append(s)
    out = []
    for v in by.values():
        busy = 0
        for s in v:
            if s["T"] < busy:
                continue
            out.append(s)
            busy = s["te"] if s["te"] is not None else 10 ** 15
    return out


def portfolio(sigs, nmax, seed):
    el = [s for s in sigs if s["rank"] <= nmax]
    rg = random.Random(seed)
    el.sort(key=lambda s: (s["T"], rg.random()))
    open_, taken = [], []
    for s in el:
        open_ = [x for x in open_ if x[0] is None or x[0] > s["T"]]
        if len(open_) >= SLOTS or any(x[1] == s["sym"] for x in open_):
            continue
        open_.append((s["te"], s["sym"]))
        taken.append(s)
    return taken, len(el)


def equity(taken, end_t, extra):
    ev = sorted(((s["te"] if s["te"] is not None else end_t), adj(s, extra)) for s in taken)
    cum = peak = dd = 0.0
    for _, r in ev:
        cum += r
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return cum, dd


def band_halves(v, mid, extra=False):
    a = [s for s in v if s["T"] < mid]
    b = [s for s in v if s["T"] >= mid]
    return a, b


def groups(v):
    coins = sorted({s["sym"] for s in v})
    ga = set(coins[::2])
    return [s for s in v if s["sym"] in ga], [s for s in v if s["sym"] not in ga]


def main():
    t0 = time.time()
    say("РАСШИРЕНИЕ СПИСКА МОНЕТ ДЛЯ ⭐-СИГНАЛОВ · %s · Binance 15m · риск %.0f $, слотов %d"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), RISK, SLOTS))
    btc = st.Btc()
    coins = [s for s in dict.fromkeys(list(st.MAIN_COINS) + list(st.NEW_COINS) + EXTRA) if s != "BTC"]
    lim = int(os.environ.get("COINS_LIMIT") or 0)
    if lim:
        coins = coins[:lim]
    P = {}
    for sym in ["BTC"] + coins:
        if sym != "BTC" and not has_data(sym):
            say("  %s: нет на Binance спот — пропуск" % sym)
            continue
        p = prep(sym, btc)
        if p:
            P[sym] = p
            say("  %s: ⭐-сигналов %d, случайных %d · %.0f с" % (sym, len(p["sig"]), len(p["rnd"]), time.time() - t0))
    end_t = max(p["end"] for p in P.values())

    byday = {}
    for sym, p in P.items():
        for dtm, v in p["vol30"].items():
            byday.setdefault(dtm, []).append((v, sym))
    rank = {}
    for dtm, lst in byday.items():
        lst.sort(reverse=True)
        for pos, (_, sym) in enumerate(lst):
            rank[(sym, dtm)] = pos + 1

    sig, rnd = [], []
    for sym, p in P.items():
        for kind, dst in (("sig", sig), ("rnd", rnd)):
            for T, d, r, te, slf, dk in p[kind]:
                rk = rank.get((sym, dk))
                if rk is not None:
                    dst.append({"sym": sym, "T": T, "d": d, "r": r, "te": te, "slf": slf, "rank": rk})
    if not sig:
        say("Нет сигналов.")
        return
    mid = (min(s["T"] for s in sig) + end_t) // 2
    sig1 = one_per_coin(sig)
    say("")
    say("Монет в наборе на начало года (с данными, считая BTC): " + " · ".join(
        "%d: %d" % (y, len(byday.get(int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), [])))
        for y in range(2019, datetime.now(timezone.utc).year + 1)))
    say("Всего ⭐-сигналов %d, из них по одной позиции на монету %d; граница половин периода %s"
        % (len(sig), len(sig1), datetime.fromtimestamp(mid / 1000, tz=timezone.utc).strftime("%Y-%m-%d")))

    say("")
    say("=" * 110)
    say("ПО ГРУППАМ ПО ОБОРОТУ (одна позиция на монету, без лимита слотов; R = результат в долях риска)")
    for lo, hi in BANDS:
        v = [s for s in sig1 if lo <= s["rank"] <= hi]
        rv = [s for s in rnd if lo <= s["rank"] <= hi]
        name = "ранг %d–%s" % (lo, hi if hi < 10 ** 6 else "…")
        if not v:
            say("  %-14s сделок 0" % name)
            continue
        a, b = band_halves(v, mid)
        ga, gb = groups(v)
        yrs = {}
        for s in v:
            yrs.setdefault(datetime.fromtimestamp(s["T"] / 1000, tz=timezone.utc).year, []).append(s)
        say("  %-14s сделок %4d · WR %3.0f%% · средний R %s (с издержками %s) · итог %+.1f R ≈ %+.0f $ · случайный вход %s (%d)"
            % (name, len(v), 100 * sum(1 for s in v if s["r"] > 0) / len(v), fmt(avg(v)), fmt(avg(v, True)),
               sum(adj(s, True) for s in v), sum(adj(s, True) for s in v) * RISK, fmt(avg(rv)), len(rv)))
        say("      половины: %s (%d) / %s (%d) · группы монет: %s (%d) / %s (%d) · по годам: %s" % (
            fmt(avg(a, True)), len(a), fmt(avg(b, True)), len(b), fmt(avg(ga, True)), len(ga), fmt(avg(gb, True)), len(gb),
            " · ".join("%d: %s (%d)" % (y, fmt(avg(s_, True)), len(s_)) for y, s_ in sorted(yrs.items()))))

    say("")
    say("=" * 110)
    say("ПОРТФЕЛЬ: %d слотов, одна позиция на монету, равные сигналы в один момент — случайный порядок (%d прогонов)"
        % (SLOTS, SCEN))
    res = {}
    for N in [50] + CANDS:
        runs = []
        for sd in range(SCEN):
            tk, n_el = portfolio(sig, N, sd)
            runs.append((equity(tk, end_t, True), equity(tk, end_t, False), len(tk), n_el))
        res[N] = {"tot_x": med([r[0][0] for r in runs]), "dd_x": med([r[0][1] for r in runs]),
                  "tot": med([r[1][0] for r in runs]), "dd": med([r[1][1] for r in runs]),
                  "n": med([r[2] for r in runs]), "elig": runs[0][3]}
        tk0, _ = portfolio(sig, N, 0)
        yrs = {}
        for s in tk0:
            yrs.setdefault(datetime.fromtimestamp(s["T"] / 1000, tz=timezone.utc).year, []).append(s)
        label = "топ-%d" % N if N < 10 ** 6 else "весь набор"
        say("  %-11s сигналов %4d → взято %4d · итог с издержками %+7.1f R ≈ %+6.0f $ (без них %+.1f R) · "
            "просадка %+.1f R ≈ %+.0f $ · по годам: %s" % (
                label, res[N]["elig"], res[N]["n"], res[N]["tot_x"], res[N]["tot_x"] * RISK, res[N]["tot"],
                res[N]["dd_x"], res[N]["dd_x"] * RISK,
                " · ".join("%d: %+.1f R (%d)" % (y, sum(adj(s, True) for s in v), len(v)) for y, v in sorted(yrs.items()))))

    say("")
    say("=" * 110)
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона)")
    base = [s for s in sig1 if s["rank"] <= 50]
    base_r = [s for s in rnd if s["rank"] <= 50]
    k0 = len(base) > 0 and avg(base) > 0 and avg(base) > avg(base_r)
    say("  0) контроль, группа 1–50: средний R %s, случайный %s → %s" % (
        fmt(avg(base)), fmt(avg(base_r)), "ОК" if k0 else "не выполнен — сравнение ненадёжно"))
    for N in CANDS:
        new = [s for s in sig1 if 50 < s["rank"] <= N]
        newr = [s for s in rnd if 50 < s["rank"] <= N]
        a, b = band_halves(new, mid)
        ga, gb = groups(new)
        k1 = len(new) >= 50 and avg(new, True) > 0
        k2 = avg(new) > avg(newr)
        k3 = (len(a) >= 15 and len(b) >= 15 and avg(a, True) > 0 and avg(b, True) > 0 and
              len(ga) >= 15 and len(gb) >= 15 and avg(ga, True) > 0 and avg(gb, True) > 0)
        k4 = res[N]["tot_x"] > res[50]["tot_x"] and -res[N]["dd_x"] <= 1.2 * -res[50]["dd_x"]
        ok = k1 and k2 and k3 and k4
        label = "топ-%d" % N if N < 10 ** 6 else "весь набор"
        say("  %-11s 1) сделок %d, R с издержками %s [%s] · 2) против случайного %s / %s [%s] · 3) половины/группы [%s] · "
            "4) портфель %+.1f R против %+.1f R, просадка %+.1f против %+.1f [%s] → %s" % (
                label, len(new), fmt(avg(new, True)), "да" if k1 else "нет", fmt(avg(new)), fmt(avg(newr)),
                "да" if k2 else "нет", "да" if k3 else "нет", res[N]["tot_x"], res[50]["tot_x"], res[N]["dd_x"],
                res[50]["dd_x"], "да" if k4 else "нет",
                "РАСШИРЯТЬ" if (ok and k0) else "не расширять"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

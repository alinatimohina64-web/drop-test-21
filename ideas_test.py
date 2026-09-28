#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ ИДЕЙ для ручной торговли по индикатору (1h + фильтр BTC 1Д). Лежит рядом с
compare_test.py и stop_test.py — данные, входы и сценарии берутся оттуда.

БАЗА (A0): сигналы Trand-Test-2 на 1h, старший ТФ индикатора 4ч, только по дневному
тренду BTC; один вход, цель 2.5 × ATR 1ч, стоп 5.2 × ATR 1ч, риск на стопе 20 $.

ВАРИАНТЫ (каждый меняет одну вещь относительно базы):
  A1 — старший ТФ индикатора 1Д вместо 4ч (EMA9 > EMA21 на дневке монеты).
  B1 — только если корреляция часовых доходностей монеты с BTC за прошлые 30 дней >= 0.6
       (считается только по прошлому). Для справки — разбивка по корреляции.
  C1 — пропустить сигнал, если сделки этой монеты, закрытые за прошлые 60 дней
       (не меньше 10), в среднем убыточны. Меньше 10 сделок — сигнал берём.
  D1 — стоп 6.5 × ATR (цель та же 2.5 × ATR).
  D2 — стоп 8.0 × ATR.
  D3 — трендовый выход: стоп 3 × ATR, цели нет, трейлинг — выход, когда цена отошла
       на 3 × ATR (ATR на входе) от лучшей цены после входа.
  Риск на стопе всегда 20 $: позиция = 20 $ / ширина стопа.

КРИТЕРИЙ (объявлен ДО прогона), для каждого варианта отдельно — лучше базы, если в
сценариях «одна сделка за раз, 30% пропущено»:
  1) медиана итога выше в ОБЕИХ половинах года;
  2) медианная просадка не больше, чем у базы.
Отчёт: ideas_report.txt
"""
import bisect
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c
import stop_test as st

REPORT = "ideas_report.txt"
RISK = st.RISK
CORR_WIN = 720             # часов = 30 дней
CORR_MIN = 0.6
PAST_WIN = 60 * c.D1
PAST_MIN_N = 10
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def ind_entries_htf(b, htf_t, htf_up, htf_ms):
    """То же, что compare_test.ind_entries, но старший ТФ задаётся (4ч или 1Д)."""
    n = len(b)
    cl = [x[4] for x in b]
    ef, es = c.ema(cl, 9), c.ema(cl, 21)
    r = c.rsi(cl)
    av = c.sma([x[5] for x in b], 20)
    adx = c.adx_arr(b)
    out = []
    last_up = last_dn = last_sig = last_pb = None
    j = -1
    for i in range(1, n):
        if ef[i] > es[i] and ef[i - 1] <= es[i - 1]:
            last_up = i
        if ef[i] < es[i] and ef[i - 1] >= es[i - 1]:
            last_dn = i
        if i < 60 or av[i] is None:
            continue
        close_t = b[i][0] + c.H1
        while j + 1 < len(htf_t) and htf_t[j + 1] + htf_ms <= close_t:
            j += 1
        if j < 25:
            continue
        up = htf_up[j]
        bs_up = None if last_up is None else i - last_up
        bs_dn = None if last_dn is None else i - last_dn
        conf_up = bs_up == 2 and (bs_dn is None or bs_dn > 2)
        conf_dn = bs_dn == 2 and (bs_up is None or bs_up > 2)
        adx_ok = adx[i] >= 20
        vol_ok = b[i][5] > av[i] * 1.3
        long_raw = conf_up and r[i] > 50 and r[i] > r[i - 1] and vol_ok and adx_ok and up
        short_raw = conf_dn and r[i] < 50 and r[i] < r[i - 1] and vol_ok and adx_ok and not up
        cd_ok = last_sig is None or i - last_sig >= 5
        ls, ss = long_raw and cd_ok, short_raw and cd_ok
        if ls or ss:
            last_sig = i
            out.append((close_t, 1 if ls else -1, i))
        o, h, l = b[i][1], b[i][2], b[i][3]
        pl = (ef[i] > es[i] and adx_ok and up and l <= ef[i] and cl[i] > ef[i] and cl[i] > o and r[i] > r[i - 1])
        ps = (ef[i] < es[i] and adx_ok and not up and h >= ef[i] and cl[i] < ef[i] and cl[i] < o and r[i] < r[i - 1])
        pb_ok = last_pb is None or i - last_pb >= 10
        pl, ps = pl and pb_ok and not ls, ps and pb_ok and not ss
        if pl or ps:
            last_pb = i
            out.append((close_t, 1 if pl else -1, i))
    return out


def sim_trail(b5, k0, d, atrp, mult=3.0):
    """Трендовый выход: стоп и трейлинг mult × ATR от лучшей цены, без цели."""
    dist = mult * atrp / 100
    notional = RISK / dist
    p0 = b5[k0][1]
    qty = notional / p0
    best = p0
    fund = 0.0
    for k in range(k0, len(b5)):
        t, o, h, l, cl = b5[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        stop = best * (1 - d * dist)
        if d * o <= d * stop:
            px = o
        elif d * adv <= d * stop:
            px = stop
        else:
            if d * fav > d * best:
                best = fav
            if (t + c.M5) % (8 * c.H1) == 0:
                fund -= d * c.FUND * qty * cl
            continue
        return (d * (qty * px - notional) - notional * c.F_TAKER - qty * px * c.F_TAKER + fund, b5[k][0] + c.M5)
    px = b5[-1][4]
    return (d * (qty * px - notional) - notional * c.F_TAKER - qty * px * c.F_TAKER + fund, b5[-1][0] + c.M5)


def rolling_corr(h1, btc_ret):
    """Корреляция часовых доходностей монеты и BTC за прошлые CORR_WIN часов (по индексу часа)."""
    xs, ys, ok = [], [], []
    for i in range(len(h1)):
        rb = btc_ret.get(h1[i][0])
        if i == 0 or rb is None or h1[i - 1][4] <= 0:
            xs.append(0.0)
            ys.append(0.0)
            ok.append(0)
        else:
            xs.append(math.log(h1[i][4] / h1[i - 1][4]))
            ys.append(rb)
            ok.append(1)
    n = len(h1)
    cs = [[0.0] * (n + 1) for _ in range(6)]
    for i in range(n):
        v = (ok[i], xs[i], ys[i], xs[i] * xs[i], ys[i] * ys[i], xs[i] * ys[i])
        for k in range(6):
            cs[k][i + 1] = cs[k][i] + v[k]
    out = [None] * n
    for i in range(n):
        a = max(0, i + 1 - CORR_WIN)
        m = cs[0][i + 1] - cs[0][a]
        if m < CORR_WIN * 0.8:
            continue
        sx, sy = cs[1][i + 1] - cs[1][a], cs[2][i + 1] - cs[2][a]
        sxx, syy, sxy = cs[3][i + 1] - cs[3][a], cs[4][i + 1] - cs[4][a], cs[5][i + 1] - cs[5][a]
        vx, vy = sxx - sx * sx / m, syy - sy * sy / m
        if vx > 0 and vy > 0:
            out[i] = (sxy - sx * sy / m) / math.sqrt(vx * vy)
    return out


def main():
    t0 = time.time()
    coins = [x for x in c.FIRST_RUN if x != "BTC"]
    try:
        coins += [x for x in c.top_coins(50) if x not in coins and x != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    say("ТЕСТ ИДЕЙ · %s · монет %d · %d дней · риск на стопе %.0f $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), c.DAYS, RISK))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(c.load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)
    bh1 = c.agg(data["BTC"][0], c.H1)
    btc_ret = {bh1[i][0]: math.log(bh1[i][4] / bh1[i - 1][4]) for i in range(1, len(bh1)) if bh1[i - 1][4] > 0}

    def btc_ind(bar_open):
        p = bar_open - bar_open % c.D1 - c.D1
        i = bisect.bisect_left(btc_t, p)
        return btc_tr[i] if i < len(btc_t) and btc_t[i] == p else 0

    names = {"A0": "база (старший ТФ 4ч, стоп 5.2)", "A1": "старший ТФ индикатора 1Д",
             "B1": "корреляция с BTC >= 0.6", "C1": "без монет с убытком за 60 дней",
             "D1": "стоп 6.5 ATR", "D2": "стоп 8.0 ATR", "D3": "трендовый выход: стоп и трейлинг 3 ATR"}
    tr = {k: [] for k in names}
    corr_b = {"< 0.4": [], "0.4–0.6": [], ">= 0.6": [], "нет данных": []}
    t_min = t_max = None
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(h4) < 400 or len(d1) < 260:
            continue
        t5 = [x[0] for x in b5]
        h1 = c.agg(b5, c.H1)
        atr1 = c.atr_pct(h1)
        h4a = c.agg(b5, c.H4)
        c4 = [x[4] for x in h4a]
        up4 = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        cd = [x[4] for x in d1]
        upd = [a > b_ for a, b_ in zip(c.ema(cd, 9), c.ema(cd, 21))]
        corr = rolling_corr(h1, btc_ret)
        base_sig = [(T, d, i) for T, d, i in ind_entries_htf(h1, [x[0] for x in h4a], up4, c.H4)
                    if btc_ind(h1[i][0]) == d]
        d1_sig = [(T, d, i) for T, d, i in ind_entries_htf(h1, [x[0] for x in d1], upd, c.D1)
                  if btc_ind(h1[i][0]) == d]
        coin_base = []
        for T, d, i in base_sig:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1 or atr1[i] <= 0:
                continue
            x = st.sim(b5, k0, d, atr1[i], 5.2)
            coin_base.append((T, x[0], x[1]))
            tr["A0"].append((T, x[0], x[1]))
            cr = corr[i]
            key = "нет данных" if cr is None else "< 0.4" if cr < 0.4 else "0.4–0.6" if cr < 0.6 else ">= 0.6"
            corr_b[key].append(x[0])
            if cr is not None and cr >= CORR_MIN:
                tr["B1"].append((T, x[0], x[1]))
            for code, m in (("D1", 6.5), ("D2", 8.0)):
                y = st.sim(b5, k0, d, atr1[i], m)
                tr[code].append((T, y[0], y[1]))
            z = sim_trail(b5, k0, d, atr1[i], 3.0)
            tr["D3"].append((T, z[0], z[1]))
        for T, pnl, te in coin_base:
            past = [p for (t_, p, e_) in coin_base if T - PAST_WIN <= e_ < T]
            if len(past) >= PAST_MIN_N and sum(past) / len(past) < 0:
                continue
            tr["C1"].append((T, pnl, te))
        for T, d, i in d1_sig:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1 or atr1[i] <= 0:
                continue
            x = st.sim(b5, k0, d, atr1[i], 5.2)
            tr["A1"].append((T, x[0], x[1]))
        t_min = h1[0][0] if t_min is None else min(t_min, h1[0][0])
        t_max = h1[-1][0] if t_max is None else max(t_max, h1[-1][0])
    if t_min is None:
        say("Нет данных.")
        return
    mid = (t_min + t_max) / 2

    say("")
    say("Все сигналы (без «одна за раз»):")
    for k, nm in names.items():
        p = [x[1] for x in tr[k]]
        if not p:
            continue
        h = [[x[1] for x in tr[k] if (x[0] < mid) == (q == 1)] for q in (1, 2)]
        say("  %s %-40s сделок %5d · винрейт %4.1f%% · средняя %+5.2f $ (%+.3f R) · половины %+.3f / %+.3f R"
            % (k, nm, len(p), 100 * sum(1 for x in p if x > 0) / len(p), sum(p) / len(p), sum(p) / len(p) / RISK,
               sum(h[0]) / max(len(h[0]), 1) / RISK, sum(h[1]) / max(len(h[1]), 1) / RISK))
    say("")
    say("Справка — сделки базы по корреляции монеты с BTC за прошлые 30 дней:")
    for key, p in corr_b.items():
        if p:
            say("  корреляция %-10s сделок %5d · винрейт %4.1f%% · средняя %+5.2f $"
                % (key, len(p), 100 * sum(1 for x in p if x > 0) / len(p), sum(p) / len(p)))
    say("")
    say("ОДНА СДЕЛКА ЗА РАЗ, 30%% пропущено, %d сценариев, депозит 1000 $, риск 20 $:" % c.SCEN)
    res = {}
    for k, nm in names.items():
        if not tr[k]:
            continue
        r = c.scenarios(tr[k], mid)
        res[k] = r
        say("  %s %-40s сделок ~%d · итог медиана %+6.0f $ (10%% худших %+5.0f) · в плюсе %3.0f%% · "
            "половины %+5.0f / %+5.0f $ · просадка медиана %4.0f $ (10%% худших %4.0f)"
            % (k, nm, c.med(r["n"]), c.med(r["tot"]), r["tot"][c.SCEN // 10],
               100 * sum(1 for x in r["tot"] if x > 0) / c.SCEN, c.med(r["h1"]), c.med(r["h2"]),
               c.med(r["dd"]), r["dd"][c.SCEN * 9 // 10]))
    say("")
    say("ВЕРДИКТ (объявлен до прогона): лучше базы, если медиана выше в обеих половинах и просадка не больше")
    b = res["A0"]
    for k, nm in names.items():
        if k == "A0" or k not in res:
            continue
        r = res[k]
        c1 = c.med(r["h1"]) > c.med(b["h1"]) and c.med(r["h2"]) > c.med(b["h2"])
        c2 = c.med(r["dd"]) <= c.med(b["dd"])
        say("  %s %-40s половины [%s] · просадка [%s] → %s"
            % (k, nm, "да" if c1 else "нет", "да" if c2 else "нет", "ЛУЧШЕ" if (c1 and c2) else "не лучше"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

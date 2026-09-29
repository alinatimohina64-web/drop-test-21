#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
МНОГОЛЕТНЯЯ ПРОВЕРКА (с 2022 года) — бот, подтверждение индикатором, связь с BTC, отбор монет.

Монеты: 21 «старая» монета с историей на OKX с 2022 года (список MONEY ниже). Фильтр — BTC.
Исполнение на 15-минутках (5m за 4.5 года не помещается в лимит GitHub), спорная свеча — в
худшую сторону, комиссии 0.055% / тейк 0.02%, фандинг 0.01% за 8 ч.

ВАРИАНТЫ (везде: входы робота 🟢 + фильтр BTC 1Д + сетка GHOST B, 4 × 300 $, шаг ATR 1ч):
  V1 — база.
  V2 — + подтверждение индикатором: за последние 3 часа (включая сигнальный) Trand-Test-2 на 1h
       дал сигнал в ту же сторону.
  V3 — + монета связана с BTC: корреляция часовых доходностей за прошлые 30 дней >= 0.6.
  V4 — БОТ НА 4Ч: вход при развороте 4Ч (1Д в ту же сторону, ADX 4Ч >= 25, MRC 4Ч не на краю,
       цена по ту же сторону 200Д, BTC 1Д за), шаг сетки = ATR 4ч (не меньше 1%), та же формула
       тейка/стопа, ордера 300 $. Сделки крупнее и реже — «запустил и забыл», как у разработчика.

КАК ТОРГУЕТ ЧЕЛОВЕК: до 3 позиций одновременно (не больше одной на монету), 30% сигналов
пропущено, 500 сценариев на каждый год отдельно.

КРИТЕРИИ (объявлены ДО прогона):
  A) V2, V3 или V4 лучше V1, если медиана итога года лучше в 3 из 4 проверочных лет
     (2023, 2024, 2025, 2026 — последний неполный) и сумма медиан по этим годам больше.
  B) Отбор монет работает для варианта, если топ-7 монет по году N в году N+1 лучше всех монет
     (и по средней сделке, и по медиане итога) в 3 из 4 переходов 2022→23, 23→24, 24→25, 25→26.

Лежит рядом с compare_test.py и ideas_test.py. Кэш — data_ind/. Отчёт: multi_report.txt
"""
import bisect
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import compare_test as c
import ideas_test as it

REPORT = "multi_report.txt"
MONEY = ["ETH", "SOL", "XRP", "DOGE", "LINK", "ADA", "AVAX", "BNB", "LTC", "UNI", "FIL", "DOT", "NEAR",
         "XLM", "BCH", "AAVE", "ATOM", "ETC", "ALGO", "CRV", "ONE"]
M15 = 900000
START = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
WARM_DAYS = 120
YEARS = [2022, 2023, 2024, 2025, 2026]
SLOTS, SKIP, SCEN = 3, 0.3, 500
CONFIRM_H = 3
TOPN = 7
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def load_coin(sym):
    now = int(time.time() * 1000)
    days = (now - START) // c.D1 + WARM_DAYS
    b15 = c.load(sym, "15m", M15, days)
    h4 = c.load(sym, "4H", c.H4, days + 70)
    d1 = c.load(sym, "1Dutc", c.D1, days + 230)
    print("  %s: 15m %d, 4Ч %d, 1Д %d" % (sym, len(b15), len(h4), len(d1)), flush=True)
    return sym, (b15, h4, d1)


def sim_grid(b, k0, d, step):
    """Сетка бота (как compare_test.sim_grid), свечи 15m."""
    s = step / 100
    tp, sl = 2.5 * s, 5.2 * s
    p0 = b[k0][1]
    prices = [p0 * (1 + d * j * s) for j in range(c.N_ORD)]
    qty, cost, n = c.VOL / p0, c.VOL, 1
    fees, fund = c.VOL * c.F_TAKER, 0.0
    sl_px, peak = p0 * (1 - d * sl), None

    def done(px, k, maker=False):
        return d * (qty * px - cost) - fees - qty * px * (c.F_MAKER if maker else c.F_TAKER) + fund, b[k][0] + M15

    for k in range(k0, len(b)):
        t, o, h, l, cl = b[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        avg = cost / qty
        stop = sl_px
        if n >= 2:
            trl = avg + c.KEEPS[n - 2] * (peak - avg)
            if d * trl > d * stop:
                stop = trl
        if d * o <= d * stop:
            return done(o, k)
        if d * adv <= d * stop:
            return done(stop, k)
        while n < c.N_ORD and d * fav >= d * prices[n]:
            px = o if d * o > d * prices[n] else prices[n]
            qty += c.VOL / px
            cost += c.VOL
            fees += c.VOL * c.F_TAKER
            n += 1
            if peak is None or d * px > d * peak:
                peak = px
        avg = cost / qty
        tp_px = avg * (1 + d * tp)
        if d * fav >= d * tp_px:
            return done(o if d * o > d * tp_px else tp_px, k, True)
        if n >= 2:
            if d * fav > d * peak:
                peak = fav
            trl = avg + c.KEEPS[n - 2] * (peak - avg)
            if d * cl <= d * trl:
                return done(trl, k)
        if (t + M15) % (8 * c.H1) == 0:
            fund -= d * c.FUND * qty * cl
    return done(b[-1][4], len(b) - 1)


def scen(trades):
    """До SLOTS позиций, не больше одной на монету, SKIP сигналов пропущено."""
    trades = sorted(trades, key=lambda x: x[0])
    rnd = random.Random(7)
    tot, dds, ns = [], [], []
    for _ in range(SCEN):
        open_ = []
        eq = pk = dd = 0.0
        n = 0
        for T, pnl, te, sym in trades:
            open_ = [o for o in open_ if o[0] > T]
            if len(open_) >= SLOTS or any(o[1] == sym for o in open_) or rnd.random() < SKIP:
                continue
            open_.append((te, sym))
            eq += pnl
            pk = max(pk, eq)
            dd = max(dd, pk - eq)
            n += 1
        tot.append(eq)
        dds.append(dd)
        ns.append(n)
    tot.sort()
    dds.sort()
    ns.sort()
    return {"med": tot[SCEN // 2], "p10": tot[SCEN // 10], "plus": 100 * sum(1 for x in tot if x > 0) / SCEN,
            "dd": dds[SCEN // 2], "n": ns[SCEN // 2]}


def bot_entries_4h(h4, d1):
    """Вход при развороте 4Ч по тренду 1Д: (время закрытия 4ч, сторона, индекс 4ч)."""
    tr4, trd, adx4 = c.trend_arr(h4), c.trend_arr(d1), c.adx_arr(h4)
    dcl = [x[4] for x in d1]
    pre = [0.0]
    for v in dcl:
        pre.append(pre[-1] + v)
    out, jd = [], -1
    for i in range(300, len(h4)):
        T = h4[i][0] + c.H4
        while jd + 1 < len(d1) and d1[jd + 1][0] + c.D1 <= T:
            jd += 1
        if jd < 199:
            continue
        d = tr4[i]
        if d == 0 or d == tr4[i - 1] or trd[jd] != d or adx4[i] < c.ADX_BOT:
            continue
        s200 = (pre[jd + 1] - pre[jd - 199]) / 200
        if (d == 1 and not dcl[jd] > s200) or (d == -1 and dcl[jd] > s200):
            continue
        z = c.mrc_zone(h4[i - 298:i + 1])
        if (d == 1 and z == "перекуп") or (d == -1 and z == "перепрод"):
            continue
        out.append((T, d, i))
    return out


def main():
    t0 = time.time()
    say("МНОГОЛЕТНЯЯ ПРОВЕРКА · %s · монет %d · с 2022 года · исполнение 15m · до %d позиций"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(MONEY), SLOTS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(load_coin, ["BTC"] + MONEY))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], c.trend_arr(bd1)
    bh1 = c.agg(data["BTC"][0], c.H1)
    btc_ret = {bh1[i][0]: math.log(bh1[i][4] / bh1[i - 1][4]) for i in range(1, len(bh1)) if bh1[i - 1][4] > 0}

    def btc_at_close(T):
        i = bisect.bisect_right(btc_t, T - c.D1) - 1
        return btc_tr[i] if i >= 0 else 0

    V = {"V1": "бот + BTC 1Д (база)", "V2": "+ подтверждение индикатором (3 ч)", "V3": "+ корреляция с BTC >= 0.6",
         "V4": "бот на 4ч (шаг ATR 4ч)"}
    tr = {v: [] for v in V}              # (T, pnl, t_exit, монета)
    for sym in MONEY:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 50000 or len(h4) < 400 or len(d1) < 260:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = [x[0] for x in b15]
        h1 = c.agg(b15, c.H1)
        atr1 = c.atr_pct(h1)
        h4a = c.agg(b15, c.H4)
        c4 = [x[4] for x in h4a]
        up4 = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        ind = it.ind_entries_htf(h1, [x[0] for x in h4a], up4, c.H4)
        ind_t = {1: sorted(T for T, d, i in ind if d == 1), -1: sorted(T for T, d, i in ind if d == -1)}
        corr = it.rolling_corr(h1, btc_ret)
        n_b = 0
        for T, d, i in c.bot_entries(h1, h4, d1):
            if T < START or btc_at_close(T) != d:
                continue
            k0 = bisect.bisect_left(t15, T)
            if k0 >= len(b15) - 1:
                continue
            pnl, te = sim_grid(b15, k0, d, max(atr1[i], c.MIN_STEP))
            rec = (T, pnl, te, sym)
            tr["V1"].append(rec)
            n_b += 1
            lst = ind_t[d]
            j = bisect.bisect_right(lst, T) - 1
            if j >= 0 and lst[j] >= T - CONFIRM_H * c.H1:
                tr["V2"].append(rec)
            if corr[i] is not None and corr[i] >= 0.6:
                tr["V3"].append(rec)
        atr4 = c.atr_pct(h4)
        n4 = 0
        for T, d, i in bot_entries_4h(h4, d1):
            if T < START or btc_at_close(T) != d:
                continue
            k0 = bisect.bisect_left(t15, T)
            if k0 >= len(b15) - 1:
                continue
            pnl, te = sim_grid(b15, k0, d, max(atr4[i], c.MIN_STEP))
            tr["V4"].append((T, pnl, te, sym))
            n4 += 1
        say("  %s: бот на 4ч — входов %d" % (sym, n4))
        say("  %s: с %s, входов бота (BTC 1Д за) %d" % (sym, datetime.fromtimestamp(b15[0][0] / 1000, tz=timezone.utc)
                                                         .strftime("%Y-%m-%d"), n_b))

    # ---- по годам
    say("")
    say("=" * 110)
    say("ПО ГОДАМ (до %d позиций, 30%% пропущено, %d сценариев; 2026 — неполный год)" % (SLOTS, SCEN))
    yr = {v: {} for v in V}
    for v, nm in V.items():
        say("  %s %s" % (v, nm))
        for y in YEARS:
            t_y = [x for x in tr[v] if year_of(x[0]) == y]
            if not t_y:
                say("    %d: нет сделок" % y)
                continue
            s = scen(t_y)
            yr[v][y] = s
            days = 365 if y < 2026 else max(1, (int(time.time() * 1000) - int(datetime(2026, 1, 1, tzinfo=timezone.utc)
                                                                               .timestamp() * 1000)) // c.D1)
            p = [x[1] for x in t_y]
            say("    %d: сигналов %4d · средняя %+5.2f $ · сценарий: сделок ~%d (%.1f в день) · итог медиана %+5.0f $ "
                "(10%% худших %+5.0f) · в плюсе %3.0f%% · просадка %4.0f $"
                % (y, len(p), sum(p) / len(p), s["n"], s["n"] / days, s["med"], s["p10"], s["plus"], s["dd"]))
    say("")
    say("КРИТЕРИЙ A (объявлен до прогона): лучше V1 в 3 из 4 лет 2023–2026 и сумма больше")
    test_years = [2023, 2024, 2025, 2026]
    for v in ("V2", "V3", "V4"):
        wins = sum(1 for y in test_years if y in yr[v] and y in yr["V1"] and yr[v][y]["med"] > yr["V1"][y]["med"])
        s_v = sum(yr[v][y]["med"] for y in test_years if y in yr[v])
        s_b = sum(yr["V1"][y]["med"] for y in test_years if y in yr["V1"])
        say("  %s: лучше в %d из 4 лет · сумма %+.0f против %+.0f → %s"
            % (v, wins, s_v, s_b, "ЛУЧШЕ БАЗЫ" if (wins >= 3 and s_v > s_b) else "не лучше"))

    # ---- отбор монет по годам
    say("")
    say("=" * 110)
    say("КРИТЕРИЙ B — отбор топ-%d монет по году N, торговля в году N+1" % TOPN)
    for v, nm in V.items():
        say("  %s %s" % (v, nm))
        ok_cnt = 0
        for y0, y1 in ((2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026)):
            t0y = [x for x in tr[v] if year_of(x[0]) == y0]
            t1y = [x for x in tr[v] if year_of(x[0]) == y1]
            cnt0 = {}
            tot0 = {}
            for x in t0y:
                tot0[x[3]] = tot0.get(x[3], 0.0) + x[1]
                cnt0[x[3]] = cnt0.get(x[3], 0) + 1
            elig = [s for s in tot0 if cnt0[s] >= 10]
            if len(elig) < TOPN + 3 or not t1y:
                say("    %d→%d: мало данных" % (y0, y1))
                continue
            top = sorted(elig, key=lambda s: -tot0[s])[:TOPN]
            t_top = [x for x in t1y if x[3] in top]
            if not t_top:
                say("    %d→%d: у топа нет сделок" % (y0, y1))
                continue
            a_all = sum(x[1] for x in t1y) / len(t1y)
            a_top = sum(x[1] for x in t_top) / len(t_top)
            s_all, s_top = scen(t1y), scen(t_top)
            good = a_top > a_all and s_top["med"] > s_all["med"]
            ok_cnt += good
            say("    %d→%d: топ %s · средняя сделка %+5.2f против %+5.2f · итог %+5.0f против %+5.0f $ → %s"
                % (y0, y1, ",".join(top), a_top, a_all, s_top["med"], s_all["med"], "лучше" if good else "не лучше"))
        say("    ВЕРДИКТ: отбор лучше в %d из 4 переходов → %s"
            % (ok_cnt, "ОТБОР РАБОТАЕТ" if ok_cnt >= 3 else "отбор не работает"))

    # ---- список на следующий период
    say("")
    say("Топ-%d по 2026 году (для каждого варианта) — список, если отбор для варианта работает:" % TOPN)
    for v in V:
        tot = {}
        for x in tr[v]:
            if year_of(x[0]) == 2026:
                tot[x[3]] = tot.get(x[3], 0.0) + x[1]
        order = sorted(tot, key=lambda s: -tot[s])
        say("  %s: %s" % (v, ", ".join("%s %+.0f" % (s, tot[s]) for s in order)))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

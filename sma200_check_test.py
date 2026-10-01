#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРОВЕРКА УСТОЙЧИВОСТИ НАХОДКИ R2 (sma200_test.py): TT2 без MRC, цель 4 / стоп 20 ATR, только когда BTC выше SMA200.
Сигналы, исполнение, комиссии, данные — как в sma200_test.py (30m, ст. ТФ 4ч, тренд BTC 1Д, 15m Binance,
позиция 1000 $, в спорной свече — стоп, 250 дневок у монеты). Оба набора монет: основной и новый.

1) СОСЕДНИЕ НАСТРОЙКИ: SMA BTC 150 / 200 / 250 × цель 3 / 4 / 5 × стоп 15 / 20 / 25 ATR (27 вариантов),
   одна позиция на монету. КРИТЕРИЙ «устойчиво» (объявлен ДО прогона), на КАЖДОМ наборе:
   а) в плюсе не меньше 75% вариантов (21 из 27);
   б) медианный вариант даёт не меньше половины итога R2 (200 / 4 / 20).
2) ЛОНГИ И ШОРТЫ R2 отдельно — справочно (новое правило из этого не делаем без отдельной проверки).
3) ЛИМИТ ОДНОВРЕМЕННЫХ ПОЗИЦИЙ на весь счёт: без лимита / 10 / 5 / 3. Сигналы всех монет по времени,
   вход берётся, если монета свободна и открытых позиций меньше лимита. Справочно: итог, просадка счёта
   с учётом незакрытых позиций, итог/просадка, депозит для просадки не больше 50% — чтобы подобрать размер
   позиции под реальный депозит.
Лежит рядом с sma200_test.py и coin_select_test.py. Отчёт: sma200_check_report.txt
"""
import bisect
import statistics
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st

c, cw = cs.c, cs.cw
REPORT = "sma200_check_report.txt"
M30, D1, H4 = cs.M30, cs.D1, cs.H4
SMAS = [150, 200, 250]
TPS = [3.0, 4.0, 5.0]
SLS = [15.0, 20.0, 25.0]
BASE = (200, 4.0, 20.0)
CAPS = [None, 10, 5, 3]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def run_set(set_name, coins, btc, above):
    t0 = time.time()
    seqs = {}       # (sma, tp, sl) -> [сделки], одна позиция на монету
    allsig = []     # для лимита: все сигналы R2 со своими исходами (T, сделка)
    days = {}
    end_t = 0
    for sym in coins:
        b15 = cs.load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        end_t = max(end_t, int(t15[-1]))
        b30 = c.agg(b15, M30)
        atrp = c.atr_pct(b30)
        h4a = c.agg(b15, H4)
        c4 = [x[4] for x in h4a]
        hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
        d1 = c.agg(b15, D1)
        dt = [x[0] for x in d1]
        days[sym] = (dt, [x[4] for x in d1])
        raw = []
        for T, d, i in cw.ind_entries_tf(b30, M30, [x[0] for x in h4a], hu, H4):
            if T < st.START or atrp[i] <= 0 or btc.trend(T) != d:
                continue
            if bisect.bisect_right(dt, T - D1) - 1 < 250:
                continue
            raw.append((T, d, atrp[i] / 100))

        def one(T, d, a, tp, sl):
            k0 = int(np.searchsorted(t15, T, side="left"))
            if k0 >= len(t15) - 1:
                return None
            pnl, r, te = cs.trade(t15, h, l, cl, k0, d, o[k0], tp * a, sl * a)
            return (T, pnl, r, te, sym, d, float(o[k0]), st.NOTIONAL * sl * a)

        for n in SMAS:
            sig = [x for x in raw if above[n](x[0])]
            for tp in TPS:
                for sl in SLS:
                    busy, out = 0, []
                    for T, d, a in sig:
                        if T < busy:
                            continue
                        tr = one(T, d, a, tp, sl)
                        if tr is None:
                            continue
                        out.append(tr)
                        busy = tr[3] if tr[3] is not None else 10 ** 15
                    seqs.setdefault((n, tp, sl), []).extend(out)
        for T, d, a in raw:
            if above[BASE[0]](T):
                tr = one(T, d, a, BASE[1], BASE[2])
                if tr is not None:
                    allsig.append(tr)
        say("  [%s] %s: сигналов %d · %.0f с" % (set_name, sym, len(raw), time.time() - t0))
    return seqs, allsig, days, end_t


def capped(allsig, cap):
    allsig = sorted(allsig)
    busy = {}
    opened = []     # времена выхода открытых позиций
    out = []
    for tr in allsig:
        T, sym, te = tr[0], tr[4], tr[3]
        if busy.get(sym, 0) > T:
            continue
        opened = [x for x in opened if x > T]
        if cap is not None and len(opened) >= cap:
            continue
        out.append(tr)
        e = te if te is not None else 10 ** 15
        busy[sym] = e
        opened.append(e)
    return out


def line(trades, days, end_t):
    n = len(trades)
    tot = sum(x[1] for x in trades)
    mo = st.money(trades, days, end_t) if trades else {"dd": 0, "maxpos": 0, "maxrisk": 0, "dd_from": None, "dd_t": None}
    rat = tot / -mo["dd"] if mo["dd"] < 0 else float("inf")
    return n, tot, mo, rat


def main():
    t0 = time.time()
    say("УСТОЙЧИВОСТЬ R2 (SMA200 BTC, без MRC, 4/20) · %s · Binance 15m · позиция 1000 $"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    smas = {n: c.sma(btc.cl, n) for n in SMAS}

    def mk(n):
        def f(T):
            i = btc.idx(T)
            return i >= 0 and smas[n][i] is not None and btc.cl[i] > smas[n][i]
        return f
    above = {n: mk(n) for n in SMAS}

    verdict = {}
    for set_name, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS)):
        say("")
        say("#" * 110)
        say("НАБОР «%s»" % set_name)
        seqs, allsig, days, end_t = run_set(set_name, coins, btc, above)

        say("")
        say("1) СОСЕДНИЕ НАСТРОЙКИ (итог в $ · сделок), строки — SMA, столбцы — цель/стоп")
        hdr = "  SMA  " + " ".join("%11s" % ("%g/%g" % (tp, sl)) for tp in TPS for sl in SLS)
        say(hdr)
        tots = {}
        for n in SMAS:
            cells = []
            for tp in TPS:
                for sl in SLS:
                    v = seqs.get((n, tp, sl), [])
                    s = sum(x[1] for x in v)
                    tots[(n, tp, sl)] = s
                    cells.append("%+7.0f(%4d)" % (s, len(v)))
            say("  %3d  " % n + " ".join(cells))
        pos = sum(1 for v in tots.values() if v > 0)
        med = statistics.median(tots.values())
        base = tots[BASE]
        ka = pos >= 21
        kb = med >= base / 2
        verdict[set_name] = ka and kb
        say("  в плюсе %d из 27 [%s] · медиана %+.0f $ против половины R2 %+.0f $ [%s] → %s"
            % (pos, "да" if ka else "нет", med, base / 2, "да" if kb else "нет", "УСТОЙЧИВО" if (ka and kb) else "не устойчиво"))

        say("")
        say("2) R2: ЛОНГИ И ШОРТЫ")
        r2 = seqs[BASE]
        for nm, dd in (("лонги", 1), ("шорты", -1)):
            v = [x for x in r2 if x[5] == dd]
            yrs = {}
            for x in v:
                yrs[cs.year_of(x[0])] = yrs.get(cs.year_of(x[0]), 0) + x[1]
            say("  %-6s сделок %4d · WR %4.1f%% · итог %+7.0f $ · %+6.2f $/сд · по годам: %s" % (
                nm, len(v), sum(1 for x in v if x[1] > 0) / max(1, len(v)) * 100, sum(x[1] for x in v),
                sum(x[1] for x in v) / max(1, len(v)), " · ".join("%d: %+.0f" % (k, s) for k, s in sorted(yrs.items()))))

        say("")
        say("3) ЛИМИТ ОДНОВРЕМЕННЫХ ПОЗИЦИЙ НА ВЕСЬ СЧЁТ (R2)")
        for cap in CAPS:
            tr = capped(allsig, cap)
            n, tot, mo, rat = line(tr, days, end_t)
            say("  %-12s сделок %4d · итог %+7.0f $ · просадка %+7.0f $ (%s → %s) · позиций до %2d · стопов до %6.0f $ · "
                "итог/просадка %.2f · депозит для просадки ≤50%% ≈ %.0f $"
                % ("без лимита" if cap is None else "до %d" % cap, n, tot, mo["dd"],
                   st.dstr(mo["dd_from"]) if mo["dd_from"] else "-", st.dstr(mo["dd_t"]) if mo["dd_t"] else "-",
                   mo["maxpos"], mo["maxrisk"], rat, -mo["dd"] * 2))

    say("")
    say("=" * 110)
    say("ИТОГ ПО КРИТЕРИЮ УСТОЙЧИВОСТИ: %s" % " · ".join("%s — %s" % (k, "устойчиво" if v else "НЕ устойчиво")
                                                       for k, v in verdict.items()))
    say("→ %s" % ("НАХОДКА УСТОЙЧИВА: правило работает и на соседних настройках" if all(verdict.values())
                  else "находка держится на узкой точке настроек — осторожно"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

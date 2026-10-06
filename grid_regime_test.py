#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
КОНТРТРЕНДОВАЯ ЛОНГ-СЕТКА И МЕДВЕЖКА: застревает ли сетка именно когда BTC ниже SMA200 и стоит ли её тогда выключать.

СЕТКА — как в «NEW_Боты_симуляция_реинвест.xlsx» (вкладка «Реинвест (расчеты)», LONG):
  депозит 10 000 $, кросс-маржа, плечо 10; первый ордер 10 $ (номинал), 9 страховочных ордеров, объём ×2 (мартингейл 2),
  ширина сетки 8% — шаг 0.889% от цены входа (ровно), полная сетка 10 230 $; тейк 0.8% от средней; стопа нет.
  После тейка новый цикл сразу (по открытию следующей 15m-свечи). Размер не растёт (без реинвеста).
Исполнение — Binance 15m. Внутри свечи худший порядок: если в свече набрался ордер, тейк в этой свече не считается.
Комиссия 0.04% с каждой стороны (вход, доливки, тейк), фандинг 0.01% / 8 ч на набранный объём.
Итог = закрытые циклы + незакрытая позиция по последней цене (застрявшая позиция в конце — тоже в итоге).
Просадка — худшее падение счёта (закрытые + незакрытая по минимуму свечи) от предыдущего максимума.
Ликвидация (кросс-маржа): счёт 10 000 $ + заработанное минус убыток позиции дошёл до 0.5% от позиции — монета
выбывает с итогом −10 000 $ (теряется депозит и всё заработанное).
Период — с 2019-01-01 (или с листинга). Режим: «BTC выше SMA200» — вчерашнее дневное закрытие выше дневной SMA200
(как в роботе); «🐻 медвежка» — BTC ниже SMA200 И SMA200 за 30 дней пошла вниз.

ВАРИАНТЫ (объявлены до прогона, без подбора):
  A  всегда включена (как сейчас);
  B  новые циклы только когда BTC выше SMA200 (идущий цикл доводится до тейка);
  C  новые циклы не начинать только в «🐻 медвежке» (ниже SMA200 и SMA200 вниз);
  D  справочно: как B, но при переходе BTC под SMA200 открытая позиция закрывается по рынку (фиксируем убыток).

ЧАСТЬ 1 — ГДЕ ЗАСТРЕВАЕТ (вариант A): сколько раз сетка набиралась целиком и сколько дней ждала тейка — при BTC выше
  и ниже SMA200; доход в месяц в каждом режиме.
ЧАСТЬ 2 — ВЫКЛЮЧАТЬ ЛИ: A / B / C / D по трём наборам монет (основной 21, новый 20, третий 40) и по половинам периода.
КОНТРОЛЬ 1 — «сдвинутый режим»: B и C с режимом BTC, сдвинутым на 12 разных сроков (та же доля и длина выключений,
  но в другие даты). Если реальный режим не лучше сдвинутого — польза от выключения случайна.
КОНТРОЛЬ 2 — случайные цены: та же сетка A на перемешанных 15m-свечах той же монеты (та же подвижность и общий
  рост/падение, но без настоящих трендов и откатов) — справочно, есть ли у сетки преимущество вообще.

КРИТЕРИЙ (объявлен ДО прогона). Выключение (B или C) принимается, если на ≥ 2 из 3 наборов монет одновременно:
  1) медианная по монетам максимальная просадка меньше, чем у A, хотя бы на 25%;
  2) итог набора не хуже A больше чем на 20% (итог ≥ A − 0.2·|A|);
  3) медианная просадка меньше, чем у A, в ОБЕИХ половинах периода (до / после 2023-01-01);
  4) реальный режим лучше сдвинутого: «итог / сумма просадок» у реального выше, чем у ≥ 10 из 12 сдвигов.
  Если проходят оба — главным считается тот, что проще (B). XRP отдельно — справочно (одна монета = шум).
Лежит рядом с coin_select_test.py, sma200_test.py, sma200_vol_test.py. Отчёт: grid_regime_report.txt
"""
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs
import sma200_test as st
import sma200_vol_test as sv

REPORT = "grid_regime_report.txt"
D1 = cs.D1
H8 = 8 * 3600 * 1000
DEPO = 10000.0
FIRST = 10.0
N_SO = 9
MART = 2.0
WIDTH = 0.08
STEP = WIDTH / N_SO
TP = 0.008
FEE = 0.0004
FUND8 = 0.0001
MMR = 0.005
START = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
HALF = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
N_SHIFT = 12
STUCK_D = 30
SETS = [("основной", list(st.MAIN_COINS)), ("новый", list(st.NEW_COINS)), ("третий (топ-50)", list(sv.THIRD_COINS))]
SIZES = [FIRST * MART ** k for k in range(N_SO + 1)]
FULL = sum(SIZES)
_out = []
G = {}


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def dstr(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def next_true(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def first_hit(lo, hi, i, L, T, lim):
    step = 256
    while i < lim:
        j = min(lim, i + step)
        m = (lo[i:j] <= L) | (hi[i:j] >= T)
        k = int(m.argmax())
        if m[k]:
            return i + k
        i = j
        step = min(step * 2, 65536)
    return lim


def sim(t, o, h, l, c, on, above, force=False):
    """Одна монета, один вариант. on[i] — можно начинать новый цикл на открытии свечи i."""
    n = len(t)
    i0 = int(np.searchsorted(t, START))
    nxt_on = next_true(on)
    nxt_off = next_true(~on) if force else None
    real = 0.0
    peak = 0.0
    mdd = [0.0, 0.0]
    cyc = 0
    full = []                     # (t полного набора, t выхода или None, режим выше SMA200?)
    inpos = 0                     # свечей в позиции
    by_reg = [0.0, 0.0]          # закрытые циклы: [ниже, выше] SMA200 на момент тейка
    by_year = {}
    forced = [0, 0.0]
    i = i0
    end_mtm = 0.0
    liq = None
    while i < n and liq is None:
        if not on[i]:
            i = int(nxt_on[i])
            continue
        p0 = o[i]
        lev = [p0 * (1 - STEP * k) for k in range(N_SO + 1)]
        qty = FIRST / p0
        cost = FIRST
        fees = FIRST * FEE
        fund = 0.0
        tl = t[i]
        k = 1
        tfull = None
        j = i
        t_open = i
        while True:
            lim = n if not force else max(j, int(nxt_off[j]))
            tp = cost / qty * (1 + TP)
            L = lev[k] if k <= N_SO else -1.0
            m = first_hit(l, h, j, L, tp, lim)
            if m > j:
                lowseg = float(l[j:m].min())
                fnow = fund + cost * (t[m - 1] - tl) / H8 * FUND8
                p_liq = (cost + fees + fnow - real - DEPO) / (qty * (1 - MMR))
                if lowseg <= p_liq:
                    m_liq = j + int(np.argmax(l[j:m] <= p_liq))
                    liq = t[m_liq]
                    break
                eq = real + qty * lowseg - cost - fees - fnow
                hf = 0 if t[j] < HALF else 1
                mdd[hf] = max(mdd[hf], peak - eq)
            if m >= lim:
                fund += cost * (t[m - 1] - tl) / H8 * FUND8 if m > 0 else 0.0
                inpos += m - t_open
                if lim == n:
                    end_mtm = qty * c[n - 1] - cost - fees - fund - qty * c[n - 1] * FEE
                    if tfull is not None:
                        full.append((tfull, None, bool(above[min(n - 1, int(np.searchsorted(t, tfull)))])))
                    i = n
                else:
                    px = o[m]
                    pnl = qty * px - cost - fees - fund - qty * px * FEE
                    real += pnl
                    forced[0] += 1
                    forced[1] += pnl
                    y = datetime.fromtimestamp(t[m] / 1000, tz=timezone.utc).year
                    by_year[y] = by_year.get(y, 0.0) + pnl
                    if tfull is not None:
                        full.append((tfull, t[m], bool(above[int(np.searchsorted(t, tfull))])))
                    peak = max(peak, real)
                    i = m
                break
            filled = False
            while k <= N_SO and l[m] <= lev[k]:
                fund += cost * (t[m] - tl) / H8 * FUND8
                tl = t[m]
                add = SIZES[k]
                qty += add / lev[k]
                cost += add
                fees += add * FEE
                k += 1
                filled = True
                if k > N_SO:
                    tfull = t[m]
            if filled:
                p_liq = (cost + fees + fund - real - DEPO) / (qty * (1 - MMR))
                if l[m] <= p_liq:
                    liq = t[m]
                    break
                eq = real + qty * l[m] - cost - fees - fund
                hf = 0 if t[m] < HALF else 1
                mdd[hf] = max(mdd[hf], peak - eq)
                j = m + 1
                if j >= n:
                    end_mtm = qty * c[n - 1] - cost - fees - fund - qty * c[n - 1] * FEE
                    inpos += n - t_open
                    if tfull is not None:
                        full.append((tfull, None, bool(above[n - 1])))
                    i = n
                    break
                continue
            fund += cost * (t[m] - tl) / H8 * FUND8
            pnl = qty * tp - cost - fees - fund - qty * tp * FEE
            real += pnl
            cyc += 1
            by_reg[1 if above[m] else 0] += pnl
            y = datetime.fromtimestamp(t[m] / 1000, tz=timezone.utc).year
            by_year[y] = by_year.get(y, 0.0) + pnl
            if tfull is not None:
                full.append((tfull, t[m], bool(above[int(np.searchsorted(t, tfull))])))
            inpos += m + 1 - t_open
            peak = max(peak, real)
            i = m + 1
            break
    if liq is not None:
        y = datetime.fromtimestamp(liq / 1000, tz=timezone.utc).year
        by_year[y] = by_year.get(y, 0.0) - DEPO - real
        hf = 0 if liq < HALF else 1
        mdd[hf] = max(mdd[hf], peak + DEPO)
        if tfull is not None:
            full.append((tfull, None, bool(above[int(np.searchsorted(t, tfull))])))
        real, end_mtm = -DEPO, 0.0
    tot = real + end_mtm
    if end_mtm:
        y = datetime.fromtimestamp(t[-1] / 1000, tz=timezone.utc).year
        by_year[y] = by_year.get(y, 0.0) + end_mtm
    hreal = [0.0, 0.0]
    for y, v in by_year.items():
        hreal[0 if y < 2023 else 1] += v
    return {"tot": tot, "real": real, "mtm": end_mtm, "mdd": max(mdd), "mdd_h": mdd, "cyc": cyc, "full": full,
            "inpos": inpos, "by_reg": by_reg, "by_year": by_year, "half": hreal, "forced": forced,
            "liq": liq, "t0": int(t[i0]) if i0 < n else None, "t1": int(t[-1])}


def regime_arrays(t, shift=0):
    a_d, bear_d, btc_t, d0 = G["above_d"], G["bear_d"], G["btc_t"], G["d0"]
    di = np.searchsorted(btc_t, t - D1, side="right") - 1
    di = np.clip(di, 0, len(btc_t) - 1)
    if shift:
        Ld = len(btc_t) - d0
        src = np.where(di >= d0, d0 + ((di - d0 - shift) % Ld), di)
    else:
        src = di
    return a_d[src], bear_d[src]


def work(sym):
    t0 = time.time()
    try:
        bars = cs.load_15m(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(bars) < 2000:
        return sym, None, "мало данных"
    a = np.array(bars, dtype=float)
    t = a[:, 0].astype(np.int64)
    o, h, l, c = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    if t[-1] < START + 200 * D1:
        return sym, None, "мало данных после 2019"
    above, bear = regime_arrays(t)
    on_all = np.ones(len(t), dtype=bool)
    res = {"A": sim(t, o, h, l, c, on_all, above),
           "B": sim(t, o, h, l, c, above.copy(), above),
           "C": sim(t, o, h, l, c, ~bear, above),
           "D": sim(t, o, h, l, c, above.copy(), above, force=True)}
    i0 = int(np.searchsorted(t, START))
    res["share_above"] = float(above[i0:].mean()) if i0 < len(t) else 0.0
    res["share_bear"] = float(bear[i0:].mean()) if i0 < len(t) else 0.0
    res["months"] = (t[-1] - t[i0]) / (30.44 * D1) if i0 < len(t) else 0.0
    res["months_above"] = res["months"] * res["share_above"]
    sh = []
    for s in G["shifts"]:
        ab_s, be_s = regime_arrays(t, s)
        rb = sim(t, o, h, l, c, ab_s.copy(), above)
        rc = sim(t, o, h, l, c, ~be_s, above)
        sh.append(({"tot": rb["tot"], "mdd": rb["mdd"]}, {"tot": rc["tot"], "mdd": rc["mdd"]}))
    res["shift"] = sh
    rng = np.random.default_rng(sum(ord(ch) * 131 ** k for k, ch in enumerate(sym)) % (2 ** 32))
    seg = slice(i0, len(t))
    oo, hh, ll, cc = o[seg], h[seg], l[seg], c[seg]
    rc_, rh_, rl_ = cc / oo, hh / oo, ll / oo
    p = rng.permutation(len(oo))
    rc_, rh_, rl_ = rc_[p], rh_[p], rl_[p]
    no = np.empty(len(oo))
    no[0] = oo[0]
    no[1:] = oo[0] * np.cumprod(rc_[:-1])
    tt = t[seg]
    res["rand"] = sim(tt, no, no * rh_, no * rl_, no * rc_, np.ones(len(tt), dtype=bool), above[seg])
    return sym, res, "%.0f с" % (time.time() - t0)


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def full_stats(rs):
    ep = [f for r in rs for f in r["full"]]
    out = {}
    for reg in (True, False):
        e = [f for f in ep if f[2] == reg]
        days = [((f[1] if f[1] is not None else G["t_end"]) - f[0]) / D1 for f in e]
        out[reg] = (len(e), sum(1 for d in days if d > STUCK_D), med(days), max(days) if days else 0.0,
                    sum(1 for f in e if f[1] is None), sum(days))
    return out


def main():
    t_start = time.time()
    say("СЕТКА И МЕДВЕЖКА · %s · Binance 15m · с 2019-01-01" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    say("Сетка: депозит %.0f $, первый ордер %.0f $, %d страховочных ×%.0f, ширина %.0f%% (шаг %.3f%%), полная сетка %.0f $, "
        "тейк %.1f%%, без стопа" % (DEPO, FIRST, N_SO, MART, WIDTH * 100, STEP * 100, FULL, TP * 100))
    btc = st.Btc()
    s200 = btc.sma
    n = len(btc.t)
    above_d = np.array([s200[i] is not None and btc.cl[i] > s200[i] for i in range(n)], dtype=bool)
    bear_d = np.array([i >= 30 and s200[i] is not None and s200[i - 30] is not None and btc.cl[i] < s200[i]
                       and s200[i] < s200[i - 30] for i in range(n)], dtype=bool)
    G["btc_t"] = np.array(btc.t, dtype=np.int64)
    G["above_d"], G["bear_d"] = above_d, bear_d
    G["d0"] = int(np.searchsorted(G["btc_t"], START))
    Ld = n - G["d0"]
    G["shifts"] = [int(Ld * (k + 1) / (N_SHIFT + 1)) for k in range(N_SHIFT)]
    G["t_end"] = int(btc.t[-1]) + D1
    say("BTC ниже SMA200 с 2019: %.0f%% дней · «🐻» (ниже и SMA200 вниз): %.0f%% дней · сдвиги режима, дней: %s"
        % (100 * (1 - above_d[G["d0"]:].mean()), 100 * bear_d[G["d0"]:].mean(), " ".join(str(s) for s in G["shifts"])))
    allc = []
    for nm, cl in SETS:
        allc += [x for x in cl if x not in allc]
    with mp.get_context("fork").Pool(max(1, os.cpu_count() or 2)) as pool:
        got = {}
        for sym, res, msg in pool.imap_unordered(work, allc):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res

    # ---------- XRP подробно ----------
    say()
    say("=" * 110)
    if "XRP" in got:
        r = got["XRP"]
        say("XRP ПОДРОБНО (справочно — одна монета)")
        for v, nm in (("A", "всегда"), ("B", "только выше SMA200"), ("C", "кроме 🐻"), ("D", "выше SMA200 + закрыть при уходе вниз")):
            x = r[v]
            say("  %s %-38s итог %+8.0f $ (закрыто %+8.0f, незакрытая %+7.0f) · циклов %5d · просадка %6.0f $ "
                "(половины %.0f / %.0f) · полных наборов %d · вынужд. закрытий %d (%+.0f $)%s"
                % (v, nm, x["tot"], x["real"], x["mtm"], x["cyc"], x["mdd"], x["mdd_h"][0], x["mdd_h"][1], len(x["full"]),
                   x["forced"][0], x["forced"][1], (" · ЛИКВИДАЦИЯ " + dstr(x["liq"])) if x["liq"] else ""))
        say("  по годам (A / B / C):")
        ys = sorted(set(r["A"]["by_year"]) | set(r["B"]["by_year"]) | set(r["C"]["by_year"]))
        for y in ys:
            say("    %d: %+7.0f / %+7.0f / %+7.0f $" % (y, r["A"]["by_year"].get(y, 0), r["B"]["by_year"].get(y, 0),
                                                       r["C"]["by_year"].get(y, 0)))
        say("  полные наборы сетки XRP (вариант A): начало → выход · дней · BTC")
        for f in r["A"]["full"]:
            d = ((f[1] if f[1] else G["t_end"]) - f[0]) / D1
            if d >= 3:
                say("    %s → %s · %5.0f дн · %s" % (dstr(f[0]), dstr(f[1]) if f[1] else "НЕ ВЫШЛА", d,
                                                    "выше SMA200" if f[2] else "НИЖЕ SMA200"))
        say("  случайные цены XRP (A): итог %+.0f $ · просадка %.0f $ · циклов %d"
            % (r["rand"]["tot"], r["rand"]["mdd"], r["rand"]["cyc"]))

    # ---------- часть 1 ----------
    say()
    say("=" * 110)
    say("ЧАСТЬ 1 — ГДЕ ЗАСТРЕВАЕТ (вариант A, всегда включена). «Полный набор» = набраны все %d страховочных; "
        "«застряла» = ждала тейка > %d дней" % (N_SO, STUCK_D))
    for nm, cl in SETS:
        rs = [got[x] for x in cl if x in got]
        if not rs:
            continue
        A = [r["A"] for r in rs]
        fs = full_stats(A)
        mo_up = sum(r["months_above"] for r in rs)
        mo_dn = sum(r["months"] - r["months_above"] for r in rs)
        inc_up = sum(a["by_reg"][1] for a in A)
        inc_dn = sum(a["by_reg"][0] for a in A)
        say("  НАБОР «%s» (%d монет) · доля времени BTC выше SMA200 %.0f%%" % (nm, len(rs), 100 * mo_up / max(1e-9, mo_up + mo_dn)))
        for reg, lab, mo in ((True, "BTC выше SMA200", mo_up), (False, "BTC НИЖЕ SMA200", mo_dn)):
            k, s30, md, mx, op, sd = fs[reg]
            say("    %-16s полных наборов %4d (%.2f на монету в год) · застряли >%dд: %3d (%.2f на монету в год) · "
                "ждали тейка: медиана %.0f, макс %.0f дн · не вышли до сих пор %d"
                % (lab, k, k / max(1e-9, mo / 12), STUCK_D, s30, s30 / max(1e-9, mo / 12), md, mx, op))
        say("    доход закрытых циклов в месяц на монету: выше SMA200 %+.0f $ · ниже %+.0f $ (без учёта незакрытых)"
            % (inc_up / max(1e-9, mo_up), inc_dn / max(1e-9, mo_dn)))

    # ---------- часть 2 ----------
    say()
    say("=" * 110)
    say("ЧАСТЬ 2 — ВЫКЛЮЧАТЬ ЛИ. Итог = закрытые циклы + незакрытая позиция по последней цене; просадка — медиана по монетам.")
    verdict = {"B": 0, "C": 0}
    detail = {"B": [], "C": []}
    for nm, cl in SETS:
        rs = [got[x] for x in cl if x in got]
        if not rs:
            continue
        say("  НАБОР «%s» (%d монет)" % (nm, len(rs)))
        S = {}
        for v, lab in (("A", "всегда"), ("B", "только выше SMA200"), ("C", "кроме 🐻"), ("D", "B + закрыть при уходе вниз")):
            X = [r[v] for r in rs]
            tot = sum(x["tot"] for x in X)
            S[v] = (tot, med([x["mdd"] for x in X]), med([x["mdd_h"][0] for x in X]), med([x["mdd_h"][1] for x in X]),
                    sum(x["mdd"] for x in X))
            pos = sum(1 for x in X if x["tot"] > 0)
            nliq = sum(1 for x in X if x["liq"] is not None)
            yrs = sum(r["months"] for r in rs) / 12
            say("    %s %-28s итог %+9.0f $ (%+6.0f $ на монету в год) · половины %+8.0f / %+8.0f · незакрытые %+8.0f · "
                "просадка медиана %5.0f (половины %5.0f / %5.0f), худшая %5.0f · в плюсе %d/%d монет · циклов %d · ЛИКВИДАЦИЙ %d"
                % (v, lab, tot, tot / max(1e-9, yrs), sum(x["half"][0] for x in X), sum(x["half"][1] for x in X),
                   sum(x["mtm"] for x in X), S[v][1], S[v][2], S[v][3], max(x["mdd"] for x in X), pos, len(X),
                   sum(x["cyc"] for x in X), nliq))
        A = S["A"]
        for vi, v in ((0, "B"), (1, "C")):
            real_ratio = S[v][0] / max(1e-9, S[v][4])
            sh_ratios = []
            for k in range(N_SHIFT):
                tt = sum(r["shift"][k][vi]["tot"] for r in rs)
                dd = sum(r["shift"][k][vi]["mdd"] for r in rs)
                sh_ratios.append(tt / max(1e-9, dd))
            beat = sum(1 for x in sh_ratios if real_ratio > x)
            c1 = S[v][1] <= 0.75 * A[1]
            c2 = S[v][0] >= A[0] - 0.2 * abs(A[0])
            c3 = S[v][2] < A[2] and S[v][3] < A[3]
            c4 = beat >= 10
            ok = c1 and c2 and c3 and c4
            verdict[v] += ok
            detail[v].append(ok)
            say("    %s: просадка −25%% [%s] · итог не хуже 20%% [%s] · обе половины [%s] · лучше сдвигов %d/12 [%s] "
                "(итог/просадки %.2f; сдвиги %s) → %s"
                % (v, "да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет", beat, "да" if c4 else "нет",
                   real_ratio, " ".join("%.2f" % x for x in sorted(sh_ratios)), "ПРОШЁЛ" if ok else "не прошёл"))
        R = [r["rand"] for r in rs]
        say("    случайные цены (A): итог %+9.0f $ · просадка медиана %5.0f · в плюсе %d/%d монет  ← справочно"
            % (sum(x["tot"] for x in R), med([x["mdd"] for x in R]), sum(1 for x in R if x["tot"] > 0), len(R)))

    say()
    say("=" * 110)
    say("ПО МОНЕТАМ (итог $ · просадка $): A | B | C | случайные цены A")
    for nm, cl in SETS:
        for x in cl:
            if x in got:
                r = got[x]
                say("  %-7s %+7.0f · %5.0f | %+7.0f · %5.0f | %+7.0f · %5.0f | %+7.0f · %5.0f · полных наборов A %d%s"
                    % (x, r["A"]["tot"], r["A"]["mdd"], r["B"]["tot"], r["B"]["mdd"], r["C"]["tot"], r["C"]["mdd"],
                       r["rand"]["tot"], r["rand"]["mdd"], len(r["A"]["full"]),
                       (" · A ЛИКВИДАЦИЯ " + dstr(r["A"]["liq"])) if r["A"]["liq"] else ""))

    say()
    say("=" * 110)
    for v, lab in (("B", "B (только выше SMA200)"), ("C", "C (кроме 🐻)")):
        say("КРИТЕРИЙ %s: прошёл на %d из 3 наборов %s → %s"
            % (lab, verdict[v], ["да" if q else "нет" for q in detail[v]], "ПРИНЯТ" if verdict[v] >= 2 else "НЕ принят"))
    if verdict["B"] >= 2:
        say("→ ИТОГ: выключать сетку, когда BTC ниже SMA200 (вариант B)")
    elif verdict["C"] >= 2:
        say("→ ИТОГ: выключать сетку только в «🐻 медвежке» (вариант C)")
    else:
        say("→ ИТОГ: выключение по режиму BTC не доказано — оставлять как есть (A)")
    say()
    say("Время %.0f мин" % ((time.time() - t_start) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

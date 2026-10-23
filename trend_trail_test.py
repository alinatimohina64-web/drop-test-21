#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПУНКТИР AIO-49 (Trend Trail = SuperTrend ATR 10 x 3, как ta.supertrend в TradingView): на каком таймфрейме он
зарабатывает и зарабатывает ли вообще — после комиссий, против Buy & Hold и против случайного тайминга.

ДАННЫЕ: Binance spot 15m (data.binance.vision), 12 монет, с 2018-01-01 (или с листинга + 250 дней прогрева).
  Старшие ТФ (30m 1h 2h 4h 1d) собираются из 15m (UTC). Spot вместо фьючерса — ради длинной истории (цены почти те же).
ИСПОЛНЕНИЕ: сигнал по закрытию бара, вход по открытию следующего (как strategy() в TradingView).
  Комиссия 0.055% + проскальзывание 0.02% с каждой стороны. Фандинг 0.01% / 8 ч: лонг платит, шорт получает.
  Просадка — по худшей цене внутри бара (low для лонга, high для шорта). Капитал 100% в сделке, плечо x1.

ВАРИАНТЫ (объявлены до прогона, без подбора):
  A  лонг + шорт всегда (как в Pine-стратегии)
  B  только лонг
  C  только лонг, вход лишь если close > EMA200 этого ТФ
  D  лонг выше EMA200, шорт ниже EMA200
  E  C + фильтр Index % (лонг только если Index <= 70)
  F  C + фильтр Trends 4 (Bull >= Bear по пивотам прошлого дня)
ОСНОВНЫЕ ПАРАМЕТРЫ: ATR 10, множитель 3.0. Сетка 3x3 (ATR 10/14/20 x 2/3/4) — только для проверки устойчивости.

КОНТРОЛЬ «сдвиг»: та же серия позиций, циклически сдвинутая на 12 разных сроков (та же доля времени в рынке,
  та же длина сделок, но в другие даты). Если реальный тайминг не лучше сдвинутого — сигналы ничего не угадывают.

КРИТЕРИЙ (объявлен ДО прогона). Монета «прошла», если одновременно:
  1) Profit Factor >= 1.2 (по сделкам, после издержек);
  2) CAGR/MaxDD стратегии выше, чем у Buy & Hold этой монеты за тот же период;
  3) результат > 0 в ОБЕИХ половинах (до / после 2022-01-01);
  4) CAGR/MaxDD реального тайминга выше, чем у >= 10 из 12 сдвигов;
  5) сделок >= 20 (иначе шум).
  Связка «ТФ + вариант» ПРИНЯТА, если прошли >= 2/3 загрузившихся монет.
ПЛЕЧО — справочно: те же сделки с плечом x2…x100 на всю сумму; ликвидация, если цена против входа дошла до 1/L − 0.5%.
Отчёт: trend_trail_report.txt
"""
import io
import multiprocessing as mp
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

REPORT = "trend_trail_report.txt"
DATA = "data_tt"
COINS = os.environ.get("COINS", "BTC,ETH,BNB,XRP,ADA,SOL,DOGE,LTC,LINK,TRX,AVAX,DOT").split(",")
TFS = [("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("4h", 240), ("1d", 1440)]
VARS = [("A", "лонг+шорт (как Pine)"), ("B", "только лонг"), ("C", "лонг > EMA200"),
        ("D", "лонг>EMA / шорт<EMA"), ("E", "C + Index% <= 70"), ("F", "C + Trends4")]
D1 = 86400 * 1000


def ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


START = ms(2018, 1, 1)
HALF = ms(2022, 1, 1)
WARM_D = 250
FEE = 0.00055 + 0.0002
FUND8 = 0.0001
MMR = 0.005
MAIN = (10, 3.0)
GRID = [(a, m) for a in (10, 14, 20) for m in (2.0, 3.0, 4.0)]
N_SHIFT = 12
LEVS = [1, 2, 3, 5, 10, 20, 50, 100]
SYNTH = os.environ.get("SYNTH") == "1"
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def dstr(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ───────────────────────── ДАННЫЕ ─────────────────────────
def fetch(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read(), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "404"
            time.sleep(2 + 3 * k)
        except Exception:                                        # noqa: BLE001
            time.sleep(2 + 3 * k)
    return None, "fail"


def parse(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    rows = []
    for line in z.read(z.namelist()[0]).decode().splitlines():
        p = line.split(",")
        if not p or not p[0].strip().isdigit():
            continue
        rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    a = np.array(rows, dtype=float).reshape(-1, 6)
    if len(a):
        a[:, 0] = np.where(a[:, 0] > 1e14, a[:, 0] // 1000, a[:, 0])   # с 2025 Binance пишет микросекунды
    return a


def synth(sym):
    rng = np.random.default_rng(sum(ord(ch) for ch in sym))
    t = np.arange(ms(2017, 8, 17), ms(2026, 10, 20), 15 * 60000, dtype=float)
    r = rng.standard_t(4, len(t)) * 0.003 + 0.000004
    c = 4000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.002, len(t)))
    h = np.maximum(o, c) * (1 + w)
    l = np.minimum(o, c) * (1 - w)
    v = rng.uniform(10, 100, len(t))
    return np.column_stack([t, o, h, l, c, v])


def load(sym):
    if SYNTH:
        return synth(sym), []
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, sym + "_15m.npy")
    old = np.load(path) if os.path.exists(path) else np.zeros((0, 6))
    now = datetime.now(timezone.utc)
    if len(old):
        d0 = datetime.fromtimestamp(old[-1, 0] / 1000, tz=timezone.utc)
        y, m = d0.year, d0.month
    else:
        y, m = 2017, 8
    urls = []
    while (y, m) < (now.year, now.month):
        urls.append("https://data.binance.vision/data/spot/monthly/klines/%sUSDT/15m/%sUSDT-15m-%d-%02d.zip" % (sym, sym, y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    d = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    while d.date() < (now - timedelta(days=1)).date():
        urls.append("https://data.binance.vision/data/spot/daily/klines/%sUSDT/15m/%sUSDT-15m-%s.zip" % (sym, sym, d.strftime("%Y-%m-%d")))
        d += timedelta(days=1)
    with ThreadPoolExecutor(12) as ex:
        got = list(ex.map(fetch, urls))
    fails = [u.rsplit("/", 1)[1] for u, (_, st) in zip(urls, got) if st == "fail"]
    parts = [old] + [parse(b) for b, st in got if b is not None]
    a = np.vstack(parts)
    a = a[np.argsort(a[:, 0], kind="stable")]
    a = a[np.r_[True, a[1:, 0] != a[:-1, 0]]]
    np.save(path, a)
    return a, fails


def resample(a, minutes):
    if minutes == 15:
        return a
    step = minutes * 60000
    key = (a[:, 0] // step).astype(np.int64)
    idx = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    last = np.r_[idx[1:] - 1, len(a) - 1]
    out = np.column_stack([key[idx] * step, a[idx, 1], np.maximum.reduceat(a[:, 2], idx),
                           np.minimum.reduceat(a[:, 3], idx), a[last, 4], np.add.reduceat(a[:, 5], idx)])
    if last[-1] - idx[-1] + 1 < minutes // 15:                  # последний бар не закрыт
        out = out[:-1]
    return out


# ───────────────────────── ИНДИКАТОРЫ ─────────────────────────
def ewm(x, alpha, seed_n=1):
    x = np.nan_to_num(np.asarray(x, dtype=float)).tolist()
    n = len(x)
    out = [float("nan")] * n
    if n < seed_n:
        return np.array(out)
    v = sum(x[:seed_n]) / seed_n
    out[seed_n - 1] = v
    b = 1 - alpha
    for i in range(seed_n, n):
        v = alpha * x[i] + b * v
        out[i] = v
    return np.array(out)


def rma(x, n):
    return ewm(x, 1.0 / n, n)


def ema(x, n):
    return ewm(x, 2.0 / (n + 1), 1)


def atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return rma(tr, n)


def supertrend_dir(h, l, c, n, mult):
    """Копия ta.supertrend TradingView. direction: -1 = вверх (пунктир снизу), 1 = вниз (пунктир сверху)."""
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    loB, upB = (hl2 - mult * a).tolist(), (hl2 + mult * a).tolist()
    cl, av = c.tolist(), a.tolist()
    N = len(cl)
    d = [1] * N
    pl = pu = 0.0
    pst = None
    for i in range(N):
        lo, up = loB[i], upB[i]
        if lo != lo:
            lo = up = 0.0
        cp = cl[i - 1] if i else 0.0
        lo = lo if (lo > pl or cp < pl) else pl
        up = up if (up < pu or cp > pu) else pu
        if i == 0 or av[i - 1] != av[i - 1]:
            di = 1
        elif pst == pu:
            di = -1 if cl[i] > up else 1
        else:
            di = 1 if cl[i] < lo else -1
        d[i] = di
        pst = lo if di == -1 else up
        pl, pu = lo, up
    return np.array(d)


def pad(x, n):
    return np.r_[np.full(n - len(x), np.nan), x]


def roll_max(x, n):
    return pad(swv(x, n).max(1), len(x))


def roll_min(x, n):
    return pad(swv(x, n).min(1), len(x))


def roll_sum(x, n):
    return pad(swv(x, n).sum(1), len(x))


def stoch(c, h, l, n):
    hh, ll = roll_max(h, n), roll_min(l, n)
    with np.errstate(all="ignore"):
        return 100 * (c - ll) / (hh - ll)


def norm(x):
    return np.clip(np.nan_to_num(x, nan=50.0, posinf=50.0, neginf=50.0), 0, 100)


def index_pct(o, h, l, c, v, n=14):
    """Index inside %: среднее 8 нормированных осцилляторов (как в AIO-49)."""
    with np.errstate(all="ignore"):
        ch = np.diff(c, prepend=c[0])
        g, ls = rma(np.maximum(ch, 0), n), rma(np.maximum(-ch, 0), n)
        rsi = np.where(ls == 0, 100.0, 100 - 100 / (1 + g / ls))
        st = pad(swv(np.nan_to_num(stoch(c, h, l, n), nan=50.0), 3).mean(1), len(c))
        W = swv(c, n)
        tt = np.arange(n) - (n - 1) / 2
        Wc = W - W.mean(1)[:, None]
        corr = pad((Wc * tt).sum(1) / np.sqrt((Wc ** 2).sum(1) * (tt ** 2).sum()), len(c))
        wpr = stoch(c, h, l, n)
        W1 = swv(c, n + 1)
        prk = pad((W1[:, :-1] <= W1[:, -1:]).sum(1) / n * 100.0, len(c))
        tp = (h + l + c) / 3
        dtp = np.diff(tp, prepend=tp[0])
        mf = v * tp
        up, dn = roll_sum(np.where(dtp > 0, mf, 0.0), n), roll_sum(np.where(dtp < 0, mf, 0.0), n)
        mfi = np.where(dn == 0, 100.0, 100 - 100 / (1 + up / dn))
        mh = ema(c, 12) - ema(c, 26)
        mh = mh - ema(mh, 9)
        macdN = stoch(mh, mh, mh, 50)
        haC = ((o + h + l + c) / 4).tolist()
        ol, cl_ = o.tolist(), c.tolist()
        haO = [0.0] * len(haC)
        haO[0] = (ol[0] + cl_[0]) / 2
        for i in range(1, len(haC)):
            haO[i] = (haO[i - 1] + haC[i - 1]) / 2
        haO, haC = np.array(haO), np.array(haC)
        jap = stoch(haC, np.maximum(h, np.maximum(haO, haC)), np.minimum(l, np.minimum(haO, haC)), n)
        s = (norm(rsi) + norm(st) + norm((corr + 1) * 50) + norm(wpr) + norm(prk) + norm(mfi) + norm(macdN) + norm(jap)) / 8
    return np.clip(s, 1, 100)


def trends4(t, c, daily):
    """Голоса Traditional/Fibonacci/Camarilla/Woodie по пивотам ПРОШЛОГО дня -> (bull, bear)."""
    dk = (daily[:, 0] // D1).astype(np.int64)
    want = (t // D1).astype(np.int64) - 1
    j = np.minimum(np.searchsorted(dk, want), len(dk) - 1)
    ok = dk[j] == want
    H, L, C = daily[j, 2], daily[j, 3], daily[j, 4]
    R = H - L
    P = (H + L + C) / 3
    Wd = (H + L + 2 * C) / 4

    def vote(p, r1, s1):
        return np.where(c > (p + r1) / 2, 1, np.where(c < (p + s1) / 2, -1, 0))
    vs = [vote(P, 2 * P - L, 2 * P - H), vote(P, P + 0.382 * R, P - 0.382 * R),
          vote(C, C + R * 1.1 / 12, C - R * 1.1 / 12), vote(Wd, 2 * Wd - L, 2 * Wd - H)]
    bull = sum((x == 1).astype(int) for x in vs) * ok
    bear = sum((x == -1).astype(int) for x in vs) * ok
    return bull, bear


# ───────────────────────── БЭКТЕСТ ─────────────────────────
def positions(d, lok, sok):
    """Как в Pine: на BUY — лонг (если фильтр разрешает) или вне рынка; на SELL — шорт или вне рынка."""
    prev = np.r_[0, d[:-1]]
    buy = (d == -1) & (prev == 1)
    sell = (d == 1) & (prev == -1)
    tgt = np.full(len(d), np.nan)
    tgt[buy] = np.where(lok[buy], 1.0, 0.0)
    tgt[sell] = np.where(sok[sell], -1.0, 0.0)
    tgt[0] = 0.0 if np.isnan(tgt[0]) else tgt[0]
    idx = np.maximum.accumulate(np.where(np.isnan(tgt), 0, np.arange(len(d))))
    return tgt[idx]


def run(t, o, h, l, c, p, i0, bar_h, funding=True):
    """p[i] — позиция после закрытия бара i; действует с открытия i+1."""
    p = p.copy()
    p[:i0] = 0
    eff = np.r_[0.0, p[:-1]]
    ret = np.r_[o[1:] / o[:-1] - 1, c[-1] / o[-1] - 1]
    chg = np.abs(np.diff(np.r_[0.0, eff]))
    fund = eff * FUND8 * bar_h / 8 if funding else 0.0
    f = 1 + eff * ret - chg * FEE - fund
    worst = np.minimum(eff * (l / o - 1), eff * (h / o - 1))
    fw = 1 + worst - chg * FEE - fund
    fw = fw[i0:]
    E = np.cumprod(f[i0:])
    Ep = np.r_[1.0, E[:-1]]
    pk = np.maximum.accumulate(np.r_[1.0, E])[:-1]
    mdd = float(np.max(1 - np.minimum(Ep * fw, E) / pk))
    ih = int(np.searchsorted(t, HALF))
    h1 = float(np.prod(f[i0:max(i0, ih)])) - 1
    h2 = float(np.prod(f[max(i0, ih):])) - 1
    yrs = (t[-1] - t[i0]) / (365.25 * D1)
    fin = float(E[-1])
    cagr = fin ** (1 / yrs) - 1 if fin > 0 else -1.0
    return {"ret": fin - 1, "cagr": cagr, "mdd": mdd, "mar": cagr / max(mdd, 1e-9), "h1": h1, "h2": h2,
            "expo": float(np.mean(eff[i0:] != 0)), "eff": eff, "retv": ret}


def trades(t, o, h, l, eff, ret, bar_h):
    n = len(eff)
    b = np.flatnonzero(np.r_[True, eff[1:] != eff[:-1]])
    e = np.r_[b[1:], n]
    out = []
    for s, en in zip(b, e):
        side = eff[s]
        if side == 0:
            continue
        gross = float(np.prod(1 + side * ret[s:en])) - 1
        net = (1 + gross) * (1 - FEE) ** 2 - 1 - side * FUND8 * bar_h / 8 * (en - s)
        ent = o[s]
        adv = 1 - l[s:en].min() / ent if side > 0 else h[s:en].max() / ent - 1
        fav = h[s:en].max() / ent - 1 if side > 0 else 1 - l[s:en].min() / ent
        out.append((int(t[s]), side, net, adv, fav))
    return out


def trade_stats(tr):
    if not tr:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "avg": 0.0, "fav": 0.0, "adv": 0.0}
    net = np.array([x[2] for x in tr])
    w, ls = net[net > 0].sum(), -net[net < 0].sum()
    return {"n": len(tr), "wr": float((net > 0).mean()), "pf": float(w / ls) if ls > 0 else 99.0,
            "avg": float(net.mean()), "fav": float(np.mean([x[4] for x in tr])), "adv": float(max(x[3] for x in tr))}


def lev_sim(tr, L):
    eq, pk, mdd = 1.0, 1.0, 0.0
    for t, side, net, adv, fav in tr:
        if adv >= 1.0 / L - MMR:
            return 0.0, 1.0, t
        mdd = max(mdd, 1 - eq * (1 - L * adv) / pk)
        eq *= max(0.0, 1 + L * net)
        if eq <= 0:
            return 0.0, 1.0, t
        pk = max(pk, eq)
        mdd = max(mdd, 1 - eq / pk)
    return eq, mdd, None


def work(sym):
    t_s = time.time()
    try:
        base, fails = load(sym)
    except Exception as e:                                       # noqa: BLE001
        return sym, None, "ошибка загрузки: %s" % e
    if len(base) < 20000:
        return sym, None, "мало данных (%d баров)" % len(base)
    gaps = int(np.sum(np.diff(base[:, 0]) > D1))
    daily = resample(base, 1440)
    t_start = max(START, int(base[0, 0]) + WARM_D * D1)
    res = {"tf": {}, "fails": fails, "gaps": gaps, "t0": t_start, "t1": int(base[-1, 0])}
    for tfn, mins in TFS:
        a = resample(base, mins)
        t, o, h, l, c, v = (a[:, k] for k in range(6))
        i0 = int(np.searchsorted(t, t_start))
        bar_h = mins / 60
        e200 = ema(c, 200)
        idx = index_pct(o, h, l, c, v)
        bull, bear = trends4(t, c, daily)
        T, F = np.ones(len(c), bool), np.zeros(len(c), bool)
        above = c > e200
        filt = {"A": (T, T), "B": (T, F), "C": (above, F), "D": (above, ~above),
                "E": (above & (idx <= 70), F), "F": (above & (bull >= bear), F)}
        bh = run(t, o, h, l, c, np.ones(len(c)), i0, bar_h, funding=False)
        R = {"bh": {k: bh[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2")}, "bars": len(c) - i0}
        for prm in GRID:
            d = supertrend_dir(h, l, c, *prm)
            for vn, _ in VARS:
                p = positions(d, *filt[vn])
                r = run(t, o, h, l, c, p, i0, bar_h)
                tr = trades(t, o, h, l, r["eff"], r["retv"], bar_h)
                x = {k: r[k] for k in ("ret", "cagr", "mdd", "mar", "h1", "h2", "expo")}
                x.update(trade_stats(tr))
                if prm == MAIN:
                    pv = p[i0:]
                    sh = []
                    for k in range(1, N_SHIFT + 1):
                        p2 = p.copy()
                        p2[i0:] = np.roll(pv, int(len(pv) * k / (N_SHIFT + 1)))
                        sh.append(run(t, o, h, l, c, p2, i0, bar_h)["mar"])
                    x["beat"] = int(sum(1 for s in sh if x["mar"] > s))
                    x["lev"] = {L: lev_sim(tr, L) for L in LEVS}
                    x["liq"] = {L: sum(1 for q in tr if q[3] >= 1.0 / L - MMR) for L in LEVS}
                R[(vn, prm)] = x
        res["tf"][tfn] = R
    return sym, res, "%.0f с, %d баров 15m, пропусков >1д: %d, не скачалось файлов: %d%s" % (
        time.time() - t_s, len(base), gaps, len(fails), (" (" + ", ".join(fails[:5]) + ")") if fails else "")


def passed(x, bh, use_shift=True):
    return [x["pf"] >= 1.2, x["mar"] > bh["mar"], x["h1"] > 0 and x["h2"] > 0,
            (x.get("beat", 0) >= 10) if use_shift else True, x["n"] >= 20]


def med(v):
    return float(np.median(v)) if len(v) else 0.0


def main():
    t0 = time.time()
    say("ПУНКТИР AIO-49 (SuperTrend) · %s · Binance spot 15m → %s · с 2018-01-01%s" % (
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), " ".join(x for x, _ in TFS),
        " · !!! СИНТЕТИКА (SYNTH=1) — цифры ничего не значат" if SYNTH else ""))
    say("Издержки %.3f%% на сторону (комиссия + проскальзывание), фандинг %.2f%% / 8ч · основные параметры ATR %d × %.1f"
        % (FEE * 100, FUND8 * 100, MAIN[0], MAIN[1]))
    got = {}
    with mp.get_context("fork").Pool(max(1, min(len(COINS), os.cpu_count() or 2))) as pool:
        for sym, res, msg in pool.imap_unordered(work, COINS):
            say("  %s: %s" % (sym, msg))
            if res is not None:
                got[sym] = res
    coins = [s for s in COINS if s in got]
    if not coins:
        say("НЕТ ДАННЫХ — проверь доступ к data.binance.vision")
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(_out) + "\n")
        return
    need = int(np.ceil(len(coins) * 2 / 3))
    say("Монет: %d · для принятия нужно >= %d · периоды: %s" % (len(coins), need, ", ".join(
        "%s %s→%s" % (s, dstr(got[s]["t0"]), dstr(got[s]["t1"])) for s in coins)))

    say()
    say("=" * 118)
    say("ЧАСТЬ 1 — СВОДКА ПО МОНЕТАМ (ATR %d × %.1f). Медианы по монетам. MAR = CAGR / MaxDD." % MAIN)
    say("  «по условиям» — сколько монет прошло каждое: 1 PF>=1.2 / 2 MAR>B&H / 3 обе половины>0 / 4 сдвиги>=10 / 5 сделок>=20")
    say("  %-4s %-22s %7s %5s %5s %8s %6s %6s %7s %6s %6s  %s" % ("ТФ", "вариант", "прошли", "PF", "WR", "CAGR", "MaxDD",
        "MAR", "MAR B&H", "сделок", "сдвиг", "по условиям"))
    accepted = []
    for tfn, _ in TFS:
        for vn, vl in VARS:
            X = [(got[s]["tf"][tfn][(vn, MAIN)], got[s]["tf"][tfn]["bh"]) for s in coins]
            cs = [passed(x, b) for x, b in X]
            npass = sum(all(q) for q in cs)
            per = "/".join(str(sum(q[k] for q in cs)) for k in range(5))
            ok = npass >= need
            if ok:
                accepted.append((tfn, vn))
            say("  %-4s %s %-20s %3d/%-3d %5.2f %4.0f%% %+7.1f%% %5.1f%% %6.2f %7.2f %6.0f %6.1f  %s%s" % (
                tfn, vn, vl, npass, len(coins), med([x["pf"] for x, _ in X]), 100 * med([x["wr"] for x, _ in X]),
                100 * med([x["cagr"] for x, _ in X]), 100 * med([x["mdd"] for x, _ in X]), med([x["mar"] for x, _ in X]),
                med([b["mar"] for _, b in X]), med([x["n"] for x, _ in X]), med([x["beat"] for x, _ in X]), per,
                "  ← ПРИНЯТ" if ok else ""))
        say("  " + "-" * 112)

    say()
    say("=" * 118)
    say("ЧАСТЬ 2 — BTC ПОДРОБНО (ATR %d × %.1f)" % MAIN)
    if "BTC" in got:
        B = got["BTC"]["tf"]
        for tfn, _ in TFS:
            bh = B[tfn]["bh"]
            say("  %s · Buy&Hold: итог %+.0f%% · CAGR %+.1f%% · MaxDD %.1f%% · MAR %.2f · половины %+.0f%% / %+.0f%%" % (
                tfn, 100 * bh["ret"], 100 * bh["cagr"], 100 * bh["mdd"], bh["mar"], 100 * bh["h1"], 100 * bh["h2"]))
            for vn, vl in VARS:
                x = B[tfn][(vn, MAIN)]
                say("     %s %-20s сделок %4d · WR %3.0f%% · PF %5.2f · итог %+8.0f%% · CAGR %+6.1f%% · MaxDD %5.1f%% · MAR %5.2f · "
                    "половины %+6.0f%% / %+6.0f%% · в рынке %3.0f%% · сдвиги %2d/12 · худший ход против %.1f%%" % (
                        vn, vl, x["n"], 100 * x["wr"], x["pf"], 100 * x["ret"], 100 * x["cagr"], 100 * x["mdd"], x["mar"],
                        100 * x["h1"], 100 * x["h2"], 100 * x["expo"], x["beat"], 100 * x["adv"]))

        say()
        say("=" * 118)
        say("ЧАСТЬ 3 — ПЛЕЧО НА BTC (все деньги в каждой сделке). «3.2x dd40%» = счёт вырос в 3.2 раза, просадка 40%; ЛИКВ = обнулён")
        say("  %-4s %-3s " % ("ТФ", "вар") + " ".join("%15s" % ("x%d" % L) for L in LEVS))
        for tfn, _ in TFS:
            for vn in ("A", "C"):
                x = B[tfn][(vn, MAIN)]
                cells = []
                for L in LEVS:
                    eq, dd, rt = x["lev"][L]
                    cells.append("%15s" % (("ЛИКВ " + dstr(rt)) if rt else "%.2fx dd%.0f%%" % (eq, 100 * dd)))
                say("  %-4s %-3s " % (tfn, vn) + " ".join(cells))
        say("  сделок A, выбитых бы ликвидацией x100/x50/x20/x10: " + " · ".join(
            "%s %s" % (tfn, "/".join(str(B[tfn][("A", MAIN)]["liq"][L]) for L in (100, 50, 20, 10))) for tfn, _ in TFS))

    say()
    say("=" * 118)
    say("ЧАСТЬ 4 — УСТОЙЧИВОСТЬ К ПАРАМЕТРАМ (ATR × множитель): медианный PF · сколько монет прошли условия 1,2,3,5")
    for vn in ("A", "B", "C", "D"):
        say("  вариант %s:" % vn)
        say("    %-4s " % "ТФ" + " ".join("%11s" % ("%d×%.0f" % g) for g in GRID))
        for tfn, _ in TFS:
            cells = []
            for g in GRID:
                X = [(got[s]["tf"][tfn][(vn, g)], got[s]["tf"][tfn]["bh"]) for s in coins]
                np_ = sum(all(passed(x, b, use_shift=False)) for x, b in X)
                cells.append("%11s" % ("%.2f %d/%d" % (med([x["pf"] for x, _ in X]), np_, len(coins))))
            say("    %-4s " % tfn + " ".join(cells))

    say()
    say("=" * 118)
    say("ПО МОНЕТАМ (итог % · MaxDD % · PF) по ТФ: " + " | ".join(t for t, _ in TFS))
    for s in coins:
        for vn in ("A", "C"):
            say("  %-5s %-3s " % (s, vn) + " | ".join("%+7.0f %3.0f %4.2f" % (
                100 * got[s]["tf"][t][(vn, MAIN)]["ret"], 100 * got[s]["tf"][t][(vn, MAIN)]["mdd"],
                got[s]["tf"][t][(vn, MAIN)]["pf"]) for t, _ in TFS))
        say("  %-5s B&H " % s + " | ".join("%+7.0f %3.0f     " % (100 * got[s]["tf"][t]["bh"]["ret"],
                                                              100 * got[s]["tf"][t]["bh"]["mdd"]) for t, _ in TFS))

    say()
    say("=" * 118)
    if accepted:
        say("→ ИТОГ: критерий прошли: " + ", ".join("%s/%s" % a for a in accepted))
        say("  Проверь в части 4, что соседние параметры тоже держатся — иначе это удачная точка, а не система.")
    else:
        say("→ ИТОГ: ни одна связка «ТФ + вариант» не прошла критерий — пунктир как самостоятельная система не доказан.")
    say("Время %.1f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

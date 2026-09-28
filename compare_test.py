#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СРАВНЕНИЕ: КТО ЛУЧШЕ ИЩЕТ ВХОД И ВЕДЁТ СДЕЛКУ — бот или индикатор.
Один и тот же год, одни и те же монеты (топ-50 OKX + 10 монет первого прогона, без BTC),
одинаковый риск на стопе, одна сделка за раз — как торгует человек.

ВАРИАНТЫ
  1 БОТ        — вход робота 🟢 (1Д/4Ч/1Ч в одну сторону, ADX 4Ч >= 25, MRC 4Ч не на краю,
                 цена по ту же сторону 200Д, 1Ч только что развернулся) + сетка GHOST B:
                 4 ордера по 300 $, шаг = ATR 1ч (не меньше 1%), тейк 2.5 шага от средней,
                 стоп 5.2 шага от первого входа, трейлинг 50/50/50 после 1-й доливки.
  2 БОТ+BTC1Д  — то же, но только входы по направлению дневного тренда BTC.
  3 ИНДИКАТОР  — сигналы Trand-Test-2 на 1h (основной + откат, как в ind_btc_test.py),
                 только по дневному тренду BTC (связка, прошедшая критерий на 36 новых
                 монетах). Один вход 300 $, цель 2.5 × ATR 1ч, стоп 5.2 × ATR 1ч.
  4 ГИБРИД     — входы индикатора (как в 3) + сетка бота (как в 1).
  Риск на стопе одинаковый: 300 $ × 5.2 × ATR 1ч (у бота — если стоп до 1-й доливки).

ИСПОЛНЕНИЕ: вход по открытию 5-минутки после закрытия сигнальной часовой свечи; дальше всё
на 5m: сначала проверяется стоп/трейлинг, потом доливки и тейк (спорная свеча — в худшую
сторону). Комиссия рынком 0.055%, тейк лимиткой 0.02%, фандинг 0.01% / 8 ч.

КАК ТОРГУЕТ ЧЕЛОВЕК: 1000 сценариев — одна сделка за раз по всем монетам; если в один
час несколько сигналов, берётся случайный; 30% сигналов пропущено (сон, дела).

КРИТЕРИЙ (объявлен ДО прогона). Вариант 2, 3 или 4 лучше текущего бота (1), если:
  1) медиана итога в $ выше, чем у бота, в ОБЕИХ половинах года;
  2) медианная просадка не больше, чем у бота.
Отдельно показывается доля сценариев в плюсе.

Файл самостоятельный. Кэш свечей — data_ind/ (тот же, что у ind_btc_test.py).
Переменные: DAYS (по умолчанию 365). Отчёт: compare_report.txt
"""
import bisect
import gzip
import json
import math
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

DAYS = int(os.environ.get("DAYS") or 365)
DATA_DIR = "data_ind"
REPORT = "compare_report.txt"
M5, M15, H1, H4, D1 = 300000, 900000, 3600000, 14400000, 86400000
VOL = 300.0
N_ORD = 4
MIN_STEP = 1.0
F_TAKER, F_MAKER, FUND = 0.055 / 100, 0.02 / 100, 0.01 / 100
KEEPS = (0.5, 0.5, 0.5)
ADX_BOT = 25
SCEN, SKIP = 1000, 0.3
PI = math.pi

FIRST_RUN = ["BTC", "ETH", "SOL", "XRP", "DOGE", "LINK", "SUI", "ADA", "AVAX", "BNB"]
EXCLUDE = {"USDC", "USDE", "DAI", "FDUSD", "TUSD", "PYUSD", "USDT", "USD1", "RLUSD",
           "XAU", "XAG", "PAXG", "XAUT", "XPT", "XPD", "TRUMP",
           "SOXL", "SNDK", "TSLA", "NVDA", "AAPL", "MSTR", "INTC", "SPCX", "MU", "AMZN", "GOOGL",
           "GOOG", "META", "MSFT", "NFLX", "AMD", "PLTR", "COIN", "HOOD", "CRCL", "GME", "AMC",
           "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "ORCL", "AVGO", "TSM", "BABA", "SMCI", "ARM",
           "CRWV", "IBM", "UBER", "DIS", "NKE", "JPM", "V", "MA", "BRKB", "COST", "WMT", "KO",
           "PEP", "XOM", "CVX", "BA", "GS", "LLY", "UNH", "MRVL", "ASML", "QCOM", "SNOW", "SHOP",
           "SKHYNIX", "SKHY"}

_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


# ---------------------------------------------------------------- данные

def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception:                                       # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("OKX не отвечает")


def top_coins(n=50):
    rows = []
    for r in http_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP").get("data") or []:
        inst = r.get("instId", "")
        if not inst.endswith("-USDT-SWAP"):
            continue
        base = inst.split("-")[0]
        if base in EXCLUDE:
            continue
        try:
            rows.append((float(r.get("volCcy24h") or 0) * float(r.get("last") or 0), base))
        except (TypeError, ValueError):
            continue
    rows.sort(reverse=True)
    return [b for _, b in rows[:n]]


def fetch(sym, bar, ms, t_from, t_stop=0):
    """История OKX [t, o, h, l, c, объём] от сейчас назад до t_from (или t_stop — есть в кэше)."""
    inst = "%s-USDT-SWAP" % sym
    stop = max(t_from, t_stop)
    got = {}
    for host in ("https://www.okx.com", "https://aws.okx.com"):
        after = None
        try:
            while True:
                url = "%s/api/v5/market/history-candles?instId=%s&bar=%s&limit=100" % (host, inst, bar)
                if after is not None:
                    url += "&after=%d" % after
                rows = http_json(url).get("data") or []
                if not rows:
                    break
                for r in rows:
                    t = int(r[0])
                    vol = float(r[7]) if len(r) > 7 and r[7] not in ("", None) else float(r[5] or 0)
                    got[t] = [t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), vol]
                oldest = min(int(r[0]) for r in rows)
                if oldest <= stop or (after is not None and oldest >= after):
                    break
                after = oldest
                time.sleep(0.12)
            break
        except Exception as e:                                  # noqa: BLE001
            print("  ! %s %s %s: %s" % (sym, bar, host, e), flush=True)
    now = int(time.time() * 1000)
    return sorted(v for v in got.values() if v[0] + ms <= now)


def load(sym, bar, ms, days):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "%s_%s.json.gz" % (sym, bar))
    t_from = int(time.time() * 1000) - days * D1
    old = []
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt") as f:
                old = json.load(f)
        except Exception:                                       # noqa: BLE001
            old = []
    if old and len(old[0]) == 6 and old[0][0] <= t_from + ms:
        new = fetch(sym, bar, ms, t_from, old[-1][0])
    else:
        old, new = [], fetch(sym, bar, ms, t_from)
    rows = {r[0]: r for r in old}
    rows.update({r[0]: r for r in new})
    bars = sorted(r for r in rows.values() if r[0] >= t_from)
    with gzip.open(path, "wt") as f:
        json.dump(bars, f)
    return bars


def load_coin(sym):
    b5 = load(sym, "5m", M5, DAYS)
    h4 = load(sym, "4H", H4, DAYS + 70)
    d1 = load(sym, "1Dutc", D1, DAYS + 230)
    print("  %s: 5m %d, 4Ч %d, 1Д %d" % (sym, len(b5), len(h4), len(d1)), flush=True)
    return sym, (b5, h4, d1)


def agg(bars, ms):
    out, cur = [], None
    for t, o, h, l, c, v in bars:
        k = t - t % ms
        if cur is None or cur[0] != k:
            if cur:
                out.append(cur)
            cur = [k, o, h, l, c, v]
        else:
            cur[2] = max(cur[2], h)
            cur[3] = min(cur[3], l)
            cur[4] = c
            cur[5] += v
    return out


# ---------------------------------------------------------------- индикаторы (как у робота и индикатора)

def ema(x, n):
    a, e, out = 2 / (n + 1), None, []
    for v in x:
        e = v if e is None else a * v + (1 - a) * e
        out.append(e)
    return out


def rma(x, n):
    r, out = None, []
    for v in x:
        r = v if r is None else (r * (n - 1) + v) / n
        out.append(r)
    return out


def sma(x, n):
    out, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        out.append(s / n if i >= n - 1 else None)
    return out


def trs(b):
    return [b[0][2] - b[0][3]] + [max(b[i][2] - b[i][3], abs(b[i][2] - b[i - 1][4]),
                                      abs(b[i][3] - b[i - 1][4])) for i in range(1, len(b))]


def atr_pct(b, n=14):
    a = rma(trs(b), n)
    return [a[i] / b[i][4] * 100 if b[i][4] else 0 for i in range(len(b))]


def adx_arr(b, n=14):
    tr, pdm, mdm = trs(b), [0.0], [0.0]
    for i in range(1, len(b)):
        up, dn = b[i][2] - b[i - 1][2], b[i - 1][3] - b[i][3]
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)
    at, p, m = rma(tr, n), rma(pdm, n), rma(mdm, n)
    dx = []
    for i in range(len(b)):
        pi_ = 100 * p[i] / at[i] if at[i] else 0
        mi_ = 100 * m[i] / at[i] if at[i] else 0
        dx.append(100 * abs(pi_ - mi_) / (pi_ + mi_) if pi_ + mi_ else 0)
    return rma(dx, n)


def trend_arr(b):
    """Тренд робота/карты/фильтра BTC: цена > EMA50 и EMA21 > EMA50 (и зеркально)."""
    c = [x[4] for x in b]
    ef, es = ema(c, 21), ema(c, 50)
    return [0 if i < 59 else 1 if (c[i] > es[i] and ef[i] > es[i]) else -1 if (c[i] < es[i] and ef[i] < es[i]) else 0
            for i in range(len(b))]


def rsi(c, n=14):
    up = [0.0] + [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    dn = [0.0] + [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    ru, rd = rma(up, n), rma(dn, n)
    return [100.0 if rd[i] == 0 else 100 - 100 / (1 + ru[i] / rd[i]) for i in range(len(c))]


def supersmoother(src, n):
    a1 = math.exp(-math.sqrt(2) * PI / n)
    b1 = 2 * a1 * math.cos(math.sqrt(2) * PI / n)
    c3, c2 = -a1 * a1, b1
    c1 = 1 - c2 - c3
    out = []
    for i, v in enumerate(src):
        s1 = out[i - 1] if i >= 1 else src[0]
        s2 = out[i - 2] if i >= 2 else (src[i - 1] if i >= 1 else src[0])
        out.append(c1 * v + c2 * s1 + c3 * s2)
    return out


def mrc_zone(b, n=200):
    if len(b) < n:
        return None
    mean = supersmoother([(x[2] + x[3] + x[4]) / 3 for x in b], n)[-1]
    rng = supersmoother(trs(b), n)[-1]
    c = b[-1][4]
    if c >= mean + rng * PI * 2.415:
        return "перекуп"
    if c >= mean + rng * PI:
        return "верх"
    if c <= mean - rng * PI * 2.415:
        return "перепрод"
    if c <= mean - rng * PI:
        return "низ"
    return "середина"


# ---------------------------------------------------------------- входы

def bot_entries(h1, h4, d1):
    """Входы робота 🟢: (время закрытия часа, сторона, индекс часа)."""
    tr1, tr4, trd, adx4 = trend_arr(h1), trend_arr(h4), trend_arr(d1), adx_arr(h4)
    dcl = [x[4] for x in d1]
    pre = [0.0]
    for v in dcl:
        pre.append(pre[-1] + v)
    out, j4, jd = [], -1, -1
    for i in range(60, len(h1)):
        T = h1[i][0] + H1
        while j4 + 1 < len(h4) and h4[j4 + 1][0] + H4 <= T:
            j4 += 1
        while jd + 1 < len(d1) and d1[jd + 1][0] + D1 <= T:
            jd += 1
        if j4 < 298 or jd < 199:
            continue
        d = tr1[i]
        if d == 0 or d == tr1[i - 1] or tr4[j4] != d or trd[jd] != d or adx4[j4] < ADX_BOT:
            continue
        s200 = (pre[jd + 1] - pre[jd - 199]) / 200
        if (d == 1 and not dcl[jd] > s200) or (d == -1 and dcl[jd] > s200):
            continue
        z = mrc_zone(h4[j4 - 298:j4 + 1])
        if (d == 1 and z == "перекуп") or (d == -1 and z == "перепрод"):
            continue
        out.append((T, d, i))
    return out


def ind_entries(b, h4_t, h4_up):
    """Сигналы индикатора Trand-Test-2 на часовике: (время закрытия, сторона, индекс)."""
    n = len(b)
    c = [x[4] for x in b]
    ef, es = ema(c, 9), ema(c, 21)
    r = rsi(c)
    av = sma([x[5] for x in b], 20)
    adx = adx_arr(b)
    out = []
    last_up = last_dn = last_sig = last_pb = None
    j4 = -1
    for i in range(1, n):
        if ef[i] > es[i] and ef[i - 1] <= es[i - 1]:
            last_up = i
        if ef[i] < es[i] and ef[i - 1] >= es[i - 1]:
            last_dn = i
        if i < 60 or av[i] is None:
            continue
        close_t = b[i][0] + H1
        while j4 + 1 < len(h4_t) and h4_t[j4 + 1] + H4 <= close_t:
            j4 += 1
        if j4 < 25:
            continue
        htf_up = h4_up[j4]
        bs_up = None if last_up is None else i - last_up
        bs_dn = None if last_dn is None else i - last_dn
        conf_up = bs_up == 2 and (bs_dn is None or bs_dn > 2)
        conf_dn = bs_dn == 2 and (bs_up is None or bs_up > 2)
        adx_ok = adx[i] >= 20
        vol_ok = b[i][5] > av[i] * 1.3
        long_raw = conf_up and r[i] > 50 and r[i] > r[i - 1] and vol_ok and adx_ok and htf_up
        short_raw = conf_dn and r[i] < 50 and r[i] < r[i - 1] and vol_ok and adx_ok and not htf_up
        cd_ok = last_sig is None or i - last_sig >= 5
        long_sig, short_sig = long_raw and cd_ok, short_raw and cd_ok
        if long_sig or short_sig:
            last_sig = i
            out.append((close_t, 1 if long_sig else -1, i))
        o, h, l = b[i][1], b[i][2], b[i][3]
        pb_long = (ef[i] > es[i] and adx_ok and htf_up and l <= ef[i] and c[i] > ef[i] and c[i] > o
                   and r[i] > r[i - 1])
        pb_short = (ef[i] < es[i] and adx_ok and not htf_up and h >= ef[i] and c[i] < ef[i] and c[i] < o
                    and r[i] < r[i - 1])
        pb_ok = last_pb is None or i - last_pb >= 10
        pl, ps = pb_long and pb_ok and not long_sig, pb_short and pb_ok and not short_sig
        if pl or ps:
            last_pb = i
            out.append((close_t, 1 if pl else -1, i))
    return out


# ---------------------------------------------------------------- сделки на 5m

def sim_grid(b5, k0, d, step):
    s = step / 100
    tp, sl = 2.5 * s, 5.2 * s
    p0 = b5[k0][1]
    prices = [p0 * (1 + d * j * s) for j in range(N_ORD)]
    qty, cost, n = VOL / p0, VOL, 1
    fees, fund = VOL * F_TAKER, 0.0
    sl_px, peak = p0 * (1 - d * sl), None

    def done(px, k, maker=False):
        return d * (qty * px - cost) - fees - qty * px * (F_MAKER if maker else F_TAKER) + fund, b5[k][0] + M5

    k = k0
    while k < len(b5):
        t, o, h, l, c = b5[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        avg = cost / qty
        stop = sl_px
        if n >= 2:
            trl = avg + KEEPS[n - 2] * (peak - avg)
            if d * trl > d * stop:
                stop = trl
        if d * o <= d * stop:
            return done(o, k)
        if d * adv <= d * stop:
            return done(stop, k)
        while n < N_ORD and d * fav >= d * prices[n]:
            px = o if d * o > d * prices[n] else prices[n]
            qty += VOL / px
            cost += VOL
            fees += VOL * F_TAKER
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
            trl = avg + KEEPS[n - 2] * (peak - avg)
            if d * c <= d * trl:
                return done(trl, k)
        if (t + M5) % (8 * H1) == 0:
            fund -= d * FUND * qty * c
        k += 1
    return done(b5[-1][4], len(b5) - 1)


def sim_single(b5, k0, d, atrp):
    tp, sl = 2.5 * atrp / 100, 5.2 * atrp / 100
    p0 = b5[k0][1]
    qty = VOL / p0
    tgt, stp = p0 * (1 + d * tp), p0 * (1 - d * sl)
    fund = 0.0

    def done(px, k, maker=False):
        return (d * (qty * px - VOL) - VOL * F_TAKER - qty * px * (F_MAKER if maker else F_TAKER) + fund,
                b5[k][0] + M5)

    for k in range(k0, len(b5)):
        t, o, h, l, c = b5[k][:5]
        fav, adv = (h, l) if d == 1 else (l, h)
        if d * o <= d * stp:
            return done(o, k)
        if d * adv <= d * stp:
            return done(stp, k)
        if d * fav >= d * tgt:
            return done(o if d * o > d * tgt else tgt, k, True)
        if (t + M5) % (8 * H1) == 0:
            fund -= d * FUND * qty * c
    return done(b5[-1][4], len(b5) - 1)


# ---------------------------------------------------------------- сценарии

def scenarios(tr, mid):
    tr = sorted(tr, key=lambda x: x[0])
    groups, cur = [], []
    for x in tr:
        if cur and x[0] != cur[0][0]:
            groups.append(cur)
            cur = []
        cur.append(x)
    if cur:
        groups.append(cur)
    res = {"tot": [], "h1": [], "h2": [], "dd": [], "n": []}
    rnd = random.Random(7)
    for _ in range(SCEN):
        free = 0
        eq = pk = dd = 0.0
        h = [0.0, 0.0]
        n = 0
        for grp in groups:
            if grp[0][0] < free or rnd.random() < SKIP:
                continue
            t, pnl, t_exit = rnd.choice(grp)
            eq += pnl
            h[0 if t < mid else 1] += pnl
            pk = max(pk, eq)
            dd = max(dd, pk - eq)
            free = t_exit
            n += 1
        res["tot"].append(eq)
        res["h1"].append(h[0])
        res["h2"].append(h[1])
        res["dd"].append(dd)
        res["n"].append(n)
    for k in res:
        res[k].sort()
    return res


def med(x):
    return x[len(x) // 2]


def main():
    t0 = time.time()
    coins = [c for c in FIRST_RUN if c != "BTC"]
    try:
        coins += [c for c in top_coins(50) if c not in coins and c != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    say("СРАВНЕНИЕ БОТ / ИНДИКАТОР · %s · монет %d · %d дней · одна сделка за раз, депозит 1000 $"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), DAYS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(load_coin, ["BTC"] + coins))
    bd1 = data["BTC"][2]
    btc_t, btc_tr = [x[0] for x in bd1], trend_arr(bd1)

    def btc_at_close(T):
        """Тренд BTC по последней закрытой дневке на момент T."""
        i = bisect.bisect_right(btc_t, T - D1) - 1
        return btc_tr[i] if i >= 0 else 0

    def btc_ind(bar_open):
        """Как в индикаторе: прошлая закрытая дневка относительно дня свечи графика."""
        p = bar_open - bar_open % D1 - D1
        i = bisect.bisect_left(btc_t, p)
        return btc_tr[i] if i < len(btc_t) and btc_t[i] == p else 0

    trades = {1: [], 2: [], 3: [], 4: []}
    t_min, t_max = None, None
    for sym in coins:
        b5, h4, d1 = data.get(sym) or ([], [], [])
        if len(b5) < 50000 or len(h4) < 400 or len(d1) < 260:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t5 = [x[0] for x in b5]
        h1 = agg(b5, H1)
        atr1 = atr_pct(h1)
        h4a = agg(b5, H4)
        h4a_up = [a > b_ for a, b_ in zip(ema([x[4] for x in h4a], 9), ema([x[4] for x in h4a], 21))]
        be = bot_entries(h1, h4, d1)
        ie = [(T, d, i) for T, d, i in ind_entries(h1, [x[0] for x in h4a], h4a_up)
              if btc_ind(h1[i][0]) == d]
        for T, d, i in be:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1:
                continue
            pnl, te = sim_grid(b5, k0, d, max(atr1[i], MIN_STEP))
            trades[1].append((T, pnl, te))
            if btc_at_close(T) == d:
                trades[2].append((T, pnl, te))
        for T, d, i in ie:
            k0 = bisect.bisect_left(t5, T)
            if k0 >= len(b5) - 1:
                continue
            pnl, te = sim_single(b5, k0, d, atr1[i])
            trades[3].append((T, pnl, te))
            pnl4, te4 = sim_grid(b5, k0, d, max(atr1[i], MIN_STEP))
            trades[4].append((T, pnl4, te4))
        t_min = h1[0][0] if t_min is None else min(t_min, h1[0][0])
        t_max = h1[-1][0] if t_max is None else max(t_max, h1[-1][0])
        say("  %s: входов бота %d, сигналов индикатора (BTC 1Д за) %d" % (sym, len(be), len(ie)))

    if t_min is None:
        say("Нет данных ни по одной монете.")
        return
    mid = (t_min + t_max) / 2
    names = {1: "БОТ (как сейчас)", 2: "БОТ + BTC 1Д", 3: "ИНДИКАТОР 1h + BTC 1Д, один вход",
             4: "ГИБРИД: вход индикатора + сетка бота"}
    say("")
    say("=" * 110)
    say("Все сделки (без ограничения «одна за раз») — для понимания, не для решения:")
    for v in (1, 2, 3, 4):
        p = [x[1] for x in trades[v]]
        if p:
            say("  %d %-40s сделок %5d · средняя %+6.2f $ · винрейт %3.0f%%"
                % (v, names[v], len(p), sum(p) / len(p), 100 * sum(1 for x in p if x > 0) / len(p)))
    say("")
    say("ОДНА СДЕЛКА ЗА РАЗ, 30%% сигналов пропущено, %d сценариев, депозит 1000 $:" % SCEN)
    res = {}
    for v in (1, 2, 3, 4):
        if not trades[v]:
            continue
        r = scenarios(trades[v], mid)
        res[v] = r
        say("  %d %-40s сделок ~%d · итог медиана %+6.0f $ (10%% худших %+5.0f) · в плюсе %3.0f%% · "
            "половины %+5.0f / %+5.0f $ · просадка медиана %4.0f $ (10%% худших %4.0f)"
            % (v, names[v], med(r["n"]), med(r["tot"]), r["tot"][SCEN // 10],
               100 * sum(1 for x in r["tot"] if x > 0) / SCEN, med(r["h1"]), med(r["h2"]),
               med(r["dd"]), r["dd"][SCEN * 9 // 10]))
    say("")
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона): лучше бота, если медиана выше в обеих половинах "
        "и просадка не больше")
    if 1 in res:
        b = res[1]
        for v in (2, 3, 4):
            if v not in res:
                continue
            r = res[v]
            c1 = med(r["h1"]) > med(b["h1"]) and med(r["h2"]) > med(b["h2"])
            c2 = med(r["dd"]) <= med(b["dd"])
            say("  %d %-40s половины [%s] · просадка [%s] → %s"
                % (v, names[v], "да" if c1 else "нет", "да" if c2 else "нет",
                   "ЛУЧШЕ БОТА" if (c1 and c2) else "не лучше"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

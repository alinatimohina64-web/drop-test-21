#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ТЕСТ ФИЛЬТРА ПО BTC ДЛЯ СИГНАЛОВ ИНДИКАТОРА Trand-Test-2 (v4.2, настройки по умолчанию).

Сигналы индикатора перенесены один в один:
  основной: пересечение EMA9/EMA21, продержавшееся 2 бара, RSI(14) > 50 и растёт,
            объём > SMA20 × 1.3, ADX(14) >= 20, старший тренд 4ч (EMA9 > EMA21) в ту же
            сторону, кулдаун 5 баров;
  по откату: EMA9 > EMA21, свеча коснулась EMA9 и закрылась выше неё (и выше открытия),
            RSI растёт, ADX >= 20, старший тренд 4ч, кулдаун 10 баров.
  Шорт — зеркально. Вход по закрытию сигнальной свечи.
  Цель = 2.5 × ATR%, стоп = 5.2 × ATR% (ATR 14 этого ТФ) — как в таблице индикатора.
  Выход проверяется со следующей свечи; если свеча задела и цель, и стоп — проигрыш
  (как в индикаторе). Каждый сигнал считается отдельно, как в статистике индикатора.
  Мат. ожидание = средний % на сделку − 0.11% комиссии за круг.

Фильтр BTC (как в индикаторе v4.2): тренд BTC — цена выше EMA50 и EMA21 выше EMA50
(шорт зеркально) на последней закрытой свече BTC прошлого периода ТФ BTC.
Связки: 5m → BTC 1ч, 15m → BTC 4ч.

КРИТЕРИЙ (объявлен ДО прогона), отдельно для каждой связки. Фильтр принимаем, если:
  1) мат. «BTC за» лучше «BTC против» на >= 70% монет (монеты, где в обеих группах
     не меньше 20 сделок);
  2) суммарный мат. «за» по всем монетам больше нуля;
  3) суммарный мат. «за» лучше «против» в обеих половинах периода.

Монеты: 10 монет первого прогона + топ-50 OKX, кроме BTC. XRP уже смотрели вручную.
Переменные: DAYS (по умолчанию 365). Отчёт: ind_btc_report.txt
"""
import bisect
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import trend_grid_test as g

DAYS = int(os.environ.get("DAYS") or 365)
DATA_DIR = "data_ind"                 # отдельный кэш: здесь свечи с объёмом
REPORT = "ind_btc_report.txt"
FEE = 0.11                            # % за круг
MIN_N = 20
M5, M15, H1, H4 = 300000, 900000, 3600000, 14400000
LINKS = [("5m", M5, "1ч", H1), ("15m", M15, "4ч", H4)]

_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


# ---------------------------------------------------------------- данные с объёмом

def fetch(sym, t_from, t_stop=0):
    inst = "%s-USDT-SWAP" % sym
    stop = max(t_from, t_stop)
    got = {}
    for host in ("https://www.okx.com", "https://aws.okx.com"):
        after = None
        try:
            while True:
                url = "%s/api/v5/market/history-candles?instId=%s&bar=5m&limit=100" % (host, inst)
                if after is not None:
                    url += "&after=%d" % after
                rows = g.http_json(url).get("data") or []
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
            print("  ! %s %s: %s" % (sym, host, e), flush=True)
    now = int(time.time() * 1000)
    return sorted(v for v in got.values() if v[0] + M5 <= now)


def load(sym):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "%s_5m.json.gz" % sym)
    t_from = int(time.time() * 1000) - DAYS * 86400000
    old = []
    if os.path.exists(path):
        try:
            with gzip.open(path, "rt") as f:
                old = json.load(f)
        except Exception:                                       # noqa: BLE001
            old = []
    if old and old[0][0] <= t_from + M5:
        new = fetch(sym, t_from, old[-1][0])
    else:
        old, new = [], fetch(sym, t_from)
    rows = {r[0]: r for r in old}
    rows.update({r[0]: r for r in new})
    bars = sorted(r for r in rows.values() if r[0] >= t_from)
    with gzip.open(path, "wt") as f:
        json.dump(bars, f)
    print("  %s: %d свечей 5m" % (sym, len(bars)), flush=True)
    return sym, bars


def agg(bars, ms):
    """Свечи старшего ТФ с объёмом; последняя (возможно незакрытая) отбрасывается."""
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


# ---------------------------------------------------------------- индикатор

def sma(x, n):
    out, s = [], 0.0
    for i, v in enumerate(x):
        s += v
        if i >= n:
            s -= x[i - n]
        out.append(s / n if i >= n - 1 else None)
    return out


def rsi(c, n=14):
    up = [0.0] + [max(c[i] - c[i - 1], 0.0) for i in range(1, len(c))]
    dn = [0.0] + [max(c[i - 1] - c[i], 0.0) for i in range(1, len(c))]
    ru, rd = g.rma(up, n), g.rma(dn, n)
    return [100.0 if rd[i] == 0 else 100 - 100 / (1 + ru[i] / rd[i]) for i in range(len(c))]


def trend_up_series(b, fast, slow):
    """Для старшего ТФ индикатора: EMA fast > EMA slow."""
    c = [x[4] for x in b]
    ef, es = g.ema(c, fast), g.ema(c, slow)
    return [ef[i] > es[i] for i in range(len(b))]


def signals(b, h4_t, h4_up):
    """Сигналы индикатора: список (индекс бара, сторона)."""
    n = len(b)
    c = [x[4] for x in b]
    ef, es = g.ema(c, 9), g.ema(c, 21)
    r = rsi(c)
    av = sma([x[5] for x in b], 20)
    adx = g.adx_arr([x[:5] for x in b])
    out = []
    last_up = last_dn = None
    last_sig = last_pb = None
    tf = b[1][0] - b[0][0] if n > 1 else M5
    j4 = -1
    for i in range(1, n):
        if ef[i] > es[i] and ef[i - 1] <= es[i - 1]:
            last_up = i
        if ef[i] < es[i] and ef[i - 1] >= es[i - 1]:
            last_dn = i
        if i < 60 or av[i] is None:
            continue
        close_t = b[i][0] + tf
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
            out.append((i, 1 if long_sig else -1))
        o, h, l = b[i][1], b[i][2], b[i][3]
        pb_long = (ef[i] > es[i] and adx_ok and htf_up and l <= ef[i] and c[i] > ef[i] and c[i] > o
                   and r[i] > r[i - 1])
        pb_short = (ef[i] < es[i] and adx_ok and not htf_up and h >= ef[i] and c[i] < ef[i] and c[i] < o
                    and r[i] < r[i - 1])
        pb_ok = last_pb is None or i - last_pb >= 10
        pl, ps = pb_long and pb_ok and not long_sig, pb_short and pb_ok and not short_sig
        if pl or ps:
            last_pb = i
            out.append((i, 1 if pl else -1))
    return out


def outcome(b, i, d, tp, sl):
    """% результата сделки по цели/стопу; спорная свеча — стоп. None — не закрылась."""
    e = b[i][4]
    tgt = e * (1 + d * tp / 100)
    stp = e * (1 - d * sl / 100)
    for k in range(i + 1, len(b)):
        h, l = b[k][2], b[k][3]
        hit_t = h >= tgt if d == 1 else l <= tgt
        hit_s = l <= stp if d == 1 else h >= stp
        if hit_s:
            return -sl
        if hit_t:
            return tp
    return None


def btc_trend(b):
    """Тренд BTC как в индикаторе v4.2: +1 / -1 / 0."""
    c = [x[4] for x in b]
    ef, es = g.ema(c, 21), g.ema(c, 50)
    return [1 if (c[i] > es[i] and ef[i] > es[i]) else -1 if (c[i] < es[i] and ef[i] < es[i]) else 0
            for i in range(len(b))]


# ---------------------------------------------------------------- отчёт

def st(tr):
    if not tr:
        return None
    w = sum(1 for x in tr if x > 0)
    return len(tr), 100 * w / len(tr), sum(tr) / len(tr) - FEE


def fmt(s):
    return "нет сделок" if s is None else "%4d сд · WR %5.1f%% · мат. %+.3f%%" % s


def main():
    t0 = time.time()
    coins = [c for c in g.FIRST_RUN if c != "BTC"]
    try:
        coins += [c for c in g.top_new_coins(50) if c not in coins and c != "BTC"]
    except Exception as e:                                      # noqa: BLE001
        say("! топ-50 не получен (%s)" % e)
    say("ТЕСТ ФИЛЬТРА BTC ДЛЯ СИГНАЛОВ ИНДИКАТОРА · %s · монет %d · %d дней"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins), DAYS))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(load, ["BTC"] + coins))
    btc5 = data.get("BTC") or []
    if len(btc5) < 20000:
        say("Нет данных BTC.")
        return
    btc = {}
    for _, _, bname, bms in LINKS:
        bb = agg(btc5, bms)
        btc[bms] = ([x[0] for x in bb], btc_trend(bb))

    for tfname, tfms, bname, bms in LINKS:
        bt_t, bt_tr = btc[bms]
        pooled = {"all": [], "for": [], "ag": []}
        halves = {1: {"for": [], "ag": []}, 2: {"for": [], "ag": []}}
        rows, better, counted = [], 0, 0
        for sym in coins:
            b5 = data.get(sym) or []
            if len(b5) < 20000:
                continue
            b = b5 if tfms == M5 else agg(b5, tfms)
            h4 = agg(b5, H4)
            sig = signals(b, [x[0] for x in h4], trend_up_series(h4, 9, 21))
            atrp = g.atr_pct([x[:5] for x in b])
            mid = (b[0][0] + b[-1][0]) / 2
            res = {"all": [], "for": [], "ag": []}
            for i, d in sig:
                res_pct = outcome(b, i, d, 2.5 * atrp[i], 5.2 * atrp[i])
                if res_pct is None:
                    continue
                # последняя закрытая свеча BTC прошлого периода (как request.security [1] + lookahead)
                p = b[i][0] - b[i][0] % bms - bms
                j = bisect.bisect_left(bt_t, p)
                if j >= len(bt_t) or bt_t[j] != p:
                    continue
                grp = "for" if bt_tr[j] == d else "ag"
                res["all"].append(res_pct)
                res[grp].append(res_pct)
                halves[1 if b[i][0] < mid else 2][grp].append(res_pct)
            for k in pooled:
                pooled[k] += res[k]
            sf, sa = st(res["for"]), st(res["ag"])
            mark = ""
            if sf and sa and sf[0] >= MIN_N and sa[0] >= MIN_N:
                counted += 1
                if sf[2] > sa[2]:
                    better += 1
                    mark = "за лучше"
                else:
                    mark = "за НЕ лучше"
            rows.append("  %-8s все: %s | за: %s | против: %s %s"
                        % (sym + ("*" if sym == "XRP" else ""), fmt(st(res["all"])), fmt(sf), fmt(sa), mark))
        say("")
        say("=" * 110)
        say("СВЯЗКА %s → BTC %s" % (tfname, bname))
        for r_ in rows:
            say(r_)
        pa, pf, pg = st(pooled["all"]), st(pooled["for"]), st(pooled["ag"])
        say("")
        say("  ВСЕ МОНЕТЫ   все: %s" % fmt(pa))
        say("               за:  %s" % fmt(pf))
        say("               против: %s" % fmt(pg))
        hs = []
        for h in (1, 2):
            f_, a_ = st(halves[h]["for"]), st(halves[h]["ag"])
            hs.append(f_ and a_ and f_[2] > a_[2])
            say("  %d-я половина  за: %s | против: %s" % (h, fmt(f_), fmt(a_)))
        c1 = counted > 0 and better >= 0.7 * counted
        c2 = pf is not None and pf[2] > 0
        c3 = all(hs)
        say("  ВЕРДИКТ: за лучше на %d из %d монет [%s] · мат. за > 0 [%s] · обе половины [%s] → %s"
            % (better, counted, "да" if c1 else "нет", "да" if c2 else "нет", "да" if c3 else "нет",
               "ПРИНЯТЬ" if (c1 and c2 and c3) else "не принимать"))
    say("")
    say("* XRP уже смотрели вручную. Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

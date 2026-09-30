#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЛАБОРАТОРИЯ BTC — все стратегии проекта и новые идеи ОДНИМ прогоном, только BTC, история с 2017 года.

ДАННЫЕ: BTCUSDT спот Binance, 5-минутки с августа 2017 (data.binance.vision, кэш data_btc/),
фандинг BTCUSDT-перпетуала Binance с сентября 2019. Исполнение сделок — на 5m.

ЕДИНЫЕ ПРАВИЛА ДЛЯ ВСЕХ (объявлены до прогона, параметры — как были в исходных стратегиях, без подкрутки):
  * сигнал на закрытии свечи своего ТФ, вход по открытию следующей 5m-свечи;
  * внутри 5m-свечи всё спорное — в худшую сторону: сначала стоп, известный на начало свечи; если
    свеча обновила максимум (трейлинг подтянулся) и в ней же задет подтянутый стоп — выход по нему;
    цель на свече входа лимиткой — только если закрытие за целью;
  * комиссия 0.11% за круг (тейкер), фандинг 0.01% за 8 ч всегда против позиции;
  * одна позиция за раз на стратегию; сделка не дольше 60 дней;
  * результат в R (R = риск сделки, как он задан в стратегии) и в $ при риске 20 $ на сделку.

ПЕРИОДЫ: разработка 2017-08…2021-12 (для справки), ПРОВЕРКА 2022-01…сейчас (по ней вердикт).

КОНТРОЛИ (встроены, в том же прогоне):
  * случайный вход — для каждой сделки проверки 3 входа в случайные моменты того же года,
    случайная сторона, те же правила выхода (показывает, дают ли что-то сами сигналы);
  * случайные цены — весь прогон повторяется на синтетических ценах без закономерностей
    (мартингал с меняющейся волатильностью). Там у любой стратегии средняя сделка должна быть
    около нуля минус комиссии. Если на случайных ценах стратегия «зарабатывает» — это дефект теста,
    её вердикт помечается «тест под подозрением».

СТРАТЕГИИ:
  S1 Бот GHOST (робот 🟢): разворот 1Ч по тренду 4Ч/1Д, ADX 4Ч >= 25, 200Д, MRC; сетка 4 × 300 $,
     шаг ATR 1ч (>= 1%), тейк 2.5 шага от средней, стоп 5.2 шага, трейлинг 50/50/50.
  S2 Бот на 4ч: разворот 4Ч по тренду 1Д, шаг ATR 4ч, та же сетка.
  S3 Индикатор TT2: 30m, старший 4ч, по дневному тренду BTC; цель 5 / стоп 3.5 × ATR.
  S4 Пробой канала 1Д (TrendBreakoutStrategy): Donchian 20, ATR 14, режим бота — 3 трендовых
     ордера, шаг 1 ATR, стоп 2.5 ATR от средней, трейлинг 50% дельты.
  S5 Пробой канала 4Ч — то же на 4-часовках.
  S6 Контртренд + раннер 4Ч (MeanReversionStrategy): отскок от SMA50 ± 1 ATR при ADX < 20 и
     объёме > 1.3 среднего, лимитка по закрытию (живёт 1 свечу), цель центр ± 0.5 ATR, стоп за
     внешней границей (2 ATR) + 0.5 ATR, на цели закрыть 10%, остаток — трейлинг 0.5 ATR.
  N1 «Черепаха» 1Д: пробой максимума/минимума 20 дней, стоп 2 ATR(20), выход по 10-дневному каналу.
  N2 Время суток: на разработке выбирается лучшее 4-часовое окно дня (лонг) и худшее (шорт),
     на проверке они торгуются каждый день.
  N3 Экстремальный фандинг: фандинг выше 95-го процентиля за 90 дней — шорт, ниже 5-го — лонг;
     держать 72 ч, стоп 2 ATR(1Д).
  N4 Отскок после резкой свечи 4Ч: свеча против тренда (цена vs SMA200 1Д) длиной >= 2.5 ATR —
     вход на отскок, цель и стоп 2 ATR 4ч, не дольше 3 дней.
  N5 Пробой после сжатия 4Ч: ширина Боллинджера (20, 2) на минимуме 120 свечей в последние 6 свечей,
     закрытие за полосой — вход, стоп 2 ATR, трейлинг 3 ATR.
  N6 Выходной гэп: цена в вс 23:00 UTC ушла от пт 21:00 UTC больше чем на 1% — ставка на
     возврат к пятничной цене, стоп на том же расстоянии, не дольше 5 дней.

ВЕРДИКТ по стратегии (объявлен до прогона) — «РАБОТАЕТ», если на проверке 2022–2026:
  1) сделок >= 30, итог в плюсе и он статистически заметен: t >= 2 (без этого на случайных ценах
     часть стратегий «проходит» просто по везению — проверено на пробном прогоне);
  2) в плюсе не меньше 3 из 5 лет (2022…2026);
  3) в плюсе обе половины проверки;
  4) средняя сделка лучше случайного входа;
  5) на случайных ценах стратегия не зарабатывает (t < 2), иначе — «тест под подозрением».

Лежит рядом с compare_test.py, multi_test.py, coin_wf_test.py (оттуда берутся входы S1–S3).
Нужен numpy. Отчёт: btc_lab_report.txt
"""
import bisect
import csv
import io
import math
import os
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np

import compare_test as c
import multi_test as mt
import coin_wf_test as cw

REPORT = "btc_lab_report.txt"
DATA_DIR = "data_btc"
M5, M30 = 300000, 1800000
H1, H4, D1 = c.H1, c.H4, c.D1
FEE = 0.11                       # % за круг
FUND = 0.01                      # % за 8 ч
RISK = 20.0
MAXB = 60 * 288                  # 60 дней 5-минуток
START = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
TEST_FROM = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
N_RAND = 3
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


# ============================================================ данные

def _get(url, tries=4):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * a)
        except Exception:                                    # noqa: BLE001
            time.sleep(2 + 3 * a)
    raise RuntimeError("не скачалось: " + url)


def _zip_rows(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    with z.open(z.namelist()[0]) as f:
        return list(csv.reader(io.TextIOWrapper(f, encoding="utf-8")))


def _cached(name, url):
    os.makedirs(DATA_DIR, exist_ok=True)
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    blob = _get(url)
    if blob is not None:
        with open(p, "wb") as f:
            f.write(blob)
    return blob


def load_btc_5m():
    base = "https://data.binance.vision/data/spot/%s/klines/BTCUSDT/5m/"
    now = datetime.now(timezone.utc)
    bars = {}
    y, m = 2017, 8
    while (y, m) <= (now.year, now.month):
        tag = "%04d-%02d" % (y, m)
        cur_month = (y, m) == (now.year, now.month)
        blob = None if cur_month else _cached("5m-%s.zip" % tag, base % "monthly" + "BTCUSDT-5m-%s.zip" % tag)
        days = []
        if blob is not None:
            days = [blob]
        else:                                              # месяц ещё не выложен целиком — по дням
            d = datetime(y, m, 1, tzinfo=timezone.utc)
            while d.month == m and d.date() < now.date():
                dt = d.strftime("%Y-%m-%d")
                name = "5m-%s.zip" % dt
                b = _cached(name, base % "daily" + "BTCUSDT-5m-%s.zip" % dt) if not cur_month or d.date() < (now - timedelta(days=1)).date() \
                    else _get(base % "daily" + "BTCUSDT-5m-%s.zip" % dt)
                if b is not None:
                    days.append(b)
                d += timedelta(days=1)
        for b in days:
            for r in _zip_rows(b):
                if not r or not r[0].isdigit():
                    continue
                t = int(r[0])
                if t > 10 ** 14:                           # с 2025 года Binance пишет микросекунды
                    t //= 1000
                bars[t] = [t, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
        m += 1
        if m == 13:
            y, m = y + 1, 1
    out = [bars[k] for k in sorted(bars)]
    say("  BTC 5m: %d свечей, %s … %s" % (len(out), ts(out[0][0]), ts(out[-1][0])))
    return out


def load_funding():
    base = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/"
    now = datetime.now(timezone.utc)
    out = {}
    y, m = 2019, 9
    while (y, m) < (now.year, now.month):
        tag = "%04d-%02d" % (y, m)
        try:
            blob = _cached("fund-%s.zip" % tag, base + "BTCUSDT-fundingRate-%s.zip" % tag)
        except RuntimeError:
            blob = None
        if blob is not None:
            for r in _zip_rows(blob):
                if not r or not r[0].isdigit():
                    continue
                t = int(r[0])
                if t > 10 ** 14:
                    t //= 1000
                out[t] = float(r[-1])
        m += 1
        if m == 13:
            y, m = y + 1, 1
    res = sorted(out.items())
    say("  фандинг: %d записей" % len(res))
    return res


def synth(n_from, n_to, seed):  # noqa: C901
    """Случайные цены: мартингал (E[следующая цена] = текущая), волатильность меняется по дням."""
    rnd = random.Random(seed)
    t, p, b = n_from - n_from % M5, 4000.0, []
    vol_d, day = 0.0025, None
    while t < n_to:
        if t // D1 != day:
            day = t // D1
            vol_d = min(0.012, max(0.0008, vol_d * math.exp(rnd.gauss(0, 0.15)) * (0.0025 / vol_d) ** 0.02))
        s = vol_d if rnd.random() > 0.005 else vol_d * 5
        o = p
        p = o * math.exp(rnd.gauss(-s * s / 2, s))
        hi = max(o, p) * (1 + abs(rnd.gauss(0, s * 0.5)))
        lo = min(o, p) * (1 - abs(rnd.gauss(0, s * 0.5)))
        b.append([t, o, hi, lo, p, 50 * math.exp(rnd.gauss(0, 0.7))])
        t += M5
    fund = [(tt, rnd.gauss(0.0001, 0.0002)) for tt in range(n_from - n_from % (8 * H1) + 8 * H1, n_to, 8 * H1)
            if tt >= int(datetime(2019, 9, 10, tzinfo=timezone.utc).timestamp() * 1000)]
    return b, fund


def ts(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ============================================================ рынок и исполнение

class Mkt:
    def __init__(self, b5):
        self.b5 = b5
        self.t = [x[0] for x in b5]
        self.o = [x[1] for x in b5]
        self.h = [x[2] for x in b5]
        self.l = [x[3] for x in b5]
        self.c = [x[4] for x in b5]
        self.n = len(b5)

    def k_at(self, T):
        return bisect.bisect_left(self.t, T)


def fund_cost(hours):
    return FUND * hours / 8.0


def ex_tp_sl(m, k0, d, tp, sl, maxb=MAXB):
    """Цель и стоп в % от входа. R = результат / стоп."""
    p0 = m.o[k0]
    tgt, stp = p0 * (1 + d * tp / 100), p0 * (1 - d * sl / 100)
    k1 = min(m.n, k0 + maxb)
    px, k = m.c[k1 - 1], k1 - 1
    for k in range(k0, k1):
        o, fav, adv = m.o[k], (m.h[k] if d == 1 else m.l[k]), (m.l[k] if d == 1 else m.h[k])
        if d * o <= d * stp:
            px = o
            break
        if d * adv <= d * stp:
            px = stp
            break
        if d * fav >= d * tgt:
            px = o if d * o > d * tgt else tgt
            break
    hours = (m.t[k] + M5 - m.t[k0]) / H1
    pnl = d * (px / p0 - 1) * 100 - FEE - fund_cost(hours)
    return pnl / sl, m.t[k] + M5


def ex_time(m, k0, d, hold, sl, risk):
    """Выход по времени (hold свечей 5m), аварийный стоп sl %. R = результат / risk %."""
    p0 = m.o[k0]
    stp = p0 * (1 - d * sl / 100)
    k1 = min(m.n - 1, k0 + hold)
    px, k = m.o[k1], k1
    for k in range(k0, k1):
        o, adv = m.o[k], (m.l[k] if d == 1 else m.h[k])
        if d * o <= d * stp:
            px = o
            break
        if d * adv <= d * stp:
            px = stp
            break
    hours = (m.t[k] - m.t[k0]) / H1
    pnl = d * (px / p0 - 1) * 100 - FEE - fund_cost(hours)
    return pnl / risk, m.t[k] + M5


def ex_trail(m, k0, d, init_abs, trail_abs, maxb=MAXB):
    """Начальный стоп init_abs, трейлинг trail_abs от лучшей цены, без цели. R = от начального стопа."""
    p0 = m.o[k0]
    stop = p0 - d * init_abs
    best = p0
    k1 = min(m.n, k0 + maxb)
    px, k = m.c[k1 - 1], k1 - 1
    for k in range(k0, k1):
        o, fav, adv = m.o[k], (m.h[k] if d == 1 else m.l[k]), (m.l[k] if d == 1 else m.h[k])
        if d * o <= d * stop:
            px = o
            break
        if d * adv <= d * stop:
            px = stop
            break
        if d * fav > d * best:
            best = fav
            ns = best - d * trail_abs
            if d * ns > d * stop:
                stop = ns
                if d * adv <= d * stop:              # спорная свеча — в худшую сторону
                    px = stop
                    break
    hours = (m.t[k] + M5 - m.t[k0]) / H1
    pnl = d * (px / p0 - 1) * 100 - FEE - fund_cost(hours)
    return pnl / (init_abs / p0 * 100), m.t[k] + M5


def ex_ghost(m, k0, d, step):
    """Сетка GHOST (4 × 300 $, тейк 2.5 шага от средней, стоп 5.2 шага, трейлинг 50%), спорное — в худшую."""
    s = step / 100
    tp, sl = 2.5 * s, 5.2 * s
    p0 = m.o[k0]
    prices = [p0 * (1 + d * j * s) for j in range(c.N_ORD)]
    qty, cost, n = c.VOL / p0, c.VOL, 1
    fees, fund = c.VOL * c.F_TAKER, 0.0
    sl_px, peak = p0 * (1 - d * sl), None
    k1 = min(m.n, k0 + MAXB)

    def done(px, k, maker=False):
        pnl = d * (qty * px - cost) - fees - qty * px * (c.F_MAKER if maker else c.F_TAKER) + fund
        return pnl / (c.VOL * sl), m.t[k] + M5

    for k in range(k0, k1):
        t, o, h, l, cl = m.t[k], m.o[k], m.h[k], m.l[k], m.c[k]
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
            if d * adv <= d * trl:                   # спорная свеча — в худшую сторону
                return done(trl, k)
        if (t + M5) % (8 * H1) == 0:
            fund -= d * c.FUND * qty * cl
    return done(m.c[k1 - 1], k1 - 1)


def ex_breakout_bot(m, k0, d, atr_abs, orders=3, step_atr=1.0, stop_atr=2.5, keep=50.0):
    """Режим бота TrendBreakoutStrategy: головной + 3 трендовых ордера, стоп от средней, трейлинг % дельты."""
    p0 = m.o[k0]
    sl_pct = atr_abs * stop_atr / p0
    step_pct = atr_abs * step_atr / p0
    fills, ftimes = [p0], [m.t[k0]]
    best = p0
    k1 = min(m.n, k0 + MAXB)

    def avg():
        return len(fills) / sum(1.0 / p for p in fills)

    def cur_stop():
        a = avg()
        st = a * (1 - d * sl_pct)
        if len(fills) >= 2:
            delta = d * (best - a)
            if delta > 0:
                tr = best - d * keep / 100.0 * delta
                st = max(st, tr) if d == 1 else min(st, tr)
        return st

    def close(px, k):
        pnl = sum(d * (px - f) / f for f in fills)                # в единицах номинала ордера
        fees = FEE / 2 / 100 * (len(fills) + sum(px / f for f in fills))
        fund = sum(FUND / 100 * ((m.t[k] + M5 - ft) / H1) / 8 for ft in ftimes)
        net = pnl - fees - fund
        return net / ((1 + orders) * sl_pct), m.t[k] + M5

    for k in range(k0, k1):
        o, fav, adv = m.o[k], (m.h[k] if d == 1 else m.l[k]), (m.l[k] if d == 1 else m.h[k])
        st = cur_stop()
        if d * o <= d * st:
            return close(o, k)
        if d * adv <= d * st:
            return close(st, k)
        while len(fills) - 1 < orders:
            j = len(fills)
            lvl = p0 * (1 + d * j * step_pct)
            if d * fav < d * lvl:
                break
            fills.append(o if d * o > d * lvl else lvl)
            ftimes.append(m.t[k])
        if d * fav > d * best:
            best = fav
            st2 = cur_stop()
            if d * adv <= d * st2:                   # спорная свеча — в худшую сторону
                return close(st2, k)
    return close(m.c[k1 - 1], k1 - 1)


def ex_mr(m, k0, d, info):
    """MeanReversionStrategy: лимитка, цель, стоп, 10% на цели, остаток — трейлинг 0.5 ATR. None — не исполнена."""
    lim, tp_px, sl_px, atr_abs = info
    kf = None
    for k in range(k0, min(m.n, k0 + 48)):                     # лимитка живёт одну 4-часовку
        if (d == 1 and m.l[k] < lim) or (d == -1 and m.h[k] > lim):
            kf = k
            break
    if kf is None:
        return None
    sl_pct = abs(lim - sl_px) / lim * 100
    k1 = min(m.n, kf + MAXB)
    part, rem = 0.10, 0.90
    realized = 0.0
    runner, trail = False, None
    dist = 0.5 * atr_abs
    px, k = m.c[k1 - 1], k1 - 1
    for k in range(kf, k1):
        o, fav, adv, cl = m.o[k], (m.h[k] if d == 1 else m.l[k]), (m.l[k] if d == 1 else m.h[k]), m.c[k]
        if not runner:
            if d * adv <= d * sl_px:
                px = sl_px if k == kf or d * o > d * sl_px else o
                break
            tp_hit = (d * cl >= d * tp_px) if k == kf else (d * fav >= d * tp_px)
            if tp_hit:
                realized = part * d * (tp_px / lim - 1) * 100
                runner, trail = True, tp_px
                if k != kf:
                    ns = fav - d * dist
                    if d * ns > d * trail:
                        trail = ns
                    if d * adv <= d * trail and d * trail > d * tp_px:
                        px = trail
                        break
            continue
        if d * o <= d * trail:
            px = o
            break
        if d * adv <= d * trail:
            px = trail
            break
        ns = fav - d * dist
        if d * ns > d * trail:
            trail = ns
            if d * adv <= d * trail:
                px = trail
                break
    hours = (m.t[k] + M5 - m.t[kf]) / H1
    main = (rem if runner else 1.0) * d * (px / lim - 1) * 100
    pnl = realized + main - FEE - fund_cost(hours)
    return pnl / sl_pct, m.t[k] + M5


def ex_turtle(m, k0, d, info, day_idx, dlo10, dhi10, dcl):
    """Стоп 2 ATR, выход, когда дневное закрытие пробило 10-дневный канал против позиции (по открытию дня)."""
    stop_abs = info
    p0 = m.o[k0]
    stp = p0 - d * stop_abs
    k1 = min(m.n, k0 + MAXB * 3)
    px, k = m.c[k1 - 1], k1 - 1
    for k in range(k0, k1):
        o, adv = m.o[k], (m.l[k] if d == 1 else m.h[k])
        if k > k0 and m.t[k] % D1 == 0:
            j = day_idx.get(m.t[k] - D1)
            if j is not None and dlo10[j] is not None:
                if (d == 1 and dcl[j] < dlo10[j]) or (d == -1 and dcl[j] > dhi10[j]):
                    px = o
                    break
        if d * o <= d * stp:
            px = o
            break
        if d * adv <= d * stp:
            px = stp
            break
    hours = (m.t[k] + M5 - m.t[k0]) / H1
    pnl = d * (px / p0 - 1) * 100 - FEE - fund_cost(hours)
    return pnl / (stop_abs / p0 * 100), m.t[k] + M5


# ============================================================ стратегии

def build(m, fund):
    """Возвращает список стратегий: (код, название, сигналы [(T, d)], info_at(T, d), trade(k0, d, info))."""
    b5 = m.b5
    h1 = c.agg(b5, H1)
    h4 = c.agg(b5, H4)
    d1 = c.agg(b5, D1)
    b30 = c.agg(b5, M30)
    t1, t4, td, t30 = [x[0] for x in h1], [x[0] for x in h4], [x[0] for x in d1], [x[0] for x in b30]
    atr1, atr4, atrd, atr30 = c.atr_pct(h1), c.atr_pct(h4), c.atr_pct(d1), c.atr_pct(b30)
    atrd20 = c.atr_pct(d1, 20)
    dcl = [x[4] for x in d1]
    trd = c.trend_arr(d1)
    sma200 = c.sma(dcl, 200)
    day_idx = {x[0]: i for i, x in enumerate(d1)}

    def last(tt, T, ms):
        return bisect.bisect_right(tt, T - ms) - 1

    S = []

    # ---- S1 бот GHOST (робот 🟢)
    sig = [(T, d) for T, d, i in c.bot_entries(h1, h4, d1)]
    S.append(("S1", "бот GHOST 1ч (робот)", sig,
              lambda T, d: max(atr1[last(t1, T, H1)], c.MIN_STEP),
              lambda k0, d, s: ex_ghost(m, k0, d, s)))

    # ---- S2 бот на 4ч
    sig = [(T, d) for T, d, i in mt.bot_entries_4h(h4, d1)]
    S.append(("S2", "бот на 4ч", sig,
              lambda T, d: max(atr4[last(t4, T, H4)], c.MIN_STEP),
              lambda k0, d, s: ex_ghost(m, k0, d, s)))

    # ---- S3 индикатор TT2 30m / 4ч / тренд BTC 1Д
    c4 = [x[4] for x in h4]
    hu = [a > b_ for a, b_ in zip(c.ema(c4, 9), c.ema(c4, 21))]
    sig = []
    for T, d, i in cw.ind_entries_tf(b30, M30, t4, hu, H4):
        j = last(td, T, D1)
        if j >= 0 and trd[j] == d:
            sig.append((T, d))
    S.append(("S3", "индикатор TT2 (цель 5 / стоп 3.5 ATR)", sig,
              lambda T, d: atr30[last(t30, T, M30)],
              lambda k0, d, a: ex_tp_sl(m, k0, d, 5 * a, 3.5 * a)))

    # ---- S4/S5 пробой канала (бот-режим)
    def donchian(bars, n=20):
        out = []
        for i in range(n + 1, len(bars)):
            hh = max(x[2] for x in bars[i - n:i])
            ll = min(x[3] for x in bars[i - n:i])
            if bars[i][4] > hh:
                out.append((i, 1))
            elif bars[i][4] < ll:
                out.append((i, -1))
        return out

    for code, nm, bars, tt, ap, ms in (("S4", "пробой канала 1Д (бот-режим)", d1, td, atrd, D1),
                                       ("S5", "пробой канала 4Ч (бот-режим)", h4, t4, atr4, H4)):
        sig = [(bars[i][0] + ms, d) for i, d in donchian(bars)]
        S.append((code, nm, sig,
                  (lambda tt_, ap_, ms_, bars_: (lambda T, d: ap_[last(tt_, T, ms_)] / 100 * bars_[last(tt_, T, ms_)][4]))(tt, ap, ms, bars),
                  lambda k0, d, a: ex_breakout_bot(m, k0, d, a)))

    # ---- S6 контртренд + раннер 4Ч
    cl4 = [x[4] for x in h4]
    basis = c.sma(cl4, 50)
    adx4 = c.adx_arr(h4)
    vavg = c.sma([x[5] for x in h4], 20)
    atr4abs = [atr4[i] / 100 * cl4[i] for i in range(len(h4))]

    def mr_info(i, d):
        if i < 0 or basis[i] is None:
            return None
        a, bs, cl = atr4abs[i], basis[i], cl4[i]
        tgt = bs + d * 0.5 * a
        stp = bs - d * (2.0 * a + 0.5 * a)
        ap = a / cl * 100
        tp_pct = max(d * (tgt - cl) / cl * 100, 0.3 * ap)
        sl_pct = max(d * (cl - stp) / cl * 100, 0.3 * ap)
        return (cl, cl * (1 + d * tp_pct / 100), cl * (1 - d * sl_pct / 100), a)

    sig, last_sig = [], -10 ** 9
    for i in range(80, len(h4)):
        if basis[i] is None or vavg[i] is None:
            continue
        o, hh, ll, cl, v = h4[i][1:6]
        a = atr4abs[i]
        li, ui = basis[i] - a, basis[i] + a
        bl = ll <= li and cl > li and cl > o
        bsh = hh >= ui and cl < ui and cl < o
        ok = adx4[i] < 20 and v > vavg[i] * 1.3 and i - last_sig >= 5
        if ok and (bl or bsh):
            last_sig = i
            sig.append((h4[i][0] + H4, 1 if bl else -1))
    S.append(("S6", "контртренд + раннер 4Ч", sig,
              lambda T, d: mr_info(last(t4, T, H4), d),
              lambda k0, d, inf: ex_mr(m, k0, d, inf)))

    # ---- N1 черепаха 1Д
    dhi10 = [None] * len(d1)
    dlo10 = [None] * len(d1)
    for i in range(10, len(d1)):
        dhi10[i] = max(x[2] for x in d1[i - 10:i])
        dlo10[i] = min(x[3] for x in d1[i - 10:i])
    sig = [(d1[i][0] + D1, d) for i, d in donchian(d1, 20)]
    S.append(("N1", "черепаха 1Д (20/10, стоп 2 ATR)", sig,
              lambda T, d: 2 * atrd20[last(td, T, D1)] / 100 * dcl[last(td, T, D1)],
              lambda k0, d, s: ex_turtle(m, k0, d, s, day_idx, dlo10, dhi10, dcl)))

    # ---- N2 время суток: окно выбирается на разработке
    by_h = {hs: [] for hs in range(24)}
    for i in range(len(h1) - 4):
        if h1[i][0] >= TEST_FROM:
            break
        hs = (h1[i][0] // H1) % 24
        if h1[i + 3][0] - h1[i][0] == 3 * H1:
            by_h[hs].append(h1[i + 3][4] / h1[i][1] - 1)
    means = {hs: (sum(v) / len(v) if v else 0.0) for hs, v in by_h.items()}
    best_h = max(means, key=means.get)
    worst_h = min(means, key=means.get)
    sig = []
    for x in h1:
        if x[0] < TEST_FROM:
            continue
        hs = (x[0] // H1) % 24
        if hs == best_h:
            sig.append((x[0], 1))
        elif hs == worst_h and means[worst_h] < 0:
            sig.append((x[0], -1))
    nm2 = "время суток: лонг %02d–%02d UTC (%+.3f%%), шорт %02d–%02d (%+.3f%%)" % (
        best_h, (best_h + 4) % 24, means[best_h] * 100, worst_h, (worst_h + 4) % 24, means[worst_h] * 100)
    S.append(("N2", nm2, sig,
              lambda T, d: 2 * atr1[max(0, last(t1, T, H1))],
              lambda k0, d, r: ex_time(m, k0, d, 48, 3 * r, r)))

    # ---- N3 экстремальный фандинг
    sig = []
    fr = [x[1] for x in fund]
    for j in range(270, len(fund)):
        win = sorted(fr[j - 270:j])
        hi, lo = win[int(0.95 * 269)], win[int(0.05 * 269)]
        if fr[j] >= hi and fr[j] > 0:
            sig.append((fund[j][0], -1))
        elif fr[j] <= lo:
            sig.append((fund[j][0], 1))
    S.append(("N3", "экстремальный фандинг (72 ч, стоп 2 ATR 1Д)", sig,
              lambda T, d: 2 * atrd[max(0, last(td, T, D1))],
              lambda k0, d, r: ex_time(m, k0, d, 72 * 12, r, r)))

    # ---- N4 отскок после резкой свечи 4Ч
    sig = []
    for i in range(20, len(h4)):
        T = h4[i][0] + H4
        j = last(td, T, D1)
        if j < 0 or sma200[j] is None:
            continue
        o, cl = h4[i][1], h4[i][4]
        a = atr4abs[i]
        if dcl[j] > sma200[j] and (o - cl) >= 2.5 * a:
            sig.append((T, 1))
        elif dcl[j] < sma200[j] and (cl - o) >= 2.5 * a:
            sig.append((T, -1))
    S.append(("N4", "отскок после резкой свечи 4Ч", sig,
              lambda T, d: 2 * atr4[last(t4, T, H4)],
              lambda k0, d, r: ex_tp_sl(m, k0, d, r, r, 3 * 288)))

    # ---- N5 пробой после сжатия 4Ч
    sd = []
    for i in range(len(cl4)):
        if i < 19:
            sd.append(None)
            continue
        w = cl4[i - 19:i + 1]
        mu = sum(w) / 20
        sd.append(math.sqrt(sum((x - mu) ** 2 for x in w) / 20))
    width = []
    for i in range(len(cl4)):
        if sd[i] is None:
            width.append(None)
        else:
            mu = sum(cl4[i - 19:i + 1]) / 20
            width.append(4 * sd[i] / mu if mu else None)
    sig = []
    for i in range(140, len(h4)):
        ws = [x for x in width[i - 119:i + 1] if x is not None]
        if len(ws) < 100:
            continue
        mn = min(ws)
        if mn not in width[i - 5:i + 1]:
            continue
        mu = sum(cl4[i - 19:i + 1]) / 20
        up, dn = mu + 2 * sd[i], mu - 2 * sd[i]
        if cl4[i] > up:
            sig.append((h4[i][0] + H4, 1))
        elif cl4[i] < dn:
            sig.append((h4[i][0] + H4, -1))
    S.append(("N5", "пробой после сжатия 4Ч (стоп 2, трейлинг 3 ATR)", sig,
              lambda T, d: atr4abs[last(t4, T, H4)],
              lambda k0, d, a: ex_trail(m, k0, d, 2 * a, 3 * a)))

    # ---- N6 выходной гэп
    def gap_info(T):
        k = m.k_at(T)
        kp = m.k_at(T - 50 * H1) - 1
        if kp < 0 or k >= m.n:
            return None
        pf, ps = m.c[kp], m.o[k]
        g = abs(ps / pf - 1) * 100
        return g if g >= 1.0 else None

    sig = []
    for T in range(START - START % D1, m.t[-1], H1):
        dt = datetime.fromtimestamp(T / 1000, tz=timezone.utc)
        if dt.weekday() == 6 and dt.hour == 23:
            k, kp = m.k_at(T), m.k_at(T - 50 * H1) - 1
            if kp < 0 or k >= m.n:
                continue
            pf, ps = m.c[kp], m.o[k]
            if abs(ps / pf - 1) >= 0.01:
                sig.append((T, -1 if ps > pf else 1))
    S.append(("N6", "выходной гэп (к пятничной цене)", sig,
              lambda T, d: gap_info(T),
              lambda k0, d, g: ex_tp_sl(m, k0, d, g, g, 5 * 288)))
    return S


# ============================================================ прогон

def run(m, fund, rnd, label):
    say("")
    say("#" * 118)
    say("ПРОГОН: %s" % label)
    S = build(m, fund)
    res = {}
    for code, nm, sig, info_at, trade in S:
        trades = []                                         # (T, R, t_exit)
        busy = 0
        for T, d in sorted(sig):
            if T < START + 120 * D1 or T < busy:
                continue
            k0 = m.k_at(T)
            if k0 >= m.n - 2:
                continue
            inf = info_at(T, d)
            if inf is None or (isinstance(inf, float) and not inf > 0):
                continue
            r = trade(k0, d, inf)
            if r is None:
                continue
            trades.append((T, r[0], r[1]))
            busy = r[1]
        # случайный вход: на каждую сделку проверки — N_RAND случайных моментов того же года
        rand = []
        per_year = {}
        for T, R_, te in trades:
            if T >= TEST_FROM:
                per_year[year_of(T)] = per_year.get(year_of(T), 0) + 1
        for y, cnt in per_year.items():
            y0 = int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            y1 = min(int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000), m.t[-1] - 10 * D1)
            got = tries = 0
            while got < cnt * N_RAND and tries < cnt * N_RAND * 20:
                tries += 1
                T = rnd.randrange(y0, y1)
                T -= T % H1
                d = rnd.choice((1, -1))
                k0 = m.k_at(T)
                if k0 >= m.n - 2:
                    continue
                inf = info_at(T, d)
                if inf is None or (isinstance(inf, float) and not inf > 0):
                    continue
                r = trade(k0, d, inf)
                if r is None:
                    continue
                rand.append(r[0])
                got += 1
        res[code] = (nm, trades, rand)
        test = [x[1] for x in trades if x[0] >= TEST_FROM]
        say("  %s %-58s сделок %4d (проверка %4d) · средняя на проверке %+.3f R" %
            (code, nm[:58], len(trades), len(test), (sum(test) / len(test)) if test else 0.0))
    return res


def stats(v):
    if not v:
        return 0, 0.0, 0.0, 0.0
    n = len(v)
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, n - 1))
    return n, mu, 100 * sum(1 for x in v if x > 0) / n, (mu / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def main():
    t0 = time.time()
    say("ЛАБОРАТОРИЯ BTC · %s · исполнение 5m · комиссия %.2f%% · риск %.0f $ на сделку"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), FEE, RISK))
    offline = os.environ.get("BTC_LAB_OFFLINE") == "1"
    now = int(time.time() * 1000)
    if offline:
        say("  ! OFFLINE: вместо данных Binance — синтетика (только проверка кода)")
        b5, fund = synth(START, now, int(os.environ.get("BTC_LAB_SEED") or 12345))
    else:
        b5 = load_btc_5m()
        try:
            fund = load_funding()
        except Exception as e:                               # noqa: BLE001
            say("  ! фандинг не загружен (%s) — N3 без сделок" % e)
            fund = []
    real = Mkt(b5)
    sb, sf = synth(real.t[0], real.t[-1] + M5, 777)
    ctrl = Mkt(sb)
    rnd = random.Random(2026)
    R_real = run(real, fund, rnd, "РЕАЛЬНЫЙ BTC")
    R_ctrl = run(ctrl, sf, rnd, "КОНТРОЛЬ — СЛУЧАЙНЫЕ ЦЕНЫ (у всех должно быть около нуля минус комиссии)")

    say("")
    say("=" * 118)
    say("ИТОГИ ПО СТРАТЕГИЯМ (R — в рисках сделки; $ — при риске %.0f $ на сделку)" % RISK)
    verdicts = []
    for code, (nm, trades, rand) in R_real.items():
        dev = [x[1] for x in trades if x[0] < TEST_FROM]
        test = [x for x in trades if x[0] >= TEST_FROM]
        tv = [x[1] for x in test]
        n_d, mu_d, wr_d, _ = stats(dev)
        n_t, mu_t, wr_t, t_t = stats(tv)
        n_r, mu_r, _, _ = stats(rand)
        cv = [x[1] for x in R_ctrl[code][1] if x[0] >= TEST_FROM]
        n_c, mu_c, _, t_c = stats(cv)
        say("")
        say("%s %s" % (code, nm))
        say("  разработка 2017–2021: сделок %4d · WR %4.1f%% · %+.3f R/сд · итог %+7.0f $"
            % (n_d, wr_d, mu_d, sum(dev) * RISK))
        say("  ПРОВЕРКА 2022–2026:   сделок %4d · WR %4.1f%% · %+.3f R/сд · итог %+7.0f $ · t = %+.1f · %.1f сделки в месяц"
            % (n_t, wr_t, mu_t, sum(tv) * RISK, t_t, n_t / max(1, (real.t[-1] - TEST_FROM) / (30 * D1))))
        yrs = {}
        for T, R_, te in test:
            yrs.setdefault(year_of(T), []).append(R_)
        say("  по годам: " + " · ".join("%d: %+.0f $ (%d)" % (y, sum(v) * RISK, len(v)) for y, v in sorted(yrs.items())))
        mid = TEST_FROM + (real.t[-1] - TEST_FROM) // 2
        h1_ = sum(x[1] for x in test if x[0] < mid)
        h2_ = sum(x[1] for x in test if x[0] >= mid)
        say("  половины проверки: %+.0f $ / %+.0f $" % (h1_ * RISK, h2_ * RISK))
        say("  случайный вход (те же выходы): %+.3f R/сд (%d входов) · случайные цены: %+.3f R/сд, t = %+.1f (%d сделок)"
            % (mu_r, n_r, mu_c, t_c, n_c))
        c1 = n_t >= 30 and sum(tv) > 0 and t_t >= 2.0
        c2 = sum(1 for y in TEST_YEARS if sum(yrs.get(y, [0.0])) > 0) >= 3
        c3 = h1_ > 0 and h2_ > 0
        c4 = mu_t > mu_r
        c5 = not (n_c >= 30 and t_c >= 2.0 and mu_c > 0)
        ok = c1 and c2 and c3 and c4
        v = ("ТЕСТ ПОД ПОДОЗРЕНИЕМ (на случайных ценах плюс)" if not c5 else
             "РАБОТАЕТ" if ok else "не работает")
        say("  критерии: итог+, >=30 сд, t>=2 [%s] · 3 из 5 лет [%s] · обе половины [%s] · лучше случайного [%s] · "
            "контроль чистый [%s] → %s" % tuple(["да" if x else "нет" for x in (c1, c2, c3, c4, c5)] + [v]))
        verdicts.append((code, nm, v, mu_t, sum(tv) * RISK))
    say("")
    say("=" * 118)
    say("СВОДКА (объявленный до прогона вердикт):")
    for code, nm, v, mu, tot in verdicts:
        say("  %s %-60s %+.3f R/сд · %+7.0f $ · %s" % (code, nm[:60], mu, tot, v))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОДБОР НАСТРОЕК ИНДИКАТОРА ПОД КАЖДУЮ МОНЕТУ — ПРОВЕРКА НА СЛЕДУЮЩЕМ ГОДУ (walk-forward).

Для каждой монеты отдельно перебираются настройки Trand-Test-2 (как вручную в TradingView):
  ТФ графика 30m / 1h · старший ТФ индикатора 4ч / 1Д · фильтр «тренд BTC» 1ч / 4ч / 1Д ·
  цель 3 / 4 / 5 / 6 × ATR · стоп 2 / 3 / 3.5 / 4 × ATR  →  192 варианта на монету.
По году N для монеты выбирается лучшая настройка (не меньше 30 сделок), и смотрим, как ИМЕННО
она отработала в году N+1 (подбор его не видел). Переходы: 2022→23, 2023→24, 2024→25, 2025→26.

Сделка: сигнал индикатора (основной + откат, как в тестах), вход по открытию 15-минутки после
закрытия сигнальной свечи, цель/стоп от цены входа в × ATR графика; внутри 15m-свечи, где
задеты и цель, и стоп, — стоп. Комиссия 0.11% за круг. Не больше 30 дней в сделке.
Результат считается в R (R = риск на стопе) и переводится в доллары при риске 20 $.

КРИТЕРИЙ (объявлен ДО прогона). Подбор под монету работает, если в 3 из 4 проверочных лет
сделки по выбранным для каждой монеты настройкам в сумме по всем монетам
  1) в плюсе и
  2) лучше одной общей настройки (30m, старший 1Д, BTC 1Д, цель 5, стоп 3.5 ATR).

Лежит рядом с compare_test.py и multi_test.py (данные 15m — из кэша multi_test).
Нужен numpy (в workflow ставится). Отчёт: coin_wf_report.txt
"""
import bisect
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

import compare_test as c
import multi_test as mt

REPORT = "coin_wf_report.txt"
M15, M30 = 900000, 1800000
TFS = [("30m", M30), ("1h", c.H1)]
HTFS = ["4ч", "1Д"]
BTCS = [("1ч", c.H1), ("4ч", c.H4), ("1Д", c.D1)]
TPS = [3.0, 4.0, 5.0, 6.0]
SLS = [2.0, 3.0, 3.5, 4.0]
EXITS = [(tp, sl) for tp in TPS for sl in SLS]
FEE = 0.11 / 100
MAX_BARS = 30 * 96                  # 30 дней 15-минуток
MIN_N = 30
RISK = 20.0
COMMON = ("30m", "1Д", "1Д", 5.0, 3.5)
YEARS = [2022, 2023, 2024, 2025, 2026]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def year_of(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).year


def ind_entries_tf(b, tf_ms, htf_t, htf_up, htf_ms):
    """Сигналы Trand-Test-2 (как ideas_test.ind_entries_htf), для любого ТФ графика."""
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
        close_t = b[i][0] + tf_ms
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


def exits_R(o, h, l, cl, k0, d, atrp):
    """R для всех EXITS одной сделки: первый бар, где задета цель, против первого бара, где стоп."""
    k1 = min(len(o), k0 + MAX_BARS)
    p0 = o[k0]
    if d == 1:
        fav = np.maximum.accumulate((h[k0:k1] - p0) / p0)
        adv = np.maximum.accumulate((p0 - l[k0:k1]) / p0)
    else:
        fav = np.maximum.accumulate((p0 - l[k0:k1]) / p0)
        adv = np.maximum.accumulate((h[k0:k1] - p0) / p0)
    last = d * (cl[k1 - 1] - p0) / p0
    a = atrp / 100
    res = []
    for tp, sl in EXITS:
        tpx, slx = tp * a, sl * a
        it = int(np.searchsorted(fav, tpx, side="left"))
        is_ = int(np.searchsorted(adv, slx, side="left"))
        n = len(fav)
        if is_ < n and is_ <= it:
            r = -1.0
        elif it < n:
            r = tp / sl
        else:
            r = last / slx
        res.append(r - FEE / slx)
    return res


def main():
    t0 = time.time()
    coins = list(mt.MONEY)
    say("ПОДБОР ПОД КАЖДУЮ МОНЕТУ, ПРОВЕРКА НА СЛЕДУЮЩЕМ ГОДУ · %s · монет %d · настроек на монету %d"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(coins),
           len(TFS) * len(HTFS) * len(BTCS) * len(EXITS)))
    with ThreadPoolExecutor(max_workers=3) as ex:
        data = dict(ex.map(mt.load_coin, ["BTC"] + coins))
    b15b, h4b, d1b = data["BTC"]
    btc = {}
    for name, ms in BTCS:
        bb = d1b if ms == c.D1 else c.agg(b15b, ms)
        btc[name] = ([x[0] for x in bb], c.trend_arr(bb), ms)

    def btc_tr(name, T):
        tt, tr, ms = btc[name]
        i = bisect.bisect_right(tt, T - ms) - 1
        return tr[i] if i >= 0 else 0

    # агрегаты: (монета, tf, htf, btc, tp, sl, год) -> [сумма R, сделок]
    agg = defaultdict(lambda: [0.0, 0])
    for sym in coins:
        b15, h4, d1 = data.get(sym) or ([], [], [])
        if len(b15) < 50000 or len(d1) < 260:
            say("  %s: мало данных — пропуск" % sym)
            continue
        t15 = np.array([x[0] for x in b15], dtype=np.int64)
        o = np.array([x[1] for x in b15])
        h = np.array([x[2] for x in b15])
        l = np.array([x[3] for x in b15])
        cl = np.array([x[4] for x in b15])
        h4a = c.agg(b15, c.H4)
        htf_src = {"4ч": (h4a, c.H4), "1Д": (d1, c.D1)}
        htf = {}
        for k, (bb, ms) in htf_src.items():
            cc = [x[4] for x in bb]
            htf[k] = ([x[0] for x in bb], [a > b_ for a, b_ in zip(c.ema(cc, 9), c.ema(cc, 21))], ms)
        n_sig = 0
        for tfn, tfms in TFS:
            b = c.agg(b15, tfms)
            atrp = c.atr_pct(b)
            for hk in HTFS:
                ht, hu, hms = htf[hk]
                for T, d, i in ind_entries_tf(b, tfms, ht, hu, hms):
                    if T < mt.START or atrp[i] <= 0:
                        continue
                    k0 = int(np.searchsorted(t15, T, side="left"))
                    if k0 >= len(t15) - 1:
                        continue
                    y = year_of(T)
                    rs = exits_R(o, h, l, cl, k0, d, atrp[i])
                    n_sig += 1
                    for bn, _ in BTCS:
                        if btc_tr(bn, T) != d:
                            continue
                        for (tp, sl), r in zip(EXITS, rs):
                            a = agg[(sym, tfn, hk, bn, tp, sl, y)]
                            a[0] += r
                            a[1] += 1
        say("  %s: сигналов (все ТФ) %d · %.0f с" % (sym, n_sig, time.time() - t0))

    keys_by_coin_year = defaultdict(list)
    for k in agg:
        keys_by_coin_year[(k[0], k[6])].append(k)

    def best(sym, y):
        cand = [k for k in keys_by_coin_year.get((sym, y), []) if agg[k][1] >= MIN_N]
        return max(cand, key=lambda k: agg[k][0]) if cand else None

    def short(k):
        return "%s/%s/BTC%s ц%g с%g" % (k[1], k[2], k[3], k[4], k[5])

    say("")
    say("=" * 118)
    say("ПО МОНЕТАМ: выбранная по году N настройка → результат в году N+1 (R на сделку; $ при риске %.0f $)" % RISK)
    wins = 0
    trans = [(y, y + 1) for y in YEARS[:-1]]
    summary = []
    for y0, y1 in trans:
        say("")
        say("  %d → %d" % (y0, y1))
        tot_r = tot_n = com_r = com_n = is_r = is_n = 0.0
        pos = cnt = 0
        for sym in coins:
            k = best(sym, y0)
            if not k:
                continue
            ko = k[:6] + (y1,)
            r1, n1 = agg.get(ko, [0.0, 0])
            r0, n0 = agg[k]
            kc = (sym,) + COMMON + (y1,)
            rc, nc = agg.get(kc, [0.0, 0])
            com_r += rc
            com_n += nc
            if n1 == 0:
                say("    %-5s %-26s подбор %+.2f R (%d сд) → в %d сделок нет" % (sym, short(k), r0 / n0, n0, y1))
                continue
            tot_r += r1
            tot_n += n1
            is_r += r0
            is_n += n0
            cnt += 1
            pos += r1 > 0
            say("    %-5s %-26s подбор %+.2f R (%3d сд) → проверка %+.2f R (%3d сд) = %+6.0f $ %s"
                % (sym, short(k), r0 / n0, n0, r1 / n1, n1, r1 * RISK, "✔" if r1 > 0 else "✖"))
        a_ind = tot_r / tot_n if tot_n else 0.0
        a_com = com_r / com_n if com_n else 0.0
        ok = tot_r > 0 and a_ind > a_com
        wins += ok
        summary.append((y0, y1, is_r / is_n if is_n else 0.0, a_ind, tot_r, a_com, com_r, pos, cnt, ok))
        say("    ИТОГО %d: своя настройка монеты %+.3f R/сделку, итог %+.0f $ · общая настройка %+.3f R/сделку, "
            "итог %+.0f $ · монет в плюсе %d из %d → %s"
            % (y1, a_ind, tot_r * RISK, a_com, com_r * RISK, pos, cnt, "да" if ok else "нет"))

    say("")
    say("=" * 118)
    say("СВОДКА (R на сделку): на подборе → на проверке · общая настройка")
    for y0, y1, isr, a, tr_, ac, cr, pos, cnt, ok in summary:
        say("  %d→%d: %+.3f → %+.3f R (итог %+.0f $) · общая %+.3f R (%+.0f $) · монет в плюсе %d/%d · %s"
            % (y0, y1, isr, a, tr_ * RISK, ac, cr * RISK, pos, cnt, "✔" if ok else "✖"))
    say("  (разница «на подборе» и «на проверке» — сколько было подгонки)")
    say("")
    say("ВЕРДИКТ (объявлен до прогона): подбор под монету работает, если в 3 из 4 лет своя настройка в плюсе "
        "и лучше общей → %s (%d из 4)" % ("ПОДБОР РАБОТАЕТ" if wins >= 3 else "подбор не работает", wins))
    say("")
    say("Настройки, выбранные по 2026 году (для торговли — только если подбор работает):")
    for sym in coins:
        k = best(sym, 2026)
        if k:
            say("  %-5s %-26s %+.2f R/сделку (%d сд) = %+.0f $ при риске %.0f $"
                % (sym, short(k), agg[k][0] / agg[k][1], agg[k][1], agg[k][0] * RISK, RISK))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

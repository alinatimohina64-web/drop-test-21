#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПРАВИЛО SMA200 + ТОРГОВАТЬ ТОЛЬКО МОНЕТЫ «В ПЛЮСЕ» — работает ли отбор монет по прошлому?

R2 = TT2 без MRC, цель 4 / стоп 20 ATR, входы только когда BTC вчера закрылся выше SMA200 (как в sma200_test.py:
30m, ст. ТФ 4ч, тренд BTC 1Д, 15m Binance, позиция 1000 $, в спорной свече — стоп, одна позиция на монету).
Монеты: основной + новый набор вместе (41).

ОТБОР БЕЗ ПОДГЛЯДЫВАНИЯ (скользящее окно по годам): в начале каждого года Y (2021…2026) монета «в плюсе»,
если её итог R2 по ЗАКРЫТЫМ до 1 января Y сделкам (с 2019 года) больше нуля и сделок не меньше 15.
В году Y торгуем только отобранные. Сравнение: отобранные против отброшенных и против всех монет.
Плюс простое деление: отбор по 2019–2022 → проверка 2023–2026.

КРИТЕРИЙ (объявлен ДО прогона), отбор «работает», если:
  1) средняя сделка отобранных за все годы проверки выше, чем у всех монет;
  2) выше, чем у отброшенных, не меньше чем в 4 из 6 лет (годы, где у обеих групп ≥ 20 сделок);
  3) на делении 2019–2022 → 2023–2026 средняя сделка отобранных выше, чем у отброшенных.
Справочно: одна и три позиции за раз среди отобранных (первый по времени сигнал), итог и просадка.
Лежит рядом с sma200_test.py, sma200_check_test.py, coin_select_test.py. Отчёт: sma200_coins_report.txt
"""
import time
from datetime import datetime, timezone

import coin_select_test as cs
import sma200_check_test as ck
import sma200_test as st

REPORT = "sma200_coins_report.txt"
CB = ("none", 4.0, 20.0)
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def ms(y):
    return int(datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def avg(v):
    return sum(x[1] for x in v) / len(v) if v else 0.0


def main():
    t0 = time.time()
    say("SMA200 + ОТБОР МОНЕТ «В ПЛЮСЕ» · %s · Binance 15m · позиция 1000 $"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    btc = st.Btc()
    trades = {}            # монета -> сделки R2
    days = {}
    end_t = 0
    for nm, coins in (("основной", st.MAIN_COINS), ("новый", st.NEW_COINS)):
        seq, rnd, dd, have, et = st.run_set(nm, coins, btc)
        for s in have:
            trades[s] = sorted(seq[(CB, True)].get(s, []))
        days.update(dd)
        end_t = max(end_t, et)
    coins = sorted(trades)

    say("")
    say("=" * 110)
    say("ИТОГ R2 ПО МОНЕТАМ (весь период): " + ", ".join(
        "%s %+.0f (%d)" % (s, sum(x[1] for x in trades[s]), len(trades[s]))
        for s in sorted(coins, key=lambda s: -sum(x[1] for x in trades[s]))))

    say("")
    say("=" * 110)
    say("СКОЛЬЗЯЩИЙ ОТБОР ПО ГОДАМ (монета «в плюсе» по закрытым сделкам до 1 января)")
    all_sel, all_rej, all_all = [], [], []
    yr_better = yr_n = 0
    sel_by_year = {}
    for y in YEARS:
        a, b = ms(y), ms(y + 1)
        sel, rej = [], []
        for s in coins:
            past = [x for x in trades[s] if x[3] is not None and x[3] < a]
            (sel if (len(past) >= 15 and sum(x[1] for x in past) > 0) else rej).append(s)
        sel_by_year[y] = set(sel)
        vs = [x for s in sel for x in trades[s] if a <= x[0] < b]
        vr = [x for s in rej for x in trades[s] if a <= x[0] < b]
        all_sel += vs
        all_rej += vr
        all_all += vs + vr
        cmp_ = ""
        if len(vs) >= 20 and len(vr) >= 20:
            yr_n += 1
            if avg(vs) > avg(vr):
                yr_better += 1
                cmp_ = " → отобранные лучше"
            else:
                cmp_ = " → отобранные НЕ лучше"
        say("  %d: отобрано %2d из %2d · отобранные %+6.0f $ (%4d сд, %+6.2f/сд) · отброшенные %+6.0f $ (%4d сд, %+6.2f/сд)%s"
            % (y, len(sel), len(coins), sum(x[1] for x in vs), len(vs), avg(vs), sum(x[1] for x in vr), len(vr), avg(vr), cmp_))
    say("  ВСЕ ГОДЫ: отобранные %+.0f $ (%d сд, %+.2f/сд) · отброшенные %+.0f $ (%d сд, %+.2f/сд) · все монеты %+.2f/сд"
        % (sum(x[1] for x in all_sel), len(all_sel), avg(all_sel), sum(x[1] for x in all_rej), len(all_rej), avg(all_rej),
           avg(all_all)))

    say("")
    say("=" * 110)
    say("ДЕЛЕНИЕ: отбор по 2019–2022 → проверка 2023–2026")
    a = ms(2023)
    sel = [s for s in coins if sum(x[1] for x in trades[s] if x[3] is not None and x[3] < a) > 0
           and sum(1 for x in trades[s] if x[3] is not None and x[3] < a) >= 15]
    rej = [s for s in coins if s not in sel]
    vs = [x for s in sel for x in trades[s] if x[0] >= a]
    vr = [x for s in rej for x in trades[s] if x[0] >= a]
    say("  отобраны (%d): %s" % (len(sel), ", ".join(sel)))
    say("  проверка 2023–2026: отобранные %+.0f $ (%d сд, %+.2f/сд) · отброшенные %+.0f $ (%d сд, %+.2f/сд)"
        % (sum(x[1] for x in vs), len(vs), avg(vs), sum(x[1] for x in vr), len(vr), avg(vr)))
    k3 = avg(vs) > avg(vr)

    say("")
    say("=" * 110)
    say("СПРАВОЧНО: МАЛО ПОЗИЦИЙ ЗА РАЗ среди монет, отобранных скользящим отбором (2021–2026)")
    pool = [x for x in all_sel]
    pool_all = [x for x in all_all]
    for cap in (1, 3, 10):
        for nm, src in (("только отобранные", pool), ("все монеты", pool_all)):
            tr = ck.capped(src, cap)
            n, tot, mo, rat = ck.line(tr, days, end_t)
            say("  до %2d позиций · %-17s сделок %4d · итог %+7.0f $ · просадка %+7.0f $ · итог/просадка %.2f"
                % (cap, nm, n, tot, mo["dd"], rat))

    k1 = avg(all_sel) > avg(all_all)
    k2 = yr_n > 0 and yr_better >= 4
    say("")
    say("=" * 110)
    say("КРИТЕРИЙ (объявлен до прогона): лучше всех монет [%s, %+.2f против %+.2f] · лучше отброшенных в 4 из 6 лет "
        "[%s, %d из %d] · деление 2019–22 → 2023–26 [%s] → %s"
        % ("да" if k1 else "нет", avg(all_sel), avg(all_all), "да" if k2 else "нет", yr_better, yr_n,
           "да" if k3 else "нет", "ОТБОР МОНЕТ РАБОТАЕТ" if (k1 and k2 and k3) else "отбор монет НЕ работает"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

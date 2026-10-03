#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
«ЗАМОК» ДЛЯ ЗАВИСШЕГО ЛОНГА (реконструкция по скриншотам калькулятора из группы) против «просто держать».

Ситуация как у счёта A: лонг купили на хаях, цена ушла на DD = 45% ниже средней (2.78 → ≈ 1.53).
Старт симуляции — в начале каждого месяца 2019…2025, когда условно «лонг уже в минусе на 45%»:
средняя лонга = цена старта / (1 − 0.45). Объём лонга = 1 монета (всё в % от вложенного в лонг).
Данные — Binance 15m → часовые свечи. Комиссия 0.11% за круг на каждый объём; фандинг не считается
(лонг и шорт его почти взаимно гасят). Внутри свечи худший порядок: сначала то, что хуже для счёта.

H0  «ДЕРЖАТЬ»: ждать, пока цена вернётся к средней лонга (+0.2%) — тогда «разрулено».
H1  «ЗАМОК» (как в калькуляторе):
    · сразу шорт того же объёма, что лонг (замок; чистая позиция ≈ 0);
    · ШОРТ: тейк 3% от средней шорта → закрыть весь шорт с прибылью и тут же открыть базовый шорт заново (цикл);
      усреднение шорта на росте: +5 / +10 / +20 / +35 / +55% от средней, объёмы 0.05 / 0.1 / 0.2 / 0.4 / 0.8 базы;
    · ЛОНГ: усреднение на падении: −5 / −10 / −20 / −35 / −55% от последней покупки, объёмы 0.05 / 0.1 / 0.2 / 0.4 / 0.8;
      тейк 2% от средней лонга → закрыть лонг И весь шорт по рынку — «разрулено».
    Результат = вся закрытая прибыль циклов + закрытие обеих сторон.
Для каждого старта: разрулено ли и за сколько дней, итог в момент выхода (или на конец данных),
худшее состояние счёта по пути (закрытое + незакрытое), сколько всего объёма было в позициях (нужная маржа).

КРИТЕРИЙ (объявлен ДО прогона): замок «помогает разрулить», если
  1) на XRP доля стартов, разрулённых за 12 месяцев, у H1 выше, чем у H0, и медианный итог H1 в момент выхода
     не хуже −10% от вложенного в лонг;
  2) то же на контрольных монетах (ETH, ADA, LINK, LTC, DOGE) — не меньше чем на 4 из 5;
  3) худшее состояние счёта H1 по всем стартам не хуже, чем у H0 (замок не должен увеличивать риск).
Лежит рядом с coin_select_test.py (данные и загрузка оттуда). Отчёт: lock_report.txt
"""
import time
from datetime import datetime, timezone

import numpy as np

import coin_select_test as cs

REPORT = "lock_report.txt"
H1MS = 3600000
DD = 0.45
FEE = 0.11 / 100
S_TP, L_TP = 0.03, 0.02
S_STEPS, L_STEPS = [0.05, 0.10, 0.20, 0.35, 0.55], [0.05, 0.10, 0.20, 0.35, 0.55]
ADD_Q = [0.05, 0.10, 0.20, 0.40, 0.80]
COINS = ["XRP", "ETH", "ADA", "LINK", "LTC", "DOGE"]
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def to_1h(b15):
    out = {}
    for t, o, h, l, c, v in b15:
        k = t - t % H1MS
        if k not in out:
            out[k] = [k, o, h, l, c]
        else:
            x = out[k]
            x[2] = max(x[2], h)
            x[3] = min(x[3], l)
            x[4] = c
    return [out[k] for k in sorted(out)]


def run_hold(bars, i0, pe):
    worst = 0.0
    for i in range(i0, len(bars)):
        t, o, h, l, c = bars[i]
        worst = min(worst, l / pe - 1)
        if h >= pe * 1.002:
            return True, (t - bars[i0][0]) / 86400000, -FEE, worst
    return False, (bars[-1][0] - bars[i0][0]) / 86400000, bars[-1][4] / pe - 1 - FEE, worst


def run_lock(bars, i0, pe):
    p0 = bars[i0][4]
    lq, lcost = 1.0, pe            # лонг: монет и стоимость (средняя pe)
    l_last, l_n = p0, 0            # последняя покупка лонга (старт — как будто «последняя» на уровне старта)
    sq, scost, s_n = 1.0, p0, 0    # шорт: монет и стоимость
    realized = -FEE * p0           # открытие шорта
    worst = lq * p0 - lcost        # незакрытое лонга на старте (≈ −45%)
    max_gross = (lq + sq) * p0
    for i in range(i0 + 1, len(bars)):
        t, o, h, l, c = bars[i]
        # худший порядок: сначала проверяем незакрытое на неблагоприятных краях свечи
        for px in (h, l):
            eq = realized + (lq * px - lcost) + (scost - sq * px)
            worst = min(worst, eq)
        # усреднение шорта на росте
        s_avg = scost / sq
        while s_n < len(S_STEPS) and h >= s_avg * (1 + S_STEPS[s_n]):
            px = s_avg * (1 + S_STEPS[s_n])
            q = ADD_Q[s_n]
            sq += q
            scost += q * px
            realized -= FEE * q * px
            s_n += 1
        # усреднение лонга на падении
        while l_n < len(L_STEPS) and l <= l_last * (1 - L_STEPS[l_n]):
            px = l_last * (1 - L_STEPS[l_n])
            q = ADD_Q[l_n]
            lq += q
            lcost += q * px
            realized -= FEE * q * px
            l_last = px
            l_n += 1
        max_gross = max(max_gross, (lq + sq) * c)
        # тейк лонга → разрулено (закрыть всё)
        l_avg = lcost / lq
        if h >= l_avg * (1 + L_TP):
            px = l_avg * (1 + L_TP)
            total = realized + (lq * px - lcost) + (scost - sq * px) - FEE * (lq + sq) * px
            return True, (t - bars[i0][0]) / 86400000, total / pe, worst / pe, max_gross / pe
        # тейк шорта → закрыть шорт, открыть базовый заново (цикл)
        s_avg = scost / sq
        if l <= s_avg * (1 - S_TP):
            px = s_avg * (1 - S_TP)
            realized += scost - sq * px - FEE * sq * px
            sq, scost, s_n = 1.0, px, 0
            realized -= FEE * px
    px = bars[-1][4]
    total = realized + (lq * px - lcost) + (scost - sq * px) - FEE * (lq + sq) * px
    return False, (bars[-1][0] - bars[i0][0]) / 86400000, total / pe, worst / pe, max_gross / pe


def main():
    t0 = time.time()
    say("ЗАМОК ДЛЯ ЗАВИСШЕГО ЛОНГА (минус %.0f%% на старте) против «держать» · %s · Binance 1h"
        % (DD * 100, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))
    verdict = {}
    worst_h0 = worst_h1 = 0.0
    for sym in COINS:
        b15 = cs.load_15m(sym)
        if len(b15) < 20000:
            say("  %s: мало данных" % sym)
            continue
        bars = to_1h(b15)
        tt = [b[0] for b in bars]
        starts = []
        y, m = 2019, 1
        while (y, m) <= (2025, 12):
            ts = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
            i0 = int(np.searchsorted(tt, ts))
            if 0 < i0 < len(bars) - 24 * 60:
                starts.append(i0)
            m += 1
            if m == 13:
                y, m = y + 1, 1
        h0 = [run_hold(bars, i0, bars[i0][4] / (1 - DD)) for i0 in starts]
        h1 = [run_lock(bars, i0, bars[i0][4] / (1 - DD)) for i0 in starts]
        n = len(starts)
        r0 = sum(1 for x in h0 if x[0] and x[1] <= 365) / max(1, n)
        r1 = sum(1 for x in h1 if x[0] and x[1] <= 365) / max(1, n)
        d0 = np.median([x[1] for x in h0 if x[0]]) if any(x[0] for x in h0) else float("nan")
        d1 = np.median([x[1] for x in h1 if x[0]]) if any(x[0] for x in h1) else float("nan")
        f0 = np.median([x[2] for x in h0])
        f1 = np.median([x[2] for x in h1])
        w0 = min(x[3] for x in h0)
        w1 = min(x[3] for x in h1)
        g1 = max(x[4] for x in h1)
        worst_h0, worst_h1 = min(worst_h0, w0), min(worst_h1, w1)
        verdict[sym] = r1 > r0 and f1 >= -0.10
        say("")
        say("=" * 110)
        say("%s · стартов %d (каждый месяц 2019–2025)" % (sym, n))
        say("  ДЕРЖАТЬ: разрулено за 12 мес %.0f%% стартов · медиана дней до разрулки %.0f · итог (медиана) %+.1f%% · "
            "худшее по пути %+.0f%%" % (100 * r0, d0, 100 * f0, 100 * w0))
        say("  ЗАМОК:   разрулено за 12 мес %.0f%% стартов · медиана дней до разрулки %.0f · итог (медиана) %+.1f%% · "
            "худшее по пути %+.0f%% · объём в позициях до %.1f× вложенного"
            % (100 * r1, d1, 100 * f1, 100 * w1, g1))
        say("  → замок лучше по критерию 1: %s" % ("да" if verdict[sym] else "нет"))
    say("")
    say("=" * 110)
    k1 = verdict.get("XRP", False)
    ctrl = [s for s in COINS if s != "XRP" and s in verdict]
    k2 = sum(1 for s in ctrl if verdict[s]) >= 4
    k3 = worst_h1 >= worst_h0
    say("КРИТЕРИЙ: XRP [%s] · контрольные монеты %d из %d [%s] · худшее по пути: замок %+.0f%% против держать %+.0f%% [%s]"
        % ("да" if k1 else "нет", sum(1 for s in ctrl if verdict[s]), len(ctrl), "да" if k2 else "нет",
           100 * worst_h1, 100 * worst_h0, "да" if k3 else "нет"))
    say("→ ИТОГ: %s" % ("ЗАМОК ПОМОГАЕТ РАЗРУЛИТЬ ЗАВИСШИЙ ЛОНГ" if (k1 and k2 and k3) else
                        "замок (в такой реконструкции) не лучше простого удержания"))
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

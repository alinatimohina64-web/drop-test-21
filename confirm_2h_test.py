#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ПОДТВЕРЖДЕНИЕ НАХОДКИ ИЗ tf2h_grid_test.py НА НЕЗАВИСИМЫХ МОНЕТАХ.

Находка (перебор 30 вариантов, 24.11.2026): на 2ч графике со старшим ТФ 1Д область цель 5–7 / стоп 6–7 ATR(2ч)
на трёх наборах обыгрывает случайный вход на +0.07…+0.12 R, обе половины в плюсе, контроль на перемешанных ценах
чистый. Но доход на единицу просадки — не лучше базы (1.5–2.0 против 2.1–3.4).

ПРОВЕРЯЕМ ОДИН вариант, объявленный ДО прогона: 2ч / 1Д, цель 6 / стоп 7 ATR(2ч), без порога ⭐, стоп не дальше 80%,
правило SMA200, риск 50 $, 5 слотов. База — 30m / 4ч, 5/20, ⭐. Всё как в tf2h_grid_test.py.
НАБОР: «четвёртый» — монеты, НЕ участвовавшие ни в одном тесте сетки (основной, новый, третий исключены).

КРИТЕРИЙ (объявлен ДО прогона). Находка подтверждена, если на четвёртом наборе ВСЕ:
  1) средний R сделки минус случайный вход ≥ +0.03;
  2) обе половины периода в плюсе (R портфеля);
  3) контроль на перемешанных ценах этого же набора: перевес над случайным ≤ +0.02;
  4) доход на единицу просадки (итог R / |просадка R|) не хуже базы на этом же наборе.
Если 1–3 выполнены, а 4 нет — правило настоящее, но для 5 слотов и риска 50 $ не выгоднее текущего; печатается,
какой риск на сделку уравнял бы просадку с базой и сколько тогда было бы дохода.

Запуск: python confirm_2h_test.py (кэш data_btc/). Отчёт: confirm_2h_report.txt
"""
import time
from datetime import datetime, timezone

import sma200_test as st
import tf_test as tt
import tf2h_grid_test as g

cs = st.cs
REPORT = "confirm_2h_report.txt"
H2, D1 = g.H2, g.D1
g.VARS = [g.BASE, ("g", H2, D1, 6.0, 7.0, 0.0)]

FOURTH = ["XMR", "GMX", "RNDR", "PEPE", "FLOKI", "BONK", "WIF", "JUP", "PYTH", "BLUR", "JTO", "STRK", "ORDI",
          "ETHFI", "W", "ENA", "AR", "CFX", "PENDLE", "RPL", "WOO", "CKB", "IOTX", "GMT", "MASK", "API3", "CVX",
          "BAL", "ZRX", "STORJ", "SKL", "COTI", "JASMY", "AUDIO", "BAND", "CTSI", "DENT", "FXS", "GLM", "ID", "LQTY",
          "MAGIC", "OCEAN", "AGIX", "ACH", "AMP", "ARPA", "CELR", "CHR", "DGB", "DUSK", "FUN", "HIVE", "KNC",
          "LINA", "LSK", "NKN", "OGN", "PERP", "REEF", "RSR", "SXP", "TLM", "TRB", "UMA", "WAVES", "XEM", "YGG"]
_used = set(st.MAIN_COINS) | set(st.NEW_COINS) | set(tt.THIRD)
FOURTH = [x for x in FOURTH if x not in _used]


def say(s=""):
    g.say(s)


def line(name, r):
    if not r:
        say("  %-26s нет сигналов" % name)
        return
    ratio = r["tot"] / -r["dd"] if r["dd"] < 0 else float("inf")
    say("  %-26s сделок %5d · WR %3.0f%% · R %s (случ. %s, перевес %s) · удерж. %.1f дн" % (
        name, r["n"], r["wr"], g.fmt(r["avg"]), g.fmt(r["rnd"]), g.fmt(r["avg"] - r["rnd"]), r["hold"]))
    say("      портфель: взято %d · итог %+.1f R ≈ %+.0f $ · просадка %+.1f R · доход/просадка %.2f · половины %+.1f / %+.1f"
        % (r["taken"], r["tot"], r["tot"] * 50, r["dd"], ratio, r["h1"], r["h2"]))


def main():
    t0 = time.time()
    say("ПОДТВЕРЖДЕНИЕ 2ч / 1Д, цель 6 / стоп 7 · %s · четвёртый набор: %d монет"
        % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), len(FOURTH)))
    say("  " + " ".join(FOURTH))
    real = g.run_set("четвёртый", FOURTH, st.Btc())
    say("")
    say("=" * 110)
    say("ЧЕТВЁРТЫЙ НАБОР")
    line("БАЗА 30m/4ч 5/20 ⭐", real.get(0))
    line("2ч/1Д цель 6 / стоп 7", real.get(1))

    btc_sh = tt.BtcFrom(tt.shuffle_days(cs.load_15m("BTC"), "coin-BTC"))
    ctrl = g.run_set("перемешанные", FOURTH, btc_sh, shuffled=True)
    say("")
    say("КОНТРОЛЬ: перемешанные цены, тот же набор")
    line("БАЗА 30m/4ч 5/20 ⭐", ctrl.get(0))
    line("2ч/1Д цель 6 / стоп 7", ctrl.get(1))

    say("")
    say("=" * 110)
    say("ВЕРДИКТ ПО КРИТЕРИЮ (объявлен до прогона)")
    r, b, cr = real.get(1), real.get(0), ctrl.get(1)
    if not (r and b and cr):
        say("  нет данных для решения")
    else:
        k1 = r["avg"] - r["rnd"] >= 0.03
        k2 = r["h1"] > 0 and r["h2"] > 0
        k3 = cr["avg"] - cr["rnd"] <= 0.02
        rr = r["tot"] / -r["dd"] if r["dd"] < 0 else float("inf")
        rb = b["tot"] / -b["dd"] if b["dd"] < 0 else float("inf")
        k4 = rr >= rb
        say("  1) перевес над случайным %s ≥ +0.03 [%s]" % (g.fmt(r["avg"] - r["rnd"]), "да" if k1 else "нет"))
        say("  2) половины %+.1f / %+.1f R [%s]" % (r["h1"], r["h2"], "да" if k2 else "нет"))
        say("  3) контроль %s ≤ +0.02 [%s]" % (g.fmt(cr["avg"] - cr["rnd"]), "да" if k3 else "нет"))
        say("  4) доход/просадка %.2f против базы %.2f [%s]" % (rr, rb, "да" if k4 else "нет"))
        if r["dd"] < 0 and b["dd"] < 0:
            scale = b["dd"] / r["dd"]
            say("  справочно: чтобы просадка была как у базы, риск на сделку ≈ %.0f $ вместо 50; доход тогда ≈ %+.0f $ "
                "против базы %+.0f $" % (50 * scale, r["tot"] * 50 * scale, b["tot"] * 50))
        if k1 and k2 and k3 and k4:
            say("  → ПОДТВЕРЖДЕНО: 2ч/1Д 6/7 лучше текущего правила и на независимых монетах")
        elif k1 and k2 and k3:
            say("  → правило настоящее (подтвердилось на независимых монетах), но по доходу на просадку не лучше текущего")
        else:
            say("  → НЕ подтвердилось — находка из перебора")
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(g._out) + "\n")


if __name__ == "__main__":
    main()

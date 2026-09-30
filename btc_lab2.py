#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЛАБОРАТОРИЯ BTC — 2. Можно ли хоть немного предугадывать НАПРАВЛЕНИЕ BTC? Одним прогоном.

Данные: BTCUSDT 5m Binance с 2017-08 (кэш data_btc/, как в btc_lab.py), дневки S&P 500, Nasdaq,
VIX, индекс доллара (FRED) и золота (Stooq). Разработка 2017-08…2021-12, ПРОВЕРКА 2022-01…сейчас.

ЧАСТЬ A — «АТЛАС СИТУАЦИЙ» (будущий индикатор вероятности).
  Горизонты: 4 часа (точки каждые 4 ч), 24 часа (каждый день 00:00 UTC), 7 дней (каждый понедельник).
  Точки не перекрываются — иначе статистика завышается. Вопрос: BTC через горизонт выше или ниже?
  Признаки ситуации (мало и заранее, чтобы не наловить случайных «70%»):
    TR  — цена относительно SMA50 и SMA200 дневок (4 состояния);
    MOM — ход за последние 7 дней (для 4 ч — за 24 ч): нижняя / средняя / верхняя треть (границы — по разработке);
    RSI — RSI14 (дневки; для 4 ч — 4-часовки): < 35 / 35–65 / > 65;
    VOL — волатильность (ATR) относительно своей медианы за 90 дней: низкая / обычная / высокая;
    HOUR (4 ч) — блок часов UTC; DOW (24 ч) — день недели.
  Ячейка = одно состояние признака или пара состояний двух признаков. На РАЗРАБОТКЕ отбираются ячейки,
  где доля «выше» отличается от обычной на >= 5 п.п. (и точек достаточно). На ПРОВЕРКЕ каждая точка
  получает прогноз голосованием отобранных ячеек, в которые она попала.
  КРИТЕРИЙ (до прогона): направление угадывается чаще, чем «просто всегда ставить на обычную долю»,
  на >= 3 п.п. и z >= 2. Иначе — закономерностей нет.

ЧАСТЬ C — что предсказуемо на самом деле: РАЗМАХ движения. Будет ли следующий день / неделя
  волатильнее медианы, если текущая волатильность выше медианы? (Нужно для сеток и ширины стопов.)

ЧАСТЬ B — МЕЖРЫНОК (дневки, только прошлое):
  B1 — вчерашний ход S&P / Nasdaq / золота / доллара (обратный знак) / VIX (обратный) → BTC сегодня;
  B2 — ход S&P в пятницу → BTC за выходные;
  B3 — режим: BTC за следующие 7 дней, когда S&P выше/ниже SMA200, доллар выше/ниже SMA50, VIX > 25;
  B4 — справка: корреляция дневных ходов BTC с S&P, Nasdaq, золотом и долларом по годам.
  КРИТЕРИЙ (до прогона): B1/B2 — угадывание на проверке выше обычной доли на >= 3 п.п., z >= 2 и тот же
  знак эффекта на разработке. B3 — разница средних недельных ходов между режимами одного знака на
  разработке и проверке и t >= 2 на проверке.

ЧАСТЬ D — ОБЪЁМ: «цена растёт, а объём падает — скоро вниз» (и зеркально для падения).
  Сигнал: закрытие на максимуме 20 свечей, а средний объём последних 10 свечей меньше 0.8 от
  предыдущих 10. Сравнение — с такими же максимумами, но без падения объёма (иначе проверяем не
  объём, а сам факт максимума). Горизонт: 4Ч-свечи → следующие 24 ч, дневки → следующие 7 дней.
  КРИТЕРИЙ (до прогона): после «максимума на падающем объёме» BTC выше реже, чем после обычного
  максимума, на >= 3 п.п., z >= 2 на проверке и тот же знак на разработке (зеркально для минимумов).

ЧАСТЬ E — МОЖНО ЛИ НА ЭТОМ ЗАРАБОТАТЬ: прогноз атласа на 4 ч как сделка (вход сейчас, выход через
  4 ч по рынку), комиссия 0.11% за круг (тейкер) и 0.04% (лимитки с обеих сторон, для справки).
  КРИТЕРИЙ: средняя сделка на проверке после комиссии тейкера > 0 и t >= 2, и прогноз даёт больше, чем
  простой рост/падение рынка за то же время (сверх дрейфа > 0, t >= 2).

КОНТРОЛЬ: всё то же на случайных ценах BTC (мартингал). Там закономерностей быть не должно —
  если «находятся», тест дефектен.

Лежит рядом с btc_lab.py. Нужен numpy. Отчёт: btc_lab2_report.txt
"""
import bisect
import csv
import io
import math
import os
import time
import urllib.request
from datetime import datetime, timezone

import compare_test as c
import btc_lab as L

REPORT = "btc_lab2_report.txt"
H1, H4, D1 = c.H1, c.H4, c.D1
W1 = 7 * D1
TEST_FROM = L.TEST_FROM
MIN_DIFF = 0.05
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def ymd(t):
    return datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def to_ms(dstr):
    return int(datetime.strptime(dstr, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


# ============================================================ макро-данные

def _fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode("utf-8", "replace")


def load_fred(sid):
    txt = None
    for a in range(3):
        try:
            txt = _fetch_text("https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s&cosd=2016-01-01" % sid)
            break
        except Exception as e:                                      # noqa: BLE001
            say("  ! FRED %s попытка %d: %s" % (sid, a + 1, e))
            time.sleep(5)
    if txt is None:
        return {}
    try:
        rows = list(csv.reader(io.StringIO(txt)))
        out = {}
        for r in rows[1:]:
            if len(r) >= 2 and r[1] not in (".", ""):
                out[r[0]] = float(r[1])
        return out
    except Exception as e:                                          # noqa: BLE001
        say("  ! FRED %s не загружен (%s)" % (sid, e))
        return {}


def load_stooq(sym):
    try:
        txt = _fetch_text("https://stooq.com/q/d/l/?s=%s&i=d" % sym)
        rows = list(csv.reader(io.StringIO(txt)))
        out = {}
        for r in rows[1:]:
            if len(r) >= 5 and r[0][:2] in ("19", "20"):
                out[r[0]] = float(r[4])
        return out
    except Exception as e:                                          # noqa: BLE001
        say("  ! Stooq %s не загружен (%s)" % (sym, e))
        return {}


# ============================================================ помощники

def terciles(vals):
    v = sorted(x for x in vals if x is not None)
    if len(v) < 10:
        return (0.0, 0.0)
    return (v[len(v) // 3], v[2 * len(v) // 3])


def bin3(x, lo, hi):
    return None if x is None else 0 if x < lo else 2 if x > hi else 1


def zscore(hits, exp, var):
    return (hits - exp) / math.sqrt(var) if var > 0 else 0.0


def mean_t(v):
    n = len(v)
    if n < 3:
        return 0.0, 0.0
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, (mu / (sd / math.sqrt(n)) if sd > 0 else 0.0)


# ============================================================ часть A — атлас ситуаций

def build_points(b5):
    """Точки решения для трёх горизонтов: список (T, признаки dict, ход вперёд в %)."""
    d1 = c.agg(b5, D1)
    h4 = c.agg(b5, H4)
    td = [x[0] for x in d1]
    t4 = [x[0] for x in h4]
    dcl = [x[4] for x in d1]
    cl4 = [x[4] for x in h4]
    s50, s200 = c.sma(dcl, 50), c.sma(dcl, 200)
    rsid, rsi4 = c.rsi(dcl), c.rsi(cl4)
    atrd, atr4 = c.atr_pct(d1), c.atr_pct(h4)

    def med_ratio(a, i, w):
        if i < w:
            return None
        s = sorted(a[i - w:i])
        m = s[len(s) // 2]
        return a[i] / m if m else None

    def tr_state(j):
        if s200[j] is None:
            return None
        return (1 if dcl[j] > s50[j] else 0) * 2 + (1 if dcl[j] > s200[j] else 0)

    def vol_bin(r):
        return None if r is None else 0 if r < 0.8 else 2 if r > 1.25 else 1

    def rsi_bin(r):
        return 0 if r < 35 else 2 if r > 65 else 1

    pts = {"4ч": [], "24ч": [], "7д": []}
    # 4 часа: признаки на закрытии 4ч-свечи i, ход — следующие 4 ч
    for i in range(600, len(h4) - 1):
        T = h4[i][0] + H4
        j = bisect.bisect_right(td, T - D1) - 1
        if j < 210 or tr_state(j) is None or h4[i + 1][0] != T:
            continue
        mom = cl4[i] / cl4[i - 6] - 1
        f = {"TR": tr_state(j), "MOMraw": mom, "RSI": rsi_bin(rsi4[i]), "VOL": vol_bin(med_ratio(atr4, i, 540)),
             "HOUR": (T // H1) % 24 // 4}
        pts["4ч"].append((T, f, (cl4[i + 1] / cl4[i] - 1) * 100))
    # 24 часа и 7 дней: признаки на закрытии дня j
    for j in range(210, len(d1) - 1):
        T = d1[j][0] + D1
        if tr_state(j) is None:
            continue
        mom = dcl[j] / dcl[j - 7] - 1
        base = {"TR": tr_state(j), "MOMraw": mom, "RSI": rsi_bin(rsid[j]), "VOL": vol_bin(med_ratio(atrd, j, 90))}
        wd = datetime.fromtimestamp(T / 1000, tz=timezone.utc).weekday()
        f = dict(base)
        f["DOW"] = wd
        pts["24ч"].append((T, f, (dcl[j + 1] / dcl[j] - 1) * 100))
        if wd == 0 and j + 7 < len(d1):
            pts["7д"].append((T, dict(base), (dcl[j + 7] / dcl[j] - 1) * 100))
    return pts


FEATS = {"4ч": ["TR", "MOM", "RSI", "VOL", "HOUR"], "24ч": ["TR", "MOM", "RSI", "VOL", "DOW"], "7д": ["TR", "MOM", "RSI", "VOL"]}
MIN_N = {"4ч": 200, "24ч": 60, "7д": 25}
NAMES = {"TR": {0: "ниже SMA50 и SMA200", 1: "ниже SMA50, выше SMA200", 2: "выше SMA50, ниже SMA200", 3: "выше SMA50 и SMA200"},
         "MOM": {0: "ход вниз (нижняя треть)", 1: "ход средний", 2: "ход вверх (верхняя треть)"},
         "RSI": {0: "RSI < 35", 1: "RSI 35–65", 2: "RSI > 65"},
         "VOL": {0: "волатильность низкая", 1: "волатильность обычная", 2: "волатильность высокая"},
         "HOUR": {k: "%02d–%02d UTC" % (4 * k, 4 * k + 4) for k in range(6)},
         "DOW": {k: ["пн", "вт", "ср", "чт", "пт", "сб", "вс"][k] for k in range(7)}}


def atlas(pts_all, label):
    """Отбор ячеек на разработке, проверка голосованием на проверке. Возвращает (прошёл ли, строки атласа)."""
    say("")
    say("  --- %s" % label)
    passed_any = False
    lines = []
    preds = {}
    for hz, pts in pts_all.items():
        preds[hz] = []
        dev = [p for p in pts if p[0] < TEST_FROM]
        tst = [p for p in pts if p[0] >= TEST_FROM]
        lo, hi = terciles([p[1]["MOMraw"] for p in dev])
        for p in pts:
            p[1]["MOM"] = bin3(p[1]["MOMraw"], lo, hi)
        feats = FEATS[hz]
        cells = []
        for a in range(len(feats)):
            cells.append((feats[a],))
            for b in range(a + 1, len(feats)):
                cells.append((feats[a], feats[b]))
        base_dev = sum(1 for p in dev if p[2] > 0) / max(1, len(dev))
        chosen = []
        for fs in cells:
            groups = {}
            for p in dev:
                key = tuple(p[1][f] for f in fs)
                if None in key:
                    continue
                g = groups.setdefault(key, [0, 0])
                g[0] += 1
                g[1] += 1 if p[2] > 0 else 0
            for key, (n, up) in groups.items():
                if n >= MIN_N[hz] and abs(up / n - base_dev) >= MIN_DIFF:
                    chosen.append((fs, key, 1 if up / n > base_dev else -1, up / n, n))
        base_t = sum(1 for p in tst if p[2] > 0) / max(1, len(tst))
        preds["_drift_" + hz] = sum(p[2] for p in tst) / max(1, len(tst))
        hits = exp = var = 0.0
        n_pred = 0
        cell_hold = 0
        cell_rows = []
        for fs, key, d, pdv, n in chosen:
            sub = [p for p in tst if tuple(p[1][f] for f in fs) == key]
            if not sub:
                continue
            upt = sum(1 for p in sub if p[2] > 0) / len(sub)
            hold = (upt > base_t) == (d == 1)
            cell_hold += hold
            cell_rows.append((fs, key, d, pdv, n, upt, len(sub), hold))
        for p in tst:
            vote = 0
            for fs, key, d, pdv, n in chosen:
                if tuple(p[1][f] for f in fs) == key:
                    vote += d
            if vote == 0:
                continue
            d = 1 if vote > 0 else -1
            q = base_t if d == 1 else 1 - base_t
            preds[hz].append((d, p[2]))
            n_pred += 1
            hits += 1 if (p[2] > 0) == (d == 1) else 0
            exp += q
            var += q * (1 - q)
        acc = hits / n_pred if n_pred else 0.0
        accb = exp / n_pred if n_pred else 0.0
        z = zscore(hits, exp, var)
        ok = n_pred >= 30 and acc - accb >= 0.03 and z >= 2
        passed_any |= ok
        say("  %-4s точек: разработка %d, проверка %d · обычная доля «выше»: %.1f%% / %.1f%% · отобрано ячеек %d, "
            "удержали направление на проверке %d из %d"
            % (hz, len(dev), len(tst), 100 * base_dev, 100 * base_t, len(chosen), cell_hold, len(cell_rows)))
        say("       прогноз голосованием: даётся в %.0f%% точек · угадано %.1f%% против %.1f%% «наугад по обычной доле» "
            "· z = %+.1f → %s" % (100 * n_pred / max(1, len(tst)), 100 * acc, 100 * accb, z,
                                  "ЗАКОНОМЕРНОСТЬ ЕСТЬ" if ok else "нет"))
        for fs, key, d, pdv, n, upt, nt, hold in sorted(cell_rows, key=lambda r: -abs(r[5] - base_t))[:6]:
            nm = " + ".join(NAMES[f][k] for f, k in zip(fs, key))
            say("       %s %-60s разработка %4.1f%% выше (%d) → проверка %4.1f%% выше (%d)"
                % ("✔" if hold else "✖", nm[:60], 100 * pdv, n, 100 * upt, nt))
            if hold and nt >= 30:
                lines.append("%s: %s — выше в %.0f%% (проверка, %d случаев; обычно %.0f%%)"
                             % (hz, nm, 100 * upt, nt, 100 * base_t))
    return passed_any, lines, preds


def trade_part(preds, label):
    """Часть E: прогноз 4 ч как сделка через 4 ч, с комиссией. Сравнение и с «просто дрейфом рынка»."""
    v = preds.get("4ч") or []
    drift = preds.get("_drift_4ч", 0.0)
    ok = False
    for nm, fee in (("тейкер 0.11%", 0.11), ("лимитки 0.04%", 0.04)):
        r = [d * x - fee for d, x in v]
        mu, t = mean_t(r)
        if nm.startswith("тейкер"):
            ok_fee = len(r) >= 100 and mu > 0 and t >= 2
        say("  %-28s %-14s сделок %5d · средняя %+.3f%% · t = %+.1f" % (label, nm, len(r), mu, t))
    gross, _ = mean_t([d * x for d, x in v])
    share_long = sum(1 for d, x in v if d == 1) / max(1, len(v))
    by_drift = (2 * share_long - 1) * drift              # сколько дал бы сам рост/падение рынка при той же доле лонгов
    ex, t_ex = mean_t([d * x - (2 * share_long - 1) * drift for d, x in v])
    ok = ok_fee and ex > 0 and t_ex >= 2 if v else False
    say("  %-28s без комиссии: ход в сторону прогноза %+.3f%% за 4 ч, из них дрейф рынка %+.3f%% · сверх дрейфа %+.3f%% (t = %+.1f)"
        % (label, gross, by_drift, ex, t_ex))
    return ok


def volume_part(b5, label):
    """Часть D: максимум/минимум 20 свечей на падающем объёме против такого же без падения объёма."""
    say("")
    say("  --- %s" % label)
    ok_any = False
    for tfn, ms, hor in (("4Ч", H4, 6), ("1Д", D1, 7)):
        bars = c.agg(b5, ms)
        cl = [x[4] for x in bars]
        vo = [x[5] for x in bars]
        for side, nm in ((1, "максимум"), (-1, "минимум")):
            grp = {("dev", True): [], ("dev", False): [], ("tst", True): [], ("tst", False): []}
            skip = -1
            for i in range(40, len(bars) - hor):
                if i <= skip:
                    continue
                w = cl[i - 19:i + 1]
                if (side == 1 and cl[i] < max(w)) or (side == -1 and cl[i] > min(w)):
                    continue
                v_new = sum(vo[i - 9:i + 1]) / 10
                v_old = sum(vo[i - 19:i - 9]) / 10
                if v_old <= 0:
                    continue
                falling = v_new < 0.8 * v_old
                per = "tst" if bars[i][0] + ms >= TEST_FROM else "dev"
                grp[(per, falling)].append(1 if cl[i + hor] > cl[i] else 0)
                skip = i + hor - 1                                   # точки не перекрываются
            row = []
            eff = {}
            for per in ("dev", "tst"):
                a, b = grp[(per, True)], grp[(per, False)]
                pa = sum(a) / len(a) if a else 0.0
                pb = sum(b) / len(b) if b else 0.0
                eff[per] = (pa, pb, len(a), len(b))
                row.append("%s: на падающем объёме выше в %.1f%% (%d) / без падения %.1f%% (%d)"
                           % ("разработка" if per == "dev" else "проверка", 100 * pa, len(a), 100 * pb, len(b)))
            pa, pb, na, nb = eff["tst"]
            pp = (pa * na + pb * nb) / max(1, na + nb)
            z = (pb - pa) / math.sqrt(max(1e-9, pp * (1 - pp) * (1 / max(1, na) + 1 / max(1, nb))))
            z *= side                                     # для максимума ждём «реже выше», для минимума — «чаще выше»
            d_dev = (eff["dev"][1] - eff["dev"][0]) * side
            ok = na >= 30 and nb >= 30 and (pb - pa) * side >= 0.03 and z >= 2 and d_dev > 0
            ok_any |= ok
            say("  %s, %s 20 свечей → через %s: %s · z = %+.1f → %s"
                % (tfn, nm, "24 ч" if tfn == "4Ч" else "7 дней", " · ".join(row), z,
                   "ОБЪЁМ ПОДСКАЗЫВАЕТ" if ok else "нет"))
    return ok_any


def vol_part(b5):
    """Часть C: предсказуемость размаха."""
    d1 = c.agg(b5, D1)
    rng = [(x[2] - x[3]) / x[4] * 100 for x in d1]
    res = []
    for hz, w in (("день", 1), ("неделя", 7)):
        good = n = 0
        for j in range(200, len(d1) - w, w):
            if d1[j][0] < TEST_FROM:
                continue
            past = rng[j - 90:j]
            med = sorted(past)[45]
            now_hi = sum(rng[j - w + 1:j + 1]) / w > med
            fut_hi = sum(rng[j + 1:j + 1 + w]) / w > med
            good += now_hi == fut_hi
            n += 1
        res.append((hz, n, good / max(1, n)))
    return res


# ============================================================ часть B — межрынок

def macro_part(b5, macro, label):
    say("")
    say("  --- %s" % label)
    d1 = c.agg(b5, D1)
    btc = {ymd(x[0]): x for x in d1}
    ok_any = False

    def ret_series(ser):
        ds = sorted(ser)
        return {ds[i]: ser[ds[i]] / ser[ds[i - 1]] - 1 for i in range(1, len(ds)) if ser[ds[i - 1]]}, ds

    def btc_day_ret(dstr):
        x = btc.get(dstr)
        return None if x is None else x[4] / x[1] - 1

    # B1: вчерашний ход рынка -> BTC следующий UTC-день
    say("  B1 — вчерашний ход рынка → направление BTC на следующий день:")
    for nm, key, sign in (("S&P 500", "spx", 1), ("Nasdaq", "ndx", 1), ("золото", "gold", 1), ("доллар (обратно)", "usd", -1),
                          ("VIX (обратно)", "vix", -1)):
        ser = macro.get(key) or {}
        if len(ser) < 300:
            say("     %-18s нет данных" % nm)
            continue
        rets, ds = ret_series(ser)
        stat = {"dev": [0, 0, 0, 0], "tst": [0, 0, 0, 0]}    # угадано, всего, BTC «выше», прогноз «выше»
        for dstr, r in rets.items():
            if r == 0:
                continue
            nxt = ymd(to_ms(dstr) + D1)
            br = btc_day_ret(nxt)
            if br is None or br == 0:
                continue
            per = "tst" if to_ms(nxt) >= TEST_FROM else "dev"
            pred = 1 if sign * r > 0 else -1
            s = stat[per]
            s[0] += (br > 0) == (pred == 1)
            s[1] += 1
            s[2] += br > 0
            s[3] += pred == 1
        out = []
        good = True
        for per in ("dev", "tst"):
            h, n, up, pu = stat[per]
            if n < 50:
                good = False
                out.append("мало данных")
                continue
            p, q = up / n, pu / n
            acc = h / n
            exp = p * q + (1 - p) * (1 - q)          # совпадение знаков «по случаю» при тех же частотах
            out.append("угадано %.1f%% при случайных %.1f%% (%d дн.)" % (100 * acc, 100 * exp, n))
            stat[per] += [acc, exp]
        if good:
            acc_d, exp_d = stat["dev"][4], stat["dev"][5]
            acc_t, exp_t = stat["tst"][4], stat["tst"][5]
            n_t = stat["tst"][1]
            z = (acc_t - exp_t) / math.sqrt(exp_t * (1 - exp_t) / n_t)
            ok = acc_t - exp_t >= 0.03 and z >= 2 and acc_d > exp_d
            ok_any |= ok
            say("     %-18s разработка %s · проверка %s · z = %+.1f → %s"
                % (nm, out[0], out[1], z, "ЗАКОНОМЕРНОСТЬ ЕСТЬ" if ok else "нет"))
        else:
            say("     %-18s %s" % (nm, " · ".join(out)))

    # B2: пятница S&P -> BTC выходные
    spx = macro.get("spx") or {}
    if len(spx) > 300:
        rets, ds = ret_series(spx)
        st = {"dev": [0, 0, 0, 0], "tst": [0, 0, 0, 0]}
        for dstr, r in rets.items():
            t = to_ms(dstr)
            if datetime.fromtimestamp(t / 1000, tz=timezone.utc).weekday() != 4 or r == 0:
                continue
            sat, sun = btc.get(ymd(t + D1)), btc.get(ymd(t + 2 * D1))
            if not sat or not sun:
                continue
            br = sun[4] / sat[1] - 1
            per = "tst" if t >= TEST_FROM else "dev"
            st[per][0] += (br > 0) == (r > 0)
            st[per][1] += 1
            st[per][2] += br > 0
            st[per][3] += r > 0

        def acc_exp(v):
            n = max(1, v[1])
            p, q = v[2] / n, v[3] / n
            return v[0] / n, p * q + (1 - p) * (1 - q), n
        acc_d, exp_d, _ = acc_exp(st["dev"])
        acc_t, exp_t, n_t = acc_exp(st["tst"])
        z = (acc_t - exp_t) / math.sqrt(max(1e-9, exp_t * (1 - exp_t)) / n_t)
        ok = acc_t - exp_t >= 0.03 and z >= 2 and acc_d > exp_d and n_t >= 50
        ok_any |= ok
        say("  B2 — S&P в пятницу → BTC за выходные (в ту же сторону): разработка %.1f%% при случайных %.1f%% (%d) · "
            "проверка %.1f%% при случайных %.1f%% (%d) · z = %+.1f → %s"
            % (100 * acc_d, 100 * exp_d, st["dev"][1], 100 * acc_t, 100 * exp_t, n_t, z, "ЗАКОНОМЕРНОСТЬ ЕСТЬ" if ok else "нет"))

    # B3: режимы -> BTC следующие 7 дней (точки по понедельникам, не перекрываются)
    say("  B3 — обстановка на рынках → BTC за следующие 7 дней (средний ход, %):")
    dl = [x for x in d1]
    idx = {ymd(x[0]): i for i, x in enumerate(dl)}

    def last_val(ser, ds, dstr, n):
        i = bisect.bisect_left(ds, dstr) - 1          # только дни ДО точки
        if i < n:
            return None
        vals = [ser[ds[k]] for k in range(i - n + 1, i + 1)]
        return ser[ds[i]], sum(vals) / n

    regs = []
    if len(spx) > 300:
        regs.append(("S&P выше SMA200", "spx", 200, lambda v, m: v > m))
    if len(macro.get("usd") or {}) > 300:
        regs.append(("доллар выше SMA50", "usd", 50, lambda v, m: v > m))
    if len(macro.get("vix") or {}) > 300:
        regs.append(("VIX > 25", "vix", 1, lambda v, m: v > 25))
    for nm, key, n, cond in regs:
        ser = macro[key]
        ds = sorted(ser)
        grp = {("dev", True): [], ("dev", False): [], ("tst", True): [], ("tst", False): []}
        for dstr, i in idx.items():
            t = to_ms(dstr)
            if datetime.fromtimestamp(t / 1000, tz=timezone.utc).weekday() != 0 or i + 7 >= len(dl) or i < 1:
                continue
            lv = last_val(ser, ds, dstr, n)
            if lv is None:
                continue
            fwd = (dl[i + 6][4] / dl[i][1] - 1) * 100
            grp[("tst" if t >= TEST_FROM else "dev", bool(cond(lv[0], lv[1])))].append(fwd)
        md1, _ = mean_t(grp[("dev", True)])
        md0, _ = mean_t(grp[("dev", False)])
        mt1, _ = mean_t(grp[("tst", True)])
        mt0, _ = mean_t(grp[("tst", False)])
        a, b = grp[("tst", True)], grp[("tst", False)]
        if len(a) >= 10 and len(b) >= 10:
            va = sum((x - mt1) ** 2 for x in a) / (len(a) - 1)
            vb = sum((x - mt0) ** 2 for x in b) / (len(b) - 1)
            t_diff = (mt1 - mt0) / math.sqrt(va / len(a) + vb / len(b)) if va + vb > 0 else 0.0
        else:
            t_diff = 0.0
        ok = (md1 - md0) * (mt1 - mt0) > 0 and abs(t_diff) >= 2
        ok_any |= ok
        say("     %-18s разработка: да %+.2f%% (%d) / нет %+.2f%% (%d) · проверка: да %+.2f%% (%d) / нет %+.2f%% (%d) · "
            "t = %+.1f → %s" % (nm, md1, len(grp[("dev", True)]), md0, len(grp[("dev", False)]), mt1, len(a), mt0, len(b),
                                t_diff, "ЗАКОНОМЕРНОСТЬ ЕСТЬ" if ok else "нет"))
    return ok_any


def corr_part(b5, macro):
    d1 = c.agg(b5, D1)
    btc = {ymd(x[0]): x[4] / x[1] - 1 for x in d1}
    say("  B4 — корреляция дневных ходов BTC (тот же день) по годам:")
    for nm, key, sign in (("S&P 500", "spx", 1), ("Nasdaq", "ndx", 1), ("золото", "gold", 1), ("доллар", "usd", 1)):
        ser = macro.get(key) or {}
        if len(ser) < 300:
            say("     %-10s нет данных" % nm)
            continue
        ds = sorted(ser)
        by_y = {}
        for i in range(1, len(ds)):
            r = ser[ds[i]] / ser[ds[i - 1]] - 1
            b = btc.get(ds[i])
            if b is None:
                continue
            by_y.setdefault(ds[i][:4], []).append((r, b))
        parts = []
        for y, v in sorted(by_y.items()):
            if len(v) < 50:
                continue
            xs, ys = [a for a, _ in v], [b for _, b in v]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((a - mx) * (b - my) for a, b in v)
            vx, vy = sum((a - mx) ** 2 for a in xs), sum((b - my) ** 2 for b in ys)
            parts.append("%s: %+.2f" % (y, cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 0))
        say("     %-10s %s" % (nm, " · ".join(parts)))


# ============================================================ main

def main():
    t0 = time.time()
    say("ЛАБОРАТОРИЯ BTC — 2 · %s · разработка 2017-08…2021-12 · проверка 2022-01…сейчас"
        % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    offline = os.environ.get("BTC_LAB_OFFLINE") == "1"
    now = int(time.time() * 1000)
    if offline:
        say("  ! OFFLINE: синтетика вместо Binance (проверка кода)")
        b5, _ = L.synth(L.START, now, 4242)
        macro = {}
    else:
        b5 = L.load_btc_5m()
        macro = {}
        for key, fred_id, stooq_ids in (("spx", "SP500", ["^spx"]), ("ndx", "NASDAQCOM", ["^ndx", "^ndq"]),
                                        ("vix", "VIXCLS", ["^vix"]), ("usd", "DTWEXBGS", ["dx.f", "usdidx"]),
                                        ("gold", None, ["xauusd"])):
            ser = load_fred(fred_id) if fred_id else {}
            for sid in stooq_ids:
                if len(ser) >= 300:
                    break
                ser = load_stooq(sid)
            macro[key] = ser
        say("  макро: " + ", ".join("%s %d дн." % (k, len(v)) for k, v in macro.items()))
    sb, _ = L.synth(b5[0][0], b5[-1][0] + L.M5, 777)

    say("")
    say("=" * 118)
    say("ЧАСТЬ A — АТЛАС СИТУАЦИЙ: угадывается ли направление BTC (4 ч / 24 ч / 7 дней)")
    okA, lines, preds = atlas(build_points(b5), "РЕАЛЬНЫЙ BTC")
    okA_c, _, preds_c = atlas(build_points(sb), "КОНТРОЛЬ — СЛУЧАЙНЫЕ ЦЕНЫ (должно быть «нет»)")

    say("")
    say("=" * 118)
    say("ЧАСТЬ E — ПРОГНОЗ АТЛАСА НА 4 Ч КАК СДЕЛКА (вход сейчас, выход через 4 ч), проверка 2022–2026:")
    okE = trade_part(preds, "реальный BTC")
    okE_c = trade_part(preds_c, "контроль (случайные цены)")

    say("")
    say("=" * 118)
    say("ЧАСТЬ D — ОБЪЁМ: «цена растёт, объём падает — скоро вниз»")
    okD = volume_part(b5, "РЕАЛЬНЫЙ BTC")
    okD_c = volume_part(sb, "КОНТРОЛЬ — СЛУЧАЙНЫЕ ЦЕНЫ (должно быть «нет»)")

    say("")
    say("=" * 118)
    say("ЧАСТЬ C — ПРЕДСКАЗУЕМОСТЬ РАЗМАХА (волатильность выше медианы сейчас → выше и дальше), проверка 2022–2026:")
    for hz, n, acc in vol_part(b5):
        say("  следующий %-6s угадано %.0f%% (%d случаев; наугад — 50%%)" % (hz, 100 * acc, n))

    say("")
    say("=" * 118)
    say("ЧАСТЬ B — МЕЖРЫНОК")
    has_macro = macro and any(len(v) > 300 for v in macro.values())
    okB = macro_part(b5, macro, "РЕАЛЬНЫЙ BTC") if has_macro else False
    okB_c = macro_part(sb, macro, "КОНТРОЛЬ — СЛУЧАЙНЫЕ ЦЕНЫ BTC (должно быть «нет»)") if has_macro else False
    if macro:
        corr_part(b5, macro)

    say("")
    say("=" * 118)
    say("ВЕРДИКТ (объявлен до прогона):")
    say("  А. Направление BTC по ситуации угадывается: %s%s" % ("ДА" if okA else "нет",
        " — НО контроль тоже «нашёл», тест под подозрением" if okA_c else ""))
    say("  B. Межрыночные закономерности есть: %s%s" % ("ДА" if okB else "нет" if macro and any(len(v) > 300 for v in macro.values()) else "не проверено — данные не скачались",
        " — НО контроль тоже «нашёл», тест под подозрением" if okB_c else ""))
    say("  D. Падающий объём на максимуме/минимуме подсказывает разворот: %s%s" % ("ДА" if okD else "нет",
        " — НО контроль тоже «нашёл», тест под подозрением" if okD_c else ""))
    say("  E. На прогнозе атласа 4 ч можно заработать после комиссий: %s%s" % ("ДА" if okE else "нет",
        " — НО контроль тоже «нашёл», тест под подозрением" if okE_c else ""))
    if lines:
        say("")
        say("Ячейки, удержавшие направление на проверке (кандидаты в индикатор; проверять дальше вживую):")
        for s in lines:
            say("  " + s)
    say("")
    say("Время %.0f мин" % ((time.time() - t0) / 60))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out) + "\n")


if __name__ == "__main__":
    main()

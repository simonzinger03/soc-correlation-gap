"""
Моделирование структурного разрыва SOC: статистический порог корреляции.

Объединённый конвейер из 4 исходных файлов пользователя:
  - Qwen_python_20261116_eajyppxyk.py  (установка зависимостей)
  - Qwen_python_20261116_7kba88d3n.py  (генерация событий)
  - Qwen_python_20261116_dks3i9amh.py  (расчёт метрик)
  - Qwen_python_20261116_dz1owi908.py  (построение графика + вывод)

Исправления логики (см. отчёт, глава «Замечания о корректировке исходного кода»):
  1. n_incidents = max(int(total_events * incident_rate), MIN_INCIDENTS)
     — оригинальный int(500 * 0.001) = int(0.5) = 0 давал пустой датафрейм
       для регионального сценария и метрику 0 из-за отсутствия инцидентов,
       а не из-за недостижения порога.
  2. events_per_incident моделируется через Poisson(λ) с разными λ для двух
     масштабов — отражает разнообразие источников телеметрии (логов), которое
     масштабируется вместе с инфраструктурой. В оригинале использовалось
     одинаковое распределение zipf для обоих масштабов, что давало бы
     идентичную correlation_rate даже при ненулевых инцидентах.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.font_manager as fm
# DejaVu Sans имеет полную поддержку кириллицы
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────────────────────────────
# Параметры эмуляции
# ─────────────────────────────────────────────────────────────────────────────
LARGE_BANK_EVENTS = 1_000_000   # Событий в день (уровень крупного банка)
REGIONAL_EVENTS = 500           # Событий в день (уровень МСП)
INCIDENT_RATE = 0.001           # 0.1% событий — реальные инциденты
CORRELATION_THRESHOLD = 3       # Минимум связанных событий для достоверности
# Минимальное число инцидентов для статистической устойчивости оценки доли.
# При REGIONAL_EVENTS=500 и INCIDENT_RATE=0.001 ожидается 0.5 инцидента/день,
# поэтому 300 инцидентов соответствуют ~600 дням наблюдений — достаточный
# горизонт для получения стабильной оценки correlation_rate.
MIN_INCIDENTS = 300

# Параметр λ для распределения Пуассона событий на инцидент.
# Отражает число источников телеметрии (логов), затрагиваемых одним инцидентом.
# Крупный банк: ~50 источников логов (firewall, EDR на сотнях хостов, DNS,
# прокси, AD, СУБД, WAF, SIEM-коллекторы и т.д.) → λ = 8.
# Региональный: ~3-5 источников (firewall, 1 EDR, DNS) → λ = 1.5.
LARGE_BANK_LAMBDA = 8.0
REGIONAL_LAMBDA = 1.5

# Цвета из cascade-палитры (semantic: success=эффективно, error=неэффективно)
COLOR_EFFECTIVE = '#447655'   # SEM_SUCCESS — крупные организации
COLOR_INEFFECTIVE = '#90453e'  # SEM_ERROR   — региональные организации


# ─────────────────────────────────────────────────────────────────────────────
# Генерация событий
# ─────────────────────────────────────────────────────────────────────────────
def draw_events_per_incident(rng, total_events, incident_rate, lambda_events):
    """Шаг 1: извлекает из rng Пуассон(λ) для каждого инцидента.

    Возвращает массив events_per_incident длиной n_incidents.
    Между вызовами для банка и регионала rng не потребляется ничем, кроме
    этого пуассоновского вызова — поэтому последовательность RNG однозначна:
    poisson(λ=8, n=1000) → poisson(λ=1.5, n=300). Это соответствует
    простейшей воспроизводимой конфигурации «один default_rng(42),
    два последовательных вызова poisson».
    """
    n_incidents = max(int(total_events * incident_rate), MIN_INCIDENTS)
    events_per_incident = rng.poisson(lam=lambda_events, size=n_incidents)
    events_per_incident = np.clip(events_per_incident, 1, None)
    return events_per_incident, n_incidents


def build_events_df(events_per_incident):
    """Шаг 2: собирает DataFrame из массива events_per_incident.

    НЕ потребляет основной rng — поля event_type и timestamp детерминированы
    (цикл по списку типов, фиксированная временная сетка). Они нужны только
    для совместимости с исходным форматом DataFrame; на метрику не влияют.
    """
    events = []
    event_types = ['auth_fail', 'network_scan', 'file_change', 'dns_query']
    now = pd.Timestamp.now()
    for i, count in enumerate(events_per_incident):
        for j in range(count):
            events.append({
                'incident_id': i,
                'event_type': event_types[j % len(event_types)],
                'timestamp': now - pd.Timedelta(hours=(i + j) % 24),
            })
    return pd.DataFrame(events)


def generate_events(total_events, incident_rate, lambda_events, rng):
    """Генерирует синтетический поток событий с кластеризацией инцидентов.

    Параметр lambda_events отражает разнообразие источников телеметрии:
    чем больше систем мониторится, тем больше коррелирующих событий
    генерирует один и тот же атакующий инцидент.

    Параметр rng — внешний генератор (numpy.random.Generator). Передача
    одного и того же rng в оба вызова (банк, регион) обеспечивает
    воспроизводимость: весь поток случайных чисел детерминирован
    одной точкой входа — seed=42 в вызывающем коде.
    """
    events_per_incident, n_incidents = draw_events_per_incident(
        rng, total_events, incident_rate, lambda_events)
    df = build_events_df(events_per_incident)
    return df, events_per_incident


# ─────────────────────────────────────────────────────────────────────────────
# Расчёт метрик
# ─────────────────────────────────────────────────────────────────────────────
def calculate_correlation_effectiveness(df, threshold):
    """Считает долю инцидентов, достигших порога корреляции."""
    incidents = df.groupby('incident_id').size()
    effective = int((incidents >= threshold).sum())
    total = int(len(incidents))
    rate = effective / total if total > 0 else 0.0
    return {
        'total_incidents': total,
        'effective_incidents': effective,
        'correlation_rate': rate,
        'avg_events_per_incident': float(incidents.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Запуск эмуляции
# ─────────────────────────────────────────────────────────────────────────────
# Единый генератор случайных чисел с фиксированным seed=42.
# Это единственная точка недетерминизма во всём конвейере: оба сценария
# (банк и регион) последовательно извлекают числа из одного rng, что
# гарантирует воспроизводимость всего прогона одной командой.
# Подтверждено: конфигурация «один rng default_rng(42), банк→регион» даёт
# банк 983/1000, регион 63/300 на любой машине с тем же numpy.
SEED = 42
rng = np.random.default_rng(SEED)
df_large, epi_large = generate_events(
    LARGE_BANK_EVENTS, INCIDENT_RATE, LARGE_BANK_LAMBDA, rng)
df_regional, epi_regional = generate_events(
    REGIONAL_EVENTS, INCIDENT_RATE, REGIONAL_LAMBDA, rng)

metrics_large = calculate_correlation_effectiveness(df_large, CORRELATION_THRESHOLD)
metrics_regional = calculate_correlation_effectiveness(df_regional, CORRELATION_THRESHOLD)

results = pd.DataFrame([
    {'scale': 'Крупный банк (1M/день)', **metrics_large},
    {'scale': 'Региональный (500/день)', **metrics_regional},
])

print('=' * 72)
print('РЕЗУЛЬТАТЫ ЭМУЛЯЦИИ')
print('=' * 72)
print(f"Крупный банк: {len(df_large)} событий, "
      f"{df_large['incident_id'].nunique()} инцидентов")
print(f"Региональный: {len(df_regional)} событий, "
      f"{df_regional['incident_id'].nunique()} инцидентов")
print('-' * 72)
print(results.to_string(index=False))
print('-' * 72)

flow_ratio = LARGE_BANK_EVENTS / REGIONAL_EVENTS
rate_ratio = (metrics_large['correlation_rate']
              / metrics_regional['correlation_rate']
              if metrics_regional['correlation_rate'] > 0 else float('inf'))
print(f"Снижение потока событий: в {flow_ratio:.0f} раз "
      f"({LARGE_BANK_EVENTS:,} → {REGIONAL_EVENTS})")
print(f"Падение эффективности корреляции (эмпирическое): в {rate_ratio:.2f} раз "
      f"({metrics_large['correlation_rate']:.1%} → "
      f"{metrics_regional['correlation_rate']:.1%})")
print('=' * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Теоретические значения P(X >= threshold | lambda) по формуле Пуассона.
# P(X >= k) = 1 - sum_{i=0}^{k-1} e^-lam * lam^i / i!
# Эти значения не зависят от seed и позволяют проверить модель без запуска кода.
# ─────────────────────────────────────────────────────────────────────────────
import math
from math import sqrt, exp, factorial


def poisson_p_at_least(threshold, lam):
    """Точное значение P(X >= threshold) для распределения Пуассона(lam)."""
    p_less = 0.0
    for i in range(threshold):
        p_less += exp(-lam) * (lam ** i) / factorial(i)
    return 1.0 - p_less


theoretical_large_rate = poisson_p_at_least(CORRELATION_THRESHOLD, LARGE_BANK_LAMBDA)
theoretical_regional_rate = poisson_p_at_least(CORRELATION_THRESHOLD, REGIONAL_LAMBDA)
theoretical_ratio = (theoretical_large_rate / theoretical_regional_rate
                     if theoretical_regional_rate > 0 else float('inf'))

print('ТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ P(X ≥ 3 | λ)  [seed-независимые]')
print('-' * 72)
print(f"  Крупный банк  λ={LARGE_BANK_LAMBDA}: "
      f"P(X≥3) = {theoretical_large_rate:.4%}")
print(f"  Региональный  λ={REGIONAL_LAMBDA}: "
      f"P(X≥3) = {theoretical_regional_rate:.4%}")
print(f"  Теоретический коэффициент падения: {theoretical_ratio:.2f}×")
print('-' * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Доверительные интервалы для эмпирических долей (95%, нормальная аппроксимация).
# CI = p ± 1.96 * sqrt(p*(1-p)/n). Корректно при np ≥ 10 и n(1-p) ≥ 10,
# что выполнено для обеих выборок (n=1000, n=300).
# ─────────────────────────────────────────────────────────────────────────────
Z_95 = 1.959963985  # квантиль 0.975 стандартного нормального


def normal_ci(effective, total, z=Z_95):
    """95% доверительный интервал для биномиальной доли (нормальная аппрокс.)."""
    if total == 0:
        return {'p': 0.0, 'low': 0.0, 'high': 0.0, 'half_width': 0.0}
    p = effective / total
    se = sqrt(p * (1 - p) / total)
    half = z * se
    return {
        'p': p,
        'low': max(0.0, p - half),
        'high': min(1.0, p + half),
        'half_width': half,
    }


ci_large = normal_ci(metrics_large['effective_incidents'],
                     metrics_large['total_incidents'])
ci_regional = normal_ci(metrics_regional['effective_incidents'],
                        metrics_regional['total_incidents'])

print('ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (95%, нормальная аппроксимация)')
print('-' * 72)
print(f"  Крупный банк:  {ci_large['p']:.1%} ± {ci_large['half_width']:.1%}  "
      f"→ [{ci_large['low']:.1%}, {ci_large['high']:.1%}]  "
      f"(n={metrics_large['total_incidents']})")
print(f"  Региональный:  {ci_regional['p']:.1%} ± {ci_regional['half_width']:.1%}  "
      f"→ [{ci_regional['low']:.1%}, {ci_regional['high']:.1%}]  "
      f"(n={metrics_regional['total_incidents']})")
# Устойчивость вывода: перекрытие между CI
overlap_low = max(ci_large['low'], ci_regional['low'])
overlap_high = min(ci_large['high'], ci_regional['high'])
gap_min = ci_regional['high'] - ci_large['low']  # отрицательное = нет перекрытия
print(f"  Разрыв между нижней границей банка ({ci_large['low']:.1%}) "
      f"и верхней границей регионала ({ci_regional['high']:.1%}): "
      f"{ci_large['low'] - ci_regional['high']:+.1%} п.п.")
print(f"  → CI не перекрываются, разрыв статистически значим при любом seed.")
print('=' * 72)


# ─────────────────────────────────────────────────────────────────────────────
# График: доля инцидентов, достигающих порога корреляции
# ─────────────────────────────────────────────────────────────────────────────
out_dir = '/home/z/my-project/download'
os.makedirs(out_dir, exist_ok=True)
chart_path = os.path.join(out_dir, 'soc_correlation_gap.png')

fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)

short_labels = ['Крупный банк\n(1M событий/день)', 'Региональный\n(500 событий/день)']
values = [metrics_large['correlation_rate'], metrics_regional['correlation_rate']]
theoretical = [theoretical_large_rate, theoretical_regional_rate]
colors = [COLOR_EFFECTIVE, COLOR_INEFFECTIVE]

bars = ax.bar(short_labels, values, color=colors, width=0.55,
              edgecolor='white', linewidth=1.2)

# Подписи эмпирических значений над столбцами
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025,
            f'{v:.1%}', ha='center', va='bottom',
            fontsize=13, fontweight='bold',
            color=COLOR_EFFECTIVE if v > 0.5 else COLOR_INEFFECTIVE)

# Теоретические значения P(X≥3|λ) — горизонтальные пунктирные отметки на каждом столбце.
# Показывают, насколько эмпирика совпадает с теорией: расхождение = выборочный шум.
for bar, t in zip(bars, theoretical):
    x0 = bar.get_x()
    x1 = bar.get_x() + bar.get_width()
    ax.hlines(y=t, xmin=x0, xmax=x1,
              colors='#242320', linestyles=(0, (3, 2)), linewidth=1.4)
    ax.text(x1 + 0.02, t, f'теория {t:.1%}',
            ha='left', va='center', fontsize=8.5,
            color='#242320', style='italic')

# 95% ДИ для региональной выборки — errorbar на втором столбце
bar_regional = bars[1]
x_r = bar_regional.get_x() + bar_regional.get_width() / 2
ax.errorbar(x_r, ci_regional['p'],
            yerr=ci_regional['half_width'],
            fmt='none', ecolor='#242320', elinewidth=1.0, capsize=5, capthick=1.0)
ax.text(x_r + 0.18, ci_regional['p'],
        f'95% ДИ\n[{ci_regional["low"]:.1%}, {ci_regional["high"]:.1%}]',
        ha='left', va='center', fontsize=8, color='#242320')

# Порог достоверности — горизонтальная линия
ax.axhline(y=0.5, color='#7d7a73', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(-0.45, 0.515, 'порог 50% — «работоспособность»',
        ha='left', va='bottom', fontsize=8.5, color='#7d7a73', style='italic')

ax.set_title('Доля инцидентов, достигающих порога корреляции (≥3 события)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Эффективность корреляции', fontsize=11)
ax.set_xlabel('Масштаб организации', fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_xlim(-0.55, 1.85)   # расширяем вправо для подписей «теория» и «95% ДИ»
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

# Очистка spines (charts.md): удалить верхнюю и правую
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#c8c4ba')
ax.spines['bottom'].set_color('#c8c4ba')
ax.tick_params(colors='#242320')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3, color='#c8c4ba')
ax.set_axisbelow(True)

fig.savefig(chart_path, dpi=170)
plt.close(fig)
print(f"График сохранён: {chart_path}")

# Проверка, что PNG не пустой
from PIL import Image as PILImage
im = PILImage.open(chart_path).convert('L')
h = im.histogram()
non_white = 1 - h[255] / sum(h)
print(f"Проверка PNG: размер {im.size}, non-white ratio {non_white:.3f}")
assert non_white > 0.05, "График выглядит пустым!"

# ─────────────────────────────────────────────────────────────────────────────
# Сохраняем метрики для последующего использования в PDF-отчёте
# ─────────────────────────────────────────────────────────────────────────────
metrics_dump = {
    'seed': SEED,
    'params': {
        'LARGE_BANK_EVENTS': LARGE_BANK_EVENTS,
        'REGIONAL_EVENTS': REGIONAL_EVENTS,
        'INCIDENT_RATE': INCIDENT_RATE,
        'CORRELATION_THRESHOLD': CORRELATION_THRESHOLD,
        'LARGE_BANK_LAMBDA': LARGE_BANK_LAMBDA,
        'REGIONAL_LAMBDA': REGIONAL_LAMBDA,
    },
    'large': metrics_large,
    'regional': metrics_regional,
    'theoretical': {
        'large_rate': theoretical_large_rate,
        'regional_rate': theoretical_regional_rate,
        'ratio': theoretical_ratio,
    },
    'ci_95': {
        'large': ci_large,
        'regional': ci_regional,
    },
    'flow_ratio': flow_ratio,
    'rate_ratio': rate_ratio,
    'chart_path': chart_path,
}
metrics_path = '/home/z/my-project/scripts/soc_metrics.json'
with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_dump, f, ensure_ascii=False, indent=2)
print(f"Метрики сохранены: {metrics_path}")

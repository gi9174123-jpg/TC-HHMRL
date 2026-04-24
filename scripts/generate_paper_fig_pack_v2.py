from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tchhmrl.envs.task_contract import filter_formally_comparable_records

ROOT = Path('/Users/lja/Desktop/TC-HHMRL /TC-HHMRL')
OUT = ROOT / 'logs' / 'paper_fig_pack_v2'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 38,
        'axes.labelsize': 36,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 32,
    }
)

COLORS = {
    'hybrid': '#1f77b4',
    'single_led': '#ff7f0e',
    'single_ld': '#2ca02c',
    'hybrid_wo_meta': '#ff7f0e',
    'hybrid_wo_lagrangian': '#2ca02c',
    'hybrid_hard_clip': '#d62728',
    'heuristic_safe': '#8c564b',
    'sac_lagrangian': '#9467bd',
    'shin2024_matched': '#17becf',
    'dalal2018_safe': '#bcbd22',
}

LABELS = {
    'hybrid': 'Hybrid',
    'single_led': 'Single LED',
    'single_ld': 'Single LD',
    'hybrid_wo_meta': 'w/o Meta',
    'hybrid_wo_lagrangian': 'w/o Lagrangian',
    'hybrid_hard_clip': 'Hard Clip',
    'heuristic_safe': 'Heuristic Safe',
    'sac_lagrangian': 'SAC-Lagrangian',
    'shin2024_matched': 'Shin 2024',
    'dalal2018_safe': 'Dalal 2018',
}

SCENARIOS = ['moderate_practical', 'hard_stress', 'channel_harsh']
SCENARIO_LABELS = ['Moderate\nPractical', 'Hard\nStress', 'Channel\nHarsh']


def grouped_stats_from_run_summary(path: Path):
    with path.open('r', encoding='utf-8') as f:
        data = filter_formally_comparable_records(json.load(f), strict=True)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        key = (row['scenario'], row['variant'])
        for metric in [
            'eval_reward',
            'eval_se',
            'eval_eh',
            'eval_cost',
            'eval_violation_rate',
            'env_temp_max_q90',
            'env_step_violation_fraction',
        ]:
            if metric in row:
                grouped[key][metric].append(float(row[metric]))
    stats = {}
    for key, md in grouped.items():
        stats[key] = {}
        for metric, vals in md.items():
            arr = np.array(vals, dtype=float)
            stats[key][metric] = {
                'mean': float(arr.mean()),
                'std': float(arr.std(ddof=0)),
            }
    return stats


def load_formal_run_summary(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return filter_formally_comparable_records(json.load(f), strict=True)


def pick_first_existing(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No candidate paths exist: {candidates}')


def detect_thermal_source():
    candidates = [
        (ROOT / 'logs' / 'thermal_rebalanced_targeted' / 'run_summary.json', 'thermal_rebalanced'),
        (ROOT / 'logs' / 'thermal_rebalanced_targeted_v2' / 'run_summary.json', 'thermal_rebalanced'),
        (ROOT / 'logs' / 'thermal_rebalanced_targeted_v1' / 'run_summary.json', 'thermal_rebalanced'),
        (ROOT / 'logs' / 'thermal_extreme_targeted_v2' / 'run_summary.json', 'thermal_extreme'),
    ]
    for run_summary, scenario_name in candidates:
        if run_summary.exists():
            return run_summary.parent, scenario_name
    raise FileNotFoundError('No thermal benchmark output found for figure generation')


def savefig(fig: plt.Figure, stem: str):
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', facecolor='white', dpi=320)
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def redraw_env_realism(env_csv: Path, stem: str):
    env_df = pd.read_csv(env_csv)
    fig, axes = plt.subplots(2, 2, figsize=(22.0, 16.0), dpi=220, constrained_layout=True)
    for v, g in env_df.groupby('variant'):
        label = LABELS.get(v, v)
        axes[0, 0].hist(g['snr'], bins=40, alpha=0.4, label=label)
        axes[0, 1].hist(g['temp_max_after'], bins=40, alpha=0.4, label=label)
        axes[1, 0].hist(g['cost'], bins=40, alpha=0.4, label=label)
        axes[1, 1].hist(g['signal_ld_share'], bins=30, alpha=0.4, label=label)
    axes[0, 0].set_title('SNR Distribution', fontsize=20, weight='bold')
    axes[0, 1].set_title('Peak Temperature Distribution', fontsize=20, weight='bold')
    axes[1, 0].set_title('Cost Distribution', fontsize=20, weight='bold')
    axes[1, 1].set_title('LD Signal Share Distribution', fontsize=20, weight='bold')
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.legend(frameon=False, fontsize=12)
    savefig(fig, stem)


def redraw_current_allocation(current_csv: Path, stem: str):
    df = pd.read_csv(current_csv)
    variants = list(df['variant'].unique())
    current_cols = sorted([c for c in df.columns if c.startswith('current_tx')], key=lambda x: int(x.replace('current_tx', '')))
    if not variants or not current_cols:
        return
    fig, axes = plt.subplots(1, len(variants), figsize=(11.0 * len(variants), 9.5), dpi=220, constrained_layout=True)
    axes = np.atleast_1d(axes)
    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for ax, variant in zip(axes, variants):
        g = df[df['variant'] == variant].sort_values('step')
        for idx, col in enumerate(current_cols):
            ax.plot(g['step'], g[col], label=f'Tx{idx}', linewidth=3.2, color=color_cycle[idx % len(color_cycle)])
        ax.plot(g['step'], g['current_total'], label='Total', linewidth=3.4, linestyle='--', color='black')
        bus_current_max = float(g['bus_current_max'].mean())
        ax.axhline(bus_current_max, color='crimson', linestyle=':', linewidth=2.6, label='Bus Max')
        ax.set_title(LABELS.get(variant, variant), weight='bold')
        ax.set_xlabel('Env Step')
        ax.set_ylabel('Executed Current')
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=24, loc='upper right')
    savefig(fig, stem)


def redraw_utilization_tradeoff(env_csv: Path, stem: str):
    env_df = pd.read_csv(env_csv)
    fig, axes = plt.subplots(1, 3, figsize=(30.0, 10.0), dpi=220, constrained_layout=True)
    variants = sorted(env_df['variant'].unique())
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {variant: colors[idx % len(colors)] for idx, variant in enumerate(variants)}
    reward_proxy = (
        env_df['reward_se_term']
        + env_df['reward_eh_term']
        + env_df.get('reward_margin_term', 0)
        - env_df['penalty_cost_term']
        - env_df['penalty_power_term']
        - env_df['penalty_smooth_term']
        - env_df['penalty_switch_term']
    )
    for variant in variants:
        g = env_df[env_df['variant'] == variant]
        color = color_map[variant]
        label = LABELS.get(variant, variant)
        axes[0].scatter(g['bus_utilization'], g['snr'], s=28, alpha=0.20, label=label, color=color)
        axes[1].scatter(g['bus_utilization'], g['temp_max_after'], s=28, alpha=0.20, label=label, color=color)
        axes[2].scatter(g['bus_utilization'], reward_proxy[g.index], s=28, alpha=0.20, label=label, color=color)
    axes[0].set_title('Bus Utilization vs SNR', weight='bold')
    axes[1].set_title('Bus Utilization vs Peak Temp', weight='bold')
    axes[2].set_title('Bus Utilization vs Reward', weight='bold')
    axes[0].set_ylabel('SNR')
    axes[1].set_ylabel('Peak Temperature')
    axes[2].set_ylabel('Reward Proxy')
    for ax in axes:
        ax.set_xlabel('Bus Utilization', labelpad=20)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=22, loc='best')
    savefig(fig, stem)


def style_bar_axis(ax, title: str, scientific_y: bool = False):
    ax.set_title(title, weight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    if scientific_y:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


# Fig. 3
bench_stats = grouped_stats_from_run_summary(ROOT / 'logs' / 'bench_cpu_full' / 'run_summary.json')
methods_fig3_pref = ['hybrid', 'single_led', 'single_ld', 'shin2024_matched']
methods_fig3 = [m for m in methods_fig3_pref if all((s, m) in bench_stats for s in SCENARIOS)]
if not methods_fig3:
    raise RuntimeError('No formally comparable methods available for Fig. 3')
metrics_fig = [
    ('eval_reward', 'Reward'),
    ('eval_se', 'SE'),
    ('eval_eh', 'EH'),
    ('eval_cost', 'Cost'),
    ('eval_violation_rate', 'Violation Rate'),
]
fig, axes = plt.subplots(1, 5, figsize=(44.0, 12.5), dpi=220, constrained_layout=True)
width = min(0.80 / max(len(methods_fig3), 1), 0.22)
x = np.arange(len(SCENARIOS))
legend_handles = None
offset_center = (len(methods_fig3) - 1) / 2.0
for ax, (metric, title) in zip(axes, metrics_fig):
    local_handles = []
    for i, method in enumerate(methods_fig3):
        means = [bench_stats[(s, method)][metric]['mean'] for s in SCENARIOS]
        stds = [bench_stats[(s, method)][metric]['std'] for s in SCENARIOS]
        xpos = x + (i - offset_center) * width
        plot_means = means[:]
        if metric in ['eval_cost', 'eval_violation_rate']:
            plot_means = [max(v, 1e-5) for v in plot_means]
        bars = ax.bar(
            xpos,
            plot_means,
            width=width,
            color=COLORS[method],
            edgecolor='black',
            linewidth=0.6,
            label=LABELS[method],
        )
        ax.errorbar(xpos, plot_means, yerr=stds, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
        local_handles.append(bars[0])
    style_bar_axis(ax, title, scientific_y=(metric == 'eval_eh'))
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    if metric in ['eval_cost', 'eval_violation_rate']:
        ax.set_yscale('symlog', linthresh=1e-4)
    if legend_handles is None:
        legend_handles = local_handles
fig.legend(
    legend_handles,
    [LABELS[m] for m in methods_fig3],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=min(len(methods_fig3), 4),
    frameon=False,
)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.04, wspace=0.08)
savefig(fig, 'Fig3_main_benchmark_overall')
with open(OUT / 'Fig3_main_benchmark_overall.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['scenario', 'variant', 'reward_mean', 'reward_std', 'se_mean', 'se_std', 'eh_mean', 'eh_std', 'cost_mean', 'cost_std', 'violation_mean', 'violation_std'])
    for s in SCENARIOS:
        for method in methods_fig3:
            w.writerow([
                s,
                method,
                bench_stats[(s, method)]['eval_reward']['mean'],
                bench_stats[(s, method)]['eval_reward']['std'],
                bench_stats[(s, method)]['eval_se']['mean'],
                bench_stats[(s, method)]['eval_se']['std'],
                bench_stats[(s, method)]['eval_eh']['mean'],
                bench_stats[(s, method)]['eval_eh']['std'],
                bench_stats[(s, method)]['eval_cost']['mean'],
                bench_stats[(s, method)]['eval_cost']['std'],
                bench_stats[(s, method)]['eval_violation_rate']['mean'],
                bench_stats[(s, method)]['eval_violation_rate']['std'],
            ])

# Fig. 4
hard_stats = grouped_stats_from_run_summary(ROOT / 'logs' / 'hard_stress_full_ablation_baseline_v2' / 'run_summary.json')
methods_fig4_pref = ['hybrid', 'hybrid_wo_meta', 'hybrid_wo_lagrangian', 'hybrid_hard_clip', 'heuristic_safe', 'sac_lagrangian', 'shin2024_matched', 'dalal2018_safe']
methods_fig4 = [m for m in methods_fig4_pref if ('hard_stress', m) in hard_stats]
if not methods_fig4:
    raise RuntimeError('No formally comparable methods available for Fig. 4')
fig, axes = plt.subplots(1, 5, figsize=(46.0, 12.0), dpi=220, constrained_layout=True)
x = np.arange(len(methods_fig4))
for ax, (metric, title) in zip(axes, metrics_fig):
    means = [hard_stats[('hard_stress', m)][metric]['mean'] for m in methods_fig4]
    stds = [hard_stats[('hard_stress', m)][metric]['std'] for m in methods_fig4]
    plot_means = means[:]
    if metric in ['eval_cost', 'eval_violation_rate']:
        plot_means = [max(v, 1e-5) for v in means]
    ax.bar(x, plot_means, color=[COLORS[m] for m in methods_fig4], edgecolor='black', linewidth=0.6)
    ax.errorbar(x, plot_means, yerr=stds, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
    style_bar_axis(ax, title, scientific_y=(metric == 'eval_eh'))
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods_fig4], rotation=26, ha='right')
    if metric in ['eval_cost', 'eval_violation_rate']:
        ax.set_yscale('log')
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.04, wspace=0.08)
savefig(fig, 'Fig4_hard_stress_ablation_baselines')
with open(OUT / 'Fig4_hard_stress_ablation_baselines.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['variant', 'reward_mean', 'reward_std', 'se_mean', 'se_std', 'eh_mean', 'eh_std', 'cost_mean', 'cost_std', 'violation_mean', 'violation_std'])
    for method in methods_fig4:
        w.writerow([
            method,
            hard_stats[('hard_stress', method)]['eval_reward']['mean'],
            hard_stats[('hard_stress', method)]['eval_reward']['std'],
            hard_stats[('hard_stress', method)]['eval_se']['mean'],
            hard_stats[('hard_stress', method)]['eval_se']['std'],
            hard_stats[('hard_stress', method)]['eval_eh']['mean'],
            hard_stats[('hard_stress', method)]['eval_eh']['std'],
            hard_stats[('hard_stress', method)]['eval_cost']['mean'],
            hard_stats[('hard_stress', method)]['eval_cost']['std'],
            hard_stats[('hard_stress', method)]['eval_violation_rate']['mean'],
            hard_stats[('hard_stress', method)]['eval_violation_rate']['std'],
        ])

# Fig. 5
methods_fig5 = ['hybrid', 'hybrid_wo_meta', 'hybrid_wo_lagrangian', 'sac_lagrangian']
base_dir = ROOT / 'logs' / 'hard_stress_full_ablation_baseline_v2' / 'hard_stress'
curves = {m: defaultdict(list) for m in methods_fig5}
for method in methods_fig5:
    for p in sorted(base_dir.glob(f'hard_stress_{method}_seed*/checkpoint_selection.csv')):
        rows = list(csv.DictReader(open(p)))
        for row in rows:
            it = int(float(row['iter']))
            curves[method][('score', it)].append(float(row['score']))
            curves[method][('reward', it)].append(float(row['reward']))
            curves[method][('violation_rate', it)].append(float(row['violation_rate']))

fig, axes = plt.subplots(1, 3, figsize=(32.0, 11.5), dpi=220, constrained_layout=True)
panels = [('score', 'Held-out Evaluation Score'), ('reward', 'Held-out Reward'), ('violation_rate', 'Held-out Violation Rate')]
for ax, (key, title) in zip(axes, panels):
    handles = []
    for method in methods_fig5:
        its = sorted({it for (k, it) in curves[method].keys() if k == key})
        means = []
        stds = []
        for it in its:
            arr = np.array(curves[method][(key, it)], dtype=float)
            means.append(arr.mean())
            stds.append(arr.std(ddof=0))
        means = np.array(means)
        stds = np.array(stds)
        line, = ax.plot(its, means, color=COLORS[method], linewidth=2.2, label=LABELS[method])
        ax.fill_between(its, means - stds, means + stds, color=COLORS[method], alpha=0.18)
        handles.append(line)
    ax.set_title(title, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
fig.legend(
    handles,
    [LABELS[m] for m in methods_fig5],
    loc='lower center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=4,
    frameon=False,
)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=10 / 72, hspace=0.03, wspace=0.04)
savefig(fig, 'Fig5_hard_stress_training_dynamics')

# Fig. 6
THERM_ROOT, THERM_SCENARIO = detect_thermal_source()
therm_run = load_formal_run_summary(THERM_ROOT / 'run_summary.json')
therm_methods_pref = ['hybrid', 'hybrid_wo_lagrangian', 'hybrid_hard_clip', 'sac_lagrangian', 'shin2024_matched', 'dalal2018_safe']
therm_methods = [m for m in therm_methods_pref if any(row['variant'] == m and row['scenario'] == THERM_SCENARIO for row in therm_run)]
if not therm_methods:
    raise RuntimeError('No formally comparable methods available for thermal figures')
therm_stats = defaultdict(lambda: defaultdict(list))
for row in therm_run:
    if row['scenario'] != THERM_SCENARIO or row['variant'] not in therm_methods:
        continue
    method = row['variant']
    for metric, dest in [
        ('eval_reward', 'reward'),
        ('eval_cost', 'cost'),
        ('eval_violation_rate', 'violation'),
        ('env_temp_max_q90', 'temp_q90'),
    ]:
        therm_stats[method][dest].append(float(row[metric]))
with open(THERM_ROOT / THERM_SCENARIO / 'env.csv', newline='') as f:
    r = csv.DictReader(f)
    by_vs = defaultdict(lambda: {'n': 0, 'thermal': 0})
    for row in r:
        if row['variant'] not in therm_methods:
            continue
        key = (row['variant'], int(float(row['seed'])))
        by_vs[key]['n'] += 1
        thermal = (float(row['cost_temp_anchor']) + float(row['cost_temp_boost1']) + float(row['cost_temp_boost2'])) > 0
        by_vs[key]['thermal'] += thermal
    for (variant, _seed), info in by_vs.items():
        therm_stats[variant]['thermal_frac'].append(info['thermal'] / info['n'])

fig, axes = plt.subplots(1, 5, figsize=(42.0, 12.0), dpi=220, constrained_layout=True)
panels = [
    ('reward', 'Reward'),
    ('cost', 'Cost'),
    ('violation', 'Violation Rate'),
    ('thermal_frac', 'Thermal Violation Fraction'),
    ('temp_q90', 'Temperature q90'),
]
x = np.arange(len(therm_methods))
for ax, (metric, title) in zip(axes, panels):
    means = []
    stds = []
    for method in therm_methods:
        arr = np.array(therm_stats[method][metric], dtype=float)
        means.append(float(arr.mean()))
        stds.append(float(arr.std(ddof=0)))
    plot_means = means[:]
    if metric == 'cost':
        plot_means = [max(v, 1e-5) for v in means]
    ax.bar(x, plot_means, color=[COLORS[m] for m in therm_methods], edgecolor='black', linewidth=0.6)
    ax.errorbar(x, plot_means, yerr=stds, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
    style_bar_axis(ax, title)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in therm_methods], rotation=22, ha='right')
    if metric == 'cost':
        ax.set_yscale('log')
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.04, wspace=0.08)
savefig(fig, 'Fig6_thermal_targeted_results_v2')

# Fig. 7
therm_env_root = THERM_ROOT / THERM_SCENARIO
by_vs = defaultdict(lambda: {'n': 0, 'qos': 0, 'thermal': 0})
with open(therm_env_root / 'env.csv', newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        if row['variant'] not in therm_methods:
            continue
        key = (row['variant'], int(float(row['seed'])))
        by_vs[key]['n'] += 1
        by_vs[key]['qos'] += float(row['cost_qos']) > 0
        by_vs[key]['thermal'] += (float(row['cost_temp_anchor']) + float(row['cost_temp_boost1']) + float(row['cost_temp_boost2'])) > 0
viol = defaultdict(lambda: defaultdict(list))
for (variant, _seed), info in by_vs.items():
    viol[variant]['qos'].append(info['qos'] / info['n'])
    viol[variant]['thermal'].append(info['thermal'] / info['n'])
lam = defaultdict(lambda: defaultdict(list))
for method in therm_methods:
    for p in sorted(therm_env_root.glob(f'{THERM_SCENARIO}_{method}_seed*/metrics.csv')):
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        last = rows[-1]
        for key in ['lambda_temp_anchor', 'lambda_temp_boost1', 'lambda_temp_boost2', 'lambda_qos']:
            if key in last:
                lam[method][key].append(float(last[key]))

fig = plt.figure(figsize=(34.0, 12.5), dpi=220, constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])

# Panel A
xA = np.arange(len(therm_methods))
widthA = 0.32
q_means = [np.mean(viol[m]['qos']) for m in therm_methods]
q_stds = [np.std(viol[m]['qos'], ddof=0) for m in therm_methods]
t_means = [np.mean(viol[m]['thermal']) for m in therm_methods]
t_stds = [np.std(viol[m]['thermal'], ddof=0) for m in therm_methods]
axA.bar(xA - widthA / 2, q_means, width=widthA, color='#4C78A8', edgecolor='black', linewidth=0.8, label='QoS violation fraction')
axA.bar(xA + widthA / 2, t_means, width=widthA, color='#E45756', edgecolor='black', linewidth=0.8, label='Thermal violation fraction')
axA.errorbar(xA - widthA / 2, q_means, yerr=q_stds, fmt='none', ecolor='black', elinewidth=1.0, capsize=3)
axA.errorbar(xA + widthA / 2, t_means, yerr=t_stds, fmt='none', ecolor='black', elinewidth=1.0, capsize=3)
axA.set_xticks(xA)
axA.set_xticklabels([LABELS[m] for m in therm_methods], rotation=16, ha='right')
axA.set_ylim(0, 1.05)
axA.set_title('Panel A. Constraint Decomposition', fontsize=34, weight='bold')
axA.set_ylabel('Violation Fraction')
axA.grid(axis='y', linestyle='--', alpha=0.35)
axA.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=26)
axA.set_axisbelow(True)

# Panel B
lam_specs = [
    ('lambda_temp_anchor', r'$\lambda_{T_A}$', '#1f77b4'),
    ('lambda_temp_boost1', r'$\lambda_{T_{B1}}$', '#2ca02c'),
    ('lambda_temp_boost2', r'$\lambda_{T_{B2}}$', '#ff7f0e'),
]
main_methods = [m for m in therm_methods if m not in {'sac_lagrangian', 'hybrid_wo_lagrangian'} and any(lam[m][spec[0]] for spec in lam_specs)]
inset_method = 'sac_lagrangian' if any(lam['sac_lagrangian'][spec[0]] for spec in lam_specs) else None
xB = np.arange(len(main_methods))
widthB = 0.22
for i, (key, label, color) in enumerate(lam_specs):
    means = [np.mean(lam[m][key]) if lam[m][key] else 0.0 for m in main_methods]
    stds = [np.std(lam[m][key], ddof=0) if lam[m][key] else 0.0 for m in main_methods]
    if main_methods:
        axB.bar(xB + (i - 1) * widthB, means, width=widthB, color=color, edgecolor='black', linewidth=0.8, label=label)
        axB.errorbar(xB + (i - 1) * widthB, means, yerr=stds, fmt='none', ecolor='black', elinewidth=1.0, capsize=3)
axB.set_xticks(xB)
axB.set_xticklabels([LABELS[m] for m in main_methods], rotation=16, ha='right')
axB.set_ylim(0, max(0.3, max((np.mean(lam[m][spec[0]]) if lam[m][spec[0]] else 0.0) for m in main_methods for spec in lam_specs) * 1.35 if main_methods else 0.3))
axB.set_title('Panel B. Thermal Multiplier Activation', fontsize=34, weight='bold')
axB.set_ylabel('Final Mean Value Across Seeds')
axB.grid(axis='y', linestyle='--', alpha=0.35)
axB.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=24)
axB.set_axisbelow(True)

if inset_method is not None:
    ins = inset_axes(axB, width='34%', height='45%', loc='upper center', borderpad=1.0)
    xi = np.arange(1)
    for i, (key, _label, color) in enumerate(lam_specs):
        arr = np.array(lam[inset_method][key], dtype=float) if lam[inset_method][key] else np.array([0.0])
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        ins.bar(xi + (i - 1) * widthB, [mean], width=widthB, color=color, edgecolor='black', linewidth=0.7)
        ins.errorbar(xi + (i - 1) * widthB, [mean], yerr=[std], fmt='none', ecolor='black', elinewidth=0.9, capsize=2)
    ins.set_xticks([])
    ins.set_xticklabels([])
    inset_max = max((np.max(lam[inset_method][spec[0]]) if lam[inset_method][spec[0]] else 0.0) for spec in lam_specs)
    ins.set_ylim(0, max(1.0, inset_max * 1.15))
    ins.set_title('Inset: SAC-Lagrangian', fontsize=18, weight='bold')
    ins.grid(axis='y', linestyle='--', alpha=0.25)
    ins.set_axisbelow(True)

fig.set_constrained_layout_pads(w_pad=5 / 72, h_pad=8 / 72, hspace=0.05, wspace=0.08)
savefig(fig, 'Fig7_thermal_activation_decomposition_v2')

optional_src = ROOT / 'logs' / 'paper_figs_bench_cpu_full' / 'Fig6_hard_stress_stepwise_stability.png'
if optional_src.exists():
    shutil.copy2(optional_src, OUT / 'Fig8_optional_hard_stress_bridge_stability.png')

copy_map = {
    ROOT / 'logs' / 'paper_figs_bench_cpu_full' / 'Fig3_scenario_realism_triptych.png': OUT / 'Appendix_Fig_A_scenario_realism_triptych.png',
}
for src, dst in copy_map.items():
    if src.exists():
        shutil.copy2(src, dst)

redraw_env_realism(
    ROOT / 'logs' / 'bench_cpu_full' / 'moderate_practical' / 'env.csv',
    'Fig8a_moderate_practical_realism',
)
redraw_env_realism(
    ROOT / 'logs' / 'bench_cpu_full' / 'hard_stress' / 'env.csv',
    'Fig8b_hard_stress_realism',
)
redraw_env_realism(
    ROOT / 'logs' / 'bench_cpu_full' / 'channel_harsh' / 'env.csv',
    'Fig8c_channel_harsh_realism',
)
redraw_current_allocation(
    ROOT / 'logs' / 'bench_cpu_full' / 'moderate_practical' / 'current_trace.csv',
    'Appendix_Fig_B_moderate_practical_current_allocation',
)
redraw_utilization_tradeoff(
    ROOT / 'logs' / 'bench_cpu_full' / 'channel_harsh' / 'env.csv',
    'Appendix_Fig_C_channel_harsh_utilization_tradeoff',
)

therm_detail = defaultdict(lambda: defaultdict(list))
with open(ROOT / 'logs' / 'thermal_extreme_targeted_v2' / 'thermal_extreme' / 'env.csv', newline='') as f:
    r = csv.DictReader(f)
    by_vs = defaultdict(lambda: {'n': 0, 'ta_v': 0, 'tb1_v': 0, 'tb2_v': 0, 'ta_cost': 0.0, 'tb1_cost': 0.0, 'tb2_cost': 0.0})
    for row in r:
        key = (row['variant'], int(float(row['seed'])))
        by_vs[key]['n'] += 1
        cta = float(row['cost_temp_anchor'])
        ctb1 = float(row['cost_temp_boost1'])
        ctb2 = float(row['cost_temp_boost2'])
        by_vs[key]['ta_v'] += cta > 0
        by_vs[key]['tb1_v'] += ctb1 > 0
        by_vs[key]['tb2_v'] += ctb2 > 0
        by_vs[key]['ta_cost'] += cta
        by_vs[key]['tb1_cost'] += ctb1
        by_vs[key]['tb2_cost'] += ctb2
    for (variant, _seed), info in by_vs.items():
        therm_detail[variant]['ta_v'].append(info['ta_v'] / info['n'])
        therm_detail[variant]['tb1_v'].append(info['tb1_v'] / info['n'])
        therm_detail[variant]['tb2_v'].append(info['tb2_v'] / info['n'])
        therm_detail[variant]['ta_cost'].append(info['ta_cost'] / info['n'])
        therm_detail[variant]['tb1_cost'].append(info['tb1_cost'] / info['n'])
        therm_detail[variant]['tb2_cost'].append(info['tb2_cost'] / info['n'])
fig, axes = plt.subplots(1, 2, figsize=(30.0, 11.0), dpi=220, constrained_layout=True)
width = 0.22
x = np.arange(len(therm_methods))
for i, (key, label, color) in enumerate([('ta_v', 'Temp A', '#1f77b4'), ('tb1_v', 'Temp B1', '#2ca02c'), ('tb2_v', 'Temp B2', '#ff7f0e')]):
    means = [np.mean(therm_detail[m][key]) for m in therm_methods]
    stds = [np.std(therm_detail[m][key], ddof=0) for m in therm_methods]
    axes[0].bar(x + (i - 1) * width, means, width=width, color=color, edgecolor='black', linewidth=0.6, label=label)
    axes[0].errorbar(x + (i - 1) * width, means, yerr=stds, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
axes[0].set_xticks(x)
axes[0].set_xticklabels([LABELS[m] for m in therm_methods], rotation=18, ha='right')
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Per-Transmitter Thermal Violation Fractions', weight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.35)
axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=30)
axes[0].set_axisbelow(True)
for i, (key, label, color) in enumerate([('ta_cost', 'Temp A', '#1f77b4'), ('tb1_cost', 'Temp B1', '#2ca02c'), ('tb2_cost', 'Temp B2', '#ff7f0e')]):
    means = [np.mean(therm_detail[m][key]) for m in therm_methods]
    stds = [np.std(therm_detail[m][key], ddof=0) for m in therm_methods]
    axes[1].bar(x + (i - 1) * width, means, width=width, color=color, edgecolor='black', linewidth=0.6, label=label)
    axes[1].errorbar(x + (i - 1) * width, means, yerr=stds, fmt='none', ecolor='black', elinewidth=0.8, capsize=2)
axes[1].set_xticks(x)
axes[1].set_xticklabels([LABELS[m] for m in therm_methods], rotation=18, ha='right')
axes[1].set_title('Per-Transmitter Mean Thermal Cost', weight='bold')
axes[1].grid(axis='y', linestyle='--', alpha=0.35)
axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=30)
axes[1].set_axisbelow(True)
savefig(fig, 'Appendix_Fig_D_detailed_thermal_breakdown_v2')

manifest = OUT / 'MANIFEST.md'
manifest.write_text(
    '# Paper Figure Pack (v2 Thermal Validation)\n\n'
    '## Main Text Figures\n'
    '- Fig3_main_benchmark_overall\n'
    '- Fig4_hard_stress_ablation_baselines\n'
    '- Fig5_hard_stress_training_dynamics\n'
    '- Fig6_thermal_targeted_results_v2\n'
    '- Fig7_thermal_activation_decomposition_v2\n'
    '- Fig8_optional_hard_stress_bridge_stability\n\n'
    '## Appendix Figures\n'
    '- Appendix_Fig_A_scenario_realism_triptych\n'
    '- Appendix_Fig_B_moderate_practical_current_allocation\n'
    '- Appendix_Fig_C_channel_harsh_utilization_tradeoff\n'
    '- Appendix_Fig_D_detailed_thermal_breakdown_v2\n\n'
    '## Notes\n'
    '- Main benchmark figures are based on logs/bench_cpu_full.\n'
    '- Hard-stress targeted figures are based on logs/hard_stress_full_ablation_baseline_v2.\n'
    '- Thermal targeted figures are based on logs/thermal_extreme_targeted_v2.\n'
    '- Fig. 3 uses symlog-style handling for cost/violation because exact zeros exist in the main benchmark results.\n'
)

print('regenerated', OUT)

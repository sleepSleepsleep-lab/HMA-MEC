# -*- coding: utf-8 -*-
"""
Unified academic color palette for all figures in the paper.

Design principles:
  - ColorBrewer-derived, perceptually uniform and printer-friendly.
  - Distinguishable in both color and grayscale.
  - High contrast between HMA-MEC variants and baselines; baselines share
    a muted gray family so the proposed method pops out.
  - Consistent mapping: a given method has the same color across all figures.

Usage:
    from visual.palette import PALETTE, method_color, set_paper_style

Public API
----------
PALETTE            : dict of all named colors used in the paper
METHOD_COLORS      : {method_name -> hex color}
PERTURB_COLORS     : {perturbation_type -> hex color}
ROLE_COLORS        : Agent role -> hex color (for fig_arch)
set_paper_style()  : apply unified matplotlib rcParams (fonts, dpi, sizes)
apply_method_colors(plt, axis_or_fig)
"""

import matplotlib
import matplotlib.pyplot as plt


# ============================================================
# 1. Named color tokens (single source of truth)
# ============================================================
PALETTE = {
    # Primary brand: deep teal-navy, used for HMA-MEC family
    'hma_primary':    '#2E5E7E',   # main HMA-Distill
    'hma_hybrid':     '#1E3A5F',   # HMA-Hybrid (slightly darker)
    'hma_fullllm':    '#5B8AA6',   # HMA-FullLLM (lighter teal)

    # Baselines: vibrant, distinct colors
    'sac':            '#D94040',   # bright red for primary DRL baseline
    'ddpg':           '#E08B2B',   # vivid orange
    'greedy':         '#3E7DAA',   # medium blue
    'alllocal':       '#8B5E3C',   # warm brown
    'alledge':        '#3E8E61',   # forest green
    'random':         '#7E5A9E',   # medium purple
    'comllm':         '#D4A574',   # warm tan (single LLM baseline)

    # Perturbation colors (E6)
    'channel_drop':   '#3E6F89',
    'server_fail':    '#C45B5B',
    'dishonest_ua':   '#7E8C5A',

    # Agent roles (fig_arch)
    'role_env':       '#8DA9C4',
    'role_ua':        '#588B8B',
    'role_ea':        '#8FB992',
    'role_oa':        '#D4A574',
    'role_va':        '#C45B5B',
    'role_distill':   '#2E5E7E',

    # Semantic
    'primary':        '#2E5E7E',
    'secondary':      '#C45B5B',
    'highlight':      '#D4A574',
    'neutral_dark':   '#3A3F45',
    'neutral_med':    '#8C99A8',
    'neutral_light':  '#F4F6F8',
    'grid':           '#D9DEE3',
    'arrow':          '#555555',
    'arrow_hot':      '#AA3333',
}

# ============================================================
# 2. Method -> color mapping (used by E1, E2, E5, E7, E8, E9)
# ============================================================
METHOD_COLORS = {
    'Greedy':        PALETTE['greedy'],
    'AllLocal':      PALETTE['alllocal'],
    'AllEdge':       PALETTE['alledge'],
    'Random':        PALETTE['random'],
    'SAC':           PALETTE['sac'],
    'DDPG':          PALETTE['ddpg'],
    'B7-COMLLM-lite':PALETTE['comllm'],
    'HMA-Distill':   PALETTE['hma_primary'],
 'HMA-Hybrid':    PALETTE['hma_hybrid'],
    'HMA-FullLLM':   PALETTE['hma_fullllm'],
    'HMA':           PALETTE['hma_primary'],
}

PERTURB_COLORS = {
    'channel_drop':   PALETTE['channel_drop'],
    'server_fail':    PALETTE['server_fail'],
    'dishonest_ua':   PALETTE['dishonest_ua'],
}

ROLE_COLORS = {
    'env':      PALETTE['role_env'],
    'ua':       PALETTE['role_ua'],
    'ea':       PALETTE['role_ea'],
    'oa':       PALETTE['role_oa'],
    'va':       PALETTE['role_va'],
    'distill':  PALETTE['role_distill'],
}


# ============================================================
# 3. Convenience accessor
# ============================================================
def method_color(name: str) -> str:
    """Return color for a method name. Falls back to neutral gray."""
    return METHOD_COLORS.get(name, PALETTE['neutral_med'])


def perturb_color(name: str) -> str:
    return PERTURB_COLORS.get(name, PALETTE['neutral_med'])


def role_color(name: str) -> str:
    return ROLE_COLORS.get(name, PALETTE['neutral_med'])


# ============================================================
# 4. Unified matplotlib style
# ============================================================
def set_paper_style():
    """Apply paper-wide matplotlib style: fonts, sizes, dpi, gridlines.

    Safe to call at the top of any fig_*.py module.
    """
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.serif':        ['DejaVu Serif', 'Times New Roman'],
        'font.sans-serif':   ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset':  'dejavusans',
        'axes.unicode_minus': False,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
        'savefig.dpi':        1200,
        'figure.dpi':         120,
        'axes.labelsize':     9,
        'axes.titlesize':     9,
        'xtick.labelsize':    8,
        'ytick.labelsize':    8,
        'legend.fontsize':     7,
        'legend.frameon':     False,
        'axes.edgecolor':     PALETTE['neutral_dark'],
        'axes.linewidth':     0.6,
        'grid.color':         PALETTE['grid'],
        'grid.linewidth':     0.4,
        'grid.alpha':         0.7,
        'axes.grid':          True,
        'axes.axisbelow':     True,
        'lines.linewidth':    1.2,
        'lines.markersize':   4,
        'patch.linewidth':    0.4,
        'patch.edgecolor':    PALETTE['neutral_dark'],
    })


# Marker / linestyle cycle to enhance grayscale distinction
_METHOD_MARKERS = {
    'Greedy':     's',
    'AllLocal':   '^',
    'AllEdge':    'v',
    'Random':     'D',
    'SAC':        'o',
    'DDPG':       'P',
    'HMA-Distill':'o',
    'HMA-Hybrid': 'D',
    'HMA-FullLLM':'^',
    'HMA':        'o',
}


def method_marker(name: str) -> str:
    return _METHOD_MARKERS.get(name, 'o')


if __name__ == "__main__":
    set_paper_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    for i, (name, c) in enumerate(METHOD_COLORS.items()):
        ax.plot([0, 1], [i, i], color=c, marker=method_marker(name),
                label=name, linewidth=2)
    ax.set_yticks(range(len(METHOD_COLORS)))
    ax.set_yticklabels(list(METHOD_COLORS.keys()))
    ax.set_xlabel('color preview')
    ax.set_title('HMA-MEC unified palette')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('palette_preview.png', dpi=200, bbox_inches='tight')
    print('palette_preview.png saved')
# -*- coding: utf-8 -*-
"""Fig 2: Distill-Agent policy network architecture. English only.

Layout note (publication QA pass):
  - Drawing space is expressed in points (1 data unit == 1 pt): the final
    canvas is sized to the content after layout, so nothing is cropped and
    no whitespace remains.
  - Every box is sized to its measured label extent + padding (PX/PY), which
    guarantees the label text never touches the box border or arrowheads.
  - Column order / row pairing / arrow topology identical to the original.
"""

import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(ROOT, "latex", "figure")
os.makedirs(OUT_DIR, exist_ok=True)

C_BOX = {'input': '#7BA7D9', 'shared': '#8DCCC2', 'head': '#EAC49A',
         'conf': '#8A6C87', 'output': '#71C58B'}

TARGET_W = 515.52                 # max canvas width in pt (7.16 in)
PX, PY, RND = 5.5, 5.0, 4.5       # box padding / corner rounding (pt)
GAP_AB, GAP_BC, GAP_CD = 18, 20, 16   # arrow-zone widths (pt)
MARGIN = 6.0                      # horizontal canvas margin (pt)
MARGIN_Y = 14.0                   # vertical canvas margin (pt)
ROWGAP = 20.0                     # vertical gap between stacked rows (pt)

# label catalogue (column -> [(text, base fontsize), ...])
A = (r"$s_t\in\mathbb{R}^{4K+2M}$", 11.0)
B = ("LN -> LkReLU\n" + r"-> 256 -> LN -> LkReLU -> 256", 7.5)
C = [  # 4 head boxes, top -> bottom
    (r"$\mu_\alpha=\sigma(W_\alpha h)$", 10.0),
    (r"$\sigma_\alpha=\exp(\mathrm{clip}(W_\sigma h))$", 10.0),
    (r"$\ell_{\text{server}}\in\mathbb{R}^{K\times M}$", 10.0),
    (r"$c_\text{min}=\sigma(w_c^\top \mathrm{LkReLU}(W_c h))$", 10.0),
]
D = [  # 4 output boxes, top -> bottom (2nd row: two-line label)
    (r"$\alpha_k\sim\mathrm{Beta}(\mu_\alpha\kappa,(1-\mu_\alpha)\kappa)$"
     "\n" + r"$\to \alpha^\star \in [0.01, 1.0]$", 9.0),
    (r"$\kappa = 1/\sigma_\alpha^2 - 1$", 9.0),
    (r"$m_k=\arg\max\mathrm{softmax}(\ell_{\text{server},k})$", 9.0),
    (r"Hybrid trigger: $c_\text{min} < \tau_\text{low}$", 9.0),
]


def main():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42; plt.rcParams['ps.fonttype'] = 42

    # ---- measure text extents on a throwaway canvas (font/DPI independent) --
    tfig = plt.figure(figsize=(8, 6), dpi=100)
    tfig.canvas.draw()
    rend = tfig.canvas.get_renderer()

    def measure(s, fs):
        t = tfig.text(0, 0, s, fontsize=fs, weight='bold',
                      ha='center', va='center')
        bb = t.get_window_extent(renderer=rend)
        w = bb.width * 72.0 / tfig.dpi
        h = bb.height * 72.0 / tfig.dpi
        t.remove()
        return w, h

    def widths(labels, k):
        return [measure(s, fs * k)[0] for s, fs in labels]

    # ---- font-size shrink factor so total width fits the target canvas ----
    def box_w_total(k):
        return (max(widths([A], k)) + max(widths([B], k))
                + max(widths(C, k)) + max(widths(D, k)) + 8 * PX)

    k = 1.0
    budget = TARGET_W - 2 * MARGIN - GAP_AB - GAP_BC - GAP_CD
    while box_w_total(k) > budget and k > 0.6:
        k *= 0.99

    # ---- final extents at the resolved font size ----
    wA, hA = measure(A[0], A[1] * k)
    wB, hB = measure(B[0], B[1] * k)
    wC = widths(C, k); hC = [measure(s, fs * k)[1] for s, fs in C]
    wD = widths(D, k); hD = [measure(s, fs * k)[1] for s, fs in D]

    WA = wA + 2 * PX; HA = hA + 2 * PY          # input box
    WB = wB + 2 * PX; HB = hB + 2 * PY          # shared box
    WC = max(wC) + 2 * PX; HC = max(hC) + 2 * PY
    WD = max(wD) + 2 * PX; HD = max(hD) + 2 * PY

    xA = MARGIN
    xB = xA + WA + GAP_AB
    xC = xB + WB + GAP_BC
    xD = xC + WC + GAP_CD
    W_PT = xD + WD + MARGIN

    rowH = max(HC, HD)
    yc0 = MARGIN_Y + rowH / 2.0
    yc = [yc0 + i * (rowH + ROWGAP) for i in range(4)]
    yMid = (yc[1] + yc[2]) / 2.0          # input / shared lane
    H_PT = yc[3] + rowH / 2.0 + MARGIN_Y

    assert xD + WD <= TARGET_W - 0.5, "layout overflows target width"
    print(f"  [layout] k={k:.3f}  page={W_PT:.0f}x{H_PT:.0f}pt  "
          f"(rowH={rowH:.0f}, fonts A/B/C/D = "
          f"{A[1]*k:.1f}/{B[1]*k:.1f}/{C[0][1]*k:.1f}/{D[0][1]*k:.1f})")

    # ---- final figure sized to content ----
    fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_PT); ax.set_ylim(0, H_PT)
    ax.set_aspect('auto'); ax.axis('off')

    def rbox(x, y, w, h, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={RND}",
            linewidth=1.0, edgecolor='black',
            facecolor=color, alpha=0.85, zorder=1))

    def label(cx, cy, s, fs, color='black'):
        ax.text(cx, cy, s, ha='center', va='center',
                fontsize=fs, weight='bold', color=color, zorder=4)

    # input / shared lane
    rbox(xA, yMid - HA / 2, WA, HA, C_BOX['input'])
    label(xA + WA / 2, yMid, A[0], A[1] * k)
    rbox(xB, yMid - HB / 2, WB, HB, C_BOX['shared'])
    label(xB + WB / 2, yMid, B[0], B[1] * k)

    # head + output boxes on the four rows
    for i in range(4):
        rbox(xC, yc[i] - HC / 2, WC, HC, C_BOX['head'])
        label(xC + WC / 2, yc[i], C[i][0], C[i][1] * k)
        col = C_BOX['conf'] if i == 3 else C_BOX['output']
        rbox(xD, yc[i] - HD / 2, WD, HD, col)
        label(xD + WD / 2, yc[i], D[i][0], D[i][1] * k)

    # ---- arrows ----
    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=3,
                    arrowprops=dict(arrowstyle='-|>', lw=0.9, color='#555'))

    arrow(xA + WA, yMid, xB, yMid)                   # input -> shared
    for i in range(4):                               # shared -> heads
        arrow(xB + WB, yMid, xC, yc[i])
        arrow(xC + WC, yc[i], xD, yc[i])             # head -> output

    fig.savefig(os.path.join(OUT_DIR, "fig_distill_net.pdf"),
                format="pdf", dpi=1200)
    fig.savefig(os.path.join(OUT_DIR, "fig_distill_net.tiff"),
                format="tiff", dpi=1200,
                pil_kwargs={"compression": "tiff_lzw"})
    print("[OK] fig_distill_net.pdf / .tiff")


if __name__ == "__main__":
    main()

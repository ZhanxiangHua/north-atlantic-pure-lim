"""Render the numbered mathematical flowchart in the annual LIM tutorial."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def render(folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42, "pdf.use14corefonts": False})
    fig = plt.figure(figsize=(12.7, 10.7), facecolor="white")
    ax = fig.add_axes([.02, .02, .96, .96], xlim=(0, 12), ylim=(0, 10))
    ax.set_axis_off()
    ink, muted = "#182f3c", "#364e5b"
    neutral, background, weather = "#415d68", "#006f65", "#405b9a"

    def box(x, y, w, h, title, formula, detail, color, fill, math_size=14):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055,rounding_size=0.10",
                                    linewidth=1.15, edgecolor=color, facecolor=fill, zorder=2))
        ax.text(x+.19, y+h-.19, title, fontsize=15, fontweight="bold", color=color, va="top")
        ax.text(x+.19, y+h*.52, formula, fontsize=math_size, color=ink, va="center")
        ax.text(x+.19, y+.18, detail, fontsize=10.8, color=muted, va="bottom")

    def line(points):
        ax.plot(*zip(*points), color="#70828a", lw=1.25, zorder=0, solid_capstyle="round")

    def arrow(x, top, bottom):
        ax.add_patch(FancyArrowPatch((x, top), (x, bottom), arrowstyle="-|>", mutation_scale=15,
                                     lw=1.25, color="#617681", zorder=1, shrinkA=0, shrinkB=0))
    left, right, width = .17, 6.22, 5.6
    lc, rc = left + width/2, right + width/2
    ax.text(.12, 9.79, "How a monthly background becomes daily wind", fontsize=24,
            weight="bold", va="top", color=ink)
    ax.text(.14, 9.35, "The numbers refer to sections in the annual pure LIM tutorial.", fontsize=13,
            va="top", color=muted)
    box(left, 7.92, width, 1.05, "1   Choose the warming and month",
        r"$g=\overline{T}_{y}-\overline{T}_{1850–1900}$",
        "The calendar month selects a fitted parameter set.", neutral, "#f1f4f5")
    box(right, 7.92, width, 1.05, "2   Represent maps with EOFs",
        r"$c=U^{\mathsf{T}}D(f-\bar{f})$",
        "Keep 78 background scores and 120 wind scores.", neutral, "#f1f4f5")
    line([(lc, 7.86), (lc, 7.59), (rc, 7.59), (rc, 7.86)])
    for x in (lc, rc):
        arrow(x, 7.59, 7.34)
    box(left, 6.0, width, 1.28, "3   Draw a monthly background",
        r"$z=m(g)+F(g)e,\qquad e\sim\mathcal{N}(0,I)$",
        "The covariance of the draw is F(g) F(g)ᵀ.", background, "#eaf5f0")
    box(right, 6.0, width, 1.28, "4   Advance the two-lag wind state",
        r"$s_t=r A s_{t-1}+\eta_t,\qquad x_t=P s_t$",
        "P selects the first 120 entries of the 240-entry state.", weather, "#eef2fa")
    for x in (lc, rc):
        arrow(x, 5.94, 5.58)
    box(left, 4.20, width, 1.32, "3   Transfer the background to wind",
        r"$\mu_w=a(g)+K^{\mathsf{T}}[z-m(g)]$",
        "At fixed g and z, every weather draw has this mean.", background, "#eaf5f0")
    box(right, 4.20, width, 1.32, "4–5   Generate a daily anomaly",
        r"$h_t=s(g)\,\kappa\,[x_t-b\,\overline{x}_{t-20:t-5}]$",
        "The bar averages 16 past days. Each day adds new noise.", weather, "#eef2fa", 13.7)
    line([(lc, 4.14), (lc, 3.91), (rc, 3.91), (rc, 4.14)])
    arrow(6, 3.91, 3.64)
    box(left, 2.30, 11.65, 1.28, "5   Add the scores and reconstruct the event",
        r"$w_t=\mu_w+h_t,\qquad f^{\mathrm{wind}}_t=\bar{f}_w+D_w^{-1}U_w w_t$",
        "Decode z once for the monthly ocean and atmosphere. Decode each wₜ for the four daily wind maps.",
        neutral, "#f2f5f6", 16)
    line([(6, 2.24), (6, 2.02), (lc, 2.02), (rc, 2.02)])
    for x in (lc, rc):
        arrow(x, 2.02, 1.77)
    box(left, .20, width, 1.51, "6, 8   Test multi-day persistence",
        r"$V_d=\frac{d C_0+2\sum_{h=1}^{d-1}(d-h)C_h}{d^2}$",
        "Check sampled variance and compare with held-out MPI.", neutral, "#f6f7f4", 15)
    box(right, .20, width, 1.51, "7, 9   Assemble a year and reproduce",
        r"$N_{\mathrm{days}}=\sum_{m=1}^{12}d_m\ \in\ \{365,366\}$",
        "Restart each month. Section 9 gives the fit and run commands.", neutral, "#f6f7f4", 14)
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "annual_lim_math_flow.png", dpi=160)
    fig.savefig(folder / "annual_lim_math_flow.pdf")
    plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1] / "reports")

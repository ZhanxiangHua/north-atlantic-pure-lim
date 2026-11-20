"""Draw the input, calculation and output at each interface in the two tutorials."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# Newlines are intentional. Each page is also displayed on its own in a notebook.
TRAINING = [
    ("Turn a gridded field into coordinates", "§§1–2", [
        ("Input", "A training map f and its average map f̄. Use one calendar month at a time.\nA background map represents a month. A wind map represents one day."),
        ("Compute", r"$c = U^{\mathsf{T}} D (f - \bar f)$"),
        ("Meaning", "D weights and scales each grid value. Each column of U is a spatial pattern\n(an EOF). Its dot product with the weighted anomaly gives one score in c."),
        ("Output", "Join the four background blocks into z (78 scores per month). The wind block\ngives wₜ (120 scores per day). Save the means, weights and patterns."),
    ]),
    ("Learn the expected wind for a given background", "§3", [
        ("Input", "For each training month, pair its warming g and background scores z with\nthe average of that month’s daily wind scores."),
        ("Compute", r"$\mu = a(g) + K^{\mathsf{T}}[z - m_z(g)]$"),
        ("Meaning", "$m_z(g)$ is the average background at warming g. Each column of K weights the 78\ndepartures for one wind score. $a(g)$ gives the wind at the average background."),
        ("Output", "Save the two mean curves and the 78 × 120 transfer matrix K. Also fit the\nbackground spread across members at each warming level to obtain the factor F(g)."),
    ]),
    ("Learn how a wind anomaly carries over to tomorrow", "§§4, 6, 9", [
        ("Input", "Subtract the expected wind μ from each training day’s 120 wind scores.\nCall that residual dₜ. Keep consecutive days within the same member-month."),
        ("Compute", r"$d_t \;\approx\; A_1 d_{t-1} + A_2 d_{t-2}$"),
        ("Meaning", "Fit the two matrices by predicting each day from the preceding two. The errors\nestimate the new random variation that must enter on each day."),
        ("Output", "Save a 240 × 240 transition, noise and stationary covariances, and the fitted\nweather scaling and memory parameters. Page 2 uses these to draw new days."),
    ]),
]

SAMPLING = [
    ("Draw one monthly ocean and atmosphere", "§§1–3", [
        ("Input", "Calendar month, year, warming g in K, seed and background ID. August at +2 K\nselects the August files and evaluates their warming response at g = 2."),
        ("Compute", r"$z=m_z(g)+F(g)e,\qquad \mathrm{Cov}(z\mid g)=F(g)F(g)^{\mathsf{T}}$"),
        ("Meaning", "e contains 78 independent standard normal draws. F mixes and scales them.\nThe covariance on the right describes the spread across many such Augusts."),
        ("Output", "One 78-score background z. Its EOF blocks decode to sea surface temperature,\nmixed-layer depth, stratification, pressure, and temperature and humidity profiles."),
    ]),
    ("Transfer that background to the expected wind", "§3", [
        ("Input", "The same g and the single z from the previous step."),
        ("Compute", r"$\delta z=z-m_z(g),\qquad \mu=a(g)+K^{\mathsf{T}}\delta z$" "\n"
                    r"$\mathrm{Cov}(\mu\mid g)=K^{\mathsf{T}}\,\mathrm{Cov}(z\mid g)\,K$"),
        ("Meaning", "δz measures how this August differs from the average at +2 K. The first line\nshifts its expected wind. The second transfers the spread across many Augusts."),
        ("Output", "One 120-score wind mean μ, shared by all days and all weather IDs for this z."),
    ]),
    ("Generate the part of the wind that changes by day", "§§4–6", [
        ("Input", "A separate weather ID selects a stationary initial history and fresh daily noise ηₜ."),
        ("Compute", r"$X_t=r A X_{t-1}+\eta_t,\quad x_t=(X_t)_{1:120}$" "\n"
                    r"$h_t=s_w(g)\,\kappa\,[x_t-b\,\overline{x}_{t-20:t-5}]$"),
        ("Meaning", "A advances the 240-entry state and r sets retention. Keep its first 120 entries.\nSubtract b times their mean 5–20 days ago. $s_w(g)$ and κ scale the result."),
        ("Output", "One 120-score weather anomaly hₜ for each day t, with a new history for each month."),
    ]),
    ("Add the two contributions and return physical maps", "§§2, 5, 7–8", [
        ("Input", "The fixed mean μ, each day’s hₜ and the saved wind EOFs, weights and mean."),
        ("Compute", r"$w_t=\mu+h_t,\qquad \widehat f_t=\bar f_w+D_w^{-1}U_w w_t$"),
        ("Meaning", "Use wₜ as the 120 weights on the wind patterns. Undo the grid scaling and\nadd the training mean. A finite month’s average includes its weather anomalies."),
        ("Output", "Four maps per day in m/s on 36 latitudes × 58 longitudes. The calendar-year\nproduct has 365 or 366 days and one ocean–atmosphere background per month."),
    ]),
]

STORMS = [
    ("Convert emissions into the warming input", "§1", [
        ("Input", "SSP2-4.5 emissions and forcing inputs, one FaIR parameter set, and a target year y."),
        ("Compute", r"$a_y=(T_y+T_{y+1})/2,\qquad g_y=a_y-\overline{a}_{1850\mathrm{–}1900}$"),
        ("Meaning", "T is global surface temperature at a year boundary. Average adjacent values\nto obtain an annual mean a. Subtract the average over the 51 baseline years."),
        ("Output", "One warming value in K. The FaIR run in this notebook gives +1.813 K for 2050."),
    ]),
    ("Supply the monthly storm environment", "§§2, 5", [
        ("Input", "September weights, warming and a background ID for a generated environment.\nThe fixed-background comparison supplies a held-out MPI month instead."),
        ("Compute", r"$z=m_z(g)+F(g)e$  (generated month)     or     $z=\mathrm{encode}(f_{\rm MPI})$  (fixed month)"),
        ("Meaning", "A draw mixes 78 random numbers into one background. Encoding projects the\nsupplied MPI fields onto the same EOFs. Both routes give a conditional wind mean."),
        ("Output", "Monthly potential intensity, humidity, entropy deficit, mixed-layer depth and\nocean stratification. The LIM and Fourier runs also receive the expected wind."),
    ]),
    ("Choose one wind source for a FAST run", "§§2–3, 5", [
        ("MPI daily", "Read the four daily wind maps from the held-out MPI member."),
        ("LIM daily", "Draw daily wind scores around the conditional mean and decode the four maps."),
        ("Fourier", "Combine the same conditional mean and fitted local covariance with native\nrandom-phase hourly oscillations. Each disturbance gets its own phase draw."),
        ("Output", "Each driver returns eastward and northward wind at 250 and 850 hPa (m/s)\nat the storm’s location and time. FAST interpolates between available wind times."),
    ]),
    ("Integrate disturbances and compare with MPI replay", "§§3–6", [
        ("Input", "Wind, monthly environment, static geography and the same seed panel for each driver."),
        ("Compute", r"$d_{48}=\mathrm{dist}(p_0,\,p_{48})$"),
        ("Meaning", "FAST records position and intensity hourly. On its six-hourly track, take the first fix\nat or above 18 m/s and the fix 48 hours later. dist gives their great-circle separation."),
        ("Output", "Six-hourly tracks, formation and landfall rates per 100 seeds, and the distribution\nof d₄₈ in km. Compare each model with MPI across eight held-out members."),
    ]),
]


def draw_page(title, subtitle, series, number, steps, footer, color):
    width = 10.9
    header = 1.16
    gap = 0.26
    line_height = 0.232
    row_gap = 0.095

    def row_height(label, value):
        line = 0.33 if label == "Compute" else line_height
        return line * (1 + value.count("\n")) + row_gap

    heights = [0.60 + sum(row_height(label, value) for label, value in rows)
               for _, _, rows in steps]
    height = header + sum(heights) + gap * (len(steps) - 1) + 0.55
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    ink, muted, rule = "#192f3b", "#405764", "#bbccd1"

    ax.text(0.34, 0.17, series, va="top", color=color, size=10.5, weight="bold")
    ax.text(width-0.35, 0.17, f"{number} / 3", ha="right", va="top", color=color, size=10.5)
    ax.text(0.31, 0.40, title, color=ink, size=24, weight="bold", va="top")
    ax.text(0.34, 0.88, subtitle, color=muted, size=12, va="top")

    top = header
    for i, ((heading, section, rows), block_height) in enumerate(zip(steps, heights)):
        ax.add_patch(FancyBboxPatch((0.34, top), width-0.68, block_height,
                                    boxstyle="round,pad=0,rounding_size=0.07",
                                    edgecolor=rule, facecolor="#fafcfc", linewidth=0.85))
        ax.text(0.54, top+0.12, heading, color=color, size=15.7, weight="bold", va="top")
        ax.text(width-0.56, top+0.16, section, color=muted, size=10.8, ha="right", va="top")
        y = top + 0.52
        for label, value in rows:
            math_row = label == "Compute"
            ax.text(0.55, y+0.018, label, color=color, size=10.8, weight="bold", va="top")
            ax.text(1.60, y, value, color=ink, size=15.0 if math_row else 12.2,
                    linespacing=1.26, va="top")
            y += row_height(label, value)
        end = top + block_height
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.91, end + 0.012), (0.91, end + gap - 0.015),
                                         arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                         color=color, shrinkA=0, shrinkB=0))
        top = end + gap
    ax.text(0.34, height-0.32, footer, size=10.7, color=muted, va="top")
    return fig


def render(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pages = [
        ("annual_lim_math_flow", "What the model learns from MPI",
         "For August, 35 training members × 15 years supply 525 backgrounds and 16,275 daily wind maps.",
         "ANNUAL PURE LIM   /   TRAINING", TRAINING,
         "Saved for each month in representation.npz, parameters.npz and model.json. Section numbers refer to the annual tutorial.",
         "#09675e"),
        ("annual_lim_sampling_flow", "How to generate a month from the weights",
         "Follow one August at +2 K. Its ocean and expected wind can be reused with different weather IDs.",
         "ANNUAL PURE LIM   /   SAMPLING", SAMPLING,
         "The event file includes interpolation nodes. Section 5 selects the calendar days and section 7 assembles a full year.",
         "#344f90"),
        ("annual_lim_tc_flow", "From a warming pathway to storm tracks",
         "Follow September 2050. The MPI replay supplies the reference for the wind and storm comparisons.",
         "FaIR TO TROPICAL CYCLONES   /   SIMULATION AND BENCHMARK", STORMS,
         "Section numbers on this page refer to the FaIR → EOF → LIM → tropical cyclones tutorial.",
         "#865016"),
    ]
    with plt.rc_context({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "pdf.fonttype": 42}), PdfPages(folder / "annual_lim_math_flow.pdf") as pdf:
        pdf.infodict()["Title"] = "North Atlantic pure LIM and tropical cyclone workflows"
        for number, (name, title, subtitle, series, steps, footer, color) in enumerate(pages, start=1):
            fig = draw_page(title, subtitle, series, number, steps, footer, color)
            fig.savefig(folder / f"{name}.png", dpi=140, facecolor="white")
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).resolve().parents[1] / "reports")

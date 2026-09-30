"""Shareable chart style (the one from the S6E9 SHAP chart that opened the X thread).

Rules:
- The title states the conclusion, not the chart type.
- A subtitle says what is measured, on how many rows and with which model.
- Annotations with guide lines point at what to look at.
- A footer names the competition and the notebook, because the image travels on its own.
- 1600x900 (16:9), so X shows it whole without cropping.
- Dark grey axis labels: light grey is hard to read on a phone.
- Every chart published outside the notebook gets alt text (axes, trend, highlight).

Usage:
    fig, ax = shareable_figure(
        title="More income, more EV purchases",
        subtitle="Mean SHAP contribution · 20,000 validation rows · LightGBM, fold 0",
        footer="Kaggle Playground S6E10 · <notebook title>",
    )
    ax.scatter(x, y, s=2, alpha=0.3)
    annotate(ax, "clipped at 30k", xy=(30_000, 0.1), xytext=(60_000, 1.0))
    save_shareable(fig, "income_shap.png", alt_text="...")
"""

from pathlib import Path

import matplotlib.pyplot as plt

DARK_GREY = "#333333"
MID_GREY = "#555555"
ACCENT = "#d62728"

WIDTH_PX, HEIGHT_PX, DPI = 1600, 900, 100


def shareable_figure(title, subtitle="", footer="", nrows=1, ncols=1, **subplots_kw):
    """Create a 1600x900 figure with a conclusion title, subtitle and footer."""
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI, **subplots_kw
    )
    fig.suptitle(title, x=0.02, y=0.97, ha="left", fontsize=22, fontweight="bold", color=DARK_GREY)
    if subtitle:
        fig.text(0.02, 0.90, subtitle, ha="left", fontsize=14, color=MID_GREY)
    if footer:
        fig.text(0.02, 0.02, footer, ha="left", fontsize=11, color=MID_GREY)
    for ax in fig.axes:
        style_axes(ax)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.12)
    return fig, axes


def style_axes(ax):
    """Dark grey labels and ticks, no top/right spines."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID_GREY)
    ax.tick_params(colors=DARK_GREY, labelsize=12)
    ax.xaxis.label.set_color(DARK_GREY)
    ax.yaxis.label.set_color(DARK_GREY)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)


def annotate(ax, text, xy, xytext, color=ACCENT):
    """Callout with a guide line from the text to the point of interest."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=12,
        color=color,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
    )


def guide_line(ax, x=None, y=None, label=None, color=ACCENT):
    """Dashed vertical (x) or horizontal (y) reference line, e.g. a base rate or a clip value."""
    if x is not None:
        ax.axvline(x, ls="--", lw=1.2, color=color)
        if label:
            ax.text(x, ax.get_ylim()[1], f" {label}", color=color, fontsize=11, va="top")
    if y is not None:
        ax.axhline(y, ls="--", lw=1.2, color=color)
        if label:
            ax.text(ax.get_xlim()[1], y, f"{label} ", color=color, fontsize=11, ha="right", va="bottom")


def save_shareable(fig, path, alt_text):
    """Save at exactly 1600x900 and write the alt text next to the image (same name, .txt)."""
    if not alt_text:
        raise ValueError("A chart published outside the notebook needs alt text.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    path.with_suffix(".txt").write_text(alt_text.strip() + "\n", encoding="utf-8")
    return path

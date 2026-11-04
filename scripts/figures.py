"""Draw the README figures from the ledger-verified CSVs, so no plotted value is typed by hand.

    python scripts/figures.py          # writes docs/figures/*.svg (needs matplotlib)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.ledger import read_ledger  # noqa: E402
from pof.verify.lira import student_t_sf  # noqa: E402
from scripts.report import MAIN_MODEL, MAIN_ROUNDS, MAIN_DATASET, ROOT, load, num  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots 1-6 in fixed order; past three colours can be confused in
# scatter plots, so every seed also has its own marker shape
SEED_COLOR = {42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a", 45: "#eda100", 46: "#e87ba4", 47: "#008300"}
SEED_MARKER = {42: "o", 43: "s", 44: "^", 45: "D", 46: "v", 47: "P"}
CHECK_COLOR = {"v5_p": "#2a78d6", "v3_p": "#eb6834", "v2_p": "#1baf7a"}
CHECK_NAME = {"v5_p": "V5  client's own data", "v3_p": "V3  LiRA (auditor)", "v2_p": "V2  backdoor canary"}
METHODS = ["do_nothing", "finetune_retain", "gradient_ascent", "scrub", "federaser_lite",
           "retrain_oracle", "retrain_oracle_alt"]
ALPHA = 0.05


def z_threshold(n_null: int = 10) -> float:
    """Smallest V3 z with p < ALPHA under the Student-t prediction test (lira.client_test)."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if student_t_sf(mid / math.sqrt(1 + 1 / n_null), n_null - 1) < ALPHA:
            hi = mid
        else:
            lo = mid
    return hi


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(w, h, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    return fig, axes


def seed_legend(fig, seeds, y=0.0):
    handles = [plt.Line2D([], [], marker=SEED_MARKER[s], color=SEED_COLOR[s], linestyle="",
                          markersize=7, label=f"seed {s}") for s in seeds]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, y))


def fig_headline(runs, out):
    """V1 vs V3 for every method and seed: the common check is blind, LiRA is not."""
    fig, (a1, a2) = figure(10, 4.2, 2)
    zt = z_threshold()
    for i, run in enumerate(runs):
        by = {r["method"]: r for r in run["rows"]}
        for j, m in enumerate(METHODS):
            y = len(METHODS) - 1 - j + (i - (len(runs) - 1) / 2) * 0.12
            kw = dict(color=SEED_COLOR[run["seed"]], marker=SEED_MARKER[run["seed"]], s=42,
                      edgecolors=SURFACE, linewidths=1.2, zorder=3)
            a1.scatter(num(by[m], "v1_matched_auc"), y, **kw)
            a2.scatter(num(by[m], "v3_z"), y, **kw)
    ticks = list(range(len(METHODS)))[::-1]
    for ax in (a1, a2):
        style(ax)
        ax.set_yticks(ticks, METHODS, color=INK, fontsize=9)
        ax.set_ylim(-0.6, len(METHODS) - 0.4)
    a1.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    a1.set_xlim(0.45, 0.55)
    a1.set_title("V1  loss-threshold MIA AUC\n(the usual check)", color=INK, fontsize=10, loc="left")
    a1.set_xlabel("AUC (0.5 = looks forgotten)", color=MUTED, fontsize=9)
    a2.axvspan(zt, 7, color="#fbe3d6", zorder=0)
    a2.axvline(zt, color="#eb6834", linewidth=1)
    a2.text(zt + 0.2, -0.45, "still detected\n(p < 0.05)", color=INK, fontsize=8, va="bottom")
    a2.set_xlim(-9, 7)
    a2.set_yticklabels([])
    a2.set_title("V3  client-level LiRA\n(against 10 retrains without the client)", color=INK,
                 fontsize=10, loc="left")
    a2.set_xlabel("z  (distance from retrains)", color=MUTED, fontsize=9)
    seed_legend(fig, [r["seed"] for r in runs])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_ascent(runs, out):
    """Each verifier's p-value along gradient ascent, one panel per seed."""
    fig, axes = figure(10, 3.4, len(runs))
    for ax, run in zip(axes, runs):
        traj = sorted((r for r in run["rows"] if num(r, "step") is not None and num(r, "step") >= 0),
                      key=lambda r: num(r, "step"))
        steps = [num(r, "step") for r in traj]
        for key in ("v5_p", "v3_p", "v2_p"):
            ys = [min(-math.log10(max(num(r, key), 1e-300)), 12) for r in traj]
            ax.plot(steps, ys, color=CHECK_COLOR[key], linewidth=2, label=CHECK_NAME[key])
        ax.axhline(-math.log10(ALPHA), color=MUTED, linewidth=1, linestyle="--")
        style(ax)
        ax.set_ylim(0, 12.5)
        ax.set_xlim(0, 64)
        ax.set_title(f"seed {run['seed']}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("gradient-ascent step", color=MUTED, fontsize=9)
    axes[0].set_ylabel("evidence the client is still in\n−log10 p  (capped at 12)", color=MUTED, fontsize=9)
    axes[0].text(63, -math.log10(ALPHA) + 0.3, "p = 0.05", color=MUTED, fontsize=8, ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_size(runs, out):
    """Median V3 z for do_nothing against the number of the client's samples audited."""
    fig, ax = figure(6.2, 3.6)
    zt = z_threshold()
    for run in runs:
        r = next(r for r in run["rows"] if r["method"] == "do_nothing")
        sizes = sorted(int(k.rsplit("_", 1)[1]) for k in r if k.startswith("v3sz_zmed_"))
        ax.plot(sizes, [num(r, f"v3sz_zmed_{m}") for m in sizes], color=SEED_COLOR[run["seed"]],
                marker=SEED_MARKER[run["seed"]], markersize=7, linewidth=2,
                markeredgecolor=SURFACE, label=f"seed {run['seed']} (client has {int(num(r, 'v3sz_n_client'))})")
    ax.axhline(zt, color="#eb6834", linewidth=1)
    ax.text(1500, zt + 0.12, "detected above this line", color=INK, fontsize=8, ha="right")
    style(ax)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 250, 500, 1000, 1500], ["50", "100", "250", "500", "1000", "1500"])
    ax.set_ylim(0, 6)
    ax.set_xlabel("samples of the forgotten client audited", color=MUTED, fontsize=9)
    ax.set_ylabel("V3 z for do_nothing (median of 200)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_controlled_size(runs, out):
    """V3 z for do_nothing when the same forgotten client is cut to fewer training
    samples (data.forget_size), one line per seed that has cuts; the uncut run is
    the rightmost point. Only the client's size changes between points."""
    fig, ax = figure(6.2, 3.8)
    zt = z_threshold()
    by_seed = {}
    for run in runs:
        dn = next(x for x in run["rows"] if x["method"] == "do_nothing")
        by_seed.setdefault(run["seed"], []).append((run["forget_size"], num(dn, "v3_z")))
    full = {42: 3360, 43: 2511, 44: 1855, 45: 1670, 46: 4593, 47: 3667}  # client 0, uncut
    xs, zs = [], []
    for seed, pts in sorted(by_seed.items()):
        if all(s is None for s, _ in pts):
            continue  # no cuts on this seed
        pts = sorted((s if s is not None else full[seed], z) for s, z in pts)
        xs += [p[0] for p in pts]; zs += [p[1] for p in pts]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SEED_COLOR[seed],
                marker=SEED_MARKER[seed], markersize=7, markeredgecolor=SURFACE,
                linewidth=2, label=f"seed {seed} (uncut: {full[seed]})")
    x_hi = max(xs) + 300
    ax.axhline(zt, color="#eb6834", linewidth=1)
    ax.text(x_hi - 20, zt + 0.12, "caught above this line (p < 0.05)", ha="right",
            color=INK, fontsize=8)
    style(ax)
    ax.set_xlim(0, x_hi)
    ax.set_ylim(min(-0.5, min(zs) - 0.3), max(zs) + 0.5)
    ax.set_xlabel("training samples the forgotten client keeps (same client, cut)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("V3 z for do_nothing", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def fig_client_size(e0_runs, size_runs, out):
    """V3 z for do_nothing against the forgotten client's size, every E0 run;
    runs that share a federation (same seed, different client) are joined."""
    fig, ax = figure(6.2, 3.8)
    zt = z_threshold()
    n_client = {r["run_id"]: num(next(x for x in r["rows"] if x["method"] == "do_nothing"), "v3sz_n_client")
                for r in size_runs}
    by_seed = {}
    for run in e0_runs:
        if run["run_id"] not in n_client:
            continue
        z = num(next(x for x in run["rows"] if x["method"] == "do_nothing"), "v3_z")
        by_seed.setdefault(run["seed"], []).append((n_client[run["run_id"]], z, run["forget"]))
    for seed, pts in sorted(by_seed.items()):
        pts.sort()
        kw = dict(color=SEED_COLOR[seed], marker=SEED_MARKER[seed])
        if len(pts) > 1:  # a client swap within one federation
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linewidth=2, zorder=2, **kw,
                    markersize=0)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=60, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3, label=f"seed {seed}", **kw)
        for n, z, f in pts:
            ax.annotate(f"client {f}", (n, z), textcoords="offset points", xytext=(6, -11),
                        fontsize=7, color=MUTED)
    # axes follow the data, so added seeds or client swaps are never clipped
    xs = [p[0] for pts in by_seed.values() for p in pts]
    zs = [p[1] for pts in by_seed.values() for p in pts]
    x_hi = max(xs) + 300
    ax.axhline(zt, color="#eb6834", linewidth=1)
    ax.text(x_hi - 20, zt + 0.12, "caught above this line (p < 0.05)", ha="right",
            color=INK, fontsize=8)
    style(ax)
    ax.set_xlim(min(xs) - 100, x_hi)
    ax.set_ylim(min(-0.5, min(zs) - 0.3), max(zs) + 0.5)
    ax.set_xlabel("training samples of the forgotten client", color=MUTED, fontsize=9)
    ax.set_ylabel("V3 z for do_nothing", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Draw the figures from the verified results.")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"), help="output folder")
    ap.add_argument("--format", default="svg", choices=["svg", "pdf"],
                    help="svg for the docs, pdf for the paper (paper/figures)")
    args = ap.parse_args(argv)
    results, out = ROOT / "results", Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for e in read_ledger(results / "ledger.jsonl"):
        meta = index.setdefault(e["run_id"], dict(e))
        if "forget_client" in e:
            meta.setdefault("forget_client", e["forget_client"])
    # these figures are CIFAR-10 (seeds are colours, and a second dataset reuses the seeds);
    # the main ones use the default forget client, client-swap runs are compared separately
    # (and the default 30 rounds: the longer-trained runs are compared separately)
    cifar = lambda exp: [r for r in load(results, exp, index)
                         if r["dataset"] == MAIN_DATASET and r["rounds"] == MAIN_ROUNDS
                         and r["model"] == MAIN_MODEL]
    main_runs = lambda exp: [r for r in cifar(exp) if r["forget"] == 0 and r["forget_size"] is None]
    fig_headline(main_runs("e0_sanity"), out / f"verifiers.{args.format}")
    fig_ascent(main_runs("e2_canary"), out / f"gradient_ascent.{args.format}")
    fig_size(main_runs("e0_sanity_v3size"), out / f"audit_size.{args.format}")
    natural = lambda exp: [r for r in cifar(exp) if r["forget_size"] is None]
    fig_client_size(natural("e0_sanity"), natural("e0_sanity_v3size"), out / f"client_size.{args.format}")
    fig_controlled_size([r for r in cifar("e0_sanity") if r["forget"] == 0], out / f"controlled_size.{args.format}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""M36 figures (T13.60, for T36.8): CKA of the representations, Grad-CAM across the checkpoints.

Sources: `build/explain/cka.json` and `build/explain/cka_gradcam.npz`, from
`tools/explain.py cka --run <recorder run>`. Reduced runs (64 probe images): `subset=True`,
so the PNGs go to `build/figures/resultats/`.
"""

import json

import numpy as np

from tools.figures import ROOT, MissingSource, figure, need
from tools.figures import style as st

CKA = ROOT / "build" / "explain" / "cka.json"
STRIP = ROOT / "build" / "explain" / "cka_gradcam.npz"


def load_cka(path=CKA):
    return json.loads(need(path).read_text())


@figure("explain_cka_float_int", "resultats", "T13.60",
        "Linear CKA between the float and the integer outputs of every conv (64 VOC test "
        "images, maps pooled 2 × 2), per network.", "build/explain/cka.json", subset=True)
def fig_cka_float_int(out_dir):
    d = load_cka()
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL * 0.6, 3.6))
    for k, (net, per) in enumerate(d["float_int"].items()):
        ids = sorted(per, key=int)
        ax.plot([int(i) for i in ids], [per[i] for i in ids], "o-", color=st.PALETTE[k],
                label=net, ms=4)
    ax.axhline(0.99, color=st.INK2, lw=0.8, ls="--")
    ax.set_xlabel("conv layer")
    ax.set_ylabel("CKA float / integer")
    ax.set_title(f"Float against integer representations ({d['probe']} images)")
    ax.legend(loc="lower left")
    return st.save(fig, out_dir, "explain_cka_float_int")


@figure("explain_cka_checkpoints", "resultats", "T13.60",
        "Linear CKA of each conv at each checkpoint to the same conv at the last checkpoint "
        "of the T36.7 recorder run: the layers that settle first are the ones that reach 1 "
        "first.", "build/explain/cka.json", subset=True)
def fig_cka_checkpoints(out_dir):
    d = load_cka()
    if "checkpoints" not in d:
        raise MissingSource("build/explain/cka.json: no checkpoints (cka --run)")
    c = d["checkpoints"]
    m = np.array(c["cka_to_last"]).T  # (conv, checkpoint)
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL * 0.6, 3.6))
    im = ax.imshow(m, aspect="auto", cmap="viridis", vmin=min(0.8, float(m.min())), vmax=1)
    ax.set_xticks(range(len(c["its"])), c["its"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("conv (order in the network)")
    ax.set_title("CKA to the last checkpoint")
    fig.colorbar(im, ax=ax)
    return st.save(fig, out_dir, "explain_cka_checkpoints")


@figure("explain_gradcam_strip", "resultats", "T13.60",
        "Grad-CAM of the first ground-truth box of 4 probe images (fixed cell and anchor) "
        "across the checkpoints of the T36.7 recorder run.", "build/explain/cka_gradcam.npz",
        subset=True)
def fig_gradcam_strip(out_dir):
    with np.load(need(STRIP)) as d:
        maps, its = d["maps"], d["its"]
    plt = st.plt()
    n_ck, n_img = maps.shape[:2]
    fig, axes = plt.subplots(n_img, n_ck, figsize=(1.6 * n_ck, 1.6 * n_img), squeeze=False)
    for i in range(n_img):
        for k in range(n_ck):
            ax = axes[i][k]
            ax.imshow(maps[k, i], cmap="jet")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"it {its[k]}", fontsize=8)
    fig.suptitle("Grad-CAM across the checkpoints")
    return st.save(fig, out_dir, "explain_gradcam_strip")

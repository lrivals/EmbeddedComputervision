"""Planches de la réimplémentation de zéro (M13, section F) : une brique, sa formule, son code.

Chaque planche suit la règle « fiche mathématique » de docs/tasks/M13-figures.md : les
courbes sont calculées **en appelant le code du dépôt** (`python/yolo/…`) sur des entrées
jouets à graine fixe, et un cartouche donne le chemin `module:fonction`, la bibliothèque
évitée et le test qui vérifie l'implémentation. Aucune donnée externe : toutes ces planches
sont du palier R et se génèrent sans VOC ni poids.
"""

import pkgutil
import time

import numpy as np

from tools.figures import ROOT, figure
from tools.figures import style as st

TESTS = ROOT / "python" / "tests"
SEED = 0


def fiche(fig, code, evite, test):
    """Cartouche en bas de la planche : code tracé, bibliothèque évitée, test."""
    missing = [t for t in test.split(", ") if not (TESTS / t.split("::")[0]).exists()]
    if missing:
        raise AssertionError(f"tests cités absents : {missing}")
    fig.text(0.01, 0.005, f"Code : {code}    Évite : {evite}    Test : {test}", fontsize=7,
             color=st.INK2, ha="left", va="bottom",
             bbox=dict(fc=st.GRID, ec="none", pad=3, alpha=0.6))


def eq(ax, text, y=0.97, x=0.02, size=8.5, **kw):
    """Formule (mathtext) dans un coin du panneau."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, va="top", ha="left",
            bbox=dict(fc="white", ec=st.GRID, pad=2.5, alpha=0.92), **kw)


def finish(fig, out_dir, name, title):
    fig.suptitle(title, fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return st.save(fig, out_dir, name)


# --- T13.31 : carte de la réimplémentation -------------------------------------------------

MODULES = {  # module → (brique, bibliothèque évitée, test, planche)
    "backend": ("NumPy / CuPy au choix", "torch.device", "test_backend.py", ""),
    "data/anchors": ("k-means des ancres", "sklearn.cluster.KMeans", "test_anchors.py", "T13.40"),
    "data/augment": ("augmentations", "albumentations, cv2", "test_data_pipeline.py", "T13.42"),
    "data/datasets": ("jeux de données", "torchvision.datasets", "test_datasets.py", ""),
    "data/letterbox": ("letterbox", "cv2.resize", "test_data_pipeline.py", "T13.42"),
    "data/loader": ("chargement par lots", "torch.utils.data", "test_data_pipeline.py", ""),
    "data/targets": ("cibles YOLO", "ultralytics", "test_targets.py", "T13.38"),
    "data/voc": ("annotations VOC", "devkit VOC", "test_voc.py", ""),
    "infer/boxes": ("boîtes et IoU", "torchvision.ops.box_iou", "test_boxes.py", "T13.37"),
    "infer/coco_eval": ("mAP COCO", "pycocotools", "test_coco_eval.py", ""),
    "infer/decode": ("décodage des têtes", "ultralytics", "test_decode.py", "T13.37"),
    "infer/hw_postproc": ("post-traitement matériel", "—", "test_hw_postproc.py", ""),
    "infer/metrics": ("mAP VOC", "devkit VOC MATLAB", "test_metrics.py", "T13.41"),
    "infer/nms": ("NMS", "torchvision.ops.nms", "test_nms.py", "T13.41"),
    "infer/pipeline": ("prétraitement", "cv2", "test_pipeline.py", "T13.42"),
    "infer/tiles": ("inférence par tuiles", "SAHI", "test_tiles.py", ""),
    "io/darknet_weights": ("format .weights", "darknet", "test_darknet_weights.py", "T13.47"),
    "io/export": ("export de l'arène", "onnx, Vitis AI", "test_export.py", "T13.47"),
    "layers/activations": ("activations", "torch.nn.LeakyReLU", "test_activations.py", "T13.34"),
    "layers/batchnorm": ("batch norm", "torch.nn.BatchNorm2d", "test_batchnorm.py", "T13.35"),
    "layers/conv": ("convolution", "torch.nn.Conv2d", "test_conv.py", "T13.32"),
    "layers/pool": ("maxpool", "torch.nn.MaxPool2d", "test_pool.py", "T13.36"),
    "layers/route": ("route", "torch.cat", "test_upsample_route.py", "T13.36"),
    "layers/upsample": ("upsample", "torch.nn.Upsample", "test_upsample_route.py", "T13.36"),
    "models/cfg": ("parser .cfg", "darknet", "test_cfg.py", "T13.47"),
    "models/graph": ("graphe et rétropropagation", "torch.autograd", "test_graph.py", "T13.33"),
    "models/specs": ("formes et coûts", "torchinfo", "test_tiny_yolo.py", ""),
    "models/tiny_yolo": ("construction", "torch.nn.Module", "test_tiny_yolo.py", ""),
    "prune": ("élagage", "torch.nn.utils.prune", "test_prune.py", ""),
    "quant/calibrate": ("calibration", "Vitis AI, TensorRT", "test_quantize.py", "T13.43"),
    "quant/fake_quant": ("fake-quant, QAT", "torch.ao.quantization", "test_fake_quant.py",
                         "T13.46"),
    "quant/fuse_bn": ("fusion BN", "torch.ao fuse_modules", "test_fuse_bn.py", "T13.35"),
    "quant/int_layers": ("arithmétique entière", "Vitis AI", "test_int_layers.py", "T13.44"),
    "quant/int_model": ("réseau entier", "Vitis AI", "test_int_model.py", "T13.44"),
    "quant/lowbit": ("4 bits", "Brevitas", "test_fake_quant.py", "T13.46"),
    "quant/lut": ("tables σ et exp", "—", "test_lut.py", "T13.45"),
    "quant/pow2": ("puissances de 2", "Brevitas", "test_pow2.py", "T13.46"),
    "quant/quantize": ("quantification", "torch.quantization", "test_quantize.py", "T13.43"),
    "testing/gradcheck": ("gradcheck", "torch.autograd.gradcheck", "test_gradcheck.py", "T13.33"),
    "train/admm": ("ADMM", "—", "test_admm.py", "T13.46"),
    "train/loss": ("perte YOLO", "torch.nn.BCEWithLogitsLoss", "test_loss.py", "T13.38"),
    "train/optim": ("SGD", "torch.optim.SGD", "test_train.py", "T13.39"),
    "train/schedule": ("taux d'apprentissage", "torch.optim.lr_scheduler", "test_train.py",
                       "T13.39"),
    "train/trainer": ("boucle d'entraînement", "pytorch-lightning", "test_train.py", "T13.39"),
}


def package_modules():
    """Modules de `python/yolo/` (hors __init__), parcourus dans le paquet."""
    import yolo

    out = []
    for m in pkgutil.walk_packages(yolo.__path__, "yolo."):
        if not m.ispkg:
            out.append(m.name.removeprefix("yolo.").replace(".", "/"))
    return sorted(out)


def load_map():
    """[(module, brique, évitée, test, test présent, planche)] ; un module absent de
    `MODULES` apparaît avec « ? » (la carte n'en oublie aucun)."""
    rows = []
    for mod in package_modules():
        brick, avoid, test, task = MODULES.get(mod, ("?", "?", "?", ""))
        rows.append((mod, brick, avoid, test, (TESTS / test).exists(), task))
    return rows


@figure("carte", "maths", "T13.31",
        "Carte de la réimplémentation : chaque module de python/yolo/, la brique qu'il "
        "réécrit, la bibliothèque évitée, son test et la planche qui l'explique.",
        "python/yolo/ (parcours du paquet), python/tests/")
def carte(out_dir):
    rows = load_map()
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL, 0.24 * len(rows) + 1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(len(rows) + 0.5, -1)
    ax.axis("off")
    xs = [0, 2.3, 4.6, 6.8, 9.1]
    for x, h in zip(xs, ["module python/yolo/", "brique", "bibliothèque évitée", "test",
                         "planche"]):
        ax.text(x, -0.4, h, fontsize=8, weight="bold")
    family = {m.split("/")[0] for m, *_ in rows}
    fam_color = {f: st.PALETTE[k % 8] for k, f in enumerate(sorted(family))}
    for y, (mod, brick, avoid, test, ok, task) in enumerate(rows, start=1):
        if y % 2:
            ax.axhspan(y - 0.5, y + 0.5, color=st.GRID, alpha=0.5, lw=0)
        ax.text(xs[0], y, mod, fontsize=7, va="center", family="monospace",
                color=fam_color[mod.split("/")[0]])
        ax.text(xs[1], y, brick, fontsize=7, va="center")
        ax.text(xs[2], y, avoid, fontsize=7, va="center", color=st.INK2)
        ax.text(xs[3], y, test if ok else f"{test} (absent)", fontsize=7, va="center",
                color=st.INK if ok else st.PALETTE[7])
        ax.text(xs[4], y, task, fontsize=7, va="center", weight="bold", color=st.PALETTE[0])
    ax.set_title(f"Réimplémentation en NumPy pur : {len(rows)} modules, aucune dépendance "
                 "d'apprentissage ni de vision (ADR 0001)")
    fig.tight_layout()
    return st.save(fig, out_dir, "carte")


# --- T13.32 : convolution -----------------------------------------------------------------

def conv_timings(sizes=(8, 16, 26, 52), c=4, f=8, reps=2):
    """[(taille, s naïve, s im2col, écart max)] sur des cartes jouets en float64."""
    from yolo.layers.conv import conv_forward, conv_forward_naive

    rng = np.random.default_rng(SEED)
    out = []
    for n in sizes:
        x = rng.standard_normal((1, c, n, n))
        W = rng.standard_normal((f, c, 3, 3))
        b = rng.standard_normal(f)
        t0 = time.perf_counter()
        y0, _ = conv_forward_naive(x, W, b)
        t1 = time.perf_counter()
        for _ in range(reps):
            y1, _ = conv_forward(x, W, b)
        t2 = time.perf_counter()
        out.append((n, t1 - t0, (t2 - t1) / reps, float(np.abs(y0 - y1).max())))
    return out


@figure("convolution", "maths", "T13.32",
        "Convolution : la formule sur un patch, le dépliage im2col en produit de matrices, "
        "et le temps des boucles face à im2col.",
        "yolo.layers.conv (conv_forward_naive, conv_forward, conv_backward)")
def convolution(out_dir):
    from matplotlib.patches import Rectangle

    from yolo.layers.conv import conv_backward, conv_forward

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 4.0))
    rng = np.random.default_rng(SEED)
    # 1. patch 3×3 sur une carte 6×6
    ax = axes[0]
    x = rng.integers(0, 9, (1, 1, 6, 6)).astype(np.float64)
    W = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=np.float64)
    y, _ = conv_forward(x, W, pad=0)
    ax.imshow(x[0, 0], cmap="Greys", alpha=0.35, extent=(0, 6, 6, 0))
    for i in range(6):
        for j in range(6):
            ax.text(j + 0.5, i + 0.5, f"{x[0, 0, i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((1, 2), 3, 3, fill=False, ec=st.PALETTE[1], lw=2))
    ax.set_xlim(0, 6)
    ax.set_ylim(6, 0)
    st.image_axes(ax, f"patch (2, 1) → y = {y[0, 0, 2, 1]:.0f} (Sobel, pas s = 1, p = 0)")
    eq(ax, r"$y_{f,i,j}=b_f+\sum_c\sum_u\sum_v W_{f,c,u,v}\,x_{c,\,si+u-p,\,sj+v-p}$",
       y=-0.04, size=7.5)
    # 2. im2col : formes de L00
    ax = axes[1]
    st.schema_axes(ax, (0, 10), (0, 8))
    st.block(ax, 0.2, 4.6, 2.6, 2.6, "W\n(F, C·k²)\n(16, 27)", st.PALETTE[0])
    ax.text(3.2, 5.9, "·", fontsize=22, ha="center", va="center")
    st.block(ax, 3.6, 3.8, 2.0, 4.0, "X_col\n(C·k², Ho·Wo)\n(27, 173 056)", st.PALETTE[2])
    ax.text(6.0, 5.9, "=", fontsize=16, ha="center", va="center")
    st.block(ax, 6.4, 4.6, 3.4, 2.6, "Y\n(F, Ho·Wo)\n(16, 416·416)", st.PALETTE[1])
    ax.text(5, 2.8, "sliding_window_view : chaque patch k×k×C\ndevient une colonne, sans copie\n"
            "(L00 de Tiny-YOLOv2 : C = 3, k = 3, F = 16)", ha="center", fontsize=7.5)
    ax.text(5, 0.6, r"$dW=\delta Y\cdot X_{col}^T,\quad dX=\mathrm{col2im}(W^T\cdot\delta Y)$",
            ha="center", fontsize=8.5)
    ax.set_title("im2col : la convolution devient un produit de matrices", fontsize=9)
    # passe arrière : vérifie la forme de dW (appel du code, pas de réécriture)
    _, cache = conv_forward(x, W, pad=0)
    _, g = conv_backward(np.ones_like(y), cache)
    assert g["W"].shape == W.shape
    # 3. temps
    ax = axes[2]
    rows = conv_timings()
    n = [r[0] for r in rows]
    ax.plot(n, [r[1] * 1e3 for r in rows], "o-", color=st.PALETTE[7], label="conv_forward_naive")
    ax.plot(n, [r[2] * 1e3 for r in rows], "s-", color=st.PALETTE[0], label="conv_forward (im2col)")
    ax.set_yscale("log")
    ax.set_xlabel("taille de la carte n × n (C = 4, F = 8)")
    ax.set_ylabel("temps (ms, cette machine)")
    ax.legend(loc="upper left")
    ax.set_title(f"même sortie : écart max {max(r[3] for r in rows):.1e} (float64)", fontsize=9)
    fiche(fig, "yolo.layers.conv:conv_forward, conv_forward_naive, conv_backward",
          "torch.nn.Conv2d", "test_conv.py")
    return finish(fig, out_dir, "convolution", "Convolution : boucles et im2col (§4.1)")


# --- T13.33 : rétropropagation ------------------------------------------------------------

def gradcheck_errors():
    """{couche: erreur relative max} de `gradcheck.check_layer` (float64), et le témoin
    « gradient faux » de test_gradcheck.py::test_detects_wrong_gradient."""
    from yolo.layers.activations import leaky_backward, leaky_forward
    from yolo.layers.batchnorm import bn_backward, bn_forward, bn_init_state
    from yolo.layers.conv import conv_backward, conv_forward
    from yolo.layers.pool import maxpool_backward, maxpool_forward
    from yolo.layers.upsample import upsample_backward, upsample_forward
    from yolo.testing.gradcheck import check_grad, check_layer

    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((2, 3, 6, 6))
    out = {}
    out["conv"] = max(check_layer(lambda x, p: conv_forward(x, p["W"], p["b"]), conv_backward, x,
                                  {"W": rng.standard_normal((4, 3, 3, 3)),
                                   "b": rng.standard_normal(4)}).values())
    rb = np.random.default_rng(10)  # données de test_batchnorm.py::_data
    xb = rb.standard_normal((2, 3, 4, 5)) * 2 + 1
    out["batchnorm"] = max(check_layer(
        lambda x, p: bn_forward(x, p["gamma"], p["beta"], bn_init_state(3)), bn_backward, xb,
        {"gamma": rb.standard_normal(3), "beta": rb.standard_normal(3)}).values())
    xl = x + np.sign(x) * 0.05  # loin du coude de la leaky
    out["leaky"] = max(check_layer(lambda x, p: leaky_forward(x), leaky_backward, xl, {}).values())
    xp = x + rng.permutation(x.size).reshape(x.shape) * 1e-3  # maxima uniques
    out["maxpool 2/2"] = max(check_layer(lambda x, p: maxpool_forward(x), maxpool_backward, xp, {}).values())
    out["maxpool 2/1"] = max(check_layer(lambda x, p: maxpool_forward(x, 2, 1), maxpool_backward, xp, {}).values())
    out["upsample"] = max(check_layer(lambda x, p: upsample_forward(x), upsample_backward, x, {}).values())
    w = np.random.default_rng(3).standard_normal(10)
    out["témoin faux (x³ ↦ 2x²)"] = check_grad(lambda: w**3, w, 2 * w**2)
    return out


@figure("retropropagation", "maths", "T13.33",
        "Rétropropagation écrite à la main : formule du gradient renvoyé par chaque couche, "
        "puis l'erreur du gradcheck par couche face au seuil 1e-7.",
        "yolo.layers.*, yolo.models.graph, yolo.testing.gradcheck")
def retropropagation(out_dir):
    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.FULL, 4.6), gridspec_kw={"width_ratios":
                                                                              [1.25, 1]})
    st.schema_axes(ax, (0, 12), (0, 9))
    layers = [("conv", r"$\delta X=\mathrm{col2im}(W^T\delta Y)$" + "\n" +
               r"$\delta W=\delta Y\,X_{col}^T$"),
              ("BN", r"$\delta\hat x=\gamma\,\delta y$" + "\n" +
               r"$\delta x=\frac{1}{\sigma}(\delta\hat x-\overline{\delta\hat x}-\hat x\,"
               r"\overline{\delta\hat x\,\hat x})$"),
              ("leaky", r"$\delta x=\delta y\cdot(1\ \mathrm{si}\ x>0,\ 0{,}1\ \mathrm{sinon})$"),
              ("maxpool", r"$\delta x=\delta y$ à l'argmax, 0 ailleurs")]
    for k, (name, formula) in enumerate(layers):
        y = 7.6 - 1.75 * k
        st.block(ax, 0.2, y - 0.5, 2.0, 1.0, name, st.PALETTE[k], fontsize=9, weight="bold")
        ax.text(2.6, y, formula, fontsize=8, va="center")
        if k:
            st.arrow(ax, (1.2, y + 1.25), (1.2, y + 0.5))
    st.block(ax, 0.2, 0.1, 5.4, 1.0, "route : la carte lue deux fois reçoit\nδ = δ₁ + δ₂ "
             "(Network.backward additionne)", st.PALETTE[3], fontsize=7.5)
    st.block(ax, 6.2, 0.1, 5.6, 1.0, "upsample ×2 : δx = somme des 4 δy\nde chaque bloc 2×2",
             st.PALETTE[4], fontsize=7.5)
    ax.set_title("passe avant ↓, gradient renvoyé (formule) →", fontsize=9)
    errs = gradcheck_errors()
    bad = {n: v for n, v in errs.items() if "faux" not in n and v > 1e-7}
    assert not bad and errs["témoin faux (x³ ↦ 2x²)"] > 1e-5, errs
    names = list(errs)
    vals = np.array([max(errs[n], 1e-16) for n in names])
    colors = [st.PALETTE[7] if "faux" in n else st.PALETTE[5] for n in names]
    ax2.barh(range(len(names)), vals, color=colors)
    ax2.axvline(1e-7, color=st.INK2, ls="--", lw=1)
    ax2.text(1.3e-7, len(names) - 0.4, "seuil 1e-7", fontsize=7, color=st.INK2)
    ax2.set_xscale("log")
    ax2.set_yticks(range(len(names)), names)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"$\|g_a-g_n\|_\infty/\max(\|g_a\|,\|g_n\|)$, $g_n=\frac{f(x+h)-f(x-h)}{2h}$")
    ax2.set_title("gradcheck en float64 (check_layer)", fontsize=9)
    fiche(fig, "yolo.models.graph:Network.backward, yolo.testing.gradcheck:check_layer",
          "torch.autograd", "test_gradcheck.py, test_graph.py")
    return finish(fig, out_dir, "retropropagation",
                  "Rétropropagation écrite à la main (§4, §11)")


# --- T13.34 : activations -----------------------------------------------------------------

@figure("activations", "maths", "T13.34",
        "Leaky ReLU et sa dérivée, sigmoïde stable face à la forme naïve, BCE sur logits et "
        "son gradient, softmax stable.",
        "yolo.layers.activations, yolo.train.loss (softplus, bce_logits)")
def activations(out_dir):
    from yolo.layers.activations import leaky_backward, leaky_forward, sigmoid
    from yolo.train.loss import bce_logits, softplus

    plt = st.plt()
    fig, axes = plt.subplots(1, 4, figsize=(st.FULL, 3.4))
    t = np.linspace(-6, 6, 601)
    y, cache = leaky_forward(t)
    ax = axes[0]
    ax.plot(t, y, color=st.PALETTE[0], label="f(x)")
    ax.plot(t, leaky_backward(np.ones_like(t), cache)[0], color=st.PALETTE[1], ls="--", label="f'(x)")
    eq(ax, r"$f(x)=\max(x,\,0{,}1x)$")
    ax.legend(loc="lower right")
    ax.set_title("leaky ReLU", fontsize=9)
    ax = axes[1]
    tt = np.linspace(-100, 100, 2001).astype(np.float32)
    with np.errstate(over="ignore", divide="ignore"):
        naive = (1 / (1 + np.exp(-tt))).astype(np.float32)
        naive_log = np.log(naive)
    stable = sigmoid(tt)
    assert np.isfinite(stable).all()
    ax.plot(tt, np.log(np.maximum(stable, 1e-300)), color=st.PALETTE[0], label="ln σ, sigmoid (2 branches)")
    ax.plot(tt, np.where(np.isfinite(naive_log), naive_log, np.nan), color=st.PALETTE[7], ls=":",
            lw=2.2, label="ln σ naïve (float32)")
    ax.set_xlabel("t")
    eq(ax, r"$\sigma(t)=\frac{1}{1+e^{-t}}$ si $t\geq0$, $\frac{e^t}{1+e^t}$ sinon", size=7.5)
    ax.legend(loc="lower right", fontsize=6.5)
    st.note(ax, "aucune inf/nan sur [−100, 100]", "lower left")
    ax.set_title("sigmoïde stable", fontsize=9)
    ax = axes[2]
    for yv, c in ((1, st.PALETTE[0]), (0, st.PALETTE[1])):
        ax.plot(t, bce_logits(t, yv), color=c, label=f"BCE(t, y={yv})")
        ax.plot(t, sigmoid(t) - yv, color=c, ls="--", lw=1.1, label=f"∂/∂t = σ(t) − {yv}")
    ax.plot(t, softplus(t), color=st.MUTED, lw=0.8, ls=":")
    eq(ax, r"$\mathrm{BCE}(t,y)=\mathrm{softplus}(t)-yt$", size=7.5)
    ax.legend(loc="center right", fontsize=6.3)
    ax.set_title("BCE sur logits", fontsize=9)
    ax = axes[3]
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(20) * 3 + 500  # logits énormes : exp déborderait sans le max
    from yolo.infer.decode import decode_head

    out = np.zeros((1, 25, 1, 1))
    out[0, 5:, 0, 0] = z
    _, _, scores = decode_head(out, [[0.1, 0.1]], 20, "v2")
    p = scores[0, 0] / 0.5  # σ(0) = 0,5 d'objectness
    ax.bar(range(20), p, color=st.PALETTE[6])
    eq(ax, r"$\mathrm{softmax}(t)_c=\frac{e^{t_c-\max t}}{\sum_k e^{t_k-\max t}}$", size=7.5)
    ax.set_xlabel("classe (logits ≈ 500 : e^t déborde sans le max)")
    ax.set_title(f"softmax stable (Σ = {p.sum():.6f})", fontsize=9)
    fiche(fig, "yolo.layers.activations:leaky_forward, sigmoid ; yolo.train.loss:softplus, "
          "bce_logits ; yolo.infer.decode:decode_head (v2)", "torch.sigmoid, torch.nn.BCEWithLogitsLoss",
          "test_activations.py, test_loss.py")
    return finish(fig, out_dir, "activations", "Activations, sigmoïde stable, BCE et softmax "
                  "(§4.3, §4.5, §6.2)")


# --- T13.35 : batch normalization ---------------------------------------------------------

@figure("batchnorm", "maths", "T13.35",
        "Batch normalization : normalisation par canal, moyennes glissantes, puis fusion dans "
        "la convolution (sortie identique).",
        "yolo.layers.batchnorm.bn_forward, yolo.quant.fuse_bn.fuse_bn")
def batchnorm(out_dir):
    from yolo.layers.batchnorm import bn_forward, bn_init_state
    from yolo.layers.conv import conv_forward
    from yolo.quant.fuse_bn import fuse_bn

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.7))
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((8, 3, 10, 10)) * np.array([3, 0.5, 1.5])[None, :, None, None] + \
        np.array([4, -2, 0.5])[None, :, None, None]
    state = bn_init_state(3)
    gamma, beta = np.ones(3), np.zeros(3)
    y, _ = bn_forward(x, gamma, beta, state)
    ax = axes[0]
    for c in range(3):
        ax.hist(x[:, c].ravel(), bins=40, histtype="step", color=st.PALETTE[c], lw=1.2,
                label=f"x canal {c}")
        ax.hist(y[:, c].ravel(), bins=40, histtype="stepfilled", color=st.PALETTE[c], alpha=0.25)
    eq(ax, r"$\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\ y=\gamma\hat x+\beta$", size=7.5)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title("par canal sur (N, H, W) : avant (trait), après (plein)", fontsize=8.5)
    ax = axes[1]
    state = bn_init_state(3)
    means = []
    for _ in range(100):
        xb = rng.standard_normal((8, 3, 10, 10)) + np.array([4, -2, 0.5])[None, :, None, None]
        bn_forward(xb, gamma, beta, state)
        means.append(state["mean"].copy())
    means = np.array(means)
    for c in range(3):
        ax.plot(means[:, c], color=st.PALETTE[c], label=f"μ_run canal {c}")
        ax.axhline([4, -2, 0.5][c], color=st.PALETTE[c], ls=":", lw=0.9)
    eq(ax, r"$\mu\leftarrow0{,}9\,\mu+0{,}1\,\mu_B$", y=0.6)
    ax.set_xlabel("itération")
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title("moyennes glissantes (pointillé : vraie moyenne)", fontsize=8.5)
    ax = axes[2]
    W = rng.standard_normal((4, 3, 3, 3))
    g, b = rng.uniform(0.5, 2, 4), rng.standard_normal(4)
    mean, var = rng.standard_normal(4), rng.uniform(0.5, 2, 4)
    xs = rng.standard_normal((2, 3, 8, 8))
    z, _ = conv_forward(xs, W)
    ref, _ = bn_forward(z, g, b, {"mean": mean, "var": var}, train=False)
    Wf, bf = fuse_bn(W, g, b, mean, var)
    fused, _ = conv_forward(xs, Wf, bf)
    ax.plot(ref.ravel(), fused.ravel(), ".", ms=2, color=st.PALETTE[0])
    lim = np.abs(ref).max()
    ax.plot([-lim, lim], [-lim, lim], color=st.MUTED, lw=0.8)
    eq(ax, r"$s_f=\gamma_f/\sqrt{\sigma_f^2+\epsilon},\ W'_f=s_fW_f,\ b'_f=\beta_f+s_f(b_f-\mu_f)$",
       size=7)
    ax.set_xlabel("conv puis BN")
    ax.set_ylabel("conv fusionnée")
    ax.set_title(f"fusion : écart max {np.abs(ref - fused).max():.1e}", fontsize=8.5)
    fiche(fig, "yolo.layers.batchnorm:bn_forward ; yolo.quant.fuse_bn:fuse_bn",
          "torch.nn.BatchNorm2d, torch.ao fuse_modules", "test_batchnorm.py, test_fuse_bn.py")
    return finish(fig, out_dir, "batchnorm", "Batch normalization et sa fusion (§4.2, §9.1)")


# --- T13.36 : maxpool, upsample, route ----------------------------------------------------

def _grid(ax, a, title, hl=None, fmt="{:.0f}", cmap="Blues"):
    ax.imshow(a, cmap=cmap, alpha=0.45, vmin=a.min() - 1, vmax=a.max() + 1)
    for (i, j), v in np.ndenumerate(a):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                weight="bold" if hl is not None and hl[i, j] else "normal")
    st.image_axes(ax, title)
    ax.title.set_fontsize(8)


@figure("pool_upsample_route", "maths", "T13.36",
        "Maxpool 2×2/2 et 2×2/1 (avec la réplication du bord), upsample ×2 et route, avec "
        "les gradients qui remontent.",
        "yolo.layers.pool, upsample, route")
def pool_upsample_route(out_dir):
    from yolo.layers.pool import maxpool_backward, maxpool_forward
    from yolo.layers.route import route_backward, route_forward
    from yolo.layers.upsample import upsample_backward, upsample_forward

    plt = st.plt()
    fig, axes = plt.subplots(2, 4, figsize=(st.FULL, 5.4))
    rng = np.random.default_rng(SEED)
    x = rng.permutation(16).reshape(1, 1, 4, 4).astype(np.float64)
    y, c = maxpool_forward(x)
    dx, _ = maxpool_backward(np.ones_like(y), c)
    _grid(axes[0, 0], x[0, 0], "x 4×4", hl=dx[0, 0] > 0)
    _grid(axes[0, 1], y[0, 0], "maxpool 2×2/2 : max par fenêtre")
    _grid(axes[1, 0], dx[0, 0], "δx : δy remonte vers l'argmax", cmap="Oranges")
    y1, c1 = maxpool_forward(x, 2, 1)
    _grid(axes[1, 1], y1[0, 0], f"maxpool 2×2/1 : {x.shape[-1]}×{x.shape[-1]} → "
          f"{y1.shape[-1]}×{y1.shape[-1]}\n(bord répliqué, couche 11)")
    u = np.arange(4, dtype=np.float64).reshape(1, 1, 2, 2) + 1
    yu, cu = upsample_forward(u)
    _grid(axes[0, 2], yu[0, 0], "upsample ×2 : plus proche voisin")
    du, _ = upsample_backward(np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4), cu)
    _grid(axes[1, 2], du[0, 0], "δx : somme des 4 δy du bloc", cmap="Oranges")
    a = np.ones((1, 2, 2, 2))
    b = 2 * np.ones((1, 1, 2, 2))
    r, cr = route_forward([a, b])
    (da, db), _ = route_backward(np.arange(r.size, dtype=np.float64).reshape(r.shape), cr)
    _grid(axes[0, 3], r[0].reshape(-1, 2), f"route : concat. des canaux\n{a.shape[1]} + "
          f"{b.shape[1]} → {r.shape[1]} canaux (empilés)")
    _grid(axes[1, 3], np.concatenate([da[0], db[0]]).reshape(-1, 2),
          "δ découpé : canaux 0-1 → a, 2 → b", cmap="Oranges")
    fiche(fig, "yolo.layers.pool:maxpool_forward/backward ; upsample ; route",
          "torch.nn.MaxPool2d, torch.nn.Upsample, torch.cat",
          "test_pool.py, test_upsample_route.py")
    return finish(fig, out_dir, "pool_upsample_route", "Maxpool, upsample et route (§4.4, §4.5)")


# --- T13.37 : boîtes, IoU, décodage -------------------------------------------------------

@figure("boites_decodage", "maths", "T13.37",
        "IoU de deux boîtes et IoU de forme des ancres, décodage du centre dans sa cellule, "
        "taille selon t_w pour les 5 ancres de Tiny-YOLOv2.",
        "yolo.infer.boxes (iou, iou_wh, cxcywh_to_xyxy), yolo.infer.decode.decode_head")
def boites_decodage(out_dir):
    from matplotlib.patches import Rectangle

    from yolo.infer.boxes import cxcywh_to_xyxy, iou, iou_wh
    from yolo.infer.decode import decode_head
    from yolo.models.tiny_yolo import load_cfg
    from yolo.data.targets import anchors_frac

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.8))
    ax = axes[0]
    A = np.array([[0.4, 0.45, 0.4, 0.5]])
    B = np.array([[0.58, 0.55, 0.4, 0.35]])
    for bx, c, lab in ((A, st.PALETTE[0], "A"), (B, st.PALETTE[1], "B")):
        x1, y1, x2, y2 = cxcywh_to_xyxy(bx)[0]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc=c, alpha=0.25, ec=c, lw=1.5))
        ax.text(x1 + 0.01, y1 + 0.01, lab, color=c, weight="bold", va="top")
    a, b = cxcywh_to_xyxy(A)[0], cxcywh_to_xyxy(B)[0]
    ix = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    ax.add_patch(Rectangle(ix[:2], ix[2] - ix[0], ix[3] - ix[1], fc="none", ec=st.INK, hatch="///"))
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.85, 0.15)
    ax.set_aspect("equal")
    st.image_axes(ax, f"IoU(A, B) = {iou(A, B)[0, 0]:.3f} ; iou_wh = {iou_wh(A[:, 2:], B[:, 2:])[0, 0]:.3f}")
    eq(ax, r"$\mathrm{IoU}=\frac{|A\cap B|}{|A\cup B|}$", y=0.12)
    ax = axes[1]
    S = 13
    t = np.linspace(-8, 8, 9)
    for k, tx in enumerate(t):
        out = np.zeros((1, 25, S, S))
        out[0, 0, 6, 6] = tx
        out[0, 1, 6, 6] = -tx / 2
        bxs, _, _ = decode_head(out, [[0.1, 0.1]], 20, "v2")
        cx, cy = bxs[0, 6 * S + 6, :2] * S
        ax.plot(cx, cy, "o", color=st.PALETTE[0], alpha=0.3 + 0.7 * k / len(t), ms=5)
    ax.add_patch(Rectangle((6, 6), 1, 1, fill=False, ec=st.PALETTE[1], lw=1.5))
    ax.set_xlim(4.5, 8.5)
    ax.set_ylim(8.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(5, 9))
    ax.set_yticks(range(5, 9))
    ax.set_title("centre décodé pour t ∈ [−8, 8] : reste dans la cellule (6, 6)", fontsize=8.5)
    eq(ax, r"$b_x=\frac{\sigma(t_x)+j}{S},\ b_y=\frac{\sigma(t_y)+i}{S}$", y=0.12)
    ax = axes[2]
    anchors = anchors_frac(load_cfg("tiny-yolov2-voc")["anchors"])
    tw = np.linspace(-2, 2, 81)
    for k, (pw, _) in enumerate(anchors):
        out = np.zeros((1, 5 * 25, 1, 1))
        res = []
        for v in tw:
            out[0, k * 25 + 2, 0, 0] = v
            res.append(decode_head(out, anchors, 20, "v2")[0][0, k, 2])
        ax.plot(tw, np.array(res) * 416, color=st.PALETTE[k], label=f"ancre {k} : p_w = {pw * 416:.0f} px")
    ax.set_yscale("log")
    ax.set_xlabel("t_w")
    ax.set_ylabel("b_w (px à 416)")
    ax.legend(loc="upper left", fontsize=6.5)
    eq(ax, r"$b_w=p_w\,e^{t_w}$", x=0.6, y=0.15)
    ax.set_title("taille de la boîte selon t_w", fontsize=8.5)
    fiche(fig, "yolo.infer.boxes:iou, iou_wh, cxcywh_to_xyxy ; yolo.infer.decode:decode_head",
          "torchvision.ops.box_iou", "test_boxes.py, test_decode.py")
    return finish(fig, out_dir, "boites_decodage", "Boîtes, IoU et décodage (§5.2, §8.1)")


# --- T13.38 : cibles et perte -------------------------------------------------------------

def toy_loss():
    """Perte v3 sur une tête 13×13 jouet (3 ancres, 20 classes, sortie aléatoire)."""
    from yolo.models.tiny_yolo import load_cfg
    from yolo.data.targets import heads
    from yolo.train.loss import yolo_loss

    spec = load_cfg("tiny-yolov3-voc")
    hl = heads(spec)
    rng = np.random.default_rng(SEED)
    outs = {hid: rng.standard_normal((2, 75, s, s)) * 0.5 for (hid, _), s in zip(hl, (13, 26))}
    for o in outs.values():  # objectness proche de celle d'un réseau entraîné (σ ≈ 0,02)
        o.reshape(2, 3, 25, *o.shape[-2:])[:, :, 4] -= 4
    gt = [np.array([[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.5, 0.6]]), np.array([[0.5, 0.5, 0.06, 0.1]])]
    lab = [np.array([3, 14]), np.array([4])]
    res = yolo_loss(outs, gt, lab, spec["anchors"], hl, 20)
    return spec, hl, outs, gt, lab, res


@figure("perte", "maths", "T13.38",
        "Cibles et perte YOLO : encodage d'une vérité, la perte terme par terme, le poids "
        "ω = 2 − g_w g_h, le masque ignore et la part de chaque terme sur un lot.",
        "yolo.data.targets.build_targets, yolo.train.loss (yolo_loss, ignore_mask)")
def perte(out_dir):
    from matplotlib.patches import Rectangle

    from yolo.data.targets import build_targets

    spec, hl, outs, gt, lab, res = toy_loss()
    plt = st.plt()
    fig = plt.figure(figsize=(st.FULL, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    # 1. encodage
    ax = fig.add_subplot(gs[0, 0])
    hid = hl[0][0]
    tg = build_targets(gt, lab, spec["anchors"], hl, {h: o.shape[-1] for h, o in outs.items()})[hid]
    S = 13
    g = gt[0][0]
    for k in range(S + 1):
        ax.axhline(k, color=st.GRID, lw=0.5)
        ax.axvline(k, color=st.GRID, lw=0.5)
    n_, a_, i_, j_ = [int(v[0]) for v in np.nonzero(tg["obj"][:1])]
    ax.add_patch(Rectangle((j_, i_), 1, 1, color=st.PALETTE[3], alpha=0.6))
    ax.add_patch(Rectangle(((g[0] - g[2] / 2) * S, (g[1] - g[3] / 2) * S), g[2] * S, g[3] * S,
                           fill=False, ec=st.PALETTE[5], lw=1.6))
    ax.plot(g[0] * S, g[1] * S, "x", color=st.PALETTE[5])
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    ax.set_aspect("equal")
    st.image_axes(ax, f"cellule ({i_}, {j_}), ancre {hl[0][1][a_]} : x* = {tg['x'][0, a_, i_, j_]:.2f}, "
                  f"t_w* = {tg['tw'][0, a_, i_, j_]:.2f}")
    ax.title.set_fontsize(8)
    eq(ax, r"$x^*=g_xS-j,\ t_w^*=\ln(g_w/p_w)$", y=0.1, size=7.5)
    # 2. la perte
    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    ax.text(0.0, 0.85, r"$L=\lambda_{coord}\sum_{obj}\omega\,[(\sigma(t_x)-x^*)^2+(\sigma(t_y)-y^*)^2"
            r"+(t_w-t_w^*)^2+(t_h-t_h^*)^2]$", fontsize=10)
    ax.text(0.04, 0.6, r"$+\sum_{obj}\mathrm{BCE}(\sigma(t_o),1)+\sum_{noobj}\mathrm{BCE}"
            r"(\sigma(t_o),0)+\sum_{obj}L_{cls}$", fontsize=10)
    ax.text(0.0, 0.35, "v3 : L_cls = Σ_c BCE(σ(t_c), 1[c = c*]) ;  v2 : L_cls = −ln softmax(t)_{c*}\n"
            "ignore : ancre non responsable dont la boîte prédite dépasse ignore_thresh d'IoU\n"
            "avec une vérité → exclue de tous les termes (yolo.train.loss, docstring)", fontsize=8,
            color=st.INK2)
    # 3. ω
    ax = fig.add_subplot(gs[1, 0])
    vals = np.linspace(0.05, 1, 20)
    grids = {h: o.shape[-1] for h, o in outs.items()}
    om = np.zeros((len(vals), len(vals)))
    for xi, w in enumerate(vals):
        for yi, h in enumerate(vals):
            t = build_targets([np.array([[0.5, 0.5, w, h]])], [np.array([0])], spec["anchors"],
                              hl, grids)
            om[yi, xi] = max(tt["scale"][tt["obj"]].max() for tt in t.values() if tt["obj"].any())
    cs = ax.contourf(vals, vals, om, levels=11, cmap="Blues")
    fig.colorbar(cs, ax=ax, fraction=0.046)
    ax.set_xlabel("g_w")
    ax.set_ylabel("g_h")
    ax.set_title("ω = 2 − g_w g_h (build_targets)", fontsize=8.5)
    # 4. masque ignore
    ax = fig.add_subplot(gs[1, 1])
    m = res.masks[hid]
    best = m["best_iou"][0].max(axis=0)
    ax.imshow(best, cmap="Greys", vmin=0, vmax=1)
    ig = m["ignore"][0].any(axis=0)
    ob = m["obj"][0].any(axis=0)
    ys, xs = np.nonzero(ig)
    ax.plot(xs, ys, "s", mfc="none", mec=st.PALETTE[7], ms=6, label=f"ignore ({ig.sum()})")
    ys, xs = np.nonzero(ob)
    ax.plot(xs, ys, "o", color=st.PALETTE[3], ms=5, label=f"obj ({ob.sum()})")
    st.image_axes(ax, "IoU prédite (gris), masques")
    ax.title.set_fontsize(8.5)
    ax.legend(loc="lower right", fontsize=6.5)
    # 5. parts
    ax = fig.add_subplot(gs[1, 2])
    parts = res.parts
    ax.bar(list(parts), list(parts.values()), color=st.PALETTE[:4])
    total = sum(parts.values())
    ax.set_title(f"Σ termes = LossResult.total = {res.total:.2f}", fontsize=8.5)
    assert np.isclose(total, res.total)
    ax.set_ylabel("perte (somme sur le lot de 2)")
    fiche(fig, "yolo.data.targets:build_targets ; yolo.train.loss:yolo_loss, ignore_mask",
          "torch.nn.BCEWithLogitsLoss, ultralytics", "test_targets.py, test_loss.py")
    return finish(fig, out_dir, "perte", "Cibles et perte YOLO (§5.1, §6.2)")


# --- T13.39 : optimiseur et taux d'apprentissage ------------------------------------------

def sgd_path(momentum, steps=80, lr=0.03):
    """Trajectoire de `SGD` sur f(p) = ½ pᵀ diag(1, 25) p (mal conditionnée), sans decay."""
    from yolo.train.optim import SGD

    H = np.array([1.0, 25.0])
    p = {"W": np.array([-4.0, 1.5])}
    opt = SGD([p], momentum=momentum, weight_decay=0.0)
    path = [p["W"].copy()]
    for _ in range(steps):
        opt.step([{"W": H * p["W"]}], lr)
        path.append(p["W"].copy())
    return np.array(path)


@figure("optimisation", "maths", "T13.39",
        "SGD avec et sans momentum sur une quadratique mal conditionnée, taux d'apprentissage "
        "(montée puis paliers) et tailles multi-échelles.",
        "yolo.train.optim.SGD, yolo.train.schedule.lr_at, yolo.train.trainer.multiscale_size")
def optimisation(out_dir):
    from yolo.train.schedule import lr_at
    from yolo.train.trainer import multiscale_size

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.6))
    ax = axes[0]
    g = np.linspace(-4.5, 4.5, 200)
    X, Y = np.meshgrid(g, np.linspace(-2, 2, 120))
    ax.contour(X, Y, 0.5 * (X**2 + 25 * Y**2), levels=12, colors=st.GRID, linewidths=0.7)
    for mu, c in ((0.0, st.PALETTE[7]), (0.9, st.PALETTE[0])):
        path = sgd_path(mu)
        ax.plot(path[:, 0], path[:, 1], ".-", color=c, ms=3, lw=1, label=f"μ = {mu}")
    ax.plot(0, 0, "*", color=st.INK, ms=9)
    ax.legend(loc="lower right")
    eq(ax, r"$v\leftarrow\mu v-\mathrm{lr}(g+\mathrm{wd}\cdot p),\ p\leftarrow p+v$", size=7.5)
    ax.set_title("SGD sur ½(x² + 25 y²)", fontsize=9)
    ax = axes[1]
    its = np.arange(0, 40200, 50)
    lr = [lr_at(i, 1e-3, burn_in=1000, steps=(32000, 36000), scales=(0.1, 0.1)) for i in its]
    ax.plot(its, lr, color=st.PALETTE[0])
    ax.set_yscale("log")
    ax.set_xlabel("itération")
    ax.set_ylabel("lr")
    ax.set_ylim(1e-9, 3e-3)
    eq(ax, r"$\mathrm{lr}\cdot(it/\mathrm{burn\_in})^4$ puis ×0,1 aux paliers", size=7.5, y=0.2)
    ax.set_title("lr_at (cfg yolov2-tiny-voc)", fontsize=9)
    ax = axes[2]
    it = np.arange(0, 300)
    sizes = [multiscale_size(i) for i in it]
    ax.step(it, sizes, where="post", color=st.PALETTE[2])
    ax.set_yticks(range(320, 609, 64))
    ax.set_xlabel("itération")
    ax.set_ylabel("taille d'entrée")
    ax.set_title("multi-échelle : 320 à 608, tirage tous les 10 lots", fontsize=9)
    fiche(fig, "yolo.train.optim:SGD ; yolo.train.schedule:lr_at ; yolo.train.trainer:multiscale_size",
          "torch.optim.SGD, torch.optim.lr_scheduler", "test_train.py")
    return finish(fig, out_dir, "optimisation", "Optimiseur et taux d'apprentissage (§7.1)")


# --- T13.40 : k-means des ancres ----------------------------------------------------------

def toy_boxes(n=1500, seed=SEED):
    """Boîtes (w, h) jouets en pixels : petites nombreuses, quelques très grandes."""
    rng = np.random.default_rng(seed)
    small = np.exp(rng.normal([3.3, 3.5], 0.35, (int(n * 0.6), 2)))
    mid = np.exp(rng.normal([4.6, 4.3], 0.3, (int(n * 0.3), 2)))
    big = np.exp(rng.normal([5.7, 5.6], 0.2, (n - int(n * 0.6) - int(n * 0.3), 2)))
    return np.clip(np.concatenate([small, mid, big]), 4, 416)


def kmeans_euclid(wh, k, iters=100, seed=SEED):
    """Témoin : k-means euclidien en (w, h) (Lloyd), pour la comparaison seulement."""
    rng = np.random.default_rng(seed)
    c = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        a = np.argmin(((wh[:, None] - c[None]) ** 2).sum(-1), axis=1)
        c = np.array([wh[a == q].mean(0) if np.any(a == q) else c[q] for q in range(k)])
    return c


@figure("kmeans", "maths", "T13.40",
        "k-means des ancres avec la distance 1 − IoU face à la distance euclidienne, et "
        "l'initialisation k-means++ sur des boîtes jouets.",
        "yolo.data.anchors (kmeans_anchors, _init_plusplus, mean_best_iou)")
def kmeans(out_dir):
    from yolo.data.anchors import _init_plusplus, kmeans_anchors, mean_best_iou
    from yolo.infer.boxes import iou_wh

    wh = toy_boxes()
    k = 5
    ours = kmeans_anchors(wh, k)
    eu = kmeans_euclid(wh, k)
    plt = st.plt()
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL, 4.0))
    for ax, (cent, name, c) in zip(axes, ((ours, "1 − IoU (kmeans_anchors)", st.PALETTE[0]),
                                          (eu, "euclidienne (témoin)", st.PALETTE[7]))):
        assign = np.argmax(iou_wh(wh, cent), axis=1)
        ax.scatter(wh[:, 0], wh[:, 1], c=[st.PALETTE[a % 8] for a in assign], s=4, alpha=0.35)
        ax.plot(cent[:, 0], cent[:, 1], "X", ms=11, color=c, mec="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("w (px)")
        ax.set_ylabel("h (px)")
        ax.set_title(f"{name} : IoU moyenne {mean_best_iou(wh, cent):.4f}", fontsize=9)
    init = _init_plusplus(wh, k, np.random.default_rng(SEED))
    axes[0].plot(init[:, 0], init[:, 1], "o", mfc="none", mec=st.INK, ms=8, label="k-means++ (∝ d²)")
    axes[0].legend(loc="upper left")
    eq(axes[0], r"$d=1-\mathrm{IoU}(\mathrm{boîte},\mathrm{centre})$", y=0.12)
    st.note(axes[1], "attirée par les grandes boîtes", "lower right")
    fiche(fig, "yolo.data.anchors:kmeans_anchors, _init_plusplus, mean_best_iou",
          "sklearn.cluster.KMeans", "test_anchors.py")
    return finish(fig, out_dir, "kmeans", "k-means des ancres avec la distance 1 − IoU (§5.2)")


# --- T13.41 : NMS et mAP ------------------------------------------------------------------

def _voc_ref():
    import importlib.util

    spec = importlib.util.spec_from_file_location("voc_eval_ref", TESTS / "voc_eval_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@figure("nms_map", "maths", "T13.41",
        "NMS pas à pas, appariement détections / vérités et courbe précision-rappel, AP VOC07 "
        "sur 11 points face à l'aire sous l'enveloppe, égale à la référence du devkit.",
        "yolo.infer.nms (nms), yolo.infer.metrics (eval_class, voc_ap), tests/voc_eval_ref.py")
def nms_map(out_dir):
    from yolo.infer.boxes import cxcywh_to_xyxy
    from yolo.infer.metrics import eval_class, voc_ap
    from yolo.infer.nms import nms

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.9))
    rng = np.random.default_rng(SEED)
    ax = axes[0]
    centers = np.array([[0.3, 0.4], [0.7, 0.6]])
    boxes = np.concatenate([np.c_[c + rng.normal(0, 0.03, (5, 2)), rng.uniform(0.2, 0.3, (5, 2))]
                            for c in centers])
    scores = rng.uniform(0.3, 0.95, len(boxes))
    keep = nms(boxes, scores, 0.45)
    xy = cxcywh_to_xyxy(boxes)
    for k, (b, s_) in enumerate(zip(xy, scores)):
        kept = k in keep
        st.draw_boxes(ax, [b], [f"{s_:.2f}" if kept else ""],
                      color=st.PALETTE[5] if kept else st.GREY, lw=1.8 if kept else 0.8,
                      ls="-" if kept else "--")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal")
    st.image_axes(ax, f"NMS (IoU > 0,45) : {len(boxes)} → {len(keep)} boîtes")
    ax = axes[1]
    # appariement sur des images jouets : vérités et détections bruitées
    gts, ids, sc, bx = {}, [], [], []
    for i in range(30):
        g = rng.uniform(20, 300, (2, 2))
        g = np.c_[g, g + rng.uniform(40, 120, (2, 2))]
        gts[f"{i}"] = (g, np.array([False, i % 7 == 0]))
        for gg in g:
            if rng.random() < 0.85:
                ids.append(f"{i}")
                sc.append(rng.uniform(0.4, 1.0))
                bx.append(gg + rng.normal(0, 8, 4))
        for _ in range(rng.integers(0, 3)):
            ids.append(f"{i}")
            sc.append(rng.uniform(0.0, 0.7))
            x0 = rng.uniform(0, 300, 2)
            bx.append(np.r_[x0, x0 + rng.uniform(30, 90, 2)])
    rec, prec, ap = eval_class(ids, np.array(sc), np.array(bx), gts)
    ax.plot(rec, prec, color=st.PALETTE[0], lw=1.2, label="précision(rappel)")
    env = np.maximum.accumulate(prec[::-1])[::-1]
    ax.plot(rec, env, color=st.PALETTE[1], ls="--", lw=1, label="enveloppe max_{r̃ ≥ r}")
    for t in np.arange(0, 1.1, 0.1):
        p = prec[rec >= t]
        ax.plot([t], [p.max() if p.size else 0], "o", color=st.PALETTE[1], ms=4)
    ax.set_xlabel("rappel")
    ax.set_ylabel("précision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("IoU ≥ 0,5, une vérité une fois, difficult ignorés", fontsize=8.5)
    ax = axes[2]
    ref = _voc_ref()
    vals = {"VOC07, 11 points": (voc_ap(rec, prec, True), ref.voc_ap(rec, prec, True)),
            "VOC2010+, aire": (voc_ap(rec, prec, False), ref.voc_ap(rec, prec, False))}
    x = np.arange(len(vals))
    ax.bar(x - 0.18, [v[0] for v in vals.values()], 0.36, color=st.PALETTE[0], label="metrics.voc_ap")
    ax.bar(x + 0.18, [v[1] for v in vals.values()], 0.36, color=st.PALETTE[2], hatch="//",
           label="voc_eval_ref (devkit)")
    for xi, (a, b) in zip(x, vals.values()):
        assert abs(a - b) < 1e-12, (a, b)
        ax.text(xi, a + 0.02, f"{a:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x, list(vals))
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower center", fontsize=7)
    eq(ax, r"$AP=\frac{1}{11}\sum_{r\in\{0,0{,}1,\ldots,1\}}\max_{\tilde r\geq r}p(\tilde r)$",
       size=7.5)
    ax.set_title("égal à la référence du devkit", fontsize=8.5)
    fiche(fig, "yolo.infer.nms:nms ; yolo.infer.metrics:eval_class, voc_ap",
          "torchvision.ops.nms, devkit VOC MATLAB", "test_nms.py, test_metrics.py")
    return finish(fig, out_dir, "nms_map", "NMS et mAP VOC (§8.2, §8.3)")


# --- T13.42 : prétraitement et augmentations ----------------------------------------------

def _toy_image(w=200, h=120):
    from PIL import Image

    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([x / w, y / h, 0.5 + 0.5 * np.sin(x / 9) * np.cos(y / 7)], -1)
    rgb[30:80, 40:110] = [0.9, 0.2, 0.1]
    return Image.fromarray((rgb * 255).astype(np.uint8)), np.array([[75 / w, 55 / h, 70 / w, 50 / h]])


@figure("pretraitement", "maths", "T13.42",
        "Letterbox face à stretch, interpolation bilinéaire de Darknet, aller-retour RGB → HSV "
        "→ RGB et transformation affine des boîtes.",
        "yolo.data.letterbox, yolo.infer.pipeline (resize_darknet, preprocess), yolo.data.augment")
def pretraitement(out_dir):
    from yolo.data.augment import affine, hsv_to_rgb, rgb_to_hsv, transform_boxes, warp_image
    from yolo.data.letterbox import boxes_to_letterbox
    from yolo.infer.boxes import cxcywh_to_xyxy
    from yolo.infer.pipeline import preprocess, resize_darknet

    img, box = _toy_image()
    plt = st.plt()
    fig, axes = plt.subplots(1, 4, figsize=(st.FULL, 3.3))
    S = 160
    lb, _ = preprocess(img, S, "letterbox")
    sr, _ = preprocess(img, S, "stretch")
    ax = axes[0]
    ax.imshow(np.concatenate([lb, np.ones((3, S, 6)), sr], axis=2).transpose(1, 2, 0))
    bl = cxcywh_to_xyxy(boxes_to_letterbox(box, *img.size, S)) * S
    st.draw_boxes(ax, bl, ["letterbox"], color=st.PALETTE[5])
    st.draw_boxes(ax, cxcywh_to_xyxy(box) * S + [S + 6, 0, S + 6, 0], ["stretch"],
                  color=st.PALETTE[1])
    st.image_axes(ax, "letterbox (facteur min(S/w, S/h)) / stretch")
    ax.title.set_fontsize(8)
    ax = axes[1]
    src = np.arange(16, dtype=np.float32).reshape(4, 4, 1).repeat(3, -1) / 15
    up = resize_darknet(src, 7, 7)
    ax.imshow(up[..., 0], cmap="viridis")
    for (i, j), v in np.ndenumerate(up[..., 0] * 15):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white")
    st.image_axes(ax, "resize_darknet 4×4 → 7×7\n(coins alignés, linéaire x puis y)")
    ax.title.set_fontsize(8)
    ax = axes[2]
    rgb = np.asarray(img, dtype=np.float32) / 255
    back = hsv_to_rgb(rgb_to_hsv(rgb))
    err = float(np.abs(back - rgb).max())
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.7, 0, 1)
    ax.imshow(np.concatenate([rgb, np.ones((rgb.shape[0], 4, 3)), hsv_to_rgb(hsv)], 1))
    st.image_axes(ax, f"RGB → HSV → RGB : écart {err:.1e}\nsaturation ×1,5, exposition ×0,7")
    ax.title.set_fontsize(8)
    ax = axes[3]
    params = {"scale": 0.8, "shift": np.array([0.3, -0.1]), "flip": True, "sat": 1.0,
              "val": 1.0}
    m = affine(*img.size, S, params)
    boxes2 = np.r_[box, [[0.97, 0.5, 0.04, 0.2]]]  # boîte au bord : sort du carré
    tb, keep = transform_boxes(boxes2, *img.size, S, m)
    ax.imshow(warp_image(img, S, m))
    st.draw_boxes(ax, cxcywh_to_xyxy(tb) * S, ["boîte suivie"], color=st.PALETTE[5])
    st.image_axes(ax, f"affine (échelle, translation, retournement)\n{keep.sum()} / {len(keep)} "
                  "boîtes gardées (côté ≥ 2 px)")
    ax.title.set_fontsize(8)
    fiche(fig, "yolo.infer.pipeline:preprocess, resize_darknet ; yolo.data.augment:affine, "
          "transform_boxes, rgb_to_hsv, hsv_to_rgb", "cv2.resize, cv2.warpAffine, albumentations",
          "test_data_pipeline.py")
    return finish(fig, out_dir, "pretraitement", "Prétraitement et augmentations (§1, §7.1)")


# --- T13.43 : quantification symétrique et calibration ------------------------------------

@figure("quantification", "maths", "T13.43",
        "Quantification symétrique : l'escalier et son erreur, échelles par canal face à une "
        "échelle par couche, choix du seuil d'écrêtage par la MSE.",
        "yolo.quant.quantize (round_half_up, quantize, weight_scales), yolo.quant.calibrate")
def quantification(out_dir):
    from yolo.quant.calibrate import clip_candidates, clip_rate, quant_mse
    from yolo.quant.quantize import QMAX, quantize, round_half_up, weight_scales

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.6))
    ax = axes[0]
    s = 0.25
    x = np.linspace(-1.6, 1.6, 2001)
    q = quantize(x, s, qmax=5)
    ax.plot(x, q * s, color=st.PALETTE[0], label="s·q")
    ax.plot(x, x - q * s, color=st.PALETTE[1], lw=1, label="x − s·q")
    assert round_half_up(-2.5) == -2 and round_half_up(2.5) == 3
    eq(ax, r"$q=\mathrm{clip}(\lfloor x/s+\frac{1}{2}\rfloor,-q_{max},q_{max})$", size=7.5)
    ax.legend(loc="lower right")
    ax.set_xlabel("x (ici s = 0,25, q_max = 5)")
    ax.set_title("escalier et dents de scie ; saturation", fontsize=8.5)
    ax = axes[1]
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((32, 16, 3, 3)) * np.exp(rng.normal(-3, 1, 32))[:, None, None, None]
    sw = weight_scales(W)
    per_ch = np.abs(W - quantize(W, sw[:, None, None, None]) * sw[:, None, None, None])
    s1 = np.abs(W).max() / QMAX
    per_layer = np.abs(W - quantize(W, s1) * s1)

    def rms(e):
        return np.sqrt((e ** 2).reshape(32, -1).mean(1) / (W ** 2).reshape(32, -1).mean(1))

    order = np.argsort(sw)
    ax.semilogy(rms(per_layer)[order], "o-", ms=3, color=st.PALETTE[7], label="une échelle par couche")
    ax.semilogy(rms(per_ch)[order], "s-", ms=3, color=st.PALETTE[0], label="s_w,f = max|W_f|/127")
    ax.set_xlabel("canal de sortie (trié par amplitude)")
    ax.set_ylabel("erreur RMS relative")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("échelles par canal", fontsize=8.5)
    ax = axes[2]
    v = np.concatenate([rng.laplace(0, 0.5, 20000), rng.laplace(0, 4, 40)])
    clips = np.linspace(0.3, np.abs(v).max(), 120)
    mse = [quant_mse(v, c) for c in clips]
    best = clips[int(np.argmin(mse))]
    ax.semilogy(clips, mse, color=st.PALETTE[0], label="quant_mse")
    for name, c in clip_candidates(v, np.abs(v).max()).items():
        ax.axvline(c, color=st.GRID, lw=0.8)
        ax.text(c, max(mse), name, rotation=90, fontsize=6, va="top", color=st.MUTED)
    ax.axvline(best, color=st.PALETTE[1], lw=1.2)
    ax.set_xlabel("seuil d'écrêtage c (s = c/127)")
    ax.set_ylabel("MSE")
    a2 = ax.twinx()
    a2.plot(clips, [100 * clip_rate(v, c / QMAX) for c in clips], color=st.PALETTE[3], ls="--", lw=1)
    a2.set_ylabel("saturées (%)", color=st.PALETTE[3])
    a2.grid(False)
    ax.set_title(f"seuil MSE minimal c = {best:.2f}", fontsize=8.5)
    fiche(fig, "yolo.quant.quantize:round_half_up, quantize, weight_scales ; "
          "yolo.quant.calibrate:clip_candidates, quant_mse, clip_rate",
          "torch.quantization, TensorRT", "test_quantize.py")
    return finish(fig, out_dir, "quantification", "Quantification symétrique et calibration (§9.2)")


# --- T13.44 : arithmétique entière ---------------------------------------------------------

@figure("entier", "maths", "T13.44",
        "Arithmétique entière du matériel : chaîne d'une sortie, multiplicateur fixe M0/2ⁿ, "
        "leaky entière 13/128 et marge des accumulateurs int32.",
        "yolo.quant.int_layers (conv_acc, requantize, leaky_int, clip_q), quantize.requant_params")
def entier(out_dir):
    from yolo.quant.int_layers import clip_q, conv_acc, leaky_int, requantize
    from yolo.quant.quantize import quantize_weights_per_channel, requant_params

    plt = st.plt()
    fig, axes = plt.subplots(1, 4, figsize=(st.FULL, 3.6), gridspec_kw={"width_ratios":
                                                                         [1.3, 1, 1, 1]})
    rng = np.random.default_rng(SEED)
    sx, sy = 0.05, 0.1
    W = rng.standard_normal((16, 8, 3, 3)) * 0.1
    qW, sw = quantize_weights_per_channel(W)
    qx = rng.integers(-127, 128, (1, 8, 10, 10))
    qb = rng.integers(-500, 500, 16)
    acc = conv_acc(qx, qW.astype(np.int64), qb)
    M0, n = requant_params(sx, sw, sy)
    y = requantize(acc, M0[None, :, None, None], n)
    yl = leaky_int(y)
    yq = clip_q(yl)
    ax = axes[0]
    st.schema_axes(ax, (0, 6), (0, 9))
    steps = [("q_x, q_w : int8", f"{qx.min()}…{qx.max()}"),
             ("acc = Σ q_x q_w + q_b : int32", f"{acc.min():_}…{acc.max():_}".replace("_", " ")),
             (f"(acc·M0 + 2ⁿ⁻¹) ≫ n, n = {n}", f"{y.min()}…{y.max()}"),
             ("leaky_int", f"{yl.min()}…{yl.max()}"), ("clip_q → int8", f"{yq.min()}…{yq.max()}")]
    for k, (lab, rngtxt) in enumerate(steps):
        yy = 8 - 1.75 * k
        st.block(ax, 0.1, yy - 0.6, 5.8, 1.2, f"{lab}\n{rngtxt}", st.PALETTE[k], fontsize=7)
        if k:
            st.arrow(ax, (3, yy + 1.15), (3, yy + 0.6))
    ax.set_title("une sortie, du produit à l'octet", fontsize=8.5)
    ax = axes[1]
    ratio = sx * sw / sy
    err = np.abs(M0 / 2.0**n - ratio) / ratio
    ax.semilogy(err, "o", color=st.PALETTE[0], ms=4)
    ax.set_xlabel("canal")
    ax.set_ylabel("|M0/2ⁿ − s_x s_w/s_y| / (s_x s_w/s_y)")
    eq(ax, r"$M_0=\lfloor\frac{s_xs_w}{s_y}2^n\rceil<2^{31}$", size=7.5, y=0.2)
    ax.set_title(f"multiplicateur fixe (n = {n} ≤ 31)", fontsize=8.5)
    ax = axes[2]
    t = np.arange(-64, 33)
    ax.plot(t, leaky_int(t), ".", ms=3, color=st.PALETTE[0], label="leaky_int")
    ax.plot(t, np.where(t > 0, t, 0.1 * t), color=st.PALETTE[1], lw=0.8, label="0,1·y")
    eq(ax, r"$y<0:\ (13y+64)\gg7$, pente $13/128\approx0{,}1016$", size=7)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("leaky entière", fontsize=8.5)
    ax = axes[3]
    a = np.abs(acc[acc != 0]).astype(np.float64)
    ax.hist(np.log2(a), bins=30, color=st.PALETTE[2])
    ax.axvline(31, color=st.PALETTE[7], lw=1.2)
    margin = 31 - np.log2(a.max())
    ax.set_xlabel("log₂ |acc|")
    ax.set_title(f"accumulateurs : marge {margin:.1f} bits", fontsize=8.5)
    fiche(fig, "yolo.quant.int_layers:conv_acc, requantize, leaky_int, clip_q ; "
          "yolo.quant.quantize:requant_params (= golden::requantize, leaky_int)",
          "Vitis AI, TFLite", "test_int_layers.py, test_golden_cpp.py")
    return finish(fig, out_dir, "entier", "Arithmétique entière du matériel (§9.3)")


# --- T13.45 : tables LUT --------------------------------------------------------------------

@figure("lut", "maths", "T13.45",
        "Sigmoïde et exponentielles par tables de 256 entrées en Q16 : valeurs, erreurs aux "
        "points de la grille et de bout en bout, seuil entier sans sigmoïde.",
        "yolo.quant.lut (sigmoid_lut, exp_lut, softmax_exp_lut, logit_threshold_q)")
def lut(out_dir):
    from yolo.layers.activations import sigmoid
    from yolo.quant.lut import (ONE, exp_frac_bits, exp_lut, logit_threshold_q, lut_inputs,
                                sigmoid_lut, softmax_exp_lut)

    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, 3.6))
    s = 1 / 8
    t = lut_inputs(s)
    sl = sigmoid_lut(s) / ONE
    ax = axes[0]
    ax.plot(t, sl, ".", ms=2.5, color=st.PALETTE[0], label="sigmoid_lut[q + 128] / 2¹⁶")
    tt = np.linspace(t[0], t[-1], 1000)
    ax.plot(tt, sigmoid(tt), color=st.PALETTE[1], lw=0.8, label="σ flottante")
    theta = 0.25
    qt = logit_threshold_q(theta, s)
    ax.axvline(qt * s, color=st.PALETTE[3], ls="--", lw=1)
    ax.text(qt * s + 0.3, 0.05, f"σ(q s) > {theta} ⟺ q ≥ {qt}", fontsize=7, color=st.PALETTE[3])
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("t = q·s (s = 1/8)")
    ax.set_title("table σ, 256 entrées Q16", fontsize=8.5)
    ax = axes[1]
    f = exp_frac_bits(s)
    ax.semilogy(t, exp_lut(s) / 2.0**f, color=st.PALETTE[0], label=f"exp_lut (Q{f})")
    d = np.arange(-255, 1)
    ax.semilogy(d * s, softmax_exp_lut(s) / ONE, color=st.PALETTE[2], label="softmax_exp_lut")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("q·s  ou  (q_c − q_max)·s")
    eq(ax, r"$b_w=p_w\,e^{t_w}$ ; softmax : $e^{(q_c-q_{max})s}$", size=7, y=0.2, x=0.3)
    ax.set_title("tables exponentielles", fontsize=8.5)
    ax = axes[2]
    grid_err = np.abs(sl - sigmoid(t))
    ax.semilogy(t, np.maximum(grid_err, 1e-12), ".", ms=2.5, color=st.PALETTE[0],
                label="aux points de la grille")
    assert grid_err.max() <= 2.0**-17 + 1e-15
    tc = np.linspace(-15, 15, 4001)
    qq = np.clip(np.floor(tc / s + 0.5), -128, 127).astype(int)
    e2e = np.abs(sigmoid_lut(s)[qq + 128] / ONE - sigmoid(tc))
    ax.semilogy(tc, e2e, color=st.PALETTE[1], lw=0.7, label="de bout en bout (pas s)")
    ax.axhline(2.0**-17, color=st.PALETTE[0], ls=":", lw=1)
    ax.axhline(s / 8, color=st.PALETTE[1], ls=":", lw=1)
    ax.text(-15, 2.0**-17 * 1.5, "2⁻¹⁷", fontsize=7, color=st.PALETTE[0])
    ax.text(-15, s / 8 * 1.3, "s/8 = 1/64", fontsize=7, color=st.PALETTE[1])
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title("erreur (bornes de la docstring)", fontsize=8.5)
    fiche(fig, "yolo.quant.lut:sigmoid_lut, exp_lut, softmax_exp_lut, logit_threshold_q",
          "torch.sigmoid, tables Vitis AI", "test_lut.py")
    return finish(fig, out_dir, "lut", "Sigmoïde et exponentielle par tables (§9.4)")


# --- T13.46 : QAT, puissances de 2 et 4 bits ----------------------------------------------

def admm_toy(kind="mixed6", iters=2000, every=50, lr=0.05):
    """ADMM (`train.admm.ADMM`) sur une perte quadratique ½‖W − W*‖² d'une conv jouette :
    trajectoire d'un poids, résidus primaux."""
    from types import SimpleNamespace

    from yolo.train.admm import ADMM

    rng = np.random.default_rng(SEED)
    target = rng.standard_normal((4, 3, 3, 3)) * 0.2
    net = SimpleNamespace(params={0: {"W": target.copy()}})
    admm = ADMM(net, {0: kind}, rho=0.05, every=every, growth=1.25, rho_max=20.0)
    w_path, res = [], []
    for it in range(iters):
        W = net.params[0]["W"]
        grads = {0: {"W": W - target}}
        admm.hook(SimpleNamespace(it=it), grads)
        net.params[0]["W"] = W - lr * grads[0]["W"]
        w_path.append((net.params[0]["W"][0, 0, 0, 0], admm.Z[0][0, 0, 0, 0]))
        if (it + 1) % every == 0:
            res.append((it + 1, admm.residuals()[0]))
    return np.array(w_path), np.array(res)


@figure("basse_precision", "maths", "T13.46",
        "Basse précision : fake-quant et estimateur straight-through, niveaux équidistants "
        "face aux puissances de 2 et multiplication par décalages, ADMM, découpage 4 bits.",
        "yolo.quant.fake_quant, yolo.quant.pow2, yolo.train.admm, yolo.quant.int_layers.split4")
def basse_precision(out_dir):
    from yolo.quant.fake_quant import fq_act_backward, fq_act_forward
    from yolo.quant.int_layers import conv_acc, conv_acc_split4, split4
    from yolo.quant.pow2 import levels, shift_add_mul

    plt = st.plt()
    fig, axes = plt.subplots(1, 4, figsize=(st.FULL, 3.6))
    ax = axes[0]
    x = np.linspace(-12, 12, 1201)
    y, cache = fq_act_forward(x, -1.0, 15)
    dx, _ = fq_act_backward(np.ones_like(x), cache)
    ax.plot(x, y, color=st.PALETTE[0], label="avant : escalier (s = 2⁻¹)")
    ax.plot(x, dx * 4, color=st.PALETTE[1], ls="--", label="arrière ×4 : STE (1 dans la plage)")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("fake-quant 5 bits, STE", fontsize=8.5)
    ax = axes[1]
    for k, kind in enumerate(("uniform6", "mixed6", "pot5")):
        lv = levels(kind)
        ax.plot(lv, np.full(len(lv), k), "|", ms=12, mew=1.5, color=st.PALETTE[k])
        ax.text(-2, k, f"{kind} ({len(lv)})", ha="right", va="center", fontsize=7)
    xs = np.arange(-20, 21)
    assert np.array_equal(shift_add_mul(xs, np.full(len(xs), 96)), 96 * xs)
    ax.set_xlim(-30, 100)
    ax.set_yticks([])
    ax.set_xlabel("magnitude entière")
    ax.text(30, -0.8, "96·x = (x ≪ 6) + (x ≪ 5)\nshift_add_mul = q·x", fontsize=7)
    ax.set_ylim(-1.3, 2.6)
    ax.set_title("niveaux et décalages", fontsize=8.5)
    ax = axes[2]
    path, res = admm_toy()
    ax.plot(path[:, 0], color=st.PALETTE[0], lw=1, label="W[0] (pas SGD)")
    ax.step(range(len(path)), path[:, 1], where="post", color=st.PALETTE[1], lw=1,
            label="Z[0] = Π_S(W + U)")
    ax.set_xlabel("itération")
    ax.legend(loc="upper right", fontsize=6.5)
    a2 = ax.twinx()
    a2.semilogy(res[:, 0], res[:, 1], "o", ms=3, color=st.PALETTE[3])
    a2.set_ylabel("‖W − Z‖/‖W‖", color=st.PALETTE[3])
    a2.grid(False)
    ax.set_title("ADMM, ρ croissant", fontsize=8.5)
    ax = axes[3]
    rng = np.random.default_rng(SEED)
    qx = rng.integers(-128, 128, (1, 3, 6, 6))
    hi, lo = split4(qx)
    assert np.array_equal(16 * hi + lo, qx)
    qW = rng.integers(-7, 8, (4, 3, 3, 3))
    qb = rng.integers(-50, 50, 4)
    same = np.array_equal(conv_acc_split4(qx, qW, qb), conv_acc(qx, qW, qb))
    ax.hist([hi.ravel(), lo.ravel()], bins=np.arange(-8.5, 16.5), color=st.PALETTE[:2],
            label=["x ≫ 4 ∈ [−8, 7]", "x & 15 ∈ [0, 15]"])
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_title(f"x = 16·(x≫4) + (x&15) ; conv égale : {same}", fontsize=8)
    fiche(fig, "yolo.quant.fake_quant:fq_act_forward/backward ; yolo.quant.pow2:levels, "
          "shift_add_mul ; yolo.train.admm:ADMM ; yolo.quant.int_layers:split4, conv_acc_split4",
          "torch.ao.quantization, Brevitas", "test_fake_quant.py, test_pow2.py, test_admm.py")
    return finish(fig, out_dir, "basse_precision", "QAT, puissances de 2 et 4 bits (§9.2)")


# --- T13.47 : formats de fichiers ---------------------------------------------------------

@figure("formats", "maths", "T13.47",
        "Formats lus et écrits à la main : un .weights Darknet octet par octet, le parser .cfg "
        "et l'arène exportée (décalages égaux au manifest).",
        "yolo.io.darknet_weights, yolo.models.cfg.parse_cfg, yolo.io.export.layout")
def formats(out_dir):
    import json

    from yolo.io.darknet_weights import read_header
    from yolo.io.export import layout
    from yolo.models.specs import infer_shapes
    from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, load_cfg

    net = "tiny-yolov2-voc"
    spec = load_cfg(net)
    shapes = infer_shapes(spec)
    plt = st.plt()
    fig, axes = plt.subplots(3, 1, figsize=(st.FULL, 7.0), gridspec_kw={"height_ratios":
                                                                         [1.2, 1.1, 0.8]})
    # 1. .weights
    ax = axes[0]
    wpath = ROOT / "weights" / "yolov2-tiny-voc.weights"
    header = None
    if wpath.exists():
        with open(wpath, "rb") as f:
            header = read_header(f)
    hdr = 4 * 3 + (8 if header and (header["version"][0] * 10 + header["version"][1]) >= 2 else 4)
    segs = [("en-tête", hdr, st.INK2)]
    for layer, (ins, outs) in zip(spec["layers"], shapes):
        if layer["type"] != "conv":
            continue
        F, C, k = outs[0], ins[0], layer["k"]
        if layer["bn"]:
            segs.append(("β γ μ σ²", 4 * 4 * F, st.PALETTE[3]))
        else:
            segs.append(("biais", 4 * F, st.PALETTE[3]))
        segs.append((f"W ({F},{C},{k},{k})", 4 * F * C * k * k, st.PALETTE[0]))
    total = sum(n for _, n, _ in segs)
    zoom = segs[:5]
    ztot = sum(n for _, n, _ in zoom)
    for y, (row, tot) in enumerate(((zoom, ztot), (segs, total))):
        x0 = 0
        for lab, n, c in row:
            ax.barh(y, n / tot, left=x0 / tot, color=c, ec="white", lw=0.4, height=0.6)
            if n / tot > 0.06:
                ax.text((x0 + n / 2) / tot, y, lab, ha="center", va="center", fontsize=6.5,
                        color="white")
            if y == 0:
                ax.text(x0 / tot, -0.42, f"{x0}", fontsize=6, color=st.INK2, ha="center")
            x0 += n
    if wpath.exists():
        assert total == wpath.stat().st_size, (total, wpath.stat().st_size)
    ax.set_yticks([0, 1], [f"zoom : {ztot:_} premiers octets".replace("_", " "), "fichier entier"])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 1.4)
    ax.set_xticks([])
    ax.set_xlabel("décalages en octets (zoom) ; float32 petit-boutiste")
    h = (f"en-tête lu : version {header['version']}, seen = {header['seen']}" if header
         else "en-tête : major, minor, revision (int32), seen (int64 si ≥ 0.2)")
    ax.set_title(f"yolov2-tiny-voc.weights : {total:_} octets ; {h}".replace("_", " "), fontsize=8.5)
    ax.grid(False)
    # 2. .cfg → dicts
    ax = axes[1]
    ax.axis("off")
    text = (CFG_DIR / CFG_FILES[net]).read_text().splitlines()
    start = next(i for i, line in enumerate(text) if line.startswith("[convolutional]"))
    snippet = [line for line in text[start:start + 16] if line.strip()][:12]
    ax.text(0.0, 1.0, "\n".join(snippet), family="monospace", fontsize=7, va="top")
    ax.text(0.3, 0.55, "parse_cfg  →", fontsize=10, weight="bold", color=st.PALETTE[0])
    dicts = "\n".join(str(d) for d in spec["layers"][:3]) + "\n…\n" + str(spec["layers"][-1])
    ax.text(0.45, 1.0, dicts, family="monospace", fontsize=6.5, va="top", wrap=True)
    ax.set_title("le parser .cfg : sections Darknet → dicts de models/specs", fontsize=8.5)
    # 3. arène
    ax = axes[2]
    entries, buffers = layout(spec)
    m = json.loads((ROOT / "model" / net / "manifest.json").read_text()) if (
        ROOT / "model" / net / "manifest.json").exists() else None
    if m is not None:
        assert buffers == m["buffers"], (buffers, m["buffers"])
    x0 = 0
    for k, (name, size) in enumerate(buffers.items()):
        ax.barh(0, size, left=x0, color=st.PALETTE[k], ec="white")
        ax.text(x0 + size / 2, 0, f"{name}\n{size / 1e3:.0f} Ko", ha="center", va="center",
                fontsize=7, color="white")
        x0 += size
    ax.set_yticks([])
    ax.set_xlim(0, x0)
    ax.set_xlabel("octets ; alignement 64")
    ax.set_title("arène d'activations (export.layout) : ping-pong A/B et tête H13"
                 + (" — égale au manifest" if m is not None else ""), fontsize=8.5)
    ax.grid(False)
    fiche(fig, "yolo.io.darknet_weights:read_header, load_darknet_weights ; yolo.models.cfg:"
          "parse_cfg ; yolo.io.export:layout", "darknet, onnx, Vitis AI",
          "test_darknet_weights.py, test_cfg.py, test_export.py")
    return finish(fig, out_dir, "formats", "Formats de fichiers lus et écrits à la main")

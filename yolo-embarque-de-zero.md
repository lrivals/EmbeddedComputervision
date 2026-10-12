# Implementing an embedded YOLO from scratch

*`fpga` base, 2026-10-04. Common thread: **Tiny-YOLOv2** and **Tiny-YOLOv3**, from the
network to the FPGA accelerator, with no deep learning library.*

This document describes everything needed to rewrite a light YOLO detector by hand and
deploy it on embedded hardware: the architecture layer by layer, the mathematical functions,
the forward and backward passes of each building block, the loss function and its gradient,
training, inference (decoding, NMS, mAP), then BN folding, integer quantization and the
organisation of an FPGA accelerator.

## 0. Sourcing conventions

- `[1506.02640#004.1]`: a passage of the base, cited by its `chunk_id` and checked with
  `bin/pdb verify`. It can be reread with `bin/pdb show <chunk_id>`.
- **(outside the base)**: what the papers of the base do not describe. It is either a
  standard mathematical derivation (gradients), a convention of the Darknet reference
  implementation (`.cfg` files), or an engineering choice. None of these statements is
  attributed to a paper.
- **(checked numerically)**: a formula checked by finite differences in a NumPy reference
  implementation (§11).

Founding papers: YOLOv1 `arxiv-1506.02640`, YOLO9000/YOLOv2 `arxiv-1612.08242`, YOLOv3
`arxiv-1804.02767`. Embedded implementations: REQ-YOLO `2019-ding`, `2023-zhai`
(YOLOv3-tiny, HLS), `2024-zhang` (YOLO on an FPGA alone, hardware post-processing),
`2025-kim` (YOLOv2 on Zynq-7000), `2026-fata` (YOLOv3-tiny), `2024-yan` (low-bit YOLOv5),
TinyissimoYOLO `arxiv-2306.00001`. The full list is in §12.

### Notation

| Symbol | Meaning |
|---|---|
| $S$ | side of the output grid (13 or 26 for a 416 input) |
| $A$ (or $B$ in v1) | number of boxes predicted per cell (anchors) |
| $C$ | number of classes (20 for VOC, 80 for COCO) |
| $c_x, c_y$ | (column, row) indices of the cell |
| $p_w, p_h$ | width and height of the anchor (*prior*) |
| $t_x, t_y, t_w, t_h, t_o$ | raw network outputs for one box |
| $\sigma(t) = 1/(1+e^{-t})$ | sigmoid (logistic) |
| $\mathbb{1}^{obj}_{ij}$ | 1 if box $j$ of cell $i$ is "responsible" for an object |

Tensors use the `(N, C, H, W)` layout: batch, channels, height, width.

---

## 1. Overview of the pipeline

```
RGB image ──► preprocessing ──► conv backbone ──► 1×1 head(s) ──► decoding ──► threshold ──► NMS ──► boxes
 (H×W×3)      416×416, /255     (conv-BN-leaky,    S×S×A(5+C)      σ, exp        score        IoU
                                 maxpool)
```

- **Detection = a single regression.** YOLO divides the image into an $S\times S$ grid; the
  cell that contains the centre of an object is in charge of detecting it
  [1506.02640#002.0]. The network is applied once per image.
- **Input.** 416×416 from YOLOv2 on. The network downsamples by a factor of 32, hence a
  13×13 output map [1612.08242#002.2]. YOLOv3-tiny adds a second output at 26×26
  [2023-zhai#004.1].
- **Preprocessing (outside the base).** Resize while keeping the aspect ratio and pad with
  grey (*letterbox*, Darknet convention), then divide the pixels by 255 to bring them into
  $[0,1]$.

---

## 2. The YOLO principle: what each version brings

### 2.1 YOLOv1: grid and direct regression

- Each cell predicts $B$ boxes. Each box carries 5 numbers $(x, y, w, h,
  \text{confidence})$; the cell also carries $C$ conditional probabilities
  $\Pr(\text{Class}_i\mid\text{Object})$, once, whatever the value of $B$
  [1506.02640#002.0].
- The confidence targets $\Pr(\text{Object})\cdot \text{IOU}^{truth}_{pred}$
  [1506.02640#002.0]. At test time, the class-specific score is
  $\Pr(\text{Class}_i\mid\text{Obj})\cdot\Pr(\text{Obj})\cdot\text{IOU} = \Pr(\text{Class}_i)\cdot\text{IOU}$
  [2019-ding#003.1].
- On VOC, $S=7$, $B=2$ and $C=20$: the output is a 7×7×30 tensor [1506.02640#002.1]. In
  general, the size is $S\times S\times(B\cdot5+C)$ [2019-ding#003.1].
- $(x,y)$ are relative to the cell and $(w,h)$ to the whole image, all in $[0,1]$. The last
  layer is linear; all the others use the *leaky ReLU* with slope 0.1 [1506.02640#004.0].
- Weakness of this version: fully connected output layers and many localisation errors.
  YOLOv2 fixes them.

### 2.2 YOLOv2 (YOLO9000): BN, anchors, bounded direct prediction

- **Batch normalization** on every convolution: +2 % mAP and no more dropout
  [1612.08242#002.0].
- **Anchors.** The fully connected layers are removed; each position of the map predicts
  offsets from prior boxes. The input moves to 416 to get an odd grid, which has a centre
  cell [1612.08242#002.2].
- **Anchors by k-means** on the training boxes, with the distance
  $d(\text{box},\text{centroid}) = 1-\text{IOU}(\text{box},\text{centroid})$; $k=5$ is a
  trade-off [1612.08242#002.4].
- **Direct location prediction**, the central formula of YOLO [1612.08242#003.1]:

$$
b_x = \sigma(t_x)+c_x,\qquad b_y=\sigma(t_y)+c_y,\qquad b_w=p_w e^{t_w},\qquad b_h=p_h e^{t_h},\qquad
\Pr(\text{object})\cdot\text{IOU}(b,\text{object})=\sigma(t_o)
$$

  The sigmoid keeps the centre in its cell and the exponential guarantees $b_w,b_h>0$.
- **Passthrough.** The 26×26×512 map is reorganised into 13×13×2048 (*space-to-depth*)
  then concatenated with the 13×13 features [1612.08242#003.2].
- **Multi-scale training.** Every 10 batches, a new input size is drawn among the
  multiples of 32 between 320 and 608 [1612.08242#003.2].
- **VOC head.** 5 boxes × (5 + 20) = 125 filters in the final 1×1 convolution
  [1612.08242#005.3].

### 2.3 YOLOv3: logistic objectness, multi-label, multi-scale

- Same decoding as YOLOv2. The coordinates are learned with a sum of squares, and the
  gradient is "ground truth minus prediction", $\hat t^* - t^*$ [1804.02767#003.0].
- **Objectness.** It is learned by logistic regression. Its target is 1 for the anchor
  that overlaps an object best. An anchor that is not the best but whose IoU with an object
  exceeds 0.5 is **ignored**. An unassigned anchor only receives the objectness loss
  [1804.02767#003.1].
- **Classes.** Independent logistic classifiers trained with binary cross-entropy replace
  the softmax [1804.02767#004.0].
- **Three scales.** Each output is $N\times N\times[3\cdot(4+1+80)]$ on COCO
  [1804.02767#005.0]. A deeper map is upsampled ×2 then concatenated with an earlier map
  [1804.02767#005.0].
- **9 COCO anchors** (in pixels for 416): (10×13), (16×30), (33×23), (30×61), (62×45),
  (59×119), (116×90), (156×198), (373×326) [1804.02767#005.1].
- Abandoned attempt: the *focal loss* loses about 2 mAP points [1804.02767#009.0].

---

## 3. Tiny architectures layer by layer

The papers of the base describe the structure of these networks; the exact channel counts
and the stride-1 maxpool come from Darknet's `yolov2-tiny-voc.cfg` and `yolov3-tiny.cfg`
files **(outside the base)**. Parameters and MACs were computed by a script (§11). A "conv"
layer means convolution, BN and leaky ReLU 0.1, except the output convolution, which is
linear with a bias.

### 3.1 Tiny-YOLOv2 (VOC, 416×416×3 input)

REQ-YOLO describes this network as follows: 9 CONV layers; in layers 1 to 8, a 3×3
convolution (stride 1, padding 1) followed by a 2×2 maxpool with stride 2; a final 1×1
convolution that reduces the feature space [2019-ding#003.1, 2019-ding#003.2].

| # | Layer | k/s | C_in→C_out | Output | Parameters | MACs (M) |
|---|---|---|---|---|---|---|
| 0 | conv | 3×3/1 | 3→16 | 416×416×16 | 464 | 74.8 |
| 1 | maxpool | 2×2/2 | | 208×208×16 | | |
| 2 | conv | 3×3/1 | 16→32 | 208×208×32 | 4,672 | 199.4 |
| 3 | maxpool | 2×2/2 | | 104×104×32 | | |
| 4 | conv | 3×3/1 | 32→64 | 104×104×64 | 18,560 | 199.4 |
| 5 | maxpool | 2×2/2 | | 52×52×64 | | |
| 6 | conv | 3×3/1 | 64→128 | 52×52×128 | 73,984 | 199.4 |
| 7 | maxpool | 2×2/2 | | 26×26×128 | | |
| 8 | conv | 3×3/1 | 128→256 | 26×26×256 | 295,424 | 199.4 |
| 9 | maxpool | 2×2/2 | | 13×13×256 | | |
| 10 | conv | 3×3/1 | 256→512 | 13×13×512 | 1,180,672 | 199.4 |
| 11 | **maxpool** | **2×2/1** | | 13×13×512 | | |
| 12 | conv | 3×3/1 | 512→1024 | 13×13×1024 | 4,720,640 | 797.4 |
| 13 | conv | 3×3/1 | 1024→1024 | 13×13×1024 | 9,439,232 | 1,594.9 |
| 14 | linear conv | 1×1/1 | 1024→125 | 13×13×125 | 128,125 | 21.6 |
| | **Total** | | | | **15.86 M** | **3.49 G** |

Cross-check: CoDeNet gives 3.49 G MACs for Tiny-YOLO at 416×416 [2006.08357#025.0], the
value computed here. Parameters are counted with the BN $\gamma,\beta$ (2 per channel),
without a convolution bias before BN (§4.2).

### 3.2 Tiny-YOLOv3 (VOC: 3 anchors × (5+20) = 75 filters per head)

Structure: 24 layers, of which 13 convolutions, 6 maxpools, 2 *route*, 1 *upsample* and
2 outputs; 3×3 and 1×1 kernels; 416×416×3 input; two scales, 13×13 and 26×26
[2023-zhai#004.1]. Layer 20 concatenates the outputs of layers 8 and 19
[2023-zhai#016.2]. With 20 classes, each head has 75 filters, following
$\text{filters} = 3\times(5+C)$ [2026-fata#013.0, 2026-fata#013.2]. The heads are linear and
all the other layers use the leaky ReLU [2026-fata#013.0, 2026-fata#013.1].

| # | Layer | k/s | C_in→C_out | Output | Parameters | MACs (M) |
|---|---|---|---|---|---|---|
| 0–11 | same as Tiny-YOLOv2 | | | 13×13×512 | 1,573,776 | 1,071.6 |
| 12 | conv | 3×3/1 | 512→1024 | 13×13×1024 | 4,720,640 | 797.4 |
| 13 | conv | 1×1/1 | 1024→256 | 13×13×256 | 262,656 | 44.3 |
| 14 | conv | 3×3/1 | 256→512 | 13×13×512 | 1,180,672 | 199.4 |
| 15 | linear conv | 1×1/1 | 512→75 | 13×13×75 | 38,475 | 6.5 |
| 16 | **yolo** (large anchors) | | | 13×13×75 | | |
| 17 | route 13 | | | 13×13×256 | | |
| 18 | conv | 1×1/1 | 256→128 | 13×13×128 | 33,024 | 5.5 |
| 19 | upsample ×2 | | | 26×26×128 | | |
| 20 | route 19, 8 (concatenation) | | | 26×26×384 | | |
| 21 | conv | 3×3/1 | 384→256 | 26×26×256 | 885,248 | 598.1 |
| 22 | linear conv | 1×1/1 | 256→75 | 26×26×75 | 19,275 | 13.0 |
| 23 | **yolo** (small anchors) | | | 26×26×75 | | |
| | **Total** | | | | **8.71 M** | **2.74 G** |

**Tiny-YOLOv3 anchors (outside the base, `yolov3-tiny.cfg`)**:
`10,14  23,27  37,58  81,82  135,169  344,319` (pixels at 416). The 13×13 head takes the
three largest, the 26×26 head the smallest. For a dataset of one's own, it is better to
recompute the anchors by k-means (§5.2), following the YOLOv2 method.

---

## 4. Building blocks: forward and backward passes

All the formulas of this section are **checked numerically** (§11). The papers of the base
describe the operations; most gradient derivations are **(outside the base)**. EF-Train, a
training on FPGA, splits each layer into forward propagation, backward propagation and
weight update (§9).

Notation: $\delta Y = \partial L/\partial Y$, the gradient of the loss with respect to the
output of the layer, received from the next layer.

### 4.1 2D convolution (stride $s$, padding $P$, $k\times k$ kernel)

**Forward:**

$$
Y[n,f,i,j] = b_f + \sum_{c=0}^{C_{in}-1}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} W[f,c,u,v]\;X_p[n,c,\,is+u,\,js+v]
$$

where $X_p$ is the input padded with $P$ zeros on each border. Output size:
$H_o=\lfloor (H+2P-k)/s\rfloor+1$.

**Backward:**

$$
\frac{\partial L}{\partial W[f,c,u,v]}=\sum_{n,i,j}\delta Y[n,f,i,j]\,X_p[n,c,is+u,js+v],\qquad
\frac{\partial L}{\partial b_f}=\sum_{n,i,j}\delta Y[n,f,i,j]
$$

$$
\delta X_p[n,c,is+u,js+v] \mathrel{+}= \sum_f \delta Y[n,f,i,j]\,W[f,c,u,v]
\quad(\text{then the padding is removed})
$$

EF-Train writes these same equations for training on FPGA. The backpropagation of the error
is a convolution of $\delta Y$ by the transposed and flipped weights, $W'_i$
[2202.10935#012.0, 2202.10935#012.1]. The weight gradient is the correlation of $\delta Y$
with the input [2202.10935#013.0]. The update subtracts the gradient times the learning rate
[2202.10935#013.0].

```python
def conv_forward(X, W, b, s=1, P=1):            # X:(N,C,H,W)  W:(F,C,k,k)
    Xp = pad_zeros(X, P); k = W.shape[2]
    Ho = (X.H + 2*P - k)//s + 1; Wo = (X.W + 2*P - k)//s + 1
    for n, f, i, j in product(N, F, Ho, Wo):
        acc = b[f]
        for c, u, v in product(C, k, k):
            acc += W[f,c,u,v] * Xp[n,c,i*s+u,j*s+v]
        Y[n,f,i,j] = acc
    return Y

def conv_backward(dY, Xp, W, s=1, P=1):
    dW = zeros_like(W); db = zeros(F); dXp = zeros_like(Xp)
    for n, f, i, j in product(N, F, Ho, Wo):
        g = dY[n,f,i,j]; db[f] += g
        for c, u, v in product(C, k, k):
            dW[f,c,u,v]           += g * Xp[n,c,i*s+u,j*s+v]
            dXp[n,c,i*s+u,j*s+v]  += g * W[f,c,u,v]
    return unpad(dXp, P), dW, db
```

**im2col (optimisation).** Each $C\cdot k\cdot k$ window is unfolded into a column: the
convolution becomes the matrix product $W_{[F\times Ck^2]}\cdot X_{col\,[Ck^2\times
H_oW_o]}$. 2023-zhai uses Im2col+GEMM and Winograd on FPGA [2023-zhai#004.0].

### 4.2 Batch normalization

**Forward (training)**, per channel $c$ and over $M=N\cdot H\cdot W$ values:

$$
\mu_c=\tfrac1M\sum x,\quad \sigma_c^2=\tfrac1M\sum(x-\mu_c)^2,\quad
\hat x=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}},\quad y=\gamma_c\hat x+\beta_c
$$

Running averages $\mu^{run}\leftarrow m\,\mu^{run}+(1-m)\mu_c$ (and likewise for
$\sigma^2$) are also kept and used at inference **(outside the base)**.

**Backward (outside the base, checked numerically):**

$$
\frac{\partial L}{\partial\gamma_c}=\sum\delta y\,\hat x,\qquad
\frac{\partial L}{\partial\beta_c}=\sum\delta y,\qquad
\delta x=\frac{\gamma_c}{M\sqrt{\sigma_c^2+\varepsilon}}\Big(M\,\delta y-\sum\delta y-\hat x\sum\delta y\,\hat x\Big)
$$

**Practical consequence.** The bias of a convolution followed by a BN has an exactly zero
gradient, because the subtraction of $\mu_c$ cancels it. The numerical checks give
$|\partial L/\partial b|\approx10^{-15}$. It is therefore left out, as Darknet does:
$\beta$ plays that role.

**Inference:** $y=\gamma_c(x-\mu^{run}_c)/\sqrt{\sigma^{2,run}_c+\varepsilon}+\beta_c$. Once
training is over, this affine transform is folded into the convolution weights (§9.1).

### 4.3 Leaky ReLU (slope $a=0.1$)

YOLOv1 defines $\varphi(x)=x$ if $x>0$ and $0.1x$ otherwise [1506.02640#004.0].

$$
\varphi(x)=\max(x,ax),\qquad \delta x=\delta y\cdot\begin{cases}1&x>0\\a&x\le0\end{cases}
$$

In hardware, the product by 0.1 is approximated with shifts, for example
$x/8+x/32+x/64\approx0.109x$, or applied in fixed point (§9).

### 4.4 2×2 max pooling, stride 2 or stride 1

**Forward:** $y[i,j]=\max_{u,v\in\{0,1\}}x[is+u,\,js+v]$; the position of the maximum
(*argmax*) is stored.

**Backward:** the gradient is routed to the argmax only:
$\delta x[\text{argmax}]\mathrel{+}=\delta y[i,j]$, and 0 elsewhere. EF-Train stores the
index of the maximum pixel (*Pooling Indexes*) during forward propagation for this step
[2202.10935#014.0].

**Stride 1 (layer 11, outside the base).** One row and one column are padded on the right
and at the bottom with $-\infty$, or by repeating the border, which gives the same maximum;
the 13×13 size is kept. It is the only "irregular" layer of the network, to be handled
separately in the accelerator (§10).

### 4.5 Upsample ×2 (nearest neighbour), route, sigmoid, exponential

| Block | Forward | Backward |
|---|---|---|
| upsample ×2 | $y[2i+u,2j+v]=x[i,j]$, $u,v\in\{0,1\}$ | $\delta x[i,j]=\sum_{u,v}\delta y[2i+u,2j+v]$ |
| route / concat | $y=[x_a; x_b]$ on the channel axis | $\delta y$ is split into $\delta x_a$, $\delta x_b$; a reused map **accumulates** its gradients |
| sigmoid | $\sigma(t)=1/(1+e^{-t})$ | $\sigma'(t)=\sigma(t)(1-\sigma(t))$ |
| exponential | $e^{t}$ | $e^{t}$ |

Stable sigmoid: if $t\ge0$, $1/(1+e^{-t})$; otherwise $e^{t}/(1+e^{t})$. In Tiny-YOLOv3,
layer 13 feeds layer 14 and *route* 17: its gradient is the sum of both contributions.

---

## 5. Target encoding

### 5.1 Assigning an object to a cell and an anchor

For a ground truth $(g_x,g_y,g_w,g_h)$ normalised to $[0,1]$ and a grid scale $S$:

1. **Cell**: $j=\lfloor g_xS\rfloor$, $i=\lfloor g_yS\rfloor$. It is the cell that contains
   the centre [1506.02640#002.0].
2. **Responsible anchor**: the one that maximises the shape IoU, centres aligned, between
   $(g_w,g_h)$ and $(p_w,p_h)$; one anchor per object [1804.02767#003.1]. In v1, it is the
   predictor whose current IoU with the truth is the highest [1506.02640#005.0]. With
   several scales, the choice is made among the 6 or 9 anchors; the selected scale is the
   one the winning anchor belongs to.
3. **Regression targets**, obtained by inverting the decoding [1804.02767#003.0]:

$$
x^*=g_xS-j\in[0,1),\quad y^*=g_yS-i,\quad t_w^*=\ln\frac{g_w}{p_w},\quad t_h^*=\ln\frac{g_h}{p_h}
$$

   with $p_w,p_h$ expressed in the same unit as $g_w,g_h$, for example as a fraction of the
   image.
4. **Masks** for each anchor of each cell: *obj* if the anchor is responsible; *ignore* if
   it is not responsible but its **predicted** box overlaps a ground truth with an
   IoU > 0.5; *noobj* otherwise [1804.02767#003.1].

### 5.2 Computing the anchors by k-means (YOLOv2 method)

```python
def kmeans_anchors(boxes_wh, k, iters=300):     # boxes_wh: list of (w, h)
    centroids = random_sample(boxes_wh, k)
    for _ in range(iters):
        # distance d = 1 - IoU(box, centroid), boxes centred at the origin [1612.08242#002.4]
        assign = [argmin([1 - iou_wh(b, c) for c in centroids]) for b in boxes_wh]
        new = [mean_wh([b for b, a in zip(boxes_wh, assign) if a == q]) for q in range(k)]
        if new == centroids: break
        centroids = new
    return sorted(centroids, key=lambda c: c[0]*c[1])

def iou_wh(a, b):                                 # IoU of two boxes with the same centre
    inter = min(a[0], b[0]) * min(a[1], b[1])
    return inter / (a[0]*a[1] + b[0]*b[1] - inter)
```

For Tiny-YOLOv3, $k=6$: three anchors per head. YOLOv3 simply splits the clusters evenly
across the scales [1804.02767#005.1].

### 5.3 IoU of two boxes (centre, width, height)

$$
\text{IoU}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|},\quad
|A\cap B|=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B))\cdot\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B))
$$

with $x_1=x-w/2$ and $x_2=x+w/2$. It is the ratio of the common area to the area of the
union [2025-hozhabr#056.0].

---

## 6. Loss function and gradients

### 6.1 YOLOv1 loss (historical reference)

Sum of squared errors in five terms [1506.02640#005.0]:

$$
\begin{aligned}
L=\;&\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}^{obj}_{ij}\big[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2\big]
+\lambda_{coord}\sum_{i,j}\mathbb{1}^{obj}_{ij}\Big[(\sqrt{w_i}-\sqrt{\hat w_i})^2+(\sqrt{h_i}-\sqrt{\hat h_i})^2\Big]\\
&+\sum_{i,j}\mathbb{1}^{obj}_{ij}(C_i-\hat C_i)^2+\lambda_{noobj}\sum_{i,j}\mathbb{1}^{noobj}_{ij}(C_i-\hat C_i)^2
+\sum_{i}\mathbb{1}^{obj}_{i}\sum_{c\in\text{classes}}(p_i(c)-\hat p_i(c))^2
\end{aligned}
$$

- $\lambda_{coord}=5$ and $\lambda_{noobj}=0.5$. Without them, the many empty cells
  overwhelm the gradient and training diverges [1506.02640#004.1, 1506.02640#004.2].
- $\sqrt{w}$ and $\sqrt{h}$ reduce the weight of errors on large boxes [1506.02640#004.1].
- Classification is only penalised if an object is present in the cell, and coordinates
  only for the responsible predictor [1506.02640#005.1].

### 6.2 YOLOv3 loss (the one to implement for Tiny-YOLOv3)

For each scale, with $\text{BCE}(p,y)=-[y\ln p+(1-y)\ln(1-p)]$:

$$
\begin{aligned}
L=\;&\lambda_{coord}\sum_{obj}\Big[(\sigma(t_x)-x^*)^2+(\sigma(t_y)-y^*)^2+(t_w-t_w^*)^2+(t_h-t_h^*)^2\Big]\\
&+\sum_{obj}\text{BCE}(\sigma(t_o),1)+\sum_{noobj}\text{BCE}(\sigma(t_o),0)
+\sum_{obj}\sum_{c=1}^{C}\text{BCE}(\sigma(t_c),\mathbb{1}[c=c^*])
\end{aligned}
$$

- The terms come from YOLOv3: sum of squares for the coordinates [1804.02767#003.0],
  logistic objectness with the "ignore" rule [1804.02767#003.1], multi-label BCE for the
  classes [1804.02767#004.0].
- *Ignore* anchors contribute to no term.
- **Coordinates (variant outside the base).** The paper puts the squared error on $t$, with
  the target $t_x^*=\sigma^{-1}(x^*)$, infinite if $x^*\in\{0,1\}$. The variant above puts
  it on $\sigma(t_x)$, which avoids inverting the sigmoid. It is the one of the reference
  implementation of §11.
- **Scale factor.** The coordinate term is usually multiplied by
  $\omega_{scale}=2-g_wg_h$, which increases the weight of small boxes. Gaussian YOLOv3
  describes this weight for the box loss of YOLOv3 [1904.04620#005.2]; its targets
  $x^G=x_G W-i$ and $w^G=\log(w_G\,IW/A^k_w)$ are those of §5.1 [1904.04620#005.1].

**Gradients with respect to the raw outputs (outside the base, checked numerically)**, for
a responsible anchor:

| Output | $\partial L/\partial t$ |
|---|---|
| $t_x$ | $2\lambda_{coord}(\sigma(t_x)-x^*)\,\sigma(t_x)(1-\sigma(t_x))$ |
| $t_y$ | same with $y$ |
| $t_w$ | $2\lambda_{coord}(t_w-t_w^*)$ |
| $t_h$ | $2\lambda_{coord}(t_h-t_h^*)$ |
| $t_o$ | $\sigma(t_o)-1$ if *obj*; $\sigma(t_o)$ if *noobj*; 0 if *ignore* |
| $t_c$ | $\sigma(t_c)-\mathbb{1}[c=c^*]$ |

Key point: the derivative of $\text{BCE}(\sigma(t),y)$ with respect to $t$ is simply
$\sigma(t)-y$, because the $\sigma(1-\sigma)$ terms cancel. This is the "truth minus
prediction" of YOLOv3 [1804.02767#003.0].

### 6.3 Other localisation losses (options)

- **Smooth L1 (Huber).** The usual choice is
  $L_{loc}=\sum_{i\in\{x,y,w,h\}}s(\hat i-i)$, with $s(x)=0.5x^2$ if $|x|<1$ and
  $|x|-0.5$ otherwise [2025-el-zeinaty#010.0]. The total loss of a detector is written
  $L=\beta L_{cls}+\alpha L_{loc}$ [2025-el-zeinaty#008.0].
- **GIoU, DIoU, CIoU.** These losses directly optimise the overlap between the predicted
  box and the truth; DIoU and CIoU add the distance between centres and the aspect ratio
  [2025-el-zeinaty#010.1]. Their formulas are not in the base. The usual definitions are
  the following **(outside the base)**:

$$
L_{GIoU}=1-\text{IoU}+\frac{|E\setminus(A\cup B)|}{|E|},\qquad
L_{DIoU}=1-\text{IoU}+\frac{\rho^2(\mathbf b,\mathbf b^{gt})}{c^2},\qquad
L_{CIoU}=L_{DIoU}+\alpha v
$$

  with $E$ the smallest enclosing box, $\rho$ the distance between centres, $c$ the
  diagonal of $E$, $v=\frac{4}{\pi^2}\big(\arctan\frac{w^{gt}}{h^{gt}}-\arctan\frac wh\big)^2$
  and $\alpha=v/((1-\text{IoU})+v)$. These losses replace the coordinate term of §6.2;
  their gradient is derived by propagating through $\min$, $\max$ and the decoding of §8.1.
- **Gaussian YOLOv3.** The coordinates become Gaussians (mean and variance), and the box
  loss becomes a negative log-likelihood; objectness and classes are unchanged
  [1904.04620#005.0]. The predicted uncertainty makes the model more robust to noisy
  annotations [1904.04620#005.3]. This is useful for a system that must estimate its own
  confidence.
- **Focal loss.** It brings nothing to YOLOv3 [1804.02767#009.0].

---

## 7. Training

### 7.1 Algorithm: backpropagation and SGD with momentum

```python
for epoch in range(E):
    for X, targets in batches(dataset, size=64):           # batch of 64 [1506.02640#005.1]
        if multiscale and it % 10 == 0:                     # [1612.08242#003.2]
            res = choice(range(320, 609, 32)); X = resize(X, res)
        out    = forward(X)                                 # keep the cache of each layer
        L, dO  = yolo_loss(out, targets, anchors)           # §6.2
        grads  = backward(dO)                               # §4, layers in reverse order
        lr     = schedule(it)
        for p in params:                                    # SGD + momentum + weight decay
            v[p] = mu * v[p] - lr * (grads[p] + wd * p)     # mu = 0.9, wd = 5e-4
            p   += v[p]
```

Published hyperparameters:

| | YOLOv1 [1506.02640#005.1] | YOLOv2 detection [1612.08242#005.3] |
|---|---|---|
| Batch | 64 | n/r |
| Momentum | 0.9 | 0.9 |
| Weight decay | 0.0005 | 0.0005 |
| Learning rate | ramp from $10^{-3}$ to $10^{-2}$, then $10^{-2}$ (75 epochs), $10^{-3}$ (30), $10^{-4}$ (30) | $10^{-3}$, divided by 10 at epochs 60 and 90 (160 epochs) |
| Regularisation | dropout 0.5, augmentation | BN, no dropout [1612.08242#002.0] |

- **Progressive ramp (*warm-up*).** Starting with a high rate makes the model diverge
  [1506.02640#005.1].
- **Pre-training.** The Darknet-19 classifier is trained on ImageNet with a rate of 0.1 and
  a polynomial decay of power 4 [1612.08242#005.2]. In practice, for an embedded Tiny, one
  starts from pre-trained weights and fine-tunes on the target dataset, as 2026-fata does
  [2026-fata#013.1].
- **Augmentation.** Random scaling and translations up to 20 % of the image size; exposure
  and saturation randomly changed by up to a factor of 1.5 in HSV [1506.02640#005.2]. The
  target boxes undergo the same geometric transforms.
- **Initialisation (outside the base).** Conv weights $\sim\mathcal N(0,\,2/(k^2C_{in}))$
  (He), $\gamma=1$, $\beta=0$, head bias at 0.

---

## 8. Inference: decoding, threshold, NMS, evaluation

### 8.1 Decoding

For each cell $(i,j)$, each anchor $a$ and each scale $S$, with anchors as a fraction of
the image:

$$
b_x=\frac{\sigma(t_x)+j}{S},\quad b_y=\frac{\sigma(t_y)+i}{S},\quad b_w=p_we^{t_w},\quad b_h=p_he^{t_h}
$$

2024-zhang computes the corners directly, for example
$x_1=[(\sigma(t_x)+c_x)-0.5\,p_we^{t_w}]\cdot\text{scale}$ [2024-zhang#014.0]. Score of
class $c$:

- YOLOv3: $\sigma(t_o)\cdot\sigma(t_c)$;
- YOLOv2: $\sigma(t_o)\cdot\text{softmax}(t)_c$. 2024-zhang computes this score with a
  stabilised softmax, by subtracting the maximum [2024-zhang#014.1].

### 8.2 Threshold and non-maximum suppression (NMS)

Scores under the threshold are filtered out first. The scores are then sorted, the best box
is kept and the boxes whose IoU with it exceeds a threshold are removed; this repeats until
none is left [2024-zhang#013.0]. According to YOLOv1, NMS adds 2 to 3 mAP points
[1506.02640#006.0].

```python
def nms(boxes, scores, iou_thr=0.45):           # per class; usual thresholds (outside the base)
    order = argsort(scores, descending=True); keep = []
    while order:
        best = order.pop(0); keep.append(best)
        order = [k for k in order if iou(boxes[best], boxes[k]) <= iou_thr]
    return keep

dets = []
for c in classes:
    cand = [(b, s[c]) for b, s in decoded if s[c] > conf_thr]   # conf_thr ≈ 0.25 to 0.5
    dets += [cand[k] for k in nms(*zip(*cand))]
```

### 8.3 Evaluation: IoU, precision, recall, mAP

- A detection is a true positive if its IoU with a ground truth not yet matched reaches the
  threshold, for example 0.5 [2025-hozhabr#056.0].
- For each class: sort the detections by score, accumulate TP and FP, then draw the
  precision/recall curve. The PASCAL VOC AP is the precision interpolated at 11 recall
  levels [2025-el-zeinaty#015.0]. The mAP is the mean of the class APs.

$$
\text{AP}_{11}=\frac1{11}\sum_{r\in\{0,0.1,\dots,1\}}\max_{\tilde r\ge r}P(\tilde r)
$$

---

## 9. Moving to embedded: BN folding, quantization, integer arithmetic

### 9.1 Folding the BN into the convolution

At inference, the BN is a per-channel affine transform. It is merged into the weights of the
convolution before it, which removes one pass over memory as well as the divisions and
square roots [2023-zhai#039.0, 2202.10935#007.1]. It is the first step of post-training
quantization (PTQ) [2607.13106#008.0]. With $s_f=\gamma_f/\sqrt{\sigma^2_f+\varepsilon}$:

$$
W'_{f}=s_f\,W_{f},\qquad b'_f=\beta_f+s_f\,(b_f-\mu_f)\quad(b_f=0\text{ if the conv has no bias})
$$

2023-zhai gives the same relation, $O_{norm}=\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}O+\big(\beta-\frac{\gamma\mu}{\sqrt{\sigma^2+\varepsilon}}\big)$
[2023-zhai#039.0]. The explicit form on $W'$ and $b'$ is a derivation **(outside the base,
checked numerically: max gap 5·10⁻¹⁵)**. EF-Train points out that during **training** the
folding is impossible, since $\mu$ and $\sigma^2$ depend on the batch; an embedded training
therefore keeps the BN, in full precision [2202.10935#007.1].

### 9.2 Quantization schemes

**Uniform affine quantization** [2024-campos#004.0]:

$$
q=\text{Clip}\big(\text{Round}(r/S)-Z,\;\alpha,\beta\big),\qquad r\approx S\,(q+Z)
$$

- $S$ is the scale and $Z$ the zero point; choosing the range $[\alpha,\beta]$ is
  *calibration*.
- For the weights, the **symmetric** ($Z=0$) **per output channel** variant is the
  simplest in hardware **(outside the base)**. A scale of its own for each filter absorbs
  the very different $s_f$ that come out of BN folding.

**Calibration.**
- PTQ: the scale and zero point are estimated on a subset of the data [2607.13106#008.0].
- 2026-fata compares clipping thresholds at the 90th, 95th and 99th percentiles and keeps
  the one that minimises the error between FP32 and INT8 weights [2026-fata#015.1].

**PTQ or QAT.**
- **QAT**: training uses "fake-quantized" weights (quantized then dequantized in forward
  propagation) [2026-fata#015.0]. 2026-fata fine-tunes this way with a learning rate of
  $10^{-4}$ [2026-fata#015.3]. TinyissimoYOLO runs 350 epochs in float, then 300 in QAT
  [2306.00001#004.1].
- In QAT, rounding has a zero derivative almost everywhere. It is crossed with the
  *straight-through* estimator: $\partial\,\text{round}(x)/\partial x:=1$ inside the range,
  0 outside **(outside the base)**.

**Bit widths used in the base for YOLO on FPGA:**

| Work | Model | Format | Remark |
|---|---|---|---|
| 2025-kim | YOLOv2 | INT16 | loss of about 0.2 % against FP32 [2025-kim#004.0]; one MAC per DSP [2025-kim#011.1] |
| 2023-zhai | pruned YOLOv3-tiny | INT16, dynamic fixed point | `benchmarks.csv`, Table 8 |
| 2024-zhang | YOLOv2/v3-Tiny | INT8, dynamic fixed point (BRECQ) | `benchmarks.csv`, Tables 3-4 |
| 2023-montgomerie-corcoran (SATAY) | YOLOv3-Tiny | W8A16 | `benchmarks.csv`, Table III |
| 2026-fata | YOLOv3-tiny | INT8 QAT, per-layer scale (weights) | [2026-fata#015.0] |
| 2024-yan | YOLOv5s | 4 bits; clipping $c$ learned and constrained to a power of 2 | lowers the hardware cost [2024-yan#006.0] |
| 2019-ding (REQ-YOLO) | Tiny-YOLOv2 | heterogeneous per layer: equidistant or "mixed powers-of-two" | chosen by ADMM [2019-ding#008.1, 2019-ding#009.4] |

**REQ-YOLO in detail.**
- A 6-bit weight is made of a sign bit and 5 magnitude bits: 3 "primary" bits and 2
  "secondary" ones. The multiplication becomes two shifts and an addition
  [2019-ding#008.1].
- The quantization problem is split into two sub-problems solved iteratively until
  convergence [2019-ding#009.1]: an SGD step on the penalised loss
  $\frac\rho2\lVert W-Z+U\rVert_F^2$, then a projection of $W+U$ onto the allowed levels.
- These levels are $\alpha\times\{-(M/2-1),\dots,M/2-1\}$ (equidistant) or
  $\alpha\times\{0,\pm2^0,\dots,\pm2^{M_1}\}$ (powers of 2) [2019-ding#009.4].

### 9.3 A conv + leaky layer in integer arithmetic (outside the base, checked numerically)

The papers of the base fix the format, but none gives the requantization chain. Here is a
standard derivation, checked in §11. $s_x$, $s_w$ and $s_y$ denote the input, weight and
output scales, all symmetric.

$$
\underbrace{acc_f=\sum q_w\,q_x}_{\text{int32}}+\underbrace{q_{b,f}}_{=\text{round}(b'_f/(s_xs_{w,f}))},\qquad
q_y=\text{clip}\Big(\big(acc_f\cdot M_{0,f}+2^{n-1}\big)\gg n,\;-127,\;127\Big),\qquad
M_{0,f}=\text{round}\Big(\frac{s_xs_{w,f}}{s_y}2^{n}\Big)
$$

```python
def conv_int8(qx, qW, qb, M0, n=31):              # qx,qW: int8; qb: int32; M0: int32 per channel
    acc = conv_forward(qx, qW, qb)                # int32 accumulator (exact)
    y = (acc * M0[:,None,None] + (1 << (n-1))) >> n   # requantization: 1 multiplication + 1 shift
    y = where(y > 0, y, (y * 13) >> 7)            # leaky 0.1 ≈ 13/128 = 0.1016
    return clip(y, -127, 127)
```

- **Accumulator width.** An int8×int8 product fits in 16 bits, and a sum of $k^2C_{in}$
  terms adds $\lceil\log_2(k^2C_{in})\rceil$ bits. The worst case of Tiny-YOLOv2 is
  layer 13: $9\times1024=9216$ terms, so 16 + 14 = 30 bits, which fit in 32 bits.
- **Leaky ReLU in hardware.** It needs a multiplication by a constant and a multiplexer for
  the positive inputs [2023-montgomerie-corcoran#012.0]. The $13/128$ slope is its "shift"
  version.
- **Changing the activation.** 2024-yan replaces SiLU with a PReLU with a trainable shift,
  which costs only an adder [2024-yan#005.0].
- **First layer.** It receives 8-bit pixels, while the rest of the network can be 4-bit.
  2024-yan splits it into two 4-bit convolutions, $\text{conv}_{odd}\times16+\text{conv}_{even}$
  [2024-yan#007.3].

### 9.4 Sigmoid, exponential and softmax in hardware

- **Lookup tables (LUT).** With $t$ quantized to 8 bits, the sigmoid and the exponential of
  the decoding are done with two 256-entry tables [2024-zhang#014.0]. Discretising the
  sigmoid is the most common solution; more steps give more precision but cost more area
  [2024-rech#007.4].
- **Precision of a LUT.** With an input step of 1/16, a 256-entry LUT gives a maximum error
  of 0.0075 on $\sigma$ (checked, §11).
- **Softmax in three stages.** The maximum is extracted, $e^{x_i-x_{max}}$ and their sum
  are computed, then the division is done in float [2024-zhang#014.1].
- **Thresholding without the sigmoid (outside the base).** $\sigma$ is monotonic, so
  $\sigma(t_o)>\theta\iff t_o>\ln\frac{\theta}{1-\theta}$. Boxes are filtered on the
  integer $t_o$ before any decoding, and the LUT is only used for the survivors.

---

## 10. The FPGA accelerator

### 10.1 Two architecture families

| | Streaming / dataflow | Single engine, layer by layer |
|---|---|---|
| Principle | one hardware block per layer, pipelined [2018-venieris#004.0] | a systolic array or a matrix unit runs the layers one after the other; software schedules them [2018-venieris#007.1] |
| Advantage | latency and throughput; in SATAY, all parameters stay on chip [2023-montgomerie-corcoran#006.0] | a single bitstream for all networks; fits on a small Zynq |
| Cost | one bitstream per network, long compilation [2018-venieris#004.0] | DDR traffic at every layer |
| YOLO examples | SATAY (YOLOv3-Tiny, W8A16) | 2025-kim, 2024-zhang, 2023-zhai, 2024-yan ("layer-by-layer") |

For a first reproduction on a Zynq or a KV260, the **single engine** is the simplest: a
single parameterised conv PE, reused by the 13 to 15 convolutions.

### 10.2 Loop nest, tiling and roofline (2015-zhang)

Tiling is required to fit a small part of the data on chip [2015-zhang#009.0]. Rows,
columns, input and output channels are tiled, and only $T_m$ and $T_n$ are unrolled; the
compute roof is then a function of $T_m$ and $T_n$ [2015-zhang#010.2].

```python
for to in range(0, M, Tm):               # output channels  (tile)
  for ti in range(0, N, Tn):             # input channels   (tile)
    for row in range(0, R, Tr):          # output rows      (tile)
      for col in range(0, C, Tc):        # output columns   (tile)
        load(in_buf[Tn][S*Tr+K-S][S*Tc+K-S]); load(w_buf[Tm][Tn][K][K])   # ping-pong
        for i in range(K):
          for j in range(K):
            for trr in range(Tr):
              for tcc in range(Tc):      # PIPELINE II=1
                for too in range(Tm):    # UNROLL: Tm × Tn multipliers (DSP)
                  for tii in range(Tn):  # UNROLL + adder tree
                    out_buf[too][trr][tcc] += w_buf[too][tii][i][j] * in_buf[tii][S*trr+i][S*tcc+j]
        if ti + Tn >= N: store(out_buf)  # bias + requant + leaky (+ fused maxpool)
```

- **Buffer sizes.** $B_{in}=T_n(ST_r+K-S)(ST_c+K-S)$, $B_w=T_mT_nK^2$ and
  $B_{out}=T_mT_rT_c$ must fit in BRAM. These formulas are read in 2015-zhang §4
  (equations 5 to 8), but the extracted text is degraded: reread `bin/pdb show
  2015-zhang#014.0`.
- **Roofline.** The attainable performance is
  $\min(\text{compute roof},\ \text{CTC}\times BW)$, where the CTC is the
  computation-to-communication ratio [2015-zhang]. The $(T_m,T_n,T_r,T_c)$ are enumerated
  and the best point under the bandwidth roof is kept.
- **Double buffering.** Ping-pong buffers overlap transfers with computation
  [2015-zhang#020.0]. The reference implementation duplicates 64 structures to unroll
  $T_m$ [2015-zhang#019.0].
- **Order of magnitude for Tiny-YOLO.** $T_m\times T_n$ sets the number of MACs per cycle:
  $T_m=T_n=16$ gives 256 MAC/cycle, so 3.49 G MAC / 256 ≈ 13.6 M cycles per image, about
  68 ms at 200 MHz at 100 % efficiency **(computation outside the base)**.

**Parallelism actually used:**
- 2024-zhang unrolls $N_m$ output channels, $N_r$ rows and $N_n$ MACs per kernel, and packs
  two kernels per DSP by multiplication *packing* [2024-zhang#011.0].
- 2025-kim uses a 3×3 conv PE ($T_r=3$, $T_c=15$, IFM reuse), a 1×1 conv PE ($T_r=1$,
  $T_c=13$) and a single maxpool PE [2025-kim#018.0]. The whole is driven by six
  controllers, including the DMAs [2025-kim#017.0].
- 2023-zhai organises the convolution in four stages, *line cache*, Im2col, GEMM and output,
  and fuses the convolution with the pooling to reduce three memory accesses to one
  [2023-zhai#034.0].
- SATAY generates the sliding window with *line buffers*, which only need
  $(K-1)\times W\times C$ words [2023-montgomerie-corcoran#011.0].

### 10.3 YOLO-specific layers

| Layer | Solution read in the base | To complete (outside the base) |
|---|---|---|
| maxpool 2×2/2 | comparators, fused at the conv output [2023-zhai#037.0, 2023-zhai#034.0]; dedicated PE [2025-kim#018.0] | — |
| **maxpool 2×2/1** | **not covered in the base** | same comparator on a sliding window; replicate the last row and the last column; 13×13 output |
| upsample ×2 | four copies of each pixel [2023-zhai#037.1] | or no copy at all: halve the read address of the next layer |
| route / concat | contiguous placement in memory of the outputs of layers 8 and 19 [2023-zhai#016.2]; stream multiplexer in SATAY [2023-montgomerie-corcoran#011.0]; on the ARM in 2025-kim [2025-kim#026.0] | write layer 19 right after layer 8 in the same buffer, which makes the concatenation free |
| decoding + NMS | on the ARM: 2025-kim [2025-kim#026.0], 2024-yan [2024-yan#008.0]; NMS on the host: 2023-zhai [2023-zhai#014.2]; in dedicated circuits: 2024-zhang [2024-zhang#013.1] | — |

**Hardware post-processing of 2024-zhang.**
- On an FPGA without a hard processor, only a softcore (MicroBlaze, NIOS) is available,
  hence dedicated circuits [2024-zhang#013.1].
- The NMS is light on resources: it does not sort, and a selected box is replaced by a box
  that overlaps it beyond the threshold with a better score [2024-zhang#014.2].
- The 845 candidate boxes (13×13×5) are processed in 0.38 ms [2024-zhang#020.3].

### 10.4 Published results (`bin/pdb bench`)

Power: *board* = measured on the board; *unspec.* = scope not stated. The figures are **not
directly comparable**: datasets, pruning, frequency and power scope differ. Check with
`bin/pdb bench --comparable` before any comparison.

| Work | Model | FPGA | Format | FPS | GOPS | W (scope) | LUT / DSP / BRAM | Source (conf.) |
|---|---|---|---|---|---|---|---|---|
| 2023-zhai | pruned YOLOv3-tiny, 1 module | Zynq XC7Z035 | INT16 | 91.65 | 67.91 | 12.51 (unspec.) | 38,228 / 144 / 132.5 | Table 8 (high) |
| 2023-zhai | pruned YOLOv3-tiny, 2 modules | Zynq XC7Z035 | INT16 | 168.72 | 124.01 | 15.18 (unspec.) | 91,108 / 294 / 263 | Table 9 (high) |
| 2024-zhang | YOLOv2-Tiny 416, batch 1 | Kintex-7 325T | INT8 | n/r (14.49 ms) | 406 | 12.3 (board) | 134,900 / 687 / 551 | Table 3, §5.3 (medium) |
| 2024-zhang | YOLOv3-Tiny 416, batch 1 | Kintex-7 325T | INT8 | n/r (15.2 ms) | 401 | 12.3 (board) | 136,500 / 687 / 553 | Tables 4-5 (high) |
| 2023-montgomerie-corcoran | YOLOv3-Tiny 416 | VCU110 | W8A16 | n/r | 418.9 | 15.4 (unspec.) | see Table III | Table III (medium) |
| 2019-ding | Tiny-YOLOv2, heterogeneous ADMM | Virtex-7 690T | FFT + powers of 2 | 314.2 | n/r | 21 (unspec.) | n/r | Tables 1-2 (high) |
| 2026-fata | YOLOv3-tiny pruned by 70 % | Kria KV260 (DPU) | INT8 QAT | 24.3 | 60.18 | 2.13 (unspec.) | n/r | Tables 7-10 (medium) |

- **2024-zhang.** Chip power estimated by Vivado: 7.8 W, against 12.3 W measured on the
  board.
- **2026-fata.** The work relies on a preconfigured Vitis AI DPU, not on a hand-designed
  accelerator.
- **2025-kim (full YOLOv2, Zybo Z7-20, INT16).** A slow working prototype, about 12 s per
  image: it is not a performance reference.

### 10.5 Roadmap of a reproduction (synthesis, outside the base)

1. **Float reference model** (§3 to §8) in pure Python or NumPy, validated by gradient
   checks (§11). Trained or taken from Darknet weights, it gives the reference mAP.
2. **Bit-exact integer model** (§9): BN folding, per-channel scales, M0/shift
   requantization, leaky 13/128 and LUT. The mAP loss against the float model is measured;
   if it is too large, move to QAT.
3. **Bit-exact C++ model** of the engine: tiled loop nest of §10.2, same inputs and outputs
   as the integer model. It serves as the *golden model*.
4. **HLS or RTL** of the conv PE with fused maxpool, then of the upsample and route units
   (addressing), with DMA and ping-pong; layer-by-layer comparison with the *golden model*.
5. **Post-processing** on the ARM first, as 2025-kim and 2024-yan do, then possibly in
   hardware, as 2024-zhang does.
6. **Measurements**: FPS, latency, power with its scope stated (chip or board) and
   resources, in the format of `benchmarks.csv`.

---

## 11. Checks and tests to reproduce

Reference implementation in NumPy, used only for the tests. The scripts of this check
(`gradcheck.py`, `quant.py`) stayed in the temporary session directory; they are not a
deliverable of the base.

| Test | Method | Result obtained |
|---|---|---|
| Gradients of conv, BN ($\gamma,\beta$), leaky, maxpool s2 and s1, route (accumulation), 1×1 head conv, full YOLOv3 loss (§6.2) | centred finite differences, $h=10^{-6}$, on a mini-network conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1 with 2 images and 3 objects | relative error ≤ 2.2·10⁻⁹ for all parameters |
| Bias before BN | analytical gradient | ≈ 10⁻¹⁵ (zero, §4.2) |
| Upsample ×2 | finite differences | 1.3·10⁻⁹ |
| BN folding (§9.1) | comparison with conv + BN | 5·10⁻¹⁵ |
| INT8 conv + M0/shift requantization + leaky 13/128 (§9.3) | comparison with the float output of an 8→16 channel layer | max error 1.2 % of the dynamic range; an 18-bit accumulator is enough here |
| 256-entry sigmoid LUT, step 1/16 | 1,000 draws | max error 0.0075 |
| Parameters and MACs (§3) | counting script | Tiny-YOLOv2: 3.49 GMAC, consistent with [2006.08357#025.0] |

**Recommended tests for a full implementation:**
- check the gradients of every new layer;
- overfit a single image: the loss must tend to 0 and the boxes must coincide;
- compare float, integer and hardware layer by layer;
- measure the VOC mAP at the three stages (float, integer, FPGA).

---

## 12. References (papers of the base used)

| short id | Title | Authors | Year | Venue | Type | Link |
|---|---|---|---|---|---|---|
| `1506.02640` | You Only Look Once: Unified, Real-Time Object Detection | Redmon et al. | 2016 | CVPR 2016 | conference | https://doi.org/10.1109/CVPR.2016.91 |
| `1612.08242` | YOLO9000: Better, Faster, Stronger | Redmon & Farhadi | 2017 | CVPR 2017 | conference | https://arxiv.org/abs/1612.08242 |
| `1804.02767` | YOLOv3: An Incremental Improvement | Redmon & Farhadi | 2018 | arXiv | **preprint** | https://arxiv.org/abs/1804.02767 |
| `1904.04620` | Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving | Choi et al. | 2019 | arXiv | **preprint** | https://arxiv.org/abs/1904.04620 |
| `2019-ding` | REQ-YOLO: A Resource-Aware, Efficient Quantization Framework for Object Detection on FPGAs | Ding et al. | 2019 | FPGA '19 (ACM/SIGDA) | conference | https://doi.org/10.1145/3289602.3293904 |
| `2023-zhai` | FPGA-Based Vehicle Detection and Tracking Accelerator | Zhai et al. | 2023 | Sensors 23(4) | journal | https://doi.org/10.3390/s23042208 |
| `2024-zhang` | End-to-end acceleration of the YOLO object detection framework on FPGA-only devices | Zhang et al. | 2024 | Neural Computing and Applications 36(3) | journal | https://doi.org/10.1007/s00521-023-09078-8 |
| `2025-kim` | Design and Implementation of a YOLOv2 Accelerator on a Zynq-7000 FPGA | Kim & Kim | 2025 | Sensors | journal | https://doi.org/10.3390/s25206359 |
| `2026-fata` | Low-Cost FPGA-Enhanced CNN Accelerator for Real-Time YOLO Object Detection and Classification | Fata & Elmannai | 2026 | IEEE Access 14 | journal | https://doi.org/10.1109/ACCESS.2026.3669451 |
| `2024-yan` | An FPGA-Based YOLOv5 Accelerator for Real-Time Industrial Vision Applications | Yan et al. | 2024 | Micromachines 15(9) | journal | https://doi.org/10.3390/mi15091164 |
| `2306.00001` | TinyissimoYOLO: A Quantized, Low-Memory Footprint, TinyML Object Detection Network for Low Power Microcontrollers | Moosmann et al. | 2023 | IEEE AICAS 2023 | conference | https://doi.org/10.1109/AICAS57966.2023.10168657 |
| `2023-montgomerie-corcoran` | SATAY: A Streaming Architecture Toolflow for Accelerating YOLO Models on FPGA Devices | Montgomerie-Corcoran et al. | 2023 | ICFPT 2023 | conference | https://doi.org/10.1109/ICFPT59805.2023.00025 |
| `2015-zhang` | Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks | Zhang et al. | 2015 | FPGA '15 (ACM/SIGDA) | conference | https://doi.org/10.1145/2684746.2689060 |
| `2018-venieris` | Toolflows for Mapping Convolutional Neural Networks on FPGAs: A Survey and Future Directions | Venieris et al. | 2018 | ACM Computing Surveys 51(3) | journal | https://doi.org/10.1145/3186332 |
| `2202.10935` | EF-Train: Enable Efficient On-device CNN Training on FPGA Through Data Reshaping for Online Adaptation or Personalization | Tang et al. | 2022 | arXiv | **preprint** | https://arxiv.org/abs/2202.10935 |
| `2024-campos` | End-to-end codesign of Hessian-aware quantized neural networks for FPGAs | Campos et al. | 2024 | ACM TRETS 17(3) | journal | https://doi.org/10.1145/3662000 |
| `2607.13106` | No Attention, No Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA | Karki et al. | 2026 | arXiv | **preprint** | https://arxiv.org/abs/2607.13106 |
| `2025-el-zeinaty` | Designing Object Detection Models for TinyML: Foundations, Comparative Analysis, Challenges, and Emerging Solutions | El Zeinaty et al. | 2025 | ACM Computing Surveys 58(2) | journal | https://doi.org/10.1145/3744339 |
| `2025-hozhabr` | A Survey on Real-Time Object Detection on FPGAs | Hozhabr & Giorgi | 2025 | IEEE Access 13 | journal | https://doi.org/10.1109/ACCESS.2025.3544515 |
| `2024-rech` | Artificial Neural Networks for Space and Safety-Critical Applications: Reliability Issues and Potential Solutions | Rech | 2024 | IEEE Trans. Nuclear Science 71(4) | journal | https://doi.org/10.1109/TNS.2024.3349956 |
| `2023-shuvo` | Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review | Shuvo et al. | 2023 | Proceedings of the IEEE 111(1) | journal | https://doi.org/10.1109/JPROC.2022.3226481 |
| `2006.08357` | CoDeNet: Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs | Dong et al. | 2021 | FPGA '21 | conference | https://doi.org/10.5281/zenodo.4341394 |

YOLOv3 is a technical report never published at a conference, but it is the de facto
reference. The papers marked **preprint** have not been peer reviewed.

### What the base does not cover (gaps)

- **2×2 maxpool with stride 1** (Tiny-YOLOv2/v3): no paper describes its hardware handling.
- **Darknet `.cfg` files** (exact channels, Tiny-YOLOv3 anchors): absent; the values of §3
  are taken from them, outside the base.
- **GIoU, DIoU and CIoU formulas**, the integer requantization chain and the BN
  derivatives: derived here, not read in the base.
- **Training YOLO directly on FPGA**: EF-Train trains classification CNNs. For the thesis,
  a YOLO that adapts online on FPGA (continual learning) remains an open direction, to be
  crossed with the continual learning axis of the base.

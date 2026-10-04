# Implémenter un YOLO embarqué de zéro

*Base `fpga`, 04/10/2026. Fil conducteur : **Tiny-YOLOv2** et **Tiny-YOLOv3**, du réseau à
l'accélérateur FPGA, sans bibliothèque d'apprentissage.*

Ce document décrit tout ce qu'il faut pour réécrire à la main un détecteur YOLO léger et
l'embarquer : l'architecture couche par couche, les fonctions mathématiques, les passes avant
et arrière de chaque brique, la fonction de perte et son gradient, l'entraînement,
l'inférence (décodage, NMS, mAP), puis la fusion BN, la quantification entière et
l'organisation d'un accélérateur FPGA.

## 0. Conventions de sourçage

- `[1506.02640#004.1]` : passage de la base, cité par son `chunk_id` et vérifié avec
  `bin/pdb verify`. On le relit avec `bin/pdb show <chunk_id>`.
- **(hors base)** : ce que les articles de la base ne décrivent pas. Il s'agit soit d'une
  dérivation mathématique standard (gradients), soit d'une convention de l'implémentation de
  référence Darknet (fichiers `.cfg`), soit d'un choix d'ingénierie. Aucune de ces
  affirmations n'est attribuée à un article.
- **(vérifié numériquement)** : formule contrôlée par différences finies dans une
  implémentation NumPy de référence (§11).

Articles fondateurs : YOLOv1 `arxiv-1506.02640`, YOLO9000/YOLOv2 `arxiv-1612.08242`, YOLOv3
`arxiv-1804.02767`. Implémentations embarquées : REQ-YOLO `2019-ding`, `2023-zhai`
(YOLOv3-tiny, HLS), `2024-zhang` (YOLO sur FPGA seul, post-traitement matériel), `2025-kim`
(YOLOv2 sur Zynq-7000), `2026-fata` (YOLOv3-tiny), `2024-yan` (YOLOv5 bas bit),
TinyissimoYOLO `arxiv-2306.00001`. La liste complète est au §12.

### Notations

| Symbole | Sens |
|---|---|
| $S$ | côté de la grille de sortie (13 ou 26 pour une entrée de 416) |
| $A$ (ou $B$ en v1) | nombre de boîtes prédites par cellule (ancres) |
| $C$ | nombre de classes (20 pour VOC, 80 pour COCO) |
| $c_x, c_y$ | indices (colonne, ligne) de la cellule |
| $p_w, p_h$ | largeur et hauteur de l'ancre (*prior*) |
| $t_x, t_y, t_w, t_h, t_o$ | sorties brutes du réseau pour une boîte |
| $\sigma(t) = 1/(1+e^{-t})$ | sigmoïde (logistique) |
| $\mathbb{1}^{obj}_{ij}$ | 1 si la boîte $j$ de la cellule $i$ est « responsable » d'un objet |

Tenseurs au format `(N, C, H, W)` : lot, canaux, hauteur, largeur.

---

## 1. Vue d'ensemble de la chaîne

```
image RGB ──► prétraitement ──► backbone conv ──► tête(s) 1×1 ──► décodage ──► seuil ──► NMS ──► boîtes
 (H×W×3)      416×416, /255     (conv-BN-leaky,    S×S×A(5+C)      σ, exp       score     IoU
                                 maxpool)
```

- **Détection = régression unique.** YOLO découpe l'image en une grille $S\times S$ ; la
  cellule qui contient le centre d'un objet est chargée de le détecter
  [1506.02640#002.0]. Le réseau est appliqué une seule fois par image.
- **Entrée.** 416×416 à partir de YOLOv2. Le réseau sous-échantillonne d'un facteur 32, d'où
  une carte de sortie de 13×13 [1612.08242#002.2]. YOLOv3-tiny ajoute une seconde sortie en
  26×26 [2023-zhai#004.1].
- **Prétraitement (hors base).** Redimensionner en conservant le rapport d'aspect et
  compléter par du gris (*letterbox*, convention Darknet), puis diviser les pixels par 255
  pour les ramener dans $[0,1]$.

---

## 2. Le principe YOLO : ce que chaque version apporte

### 2.1 YOLOv1 : grille et régression directe

- Chaque cellule prédit $B$ boîtes. Chaque boîte porte 5 nombres $(x, y, w, h,
  \text{confiance})$ ; la cellule porte en plus $C$ probabilités conditionnelles
  $\Pr(\text{Class}_i\mid\text{Object})$, une seule fois quelle que soit la valeur de $B$
  [1506.02640#002.0].
- La confiance vise $\Pr(\text{Object})\cdot \text{IOU}^{truth}_{pred}$ [1506.02640#002.0].
  En test, le score spécifique à une classe vaut
  $\Pr(\text{Class}_i\mid\text{Obj})\cdot\Pr(\text{Obj})\cdot\text{IOU} = \Pr(\text{Class}_i)\cdot\text{IOU}$
  [2019-ding#003.1].
- Sur VOC, $S=7$, $B=2$ et $C=20$ : la sortie est un tenseur 7×7×30 [1506.02640#002.1].
  Dans le cas général, la taille est $S\times S\times(B\cdot5+C)$ [2019-ding#003.1].
- $(x,y)$ sont relatifs à la cellule et $(w,h)$ à l'image entière, tous dans $[0,1]$. La
  dernière couche est linéaire ; toutes les autres utilisent la *leaky ReLU* de pente 0,1
  [1506.02640#004.0].
- Défaut de cette version : des couches entièrement connectées en sortie et de nombreuses
  erreurs de localisation. YOLOv2 les corrige.

### 2.2 YOLOv2 (YOLO9000) : BN, ancres, prédiction directe bornée

- **Batch normalization** sur toutes les convolutions : +2 % de mAP et suppression du
  dropout [1612.08242#002.0].
- **Ancres.** Les couches entièrement connectées sont retirées ; chaque position de la carte
  prédit des décalages par rapport à des boîtes a priori. L'entrée passe à 416 pour obtenir
  une grille impaire, qui a une cellule centrale [1612.08242#002.2].
- **Ancres par k-means** sur les boîtes d'entraînement, avec la distance
  $d(\text{box},\text{centroid}) = 1-\text{IOU}(\text{box},\text{centroid})$ ; $k=5$ est un
  compromis [1612.08242#002.4].
- **Prédiction directe de la position**, la formule centrale de YOLO [1612.08242#003.1] :

$$
b_x = \sigma(t_x)+c_x,\qquad b_y=\sigma(t_y)+c_y,\qquad b_w=p_w e^{t_w},\qquad b_h=p_h e^{t_h},\qquad
\Pr(\text{object})\cdot\text{IOU}(b,\text{object})=\sigma(t_o)
$$

  La sigmoïde maintient le centre dans sa cellule et l'exponentielle garantit $b_w,b_h>0$.
- **Passthrough.** La carte 26×26×512 est réorganisée en 13×13×2048 (*space-to-depth*) puis
  concaténée aux caractéristiques 13×13 [1612.08242#003.2].
- **Entraînement multi-échelle.** Tous les 10 lots, une nouvelle taille d'entrée est tirée
  parmi les multiples de 32 entre 320 et 608 [1612.08242#003.2].
- **Tête VOC.** 5 boîtes × (5 + 20) = 125 filtres dans la convolution 1×1 finale
  [1612.08242#005.3].

### 2.3 YOLOv3 : objectness logistique, multi-label, multi-échelle

- Même décodage que YOLOv2. Les coordonnées sont apprises par somme des carrés, et le
  gradient vaut « la vérité terrain moins la prédiction », $\hat t^* - t^*$ [1804.02767#003.0].
- **Objectness.** Elle est apprise par régression logistique. Sa cible vaut 1 pour l'ancre
  qui recouvre le mieux un objet. Une ancre qui n'est pas la meilleure mais dont l'IoU avec un
  objet dépasse 0,5 est **ignorée**. Une ancre non assignée ne reçoit que la perte
  d'objectness [1804.02767#003.1].
- **Classes.** Des classifieurs logistiques indépendants entraînés par entropie croisée
  binaire remplacent le softmax [1804.02767#004.0].
- **Trois échelles.** Chaque sortie est $N\times N\times[3\cdot(4+1+80)]$ sur COCO
  [1804.02767#005.0]. Une carte plus profonde est sur-échantillonnée ×2 puis concaténée à une
  carte plus ancienne [1804.02767#005.0].
- **9 ancres COCO** (en pixels pour 416) : (10×13), (16×30), (33×23), (30×61), (62×45),
  (59×119), (116×90), (156×198), (373×326) [1804.02767#005.1].
- Essai abandonné : la *focal loss* fait perdre environ 2 points de mAP [1804.02767#009.0].

---

## 3. Architectures Tiny couche par couche

Les articles de la base décrivent la structure de ces réseaux ; les nombres de canaux exacts
et le maxpool de stride 1 viennent des fichiers `yolov2-tiny-voc.cfg` et `yolov3-tiny.cfg`
de Darknet **(hors base)**. Les paramètres et les MACs ont été calculés par un script (§11).
Une couche « conv » comprend convolution, BN et leaky ReLU 0,1, sauf la convolution de
sortie, qui est linéaire avec un biais.

### 3.1 Tiny-YOLOv2 (VOC, entrée 416×416×3)

REQ-YOLO décrit ce réseau ainsi : 9 couches CONV ; des couches 1 à 8, une convolution 3×3
(stride 1, padding 1) suivie d'un maxpool 2×2 de stride 2 ; une convolution 1×1 finale qui
réduit l'espace des caractéristiques [2019-ding#003.1, 2019-ding#003.2].

| # | Couche | k/s | C_in→C_out | Sortie | Paramètres | MACs (M) |
|---|---|---|---|---|---|---|
| 0 | conv | 3×3/1 | 3→16 | 416×416×16 | 464 | 74,8 |
| 1 | maxpool | 2×2/2 | | 208×208×16 | | |
| 2 | conv | 3×3/1 | 16→32 | 208×208×32 | 4 672 | 199,4 |
| 3 | maxpool | 2×2/2 | | 104×104×32 | | |
| 4 | conv | 3×3/1 | 32→64 | 104×104×64 | 18 560 | 199,4 |
| 5 | maxpool | 2×2/2 | | 52×52×64 | | |
| 6 | conv | 3×3/1 | 64→128 | 52×52×128 | 73 984 | 199,4 |
| 7 | maxpool | 2×2/2 | | 26×26×128 | | |
| 8 | conv | 3×3/1 | 128→256 | 26×26×256 | 295 424 | 199,4 |
| 9 | maxpool | 2×2/2 | | 13×13×256 | | |
| 10 | conv | 3×3/1 | 256→512 | 13×13×512 | 1 180 672 | 199,4 |
| 11 | **maxpool** | **2×2/1** | | 13×13×512 | | |
| 12 | conv | 3×3/1 | 512→1024 | 13×13×1024 | 4 720 640 | 797,4 |
| 13 | conv | 3×3/1 | 1024→1024 | 13×13×1024 | 9 439 232 | 1 594,9 |
| 14 | conv linéaire | 1×1/1 | 1024→125 | 13×13×125 | 128 125 | 21,6 |
| | **Total** | | | | **15,86 M** | **3,49 G** |

Contrôle croisé : CoDeNet donne 3,49 G MACs pour Tiny-YOLO en 416×416 [2006.08357#025.0],
soit la valeur calculée ici. Paramètres comptés avec $\gamma,\beta$ de BN (2 par canal),
sans biais de convolution avant BN (§4.2).

### 3.2 Tiny-YOLOv3 (VOC : 3 ancres × (5+20) = 75 filtres par tête)

Structure : 24 couches, dont 13 convolutions, 6 maxpool, 2 *route*, 1 *upsample* et
2 sorties ; noyaux 3×3 et 1×1 ; entrée 416×416×3 ; deux échelles, 13×13 et 26×26
[2023-zhai#004.1]. La couche 20 concatène les sorties des couches 8 et 19
[2023-zhai#016.2]. Avec 20 classes, chaque tête a 75 filtres, selon
$\text{filtres} = 3\times(5+C)$ [2026-fata#013.0, 2026-fata#013.2]. Les têtes sont
linéaires et toutes les autres couches utilisent la leaky ReLU [2026-fata#013.0,
2026-fata#013.1].

| # | Couche | k/s | C_in→C_out | Sortie | Paramètres | MACs (M) |
|---|---|---|---|---|---|---|
| 0–11 | identiques à Tiny-YOLOv2 | | | 13×13×512 | 1 573 776 | 1 071,6 |
| 12 | conv | 3×3/1 | 512→1024 | 13×13×1024 | 4 720 640 | 797,4 |
| 13 | conv | 1×1/1 | 1024→256 | 13×13×256 | 262 656 | 44,3 |
| 14 | conv | 3×3/1 | 256→512 | 13×13×512 | 1 180 672 | 199,4 |
| 15 | conv linéaire | 1×1/1 | 512→75 | 13×13×75 | 38 475 | 6,5 |
| 16 | **yolo** (grandes ancres) | | | 13×13×75 | | |
| 17 | route 13 | | | 13×13×256 | | |
| 18 | conv | 1×1/1 | 256→128 | 13×13×128 | 33 024 | 5,5 |
| 19 | upsample ×2 | | | 26×26×128 | | |
| 20 | route 19, 8 (concaténation) | | | 26×26×384 | | |
| 21 | conv | 3×3/1 | 384→256 | 26×26×256 | 885 248 | 598,1 |
| 22 | conv linéaire | 1×1/1 | 256→75 | 26×26×75 | 19 275 | 13,0 |
| 23 | **yolo** (petites ancres) | | | 26×26×75 | | |
| | **Total** | | | | **8,71 M** | **2,74 G** |

**Ancres de Tiny-YOLOv3 (hors base, `yolov3-tiny.cfg`)** :
`10,14  23,27  37,58  81,82  135,169  344,319` (pixels à 416). La tête 13×13 prend les trois
plus grandes, la tête 26×26 les plus petites. Pour un jeu de données propre, il vaut mieux
recalculer les ancres par k-means (§5.2), selon la méthode de YOLOv2.

---

## 4. Briques : passes avant et arrière

Toutes les formules de cette section sont **vérifiées numériquement** (§11). Les articles de
la base décrivent les opérations ; la plupart des dérivations des gradients sont
**(hors base)**. EF-Train, un entraînement sur FPGA, décompose chaque couche en propagation
avant, propagation arrière et mise à jour des poids (§9).

Notation : $\delta Y = \partial L/\partial Y$, gradient de la perte par rapport à la sortie
de la couche, reçu de la couche suivante.

### 4.1 Convolution 2D (stride $s$, padding $P$, noyau $k\times k$)

**Avant :**

$$
Y[n,f,i,j] = b_f + \sum_{c=0}^{C_{in}-1}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} W[f,c,u,v]\;X_p[n,c,\,is+u,\,js+v]
$$

où $X_p$ est l'entrée complétée de $P$ zéros sur chaque bord. Taille de sortie :
$H_o=\lfloor (H+2P-k)/s\rfloor+1$.

**Arrière :**

$$
\frac{\partial L}{\partial W[f,c,u,v]}=\sum_{n,i,j}\delta Y[n,f,i,j]\,X_p[n,c,is+u,js+v],\qquad
\frac{\partial L}{\partial b_f}=\sum_{n,i,j}\delta Y[n,f,i,j]
$$

$$
\delta X_p[n,c,is+u,js+v] \mathrel{+}= \sum_f \delta Y[n,f,i,j]\,W[f,c,u,v]
\quad(\text{puis on retire le padding})
$$

EF-Train écrit ces mêmes équations pour un entraînement sur FPGA. La rétropropagation de
l'erreur est une convolution de $\delta Y$ par les poids transposés et retournés,
$W'_i$ [2202.10935#012.0, 2202.10935#012.1]. Le gradient des poids est la corrélation de
$\delta Y$ avec l'entrée [2202.10935#013.0]. La mise à jour soustrait le produit du
gradient par le taux d'apprentissage [2202.10935#013.0].

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

**im2col (optimisation).** On déplie chaque fenêtre $C\cdot k\cdot k$ en colonne : la
convolution devient le produit matriciel $W_{[F\times Ck^2]}\cdot X_{col\,[Ck^2\times
H_oW_o]}$. 2023-zhai utilise Im2col+GEMM et Winograd sur FPGA [2023-zhai#004.0].

### 4.2 Batch normalization

**Avant (entraînement)**, par canal $c$ et sur $M=N\cdot H\cdot W$ valeurs :

$$
\mu_c=\tfrac1M\sum x,\quad \sigma_c^2=\tfrac1M\sum(x-\mu_c)^2,\quad
\hat x=\frac{x-\mu_c}{\sqrt{\sigma_c^2+\varepsilon}},\quad y=\gamma_c\hat x+\beta_c
$$

On tient aussi des moyennes glissantes $\mu^{run}\leftarrow m\,\mu^{run}+(1-m)\mu_c$ (et de
même pour $\sigma^2$), qui servent en inférence **(hors base)**.

**Arrière (hors base, vérifié numériquement) :**

$$
\frac{\partial L}{\partial\gamma_c}=\sum\delta y\,\hat x,\qquad
\frac{\partial L}{\partial\beta_c}=\sum\delta y,\qquad
\delta x=\frac{\gamma_c}{M\sqrt{\sigma_c^2+\varepsilon}}\Big(M\,\delta y-\sum\delta y-\hat x\sum\delta y\,\hat x\Big)
$$

**Conséquence pratique.** Le biais d'une convolution suivie d'une BN a un gradient
exactement nul, car la soustraction de $\mu_c$ l'annule. Les vérifications numériques
donnent $|\partial L/\partial b|\approx10^{-15}$. On l'omet donc, comme le fait Darknet :
$\beta$ joue ce rôle.

**Inférence :** $y=\gamma_c(x-\mu^{run}_c)/\sqrt{\sigma^{2,run}_c+\varepsilon}+\beta_c$. Une
fois l'entraînement terminé, cette transformation affine se fusionne dans les poids de la
convolution (§9.1).

### 4.3 Leaky ReLU (pente $a=0{,}1$)

YOLOv1 définit $\varphi(x)=x$ si $x>0$ et $0{,}1x$ sinon [1506.02640#004.0].

$$
\varphi(x)=\max(x,ax),\qquad \delta x=\delta y\cdot\begin{cases}1&x>0\\a&x\le0\end{cases}
$$

En matériel, le produit par 0,1 s'approche par décalages, par exemple
$x/8+x/32+x/64\approx0{,}109x$, ou s'applique en point fixe (§9).

### 4.4 Max pooling 2×2, stride 2 ou stride 1

**Avant :** $y[i,j]=\max_{u,v\in\{0,1\}}x[is+u,\,js+v]$ ; on mémorise la position du
maximum (*argmax*).

**Arrière :** le gradient est routé uniquement vers l'argmax :
$\delta x[\text{argmax}]\mathrel{+}=\delta y[i,j]$, et 0 ailleurs. EF-Train mémorise en
propagation avant l'indice du pixel maximal (*Pooling Indexes*) pour cette étape
[2202.10935#014.0].

**Stride 1 (couche 11, hors base).** On complète d'une ligne et d'une colonne à droite et
en bas avec $-\infty$, ou en répétant le bord, ce qui donne le même maximum ; la taille
13×13 est conservée. C'est la seule couche « irrégulière » du réseau, à traiter à part dans
l'accélérateur (§10).

### 4.5 Upsample ×2 (plus proche voisin), route, sigmoïde, exponentielle

| Brique | Avant | Arrière |
|---|---|---|
| upsample ×2 | $y[2i+u,2j+v]=x[i,j]$, $u,v\in\{0,1\}$ | $\delta x[i,j]=\sum_{u,v}\delta y[2i+u,2j+v]$ |
| route / concat | $y=[x_a; x_b]$ sur l'axe des canaux | on découpe $\delta y$ en $\delta x_a$, $\delta x_b$ ; une carte réutilisée **cumule** ses gradients |
| sigmoïde | $\sigma(t)=1/(1+e^{-t})$ | $\sigma'(t)=\sigma(t)(1-\sigma(t))$ |
| exponentielle | $e^{t}$ | $e^{t}$ |

Calcul stable de la sigmoïde : si $t\ge0$, $1/(1+e^{-t})$ ; sinon $e^{t}/(1+e^{t})$.
Dans Tiny-YOLOv3, la couche 13 alimente la couche 14 et la *route* 17 : son gradient est la
somme des deux contributions.

---

## 5. Encodage des cibles

### 5.1 Assignation d'un objet à une cellule et à une ancre

Pour une vérité terrain $(g_x,g_y,g_w,g_h)$ normalisée dans $[0,1]$ et une échelle de grille
$S$ :

1. **Cellule** : $j=\lfloor g_xS\rfloor$, $i=\lfloor g_yS\rfloor$. C'est la cellule qui
   contient le centre [1506.02640#002.0].
2. **Ancre responsable** : celle qui maximise l'IoU de forme, centres alignés, entre
   $(g_w,g_h)$ et $(p_w,p_h)$ ; une seule ancre par objet [1804.02767#003.1]. En v1, c'est le
   prédicteur dont l'IoU courante avec la vérité est la plus haute [1506.02640#005.0]. Avec
   plusieurs échelles, on choisit parmi les 6 ou 9 ancres ; l'échelle retenue est celle à
   laquelle appartient l'ancre gagnante.
3. **Cibles de régression**, obtenues en inversant le décodage [1804.02767#003.0] :

$$
x^*=g_xS-j\in[0,1),\quad y^*=g_yS-i,\quad t_w^*=\ln\frac{g_w}{p_w},\quad t_h^*=\ln\frac{g_h}{p_h}
$$

   avec $p_w,p_h$ exprimés dans la même unité que $g_w,g_h$, par exemple en fraction de
   l'image.
4. **Masques** pour chaque ancre de chaque cellule : *obj* si l'ancre est responsable ;
   *ignore* si elle n'est pas responsable mais que sa boîte **prédite** recouvre une vérité
   avec une IoU > 0,5 ; *noobj* sinon [1804.02767#003.1].

### 5.2 Calcul des ancres par k-means (méthode YOLOv2)

```python
def kmeans_anchors(boxes_wh, k, iters=300):     # boxes_wh: liste de (w, h)
    centroids = random_sample(boxes_wh, k)
    for _ in range(iters):
        # distance d = 1 - IoU(box, centroid), boîtes centrées à l'origine [1612.08242#002.4]
        assign = [argmin([1 - iou_wh(b, c) for c in centroids]) for b in boxes_wh]
        new = [mean_wh([b for b, a in zip(boxes_wh, assign) if a == q]) for q in range(k)]
        if new == centroids: break
        centroids = new
    return sorted(centroids, key=lambda c: c[0]*c[1])

def iou_wh(a, b):                                 # IoU de deux boîtes de même centre
    inter = min(a[0], b[0]) * min(a[1], b[1])
    return inter / (a[0]*a[1] + b[0]*b[1] - inter)
```

Pour Tiny-YOLOv3, $k=6$ : trois ancres par tête. YOLOv3 répartit simplement les clusters à
parts égales entre les échelles [1804.02767#005.1].

### 5.3 IoU de deux boîtes (centre, largeur, hauteur)

$$
\text{IoU}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|},\quad
|A\cap B|=\max(0,\min(x_2^A,x_2^B)-\max(x_1^A,x_1^B))\cdot\max(0,\min(y_2^A,y_2^B)-\max(y_1^A,y_1^B))
$$

avec $x_1=x-w/2$ et $x_2=x+w/2$. C'est le rapport entre l'aire commune et l'aire de l'union
[2025-hozhabr#056.0].

---

## 6. Fonction de perte et gradients

### 6.1 Perte de YOLOv1 (référence historique)

Somme des erreurs quadratiques en cinq termes [1506.02640#005.0] :

$$
\begin{aligned}
L=\;&\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^{B}\mathbb{1}^{obj}_{ij}\big[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2\big]
+\lambda_{coord}\sum_{i,j}\mathbb{1}^{obj}_{ij}\Big[(\sqrt{w_i}-\sqrt{\hat w_i})^2+(\sqrt{h_i}-\sqrt{\hat h_i})^2\Big]\\
&+\sum_{i,j}\mathbb{1}^{obj}_{ij}(C_i-\hat C_i)^2+\lambda_{noobj}\sum_{i,j}\mathbb{1}^{noobj}_{ij}(C_i-\hat C_i)^2
+\sum_{i}\mathbb{1}^{obj}_{i}\sum_{c\in\text{classes}}(p_i(c)-\hat p_i(c))^2
\end{aligned}
$$

- $\lambda_{coord}=5$ et $\lambda_{noobj}=0{,}5$. Sans eux, les nombreuses cellules vides
  écrasent le gradient et l'entraînement diverge [1506.02640#004.1, 1506.02640#004.2].
- $\sqrt{w}$ et $\sqrt{h}$ réduisent le poids des erreurs sur les grandes boîtes
  [1506.02640#004.1].
- La classification n'est pénalisée que si un objet est présent dans la cellule, et les
  coordonnées que pour le prédicteur responsable [1506.02640#005.1].

### 6.2 Perte de YOLOv3 (celle à implémenter pour Tiny-YOLOv3)

Pour chaque échelle, avec $\text{BCE}(p,y)=-[y\ln p+(1-y)\ln(1-p)]$ :

$$
\begin{aligned}
L=\;&\lambda_{coord}\sum_{obj}\Big[(\sigma(t_x)-x^*)^2+(\sigma(t_y)-y^*)^2+(t_w-t_w^*)^2+(t_h-t_h^*)^2\Big]\\
&+\sum_{obj}\text{BCE}(\sigma(t_o),1)+\sum_{noobj}\text{BCE}(\sigma(t_o),0)
+\sum_{obj}\sum_{c=1}^{C}\text{BCE}(\sigma(t_c),\mathbb{1}[c=c^*])
\end{aligned}
$$

- Les termes viennent de YOLOv3 : somme des carrés pour les coordonnées
  [1804.02767#003.0], objectness logistique avec règle « ignore » [1804.02767#003.1], BCE
  multi-label pour les classes [1804.02767#004.0].
- Les ancres *ignore* ne contribuent à aucun terme.
- **Coordonnées (variante hors base).** L'article place l'erreur quadratique sur $t$, avec
  la cible $t_x^*=\sigma^{-1}(x^*)$, infinie si $x^*\in\{0,1\}$. La variante ci-dessus la
  place sur $\sigma(t_x)$, ce qui évite d'inverser la sigmoïde. C'est celle de
  l'implémentation de référence du §11.
- **Facteur d'échelle.** On multiplie en général le terme de coordonnées par
  $\omega_{scale}=2-g_wg_h$, qui renforce le poids des petites boîtes. Gaussian YOLOv3 décrit
  ce poids pour la perte de boîte de YOLOv3 [1904.04620#005.2] ; ses cibles $x^G=x_G W-i$ et
  $w^G=\log(w_G\,IW/A^k_w)$ sont celles du §5.1 [1904.04620#005.1].

**Gradients par rapport aux sorties brutes (hors base, vérifiés numériquement)**, pour une
ancre responsable :

| Sortie | $\partial L/\partial t$ |
|---|---|
| $t_x$ | $2\lambda_{coord}(\sigma(t_x)-x^*)\,\sigma(t_x)(1-\sigma(t_x))$ |
| $t_y$ | idem avec $y$ |
| $t_w$ | $2\lambda_{coord}(t_w-t_w^*)$ |
| $t_h$ | $2\lambda_{coord}(t_h-t_h^*)$ |
| $t_o$ | $\sigma(t_o)-1$ si *obj* ; $\sigma(t_o)$ si *noobj* ; 0 si *ignore* |
| $t_c$ | $\sigma(t_c)-\mathbb{1}[c=c^*]$ |

Point clé : la dérivée de $\text{BCE}(\sigma(t),y)$ par rapport à $t$ vaut simplement
$\sigma(t)-y$, car les termes en $\sigma(1-\sigma)$ se simplifient. On retrouve le
« vérité moins prédiction » de YOLOv3 [1804.02767#003.0].

### 6.3 Autres pertes de localisation (options)

- **Smooth L1 (Huber).** C'est le choix courant
  $L_{loc}=\sum_{i\in\{x,y,w,h\}}s(\hat i-i)$, avec $s(x)=0{,}5x^2$ si $|x|<1$ et
  $|x|-0{,}5$ sinon [2025-el-zeinaty#010.0]. La perte totale d'un détecteur s'écrit
  $L=\beta L_{cls}+\alpha L_{loc}$ [2025-el-zeinaty#008.0].
- **GIoU, DIoU, CIoU.** Ces pertes optimisent directement le recouvrement entre la boîte
  prédite et la vérité ; DIoU et CIoU ajoutent la distance des centres et le rapport d'aspect
  [2025-el-zeinaty#010.1]. Leurs formules ne sont pas dans la base. Les définitions
  usuelles sont les suivantes **(hors base)** :

$$
L_{GIoU}=1-\text{IoU}+\frac{|E\setminus(A\cup B)|}{|E|},\qquad
L_{DIoU}=1-\text{IoU}+\frac{\rho^2(\mathbf b,\mathbf b^{gt})}{c^2},\qquad
L_{CIoU}=L_{DIoU}+\alpha v
$$

  avec $E$ la plus petite boîte englobante, $\rho$ la distance des centres, $c$ la diagonale
  de $E$, $v=\frac{4}{\pi^2}\big(\arctan\frac{w^{gt}}{h^{gt}}-\arctan\frac wh\big)^2$ et
  $\alpha=v/((1-\text{IoU})+v)$. Ces pertes remplacent le terme de coordonnées du §6.2 ;
  leur gradient se dérive en propageant à travers $\min$, $\max$ et le décodage du §8.1.
- **Gaussian YOLOv3.** Les coordonnées deviennent des gaussiennes (moyenne et variance), et
  la perte de boîte devient une log-vraisemblance négative ; objectness et classes sont
  inchangées [1904.04620#005.0]. L'incertitude prédite rend le modèle plus robuste aux
  annotations bruitées [1904.04620#005.3]. C'est utile pour un système qui doit estimer sa
  propre confiance.
- **Focal loss.** Elle n'apporte rien à YOLOv3 [1804.02767#009.0].

---

## 7. Entraînement

### 7.1 Algorithme : rétropropagation et SGD avec momentum

```python
for epoch in range(E):
    for X, targets in batches(dataset, size=64):           # lot de 64 [1506.02640#005.1]
        if multiscale and it % 10 == 0:                     # [1612.08242#003.2]
            res = choice(range(320, 609, 32)); X = resize(X, res)
        out    = forward(X)                                 # garder les caches de chaque couche
        L, dO  = yolo_loss(out, targets, anchors)           # §6.2
        grads  = backward(dO)                               # §4, couches en ordre inverse
        lr     = schedule(it)
        for p in params:                                    # SGD + momentum + weight decay
            v[p] = mu * v[p] - lr * (grads[p] + wd * p)     # mu = 0.9, wd = 5e-4
            p   += v[p]
```

Hyperparamètres publiés :

| | YOLOv1 [1506.02640#005.1] | YOLOv2 détection [1612.08242#005.3] |
|---|---|---|
| Lot | 64 | n/r |
| Momentum | 0,9 | 0,9 |
| Weight decay | 0,0005 | 0,0005 |
| Taux d'apprentissage | montée de $10^{-3}$ à $10^{-2}$, puis $10^{-2}$ (75 époques), $10^{-3}$ (30), $10^{-4}$ (30) | $10^{-3}$, divisé par 10 aux époques 60 et 90 (160 époques) |
| Régularisation | dropout 0,5, augmentation | BN, pas de dropout [1612.08242#002.0] |

- **Montée progressive (*warm-up*).** Commencer avec un taux élevé fait diverger le modèle
  [1506.02640#005.1].
- **Pré-entraînement.** Le classifieur Darknet-19 est entraîné sur ImageNet avec un taux de
  0,1 et une décroissance polynomiale de puissance 4 [1612.08242#005.2]. En pratique pour un
  Tiny embarqué, on part de poids pré-entraînés et on affine sur le jeu cible, comme
  2026-fata [2026-fata#013.1].
- **Augmentation.** Mise à l'échelle et translations aléatoires jusqu'à 20 % de la taille de
  l'image ; exposition et saturation modifiées aléatoirement jusqu'à un facteur 1,5 en HSV
  [1506.02640#005.2]. Les boîtes cibles subissent les mêmes transformations géométriques.
- **Initialisation (hors base).** Poids conv $\sim\mathcal N(0,\,2/(k^2C_{in}))$ (He),
  $\gamma=1$, $\beta=0$, biais de la tête à 0.

---

## 8. Inférence : décodage, seuil, NMS, évaluation

### 8.1 Décodage

Pour chaque cellule $(i,j)$, chaque ancre $a$ et chaque échelle $S$, avec des ancres en
fraction de l'image :

$$
b_x=\frac{\sigma(t_x)+j}{S},\quad b_y=\frac{\sigma(t_y)+i}{S},\quad b_w=p_we^{t_w},\quad b_h=p_he^{t_h}
$$

2024-zhang calcule directement les coins, par exemple
$x_1=[(\sigma(t_x)+c_x)-0{,}5\,p_we^{t_w}]\cdot\text{scale}$ [2024-zhang#014.0]. Score de
la classe $c$ :

- YOLOv3 : $\sigma(t_o)\cdot\sigma(t_c)$ ;
- YOLOv2 : $\sigma(t_o)\cdot\text{softmax}(t)_c$. 2024-zhang calcule ce score avec un softmax
  stabilisé, en soustrayant le maximum [2024-zhang#014.1].

### 8.2 Seuil et suppression des non-maxima (NMS)

On filtre d'abord les scores sous le seuil. On trie ensuite les scores, on garde la meilleure
boîte et on élimine celles dont l'IoU avec elle dépasse un seuil ; on recommence jusqu'à
épuisement [2024-zhang#013.0]. Selon YOLOv1, la NMS apporte 2 à 3 points de mAP
[1506.02640#006.0].

```python
def nms(boxes, scores, iou_thr=0.45):           # par classe ; seuils usuels (hors base)
    order = argsort(scores, descending=True); keep = []
    while order:
        best = order.pop(0); keep.append(best)
        order = [k for k in order if iou(boxes[best], boxes[k]) <= iou_thr]
    return keep

dets = []
for c in classes:
    cand = [(b, s[c]) for b, s in decoded if s[c] > conf_thr]   # conf_thr ≈ 0,25 à 0,5
    dets += [cand[k] for k in nms(*zip(*cand))]
```

### 8.3 Évaluation : IoU, précision, rappel, mAP

- Une détection est un vrai positif si son IoU avec une vérité non encore appariée atteint le
  seuil, par exemple 0,5 [2025-hozhabr#056.0].
- Pour chaque classe : trier les détections par score, cumuler TP et FP, puis tracer la
  courbe précision/rappel. L'AP de PASCAL VOC est la précision interpolée en 11 niveaux de
  rappel [2025-el-zeinaty#015.0]. La mAP est la moyenne des AP des classes.

$$
\text{AP}_{11}=\frac1{11}\sum_{r\in\{0,0.1,\dots,1\}}\max_{\tilde r\ge r}P(\tilde r)
$$

---

## 9. Passage à l'embarqué : fusion BN, quantification, arithmétique entière

### 9.1 Fusion de la BN dans la convolution

En inférence, la BN est une transformation affine par canal. On l'intègre aux poids de la
convolution qui la précède, ce qui supprime une passe sur la mémoire et les divisions et
racines carrées [2023-zhai#039.0, 2202.10935#007.1]. C'est la première étape d'une
quantification après entraînement (PTQ) [2607.13106#008.0]. Avec
$s_f=\gamma_f/\sqrt{\sigma^2_f+\varepsilon}$ :

$$
W'_{f}=s_f\,W_{f},\qquad b'_f=\beta_f+s_f\,(b_f-\mu_f)\quad(b_f=0\text{ si la conv n'a pas de biais})
$$

2023-zhai donne la même relation, $O_{norm}=\frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}O+\big(\beta-\frac{\gamma\mu}{\sqrt{\sigma^2+\varepsilon}}\big)$
[2023-zhai#039.0]. La forme explicite sur $W'$ et $b'$ est une dérivation **(hors base,
vérifiée numériquement : écart max 5·10⁻¹⁵)**. EF-Train rappelle qu'à l'**entraînement** la
fusion est impossible, puisque $\mu$ et $\sigma^2$ dépendent du lot ; un entraînement
embarqué garde donc la BN, en pleine précision [2202.10935#007.1].

### 9.2 Schémas de quantification

**Quantification uniforme affine** [2024-campos#004.0] :

$$
q=\text{Clip}\big(\text{Round}(r/S)-Z,\;\alpha,\beta\big),\qquad r\approx S\,(q+Z)
$$

- $S$ est l'échelle et $Z$ le zéro ; le choix de la plage $[\alpha,\beta]$ est la
  *calibration*.
- Pour les poids, la variante **symétrique** ($Z=0$) **par canal de sortie** est la plus
  simple en matériel **(hors base)**. Une échelle propre à chaque filtre absorbe les
  $s_f$ très différents issus de la fusion BN.

**Calibration.**
- PTQ : on estime l'échelle et le zéro sur un sous-ensemble des données
  [2607.13106#008.0].
- 2026-fata compare des seuils d'écrêtage aux 90e, 95e et 99e percentiles et garde celui qui
  minimise l'erreur entre poids FP32 et INT8 [2026-fata#015.1].

**PTQ ou QAT.**
- **QAT** : on entraîne avec des poids « fake-quantized » (quantifiés puis déquantifiés en
  propagation avant) [2026-fata#015.0]. 2026-fata affine ainsi avec un taux
  d'apprentissage de $10^{-4}$ [2026-fata#015.3]. TinyissimoYOLO fait 350 époques en
  flottant, puis 300 en QAT [2306.00001#004.1].
- En QAT, l'arrondi a une dérivée nulle presque partout. On le traverse avec l'estimateur
  *straight-through* : $\partial\,\text{round}(x)/\partial x:=1$ dans la plage, 0 hors
  plage **(hors base)**.

**Largeurs de bits utilisées dans la base pour des YOLO sur FPGA :**

| Travail | Modèle | Format | Remarque |
|---|---|---|---|
| 2025-kim | YOLOv2 | INT16 | perte d'environ 0,2 % par rapport au FP32 [2025-kim#004.0] ; une MAC par DSP [2025-kim#011.1] |
| 2023-zhai | YOLOv3-tiny élagué | INT16, virgule fixe dynamique | `benchmarks.csv`, Table 8 |
| 2024-zhang | YOLOv2/v3-Tiny | INT8, virgule fixe dynamique (BRECQ) | `benchmarks.csv`, Tables 3-4 |
| 2023-montgomerie-corcoran (SATAY) | YOLOv3-Tiny | W8A16 | `benchmarks.csv`, Table III |
| 2026-fata | YOLOv3-tiny | INT8 QAT, échelle par couche (poids) | [2026-fata#015.0] |
| 2024-yan | YOLOv5s | 4 bits ; écrêtage $c$ appris et contraint à une puissance de 2 | réduit le coût matériel [2024-yan#006.0] |
| 2019-ding (REQ-YOLO) | Tiny-YOLOv2 | hétérogène par couche : équidistante ou « mixed powers-of-two » | choisie par ADMM [2019-ding#008.1, 2019-ding#009.4] |

**REQ-YOLO en détail.**
- Un poids de 6 bits se compose d'un bit de signe et de 5 bits de magnitude : 3 bits
  « primaires » et 2 « secondaires ». La multiplication devient deux décalages et une
  addition [2019-ding#008.1].
- Le problème de quantification est décomposé en deux sous-problèmes résolus itérativement
  jusqu'à convergence [2019-ding#009.1] : un pas de SGD sur la perte pénalisée
  $\frac\rho2\lVert W-Z+U\rVert_F^2$, puis une projection de $W+U$ sur les niveaux autorisés.
- Ces niveaux valent $\alpha\times\{-(M/2-1),\dots,M/2-1\}$ (équidistants) ou
  $\alpha\times\{0,\pm2^0,\dots,\pm2^{M_1}\}$ (puissances de 2) [2019-ding#009.4].

### 9.3 Une couche conv + leaky en arithmétique entière (hors base, vérifié numériquement)

Les papiers de la base fixent le format, mais aucun ne donne la chaîne de requantification.
Voici une dérivation standard, contrôlée au §11. On note $s_x$, $s_w$ et $s_y$ les
échelles d'entrée, de poids et de sortie, toutes symétriques.

$$
\underbrace{acc_f=\sum q_w\,q_x}_{\text{int32}}+\underbrace{q_{b,f}}_{=\text{round}(b'_f/(s_xs_{w,f}))},\qquad
q_y=\text{clip}\Big(\big(acc_f\cdot M_{0,f}+2^{n-1}\big)\gg n,\;-127,\;127\Big),\qquad
M_{0,f}=\text{round}\Big(\frac{s_xs_{w,f}}{s_y}2^{n}\Big)
$$

```python
def conv_int8(qx, qW, qb, M0, n=31):              # qx,qW : int8 ; qb : int32 ; M0 : int32 par canal
    acc = conv_forward(qx, qW, qb)                # accumulateur int32 (exact)
    y = (acc * M0[:,None,None] + (1 << (n-1))) >> n   # requantification : 1 multiplication + 1 décalage
    y = where(y > 0, y, (y * 13) >> 7)            # leaky 0,1 ≈ 13/128 = 0,1016
    return clip(y, -127, 127)
```

- **Largeur de l'accumulateur.** Un produit int8×int8 tient sur 16 bits, et une somme de
  $k^2C_{in}$ termes ajoute $\lceil\log_2(k^2C_{in})\rceil$ bits. Le pire cas de Tiny-YOLOv2
  est la couche 13 : $9\times1024=9216$ termes, soit 16 + 14 = 30 bits, qui tiennent dans
  32 bits.
- **Leaky ReLU en matériel.** Elle demande une multiplication par une constante et un
  multiplexeur pour les entrées positives [2023-montgomerie-corcoran#012.0]. La pente
  $13/128$ en est la version « décalage ».
- **Changer d'activation.** 2024-yan remplace SiLU par une PReLU à décalage entraînable,
  qui ne coûte qu'un additionneur [2024-yan#005.0].
- **Première couche.** Elle reçoit des pixels sur 8 bits, alors que le reste du réseau peut
  être en 4 bits. 2024-yan la décompose en deux convolutions 4 bits, $\text{conv}_{odd}\times16+\text{conv}_{even}$
  [2024-yan#007.3].

### 9.4 Sigmoïde, exponentielle et softmax en matériel

- **Tables de correspondance (LUT).** Avec des $t$ quantifiés sur 8 bits, la sigmoïde et
  l'exponentielle du décodage se font par deux tables de 256 entrées [2024-zhang#014.0].
  Discrétiser la sigmoïde est la solution la plus répandue ; plus de pas donnent plus de
  précision mais coûtent plus de surface [2024-rech#007.4].
- **Précision d'une LUT.** Avec un pas d'entrée de 1/16, une LUT de 256 entrées donne une
  erreur maximale de 0,0075 sur $\sigma$ (vérifié, §11).
- **Softmax en trois étages.** On extrait le maximum, on calcule $e^{x_i-x_{max}}$ et leur
  somme, puis on divise en flottant [2024-zhang#014.1].
- **Seuillage sans sigmoïde (hors base).** $\sigma$ est monotone, donc
  $\sigma(t_o)>\theta\iff t_o>\ln\frac{\theta}{1-\theta}$. On filtre les boîtes sur
  l'entier $t_o$ avant tout décodage, et la LUT ne sert qu'aux survivantes.

---

## 10. L'accélérateur FPGA

### 10.1 Deux familles d'architecture

| | Streaming / dataflow | Moteur unique, couche par couche |
|---|---|---|
| Principe | un bloc matériel par couche, en pipeline [2018-venieris#004.0] | un réseau systolique ou une unité matricielle exécute les couches l'une après l'autre ; le logiciel ordonnance [2018-venieris#007.1] |
| Avantage | latence et débit ; dans SATAY, tous les paramètres restent sur la puce [2023-montgomerie-corcoran#006.0] | un seul bitstream pour tous les réseaux ; tient sur un petit Zynq |
| Coût | un bitstream par réseau, compilation longue [2018-venieris#004.0] | trafic DDR à chaque couche |
| Exemples YOLO | SATAY (YOLOv3-Tiny, W8A16) | 2025-kim, 2024-zhang, 2023-zhai, 2024-yan (« layer-by-layer ») |

Pour une première reproduction sur un Zynq ou un KV260, le **moteur unique** est le plus
simple : une seule PE conv paramétrée, réutilisée par les 13 à 15 convolutions.

### 10.2 Nid de boucles, tuilage et roofline (2015-zhang)

Le tuilage est obligatoire pour faire tenir une petite partie des données sur la puce
[2015-zhang#009.0]. On tuile les lignes, les colonnes et les canaux d'entrée et de sortie,
et l'on ne déroule que $T_m$ et $T_n$ ; le toit de calcul est alors une fonction de $T_m$ et
$T_n$ [2015-zhang#010.2].

```python
for to in range(0, M, Tm):               # canaux de sortie   (tuile)
  for ti in range(0, N, Tn):             # canaux d'entrée    (tuile)
    for row in range(0, R, Tr):          # lignes de sortie   (tuile)
      for col in range(0, C, Tc):        # colonnes de sortie (tuile)
        load(in_buf[Tn][S*Tr+K-S][S*Tc+K-S]); load(w_buf[Tm][Tn][K][K])   # ping-pong
        for i in range(K):
          for j in range(K):
            for trr in range(Tr):
              for tcc in range(Tc):      # PIPELINE II=1
                for too in range(Tm):    # UNROLL : Tm × Tn multiplieurs (DSP)
                  for tii in range(Tn):  # UNROLL + arbre d'additions
                    out_buf[too][trr][tcc] += w_buf[too][tii][i][j] * in_buf[tii][S*trr+i][S*tcc+j]
        if ti + Tn >= N: store(out_buf)  # biais + requantif + leaky (+ maxpool fusionné)
```

- **Taille des tampons.** $B_{in}=T_n(ST_r+K-S)(ST_c+K-S)$, $B_w=T_mT_nK^2$ et
  $B_{out}=T_mT_rT_c$ doivent tenir dans la BRAM. Ces formules sont lues dans 2015-zhang
  §4 (équations 5 à 8), mais le texte extrait est dégradé : relire `bin/pdb show
  2015-zhang#014.0`.
- **Roofline.** La performance atteignable vaut
  $\min(\text{toit de calcul},\ \text{CTC}\times BW)$, où le CTC est le rapport
  calcul/communication [2015-zhang]. On énumère les $(T_m,T_n,T_r,T_c)$ et on garde le
  meilleur point sous le toit de bande passante.
- **Double tampon.** Des buffers en ping-pong recouvrent les transferts par le calcul
  [2015-zhang#020.0]. L'implémentation de référence duplique 64 structures pour dérouler
  $T_m$ [2015-zhang#019.0].
- **Ordre de grandeur pour Tiny-YOLO.** $T_m\times T_n$ fixe le nombre de MAC par cycle :
  $T_m=T_n=16$ donne 256 MAC/cycle, soit 3,49 G MAC / 256 ≈ 13,6 M cycles par image,
  environ 68 ms à 200 MHz pour une efficacité de 100 % **(calcul hors base)**.

**Parallélisme effectivement utilisé :**
- 2024-zhang déroule $N_m$ canaux de sortie, $N_r$ lignes et $N_n$ MAC par noyau, et range
  deux noyaux par DSP par *packing* de multiplications [2024-zhang#011.0].
- 2025-kim utilise une PE conv 3×3 ($T_r=3$, $T_c=15$, réutilisation de l'IFM), une PE conv
  1×1 ($T_r=1$, $T_c=13$) et une seule PE maxpool [2025-kim#018.0]. Le tout est piloté par
  six contrôleurs, dont les DMA [2025-kim#017.0].
- 2023-zhai organise la convolution en quatre étages, *line cache*, Im2col, GEMM et sortie,
  et fusionne la convolution avec le pooling pour réduire trois accès mémoire à un seul
  [2023-zhai#034.0].
- SATAY génère la fenêtre glissante par *line buffers*, qui ne demandent que
  $(K-1)\times W\times C$ mots [2023-montgomerie-corcoran#011.0].

### 10.3 Couches spécifiques à YOLO

| Couche | Solution lue dans la base | À compléter (hors base) |
|---|---|---|
| maxpool 2×2/2 | comparateurs, fusionnés en sortie de conv [2023-zhai#037.0, 2023-zhai#034.0] ; PE dédiée [2025-kim#018.0] | — |
| **maxpool 2×2/1** | **non traité dans la base** | même comparateur sur une fenêtre glissante ; réplique de la dernière ligne et de la dernière colonne ; sortie 13×13 |
| upsample ×2 | quatre copies de chaque pixel [2023-zhai#037.1] | ou aucune copie : diviser par 2 l'adresse de lecture de la couche suivante |
| route / concat | placement contigu en mémoire de la sortie des couches 8 et 19 [2023-zhai#016.2] ; multiplexeur de flux dans SATAY [2023-montgomerie-corcoran#011.0] ; sur ARM chez 2025-kim [2025-kim#026.0] | écrire la couche 19 juste après la couche 8 dans le même tampon, ce qui rend la concaténation gratuite |
| décodage + NMS | sur l'ARM : 2025-kim [2025-kim#026.0], 2024-yan [2024-yan#008.0] ; NMS sur l'hôte : 2023-zhai [2023-zhai#014.2] ; en circuits dédiés : 2024-zhang [2024-zhang#013.1] | — |

**Post-traitement matériel de 2024-zhang.**
- Sur un FPGA sans processeur dur, seul un softcore (MicroBlaze, NIOS) est disponible, d'où
  des circuits dédiés [2024-zhang#013.1].
- Le NMS est économe en ressources : il ne trie pas, et une boîte sélectionnée est
  remplacée par une boîte qui la recouvre au-delà du seuil avec un meilleur score
  [2024-zhang#014.2].
- Les 845 boîtes candidates (13×13×5) sont traitées en 0,38 ms [2024-zhang#020.3].

### 10.4 Résultats publiés (`bin/pdb bench`)

Puissance : *board* = mesurée sur la carte ; *unspec.* = périmètre non précisé. Les chiffres
ne sont **pas directement comparables** : jeux de données, élagage, fréquence et périmètre
de puissance diffèrent. Vérifier avec `bin/pdb bench --comparable` avant toute comparaison.

| Travail | Modèle | FPGA | Format | FPS | GOPS | W (périmètre) | LUT / DSP / BRAM | Source (conf.) |
|---|---|---|---|---|---|---|---|---|
| 2023-zhai | YOLOv3-tiny élagué, 1 module | Zynq XC7Z035 | INT16 | 91,65 | 67,91 | 12,51 (unspec.) | 38 228 / 144 / 132,5 | Table 8 (high) |
| 2023-zhai | YOLOv3-tiny élagué, 2 modules | Zynq XC7Z035 | INT16 | 168,72 | 124,01 | 15,18 (unspec.) | 91 108 / 294 / 263 | Table 9 (high) |
| 2024-zhang | YOLOv2-Tiny 416, lot 1 | Kintex-7 325T | INT8 | n/r (14,49 ms) | 406 | 12,3 (board) | 134 900 / 687 / 551 | Table 3, §5.3 (medium) |
| 2024-zhang | YOLOv3-Tiny 416, lot 1 | Kintex-7 325T | INT8 | n/r (15,2 ms) | 401 | 12,3 (board) | 136 500 / 687 / 553 | Tables 4-5 (high) |
| 2023-montgomerie-corcoran | YOLOv3-Tiny 416 | VCU110 | W8A16 | n/r | 418,9 | 15,4 (unspec.) | voir Table III | Table III (medium) |
| 2019-ding | Tiny-YOLOv2, ADMM hétérogène | Virtex-7 690T | FFT + puissances de 2 | 314,2 | n/r | 21 (unspec.) | n/r | Tables 1-2 (high) |
| 2026-fata | YOLOv3-tiny élagué à 70 % | Kria KV260 (DPU) | INT8 QAT | 24,3 | 60,18 | 2,13 (unspec.) | n/r | Tables 7-10 (medium) |

- **2024-zhang.** Puissance de la puce estimée par Vivado : 7,8 W, contre 12,3 W mesurés
  sur la carte.
- **2026-fata.** Le travail repose sur un DPU Vitis AI préconfiguré, pas sur un
  accélérateur conçu à la main.
- **2025-kim (YOLOv2 complet, Zybo Z7-20, INT16).** Prototype fonctionnel lent, environ
  12 s par image : ce n'est pas une référence de performance.

### 10.5 Feuille de route d'une reproduction (synthèse, hors base)

1. **Modèle flottant de référence** (§3 à §8) en Python pur ou NumPy, validé par vérification
   des gradients (§11). Entraîné ou repris de poids Darknet, il donne la mAP de référence.
2. **Modèle entier bit-exact** (§9) : fusion BN, échelles par canal, requantification
   M0/décalage, leaky 13/128 et LUT. On mesure la perte de mAP par rapport au flottant ; si
   elle est trop forte, on passe au QAT.
3. **Modèle C++ bit-exact** du moteur : nid de boucles tuilé du §10.2, mêmes entrées et
   sorties que le modèle entier. Il sert de *golden model*.
4. **HLS ou RTL** de la PE conv avec maxpool fusionné, puis des unités upsample et route
   (adressage), avec DMA et ping-pong ; comparaison couche par couche avec le *golden
   model*.
5. **Post-traitement** sur l'ARM d'abord, comme 2025-kim et 2024-yan, puis éventuellement
   en matériel, comme 2024-zhang.
6. **Mesures** : FPS, latence, puissance en précisant le périmètre (puce ou carte) et
   ressources, au format de `benchmarks.csv`.

---

## 11. Vérifications et tests à reproduire

Implémentation de référence en NumPy, utilisée seulement pour les tests. Les scripts de
cette vérification (`gradcheck.py`, `quant.py`) sont restés dans le répertoire temporaire de
session ; ce n'est pas un livrable de la base.

| Test | Méthode | Résultat obtenu |
|---|---|---|
| Gradients conv, BN ($\gamma,\beta$), leaky, maxpool s2 et s1, route (cumul), conv 1×1 de tête, perte YOLOv3 complète (§6.2) | différences finies centrées, $h=10^{-6}$, sur un mini-réseau conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1 avec 2 images et 3 objets | erreur relative ≤ 2,2·10⁻⁹ pour tous les paramètres |
| Biais avant BN | gradient analytique | ≈ 10⁻¹⁵ (nul, §4.2) |
| Upsample ×2 | différences finies | 1,3·10⁻⁹ |
| Fusion BN (§9.1) | comparaison avec conv + BN | 5·10⁻¹⁵ |
| Conv INT8 + requantification M0/décalage + leaky 13/128 (§9.3) | comparaison avec la sortie flottante d'une couche 8→16 canaux | erreur max 1,2 % de la dynamique ; accumulateur de 18 bits suffisant ici |
| LUT sigmoïde 256 entrées, pas 1/16 | 1 000 tirages | erreur max 0,0075 |
| Paramètres et MACs (§3) | script de comptage | Tiny-YOLOv2 : 3,49 GMAC, conforme à [2006.08357#025.0] |

**Tests recommandés pour une implémentation complète :**
- vérifier les gradients de chaque nouvelle couche ;
- surapprendre une seule image : la perte doit tendre vers 0 et les boîtes coïncider ;
- comparer couche par couche le flottant, l'entier et le matériel ;
- mesurer la mAP VOC aux trois stades (flottant, entier, FPGA).

---

## 12. Références (articles de la base utilisés)

| id court | Titre | Auteurs | Année | Venue | Type | Lien |
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

YOLOv3 est un rapport technique jamais publié en conférence, mais c'est la référence de fait.
Les articles marqués **preprint** n'ont pas été relus par les pairs.

### Ce que la base ne couvre pas (gaps)

- **Maxpool 2×2 de stride 1** (Tiny-YOLOv2/v3) : aucun papier ne décrit son traitement
  matériel.
- **Fichiers `.cfg` de Darknet** (canaux exacts, ancres de Tiny-YOLOv3) : absents ; les
  valeurs du §3 en sont tirées hors base.
- **Formules GIoU, DIoU et CIoU**, chaîne de requantification entière et dérivées de BN :
  dérivées ici, non lues dans la base.
- **Entraînement de YOLO directement sur FPGA** : EF-Train entraîne des CNN de
  classification. Pour la thèse, un YOLO qui s'adapte en ligne sur FPGA (apprentissage
  continu) reste une piste ouverte, à croiser avec l'axe continual learning de la base.

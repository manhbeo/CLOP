# CLOP: Semi-Supervised Contrastive Learning with Orthonormal Prototypes

This repository contains the official implementation of **CLOP** (Contrastive Learning with Orthonormal Prototypes) from the paper

> **Semi-Supervised Contrastive Learning with Orthonormal Prototypes**  
> Huanran Li, Manh Nguyen, Daniel Pimentel-Alarcón :contentReference[oaicite:0]{index=0}

CLOP is a semi-supervised contrastive learning loss that:

- Analyzes **dimensional collapse** in cosine-similarity based contrastive losses from a **learning-rate** perspective.
- Introduces a **prototype-based loss** that pulls labeled embeddings toward **orthonormal class prototypes**, encouraging high-rank, well-separated representations.
- Achieves strong and stable performance across learning rates, batch sizes, and label fractions on CIFAR-100, ImageNet and related benchmarks. :contentReference[oaicite:1]{index=1}

---

## 1. Method Overview

### 1.1 Background: Dimensional Collapse in InfoNCE

Given embeddings $\{z_i\}$ on the unit sphere, the InfoNCE loss is

$$
L_{\text{InfoNCE}}
= - \sum_{i \in I}
\log \frac{\exp(z_i^\top z_{j(i)} / \tau)}
{\sum_{a \neq i} \exp(z_i^\top z_a / \tau)}.
$$

The paper shows that, when all embeddings are equal or co-linear, InfoNCE has **stationary points**: the gradient can vanish even though the representation is useless. :contentReference[oaicite:2]{index=2}

A gradient analysis reveals:

- There is a **critical range of learning rates**; outside this range, gradient steps move the **mean embedding** in a way that drives the system toward a low-rank collapsed state.
- Under a simplified setting where positives are already merged, the “safe” learning rates are centered around approximately $\tau / 2$. :contentReference[oaicite:3]{index=3}

This explains why contrastive models can suddenly collapse when the learning rate is too large.

### 1.2 CLOP Loss with Orthonormal Prototypes

CLOP adds a **prototype-pulling term** on top of InfoNCE (or SupCon), using a small labeled subset.

Let:

- $S = \{(z_i, y_i)\}$ be labeled embeddings and labels,
- $k$ be the number of classes,
- $C = \{c_1, \dots, c_k\}$ be **orthonormal prototypes** in $\mathbb{R}^{d}$:

  - Randomly sample $k$ vectors in $\mathbb{R}^d$,
  - Run SVD (or QR) and take $k$ orthonormal unit vectors as prototypes. :contentReference[oaicite:4]{index=4}

The **CLOP loss** is

$$
L_{\text{CLOP}}
= L_{\text{InfoNCE}}
+ \lambda \, \frac{1}{|S|}
\sum_{(z_i, y_i) \in S}
\bigl(1 - s(z_i, c_{y_i})\bigr),
$$

where $s(\cdot,\cdot)$ is typically cosine similarity and $\lambda > 0$ weights the prototype term. :contentReference[oaicite:5]{index=5}

Intuitively:

- InfoNCE gives **repulsive** and **attractive** forces between augmented samples.
- CLOP adds a **class-specific pulling force**: labeled points are encouraged to lie near orthogonal directions $c_{y_i}$.
- This populates multiple dimensions and discourages rank-1 collapse, while still being compatible with standard SimCLR / SupCon pipelines. :contentReference[oaicite:6]{index=6}

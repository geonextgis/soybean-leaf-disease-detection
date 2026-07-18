# Building DINOv3 From Scratch — A Roadmap

This guide walks you through re-implementing the DINOv3 vision foundation model from scratch, using the reference code at `references/dinov3/` as the source of truth. It is scoped so that you (a) understand every mechanism, and (b) end up with a model you can plug into the soybean leaf disease detection pipeline.

> **Realistic scope note.** The full DINOv3 recipe was trained on 32 nodes / 256 GPUs for weeks on a 1.7B-image dataset (LVD-1689M). *Do not try to replicate that.* The goal here is to build the architecture and training loop end-to-end, verify it on a tiny dataset (e.g. ImageNet-100 or CIFAR), then fine-tune the released weights on soybean leaves. Everything below is written with that pragmatic path in mind.

---

## 0. Prerequisites

Before writing a single line of code:

1. **Read the paper.** DINOv3 — [arXiv:2508.10104](https://arxiv.org/abs/2508.10104). Focus on §3 (architecture), §4 (losses), §5 (training recipe).
2. **Read the two ancestors:**
   - DINO (Caron et al., 2021) — self-distillation with a momentum teacher.
   - DINOv2 (Oquab et al., 2023) — iBOT + KoLeo + LVD curation. DINOv3 is largely DINOv2 + Gram anchoring + RoPE + high-res adaptation.
3. **Environment.** PyTorch ≥ 2.7.1 with CUDA. On Windows, use WSL2 — the reference code assumes Linux. The reference `conda.yaml` at `references/dinov3/conda.yaml` is the canonical setup.
4. **Skim the reference tree once** so you have a map:
   - `dinov3/layers/` — attention, blocks, patch embed, RoPE, RMSNorm, LayerScale, FFNs
   - `dinov3/models/` — `vision_transformer.py`, `convnext.py`
   - `dinov3/loss/` — DINO, iBOT, KoLeo, Gram
   - `dinov3/train/` — `ssl_meta_arch.py` (student/teacher wiring), `train.py` (loop)
   - `dinov3/data/` — augmentations, masking, samplers, collate
   - `dinov3/configs/train/*.yaml` — reference hyperparameters

---

## 1. Suggested build order

Build bottom-up. Each stage should be runnable and testable in isolation before moving on.

### Stage 1 — Core layers (`layers/`)

Start with the ViT primitives. Each is a small, self-contained file. Write them, then unit-test with random tensors and compare shapes/outputs against the reference.

| Order | File to reproduce | What it does |
|-------|------------------|--------------|
| 1 | `patch_embed.py` | Splits `[B, 3, H, W]` into `[B, N, D]` patch tokens via a strided conv. |
| 2 | `rms_norm.py` | Root-mean-square LayerNorm (cheaper than LayerNorm, used in ViT-7B). |
| 3 | `layer_scale.py` | Learnable per-channel scaler on residual branches. |
| 4 | `ffn_layers.py` | `Mlp` and `SwiGLUFFN`. SwiGLU is the default for large models. |
| 5 | `rope_position_encoding.py` | 2D Rotary positional embedding — replaces DINOv2's learned pos-embed and is *the* key change enabling variable resolution. |
| 6 | `attention.py` | `SelfAttention` with RoPE applied to q/k. |
| 7 | `block.py` | Transformer block: `LayerNorm → Attn → LayerScale → residual → LayerNorm → FFN → LayerScale → residual`. |
| 8 | `dino_head.py` | 3-layer MLP + weight-normed linear projector — projects tokens to the DINO/iBOT prototype space. |

**Sanity check:** instantiate one block, push a random `[1, 197, 384]` through it, confirm output shape and non-NaN. Then diff parameter counts against the reference.

### Stage 2 — The ViT itself (`models/vision_transformer.py`)

Compose the layers into `DinoVisionTransformer`:

- CLS token + optional register tokens (registers stabilize dense features — from Darcet et al. 2024, included in DINOv3).
- Sequence of `SelfAttentionBlock`s.
- Two "views" of the output: (a) CLS token for global tasks, (b) patch tokens for dense tasks.
- A `forward_features()` that returns both.

Match one small variant first — **ViT-S/16** (embed_dim=384, 12 blocks, 6 heads). That's the smallest released config and gives you a target parameter count (~21M) to verify against.

**Sanity check:** load the released ViT-S/16 checkpoint into *your* class using `torch.load` + `load_state_dict(strict=True)`. If keys mismatch, your layer naming is off — fix it before moving on. This single check saves days.

### Stage 3 — Data pipeline (`data/`)

DINOv3 self-supervised training needs a very specific data recipe:

1. **Multi-crop augmentation** (`augmentations.py`) — 2 global crops (224×224) + N local crops (96×96) per image. Both student and teacher see globals; only student sees locals.
2. **Masking** (`masking.py`) — random patch masking for the iBOT loss. Teacher sees unmasked patches, student predicts them.
3. **Collate** (`collate.py`) — batches crops from multiple images while keeping the global/local distinction.

You can skip the LVD dataset entirely. For a from-scratch verification run, use `torchvision.datasets.ImageFolder` on any labeled dataset (soybean leaves, ImageNet-100, whatever). The SSL loop ignores labels.

### Stage 4 — Losses (`loss/`)

Four losses, added independently. Implement + unit-test each on random logits before wiring them in.

| Loss | Purpose | Where it comes from |
|------|---------|---------------------|
| `DINOLoss` (CLS token) | Student CLS matches sharpened teacher CLS via cross-entropy over prototypes. | DINO (2021) |
| `iBOTPatchLoss` | Student's masked-patch predictions match teacher's unmasked patches. | iBOT (2022) |
| `KoLeoLoss` | Regularizer pushing feature vectors apart in a batch — prevents collapse. | DINOv2 (2023) |
| `GramLoss` | Aligns student's patch-token Gram matrix to a frozen "Gram teacher" — new in DINOv3, prevents dense-feature degradation late in training. | DINOv3 (2025) |

**Note on Gram loss:** it's only enabled in the *second* training stage (`_gram_anchor.yaml` config). For a first from-scratch pass, ignore it — just DINO + iBOT + KoLeo already trains a usable model.

### Stage 5 — SSL wiring (`train/ssl_meta_arch.py`)

This is where student and teacher meet. The pattern:

1. Build a student ViT + heads (DINO head, iBOT head, optional projection heads).
2. Build a teacher as an EMA-updated deep copy of the student.
3. Forward pass:
   - Teacher sees global crops (no grad, `torch.no_grad()`).
   - Student sees global + local crops.
   - Compute all losses, sum with weights from config.
4. Backward on student only.
5. After optimizer step, EMA-update teacher: `θ_teacher ← m·θ_teacher + (1−m)·θ_student` (m ~0.994 → 1.0 over training).

Skip FSDP / multi-node distributed sharding on your first pass — those live in `fsdp/` and `distributed/` and are performance code, not model code. Use plain `DistributedDataParallel` or single-GPU.

### Stage 6 — Training loop (`train/train.py`)

Minimal loop:

```
for it, batch in enumerate(loader):
    schedule = get_schedules(it)          # lr, wd, momentum, teacher_temp
    apply_schedules(optimizer, schedule)
    loss = meta_arch(batch)
    loss.backward()
    optimizer.step(); optimizer.zero_grad()
    meta_arch.update_teacher(schedule.momentum)
    if it % save_freq == 0: save_teacher_ckpt()
```

Cosine LR schedule is in `cosine_lr_scheduler.py`. Parameter groups (different WD for norms vs. weights) are in `param_groups.py`.

### Stage 7 — Evaluation (`eval/`)

You don't need to reproduce full ImageNet-1k eval to know the model works. Two cheap sanity checks:

1. **k-NN eval** (`eval/knn.py`) — extract teacher features on a small labeled set, run k-NN classifier. If accuracy climbs during training, SSL is working.
2. **Linear probe** (`eval/linear.py`) — freeze teacher, train a linear layer on top. Same signal, slightly stronger.

For the actual soybean task, this *is* your endpoint: freeze the ViT, add a classification head, train the head on your labeled soybean leaves.

---

## 2. Milestones and how to know each works

| Milestone | Verification |
|-----------|-------------|
| M1 — layers built | Unit-test each layer's output shape; parameter count matches reference. |
| M2 — ViT-S/16 built | Load released `.pth` weights with `strict=True` — no key errors. |
| M3 — Data pipeline | One batch yields correct number of global/local crops with correct shapes and masks. |
| M4 — Losses | Each loss returns a scalar > 0 on random inputs; gradient flows. |
| M5 — Meta-arch | 100 iterations on random data run without NaN; teacher params drift slowly from student. |
| M6 — Real training | Loss decreases on a real dataset over ~1 epoch on a single GPU. |
| M7 — k-NN eval | Feature k-NN accuracy on a held-out split beats random. |
| M8 — Soybean transfer | Linear probe or fine-tune on `data/raw/` beats an ImageNet-pretrained ResNet baseline. |

---

## 3. Recommended pragmatic path for the soybean task

Building DINOv3 from scratch is instructive. Reaching **soybean disease classification results** from scratch is not the fastest path. A defensible workflow:

1. Do stages 1–2 (build ViT + load released weights). You gain full understanding and a working model.
2. Skip pre-training. Use released **DINOv3 ViT-S/16 or ViT-B/16 LVD-1689M** weights.
3. Add a linear or MLP classifier head on top of the CLS token (or GAP over patch tokens).
4. Fine-tune on soybean leaf images from `data/raw/`.
5. If you *want* to do SSL pre-training, do it as **continued pre-training** on unlabeled soybean field imagery — this is much more useful research than replicating LVD-1689M pretraining.

---

## 4. Where each piece lives in this repo

Once you build, wire it into the existing project structure:

- Model code → `soybean_leaf_disease_detection/modeling/` (add e.g. `vit.py`, `losses.py`, `ssl.py`)
- Training entrypoint → extend `soybean_leaf_disease_detection/modeling/train.py`
- Data loading → extend `soybean_leaf_disease_detection/dataset.py`
- Configs → keep hydra-style YAMLs in a new `configs/` folder, mirroring `references/dinov3/dinov3/configs/`
- Notebooks for exploration → `notebooks/`

---

## 5. Useful reading order alongside the code

1. Paper §3 — architecture ↔ `models/vision_transformer.py`
2. Paper §4.1 — DINO/iBOT losses ↔ `loss/dino_clstoken_loss.py`, `loss/ibot_patch_loss.py`
3. Paper §4.2 — Gram anchoring ↔ `loss/gram_loss.py`
4. Paper §4.3 — RoPE and high-res adaptation ↔ `layers/rope_position_encoding.py`
5. Paper §5 — training recipe ↔ `configs/train/*.yaml`, `train/ssl_meta_arch.py`

---

## 6. Common pitfalls

- **Teacher gradient leakage.** Any teacher forward pass must be inside `torch.no_grad()`. Missing this trains the teacher and causes collapse.
- **EMA momentum schedule.** Momentum starts low-ish (~0.994) and warms to ~1.0. A constant momentum causes instability or collapse.
- **Centering / sharpening.** The DINO loss centers teacher outputs (subtracts a running mean) and sharpens with low temperature. Both are required to prevent trivial solutions.
- **Loading pretrained weights.** State-dict keys are your ground truth for correctness. Diff them early and often.
- **Windows paths.** Reference training scripts assume Linux + SLURM. Do experiments in WSL2 or run only the pieces you need (model + eval) natively.

---

## 7. First actionable step

Create `soybean_leaf_disease_detection/modeling/vit_layers.py` and port `patch_embed.py` from the reference. Write a `pytest` that:

1. Instantiates your `PatchEmbed(img_size=224, patch_size=16, embed_dim=384)`
2. Passes a `torch.randn(1, 3, 224, 224)` through it
3. Asserts output shape `[1, 196, 384]`
4. Loads the corresponding weights from the reference `PatchEmbed` and confirms bit-exact output on the same input

If that test passes, you've validated the whole workflow (build → test → weight-parity) on the smallest possible piece. Everything else is the same pattern, scaled up.

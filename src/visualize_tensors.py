"""
Snippet 4 — Tensor Shape Visualisation  (PatchTST v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs a live forward pass and prints + plots every tensor shape.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── config (must match 2_train.py) ────────────────────────────────────────────
DATA_DIR  = r"C:\Work\sorbolo_patchtst\data\processed"
OUT_DIR   = r"C:\Work\sorbolo_patchtst\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

B         = 4        # small batch for the trace
SEQ_LEN   = 60
PATCH_LEN = 10
STRIDE    = 5
N_PATCHES = (SEQ_LEN - PATCH_LEN) // STRIDE + 1   # 11
D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 4
D_FF      = 512
DROPOUT   = 0.10
N_FEAT    = 17       # features from data; overridden below if data is present

# ── try to load real data; fall back to random ─────────────────────────────────
x_path = os.path.join(DATA_DIR, "X_train.npy")
if os.path.exists(x_path):
    arr    = np.load(x_path)
    N_FEAT = arr.shape[2]
    x_real = torch.tensor(arr[:B], dtype=torch.float32)
    print(f"Loaded real data  X_train[:{B}]  shape={tuple(x_real.shape)}")
else:
    x_real = torch.randn(B, SEQ_LEN, N_FEAT)
    print(f"No processed data found — using random input  {tuple(x_real.shape)}")

TRUNK_IN = N_FEAT * N_PATCHES * D_MODEL   # 17 * 11 * 128 = 23936

# ── instrumented model (captures all intermediate shapes) ─────────────────────
class PatchTSTTraced(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed   = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed     = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        self.input_dropout = nn.Dropout(DROPOUT)

        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS,
                                             enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(D_MODEL)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(TRUNK_IN),
            nn.Linear(TRUNK_IN, D_MODEL * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.head1 = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL), nn.GELU(),
            nn.Dropout(DROPOUT * 0.5), nn.Linear(D_MODEL, 1),
        )
        self.head2 = nn.Sequential(
            nn.Linear(D_MODEL * 2, D_MODEL), nn.GELU(),
            nn.Dropout(DROPOUT * 0.5), nn.Linear(D_MODEL, 1), nn.Softplus(),
        )

    def forward(self, x, record):
        record["0_input"] = tuple(x.shape)                           # (B, T, C)

        channel_out = []
        for c in range(N_FEAT):
            xc = x[:, :, c]
            if c == 0:
                record["1_single_channel"] = tuple(xc.shape)         # (B, T)

            p = xc.unfold(1, PATCH_LEN, STRIDE)
            if c == 0:
                record["2_unfold_patches"] = tuple(p.shape)          # (B, P, patch_len)

            p = self.patch_embed(p) + self.pos_embed
            if c == 0:
                record["3_patch_embed+pos"] = tuple(p.shape)         # (B, P, D)

            p = self.input_dropout(p)
            p = self.norm(self.encoder(p))
            if c == 0:
                record["4_after_encoder"]  = tuple(p.shape)          # (B, P, D)

            channel_out.append(p)

        out = torch.stack(channel_out, dim=1)
        record["5_stack_channels"] = tuple(out.shape)                # (B, C, P, D)

        trunk = self.trunk(out)
        record["6_after_trunk"]    = tuple(trunk.shape)              # (B, D*2)

        h1 = self.head1(trunk)
        h2 = self.head2(trunk)
        record["7_head1_out"] = tuple(h1.shape)                      # (B, 1)
        record["8_head2_out"] = tuple(h2.shape)                      # (B, 1)

        return h1, h2

model = PatchTSTTraced()
model.eval()
record = {}
with torch.no_grad():
    h1, h2 = model(x_real, record)

# ── print tensor shape table ───────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  TENSOR SHAPES  (batch={B}, features={N_FEAT})")
print("="*60)
labels = {
    "0_input":           "Raw input window",
    "1_single_channel":  "One channel extracted",
    "2_unfold_patches":  "Unfolded into patches",
    "3_patch_embed+pos": "After patch embed + pos",
    "4_after_encoder":   "After Transformer encoder",
    "5_stack_channels":  "All channels stacked",
    "6_after_trunk":     "After shared trunk",
    "7_head1_out":       "Head-1 output (log-space)",
    "8_head2_out":       "Head-2 output (magnitude)",
}
for key, desc in labels.items():
    shape = record[key]
    print(f"  {desc:<32}  {str(shape):<25}")
print("="*60)

# ── dimension legend ──────────────────────────────────────────────────────────
print(f"""
Dimension key
  B  = {B}         (batch size)
  T  = {SEQ_LEN}        (sequence / look-back days)
  C  = {N_FEAT}        (input features)
  P  = {N_PATCHES}        (patches per channel: (T-patch_len)//stride + 1)
  p  = {PATCH_LEN}        (patch length in days)
  D  = {D_MODEL}       (model / embedding dimension)
  D2 = {D_MODEL*2}       (trunk hidden dim = D*2)
  FF = {D_FF}      (Transformer feed-forward dim)
""")

# ── visualisation ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 18); ax.set_ylim(0, 11)
ax.axis("off")

COLORS = {
    "data":      "#E3F2FD",
    "patch":     "#FFF9C4",
    "encoder":   "#E8F5E9",
    "merge":     "#F3E5F5",
    "trunk":     "#FBE9E7",
    "head":      "#E0F2F1",
    "output":    "#FCE4EC",
    "border":    "#37474F",
    "arrow":     "#546E7A",
    "title":     "#1A237E",
}

def box(ax, x, y, w, h, shape_str, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=COLORS["border"], linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h*0.66, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=COLORS["border"])
    ax.text(x + w/2, y + h*0.26, shape_str,
            ha="center", va="center", fontsize=8,
            color="#BF360C", family="monospace")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"],
                                lw=1.5, connectionstyle="arc3,rad=0"))

# title
ax.text(9, 10.55, "PatchTST v4 — Tensor Shape Flow",
        ha="center", va="center", fontsize=16,
        fontweight="bold", color=COLORS["title"])
ax.text(9, 10.15,
        f"B={B}  T={SEQ_LEN}  C={N_FEAT}  patch_len={PATCH_LEN}  "
        f"stride={STRIDE}  P={N_PATCHES}  D={D_MODEL}  n_layers={N_LAYERS}",
        ha="center", va="center", fontsize=9, color="#37474F")

# ── ROW 1: input ──────────────────────────────────────────────────────────────
box(ax, 6.5, 9.0, 5, 0.75,
    f"({B}, {SEQ_LEN}, {N_FEAT})",
    "Input window  x", COLORS["data"], fontsize=10)

# ── ROW 2: per-channel loop ───────────────────────────────────────────────────
ax.text(9, 8.45, f"Channel-independent loop  ×{N_FEAT}  (one column of x per iteration)",
        ha="center", va="center", fontsize=8.5, color="#4A148C",
        style="italic")

# channel box
loop_rect = mpatches.FancyBboxPatch(
    (0.5, 6.2), 17, 2.0,
    boxstyle="round,pad=0.15",
    facecolor="#F9FBE7", edgecolor="#827717", linewidth=1.5, linestyle="--",
)
ax.add_patch(loop_rect)
ax.text(0.75, 8.05, f"for c in range({N_FEAT}):", fontsize=8,
        color="#827717", fontweight="bold")

box(ax, 1.0, 7.4, 3.5, 0.65,
    f"({B}, {SEQ_LEN})",
    "xc = x[:,:,c]", COLORS["data"])
arrow(ax, 9.0, 9.0, 9.0, 8.55)  # input → loop entrance
arrow(ax, 2.75, 7.4, 2.75, 7.05 + 0.65)  # xc → unfold (adjusted below)

box(ax, 1.0, 6.5, 3.5, 0.65,
    f"({B}, {N_PATCHES}, {PATCH_LEN})",
    f"unfold(patch={PATCH_LEN}, stride={STRIDE})", COLORS["patch"])
arrow(ax, 2.75, 7.4, 2.75, 7.15)

box(ax, 5.5, 6.5, 4.5, 0.65,
    f"({B}, {N_PATCHES}, {D_MODEL})",
    f"patch_embed  Linear({PATCH_LEN}→{D_MODEL})  + pos_embed", COLORS["patch"])
arrow(ax, 4.5, 6.82, 5.5, 6.82)

box(ax, 11.0, 6.5, 6.0, 0.65,
    f"({B}, {N_PATCHES}, {D_MODEL})  ×{N_LAYERS} layers",
    f"TransformerEncoder  (D={D_MODEL}, h={N_HEADS}, FF={D_FF})", COLORS["encoder"])
arrow(ax, 10.0, 6.82, 11.0, 6.82)

# ── ROW 3: stack + trunk ──────────────────────────────────────────────────────
box(ax, 6.0, 5.2, 6.0, 0.75,
    f"({B}, {N_FEAT}, {N_PATCHES}, {D_MODEL})",
    f"torch.stack(channel_out, dim=1)", COLORS["merge"], fontsize=9)
arrow(ax, 14.0, 6.5, 14.0, 6.0)
arrow(ax, 14.0, 6.0, 9.0, 5.95)

# trunk sub-boxes
trunk_x = [3.0, 5.8, 8.5, 11.5, 14.2]
trunk_labels = [
    f"Flatten\n→ ({B}, {TRUNK_IN})",
    f"LayerNorm\n({TRUNK_IN})",
    f"Linear\n{TRUNK_IN}→{D_MODEL*2}",
    f"GELU + Drop\n({B}, {D_MODEL*2})",
]
trunk_colors = [COLORS["trunk"]]*4
trunk_shapes = [
    f"({B}, {TRUNK_IN})",
    f"({B}, {TRUNK_IN})",
    f"({B}, {D_MODEL*2})",
    f"({B}, {D_MODEL*2})",
]

# draw shared trunk label
ax.text(9.0, 4.85, "Shared Trunk", ha="center", va="center",
        fontsize=9, color="#BF360C", fontweight="bold")

trunk_x0 = 0.5
step      = 4.25
for i, (lbl, shp, col) in enumerate(zip(trunk_labels, trunk_shapes, trunk_colors)):
    bx = trunk_x0 + i * step
    box(ax, bx, 3.8, 4.0, 0.85, shp, lbl, col, fontsize=8)
    if i < len(trunk_labels) - 1:
        arrow(ax, bx + 4.0, 4.22, bx + 4.25, 4.22)

arrow(ax, 9.0, 5.2, 9.0, 4.65)

# ── ROW 4: two heads ──────────────────────────────────────────────────────────
# arrow from trunk out to both heads
arrow(ax, 2.5, 3.8, 2.5, 3.35)   # to head1 left
arrow(ax, 15.5, 3.8, 15.5, 3.35) # to head2 right
# re-route: from end of trunk
arrow(ax, 4.5 + 4.0, 4.22, 4.5, 3.35)
arrow(ax, 4.5 + 4.0, 4.22, 13.5, 3.35)

def head_box(ax, x, label, color_bg, head_layers, output_shape, activation_note):
    bx = x
    box(ax, bx, 2.6, 4.0, 0.65,
        f"({B}, {D_MODEL})",
        f"Linear({D_MODEL*2}→{D_MODEL}) + GELU", color_bg, fontsize=8)
    box(ax, bx, 1.8, 4.0, 0.65,
        output_shape,
        f"Linear({D_MODEL}→1){activation_note}", color_bg, fontsize=8)
    box(ax, bx, 0.9, 4.0, 0.70,
        f"({B}, 1)  →  scalar/sample",
        label, COLORS["output"], fontsize=8.5)
    arrow(ax, bx+2, 3.35, bx+2, 2.6+0.65)  # from top
    arrow(ax, bx+2, 2.6, bx+2, 1.8+0.65)
    arrow(ax, bx+2, 1.8, bx+2, 0.9+0.70)

head_box(ax, 0.5,  "Head-1  log₁p(QM) standardised\n(unbounded — low/medium flows)",
         COLORS["head"], [], f"({B}, 1)", "")
head_box(ax, 13.5, "Head-2  QM / QM_train_max\n(Softplus → always ≥ 0, peaks)",
         COLORS["head"], [], f"({B}, 1)", " + Softplus")

# label the two head columns
ax.text(2.5, 3.55, "HEAD 1", ha="center", fontsize=9,
        color="#1B5E20", fontweight="bold")
ax.text(15.5, 3.55, "HEAD 2", ha="center", fontsize=9,
        color="#BF360C", fontweight="bold")

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (COLORS["data"],    "Input / channel slice"),
    (COLORS["patch"],   "Patch embedding stage"),
    (COLORS["encoder"], "Transformer encoder"),
    (COLORS["merge"],   "Channel aggregation"),
    (COLORS["trunk"],   "Shared trunk"),
    (COLORS["head"],    "Per-head layers"),
    (COLORS["output"],  "Output (final shape)"),
]
for i, (c, lbl) in enumerate(legend_items):
    rx, ry = 0.3 + i * 2.55, 0.3
    ax.add_patch(plt.Rectangle((rx, ry), 0.35, 0.35,
                                facecolor=c, edgecolor=COLORS["border"], lw=0.8))
    ax.text(rx + 0.45, ry + 0.17, lbl, va="center", fontsize=7, color="#37474F")

# save
out_path = os.path.join(OUT_DIR, "tensor_flow.png")
fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nDiagram saved: {out_path}")

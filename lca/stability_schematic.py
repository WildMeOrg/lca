"""
Generate a schematic diagram of the LCA Stability-Driven Graph Clustering Algorithm.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 24))
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
C_INIT   = '#E8F4FD'   # light blue  - initialization
C_PH0    = '#FFF3E0'   # light orange - phase 0
C_ACTIVE = '#E8F5E9'   # light green  - active review
C_DONE   = '#F3E5F5'   # light purple - output
C_GRAPH  = '#FAFAFA'   # graph inset background
C_EDGE_P = '#4CAF50'   # positive edge green
C_EDGE_N = '#F44336'   # negative edge red
C_EDGE_I = '#9E9E9E'   # inactive edge grey
C_ARROW  = '#37474F'

def box(x, y, w, h, text, color, fontsize=9, bold=False, align='center'):
    """Draw a rounded box with text."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#455A64', linewidth=1.2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha=align, va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            family='sans-serif')

def arrow(x1, y1, x2, y2, label='', curved=False):
    """Draw an arrow between two points."""
    style = "Simple,tail_width=1,head_width=6,head_length=4"
    if curved:
        arrow_patch = FancyArrowPatch((x1, y1), (x2, y2),
                                       connectionstyle="arc3,rad=0.3",
                                       arrowstyle=style, color=C_ARROW, lw=1.2)
    else:
        arrow_patch = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle=style, color=C_ARROW, lw=1.2)
    ax.add_patch(arrow_patch)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7.5, color='#555', style='italic')

def section_label(x, y, text):
    ax.text(x, y, text, fontsize=12, fontweight='bold', color='#263238',
            family='sans-serif')

# ───────────────────────────── TITLE ─────────────────────────────
ax.text(9, 23.4, 'LCA v3: Stability-Driven Graph Clustering', ha='center',
        fontsize=16, fontweight='bold', color='#1B5E20', family='sans-serif')
ax.text(9, 23.0, 'Schematic Overview', ha='center',
        fontsize=11, color='#555', family='sans-serif')

# ───────────────────── 1. INITIALIZATION ─────────────────────
section_label(0.3, 22.3, '1  Initialization')

box(0.5, 21.2, 3.5, 0.9, 'Load Config\n(YAML)', C_INIT, bold=True)
box(4.5, 21.2, 3.5, 0.9, 'Load Embeddings\n& Annotations', C_INIT, bold=True)
box(8.5, 21.2, 4.0, 0.9, 'Build Classifier\n(miewid threshold)', C_INIT, bold=True)
box(13.0, 21.2, 4.5, 0.9, 'Init Empty\nStabilityGraph', C_INIT, bold=True)

arrow(4.0, 21.65, 4.5, 21.65)
arrow(8.0, 21.65, 8.5, 21.65)
arrow(12.5, 21.65, 13.0, 21.65)

# ───────────────────── 2. INITIAL EDGES ──────────────────────
section_label(0.3, 20.5, '2  Initial Edge Generation')

box(0.5, 19.3, 5.5, 1.0,
    'Compute Pairwise Distances\n(embedding space, top-k neighbors)', C_INIT, fontsize=9)

box(6.5, 19.3, 5.0, 1.0,
    'Classify Edges\nscore > \u03b8 \u2192 POSITIVE\nscore \u2264 \u03b8 \u2192 NEGATIVE', C_INIT, fontsize=9)

box(12.0, 19.3, 5.5, 1.0,
    'Add edges to graph\n\u2192 Form initial PCCs\n(Positive Connected Components)', C_INIT, fontsize=9)

arrow(6.0, 19.8, 6.5, 19.8)
arrow(11.5, 19.8, 12.0, 19.8)

# ───────────────── MINI GRAPH LEGEND ─────────────────────────
# Small legend for edge types
lx, ly = 0.5, 18.3
ax.text(lx, ly + 0.15, 'Edge Legend:', fontsize=8, fontweight='bold', color='#333')
ax.plot([lx+2.0, lx+2.8], [ly+0.2, ly+0.2], color=C_EDGE_P, lw=2.5, solid_capstyle='round')
ax.text(lx+3.0, ly+0.2, 'Positive (same)', fontsize=7.5, va='center', color=C_EDGE_P)
ax.plot([lx+5.5, lx+6.3], [ly+0.2, ly+0.2], color=C_EDGE_N, lw=2.5, linestyle='--')
ax.text(lx+6.5, ly+0.2, 'Negative (diff)', fontsize=7.5, va='center', color=C_EDGE_N)
ax.plot([lx+9.0, lx+9.8], [ly+0.2, ly+0.2], color=C_EDGE_I, lw=2.5, linestyle=':')
ax.text(lx+10.0, ly+0.2, 'Deactivated', fontsize=7.5, va='center', color=C_EDGE_I)

# ═══════════════════ 3. PHASE 0 ═══════════════════════════════
section_label(0.3, 17.7, '3  Phase 0: Automatic 0-Stability  (no human input)')

# Phase 0 steps
y0 = 14.7
box(0.5, y0, 4.0, 2.7,
    'Densify PCCs\n\n'
    'For each PCC:\n'
    '  \u2022 Add all within-PCC\n'
    '    node pairs\n'
    '  \u2022 Classify new edges\n'
    '    as P or N',
    C_PH0, fontsize=8.5)

box(5.0, y0, 4.2, 2.7,
    'Discover Cross-PCC\nEdges\n\n'
    '  \u2022 Sample pairs between\n'
    '    different PCCs\n'
    '  \u2022 Classify as N edges\n'
    '    (expected different)',
    C_PH0, fontsize=8.5)

box(9.7, y0, 4.0, 2.7,
    'Make 0-Stable\n\n'
    'For each N edge (u,v)\n'
    'inside a PCC:\n'
    '  \u2022 Find MST path u\u2192v\n'
    '  \u2022 Get weakest P edge\n'
    '  \u2022 If weak < N conf:\n'
    '    deactivate P edge\n'
    '    \u2192 PCC splits!',
    C_PH0, fontsize=8.5)

box(14.2, y0+0.3, 3.3, 2.1,
    'Convergence?\n\n'
    '\u2022 No new edges\n'
    '\u2022 No deactivations\n'
    '\u2022 OR max iterations',
    C_PH0, fontsize=8.5, bold=False)

arrow(4.5, y0+1.35, 5.0, y0+1.35)
arrow(9.2, y0+1.35, 9.7, y0+1.35)
arrow(13.7, y0+1.35, 14.2, y0+1.35)

# Loop back arrow for Phase 0
arrow(15.85, y0+2.4, 15.85, y0+2.9, label='No')
ax.annotate('', xy=(2.5, y0+2.9), xytext=(15.85, y0+2.9),
            arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1.5))
ax.annotate('', xy=(2.5, y0+2.7), xytext=(2.5, y0+2.9),
            arrowprops=dict(arrowstyle='->', color='#F57C00', lw=1.5))
ax.text(9, y0+3.1, 'iterate', fontsize=8, color='#F57C00', ha='center', style='italic')

# ──── MINI GRAPH ILLUSTRATIONS ────
# Before Phase 0: messy graph
gx, gy = 1.5, 13.1
ax.text(gx+1.5, gy+1.1, 'Before: Mixed PCC', fontsize=7.5, ha='center', fontweight='bold', color='#555')
nodes_before = [(gx+0.3, gy+0.5), (gx+1.0, gy+0.8), (gx+1.7, gy+0.2),
                (gx+2.3, gy+0.7), (gx+2.8, gy+0.4)]
for (nx, ny) in nodes_before:
    ax.plot(nx, ny, 'o', color='#1976D2', markersize=8, zorder=5)
# Positive edges
ax.plot([nodes_before[0][0], nodes_before[1][0]], [nodes_before[0][1], nodes_before[1][1]],
        color=C_EDGE_P, lw=2, zorder=3)
ax.plot([nodes_before[1][0], nodes_before[2][0]], [nodes_before[1][1], nodes_before[2][1]],
        color=C_EDGE_P, lw=2, zorder=3)
ax.plot([nodes_before[2][0], nodes_before[3][0]], [nodes_before[2][1], nodes_before[3][1]],
        color=C_EDGE_P, lw=1.2, zorder=3)  # weak positive
ax.plot([nodes_before[3][0], nodes_before[4][0]], [nodes_before[3][1], nodes_before[4][1]],
        color=C_EDGE_P, lw=2, zorder=3)
# Negative edge crossing
ax.plot([nodes_before[1][0], nodes_before[3][0]], [nodes_before[1][1], nodes_before[3][1]],
        color=C_EDGE_N, lw=2, linestyle='--', zorder=3)
ax.text(gx+1.2, gy+0.45, 'N', fontsize=7, color=C_EDGE_N, fontweight='bold')

# Arrow between mini graphs
ax.annotate('', xy=(7.5, gy+0.5), xytext=(5.5, gy+0.5),
            arrowprops=dict(arrowstyle='->', color='#F57C00', lw=2))
ax.text(6.5, gy+0.7, 'deactivate\nweak P edge', fontsize=7, ha='center', color='#F57C00')

# After Phase 0: split into 2 PCCs
gx2, gy2 = 8.0, 13.1
ax.text(gx2+2.0, gy2+1.1, 'After: Two Stable PCCs', fontsize=7.5, ha='center', fontweight='bold', color='#555')
nodes_after1 = [(gx2+0.3, gy2+0.5), (gx2+1.0, gy2+0.8), (gx2+1.7, gy2+0.2)]
nodes_after2 = [(gx2+2.8, gy2+0.7), (gx2+3.5, gy2+0.4)]
for (nx, ny) in nodes_after1:
    ax.plot(nx, ny, 'o', color='#1976D2', markersize=8, zorder=5)
for (nx, ny) in nodes_after2:
    ax.plot(nx, ny, 'o', color='#E65100', markersize=8, zorder=5)
ax.plot([nodes_after1[0][0], nodes_after1[1][0]], [nodes_after1[0][1], nodes_after1[1][1]],
        color=C_EDGE_P, lw=2, zorder=3)
ax.plot([nodes_after1[1][0], nodes_after1[2][0]], [nodes_after1[1][1], nodes_after1[2][1]],
        color=C_EDGE_P, lw=2, zorder=3)
ax.plot([nodes_after2[0][0], nodes_after2[1][0]], [nodes_after2[0][1], nodes_after2[1][1]],
        color=C_EDGE_P, lw=2, zorder=3)
# Deactivated edge
ax.plot([nodes_after1[2][0], nodes_after2[0][0]], [nodes_after1[2][1], nodes_after2[0][1]],
        color=C_EDGE_I, lw=1.5, linestyle=':', zorder=3)
# Negative still there
ax.plot([nodes_after1[1][0], nodes_after2[0][0]], [nodes_after1[1][1], nodes_after2[0][1]],
        color=C_EDGE_N, lw=2, linestyle='--', zorder=3)

# PCC labels
from matplotlib.patches import Ellipse
e1 = Ellipse((gx2+1.0, gy2+0.5), 2.0, 1.0, fill=False, edgecolor='#1976D2',
             linestyle='-', linewidth=1.2, alpha=0.5)
ax.add_patch(e1)
e2 = Ellipse((gx2+3.15, gy2+0.55), 1.3, 0.7, fill=False, edgecolor='#E65100',
             linestyle='-', linewidth=1.2, alpha=0.5)
ax.add_patch(e2)
ax.text(gx2+1.0, gy2-0.15, 'PCC A', fontsize=7, ha='center', color='#1976D2')
ax.text(gx2+3.15, gy2+0.05, 'PCC B', fontsize=7, ha='center', color='#E65100')

# ═══════════════════ 4. STABILITY FORMULA ═══════════════════
section_label(0.3, 12.4, '4  Stability Definitions')

box(0.5, 11.2, 8.0, 1.0,
    'Internal Stability(u,v) = MSP(u,v) \u2013 conf(N_edge)\n'
    'MSP = Max Strength Path = widest bottleneck through MST',
    '#E3F2FD', fontsize=9)

box(9.0, 11.2, 8.5, 1.0,
    'External Stability(PCC_a, PCC_b) =\n'
    'max(N_conf between PCCs) \u2013 max(deactivated_P_conf)',
    '#E3F2FD', fontsize=9)

# ═══════════════════ 5. ACTIVE REVIEW ═══════════════════════
section_label(0.3, 10.5, '5  Active Review Phase  (human-in-the-loop)')

y_ar = 7.2
box(0.3, y_ar, 3.0, 3.0,
    'Find Unstable\nPairs\n\n'
    '\u2022 Internal: N edges\n'
    '  within PCCs where\n'
    '  stability < \u03b1\n\n'
    '\u2022 External: N edges\n'
    '  between PCCs where\n'
    '  stability < \u03b1',
    C_ACTIVE, fontsize=8.5)

box(3.8, y_ar, 3.0, 3.0,
    'Select Review\nBatch\n\n'
    '\u2022 Sort by stability\n'
    '  (lowest first)\n\n'
    '\u2022 merge_priority:\n'
    '  70% merges\n'
    '  30% splits\n\n'
    '\u2022 ~200 edges/batch',
    C_ACTIVE, fontsize=8.5)

box(7.3, y_ar, 3.2, 3.0,
    'Human Review\n\n'
    'For each edge (u,v):\n'
    '"Are these the\n same individual?"\n\n'
    '  \u2192 Yes (agree)\n'
    '  \u2192 No  (disagree)',
    C_ACTIVE, fontsize=8.5)

box(11.0, y_ar, 3.0, 3.0,
    'Update\nConfidences\n\n'
    'Agree:\n'
    '  conf += ch (0.97)\n\n'
    'Disagree:\n'
    '  conf \u2212= ch\n'
    '  may flip label\n'
    '  P \u2194 N',
    C_ACTIVE, fontsize=8.5)

box(14.5, y_ar, 3.0, 3.0,
    'Re-Stabilize\n\n'
    '\u2022 make_zero_stable()\n'
    '\u2022 Deactivate new\n'
    '  conflicts\n'
    '\u2022 PCCs may split\n'
    '  or merge\n\n'
    'Check: stability\n'
    '\u2265 target_\u03b1 ?',
    C_ACTIVE, fontsize=8.5)

arrow(3.3, y_ar+1.5, 3.8, y_ar+1.5)
arrow(6.8, y_ar+1.5, 7.3, y_ar+1.5)
arrow(10.5, y_ar+1.5, 11.0, y_ar+1.5)
arrow(14.0, y_ar+1.5, 14.5, y_ar+1.5)

# Loop back arrow for active review
ax.annotate('', xy=(16.0, y_ar+3.0), xytext=(16.0, y_ar+3.6),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.annotate('', xy=(1.8, y_ar+3.6), xytext=(16.0, y_ar+3.6),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.annotate('', xy=(1.8, y_ar+3.0), xytext=(1.8, y_ar+3.6),
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5))
ax.text(9, y_ar+3.8, 'iterate until: reviews \u2265 max  OR  stability \u2265 target_\u03b1',
        fontsize=8.5, color='#2E7D32', ha='center', style='italic', fontweight='bold')

# ═══════════════════ 6. GRAPH EVOLUTION ILLUSTRATION ═════════
section_label(0.3, 6.5, '6  Graph Evolution Through the Algorithm')

# Three mini graphs showing evolution
titles = ['Initial Graph\n(after edge generation)',
          'After Phase 0\n(0-stable)',
          'After Active Review\n(\u03b1-stable)']

for i, (gx, title) in enumerate(zip([0.5, 6.2, 12.0], titles)):
    gy_g = 4.6
    rect = FancyBboxPatch((gx, gy_g), 5.0, 1.6,
                          boxstyle="round,pad=0.1",
                          facecolor=C_GRAPH, edgecolor='#90A4AE', linewidth=1)
    ax.add_patch(rect)
    ax.text(gx+2.5, gy_g+1.45, title, ha='center', fontsize=7.5, fontweight='bold', color='#455A64')

    # Nodes
    np.random.seed(42+i)
    if i == 0:
        # Many nodes, one big messy cluster
        pts = [(gx+0.5, gy_g+0.4), (gx+1.2, gy_g+0.9), (gx+1.8, gy_g+0.3),
               (gx+2.5, gy_g+0.7), (gx+3.2, gy_g+0.4), (gx+3.8, gy_g+0.8), (gx+4.4, gy_g+0.5)]
        colors = ['#1976D2']*7
        # All connected positively with some negatives
        pedges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(0,2),(3,5)]
        nedges = [(1,4),(2,5)]
        iedges = []
    elif i == 1:
        pts = [(gx+0.5, gy_g+0.4), (gx+1.2, gy_g+0.9), (gx+1.8, gy_g+0.3),
               (gx+2.5, gy_g+0.7), (gx+3.2, gy_g+0.4), (gx+3.8, gy_g+0.8), (gx+4.4, gy_g+0.5)]
        colors = ['#1976D2','#1976D2','#1976D2','#E65100','#E65100','#7B1FA2','#7B1FA2']
        pedges = [(0,1),(1,2),(3,4),(5,6)]
        nedges = [(2,3),(4,5)]
        iedges = [(0,2),(2,3)]  # deactivated
    else:
        pts = [(gx+0.5, gy_g+0.4), (gx+1.2, gy_g+0.9), (gx+1.8, gy_g+0.3),
               (gx+2.5, gy_g+0.7), (gx+3.2, gy_g+0.4), (gx+3.8, gy_g+0.8), (gx+4.4, gy_g+0.5)]
        colors = ['#1976D2','#1976D2','#1976D2','#1976D2','#E65100','#E65100','#E65100']
        pedges = [(0,1),(1,2),(2,3),(4,5),(5,6)]
        nedges = [(3,4)]
        iedges = []

    for (px, py), c in zip(pts, colors):
        ax.plot(px, py, 'o', color=c, markersize=6, zorder=5)

    for (a,b) in pedges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=C_EDGE_P, lw=1.5, zorder=3)
    for (a,b) in nedges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=C_EDGE_N, lw=1.5, linestyle='--', zorder=3)
    for (a,b) in iedges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=C_EDGE_I, lw=1.5, linestyle=':', zorder=3)

# Arrows between mini graphs
ax.annotate('', xy=(6.2, 5.35), xytext=(5.5, 5.35),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))
ax.annotate('', xy=(12.0, 5.35), xytext=(11.2, 5.35),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5))

# ═══════════════════ 7. KEY CONCEPT ══════════════════════════
section_label(0.3, 3.8, '7  Key Concept: Why Stability Matters')

box(0.5, 2.3, 17.0, 1.3,
    'Stability = confidence that a clustering decision is correct.\n'
    'A POSITIVE edge is "unstable" if a NEGATIVE edge in the same PCC has higher confidence than the weakest link.\n'
    'The algorithm finds these conflicts, resolves them automatically (Phase 0), then asks humans about the hardest cases (Active Review).\n'
    'Final clusters = Positive Connected Components of the stabilized graph.',
    '#FFFDE7', fontsize=9)

# ═══════════════════ 8. OUTPUT ═══════════════════════════════
section_label(0.3, 1.8, '8  Output')

box(0.5, 0.5, 4.0, 1.0, 'clustering.json\ncluster_id \u2192 [nodes]', C_DONE, fontsize=9, bold=True)
box(5.0, 0.5, 4.0, 1.0, 'node2uuid.json\nnode \u2192 annotation ID', C_DONE, fontsize=9, bold=True)
box(9.5, 0.5, 4.0, 1.0, 'graph.json\nedges + labels + conf', C_DONE, fontsize=9, bold=True)
box(14.0, 0.5, 3.5, 1.0, 'metrics.log\nF1, precision, recall', C_DONE, fontsize=9, bold=True)

plt.tight_layout()
plt.savefig('/users/PAS2136/nepove/code/lca/lca/stability_clustering_schematic.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: stability_clustering_schematic.png")

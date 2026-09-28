"""Line-art logo -> satin-column DST digitizer.

Skeletonizes the artwork, turns every stroke into a satin column whose width
follows the artwork (distance transform), with center-walk underlay, pull
compensation, tie-in/tie-off and trims between separate objects.
"""
import sys, math
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import gaussian_filter1d
import pyembroidery as pe
from PIL import Image, ImageDraw

SRC = sys.argv[1]
OUT = sys.argv[2]            # basename without extension
HEIGHT_MM = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

SATIN_SPACING = 0.40   # mm between satin passes
PULL_COMP = 0.15       # mm added on each side
UNDER_LEN = 1.8        # mm center-walk underlay stitch length
ZZ_UNDER_MIN_W = 3.0   # mm: wider columns also get zigzag underlay
TRIM_DIST = 1.5        # mm: longer gaps get a trim
MIN_W = 0.9            # mm: narrowest satin allowed
MAX_W = 7.0            # mm: widest satin allowed

# ---------------------------------------------------------------- image
img = np.array(Image.open(SRC).convert("L"))
mask = img < 128
mask = remove_small_objects(mask, max_size=30)
H, W = mask.shape
S = HEIGHT_MM / H                          # mm per pixel
dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
skel = skeletonize(mask)

# ---------------------------------------------------------------- skeleton graph
ys, xs = np.nonzero(skel)
pix = set(zip(ys.tolist(), xs.tolist()))
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def nbrs(p):
    y, x = p
    return [(y + dy, x + dx) for dy, dx in N8 if (y + dy, x + dx) in pix]

deg = {p: len(nbrs(p)) for p in pix}
nodes = {p for p in pix if deg[p] != 2}

# group adjacent junction pixels into one node id
node_id = {}
nid = 0
for p in nodes:
    if p in node_id:
        continue
    stack = [p]
    node_id[p] = nid
    while stack:
        q = stack.pop()
        for r in nbrs(q):
            if r in nodes and r not in node_id and deg[r] > 2 and deg[q] > 2:
                node_id[r] = nid
                stack.append(r)
    nid += 1

edges = []   # (n0, n1, [pixels])
visited_steps = set()
for p in nodes:
    for q in nbrs(p):
        if q in nodes:
            if node_id[p] != node_id[q] and (p, q) not in visited_steps:
                visited_steps.add((p, q)); visited_steps.add((q, p))
                edges.append([node_id[p], node_id[q], [p, q]])
            continue
        if (p, q) in visited_steps:
            continue
        path = [p, q]
        visited_steps.add((p, q))
        prev, cur = p, q
        while cur not in nodes:
            nxt = [r for r in nbrs(cur) if r != prev and (cur, r) not in visited_steps]
            if not nxt:
                break
            # prefer 4-neighbours to avoid diagonal shortcuts
            nxt.sort(key=lambda r: abs(r[0] - cur[0]) + abs(r[1] - cur[1]))
            r = nxt[0]
            visited_steps.add((cur, r)); visited_steps.add((r, cur))
            path.append(r)
            prev, cur = cur, r
        visited_steps.add((path[-1], path[-2]))
        end = node_id.get(cur, None)
        if end is None:
            continue
        edges.append([node_id[p], end, path])

# isolated loops (no nodes at all, e.g. a perfect ring)
covered = set(p for e in edges for p in e[2])
for p in pix:
    if p in covered or p in nodes:
        continue
    loop = [p]; covered.add(p); cur = p
    while True:
        nx = [r for r in nbrs(cur) if r not in covered]
        if not nx:
            break
        cur = nx[0]; covered.add(cur); loop.append(cur)
    if len(loop) > 10:
        loop.append(loop[0])
        n = nid; nid += 1
        edges.append([n, n, loop])

def node_degree(eds):
    d = {}
    for a, b, _ in eds:
        d[a] = d.get(a, 0) + 1
        d[b] = d.get(b, 0) + 1
    return d

# prune spurs: short edges ending in a tip
for _ in range(3):
    d = node_degree(edges)
    keep = []
    for e in edges:
        a, b, pth = e
        tip = (d[a] == 1) != (d[b] == 1)
        if tip:
            j = pth[0] if d[a] > 1 else pth[-1]
            r = dist[j]
            if len(pth) < 1.6 * r + 2:
                continue
        keep.append(e)
    edges = keep

# merge degree-2 chains and pair straightest edges through junctions
def direction(pth, at_start, k=12):
    seg = pth[:k] if at_start else pth[::-1][:k]
    p0, p1 = np.array(seg[0], float), np.array(seg[-1], float)
    v = p1 - p0
    n = np.linalg.norm(v)
    return v / n if n else v

strokes = []
used = [False] * len(edges)
inc = {}
for i, (a, b, _) in enumerate(edges):
    inc.setdefault(a, []).append((i, True))
    inc.setdefault(b, []).append((i, False))

# at each node, pair incident edge-ends whose directions are nearly opposite
pair = {}
for n, ends in inc.items():
    cands = []
    for x in range(len(ends)):
        for y in range(x + 1, len(ends)):
            (i, si), (j, sj) = ends[x], ends[y]
            if i == j:
                continue
            di = direction(edges[i][2], si)
            dj = direction(edges[j][2], sj)
            c = float(np.dot(di, dj))       # -1 = perfectly straight through
            if c < -0.6:
                cands.append((c, ends[x], ends[y]))
    cands.sort()
    taken = set()
    for c, ex, ey in cands:
        if ex in taken or ey in taken:
            continue
        taken.add(ex); taken.add(ey)
        pair[(n, ex)] = ey
        pair[(n, ey)] = ex

def end_node(i, at_start):
    return edges[i][0] if at_start else edges[i][1]

for i in range(len(edges)):
    if used[i]:
        continue
    used[i] = True
    chain = list(edges[i][2])
    # extend forward (from end of edge i)
    cur_i, cur_start = i, False
    while True:
        n = end_node(cur_i, cur_start)
        nxt = pair.get((n, (cur_i, cur_start)))
        if not nxt or used[nxt[0]]:
            break
        j, sj = nxt
        used[j] = True
        p = edges[j][2] if sj else edges[j][2][::-1]
        chain += p[1:]
        cur_i, cur_start = j, not sj
    # extend backward (from start of edge i)
    cur_i, cur_start = i, True
    while True:
        n = end_node(cur_i, cur_start)
        nxt = pair.get((n, (cur_i, cur_start)))
        if not nxt or used[nxt[0]]:
            break
        j, sj = nxt
        used[j] = True
        p = edges[j][2][::-1] if sj else edges[j][2]
        chain = p[:-1] + chain
        cur_i, cur_start = j, sj
    strokes.append(chain)

# tiny blobs with no skeleton edges (dots)
lab_n, labels = cv2.connectedComponents(mask.astype(np.uint8))
stroke_labels = set()
for s in strokes:
    for p in s:
        stroke_labels.add(labels[p])
for l in range(1, lab_n):
    if l in stroke_labels:
        continue
    yy, xx = np.nonzero(labels == l)
    if len(yy) < 30:
        continue
    # principal axis
    c = np.array([yy.mean(), xx.mean()])
    cov = np.cov(np.vstack([yy - c[0], xx - c[1]]))
    w, v = np.linalg.eigh(cov)
    ax = v[:, 1]
    proj = (np.vstack([yy, xx]).T - c) @ ax
    a, b = proj.min(), proj.max()
    strokes.append([tuple((c + ax * t).round().astype(int)) for t in np.linspace(a, b, 8)])

junction_pts = np.array([p for p in pix if deg[p] > 2], float) if any(deg[p] > 2 for p in pix) else np.zeros((0, 2))

# ---------------------------------------------------------------- satin geometry
def resample(pts, step):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] < 1e-6:
        return pts[:1], d[:1]
    n = max(2, int(math.ceil(d[-1] / step)) + 1)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], t

def local_width(p):
    y, x = int(round(p[0])), int(round(p[1]))
    y = min(max(y, 0), H - 1); x = min(max(x, 0), W - 1)
    # search nearby for the skeleton max (path may be smoothed off-center)
    win = dist[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    return 2 * float(win.max())

def build_column(stroke):
    pts = np.array(stroke, float)
    closed = len(pts) > 10 and np.linalg.norm(pts[0] - pts[-1]) < 2
    if len(pts) >= 5:
        sig = 3.0
        mode = "wrap" if closed else "nearest"
        pts = np.c_[gaussian_filter1d(pts[:, 0], sig, mode=mode),
                    gaussian_filter1d(pts[:, 1], sig, mode=mode)]
    # widths in px along stroke
    wpx = np.array([local_width(p) for p in pts])
    # near junctions the inscribed circle is bigger than the stroke: clamp
    if len(junction_pts):
        from scipy.spatial import cKDTree
        tree = cKDTree(junction_pts)
        dj, _ = tree.query(pts)
        med = np.median(wpx[dj > wpx]) if np.any(dj > wpx) else np.median(wpx)
        near = dj < wpx * 1.1
        wpx[near] = np.minimum(wpx[near], med * 1.1)
    wpx = gaussian_filter1d(wpx, 2, mode="nearest") if len(wpx) > 3 else wpx
    if not closed and len(pts) >= 2:
        # extend both ends by local radius so satin reaches the artwork edge
        # (and overlaps into strokes it joins)
        for end in (0, -1):
            nb = 1 if end == 0 else -2
            k = min(6, len(pts) - 1)
            ref = pts[k] if end == 0 else pts[-1 - k]
            v = pts[end] - ref
            n = np.linalg.norm(v)
            if n == 0:
                continue
            v /= n
            # march along the tangent while still inside the artwork, so
            # intentional gaps in the design stay open
            ext = 0.0
            while ext < wpx[end]:
                q = pts[end] + v * (ext + 1)
                qy, qx = int(round(q[0])), int(round(q[1]))
                if not (0 <= qy < H and 0 <= qx < W and mask[qy, qx]):
                    break
                ext += 1
            ext = max(0.0, ext - PULL_COMP / S)
            if end == 0:
                pts = np.vstack([pts[0] + v * ext, pts]); wpx = np.r_[wpx[0] * 0.8, wpx]
            else:
                pts = np.vstack([pts, pts[-1] + v * ext]); wpx = np.r_[wpx, wpx[-1] * 0.8]
    mm = pts[:, ::-1] * S                       # (x, y) in mm
    wmm = np.clip(wpx * S, MIN_W, MAX_W)
    return mm, wmm, closed

def normals(p):
    t = np.gradient(p, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1
    t = t / n
    return np.c_[-t[:, 1], t[:, 0]]

def column_stitches(mm, wmm):
    """underlay forward, satin back. returns list of (x,y) mm."""
    out = []
    # center walk underlay
    u, _ = resample(mm, UNDER_LEN)
    out += [tuple(p) for p in u]
    # satin (dense) sampled on centerline
    c, t = resample(mm, SATIN_SPACING / 2)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(mm, axis=0), axis=1))]
    w = np.interp(t, d, wmm)
    nrm = normals(c)
    if w.mean() >= ZZ_UNDER_MIN_W:
        cz, tz = resample(mm, 1.6)
        wz = np.interp(tz, d, wmm) * 0.5 - 0.4
        nz = normals(cz)
        zz = []
        for i in range(len(cz) - 1, -1, -1):
            s = 1 if i % 2 else -1
            zz.append(tuple(cz[i] + nz[i] * s * wz[i] / 2))
        out += zz
        rng = range(len(c))
    else:
        rng = range(len(c) - 1, -1, -1)
    half = w / 2 + PULL_COMP
    for k, i in enumerate(rng):
        s = 1 if k % 2 else -1
        out.append(tuple(c[i] + nrm[i] * s * half[i]))
    return out

columns = []
for s in strokes:
    if len(s) < 2:
        continue
    mm, wmm, closed = build_column(s)
    if len(mm) < 2:
        continue
    st = column_stitches(mm, wmm)
    columns.append(st)

# ---------------------------------------------------------------- ordering
# work from the top (arch) down to the text, nearest neighbour in between
def ends(col):
    return np.array(col[0]), np.array(col[-1])

remaining = list(range(len(columns)))
order = []
cur = np.array([W * S / 2, 0.0])
while remaining:
    best, bd, rev = None, 1e18, False
    for i in remaining:
        a, b = ends(columns[i])
        da = np.linalg.norm(a - cur)
        if da < bd:
            best, bd, rev = i, da, False
    remaining.remove(best)
    col = columns[best]
    order.append(col)
    cur = np.array(col[-1])

# ---------------------------------------------------------------- pattern
pat = pe.EmbPattern()
pat.add_thread(pe.EmbThread(0x28324F))
cx, cy = W * S / 2, H * S / 2
U = 10.0  # 0.1 mm units

def P(p):
    return (p[0] - cx) * U, (p[1] - cy) * U

def tie(pt, nxt):
    v = np.array(nxt) - np.array(pt)
    n = np.linalg.norm(v)
    v = v / n * 0.5 if n > 0 else np.array([0.5, 0])
    return [tuple(np.array(pt) + v), tuple(pt), tuple(np.array(pt) + v), tuple(pt)]

last = None
for col in order:
    start = col[0]
    if last is None or np.linalg.norm(np.array(start) - np.array(last)) > TRIM_DIST:
        if last is not None:
            for p in tie(last, col[-1])[:3]:
                pat.add_stitch_absolute(pe.STITCH, *P(p))
            pat.add_command(pe.TRIM)
        pat.add_stitch_absolute(pe.JUMP, *P(start))
        for p in tie(start, col[1] if len(col) > 1 else start):
            pat.add_stitch_absolute(pe.STITCH, *P(p))
    for p in col:
        pat.add_stitch_absolute(pe.STITCH, *P(p))
    last = col[-1]
for p in tie(last, order[-1][-2]):
    pat.add_stitch_absolute(pe.STITCH, *P(p))
pat.add_command(pe.TRIM)
pat.end()

settings = {"max_stitch": 70, "max_jump": 121, "long_stitch_contingency": pe.CONTINGENCY_LONG_STITCH_SEW_TO}
pe.write_dst(pat, OUT + ".dst", settings)

# ---------------------------------------------------------------- preview
chk = pe.read_dst(OUT + ".dst")
Z = 12  # px per mm
pad = 5
bx = chk.bounds()
img_w = int((bx[2] - bx[0]) / U * Z) + 2 * pad * Z
img_h = int((bx[3] - bx[1]) / U * Z) + 2 * pad * Z
prev = Image.new("RGB", (img_w, img_h), (245, 243, 238))
dr = ImageDraw.Draw(prev)
def Q(x, y):
    return ((x - bx[0]) / U * Z + pad * Z, (y - bx[1]) / U * Z + pad * Z)
stitches = chk.stitches
count = 0
for i in range(1, len(stitches)):
    x0, y0, c0 = stitches[i - 1]
    x1, y1, c1 = stitches[i]
    if (c1 & pe.COMMAND_MASK) == pe.STITCH and (c0 & pe.COMMAND_MASK) == pe.STITCH:
        dr.line([Q(x0, y0), Q(x1, y1)], fill=(40, 50, 79), width=max(1, int(0.3 * Z)))
    if (c1 & pe.COMMAND_MASK) == pe.STITCH:
        count += 1
prev.save(OUT + "_preview.png")

trims = sum(1 for s in stitches if (s[2] & pe.COMMAND_MASK) == pe.TRIM)
print(f"size: {(bx[2]-bx[0])/U:.1f} x {(bx[3]-bx[1])/U:.1f} mm, stitches: {count}, trims: {trims}, columns: {len(columns)}")

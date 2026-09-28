import sys, numpy as np, pyembroidery as pe
from PIL import Image, ImageDraw, ImageFilter
dst, out = sys.argv[1], sys.argv[2]
Z = int(sys.argv[3]) if len(sys.argv) > 3 else 20
p = pe.read_dst(dst); b = p.bounds(); pad = 4
w = int((b[2]-b[0])/10*Z)+2*pad*Z; h = int((b[3]-b[1])/10*Z)+2*pad*Z
im = Image.new('RGB', (w, h), (238, 236, 230)); d = ImageDraw.Draw(im)
f = lambda s: ((s[0]-b[0])/10*Z+pad*Z, (s[1]-b[1])/10*Z+pad*Z)
st = p.stitches
tw = max(2, int(0.32*Z))
for i in range(1, len(st)):
    a, c = st[i-1], st[i]
    if (a[2]&pe.COMMAND_MASK)==pe.STITCH and (c[2]&pe.COMMAND_MASK)==pe.STITCH:
        d.line([f(a), f(c)], fill=(24, 30, 52), width=tw+2)
for i in range(1, len(st)):
    a, c = st[i-1], st[i]
    if (a[2]&pe.COMMAND_MASK)==pe.STITCH and (c[2]&pe.COMMAND_MASK)==pe.STITCH:
        d.line([f(a), f(c)], fill=(52, 64, 102), width=tw)
im.save(out)

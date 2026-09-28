import sys, numpy as np, cv2, pyembroidery as pe
from PIL import Image, ImageDraw
src, dst, hmm = sys.argv[1], sys.argv[2], float(sys.argv[3])
m = np.array(Image.open(src).convert('L')) < 128
H, W = m.shape; S = hmm / H
p = pe.read_dst(dst)
im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
st = p.stitches
# dst is centered: shift back
xs=[s[0] for s in st]; ys=[s[1] for s in st]
for i in range(1, len(st)):
    a, b = st[i-1], st[i]
    if (a[2]&pe.COMMAND_MASK)==pe.STITCH and (b[2]&pe.COMMAND_MASK)==pe.STITCH:
        f = lambda s: (s[0]/10/S + W/2, s[1]/10/S + H/2)
        d.line([f(a), f(b)], fill=255, width=max(1,int(0.25/S)))
c = np.array(im) > 0
c = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7,7),np.uint8)) > 0
print('coverage of art: %.3f  stitches outside art: %.3f' % ((c&m).sum()/m.sum(), (c&~m).sum()/c.sum()))
vis = np.zeros((H, W, 3), np.uint8) + 255
vis[m & ~c] = (220, 40, 40)     # missed art: red
vis[c & ~m] = (40, 120, 220)    # spill: blue
vis[c & m] = (60, 60, 60)
Image.fromarray(vis).save(dst.replace('.dst', '_check.png'))

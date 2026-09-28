# Embroidery (DST) digitizer

Turns single-colour line-art logos into satin-stitch DST files for the
Voltex PM-1501R (reads Tajima `.dst` from USB).

```
pip install pyembroidery opencv-python-headless scikit-image scipy pillow numpy
python3 digitize.py source.png out_name 120     # last arg = design height in mm
python3 preview.py out_name.dst preview.png     # stitch preview
python3 check.py source.png out_name.dst 120    # coverage vs artwork
```

Settings (top of `digitize.py`): satin spacing 0.40 mm, pull compensation
0.15 mm per side, center-walk underlay (+ zigzag underlay for columns ≥ 3 mm),
tie-in/tie-off and trim for gaps > 1.5 mm.

## Designs
- `khane-mixology/` — خانه میکسولوژی ایران, 120 mm and 90 mm tall, 1 colour.

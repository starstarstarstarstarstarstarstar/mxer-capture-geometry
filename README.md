# mxer-capture-geometry

An extract of one file, `mxer_sim/capture_geometry.py`, from the Space-Tether
repository at commit `354dbed`. The file is unchanged. It was extracted so the
module can be ported and verified independently.

The module maps a payload's entry position and velocity at a conical funnel to
one of four outcomes (`MISS`, `GUIDED`, `THROAT_ENTRY`, `LATCH_CAPTURED`) using
a fixed-step Euler integration of the radial motion.

`mxer_sim/__init__.py`, `tests/__init__.py` and `tests/test_capture_geometry.py`
were added for this extract.

## Tests

Python 3 standard library only:

```
python3 -m unittest discover -s . -t . -p 'test*.py'
```

# FourHash

This repository contains a compact implementation of RZ-Merge, a serving
layer for merging results from two frozen binary indexes, and its optional
RZ-Shell tie refinement.

Each index keeps its native Hamming ranking. A fixed, unlabeled reference
bank supplies query-conditioned radius moments, recovered from precompiled
first- and second-order bit statistics. The resulting strict float32 lookup
tables change only cross-index comparisons; items at the same radius remain
tied.

RZ-Shell can use a candidate-level CSLS key inside one native
``(stream, Hamming radius)`` shell. It never changes a strict RZ comparison,
and it leaves coalesced scores from different shells tied.

## Install

```bash
python -m pip install -r requirements.txt
```

## Test

```bash
python -m unittest tests.test_rz_merge
```

## Minimal use

```python
import numpy as np

from src.rz_merge import compile_reference, dual_radius_tables, merge_two_lists

image_bank = np.random.choice((-1, 1), size=(1000, 64))
text_bank = np.random.choice((-1, 1), size=(1000, 64))
query = np.random.choice((-1, 1), size=64)

image_table, text_table = dual_radius_tables(
    query,
    compile_reference(image_bank),
    compile_reference(text_bank),
)

topk = merge_two_lists(
    image_ids=[10, 11],
    image_radii=[4, 7],
    text_ids=[20, 21],
    text_radii=[5, 6],
    image_table=image_table,
    text_table=text_table,
    k=3,
)
```

The paper and experimental outputs are maintained separately from this public
code repository.

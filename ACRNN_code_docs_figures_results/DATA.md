# Data Notice

This repository does not include the DEAP dataset files. DEAP requires separate
access from the dataset provider, and the raw `.dat` files should not be
committed to GitHub.

Expected local layout for running the scripts:

```text
data/
  deap_shuffled_data_3s/
    s01.dat
    ...
    s32.dat
```

For the stricter trial-level split experiment, the scripts also expect the
original DEAP raw `.dat` files when using `--split-protocol trial`.

```text
data/
  data_preprocessed_python/
    s01.dat
    ...
    s32.dat
```

Use `code/deap_pre_process_from_dat.py` to generate the 3-second shuffled
segments used by the sample-level reproduction experiments.

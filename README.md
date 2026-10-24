# Validity forgetting

Code for *"Validity forgetting: continual learners keep their accuracy but lose their
uncertainty guarantees"* (submitted to *Neurocomputing*).

Class-incremental learners are evaluated on accuracy. This repository measures a second
quantity — whether a split-conformal prediction set still covers the true label at the
promised rate on earlier tasks — and implements Reserved-Buffer Conformal Calibration (RBCC),
which reserves a uniformly sampled, never-trained-on part of an existing replay memory for
calibration and supports per-class (Mondrian) thresholds.

**Data.** Pre-extracted features, every run record, every derived table and the end-to-end
training logs are archived on Zenodo: **https://doi.org/10.5281/zenodo.23612863**. The `results/` folder here is the same
set of records and tables, so every number and figure in the paper can be checked without
downloading anything else.

## Layout

```
src/                     measurement module, methods, experiment drivers
  reliability.py         split conformal prediction (LAC, APS), coverage, ECE, AURC
  cl_methods.py          Seq-FT, EWC, LwF, experience replay, iCaRL, NCM, RanPAC; RBCC reservation
  experiment.py          one run: train along the stream, calibrate under four protocols, evaluate
  sweep.py               the 280-run study on frozen pre-trained features
  extract_features.py    ViT-B/16 (in21k) and DINOv2-B feature extraction
  make_medmix26_split.py class-stratified re-split of the MedMix-26 benchmark
  derive_main_tables.py  Tables 2-7 from results/main/sweep_records.csv
  e2e_cl.py              end-to-end ResNet-18 experiments (GPU)
  figures/               figure builders; build_figures.py rebuilds Figs. 1-5 from results/
results/                 run records and derived tables (identical to the Zenodo deposit)
figures/                 Figures 1-5 (PDF) and the four schematic panels
scripts/run_e2e_rbcc.sh  end-to-end RBCC, five class orderings in parallel on one GPU
```

## Quick check (minutes, CPU)

```bash
pip install -r requirements.txt
# every main-results table from the released run records
mkdir -p /tmp/check && cp results/main/sweep_records.csv /tmp/check/
python src/derive_main_tables.py /tmp/check            # compare with results/main/*.csv
# every figure from results/ (Arial required; pass a folder of Arial TTFs if not installed)
VF_ROOT=$PWD python src/figures/build_figures.py --fonts /path/to/arial
```

## Reproducing from features (hours, CPU)

Download `features/` from Zenodo into `src/features/` (or extract them yourself with
`extract_features.py` and `make_medmix26_split.py`). Then, from `src/`:

```bash
python sweep.py cifar100,cub200,imagenet_r vit_b16_in21k,dinov2_base && mv sweep_records.csv sweep_records_sweep.csv
python run_main_memory_methods.py .          # Replay and iCaRL, slots reserved before training
python run_main_medmix26.py .                # all methods on MedMix-26
python run_reservation_size.py .             # reservation size k (Section 3.10)
python run_reservation_size_medmix26.py .
python assemble_records.py .                 # -> sweep_records.csv, reservation_size_records.csv
python derive_main_tables.py .
python run_extra_buffer.py .                 # extra calibration buffer (Section 3.8)
python exp_grouped_ts.py .                   # per-task temperature scaling (Section 3.7)
python exp_score.py                          # LAC vs APS (Section 3.6)
HOLD_OUT=1 TAG=replay20_ho python exp_class_conditional.py features/cifar100__vit_b16_in21k__classcond.npz   # Section 3.12
python exp_reservation_timing.py features/cifar100__vit_b16_in21k__classcond.npz                             # Section 2.5
```

The class-conditional experiments loop over replay budgets through the `REPLAY_SLOTS` and `KS`
environment variables (see the header of `exp_class_conditional.py`);
`agg_class_conditional.py` collects their outputs.

## End-to-end experiments (GPU)

Place the official CIFAR-100 python archive under `src/data/cifar100/cifar-100-python/`, then
`bash scripts/run_e2e_rbcc.sh` for RBCC and `MATRIX=full EPOCHS=150 python src/e2e_cl.py` for
the other methods. `summarise_e2e.py` and `compare_e2e.py` produce the end-to-end tables.

## Licence and citation

Code: MIT (`LICENSE`). Data: CC BY 4.0 (Zenodo). Please cite the article; `CITATION.cff`
will carry the final reference.

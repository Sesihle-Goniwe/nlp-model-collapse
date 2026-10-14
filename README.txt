Interpretability of Model Collapse
COMS4054A / COMS7062A / COMS7066A - Natural Language Processing, Project 1
University of the Witwatersrand

------------------------------------------------------------
WHAT THIS PROJECT DOES
------------------------------------------------------------
We train a small transformer on the symbol task from Singh et al. (2024).
The model sees symbol-label pairs and must give the label of a repeated
symbol. It does this with an induction head.

We then train new models on data made by older models (model collapse).
We check if the induction head circuit gets weaker over generations.

There are two versions:
  - Base:     the model predicts one label.
  - Extended: the model predicts the label, the next symbol, and that
              symbol's label.

------------------------------------------------------------
SETUP
------------------------------------------------------------
1. Use Python 3.10 or newer.
2. Make a virtual environment (optional but recommended):
       python -m venv venv
       source venv/bin/activate        (Windows: venv\Scripts\activate)
3. Install the packages:
       pip install -r requirements.txt

All code runs on a CPU. A GPU makes training faster but is not needed.

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
Put data.py, model.py and train.py in the same folder, and run all
commands from that folder. train.py uses data.py and model.py by itself,
so Steps 1a and 1b are only quick checks.

Step 1a - Check the data (a few seconds):
       python data.py
   This prints example sequences for the base and extended tasks.
   Expected first line:  S810 L13 S1516 L7 S226 L26 S1514 L9 | S1516 L7
   Expected:             train/test overlap: 0

Step 1b - Check the model (a few seconds):
       python model.py
   Expected: parameters: 242368, and an untrained loss of about 7.4
   (this is guessing level, which is correct before training).

Step 2 - Train the base model (about 2 to 5 minutes on a laptop CPU):
       python train.py
   Saves the model to results/base_single/model.pt and the training
   curves to results/figures/base_single_curves.png.

   Expected results (our run, CPU):
       step    val acc   prev-token score   induction score
          0      0.00         0.26               0.11
        800      0.30         0.29               0.27
       1000      0.96         0.72               0.90
       4000      0.99         0.97               0.97
       final test accuracy: about 0.997

   Small differences are normal (another PyTorch version, or a GPU).
   What matters: accuracy stays low, then jumps suddenly around step
   1000, ends near 0.99, and both circuit scores end above about 0.9.

   Tuning sweep (learning rate x model width, picked by validation loss;
   the test split is used once, for the best setting only; about 15 to
   30 minutes):
       python train.py --mode sweep

   Extended task, started from the trained base model:
       python train.py --n_queries 2 --init_from results/base_single/model.pt --out_dir results/extended_warm

Step 3 - Run the model collapse loop:

   Base task (10 generations):
       python collapse.py --task base --init_from results/base_single/model.pt --generations 10

   This uses the trained base model as Generation 0. Each later generation is
   trained on synthetic answers produced by the previous generation.

   Base collapse results are saved to:
       results/collapse/base/

   Extended task:
   First create the warm-started Generation 0 extended model:
       python train.py --n_queries 2 --init_from results/base_single/model.pt --out_dir results/extended_warm

   Then run the recursive collapse experiment:
       python collapse.py --task extended --init_from results/extended_warm/model.pt --base_init_from results/base_single/model.pt --generations 10 --out_dir results/collapse/extended_warmstart

   For the extended task, every later generation is warm-started from the same
   trained base checkpoint, while its synthetic training data is produced by
   the previous extended generation. This is necessary because the extended
   task does not reliably learn the induction circuit from random
   initialization.

   collapse.py evaluates every generation on the same real validation/test
   distribution and records:
       - loss and perplexity
       - overall and answer accuracy
       - previous-token and induction scores
       - for the extended task: next-answer accuracy, q2 context-slot
         distribution, q2 entropy, q2 symbol diversity and top-symbol share

   Extended collapse results are saved to:
       results/collapse/extended_warmstart/

   Each collapse output folder contains collapse_metrics.json and one folder
   per generation with the saved model checkpoint and training history.

Step 4 - Run the interpretability analysis:
       (to be added: interpret/)

All results are saved in results/. All runs use fixed random seeds,
so results repeat on the same machine and package versions.

------------------------------------------------------------
FILES
------------------------------------------------------------
data.py          Makes the symbol task data and the train/val/test split.
model.py         The 2-layer attention-only transformer.
train.py         Trains and tunes the model. Logs loss, accuracy,
                 perplexity and two circuit scores.
collapse.py      The model collapse loop (to be added).
interpret/       Attention maps, embeddings and circuit analysis (to be added).
results/         Plots and logs.

------------------------------------------------------------
NOTES ON DESIGN CHOICES
------------------------------------------------------------
- We use 4 symbol-label pairs, not 2. With 2 pairs at fixed positions,
  the model can solve the task with a position-based shortcut instead of
  an induction head (99% accuracy, but induction score 0.50). With 4 pairs
  the induction circuit forms reliably.
- The extended task does not learn from scratch in our runs. We start it
  from the trained base model (warm start).
- The next-symbol accuracy in the extended task is at most about 25%,
  because the true next symbol is picked at random from the 4 context
  symbols. Look at its distribution, not its accuracy.

------------------------------------------------------------
REFERENCES
------------------------------------------------------------
Singh et al. (2024). What needs to go right for an induction head?
  Original code: https://github.com/aadityasingh/icl-dynamics
Shumailov et al. (2024). AI models collapse when trained on recursively
  generated data. Nature.

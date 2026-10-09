"""
Data for the in-context learning symbol task (Singh et al., 2024 style).

Each sequence looks like this (2 pairs, base task):

    s1 l1 s2 l2 | q a

    s1, s2 : symbols (all different)
    l1, l2 : labels (numbers), all different
    q      : one of the context symbols
    a      : the label paired with q  <- the model must predict this

The symbol-label pairing is random in every sequence, so the model cannot
memorise it. It has to look back in the context (in-context learning).

Extended task (n_queries = 2):

    s1 l1 s2 l2 | q a q2 a2

After the first answer, the model also predicts the next symbol q2 and its
label a2. In real data, q2 is drawn uniformly from the context symbols.
In the collapse loop, the model will choose q2 itself, so any bias it has
will show up in the next generation's data.

Token ids:
    0 .. n_symbols-1                     -> symbols
    n_symbols .. n_symbols+n_labels-1    -> labels

The core code uses only NumPy. PyTorch is only needed for `to_torch`.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TaskConfig:
    n_symbols: int = 1600      # size of the symbol set (Omniglot has ~1600 classes)
    n_labels: int = 32         # size of the label set
    n_pairs: int = 4           # symbol-label pairs in the context (2 allows a shortcut, see README)
    n_queries: int = 1         # 1 = base task, 2 = extended task
    zipf_alpha: float = 0.0    # 0 = uniform symbols; >0 = some symbols rarer (Zipf)

    @property
    def vocab_size(self):
        return self.n_symbols + self.n_labels

    @property
    def seq_len(self):
        # full token sequence length (model input is seq_len - 1)
        return 2 * self.n_pairs + 2 * self.n_queries

    def label_token(self, label):
        return self.n_symbols + label

    def is_symbol(self, tok):
        return tok < self.n_symbols

    def check(self):
        assert self.n_pairs >= 1 and self.n_queries in (1, 2)
        assert self.n_labels >= self.n_pairs, "need enough labels to keep them distinct"
        assert self.n_symbols >= self.n_pairs


def symbol_probs(cfg, symbol_pool):
    """Sampling probability for each symbol in the pool."""
    k = len(symbol_pool)
    if cfg.zipf_alpha <= 0:
        return np.full(k, 1.0 / k)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    p = ranks ** (-cfg.zipf_alpha)
    return p / p.sum()


def make_dataset(cfg, n, seed, symbol_pool=None):
    """
    Make n real sequences.

    Returns a dict:
        tokens     int64 [n, seq_len]   full sequences
        loss_mask  bool  [n, seq_len-1] True where the target (tokens[:, 1:]) is scored
    """
    cfg.check()
    rng = np.random.default_rng(seed)
    if symbol_pool is None:
        symbol_pool = np.arange(cfg.n_symbols)
    symbol_pool = np.asarray(symbol_pool)
    p = symbol_probs(cfg, symbol_pool)

    P, Q = cfg.n_pairs, cfg.n_queries
    tokens = np.empty((n, cfg.seq_len), dtype=np.int64)

    for i in range(n):
        syms = rng.choice(symbol_pool, size=P, replace=False, p=p)
        labs = rng.choice(cfg.n_labels, size=P, replace=False)
        ctx = np.empty(2 * P, dtype=np.int64)
        ctx[0::2] = syms
        ctx[1::2] = cfg.n_symbols + labs

        qidx = rng.integers(0, P, size=Q)          # which context pair each query refers to
        qry = np.empty(2 * Q, dtype=np.int64)
        qry[0::2] = syms[qidx]
        qry[1::2] = cfg.n_symbols + labs[qidx]

        tokens[i, :2 * P] = ctx
        tokens[i, 2 * P:] = qry

    return {"tokens": tokens, "loss_mask": make_loss_mask(cfg, n)}


def make_loss_mask(cfg, n):
    """
    Which next-token predictions are scored.

    Base:     only the answer a.
    Extended: a, q2 and a2.
    """
    mask = np.zeros((n, cfg.seq_len - 1), dtype=bool)
    first_answer = 2 * cfg.n_pairs + 1                # position of a in tokens
    mask[:, first_answer - 1] = True                  # predict a from q
    if cfg.n_queries == 2:
        mask[:, first_answer] = True                  # predict q2 from a
        mask[:, first_answer + 1] = True              # predict a2 from q2
    return mask


def make_splits(cfg, n_train, n_val, n_test, seed=0, holdout_frac=0.0):
    """
    Train / validation / test splits from separate random seeds.

    Every sequence has its own random symbol-label pairing, so the splits
    can only be solved by in-context learning, never by memorising.

    holdout_frac > 0 also makes an extra 'test_new_symbols' split that only
    uses symbols never seen in training. Note: with learned token embeddings
    these symbols have untrained (random) embeddings, so this is a hard
    out-of-distribution test, not the main one.
    """
    rng = np.random.default_rng(seed)
    all_syms = rng.permutation(cfg.n_symbols)
    n_hold = int(round(holdout_frac * cfg.n_symbols))
    held, train_pool = all_syms[:n_hold], np.sort(all_syms[n_hold:])

    splits = {
        "train": make_dataset(cfg, n_train, seed + 1, train_pool),
        "val":   make_dataset(cfg, n_val,   seed + 2, train_pool),
        "test":  make_dataset(cfg, n_test,  seed + 3, train_pool),
    }
    if n_hold > 0:
        splits["test_new_symbols"] = make_dataset(cfg, n_test, seed + 4, np.sort(held))
    splits["meta"] = {"config": asdict(cfg), "seed": seed,
                      "train_pool": train_pool, "held_out_symbols": np.sort(held)}
    return splits


def overlap_count(a, b):
    """How many sequences in a also appear exactly in b (should be ~0)."""
    sa = {row.tobytes() for row in a["tokens"]}
    return sum(row.tobytes() in sa for row in b["tokens"])


def symbol_counts(cfg, data, positions="all"):
    """
    How often each symbol appears. Useful for spotting collapse
    (the model favouring some symbols).

    positions: "all", "context", or "q2" (the generated next symbol, extended task).
    """
    t = data["tokens"]
    if positions == "context":
        t = t[:, 0:2 * cfg.n_pairs:2]
    elif positions == "q2":
        assert cfg.n_queries == 2
        t = t[:, 2 * cfg.n_pairs + 2:2 * cfg.n_pairs + 3]
    t = t[t < cfg.n_symbols]
    return np.bincount(t.ravel(), minlength=cfg.n_symbols)


def decode(cfg, seq):
    """Human-readable string, e.g. 'S12 L3 S7 L0 | S7 L0'."""
    out = []
    for j, tok in enumerate(seq):
        if j == 2 * cfg.n_pairs:
            out.append("|")
        out.append(f"S{tok}" if tok < cfg.n_symbols else f"L{tok - cfg.n_symbols}")
    return " ".join(out)


def to_torch(data, device="cpu"):
    """Turn a split into PyTorch tensors: inputs, targets, loss mask."""
    import torch
    t = torch.as_tensor(data["tokens"], device=device)
    m = torch.as_tensor(data["loss_mask"], device=device)
    return t[:, :-1], t[:, 1:], m


if __name__ == "__main__":
    for nq in (1, 2):
        cfg = TaskConfig(n_queries=nq)
        s = make_splits(cfg, 1000, 200, 200, seed=0, holdout_frac=0.1)
        print(f"\n--- n_queries={nq}  vocab={cfg.vocab_size}  seq_len={cfg.seq_len}")
        for row in s["train"]["tokens"][:3]:
            print("  ", decode(cfg, row))
        print("  loss mask:", s["train"]["loss_mask"][0].astype(int))
        print("  train/test overlap:", overlap_count(s["train"], s["test"]))

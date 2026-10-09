"""
A small attention-only transformer for the in-context learning symbol task.

Why this design:
  - 2 layers, 1 head each, no MLPs. This is the smallest model that can hold
    the induction circuit: a previous token head in layer 1, and an
    induction head in layer 2 (Olsson et al., 2022; Singh et al., 2024).
  - Learned position embeddings. The previous token head needs to know
    "one step back", so it needs position information.
  - Causal mask. Each token can only look at itself and earlier tokens.

Extra features for the team:
  - return_cache=True gives the attention maps and the residual stream
    after each layer (for Person 3).
  - ablate={(layer, head), ...} switches off chosen heads (for Person 3).
  - generate() continues a sequence (for Person 2's extended collapse loop).
  - Optional MLPs and LayerNorm, for side experiments. Both are off by default.
"""

from dataclasses import dataclass, asdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int               # longest input the model will see
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 1
    use_mlp: bool = False
    use_layernorm: bool = False
    d_mlp: int = 256

    @property
    def d_head(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H, D, Dh = cfg.n_heads, cfg.d_model, cfg.d_head
        # Separate Q, K, V, O weights per head, so each head is easy to inspect
        self.W_Q = nn.Parameter(torch.randn(H, D, Dh) / math.sqrt(D))
        self.W_K = nn.Parameter(torch.randn(H, D, Dh) / math.sqrt(D))
        self.W_V = nn.Parameter(torch.randn(H, D, Dh) / math.sqrt(D))
        self.W_O = nn.Parameter(torch.randn(H, Dh, D) / math.sqrt(Dh))
        mask = torch.tril(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, ablate_heads=()):
        # x: [B, T, D]
        T = x.shape[1]
        q = torch.einsum("btd,hde->bhte", x, self.W_Q)
        k = torch.einsum("btd,hde->bhte", x, self.W_K)
        v = torch.einsum("btd,hde->bhte", x, self.W_V)

        scores = torch.einsum("bhqe,bhke->bhqk", q, k) / math.sqrt(self.cfg.d_head)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        pattern = scores.softmax(dim=-1)                       # [B, H, T, T]

        z = torch.einsum("bhqk,bhke->bhqe", pattern, v)        # [B, H, T, Dh]
        head_out = torch.einsum("bhte,hed->bhtd", z, self.W_O) # [B, H, T, D]
        for h in ablate_heads:                                 # switch off heads
            head_out[:, h] = 0.0
        return head_out.sum(dim=1), pattern, head_out


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.attn = Attention(cfg)
        self.mlp = None
        if cfg.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
            self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(),
                                     nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, ablate_heads=()):
        a, pattern, head_out = self.attn(self.ln1(x), ablate_heads)
        x = x + a                                   # residual (skip) connection
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x, pattern, head_out


class AttnOnlyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.use_layernorm else nn.Identity()
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.tok_emb.weight, std=0.02 * math.sqrt(cfg.d_model))
        nn.init.normal_(self.pos_emb.weight, std=0.02 * math.sqrt(cfg.d_model))

    def forward(self, idx, return_cache=False, ablate=()):
        """
        idx:     LongTensor [B, T] of token ids
        ablate:  set of (layer, head) pairs to switch off
        returns: logits [B, T, vocab]  (and a cache dict if return_cache)
        """
        B, T = idx.shape
        assert T <= self.cfg.max_len, f"input length {T} > max_len {self.cfg.max_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        cache = {"resid": [x], "attn": [], "head_out": []}
        for layer, block in enumerate(self.blocks):
            heads_off = [h for (l, h) in ablate if l == layer]
            x, pattern, head_out = block(x, heads_off)
            if return_cache:
                cache["attn"].append(pattern.detach())       # [B, H, T, T]
                cache["head_out"].append(head_out.detach())  # [B, H, T, D]
                cache["resid"].append(x.detach())

        logits = self.unembed(self.ln_f(x))
        if return_cache:
            cache["resid"][0] = cache["resid"][0].detach()
            return logits, cache
        return logits

    @torch.no_grad()
    def generate(self, prompt, n_new, temperature=1.0, greedy=False, allowed=None):
        """
        Continue each prompt by n_new tokens.

        prompt:  LongTensor [B, T]
        allowed: optional list of LongTensor masks [vocab] (one per new token),
                 True where a token may be picked. Use this to force, e.g.,
                 'a symbol' or 'a label' at a given position.
        """
        self.eval()
        seq = prompt
        for step in range(n_new):
            logits = self(seq)[:, -1, :]
            if allowed is not None:
                logits = logits.masked_fill(~allowed[step].to(logits.device), float("-inf"))
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def masked_loss_and_acc(logits, targets, loss_mask):
    """
    Cross-entropy and accuracy, only on the scored positions.
    Also returns per-position accuracy (useful for the extended task,
    where positions are: answer, next symbol, next answer).
    """
    lp = logits[loss_mask]                  # [N, vocab]
    tg = targets[loss_mask]                 # [N]
    loss = F.cross_entropy(lp, tg)
    correct = (logits.argmax(-1) == targets) & loss_mask
    acc = correct.sum().float() / loss_mask.sum()
    per_pos = correct.sum(0).float() / loss_mask.sum(0).clamp(min=1)
    per_pos = per_pos[loss_mask.any(0)]
    return loss, acc, per_pos


def save_model(model, path, extra=None):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "extra": extra or {}}, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = AttnOnlyTransformer(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt.get("extra", {})


def build_model(task_cfg, **kwargs):
    """Make a model sized for a data.TaskConfig."""
    cfg = ModelConfig(vocab_size=task_cfg.vocab_size,
                      max_len=task_cfg.seq_len - 1, **kwargs)
    return AttnOnlyTransformer(cfg)


def warm_start(src, task_cfg):
    """
    Make a model for task_cfg that starts from the weights of src.
    Used to start the extended-task model from a trained base model:
    the extended task does not learn the circuit well from scratch.
    Extra positions (if the new task is longer) keep their fresh init.
    """
    dst = build_model(task_cfg, **{k: getattr(src.cfg, k) for k in
                                   ("d_model", "n_layers", "n_heads", "use_mlp",
                                    "use_layernorm", "d_mlp")})
    assert dst.cfg.vocab_size == src.cfg.vocab_size, "same symbol and label sets needed"
    sd = src.state_dict()
    n = min(src.cfg.max_len, dst.cfg.max_len)
    with torch.no_grad():
        for k, v in dst.state_dict().items():
            if k == "pos_emb.weight":
                v[:n] = sd[k][:n]
            elif k in sd and sd[k].shape == v.shape:
                v.copy_(sd[k])
    return dst


if __name__ == "__main__":
    from data import TaskConfig, make_dataset, to_torch
    torch.manual_seed(0)
    tcfg = TaskConfig(n_queries=2)
    model = build_model(tcfg)
    print("parameters:", model.n_params())

    d = make_dataset(tcfg, 8, seed=0)
    x, y, m = to_torch(d)
    logits, cache = model(x, return_cache=True)
    print("logits:", tuple(logits.shape))
    print("attention maps:", [tuple(a.shape) for a in cache["attn"]])
    loss, acc, per_pos = masked_loss_and_acc(logits, y, m)
    print(f"untrained loss {loss.item():.3f} (chance = ln {tcfg.vocab_size} = "
          f"{math.log(tcfg.vocab_size):.3f}), acc {acc.item():.3f}")

import importlib.util
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('sn', os.path.join(ROOT, 'SproutNet_scale_0.1.0.py'))
sn = importlib.util.module_from_spec(spec)
sys.argv = ['transformer_baseline']
spec.loader.exec_module(sn)

torch.manual_seed(0)
np.random.seed(0)

tok = sn.FastBPE(sn.VOCAB_SIZE)
tok.load(os.path.join(ROOT, 'tokenizer_incremental.bin'))
V = tok.vocab_size
lines = sn.load_lines(os.path.join(ROOT, 'dsr1_v5_0001.jsonl'))
train_recs, val_recs = sn.split_records(lines)
MAXLEN = getattr(sn, 'MAX_SEQ_LEN', 512)


def build(recs, txt=False):
    seqs = []
    for _ln, line in recs:
        r = sn.render_sample(tok, line, txt)
        if r is None:
            continue
        ids, mask = r
        if len(ids) < 2:
            continue
        ids = ids[:MAXLEN]
        mask = mask[:MAXLEN]
        seqs.append((np.asarray(ids[:-1]), np.asarray(ids[1:]), np.asarray(mask[1:])))
    return seqs


D_TXT = os.path.join(ROOT, 'code_corpus.txt')
TXT = len(sys.argv) > 1 and sys.argv[1] == 'code'
if TXT:
    lines = sn.load_lines(D_TXT)
    train_recs, val_recs = sn.split_records(lines)
tr = build(train_recs, TXT)
va = build(val_recs, TXT)
print('train recs', len(tr), 'val recs', len(va))

D = 128
LAYERS = 2
HEADS = 4
CTX = 1024
BATCH = 16
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, h, attn_mask):
        a = self.ln1(h)
        a, _ = self.att(a, a, a, attn_mask=attn_mask, need_weights=False)
        h = h + a
        h = h + self.ff(self.ln2(h))
        return h


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(D)

    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.emb(x) + self.pos(torch.arange(T))
        for b in self.blocks:
            h = b(h, mask)
        return F.linear(self.ln(h), self.emb.weight)


model = TinyGPT()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
total_steps = EPOCHS * ((len(tr) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, pct_start=0.1, total_steps=total_steps)


def batch(seqs, i):
    chunk = seqs[i:i + BATCH]
    T = min(CTX, max(len(s[0]) for s in chunk))
    X = np.zeros((len(chunk), T), dtype=np.int64)
    Y = np.full((len(chunk), T), -100, dtype=np.int64)
    for b, (xs, ys, ms) in enumerate(chunk):
        n = min(T, len(xs))
        X[b, :n] = xs[:n]
        Y[b, :n] = np.where(ms[:n], ys[:n], -100)
    return torch.from_numpy(X), torch.from_numpy(Y)


step = 0
for ep in range(EPOCHS):
    model.train()
    order = np.random.permutation(len(tr))
    seqs = [tr[i] for i in order]
    nll = 0.0
    n = 0
    for i in range(0, len(seqs), BATCH):
        X, Y = batch(seqs, i)
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            nt = int((Y != -100).sum())
            nll += float(loss) * nt
            n += nt
        step += 1
    model.eval()
    vnll = 0.0
    vn = 0
    vhits = 0
    with torch.no_grad():
        for i in range(0, len(va), BATCH):
            X, Y = batch(va, i)
            logits = model(X)
            l = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1), ignore_index=-100, reduction='none')
            m = (Y != -100).reshape(-1)
            vnll += float(l[m].sum())
            vn += int(m.sum())
            vhits += int((logits.reshape(-1, V)[m].argmax(-1) == Y.reshape(-1)[m]).sum())
    print('[ep %d] trainPPL %.1f | VAL PPL %.1f acc %.3f (n=%d)' % (
        ep + 1, math.exp(nll / max(1, n)), math.exp(vnll / max(1, vn)), vhits / max(1, vn), vn))

import importlib.util
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('sn', os.path.join(ROOT, 'SproutNet_scale_0.1.0.py'))
sn = importlib.util.module_from_spec(spec)
sys.argv = ['baselines_smooth_code']
spec.loader.exec_module(sn)

sn.init_log_file()
tok = sn.FastBPE(sn.VOCAB_SIZE)
sn.TOKENIZER_CHECKPOINT = os.path.join(ROOT, 'tokenizer_incremental.bin')
ok = tok.load(sn.TOKENIZER_CHECKPOINT)
print('tokenizer loaded:', ok, 'V=', tok.vocab_size)
V = tok.vocab_size

lines = sn.load_lines(os.path.join(ROOT, 'code_corpus.txt'))
train_recs, val_recs = sn.split_records(lines)
print('train', len(train_recs), 'val', len(val_recs))

uni = Counter()
bi = defaultdict(Counter)
tri = defaultdict(Counter)
quad = defaultdict(Counter)
n_tok = 0
for _ln, line in train_recs:
    r = sn.render_sample(tok, line, True)
    if r is None:
        continue
    ids, mask = r
    for i, x in enumerate(ids):
        uni[x] += 1
        n_tok += 1
        if i >= 1:
            bi[ids[i - 1]][x] += 1
        if i >= 2:
            tri[(ids[i - 2], ids[i - 1])][x] += 1
        if i >= 3:
            quad[(ids[i - 3], ids[i - 2], ids[i - 1])][x] += 1
print('train tokens', n_tok, 'types seen', len(uni))

ALPHA = 0.4
uni_total = n_tok


def p_uni(x):
    return (uni.get(x, 0) + 0.5) / (uni_total + 0.5 * V)


def sb(ids, i, x, order):
    p = p_uni(x)
    pred = None
    ctxs = []
    if order >= 4 and i >= 3:
        ctxs.append((quad, (ids[i - 3], ids[i - 2], ids[i - 1])))
    if order >= 3 and i >= 2:
        ctxs.append((tri, (ids[i - 2], ids[i - 1])))
    if order >= 2 and i >= 1:
        ctxs.append((bi, ids[i - 1]))
    for cmap, key in ctxs:
        c = cmap.get(key)
        if c and c.get(x, 0) > 0:
            tot = sum(c.values())
            return c[x] / tot, max(c, key=c.get)
        p *= ALPHA
        if pred is None and c:
            pred = max(c, key=c.get)
    return p, pred


def eval_order(name, order):
    nll = 0.0
    n = 0
    hits = 0
    for _ln, line in val_recs:
        r = sn.render_sample(tok, line, True)
        if r is None:
            continue
        ids, mask = r
        for i, x in enumerate(ids):
            if i == 0:
                continue
            p, pred = sb(ids, i, x, order)
            nll += -math.log(max(p, 1e-12))
            n += 1
            if pred is None:
                pred = max(uni, key=uni.get)
            if pred == x:
                hits += 1
    print('%s: PPL %.1f | acc %.3f (n=%d)' % (name, math.exp(nll / max(n, 1)), hits / max(n, 1), n))


eval_order('code smooth bigram ', 2)
eval_order('code smooth trigram', 3)
eval_order('code smooth 4-gram ', 4)
sn.close_log_file()

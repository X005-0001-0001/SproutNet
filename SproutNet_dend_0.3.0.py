# -*- coding: utf-8 -*-
"""SproutNet · dend 分支 · 0.2.0
干净重写（0.1.0 反复打补丁失败，重写以避免历史包袱）。

开关（环境变量）：
  B2_DSEG      树突段数（1=单层对照）       B2_NONLIN  段内非线性 clip/tanh/sqrt
  B2_INT_CAP   内部神经元数                  B2_FAN_IN  每个神经元的输入边数
  B2_CO_READ   每个神经元的读出边数          B2_UPD     内部输入更新 none/stdp/blame
  B2_SYN_SCALE 突触缩放开关                  B2_SCALE_TARGET 单神经元输入 L1 目标

核心结构（每条都有依据）：
  感受器 快 λ=0.35 / 慢 λ=0.85
  内部神经元  瞬时单元（每步从输入重算，不积分）：
     段内 a_k = f(Σ_{i∈段k} w_i x_i)   段内非线性
     胞体 a   = Σ_k v_k a_k
  读出  每神经元接 CO_READ 个 token 行，LMS（误差驱动）
  生长  共激活建一阶边；满员按使用率淘汰
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(int(os.environ.get('TB_THREADS', '6')))

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sn', os.path.join(ROOT, 'SproutNet_coverage_0.9.2.py'))
sn = importlib.util.module_from_spec(spec)
sys.argv = ['dend2']
spec.loader.exec_module(sn)
sn._LOG_FILE = None

DATA = os.environ.get('B2_DATA', 'code_corpus.txt')
TOKENS = int(os.environ.get('B2_TOKENS', '100000'))
B = int(os.environ.get('B2_B', '32'))
DEV = os.environ.get('B2_DEV', 'cpu')
LR = float(os.environ.get('B2_LR', '0.25'))
CLIP = float(os.environ.get('B2_CLIP', '9.0'))
INT_CAP = int(os.environ.get('B2_INT_CAP', '96'))
FAN_IN = int(os.environ.get('B2_FAN_IN', '1000'))
CO_READ = int(os.environ.get('B2_CO_READ', '500'))
DSEG = int(os.environ.get('B2_DSEG', '1'))
NONLIN = os.environ.get('B2_NONLIN', 'clip')
UPD = os.environ.get('B2_UPD', 'none')
ETA_STDP = float(os.environ.get('B2_ETA_STDP', '0.10'))
ETA_BLAME = float(os.environ.get('B2_ETA_BLAME', '0.05'))
SYN_SCALE = os.environ.get('B2_SYN_SCALE', '1') == '1'
SCALE_TARGET = float(os.environ.get('B2_SCALE_TARGET', '4.0'))
G_NEURO = float(os.environ.get('B2_G_NEURO', '0.05'))
SYN_CAP = int(os.environ.get('B2_SYN', '3000000'))
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '3'))
LAM_F = float(os.environ.get('B2_LAMF', '0.35'))
LAM_S = float(os.environ.get('B2_LAMS', '0.85'))
KSCALE = int(os.environ.get('B2_KSCALE', '4'))     # 感受器时间尺度数（1=原来快慢两档）

dev = torch.device(DEV)
sn.DATA_RECORDS_LIMIT = 100000
tok = sn.FastBPE(sn.VOCAB_SIZE)
sn.TOKENIZER_CHECKPOINT = os.path.join(ROOT, 'tokenizer_incremental.bin')
tok.load(sn.TOKENIZER_CHECKPOINT)
V = tok.vocab_size
lines = sn.load_lines(os.path.join(ROOT, DATA))
is_txt = DATA.endswith('.txt')
train_recs, val_recs = sn.split_records(lines)


def nonlin(x):
    if NONLIN == 'tanh':
        return torch.tanh(x)
    if NONLIN == 'sqrt':
        return torch.sqrt(x.clamp(min=0.0))
    if NONLIN == 'relu6':
        return x.clamp(0.0, 6.0) / 6.0
    return x.clamp(0.0, 1.0)


class Brain:
    def __init__(self):
        # 0.3.0：K 组感受器，时间常数按对数均匀覆盖 1..40 步
        self.KV = KSCALE * V
        self.n = self.KV + INT_CAP
        taus = torch.logspace(0, math.log10(40.0), KSCALE)     # 1..40 步
        lam_s = (1.0 - 1.0 / taus).repeat_interleave(V)
        lam_i = torch.rand(INT_CAP) * 0.8 + 0.1
        self.lam = torch.cat([lam_s, lam_i]).to(dev)
        self.v = torch.zeros(B, self.n, device=dev)
        self.b = torch.zeros(V, device=dev)
        self.si = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.sj = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.sw = torch.zeros(SYN_CAP, device=dev)
        self.ns = 0
        self.keys = set()
        self.alive = torch.zeros(INT_CAP, dtype=torch.bool, device=dev)
        self.vk = torch.full((INT_CAP, DSEG), 1.0, device=dev)
        self.free = list(range(INT_CAP))
        self.prev = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.last = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.raw_last = torch.zeros(B, self.n, device=dev)
        self.a_cur = torch.zeros(B, INT_CAP, device=dev)
        self.a_last = torch.zeros(B, INT_CAP, device=dev)
        self.err = torch.ones(V, device=dev)
        self.clock = 0
        self.births = 0

    # 边
    def add_edge(self, i, j, w=0.0):
        if (i, j) in self.keys or self.ns >= SYN_CAP:
            return
        k = self.ns
        self.si[k], self.sj[k], self.sw[k] = i, j, w
        self.ns += 1
        self.keys.add((i, j))

    def bulk(self, srcs, dst, w):
        m = min(len(srcs), SYN_CAP - self.ns)
        if m <= 0:
            return
        sl = slice(self.ns, self.ns + m)
        self.si[sl] = torch.from_numpy(np.asarray(srcs[:m], dtype=np.int64)).to(dev)
        if np.isscalar(dst):
            self.sj[sl] = int(dst)
        else:
            self.sj[sl] = torch.from_numpy(np.asarray(dst[:m], dtype=np.int64)).to(dev)
        self.sw[sl] = torch.from_numpy(np.asarray(w[:m], dtype=np.float32)).to(dev)
        self.ns += m

    # 出生
    def birth(self):
        if not self.free:
            return
        k = self.free.pop(0)
        nid = 2 * V + k
        self.alive[k] = True
        kf = max(1, min(FAN_IN // (2 * KSCALE), V))
        order = torch.argsort(-self.err)[:kf].cpu().numpy().astype(np.int64)
        tk = np.concatenate([order + g * V for g in range(KSCALE)])
        self.bulk(tk, nid, np.random.randn(len(tk)) * 0.05)
        kr = max(1, min(CO_READ, V))
        top = torch.argsort(-self.err)[:kr].cpu().numpy().astype(np.int64)
        self.bulk(np.full(kr, nid, dtype=np.int64), top, np.random.randn(kr) * 0.01)
        self.births += 1

    # 前向：内部神经元 = 段内非线性 + 胞体汇总
    def act_int(self, ap):
        if not self.alive.any():
            return torch.zeros(B, INT_CAP, device=dev)
        den = (self.sj[:self.ns] >= self.KV) & (self.si[:self.ns] < self.KV)
        if not den.any():
            return torch.zeros(B, INT_CAP, device=dev)
        si_d, sj_d = self.si[:self.ns][den], self.sj[:self.ns][den]
        w_d = self.sw[:self.ns][den]
        nid = sj_d - 2 * V
        val = ap[:, si_d] * w_d.unsqueeze(0)
        if DSEG <= 1:
            acc = torch.zeros(B, INT_CAP, device=dev)
            acc.scatter_add_(1, nid.unsqueeze(0).expand(B, -1), val)
            return nonlin(acc)
        seg = (si_d % V) % DSEG
        flat = nid * DSEG + seg
        acc = torch.zeros(B, INT_CAP * DSEG, device=dev)
        acc.scatter_add_(1, flat.unsqueeze(0).expand(B, -1), val)
        seg_act = nonlin(acc.view(B, INT_CAP, DSEG) / max(1.0, float(DSEG)))
        return (seg_act * self.vk.unsqueeze(0)).sum(dim=2).clamp(0.0, 1.0)

    def logits(self, ap, a_int):
        l = self.b.unsqueeze(0).expand(B, -1).clone()
        if self.ns:
            ro = self.sj[:self.ns] < V
            if ro.any():
                # ap 的布局就是 [raw(2V), a_int(INT_CAP)]，与 si 的语义一致，直接索引
                l.scatter_add_(1, self.sj[:self.ns][ro].unsqueeze(0).expand(B, -1),
                               ap[:, self.si[:self.ns][ro]] * self.sw[:self.ns][ro].unsqueeze(0))
        return l

    # 一步
    def step(self, x, learn=True):
        raw = self.v.clamp(0.0, 1.0)
        a_int = self.act_int(raw)
        ap = torch.cat([raw[:, :self.KV], a_int], dim=1)
        logits = self.logits(ap, a_int)
        if learn and self.clock > 0:
            p = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
            e = p.clone()
            e[torch.arange(B), x] -= 1.0
            scale = 1.0 / math.sqrt(B)
            if self.ns:
                is_ro = self.sj[:self.ns] < V
                is_den = ~is_ro
                w = self.sw[:self.ns]
                if is_ro.any():
                    si_r, sj_r = self.si[:self.ns][is_ro], self.sj[:self.ns][is_ro]
                    gr = (ap[:, si_r] * e[:, sj_r]).sum(dim=0) * scale
                    w[is_ro] = torch.clamp(w[is_ro] - LR * gr, -CLIP, CLIP)
                if is_den.any() and UPD != 'none':
                    si_d, sj_d = self.si[:self.ns][is_den], self.sj[:self.ns][is_den]
                    if UPD == 'stdp':
                        post_now = a_int[:, sj_d - 2 * V]
                        ltp = (self.raw_last[:, si_d] * post_now).sum(dim=0)
                        ltd = (raw[:, si_d] * self.a_last[:, sj_d - 2 * V]).sum(dim=0)
                        w[is_den] = torch.clamp(w[is_den] + (ltp - ltd) * scale * ETA_STDP, -CLIP, CLIP)
                    else:  # blame
                        ro = is_ro
                        blame = torch.zeros(INT_CAP, device=dev)
                        if ro.any():
                            si_r, sj_r = self.si[:self.ns][ro], self.sj[:self.ns][ro]
                            contrib = (w[ro].unsqueeze(0) * ap[:, si_r] * e[:, sj_r]).sum(dim=0)
                            blame.scatter_add_(0, sj_r - 2 * V, contrib)
                        d = (raw[:, si_d] * blame[sj_d - 2 * V].unsqueeze(0)).sum(dim=0) * scale
                        w[is_den] = torch.clamp(w[is_den] - ETA_BLAME * d, -CLIP, CLIP)
                if is_den.any() and SYN_SCALE:
                    si_d, sj_d = self.si[:self.ns][is_den], self.sj[:self.ns][is_den]
                    tot = torch.zeros(INT_CAP, device=dev)
                    tot.scatter_add_(0, sj_d - 2 * V, self.sw[:self.ns][is_den].abs())
                    f = torch.where(tot > SCALE_TARGET, SCALE_TARGET / (tot + 1e-9), torch.ones_like(tot))
                    w[is_den] = w[is_den] * f[sj_d - 2 * V]
            self.b -= LR * e.sum(dim=0) * scale
            self.err = 0.98 * self.err + 0.02 * e.abs().mean(dim=0)
            if torch.rand(1).item() < G_NEURO:
                self.birth()
            ok = self.prev >= 0
            if ok.any():
                for s_, d_ in zip(self.prev[ok].tolist(), x[ok].tolist()):
                    if d_ < V:
                        self.add_edge(s_, d_, 0.0)
        self.v = self.lam.unsqueeze(0) * self.v
        ar = torch.arange(B, device=dev)
        # 多尺度注入：当前 token 同时进入所有 K 组（各自不同衰减），
        # 上一 token 进入除最慢组外的各组（保证顺序信息）
        for g in range(KSCALE):
            self.v[ar, g * V + x] += 1.0
        ok_slow = self.last >= 0
        if ok_slow.any():
            for g in range(max(0, KSCALE - 1)):
                self.v[ar[ok_slow], g * V + self.last[ok_slow]] += 1.0
        self.raw_last = raw.clone()
        self.a_last = self.a_cur
        self.a_cur = a_int
        self.last = x.clone()
        self.prev = x.clone()
        self.clock += 1
        raw2 = self.v.clamp(0.0, 1.0)
        a2 = self.act_int(raw2)
        return self.logits(torch.cat([raw2[:, :self.KV], a2], dim=1), a2)


def eval_ppl(brain, limit):
    nll, n = 0.0, 0
    for _ln, line in val_recs[:limit]:
        r = sn.render_sample(tok, line, is_txt)
        if r is None:
            continue
        ids, mask = r
        if len(ids) < 2:
            continue
        brain.v.zero_()
        brain.raw_last.zero_()
        brain.a_cur.zero_()
        brain.a_last.zero_()
        brain.prev.fill_(-1)
        brain.last.fill_(-1)
        brain.clock = 0
        for i, tid in enumerate(ids):
            x = torch.full((B,), tid, dtype=torch.long, device=dev)
            if i > 0 and mask[i]:
                raw = brain.v.clamp(0.0, 1.0)
                a = brain.act_int(raw)
                lg = brain.logits(torch.cat([raw[:, :brain.KV], a], dim=1), a)
                p = torch.softmax(lg[0], dim=0)
                nll -= math.log(max(float(p[tid]), 1e-12))
                n += 1
            brain.step(x, learn=False)
    return math.exp(nll / max(n, 1)), n


all_ids = []
for _ln, line in train_recs:
    r = sn.render_sample(tok, line, is_txt)
    if r is None:
        continue
    ids, _m = r
    all_ids.extend(ids)
all_ids = np.asarray(all_ids, dtype=np.int64)
total = int(min(len(all_ids), TOKENS))
seg = total // B
mat = all_ids[:seg * B].reshape(B, seg).T.copy()

brain = Brain()
t0 = time.time()
for t in range(seg - 1):
    brain.step(torch.from_numpy(mat[t]).to(dev), learn=True)

ppl, n = eval_ppl(brain, EVAL_RECS)
msg = '[d2] done | DSEG=%d NONLIN=%s UPD=%s cap=%d tok=%d | VAL PPL %.1f (n=%d) | syn=%d ints=%d | %.0f tok/s' % (
    DSEG, NONLIN, UPD, INT_CAP, seg * B, ppl, n, brain.ns, int(brain.alive.sum()),
    seg * B / (time.time() - t0))
print(msg)

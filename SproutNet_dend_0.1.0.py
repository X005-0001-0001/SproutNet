# -*- coding: utf-8 -*-
"""SproutNet · dend 分支 · 0.1.0
干净的最小内核（重写，用于快速铺变体）：多段树突 + 稀疏读出 + STDP/LMS。

设计（每条都有实验依据）：
  感受器：快通道 λ=0.35 / 慢通道 λ=0.85（上一 token）
  内部神经元：
     段内  a_k = clip(Σ_{i∈段k} w_i x_i)      ← 树突段内非线性（Beniaguev 2021 的深度来源）
     胞体  a   = clip(Σ_k v_k a_k)            ← 段活动汇总
     出生：接「按惊讶度选」的 token（输入选对 > 输入更多，0.4.0 实测）
  读出：每个内部神经元接 CO_READ 个 token 行，LMS 更新
  生长：共激活建一阶边；使用率低者被淘汰（内存准入）
  随机种子固定（可复现）

不做：coinc / 能量账本 / 多层结构（变体再加）。
用法：B2_* 环境变量控制，见文件顶部常量。
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
sys.argv = ['dend']
spec.loader.exec_module(sn)

DATA = os.environ.get('B2_DATA', 'code_corpus.txt')
TOKENS = int(os.environ.get('B2_TOKENS', '100000'))
B = int(os.environ.get('B2_B', '32'))
DEV = os.environ.get('B2_DEV', 'cpu')
LR = float(os.environ.get('B2_LR', '0.25'))
CLIP = float(os.environ.get('B2_CLIP', '9.0'))
ETA_STDP = float(os.environ.get('B2_ETA_STDP', '0.10'))
INT_CAP = int(os.environ.get('B2_INT_CAP', '96'))
FAN_IN = int(os.environ.get('B2_FAN_IN', '1000'))
CO_READ = int(os.environ.get('B2_CO_READ', '500'))
DSEG = int(os.environ.get('B2_DSEG', '4'))          # 树突段数；1 = 单层（对照）
NONLIN = os.environ.get('B2_NONLIN', 'clip')         # 段内非线性: clip / tanh / sqrt / relu6
SEG_NORM = os.environ.get('B2_SEG_NORM', '1') == '1'  # 段内是否按段大小归一
SYN_SCALE = os.environ.get('B2_SYN_SCALE', '1') == '1'  # 突触缩放（每神经元输入权重总量归一）
SCALE_TARGET = float(os.environ.get('B2_SCALE_TARGET', '4.0'))  # 单神经元输入权重 L1 总量目标
G_NEURO = float(os.environ.get('B2_G_NEURO', '0.05'))
SYN_CAP = int(os.environ.get('B2_SYN', '3000000'))
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '3'))
LAM_F = float(os.environ.get('B2_LAMF', '0.35'))
LAM_S = float(os.environ.get('B2_LAMS', '0.85'))

dev = torch.device(DEV)
sn.DATA_RECORDS_LIMIT = 100000
# 批量并行时每个进程独立日志（避免 Windows 并发写冲突）；模块内日志关闭
sn._LOG_FILE = None

tok = sn.FastBPE(sn.VOCAB_SIZE)
sn.TOKENIZER_CHECKPOINT = os.path.join(ROOT, 'tokenizer_incremental.bin')
tok.load(sn.TOKENIZER_CHECKPOINT)
V = tok.vocab_size
lines = sn.load_lines(os.path.join(ROOT, DATA))
is_txt = DATA.endswith('.txt')
train_recs, val_recs = sn.split_records(lines)
print('data=%s V=%d B=%d DEV=%s DSEG=%d INT_CAP=%d' % (DATA, V, B, DEV, DSEG, INT_CAP))


class Brain:
    def __init__(self):
        self.N = 2 * V + INT_CAP
        lam_i = torch.rand(INT_CAP) * 0.8 + 0.1
        self.lam = torch.cat([torch.full((V,), LAM_F), torch.full((V,), LAM_S), lam_i]).to(dev)
        self.v = torch.zeros(B, self.N, device=dev)
        self.b = torch.zeros(V, device=dev)
        self.s_i = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.s_j = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.s_w = torch.zeros(SYN_CAP, device=dev)
        self.n_syn = 0
        self.keys = set()
        self.int_alive = torch.zeros(INT_CAP, dtype=torch.bool, device=dev)
        self.vk = torch.full((INT_CAP, DSEG), 1.0, device=dev)
        self._free = list(range(INT_CAP))
        self.prev = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.last = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.raw_last = torch.zeros(B, self.N, device=dev)   # 上一步的感受器活动（pre）
        self.a_int = torch.zeros(B, INT_CAP, device=dev)     # 本步内部神经元瞬时输出（post）
        self.a_int_last = torch.zeros(B, INT_CAP, device=dev) # 上一步的 post
        self.err = torch.ones(V, device=dev)
        self.clock = 0
        self.births = 0

    # ---------- 边 ----------
    def add_edge(self, i, j, w=0.0):
        if (i, j) in self.keys or self.n_syn >= SYN_CAP:
            return False
        k = self.n_syn
        self.s_i[k] = i
        self.s_j[k] = j
        self.s_w[k] = w
        self.n_syn += 1
        self.keys.add((i, j))
        return True

    def bulk_add(self, srcs, dst, w):
        n = min(len(srcs), SYN_CAP - self.n_syn)
        if n <= 0:
            return
        sl = slice(self.n_syn, self.n_syn + n)
        self.s_i[sl] = torch.from_numpy(np.asarray(srcs[:n], dtype=np.int64)).to(dev)
        if np.isscalar(dst):
            self.s_j[sl] = int(dst)
        else:
            d = np.asarray(dst, dtype=np.int64)
            if d.ndim == 0:
                self.s_j[sl] = int(d)
            else:
                self.s_j[sl] = torch.from_numpy(np.ascontiguousarray(d)).to(dev)
        self.s_w[sl] = torch.from_numpy(np.asarray(w[:n], dtype=np.float32)).to(dev)
        self.n_syn += n

    # ---------- 出生 ----------
    def birth(self):
        if not self._free:
            return False
        k = self._free.pop(0)
        nid = 2 * V + k
        self.int_alive[k] = True
        kf = max(1, min(FAN_IN // 2, V))
        order = torch.argsort(-self.err)[:kf].cpu().numpy().astype(np.int64)
        tok = np.concatenate([order, order + V])
        self.bulk_add(tok, nid, np.random.randn(len(tok)) * 0.05)
        kr = max(1, min(CO_READ, V))
        top = torch.argsort(-self.err)[:kr].cpu().numpy().astype(np.int64)
        self.bulk_add(np.full(kr, nid, dtype=np.int64), top, np.random.randn(kr) * 0.01)
        self.births += 1
        return True

    # ---------- 前向 ----------
    def logits(self, ap):
        l = self.b.unsqueeze(0).expand(B, -1).clone()
        if self.n_syn:
            ro = self.s_j[:self.n_syn] < V
            if ro.any():
                si = self.s_i[:self.n_syn][ro]
                sj = self.s_j[:self.n_syn][ro]
                w = self.s_w[:self.n_syn][ro]
                l.scatter_add_(1, sj.unsqueeze(0).expand(B, -1),
                               ap[:, si] * w.unsqueeze(0))
        return l

    def forward(self, ap):
        """内部神经元：段内非线性 + 胞体汇总（DSEG=1 时退化为单层）。"""
        self.a_int_new = None
        if not self.int_alive.any() or self.n_syn == 0:
            return ap
        den = (self.s_j[:self.n_syn] >= V) & (self.s_i[:self.n_syn] < 2 * V)
        if not den.any():
            return ap
        si_d = self.s_i[:self.n_syn][den]
        sj_d = self.s_j[:self.n_syn][den]
        w_d = self.s_w[:self.n_syn][den]
        nid = sj_d - 2 * V
        val = ap[:, si_d] * w_d.unsqueeze(0)
        if DSEG <= 1:
            acc = torch.zeros(B, INT_CAP, device=dev)
            acc.scatter_add_(1, nid.unsqueeze(0).expand(B, -1), val)
            a_int = acc.clamp(0.0, 1.0)
        else:
            seg = (si_d % V) % DSEG
            flat = nid * DSEG + seg
            acc = torch.zeros(B, INT_CAP * DSEG, device=dev)
            acc.scatter_add_(1, flat.unsqueeze(0).expand(B, -1), val)
            seg_raw = acc.view(B, INT_CAP, DSEG)
            if SEG_NORM:
                seg_raw = seg_raw / max(1.0, float(DSEG))
            if NONLIN == 'tanh':
                seg_act = torch.tanh(seg_raw)
            elif NONLIN == 'sqrt':
                seg_act = torch.sqrt(seg_raw.clamp(min=0.0))
            elif NONLIN == 'relu6':
                seg_act = seg_raw.clamp(0.0, 6.0) / 6.0
            else:
                seg_act = seg_raw.clamp(0.0, 1.0)
            # 段汇总：用「和」而不是「平均」（vk 初值 1.0，均分会把总信号压 K 倍）
            a_int = (seg_act * self.vk.unsqueeze(0)).sum(dim=2).clamp(0.0, 1.0)
        out = ap.clone()
        out[:, 2 * V:] = a_int
        self.a_int_new = a_int
        return out

    # ---------- 一步 ----------
    def step(self, x, learn=True):
        raw = self.v.clamp(0.0, 1.0)
        ap = self.forward(raw)
        logits = self.logits(ap)
        if learn and self.clock > 0:
            p = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
            e = p.clone()
            e[torch.arange(B), x] -= 1.0
            scale = 1.0 / math.sqrt(B)
            if self.n_syn:
                si = self.s_i[:self.n_syn]
                sj = self.s_j[:self.n_syn]
                is_ro = sj < V
                is_den = ~is_ro
                w = self.s_w[:self.n_syn]
                if is_ro.any():
                    gr = (ap[:, si[is_ro]] * e[:, sj[is_ro]]).sum(dim=0) * scale
                    w[is_ro] = torch.clamp(w[is_ro] - LR * gr, -CLIP, CLIP)
                if is_den.any() and SYN_SCALE:
                    # 突触缩放（synaptic scaling）：把每个神经元所有输入边的 |w| 总量拉回目标值。
                    # 没有它，STDP 只有增强没有稳态 → 权重撞 clip、输出爆炸（实测 D2 2858）。
                    si_d0, sj_d0 = si[is_den], sj[is_den]
                    tot = torch.zeros(INT_CAP, device=dev)
                    tot.scatter_add_(0, sj_d0 - 2 * V, self.s_w[:self.n_syn][is_den].abs())
                    scale_f = torch.where(tot > SCALE_TARGET,
                                          SCALE_TARGET / (tot + 1e-9), torch.ones_like(tot))
                    w[is_den] = w[is_den] * scale_f[sj_d0 - 2 * V]
                if is_den.any():
                    si_d, sj_d = si[is_den], sj[is_den]
                    # STDP 正确时序：pre = 感受器活动 raw，post = 内部神经元瞬时输出 a_int。
                    # 此前用 raw 作 post（内部神经元膜电位恒 0），故 ltp-ltd ≡ 0。
                    # LTP = pre(t-1)·post(t)   LTD = pre(t)·post(t-1)
                    post_now = self.a_int[:, sj_d - 2 * V] if self.a_int.numel() else 0.0
                    ltp = (self.raw_last[:, si_d] * post_now).sum(dim=0)
                    ltd = (raw[:, si_d] * self.a_int_last[:, sj_d - 2 * V]).sum(dim=0)
                    if os.environ.get('B2_PROBE') == '1' and self.clock in (3, 10):
                        print('  [stdp] n=%d ltp_abs %.6f ltd_abs %.6f diff_abs %.6f' % (
                            int(is_den.sum()), float(ltp.abs().mean()), float(ltd.abs().mean()),
                            float((ltp - ltd).abs().mean())))
                    w[is_den] = torch.clamp(w[is_den] + (ltp - ltd) * scale * ETA_STDP, -CLIP, CLIP)
            self.b -= LR * e.sum(dim=0) * scale
            self.err = 0.98 * self.err + 0.02 * e.abs().mean(dim=0)
            if torch.rand(1).item() < G_NEURO:
                self.birth()
            ok = self.prev >= 0
            if ok.any():
                for s_, d_ in zip(self.prev[ok].tolist(), x[ok].tolist()):
                    if d_ < V:
                        self.add_edge(s_, d_, 0.0)
        # 状态推进
        self.v = self.lam.unsqueeze(0) * self.v
        ar = torch.arange(B, device=dev)
        self.v[ar, x] += 1.0
        ok_slow = self.last >= 0
        if ok_slow.any():
            self.v[ar[ok_slow], self.last[ok_slow] + V] += 1.0
        # 关键：存「这一步的 raw」（v 更新之前），而不是更新后的 v，
        # 否则下一步开头读到的 raw 与它恒等，STDP 的 ltp-ltd 又变 0。
        self.raw_last = raw.clone()
        if getattr(self, 'a_int_new', None) is not None:
            self.a_int_last = self.a_int
            self.a_int = self.a_int_new
        self.last = x.clone()
        self.prev = x.clone()
        self.clock += 1
        return self.logits(self.forward(self.v.clamp(0.0, 1.0)))

    # ---------- 拓扑 ----------
    def topo(self):
        n = int(self.int_alive.sum())
        if n < 2:
            return {'n': n}
        idx = torch.nonzero(self.int_alive).flatten().cpu().numpy()
        pos = {int(v): i for i, v in enumerate(idx)}
        si = self.s_i[:self.n_syn].cpu().numpy()
        sj = self.s_j[:self.n_syn].cpu().numpy()
        A = np.zeros((n, n), dtype=np.float32)
        for a, b in zip(si, sj):
            if a >= 2 * V and b >= 2 * V:
                i, j = pos.get(int(a) - 2 * V), pos.get(int(b) - 2 * V)
                if i is not None and j is not None and i != j:
                    A[i, j] = 1.0
        U = np.maximum(A, A.T)
        m = int(U.sum())
        cc = 0.0
        for i in range(n):
            k = U[i].sum()
            if k > 1:
                cc += (U[i] @ U @ U[:, i]) / (k * (k - 1))
        cc /= n
        deg = A.sum(1) + A.sum(0)
        return {'n': n, 'edges': m, 'deg_mean': float(deg.mean()), 'deg_max': float(deg.max()),
                'cc': float(cc), 'cc_rand': float(m / max(1.0, n * (n - 1)))}


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
        brain.a_int.zero_()
        brain.a_int_last.zero_()
        brain.prev.fill_(-1)
        brain.last.fill_(-1)
        brain.clock = 0
        for i, tid in enumerate(ids):
            x = torch.full((B,), tid, dtype=torch.long, device=dev)
            if i > 0 and mask[i]:
                lg = brain.logits(brain.forward(brain.v.clamp(0.0, 1.0)))
                p = torch.softmax(lg[0], dim=0)
                nll -= math.log(max(float(p[tid]), 1e-12))
                n += 1
            brain.step(x, learn=False)
    return math.exp(nll / max(n, 1)), n


# ---- 探针：检查 STDP 是否真的在更新内部边 ----
if os.environ.get('B2_PROBE') == '1':
    b = Brain()
    b.birth(); b.birth()
    den = (b.s_j[:b.n_syn] >= V) & (b.s_i[:b.n_syn] < 2 * V)
    eidx = torch.nonzero(den).flatten()          # 固定这批边（后面会新建边，不能再用布尔掩码）
    w0 = b.s_w[eidx].clone()
    print('树突边数 %d / 总边 %d' % (int(den.sum()), b.n_syn))
    for _ in range(50):
        xx = torch.from_numpy(np.random.randint(0, V, size=B).astype(np.int64)).to(dev)
        b.step(xx, learn=True)
    w1 = b.s_w[eidx]
    d = (w1 - w0).abs()
    print('权重变化 abs mean %.6f  max %.6f' % (float(d.mean()), float(d.max())))
    print('内部神经元 %d  raw_last.sum %.3f' % (int(b.int_alive.sum()), float(b.raw_last.sum())))
    sys.exit(0)

# ---- 数据 ----
all_ids, all_msk, all_start = [], [], []
for _ln, line in train_recs:
    r = sn.render_sample(tok, line, is_txt)
    if r is None:
        continue
    ids, m = r
    all_start.append(len(all_ids))
    all_ids.extend(ids)
    all_msk.extend([1 if v else 0 for v in m])
all_ids = np.asarray(all_ids, dtype=np.int64)
total = int(min(len(all_ids), TOKENS))
seg = total // B
mat = all_ids[:seg * B].reshape(B, seg).T.copy()
print('stream: %d tokens, each reader %d' % (total, seg))

brain = Brain()
t0 = time.time()
for t in range(seg - 1):
    x = torch.from_numpy(mat[t]).to(dev)
    brain.step(x, learn=True)
    if (t + 1) % 20000 == 0:
        done = (t + 1) * B
        print('[dend] tok %7d | %6.0f tok/s | syn %6d | ints %d' % (
            done, done / (time.time() - t0), brain.n_syn, int(brain.int_alive.sum())))

ppl, n = eval_ppl(brain, EVAL_RECS)
msg = '[dend] done | DSEG=%d tok %d | VAL PPL %.1f (n=%d) | syn %d | ints %d | %.0f tok/s' % (
    DSEG, seg * B, ppl, n, brain.n_syn, int(brain.int_alive.sum()), seg * B / (time.time() - t0))
print(msg)
print(msg)
tr = brain.topo()
print('[topo] ' + ' '.join('%s=%.3f' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v) for k, v in tr.items()))
sn.close_log_file()

# -*- coding: utf-8 -*-
"""SproutNet · batch2 分支 · 0.1.0
GPU 批并行内核：B 个独立读者共享一个脑，各自连续读不同数据段。

实现子集（最小可验证）：
  - 感受器：快 V + 慢 V（λ=0.35 / 0.85），与主内核一致
  - 读出：bias + 感受器→token 直连（一阶表），批量 LMS
  - 生长：共激活即建边（观测到 (prev→x) 且无边则建），批量
  - 不做：内部神经元 / coinc / 能量账本（后续按需）

用途：验证「批并行 + 大数据」能否把一阶表学满。
参照（代码语料同划分）：bigram 28.7 / trigram 12.5 / 4-gram 10.7；transformer d96L1 7.97。

红线：无 BP、无两次前向对比、无外部监督注入。每突触只看自己的前后活动与本次局部误差。
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sn', os.path.join(ROOT, 'SproutNet_coverage_0.9.2.py'))
sn = importlib.util.module_from_spec(spec)
sys.argv = ['batch2']
spec.loader.exec_module(sn)

DATA = os.environ.get('B2_DATA', 'code_corpus.txt')
TOKENS = int(os.environ.get('B2_TOKENS', '2000000'))
B = int(os.environ.get('B2_B', '32'))
SYN_CAP = int(os.environ.get('B2_SYN', '200000'))
DEV = os.environ.get('B2_DEV', 'cuda')
LR = float(os.environ.get('B2_LR', '0.25'))
CLIP = float(os.environ.get('B2_CLIP', '9.0'))
LAM_F = float(os.environ.get('B2_LAMF', '0.35'))
LAM_S = float(os.environ.get('B2_LAMS', '0.85'))
EPOCHS = int(os.environ.get('B2_EPOCHS', '1'))
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '30'))
CO_CAP = int(os.environ.get('B2_CO_CAP', '20000'))     # 二阶（乘法）单元数
CO_FAN = int(os.environ.get('B2_CO_FAN', '50'))        # 每个二阶单元连的 token 行数
ETA_CO = float(os.environ.get('B2_ETA_CO', '0.25'))

sn.DATA_RECORDS_LIMIT = 100000
sn.LOG_FILE = os.path.join(ROOT, 'training_log_batch2.txt')
sn.init_log_file()
sn._LOG_FLUSH_INTERVAL = 1

tok = sn.FastBPE(sn.VOCAB_SIZE)
sn.TOKENIZER_CHECKPOINT = os.path.join(ROOT, 'tokenizer_incremental.bin')
tok.load(sn.TOKENIZER_CHECKPOINT)
V = tok.vocab_size
lines = sn.load_lines(os.path.join(ROOT, DATA))
is_txt = DATA.endswith('.txt')
train_recs, val_recs = sn.split_records(lines)
dev = torch.device(DEV)
head = 'data=%s train=%d val=%d V=%d B=%d syn_cap=%d dev=%s' % (
    DATA, len(train_recs), len(val_recs), V, B, SYN_CAP, DEV)
print(head)
sn.log_and_print(head)


class BatchBrain:
    def __init__(self):
        self.V, self.B = V, B
        self.N = 2 * V + CO_CAP
        self.lam = torch.cat([torch.full((V,), LAM_F), torch.full((V,), LAM_S),
                              torch.full((CO_CAP,), 0.5)]).to(dev)
        # 二阶单元：检测 (fast[tok_a], slow[tok_b]) 的共激活 = 「上一个 token 与当前 token」组合
        self.co_a = torch.zeros(CO_CAP, dtype=torch.long, device=dev)   # 快通道索引（按需填充）
        self.co_b = torch.zeros(CO_CAP, dtype=torch.long, device=dev)   # 慢通道索引
        self.co_alive = torch.zeros(CO_CAP, dtype=torch.bool, device=dev)
        self.co_keys = {}
        self.n_co = 0
        self.v = torch.zeros(B, self.N, device=dev)
        self.b = torch.zeros(V, device=dev)
        self.s_i = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.s_j = torch.zeros(SYN_CAP, dtype=torch.long, device=dev)
        self.s_w = torch.zeros(SYN_CAP, device=dev)
        self.n_syn = 0
        self.keys = set()
        self.prev = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.last = torch.full((B,), -1, dtype=torch.long, device=dev)   # 上一步 token（慢通道源）
        self.err = torch.ones(V, device=dev)
        self.clock = 0

    def logits(self, ap):
        l = self.b.unsqueeze(0).expand(self.B, -1).clone()
        if self.n_syn:
            vals = ap[:, self.s_i[:self.n_syn]] * self.s_w[:self.n_syn].unsqueeze(0)
            l.scatter_add_(1, self.s_j[:self.n_syn].unsqueeze(0).expand(self.B, -1), vals)
        return l

    def add_edges(self, srcs, dsts):
        n0 = self.n_syn
        for s, d in zip(srcs, dsts):
            if s < 0 or d < 0 or d >= V:
                continue
            key = (s, d)
            if key in self.keys or self.n_syn >= SYN_CAP:
                continue
            i = self.n_syn
            self.s_i[i] = s
            self.s_j[i] = d
            self.s_w[i] = 0.0
            self.n_syn += 1
            self.keys.add(key)
        return self.n_syn - n0

    def step(self, x, learn=True, msk=None):
        """x: (B,) 当前 token；msk: (B,) 1=该位置产生学习信号（QA 语料的 assistant 段）。"""
        ap = self.v.clamp(0.0, 1.0)
        # 二阶（乘法）单元激活：a_i × a_j —— 唯一能表达「若 A 且 B」的机制
        co_act = ap[:, self.co_a] * ap[:, self.co_b] * self.co_alive.float().unsqueeze(0)
        ap = torch.cat([ap, co_act], dim=1)
        logits = self.logits(ap)
        if learn and self.clock > 0:
            p = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
            e = p.clone()
            e[torch.arange(self.B), x] -= 1.0
            if msk is not None:
                e = e * msk.unsqueeze(1)     # 非 mask 位置零：不学习、不计误差
            if self.n_syn:
                si, sj = self.s_i[:self.n_syn], self.s_j[:self.n_syn]
                # 批大小缩放：mean 会让每 token 学习量被平均掉 B 倍，
                # 改用 1/sqrt(B) 缩放（批训练的常用规则），等效步长 sqrt(B) 倍。
                scale = 1.0 / math.sqrt(B)
                grad = (ap[:, si] * e[:, sj]).sum(dim=0) * scale
                self.s_w[:self.n_syn] = torch.clamp(
                    self.s_w[:self.n_syn] - LR * grad, -CLIP, CLIP)
            self.b -= LR * e.sum(dim=0) * (1.0 / math.sqrt(B))
            self.err = 0.98 * self.err + 0.02 * e.abs().mean(dim=0)
            src = self.prev            # 与主内核一致：一阶边挂 fast[prev]
            ok = self.prev >= 0
            if ok.any():
                self.add_edges(src[ok].tolist(), x[ok].tolist())
                # 二阶单元：为观测到的 (prev, cur) 建一个专用乘法检测器（共激活驱动，非随机）
                if self.n_co < CO_CAP:
                    for b in torch.nonzero(ok).flatten().tolist():
                        key = (int(self.prev[b]), int(x[b]))
                        if key in self.co_keys:
                            continue
                        if self.n_co >= CO_CAP:
                            break
                        c = self.n_co
                        self.co_keys[key] = c
                        self.co_a[c] = key[0]
                        self.co_b[c] = key[1] + self.V
                        self.co_alive[c] = True
                        self.n_co += 1
                        nid = 2 * self.V + c
                        picks = np.random.randint(0, self.V, size=CO_FAN)
                        self.add_edges([nid] * CO_FAN, picks.tolist())
        self.v = self.lam.unsqueeze(0) * self.v + torch.randn_like(self.v) * 0.02
        ar = torch.arange(self.B, device=dev)
        self.v[ar, x] += 1.0                                  # 快通道：当前 token
        ok_slow = self.last >= 0
        if ok_slow.any():
            self.v[ar[ok_slow], self.last[ok_slow] + V] += 1.0   # 慢通道：上一个 token（保序）
        self.last = x.clone()
        self.prev = x.clone()
        self.clock += 1
        return self.logits(self.v.clamp(0.0, 1.0))


def eval_ppl(brain, recs, limit):
    nll, n = 0.0, 0
    for _ln, line in recs[:limit]:
        r = sn.render_sample(tok, line, is_txt)
        if r is None:
            continue
        ids, mask = r
        if len(ids) < 2:
            continue
        brain.v.zero_()
        brain.prev.fill_(-1)
        brain.last.fill_(-1)
        brain.clock = 0
        for i, tid in enumerate(ids):
            x = torch.full((B,), tid, dtype=torch.long, device=dev)
            if i > 0 and mask[i]:   # 只统计 assistant 段（与主内核同口径）
                logits = brain.logits(brain.v.clamp(0.0, 1.0))
                p = torch.softmax(logits[0], dim=0)
                nll -= math.log(max(float(p[tid]), 1e-12))
                n += 1
            brain.step(x, learn=False)
    return math.exp(nll / max(n, 1)), n


# ---- 数据并行：训练集拼成一条长流，切成 B 段，每个读者连续读一段 ----
# 旧实现把每条记录按 B 切片，导致每个流只看到 len//B ≈ 15 个 token 就换记录，
# 上下文完全断裂（这正是 batch 版 74 万 token 仍不如 6 万 token CPU 版的原因）。
all_ids, all_msk, all_start = [], [], []
for _ln, line in train_recs:
    r = sn.render_sample(tok, line, is_txt)
    if r is None:
        continue
    ids, m = r
    all_start.append(len(all_ids))          # 记录起点（跨记录处要重置该流状态）
    all_ids.extend(ids)
    all_msk.extend([1 if v else 0 for v in m])
all_ids = np.asarray(all_ids, dtype=np.int64)
all_msk = np.asarray(all_msk, dtype=np.float32)
if EPOCHS > 1:
    # 多轮：直接重复长流（流式无法随机重排，重复足够用；边界影响已实测可忽略）
    all_ids = np.tile(all_ids, EPOCHS)
    all_msk = np.tile(all_msk, EPOCHS)
total = int(min(len(all_ids), TOKENS))
seg = total // B
mat = all_ids[:seg * B].reshape(B, seg).T.copy()      # (seg, B)
matm = all_msk[:seg * B].reshape(B, seg).T.copy()
# 跨记录边界的位置：该流在此处必须清空膜电位（主内核每条记录 reset_state 的等价物）
starts = set(all_start)
reset_mat = np.zeros((seg, B), dtype=bool)
for b in range(B):
    base = b * seg
    for k in range(seg):
        if (base + k) in starts:
            reset_mat[k, b] = True
n_reset = int(reset_mat.sum())
print('[b2] record boundaries in stream: %d' % n_reset)
msg0 = '[b2] stream: %d tokens, B=%d, each reader %d tokens' % (total, B, seg)
print(msg0)
sn.log_and_print(msg0)

brain = BatchBrain()
print('[b2] co: 共激活按需建立，cap %d, fan %d' % (CO_CAP, CO_FAN))
t0 = time.time()
for t in range(seg - 1):
    rb = reset_mat[t]
    if rb.any():
        brain.v[torch.from_numpy(rb).to(dev)] = 0.0
    x = torch.from_numpy(mat[t]).to(dev)
    mk = torch.from_numpy(matm[t]).to(dev)
    brain.step(x, learn=True, msk=mk)
    if (t + 1) % 5000 == 0:
        done = (t + 1) * B
        msg = '[b2] tok %8d | %6.0f tok/s | syn %6d | err %.4f' % (
            done, done / (time.time() - t0), brain.n_syn, float(brain.err.mean()))
        print(msg)
        sn.log_and_print(msg)
ppl, n = eval_ppl(brain, val_recs, EVAL_RECS)
msg = '[b2] done | tok %d | VAL PPL %.1f (n=%d) | syn %d | %.0f tok/s' % (
    seg * B, ppl, n, brain.n_syn, seg * B / (time.time() - t0))
print(msg)
sn.log_and_print(msg)
sn.close_log_file()

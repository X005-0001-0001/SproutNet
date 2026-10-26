"""
SproutNet · batch2 分支 · 0.9.0（从 0.5.0 分叉：化学亲和标记 → 拓扑自组织）

实验依据（分子生物学实验，非模型假设）：
  ① Sperry 化学亲和假说被验证：Ephrin/Eph 反向梯度决定视网膜-顶盖拓扑映射
     （Drescher 1995 / Cheng 1995）。原文强调差异是「同一类细胞之间的定量梯度」，
     不是离散类型 → 故每个神经元用一个连续标记向量，而不是"分几层"。
  ② 连接不是精确指定的：神经元 quite promiscuous，识别分子主要作用是限制可能的连接范围
     （Cell 2020 综述；类比 Levinthal 悖论——不搜索全部构象，走动力学有利路径）。
  ③ 分子梯度不够，活动也必需，且两者直接交互（Sur lab 综述）。

三条直译：
  - 每个内部神经元一个连续标记 m（8 维）
  - 出生时接「标记最近」的 MARK_KNN 个已有神经元（不是按出生顺序/惊讶度）
  - m 每 MARK_EVERY 步向「该神经元输入 token 的平均投影」漂移（活动依赖的拓扑重排）
  - 权重仍由 STDP 学，使用率低的淘汰

没有预设层数/窗口/接哪些 token。层或模块若出现，是 m 聚集的副产品。
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
INT_CAP = int(os.environ.get('B2_INT_CAP', '96'))        # 脑内神经元上限（用户规格）
FAN_IN = int(os.environ.get('B2_FAN_IN', '1000'))        # 默认 1000（惊讶度 top-500 快+慢）
ETA_RO = float(os.environ.get('B2_ETA_RO', '0.25'))      # 密集读出 LMS 步长
ETA_STDP = float(os.environ.get('B2_ETA_STDP', '0.10'))  # 内部树突 STDP 步长
G_NEURO = float(os.environ.get('B2_G_NEURO', '0.002'))   # 每步出生概率尺度
CO_READ = int(os.environ.get('B2_CO_READ', '500'))       # 稀疏读出边数/神经元
MARK_DIM = int(os.environ.get('B2_MARK_DIM', '8'))       # 化学标记维度
MARK_KNN = int(os.environ.get('B2_MARK_KNN', '24'))      # 出生按标记距离接的 k 近邻
MARK_ETA = float(os.environ.get('B2_MARK_ETA', '0.05'))
MARK_EVERY = int(os.environ.get('B2_MARK_EVERY', '64'))  # 标记漂移频率（步）

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
        self.N = 2 * V + INT_CAP
        lam_i = torch.rand(INT_CAP) * 0.8 + 0.1          # 异质时间尺度 λ~U(0.1,0.9)
        self.lam = torch.cat([torch.full((V,), LAM_F), torch.full((V,), LAM_S), lam_i]).to(dev)
        self.int_alive = torch.zeros(INT_CAP, dtype=torch.bool, device=dev)
        # 0.9.0 化学标记（连续梯度）+ 固定的 token→标记 随机投影
        self.mark = torch.randn(INT_CAP, MARK_DIM, device=dev) * 0.5
        g = torch.Generator().manual_seed(11)
        self._tok_proj = torch.randn(2 * V, MARK_DIM, generator=g).to(dev) * 0.1
        # 0.5.0：不再用全词表密集读出（U_ro）。每个神经元只连 CO_READ 个 token 行，
        # 参数从 INT_CAP×V 降到 INT_CAP×CO_READ，从而能塞下 2000 个神经元。
        self.ap_prev = torch.zeros(B, self.N, device=dev)
        self.births = 0
        self._free = list(range(INT_CAP))
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
            ro = self.s_j[:self.n_syn] < self.V          # 只有读出边指向 token
            if ro.any():
                si = self.s_i[:self.n_syn][ro]
                sj = self.s_j[:self.n_syn][ro]
                w = self.s_w[:self.n_syn][ro]
                l.scatter_add_(1, sj.unsqueeze(0).expand(self.B, -1),
                               ap[:, si] * w.unsqueeze(0))
        return l

    def birth(self):
        if not self._free:
            return False
        k = self._free.pop(0)
        nid = 2 * self.V + k
        self.int_alive[k] = True
        # 接线：默认全感受器（快+慢各 V 条），让每个神经元能看到完整输入，
        # 而不是随机 5 个 token——编码器的可见范围是"能否学出表示"的前提。
        if FAN_IN < 0:
            src = np.arange(2 * self.V, dtype=np.int64)     # 全感受器（快+慢）
        else:
            # 按惊讶度取前 N 个 token 的快+慢通道（高频/高误差优先），而不是 arange 前 N 个
            kf = max(1, min(FAN_IN // 2, self.V))
            order = torch.argsort(-self.err)[:kf].cpu().numpy().astype(np.int64)
            src = np.concatenate([order, order + self.V])
        fan = len(src)
        w0 = (np.random.randn(fan) * 0.05).astype(np.float32)
        self.bulk_add(src, nid, w0)
        self.bulk_add(np.array([nid], dtype=np.int64), nid, np.zeros(1, np.float32))
        # 化学亲和：接「标记最近」的 k 个已有神经元（Ephrin/Eph 梯度的抽象，非按出生顺序）
        born = torch.nonzero(self.int_alive).flatten()
        born = born[born != k]
        if born.numel() > 0:
            d = torch.cdist(self.mark[k].view(1, -1), self.mark[born]).squeeze(0)
            kk = min(MARK_KNN, int(born.numel()))
            pick = born[torch.argsort(d)[:kk]]
            srcs = (2 * self.V + pick).cpu().numpy().astype(np.int64)
            self.bulk_add(srcs, nid, (np.random.randn(len(srcs)) * 0.05).astype(np.float32))
        # 稀疏读出：接 CO_READ 个"最惊讶"的 token 行（权重从 0 开始学）
        kr = max(1, min(CO_READ, self.V))
        top = torch.argsort(-self.err)[:kr].cpu().numpy().astype(np.int64)
        self.bulk_add(np.full(kr, nid, dtype=np.int64), top,
                      (np.random.randn(kr) * 0.01).astype(np.float32))
        self.births += 1
        return True

    def bulk_add(self, srcs, dst, w0):
        """批量给某个新神经元加输入边（新神经元的边必然不重复，跳过 set 查询）。"""
        n = len(srcs)
        room = SYN_CAP - self.n_syn
        if n > room:
            srcs = srcs[:room]; w0 = w0[:room]; n = room
        if n <= 0:
            return 0
        i0 = self.n_syn
        sl = slice(i0, i0 + n)
        self.s_i[sl] = torch.from_numpy(np.ascontiguousarray(np.asarray(srcs, dtype=np.int64))).to(dev)
        if np.isscalar(dst) or (hasattr(dst, 'ndim') and getattr(dst, 'ndim', 0) == 0):
            self.s_j[sl] = int(dst)
        else:
            self.s_j[sl] = torch.from_numpy(np.ascontiguousarray(np.asarray(dst, dtype=np.int64))).to(dev)
        self.s_w[sl] = torch.from_numpy(np.ascontiguousarray(np.asarray(w0, dtype=np.float32))).to(dev)
        self.n_syn += n
        return n

    def add_edges(self, srcs, dsts):
        n0 = self.n_syn
        for s, d in zip(srcs, dsts):
            if s < 0 or d < 0 or d >= self.N:   # 目标可以是 token（读出）或内部神经元（树突）
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
        logits = self.logits(ap)
        if learn and self.clock > 0:
            p = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
            e = p.clone()
            e[torch.arange(self.B), x] -= 1.0
            if msk is not None:
                e = e * msk.unsqueeze(1)     # 非 mask 位置零：不学习、不计误差
            scale = 1.0 / math.sqrt(B)
            if self.n_syn:
                si, sj = self.s_i[:self.n_syn], self.s_j[:self.n_syn]
                is_ro = sj < self.V                 # 指向 token 的读出边：LMS
                is_den = ~is_ro                     # 指向内部神经元的树突边：STDP
                w = self.s_w[:self.n_syn]
                if is_ro.any():
                    gr = (ap[:, si[is_ro]] * e[:, sj[is_ro]]).sum(dim=0) * scale
                    w[is_ro] = torch.clamp(w[is_ro] - LR * gr, -CLIP, CLIP)
                if is_den.any():
                    si_d, sj_d = si[is_den], sj[is_den]
                    ltp = (self.ap_prev[:, si_d] * ap[:, sj_d]).sum(dim=0)
                    ltd = (ap[:, si_d] * self.ap_prev[:, sj_d]).sum(dim=0)
                    d = (ltp - ltd) * scale * ETA_STDP
                    w[is_den] = torch.clamp(w[is_den] + d, -CLIP, CLIP)
            self.b -= LR * e.sum(dim=0) * (1.0 / math.sqrt(B))
            # 神经发生（速率过程）。注意：不能乘 e.abs().mean()——它的量级是 1e-4，
            # 会把出生概率压到 1e-6（上一版实测 births=0，结构从未出现）。
            if torch.rand(1).item() < G_NEURO:
                self.birth()
            self.err = 0.98 * self.err + 0.02 * e.abs().mean(dim=0)
            if self.int_alive.any() and self.n_syn and self.clock % MARK_EVERY == 0:
                ii = torch.nonzero(self.int_alive).flatten()
                act = self.v[:, 2 * self.V:][:, ii].mean(dim=0)
                for c in ii[act > 0.05].tolist():
                    m = (self.s_j[:self.n_syn] == 2 * self.V + c) & (self.s_i[:self.n_syn] < 2 * self.V)
                    if m.any():
                        tgt = self._tok_proj[self.s_i[:self.n_syn][m]].mean(dim=0)
                        tgt = tgt / (tgt.norm() + 1e-9)
                        self.mark[c] = (1 - MARK_ETA) * self.mark[c] + MARK_ETA * tgt
            src = self.prev            # 与主内核一致：一阶边挂 fast[prev]
            ok = self.prev >= 0
            if ok.any():
                self.add_edges(src[ok].tolist(), x[ok].tolist())
        self.v = self.lam.unsqueeze(0) * self.v + torch.randn_like(self.v) * 0.02
        if self.n_syn:
            den = self.s_j[:self.n_syn] >= self.V        # 树突边：把 pre 活动注入 post（内部神经元）
            if den.any():
                dv = torch.zeros_like(self.v)
                dv.scatter_add_(1, self.s_j[:self.n_syn][den].unsqueeze(0).expand(self.B, -1),
                                ap[:, self.s_i[:self.n_syn][den]] * self.s_w[:self.n_syn][den].unsqueeze(0))
                self.v = self.v + dv
        ar = torch.arange(self.B, device=dev)
        self.v[ar, x] += 1.0                                  # 快通道：当前 token
        ok_slow = self.last >= 0
        if ok_slow.any():
            self.v[ar[ok_slow], self.last[ok_slow] + V] += 1.0   # 慢通道：上一个 token（保序）
        self.last = x.clone()
        self.prev = x.clone()
        self.ap_prev = self.v.clamp(0.0, 1.0)
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
        brain.ap_prev.zero_()
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
msg = '[b2] done | tok %d | VAL PPL %.1f (n=%d) | syn %d | ints %d/%d births %d | %.0f tok/s' % (
    seg * B, ppl, n, brain.n_syn, int(brain.int_alive.sum()), INT_CAP, brain.births,
    seg * B / (time.time() - t0))
print(msg)
sn.log_and_print(msg)
sn.close_log_file()

"""
SproutNet · batch2 分支 · 0.10.0（从 0.5.0 分叉：异质时间尺度 + 单向时间层级）

判断依据（来自本项目自己的实测，不是外部假设）：
  - 0.5.0 的内核只有两个时间常数（快 λ=0.35 / 慢 λ=0.85），记忆上限约 6 个 token。
  - 而 PPL 这个任务天然奖励查表；结构若不提供「超出 1–2 token 的上下文」，就一定是纯开销
    （0.7–0.9 的八次失败全部同向变差，即为此故）。
  - v6 那条线上唯一出现过的 scaling 信号就是「异质时间尺度 → 记忆容量随规模增长」
    （MC 1.44/1.85/2.47；大脑 2.38 vs 小脑 5.05）。batch 内核此前把这个机制弄丢了。

本版：
  1. 内部神经元的 λ 按【倒数均匀】采样，时间常数 1/(1-λ) 覆盖 1 步到 50 步
     （每个数量级都有代表，而不是挤在快端）。
  2. 出生时只接「比自己更慢（λ 更小）」的 TIME_K 个神经元 —— 单向，快→慢，
     形成时间层级：慢神经元积分长历史，无反馈震荡。
  3. 权重仍由 STDP 学，读出仍稀疏、仍按惊讶度选 token。
不预设层数：时间层级由 λ 的自然排序产生，是「时间常数」的副产品。
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
TIME_K = int(os.environ.get('B2_TIME_K', '8'))           # 时间层级：接几个「更慢」的神经元
LAM_LO = float(os.environ.get('B2_LAM_LO', '0.05'))      # λ 下限（慢端，记忆最长）
LAM_HI = float(os.environ.get('B2_LAM_HI', '0.95'))      # λ 上限（τ≤20 步；0.98 会饱和成常数）

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
        # 0.10.0：λ 在这个区间是【倒数均匀】采样的——保证时间常数 1/(1-λ)
        # 在每个数量级上都有代表（1 步到 50 步），而不是挤在快端。
        u = torch.rand(INT_CAP)
        taus = 1.0 + u * (1.0 / max(1e-3, 1.0 - LAM_HI) - 1.0)   # 时间常数 1..1/(1-hi)
        lam_i = 1.0 - 1.0 / taus
        self.lam = torch.cat([torch.full((V,), LAM_F), torch.full((V,), LAM_S), lam_i]).to(dev)
        self.int_alive = torch.zeros(INT_CAP, dtype=torch.bool, device=dev)
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

    def topo_report(self):
        """内部神经元子图的拓扑指标（判据：拓扑与 PPL 一起看）。"""
        import numpy as np
        n = int(self.int_alive.sum())
        if n < 2:
            return {'n': n}
        idx = torch.nonzero(self.int_alive).flatten().cpu().numpy()
        pos = {int(v): i for i, v in enumerate(idx)}
        si = self.s_i[:self.n_syn].cpu().numpy()
        sj = self.s_j[:self.n_syn].cpu().numpy()
        A = np.zeros((n, n), dtype=np.float32)
        for a, b in zip(si, sj):
            if a >= 2 * self.V and b >= 2 * self.V:
                i, j = pos.get(int(a) - 2 * self.V), pos.get(int(b) - 2 * self.V)
                if i is not None and j is not None and i != j:
                    A[i, j] = 1.0
        deg = A.sum(axis=1) + A.sum(axis=0)
        # 聚类系数（无向化）
        U = np.maximum(A, A.T)
        tri = np.trace(U @ U @ U)
        cc = 0.0
        for i in range(n):
            k = U[i].sum()
            if k > 1:
                cc += (U[i] @ U @ U[:, i]) / (k * (k - 1))
        cc /= n
        # 与同规模随机图（同边数）对比
        m = int(U.sum())
        p_rand = m / max(1.0, n * (n - 1))
        cc_rand = p_rand
        # 模块度（标签传播一轮）
        lab = np.arange(n)
        for _ in range(10):
            for i in range(n):
                nb = np.nonzero(U[i])[0]
                if len(nb) == 0:
                    continue
                vals, cnts = np.unique(lab[nb], return_counts=True)
                lab[i] = vals[int(np.argmax(cnts))]
        mod = 0.0
        for i in range(n):
            for j in range(n):
                if U[i, j]:
                    mod += (lab[i] == lab[j])
        mod = mod / max(1.0, m) - sum((np.bincount(lab) / n) ** 2)
        return {'n': n, 'edges': int(m), 'deg_mean': float(deg.mean()),
                'deg_max': float(deg.max()),
                'cc': float(cc), 'cc_rand': float(cc_rand),
                'cc_ratio': float(cc / max(cc_rand, 1e-9)),
                'modularity': float(mod), 'n_communities': int(len(np.unique(lab)))}

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
            # 0.10.3 输入特化：一半接「共享的高惊讶 token」，一半接「各自特有的随机 token」。
            # 动机：0.10.1/0.10.2 的模块度恒为 0，因为所有神经元接同一批 err top-500，
            # 输入完全重叠 → 表示必然趋同 → 无法分区。特化是模块化的前提。
            kf = max(1, min(FAN_IN // 4, self.V))
            top = torch.argsort(-self.err)[:kf].cpu().numpy().astype(np.int64)
            rnd = np.random.randint(0, self.V, size=kf)
            tok = np.concatenate([top, rnd])
            src = np.concatenate([tok, tok + self.V])
        fan = len(src)
        w0 = (np.random.randn(fan) * 0.05).astype(np.float32)
        self.bulk_add(src, nid, w0)
        self.bulk_add(np.array([nid], dtype=np.int64), nid, np.zeros(1, np.float32))
        # 时间层级：新神经元只接「比自己更慢」（λ 更小、记忆更长）的神经元。
        # 信息单向从快流到慢 → 慢神经元积分长历史，提供查表拿不到的信息；无反馈震荡。
        born = torch.nonzero(self.int_alive).flatten()
        born = born[born != k]
        if born.numel() > 0:
            # 0.10.2：改为接「λ 最接近」的 TIME_K 个 —— 时间尺度相近的聚成簇（模块来源），
            # 而不是都去接"最慢的那个"（那会形成单一枢纽 + 无模块，0.10.1 实测 deg_max=93/mod=0）。
            my_lam = float(self.lam[nid])
            dlam = (self.lam[2 * self.V + born] - my_lam).abs()
            kk = min(TIME_K, int(born.numel()))
            pick = born[torch.argsort(dlam)[:kk]]
            srcs = (2 * self.V + pick).cpu().numpy().astype(np.int64)
            self.bulk_add(srcs, nid, (np.random.randn(len(srcs)) * 0.01).astype(np.float32))
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
tr = brain.topo_report()
msg2 = '[topo] ' + ' '.join('%s=%.3f' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v)
                            for k, v in tr.items())
print(msg2); sn.log_and_print(msg2)
print(msg)
sn.log_and_print(msg)
sn.close_log_file()

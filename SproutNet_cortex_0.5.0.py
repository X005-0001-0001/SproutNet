# -*- coding: utf-8 -*-
"""SproutNet · cortex 0.5.0
方案见设计文档。核心：生成机制产生层/跳跃/比例，而不是我填。

生成机制
  出生：每批 N_batch=128，触发条件=近期误差不再下降
  距离：dist = 批次序号差
  连接概率 p(dist) = 0.10 · exp(−dist / 1.5)
    → 同批 0.10、前一批 0.051、前两批 0.026、更远 <0.01
    → 「层数」「不跳跃」「层内比例」都是它的副产品

神经元    LIF + 4 段树突（段内 sqrt）+ θ 稳态
可塑性    三因子：迹式 STDP（A-/A+=0.55）× 全局标量 m，配突触缩放
输出      只有末批 → 词（列稀疏 + LMS）
无表      没有 bigram
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

SEED = int(os.environ.get('B2_SEED', '0'))
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get('TB_THREADS', '6')))

ROOT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sn', os.path.join(ROOT, 'SproutNet_coverage_0.9.2.py'))
sn = importlib.util.module_from_spec(spec)
sys.argv = ['cortex5']
spec.loader.exec_module(sn)
sn._LOG_FILE = None

DATA = os.environ.get('B2_DATA', '_pretrain_300k.txt')
TOKENS = int(os.environ.get('B2_TOKENS', '300000'))
EPOCHS = int(os.environ.get('B2_EPOCHS', '1'))
B = min(32, int(os.environ.get('B2_B', '16')))
DEV = os.environ.get('B2_DEV', 'cuda')
dev = torch.device(DEV)

MAXN = int(os.environ.get('B2_MAXN', '2048'))          # 神经元容量上限
N_BATCH = int(os.environ.get('B2_NBATCH', '128'))      # 每批出生数
L_AXON = float(os.environ.get('B2_LAXON', '1.5'))      # 轴突长度（批间距）
P0 = float(os.environ.get('B2_P0', '0.10'))            # 连接概率基数
SYN_PER_CONN = 2.5                                     # 每连接平均突触数（实测 1.8-4.5）
DSEG = int(os.environ.get('B2_DSEG', '4'))             # 树突段数
KSCALE = int(os.environ.get('B2_KSCALE', '4'))
LR_OUT = float(os.environ.get('B2_LR', '0.05'))
ETA = float(os.environ.get('B2_ETA', '0.01'))
A_PLUS, A_MINUS = 1.0, 0.55
RHO_TRACE = 0.9
SCALE_TARGET = 4.0
TARGET_RATE = 0.05
ENERGY_MB = float(os.environ.get('B2_ENERGY_MB', '512'))
SYN_COST, NEURON_COST = 20.0, 200.0
GROW_WIN = int(os.environ.get('B2_GROW_WIN', '600'))
OUT_BATCHES = int(os.environ.get('B2_OUT_BATCHES', '4'))   # 输出接末几批
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '5'))
PROF = os.environ.get('B2_PROF', '0') == '1'
CLIP = 6.0

sn.DATA_RECORDS_LIMIT = 100000
tok = sn.FastBPE(sn.VOCAB_SIZE)
sn.TOKENIZER_CHECKPOINT = os.path.join(ROOT, 'tokenizer_incremental.bin')
tok.load(sn.TOKENIZER_CHECKPOINT)
V = tok.vocab_size
lines = sn.load_lines(os.path.join(ROOT, DATA))
is_txt = DATA.endswith('.txt')
train_recs, val_recs = sn.split_records(lines)

# ---- 显存守卫 ----
MAX_SYN = 4_000_000
if dev.type == 'cuda':
    try:
        free, total = torch.cuda.mem_get_info()
        budget = total * 0.55
        per_syn = 20.0 * 4          # si/sj(long=8B×2) + sw(4B) + 迹/辅助 ≈ 20B×4
        est = MAXN * (N_BATCH * SYN_PER_CONN * 2) * 20 + V * 400 * 20
        if est > budget:
            scale = budget / est
            MAXN = max(256, int(MAXN * scale))
            print('[guard] MAXN -> %d (预算 %.1fGB, 估算 %.1fGB)' % (
                MAXN, budget / 1024**3, est / 1024**3))
    except Exception as e:
        print('[guard] skip:', e)


class Cortex:
    def __init__(self):
        self.KV = KSCALE * V
        self.Nn = self.KV + MAXN
        # 感受器：K 组时间尺度
        taus = torch.logspace(0, math.log10(40.0), KSCALE)
        self.lam_s = (1.0 - 1.0 / taus).repeat_interleave(V).to(dev)
        self.v = torch.zeros(B, self.Nn, device=dev)
        # 神经元
        self.alive = torch.zeros(MAXN, dtype=torch.bool, device=dev)
        self.vm = torch.zeros(B, MAXN, device=dev)
        self.theta = torch.full((MAXN,), 0.02, device=dev)
        self.spk = torch.zeros(B, MAXN, device=dev)
        self.lam_n = (torch.rand(MAXN) * 0.5 + 0.4).to(dev)
        self.vk = torch.full((MAXN, DSEG), 1.0, device=dev)
        self.batch_of = torch.full((MAXN,), -1, dtype=torch.long, device=dev)
        # 连续位置标记：mark = 批号 + U(0,1)。层是 mark 的聚集（连续），不是离散批。
        self.mark = torch.zeros(MAXN, device=dev)
        self.n_alive = 0
        self.n_batch = 0
        # 突触
        self.si = torch.zeros(MAX_SYN, dtype=torch.long, device=dev)
        self.sj = torch.zeros(MAX_SYN, dtype=torch.long, device=dev)
        self.sw = torch.zeros(MAX_SYN, device=dev)
        self.seg = torch.zeros(MAX_SYN, dtype=torch.long, device=dev)
        self.pre_tr = torch.zeros(MAX_SYN, device=dev)
        self.post_tr = torch.zeros(MAX_SYN, device=dev)
        self.use_cnt = torch.zeros(MAX_SYN, device=dev)
        self.ns = 0
        # 输出
        self.out_n = torch.zeros(V, 2048, dtype=torch.long, device=dev)
        self.out_w = torch.zeros(V, 2048, device=dev)
        self.n_out = 0
        self.b = torch.zeros(V, device=dev)
        # 统计
        self.err_slow = None
        # 预分配缓冲（每步 new 大张量是主要开销：82ms/step 里大头在这）
        self._node = torch.zeros(B, self.KV + MAXN, device=dev)
        self._acc = torch.zeros(B, MAXN * DSEG, device=dev)
        self._val = torch.zeros(B, MAX_SYN, device=dev)
        self.err_hist = []
        self.energy_used = 0.0
        self.clock = 0
        self.grow(0)          # 第一批

    # ---------- 生成 ----------
    def grow(self, parent_batch):
        """长一批神经元，并按 p(dist) 接线。"""
        if self.n_alive + N_BATCH > MAXN:
            return False
        need = N_BATCH * (NEURON_COST + N_BATCH * SYN_PER_CONN * SYN_COST)
        if self.energy_used + need > ENERGY_MB * 1024 * 1024:
            return False
        b = self.n_batch
        lo = self.n_alive
        hi = lo + N_BATCH
        self.batch_of[lo:hi] = b
        self.mark[lo:hi] = torch.rand(N_BATCH, device=dev) + b   # 连续位置
        self.alive[lo:hi] = True
        self.n_alive = hi
        self.n_batch += 1
        g = torch.Generator().manual_seed(1000 + b)
        # 输入：第 0 批接感受器；其余接更早的批
        for k in range(lo, hi):
            # 所有批都接感觉输入（实测皮层各层都接收丘脑输入，不只第 0 层）。
            # 数量用统一规则 fan_in_sens = p0 × V（词表规模），不是魔数。
            n_sens = max(8, int(P0 * V))
            toks = torch.randint(0, V, (max(4, n_sens // KSCALE),), generator=g)
            src = torch.cat([toks + s_ * V for s_ in range(KSCALE)]).to(dev)
            self._add(src, torch.full((src.numel(),), k, dtype=torch.long, device=dev))
            if b == 0:
                # 第一批还要同批循环（否则图是碎的）
                m = int(N_BATCH * P0)
                if m > 0:
                    pick = torch.randint(lo, hi, (m,), generator=g)
                    self._add((pick + self.KV).to(dev),
                              torch.full((m,), k, dtype=torch.long, device=dev))
            else:
                # 连续距离采样：从所有更早的神经元里按 p₀·exp(−|Δmark|/L) 选。
                # 不再按「批序号差」——层是 mark 的连续聚集，"不跳跃"由距离衰减自然产生。
                prev = torch.arange(0, lo, device=dev)
                dist = (self.mark[k] - self.mark[:lo]).abs()
                prob = P0 * torch.exp(-dist / L_AXON)
                m = int(min(lo, max(4, lo * P0)))          # 期望 fan_in = p₀·N
                pick = torch.multinomial(prob.clamp(min=1e-9), min(m, lo), replacement=False)
                self._add((prev[pick]).to(dev),
                          torch.full((pick.numel(),), k, dtype=torch.long, device=dev))
        self.energy_used += need
        return True

    def _add(self, src_b, dst_b):
        """src_b/dst_b 是神经元局部序号（0..MAXN-1），转成突触存储。"""
        n = src_b.numel()
        if n == 0 or self.ns + n > MAX_SYN:
            return
        sl = slice(self.ns, self.ns + n)
        self.si[sl] = src_b
        self.sj[sl] = dst_b
        self.sw[sl] = torch.randn(n, device=dev) * 0.05
        self.seg[sl] = torch.randint(0, DSEG, (n,), device=dev)
        self.pre_tr[sl] = 0.0
        self.post_tr[sl] = 0.0
        self.ns += n

    # ---------- 前向 ----------
    def step(self, x, learn=True):
        # 传入节点的活动（感受器 + 神经元）
        # 节点向量 = [感受器(KV) | 神经元(MAXN)]，复用缓冲
        node = self._node
        node[:, :self.KV].copy_(self.v[:, :self.KV].clamp(0.0, 1.0))
        node[:, self.KV:self.KV + self.n_alive].copy_(self.spk[:, :self.n_alive])

        # 段内聚合 + sqrt + 胞体汇总
        if self.ns:
            ns = self.ns
            si, sw = self.si[:ns], self.sw[:ns]
            val = torch.index_select(node, 1, si) * sw.unsqueeze(0)
            acc = self._acc.zero_()
            flat = self.sj[:ns] * DSEG + self.seg[:ns]
            acc.scatter_add_(1, flat.unsqueeze(0).expand(B, -1), val)
            seg_act = torch.sqrt(acc.clamp(min=0.0) / max(1.0, float(DSEG)))
            drive = (seg_act.view(B, MAXN, DSEG) * self.vk.unsqueeze(0)).sum(dim=2)

        else:
            drive = torch.zeros(B, MAXN, device=dev)
        # 第 0 批还要接收感受器（感受器不进上面那张表）
        # LIF
        live = self.alive.unsqueeze(0)
        vm = self.lam_n.unsqueeze(0) * self.vm + (1.0 - self.lam_n.unsqueeze(0)) * drive
        spk = ((vm >= self.theta.unsqueeze(0)) & live).float()
        self.vm = vm * (1.0 - spk)
        self.spk = spk

        # 输出接「末 OUT_BATCHES 批」：固定 N_OUT_TOTAL 列，实际活动放在最右端，
        # 未出生的神经元保持 0（对输出无贡献）。这样稀疏矩阵结构固定、不随生长变化。
        N_OUT_TOTAL_ = OUT_BATCHES * N_BATCH
        last_lo = max(0, self.n_alive - N_OUT_TOTAL_)
        act = spk[:, last_lo:self.n_alive]
        if act.shape[1] < N_OUT_TOTAL_:
            last = torch.zeros(B, N_OUT_TOTAL_, device=dev)
            last[:, N_OUT_TOTAL_ - act.shape[1]:] = act
        else:
            last = act
        logits = self.b.unsqueeze(0).expand(B, -1).clone()
        if self._out_sp is None and self.n_alive >= MAXN:
            self.build_output()
        if self._out_sp is not None:
            # l = last @ W，W 是 (N_last, V) 稀疏矩阵（结构固定，只换 values）
            self._out_sp._values().copy_(self.out_w_flat)
            logits = logits + torch.sparse.mm(self._out_sp.t(), last.t()).t()
        if learn and self.clock > 0:
            p = torch.softmax(logits - logits.max(dim=1, keepdim=True).values, dim=1)
            e = p.clone()
            e[torch.arange(B), x] -= 1.0
            # 奖励预测误差：逐样本的惊讶度（NLL），减去其慢平均。
            # 原来的 err=e.abs().mean() 变化极慢，m≈0，dw≈1.6e-8 等于没学。
            nll = -(p.gather(1, x.view(-1, 1)).clamp(min=1e-12).log()).squeeze(1)   # (B,)
            err_t = nll.mean()
            # 输出 LMS
            if self.n_out > 0 and self._out_sp is not None:
                # 直接更新稀疏矩阵的 values（前向用的就是它；原来更新的是已废弃的 self.out_w）
                gw = (e[:, self._out_r] * last[:, self._out_c]).mean(dim=0) * (B ** 0.5)
                # 学习率按 sqrt(n_out) 缩放：fan 从 48 提到 1024 时梯度大 20 倍，不缩放会爆炸
                _lr = LR_OUT * math.sqrt(48.0 / max(48, self.n_out))
                self.out_w_flat.sub_(_lr * gw).clamp_(-CLIP, CLIP)
            self.b -= LR_OUT * e.sum(dim=0) / math.sqrt(B)

            # 三因子：迹式 STDP × 全局标量 m
            if self.err_slow is None:
                self.err_slow = err_t.detach()
            # m = 惊讶度相对慢平均的偏差（一个标量）
            m = (err_t - self.err_slow).clamp(-3.0, 3.0)
            self.err_slow = 0.99 * self.err_slow + 0.01 * err_t.detach()
            if self.ns and (self.clock % 2 == 0):
                # 一次索引、一次归约，避免多次 (B, ns) 的 mean（每次都是一个 kernel）
                ns = self.ns
                pre_now = torch.index_select(node, 1, self.si[:ns]).mean(dim=0)
                post_now = torch.index_select(node, 1, self.KV + self.sj[:ns]).mean(dim=0)
                self.pre_tr[:ns].mul_(RHO_TRACE).add_(pre_now)
                self.post_tr[:ns].mul_(RHO_TRACE).add_(post_now)
                elig = (A_PLUS * self.pre_tr[:ns] * post_now
                        - A_MINUS * pre_now * self.post_tr[:ns])
                _dw = ETA * elig * m
                self.sw[:ns].add_(_dw).clamp_(-CLIP, CLIP)
                self._dw_stat = float(_dw.abs().mean()) if self.clock % 500 == 0 else getattr(self, '_dw_stat', 0.0)
                self._m_stat = float(m) if self.clock % 500 == 0 else getattr(self, '_m_stat', 0.0)
                self.use_cnt[:ns].add_(pre_now)

            # 生长
            self._rate = float(spk.mean()) if self.clock % 200 == 0 else getattr(self, '_rate', -1)
            self._last_act = float(last.mean()) if self.clock % 200 == 0 else getattr(self, '_last_act', -1)
            self.err_hist.append(err_t.detach())
            if len(self.err_hist) >= GROW_WIN:
                # 只在检查点同步一次
                recent = float(torch.stack(self.err_hist[-200:]).mean())
                older = float(torch.stack(self.err_hist[:200]).mean())
                self.err_hist = self.err_hist[-GROW_WIN:]
                if recent > older * 0.995:
                    self.grow(self.n_batch - 1)
        # 感受器推进
        ar = torch.arange(B, device=dev)
        for g_ in range(KSCALE):
            self.v[ar, g_ * V + x] += 1.0
        self.v[:, :self.KV] = self.lam_s.unsqueeze(0) * self.v[:, :self.KV]
        self.clock += 1
        return logits

    def build_output(self):
        """长满后建立输出接线：每个词接 fan_in = p0 × n_alive 个末批神经元。"""
        N_OUT_TOTAL = min(MAXN, OUT_BATCHES * N_BATCH)
        self.n_out = max(16, min(int(P0 * self.n_alive), N_OUT_TOTAL))
        gn = torch.Generator().manual_seed(77)
        self.out_n[:, :self.n_out] = torch.randint(0, N_OUT_TOTAL, (V, self.n_out),
                                                   generator=gn).to(dev)
        self.out_w[:, :self.n_out] = (torch.randn(V, self.n_out, generator=gn) * 0.01).to(dev)
        _idx_r = torch.arange(V, device=dev).repeat_interleave(self.n_out)
        _idx_c = self.out_n[:, :self.n_out].flatten().to(dev)
        _w = self.out_w[:, :self.n_out].flatten().contiguous()
        self._out_sp = torch.sparse_coo_tensor(
            torch.stack([_idx_c, _idx_r]), _w, (N_OUT_TOTAL, V), device=dev).coalesce()
        self.out_w_flat = self._out_sp._values().clone()
        _ind = self._out_sp.indices()
        self._out_c, self._out_r = _ind[0], _ind[1]
        print('[cortex5] output built: n_out=%d nnz=%d (p0×n_alive=%d)' % (
            self.n_out, self._out_sp._nnz(), int(P0 * self.n_alive)), flush=True)

    def reset(self):
        self.v.zero_()
        self.vm.zero_()
        self.spk.zero_()
        self.clock = 0

    def topo(self):
        n = self.n_alive
        if n < 2:
            return {'n': n}
        si = self.si[:self.ns].cpu().numpy()
        sj = self.sj[:self.ns].cpu().numpy()
        A = np.zeros((n, n), dtype=np.float32)
        for a, b in zip(si, sj):
            if a < n and b < n and a != b:
                A[a, b] = 1.0
        U = np.maximum(A, A.T)
        mm = int(U.sum())
        cc = 0.0
        for i in range(n):
            k = U[i].sum()
            if k > 1:
                cc += (U[i] @ U @ U[:, i]) / (k * (k - 1))
        cc /= n
        deg = A.sum(1) + A.sum(0)
        return {'n': n, 'edges': mm, 'deg_mean': float(deg.mean()), 'deg_max': float(deg.max()),
                'cc': float(cc), 'cc_rand': float(mm / max(1.0, n * (n - 1))),
                'batches': self.n_batch}


def eval_ppl(br, limit):
    nll, n = 0.0, 0
    for _ln, line in val_recs[:limit]:
        r = sn.render_sample(tok, line, is_txt)
        if r is None:
            continue
        ids, mask = r
        if len(ids) < 2:
            continue
        br.reset()
        for i, tid in enumerate(ids):
            x = torch.full((B,), tid, dtype=torch.long, device=dev)
            lg = br.step(x, learn=False)
            if i > 0 and mask[i]:
                p = torch.softmax(lg[0], dim=0)
                nll -= math.log(max(float(p[tid]), 1e-12))
                n += 1
    return math.exp(nll / max(n, 1)), n


# ---- 数据 ----
cache = os.path.join(ROOT, '_sprout_build', 'tokcache_cortex.npz')
if os.path.exists(cache):
    all_ids = np.load(cache)['ids']
else:
    lst = []
    for _ln, line in train_recs:
        r = sn.render_sample(tok, line, is_txt)
        if r is None:
            continue
        a, _m = r
        lst.extend(a)
    all_ids = np.asarray(lst, dtype=np.int64)
    np.savez_compressed(cache, ids=all_ids)
total = int(min(len(all_ids), TOKENS))
seg = total // B
mat = all_ids[:seg * B].reshape(B, seg).T.copy()

br = Cortex()
# 输出接线：每个词连末批 n_out 个神经元（0.1 × N_BATCH）
br.n_out = max(16, min(int(P0 * br.n_alive), N_BATCH * OUT_BATCHES))   # 与内部连接同一规则
gn = torch.Generator().manual_seed(77)
br.out_n[:, :br.n_out] = torch.randint(0, N_BATCH * OUT_BATCHES, (V, br.n_out), generator=gn).to(dev)
br.out_w[:, :br.n_out] = (torch.randn(V, br.n_out, generator=gn) * 0.01).to(dev)
# 输出接线推迟到「脑长满之后」（生物学：突触形成有先后）。
# n_out 由统一规则 fan_in = p0 × n_alive 给出，不是魔数。
br._out_sp = None
print('[cortex5] V=%d KV=%d MAXN=%d B=%d (输出待长满后建立)' % (V, br.KV, MAXN, B))

t0 = time.time()
_prof = None
if os.environ.get('B2_PROF', '0') == '1':
    _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                   record_shapes=False)
for ep in range(EPOCHS):
    for t in range(seg - 1):
        if _prof is not None and t == 100:
            _prof.start()
        br.step(torch.from_numpy(mat[t]).to(dev), learn=True)
        if _prof is not None and t == 200:
            _prof.stop()
            print(_prof.key_averages().table(sort_by='cuda_time_total', row_limit=12))
            _prof = None
        if (t + 1) % 1000 == 0:
            print('  t=%d %.0f tok/s ns=%d alive=%d batches=%d' % (
                t + 1, (t + 1) * B / (time.time() - t0), br.ns, br.n_alive, br.n_batch),
                flush=True)

ppl, n = eval_ppl(br, EVAL_RECS)
print('[verify] 发放率=%.4f  末层活动均值=%.4f  theta均值=%.4f' % (
    getattr(br, '_rate', -1), getattr(br, '_last_act', -1), float(br.theta[:br.n_alive].mean())))
print('[verify] |dw|mean=%.3e  m=%.3e  (m 在 step 内是 0 维张量)' % (
    getattr(br, '_dw_stat', -1.0), getattr(br, '_m_stat', 0.0)))
print('[cortex5] done | tok=%d | VAL PPL %.1f (n=%d) | syn=%d alive=%d batches=%d | %.0f tok/s' % (
    seg * B, ppl, n, br.ns, br.n_alive, br.n_batch, seg * B / (time.time() - t0)))
tr = br.topo()
print('[topo] ' + ' '.join('%s=%.3f' % (k, v) if isinstance(v, float) else '%s=%s' % (k, v)
                          for k, v in tr.items()))

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
# 0.10.0：Hebbian + 竞争（Oja 规则），替代失败的三因子。
# 实验依据：von der Malsburg 1973 / Linsker 1986 / Miller 1989 —— Hebbian + 资源竞争
# 能自发产生朝向选择性感受野；原文强调 "as some inputs grow others must shrink"。
# Oja 1982：Δw = η·post·(pre − post·w)  —— Hebbian 的自然归一化形式，可学出 PCA 分量。
# Hyvärinen 2009：ICA 在自然图像上产生朝向感受野（与 V1 实测一致）。
OJA = os.environ.get('B2_OJA', '1') == '1'
OJA_ETA = float(os.environ.get('B2_OJA_ETA', '0.002'))
# 局部预测编码：w += η · (实际下游活动 − 预测的驱动) · 上游活动
# 依据 Rao & Ballard 1999 (predictive coding in V1) / Bastos 2012 (canonical microcircuit)：
# 皮层用「自上而下的预测」与「自下而上的误差」就地更新，不需要全局反向传播。
PRED = os.environ.get('B2_PRED', '1') == '1'
PRED_ETA = float(os.environ.get('B2_PRED_ETA', '0.02'))
# 诊断开关：输出额外接「前一 token」（慢通道 v[:, V:2V]），用来测量输出机制本身是否正常。
# 若接上后 PPL 掉到 bigram 附近，说明输出没问题、问题在神经元；否则输出机制也有问题。
DIAG_PREV = os.environ.get('B2_DIAG_PREV', '0') == '1'
FAN_PREV = int(os.environ.get('B2_FAN_PREV', '500'))
# 按共现选输入（化学亲和的可操作形式：经常一起激活的连在一起）
COOC = os.environ.get('B2_COOC', '1') == '1'
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
# 突触上限按「神经元数 × 每神经元突触数」给足，否则容量扩大后会被截断
# （实测 MAXN=5000 时 5000×1000=500 万条感受器边，撞 400 万上限）
MAX_SYN = int(os.environ.get('B2_SYN', str(8_000_000)))
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
        self.act_cnt = torch.zeros(MAXN, device=dev)   # 每个神经元的累积发放数
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
        # 预计算「每个 token 的高频后继」（共现表），供 grow 选输入用
        if COOC:
            _co = np.load(os.path.join(ROOT, '_sprout_build', 'cooc_bigram.npz'))
            _k, _v = _co['keys'].astype(np.int64), _co['vals']
            _src = _k // V
            _dst = _k % V
            _nsucc = int(os.environ.get('B2_NSUCC', '1000'))
            self.succ = torch.zeros(V, _nsucc, dtype=torch.long, device=dev)
            for t in range(V):
                m = _src == t
                if m.any():
                    d, v = _dst[m], _v[m]
                    top = np.argsort(-v)[:_nsucc]
                    self.succ[t, :len(top)] = torch.from_numpy(d[top]).to(dev)
                else:
                    self.succ[t] = torch.randint(0, V, (_nsucc,))
            print('[cooc] succ table built')
        # 诊断用的「前一 token → 词」稀疏矩阵（每词接 FAN_PREV 个前驱）
        _gp = torch.Generator().manual_seed(SEED * 10000 + 41)
        self.prev_idx = torch.randint(0, V, (V, FAN_PREV), generator=_gp).to(dev)
        self.prev_w = torch.zeros(V, FAN_PREV, device=dev)
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
        g = torch.Generator().manual_seed(SEED * 10000 + 1000 + b)
        # 输入接线：整批一次写入（原来每个神经元单独 _add，4992 个神经元 = 4992 次 GPU 操作，极慢）
        n_sens = max(8, int(P0 * V))   # 每神经元的感受器输入数（p0 × 词表）
        n_tok = max(4, n_sens // KSCALE)
        if COOC:
            # 每个神经元分配一个种子 token，输入 = 该种子的高频后继（化学亲和）
            seeds = torch.randint(0, V, (N_BATCH,), generator=g)
            toks = self.succ[seeds.to(dev), :n_tok].cpu()
        else:
            toks = torch.randint(0, V, (N_BATCH, n_tok), generator=g)
        src_mat = torch.cat([toks + s_ * V for s_ in range(KSCALE)], dim=1)   # (N_BATCH, n_tok*K)
        n_edge = src_mat.shape[1]
        self._add(src_mat.reshape(-1), torch.arange(lo, hi, device=dev).repeat_interleave(n_edge))
        if b == 0:
            m = int(N_BATCH * P0)
            if m > 0:
                pick = torch.randint(lo, hi, (N_BATCH, m), generator=g).to(dev)
                self._add((pick + self.KV).reshape(-1), torch.arange(lo, hi, device=dev).repeat_interleave(m))
        else:
            # 跨批：从更早的神经元里按 p0·exp(−dist/L) 采样（整批做，每神经元 m 个）
            prev = torch.arange(0, lo, device=dev)
            m_per = max(4, int(lo * P0))
            # 整批矩阵采样（原来每神经元单独 multinomial，4992 次是主要开销）
            _dist = (self.mark[lo:hi].unsqueeze(1) - self.mark[:lo].unsqueeze(0)).abs()
            _prob = (P0 * torch.exp(-_dist / L_AXON)).clamp(min=1e-9)
            _sel = torch.multinomial(_prob, min(m_per, lo), replacement=True)
            _src = (prev[_sel] + self.KV).reshape(-1)
            _dst = torch.arange(lo, hi, device=dev).repeat_interleave(_sel.shape[1])
            self._add(_src, _dst)
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

            drive = torch.zeros(B, MAXN, device=dev)
        # 第 0 批还要接收感受器（感受器不进上面那张表）
        # LIF
        live = self.alive.unsqueeze(0)
        vm = self.lam_n.unsqueeze(0) * self.vm + (1.0 - self.lam_n.unsqueeze(0)) * drive
        spk = ((vm >= self.theta.unsqueeze(0)) & live).float()
        self.vm = vm * (1.0 - spk)
        self.spk = spk
        self.act_cnt.add_(spk.mean(dim=0))

        # 输出接「末 OUT_BATCHES 批」：固定 N_OUT_TOTAL 列，实际活动放在最右端，
        # 未出生的神经元保持 0（对输出无贡献）。这样稀疏矩阵结构固定、不随生长变化。
        # 输出读全部神经元（不再只读末几批）
        last = spk
        logits = self.b.unsqueeze(0).expand(B, -1).clone()
        if DIAG_PREV:
            _prev = self.v[:, V:2 * V].clamp(0.0, 1.0)   # 慢通道 ≈ 前一 token
            logits = logits + (self.prev_w.unsqueeze(0) * _prev[:, self.prev_idx]).sum(dim=2)
        if self._out_sp is None and self.n_alive >= MAXN:
            self.build_output()
        if self._out_sp is not None:
            # l = last @ W，W 是 (N_last, V) 稀疏矩阵（结构固定，只换 values）
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
            if DIAG_PREV:
                _prev = self.v[:, V:2 * V].clamp(0.0, 1.0)
                _g = (e.unsqueeze(2) * _prev[:, self.prev_idx]).mean(dim=0) * (B ** 0.5)
                self.prev_w.sub_(LR_OUT * _g).clamp_(-CLIP, CLIP)

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
            if OJA and self.ns and (self.clock % 2 == 0):
                # Oja 规则：Δw = η · post · (pre − post·w)
                # post 用该神经元的 NMDA 平滑活动（不是单步发放，避免过噪）
                _ns = self.ns
                _si, _sj, _sw = self.si[:_ns], self.sj[:_ns], self.sw[:_ns]
                _den = _sj >= self.KV          # 只更新「内部连接」的突触
                if _den.any():
                    _si_d, _sj_d = _si[_den], _sj[_den]
                    # pre：源节点活动（按 batch 平均）；post：目标神经元活动
                    _pre = node[:, _si_d].mean(dim=0)
                    _post = node[:, _sj_d].mean(dim=0)
                    _dw = _post * (_pre - _post * self.sw[:_ns][_den])
                    _idx = torch.nonzero(_den).flatten()
                    self.sw[_idx] = torch.clamp(
                        self.sw[_idx] + OJA_ETA * _dw, -CLIP, CLIP)
            if PRED and self.ns and (self.clock % 2 == 0):
                # 预测误差 = 实际发放 − 预测驱动（drive 在 LIF 之前就是"预测的输入和"）
                # 只更新「内部连接」（感受器→神经元的权重是固定的输入编码）
                _ns = self.ns
                _den = self.sj[:_ns] >= self.KV
                if _den.any():
                    _idx = torch.nonzero(_den).flatten()
                    _si_d = self.si[:_ns][_den] - self.KV
                    _sj_d = self.sj[:_ns][_den]
                    _pre = node[:, _si_d].mean(dim=0)              # 上游活动
                    _post = self.spk[:, _sj_d].mean(dim=0)         # 实际下游发放
                    _predv = (self.sw[:_ns][_den] * _pre)          # 这条突触"预测"的贡献
                    _err = _post - _predv
                    self.sw[_idx] = torch.clamp(
                        self.sw[_idx] + PRED_ETA * _err * _pre / max(1, _pre.numel() // 4096),
                        -CLIP, CLIP)
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
        # 输出从【所有】神经元里选（不再只接末几批——那是我手填的"深度"，
        # 违反"不要固定深度/维度"；真实的输出层按化学亲和从全脑选输入）。
        N_OUT_TOTAL = MAXN
        self.n_out = max(16, min(int(P0 * self.n_alive), self.n_alive))
        gn = torch.Generator().manual_seed(SEED * 10000 + 77)
        # 化学亲和：输出选「活跃度高」的神经元（成熟的），而不是随机全部。
        # 依据：早期神经元训练充分、晚期不成熟；输出层在发育中只与"有功能的"输入形成连接。
        _use = self.act_cnt[:self.n_alive]
        _top = torch.argsort(-_use)[:max(1, self.n_out)]
        self.out_n[:, :self.n_out] = _top[
            torch.randint(0, _top.numel(), (V, self.n_out), generator=gn).to(dev)]
        self.out_w[:, :self.n_out] = (torch.randn(V, self.n_out, generator=gn) * 0.01).to(dev)
        _idx_r = torch.arange(V, device=dev).repeat_interleave(self.n_out)
        _idx_c = self.out_n[:, :self.n_out].flatten().to(dev)
        _w = self.out_w[:, :self.n_out].flatten().contiguous()
        self._out_sp = torch.sparse_coo_tensor(
            torch.stack([_idx_c, _idx_r]), _w, (N_OUT_TOTAL, V), device=dev).coalesce()
        # 关键：必须用 _values() 的视图（不是 clone），否则 LMS 更新的是副本、前向用的是原件
        self.out_w_flat = self._out_sp._values()
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
gn = torch.Generator().manual_seed(SEED * 10000 + 77)
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

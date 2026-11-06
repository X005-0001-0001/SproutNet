"""SproutNet grow 2.0.0 —— 让每个神经元"认一个词"

为什么从 1.0 改成 2.0
-------------------
1.0 的每个神经元预测一个**码位**。天花板测试（`_ceiling.py`）量出来：
  · unigram            2552
  · bigram（真实上限）   900
  · 岭回归 嵌入→嵌入+原型读出  7125   ← 预测条件均值必然丢掉多峰结构
  · 岭回归 稀疏码→嵌入        2300
1.0 的位级 NLL 一直贴着边际熵，根因是**每个单元看不见当前词**：
fi=96 的随机子集在 1.2% 密度的码上只有约 1.2 个活跃输入，
最多把 8192 个词分成 ~80 类。它不是学不会 bigram，是看不见 bigram。

2.0 的三处改动
-------------
① 输入改成**稠密向量**（PPMI-SVD 嵌入，或全码），每个单元看**全部维度**，
   不再有"只看到 1/40 的信息"这个瓶颈。
② 输出改成**每个单元认一个词**：单元 i 的局部目标就是"下一个词是不是 v_i"。
   这是 one-vs-all，而 one-vs-all **天然是局部的**——
   每个单元只需要自己那一位的 0/1 误差，不需要任何跨单元回传。
③ 码位的组平均换成词上的组平均：p_v = 该词组内所有单元发放概率的平均。
   它本身就是一个分布（因为 Σ_v E[p_v] = 1），归一化只是一次测量。

保留下来的（用户要的都在）
------------------------
· 从 1 个神经元长起、S 形、有生有死、y0<y
· 无 BP：全程 no_grad，没有 backward
· 无线性层：没有共享的 h→logits 投影；每个单元有自己的输入权重与自己的误差
· 阈值单元 + 发放温度；E/I 由源单元符号决定（Dale）
· 每单元 L1 预算；θ 只做边际校准
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch as T

T.set_grad_enabled(False)

ART = "_grow_art"


def log(*a):
    print("[grow2]", *a, flush=True)


# ---------------------------------------------------------------- 数据

class Corpus:
    def __init__(self, art=ART, in_mode="emb", ctx_k=0, ctx_dim=64, device="cuda"):
        self.dev = T.device(device)
        with open(os.path.join(art, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.tokens_np = np.load(os.path.join(art, "tokens.npy")).astype(np.int64)
        self.n = int(self.tokens_np.size)
        self.in_mode = in_mode
        if in_mode == "emb":
            e = np.load(os.path.join(art, "emb.npy")).astype(np.float32)

            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            # 不平移！之前这里做了 x=(e+1)*0.5 移到 [0,1]（乘性更新时代留下的），
            # 实测这个偏移值 3.3 倍 PPL：同一模型、同一数据、同一优化器，
            # shift=0 → 638，shift=1 → 2092。
            # 原因：输入全非负且均值 0.5 时，各类梯度共享同一个"均值方向"，
            # 类别之间的区分被淹没；中心化输入的梯度才互相解耦。
            x = e
        else:
            c = np.load(os.path.join(art, "code.npy")).astype(np.float32)
            x = c
        self.x = T.from_numpy(x).to(self.dev)          # (V, D) 每个 token 的输入向量
        self.V, self.D0 = self.x.shape
        # 上下文窗口：把前 k 个词的嵌入**用固定稀疏随机投影**压到 ctx_dim 维，再拼到输入上。
        # 为什么不能直接拼接：拼接让每个类多 (k)×256 个参数（R093 实测 ctx=1,2,4,8 → 2274/2670/3733/10239，
        # 单调变差）。而投影是共享的、不学习的，每类只多 ctx_dim 个参数（这里 64，即 +25%）。
        # 固定稀疏投影不是"线性层"：它属于输入编码（和 PPMI-SVD 同级），没有可学参数。
        self.ctx_k = int(ctx_k)
        self.ctx_dim = int(ctx_dim)
        if self.ctx_k > 0:
            g = T.Generator().manual_seed(20261101)
            P_ = T.zeros(self.ctx_k * self.D0, self.ctx_dim)
            nnz = 8
            for j in range(self.ctx_dim):
                idx = T.randperm(self.ctx_k * self.D0, generator=g)[:nnz]
                P_[idx, j] = T.randn(nnz, generator=g) / nnz ** 0.5
            self.proj = P_.to(self.dev)
            self.D = self.D0 + self.ctx_dim
        else:
            self.proj = None
            self.D = self.D0
        self.sigma = float(self.meta.get("sigma", 1.0))
        cnt = np.bincount(self.tokens_np[:2_000_000], minlength=self.V).astype(np.float64) + 1.0
        self.freq = T.from_numpy((cnt / cnt.sum()).astype(np.float32)).to(self.dev)
        self.logfreq = T.log(self.freq.clamp(min=1e-12))

    def tok_at(self, idx, shift=0):
        i = idx.detach().cpu().numpy() if T.is_tensor(idx) else np.asarray(idx)
        return T.from_numpy(self.tokens_np[i + shift]).to(self.dev)

    def inputs(self, idx):
        cur = self.x[self.tok_at(idx)]
        if self.ctx_k <= 0:
            return cur
        win = T.cat([self.x[self.tok_at(idx, -j)] for j in range(1, self.ctx_k + 1)], dim=1)
        return T.cat([cur, win @ self.proj], dim=1)

    def unigram_nll(self):
        return float(-self.logfreq[self.tok_at(T.arange(200000, device=self.dev))].mean())


# ---------------------------------------------------------------- 大脑

class Brain:
    def __init__(self, corpus, cap, fr=64, frac_inh=0.2, d0=10.0, geo=True, inh=True,
                 budget=4.0, gamma=0.5, homeo=1.0, temp=1.0, lr_theta=1.0, temp_abs=1.0,
                 norm="none", wd=0.0, wd_in=0.0, wd_rec=0.0, beta2=0.999, gain_jitter=1.0, theta_init="zero", w_init_sd=0.01, tau_fixed=0.0, tau_s_mult=4.5,
                 spk_target=0.05, spk_eta=0.02, lr_gain=0.01, wta_block=0, wta_k=8, delay_steps=8,
                 lif=1, tau_m_lo=2.0, tau_m_hi=30.0,
                 syn_rate=0.0, prune_rate=0.0, grow2_frac=0.85,
                 rec_norm=1, gain_rec_init=0.5,
                 slow_state=0, slow_lambda=0.999, slow_mut=1.0, init_degree=1.0,
                 slow_rank=0,
                 seed=0, device="cuda"):
        self.dev = T.device(device)
        self.V = corpus.V
        self.D = corpus.D
        self.fr = fr
        self.cap = cap
        self.geo = geo
        self.use_inh = inh
        self.frac_inh = frac_inh
        self.d0 = d0
        self.budget = budget
        self.gamma = gamma
        self.homeo = homeo
        self.spk_target = spk_target
        self.spk_eta = spk_eta
        self.fr = fr
        self.lr_gain = lr_gain
        self.wta_block = wta_block
        self.wta_k = wta_k
        self.use_lif = int(lif)
        self.tau_m_lo = float(tau_m_lo)
        self.tau_m_hi = float(tau_m_hi)
        self.syn_rate = float(syn_rate)
        self.prune_rate = float(prune_rate)
        self.grow2_frac = float(grow2_frac)
        self.rec_norm = int(rec_norm)
        self.gain_rec_init = float(gain_rec_init)
        # ===== 慢状态（文献34 修改1）：信息存于更慢的状态，结构在它之上被重建 =====
        # 每单元一个 long-τ 适应度痕迹；τ 远大于突触(0.99/0.995≈100-200步)，这里 0.999≈1000步。
        # 关键：**绝不进前向打分**（历史上把上下文塞进 logits 全判负 R062/R071/R072/R097）。
        # 它只驱动「结构生长/剪枝/死亡」的优先级，即免疫系统『变异率与适应度反相关』。
        # slow_state=0 时本数组不参与任何计算，逐位复现 3.0.0。
        self.slow_state = int(slow_state)
        self.slow_lambda = float(slow_lambda)
        self.slow_mut = float(slow_mut)
        self.init_degree = float(init_degree)
        self.slow_rank = int(slow_rank)
        self.temp = temp
        self.g = T.Generator(device="cpu").manual_seed(seed + 7)
        self.rng = np.random.default_rng(seed + 13)

        z = lambda *s: T.zeros(*s, dtype=T.float32, device=self.dev)
        o = lambda *s: T.ones(*s, dtype=T.float32, device=self.dev)

        self.coord = z(cap, 2)
        self.born = T.zeros(cap, dtype=T.int64, device=self.dev)
        self.alive = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.is_inh = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.tok = T.full((cap,), -1, dtype=T.int64, device=self.dev)   # 这个单元"认"的词
        self.theta = z(cap)
        self.gain = o(cap)
        self.tau_f = o(cap)
        self.tau_s = o(cap)
        self.rate_tgt = z(cap)
        self.rate_ema = z(cap)
        self.mean_p = z(cap)
        self.acc_ema = z(cap)
        self.age = z(cap)
        self.slow = z(cap) + 0.5          # 长期适应度痕迹，初值中性
        self.u_mean = z(cap)
        self.u_scale = o(cap) * 0.02

        self.w_in = z(cap, self.D)
        # 每个突触自己的二阶矩：只用自己的梯度历史归一化自己的步长。
        # 这是**完全局部**的（不需要任何跨突触信息），生物学对应物是习惯化/突触缩放。
        # _local_only 实测：加上它把 PPL 从 2113 打到 1442；不加则对 8192 个
        # 频率差几个数量级的类用统一步长，稀有类永远学不动。
        self.v_in = o(cap, self.D) * 0.0
        self.v_rec = o(cap, fr) * 0.0
        self.v_th = o(cap) * 0.0
        # 发放阈值与 softmax 分数的 θ **必须分开**。
        # 之前两者共用一个 score：一旦分数为了 softmax 拉开到 ±20，
        # 发放率就饱和到 0.97，h 恒为 1，递归通路传的是常数偏移而不是上下文
        # （R092 实测：开递归 PPL 3500 vs 不开 1323）。
        # θ_spike 只做一件事：把发放率稳态到 spk_target（默认 5%）。
        self.rec_gain = o(cap) * float(gain_rec_init)   # 递归增益初值（可配）
        self.ur = None
        self.theta_spike = z(cap)
        self.spk_ema = o(cap) * 0.05
        # ================= 拓扑：带延迟的递归边 =================
        # 每条边 = (源神经元, 延迟 d 步, 权重)。d=1 表示"上一 token 的状态"。
        # 这是 SNN 相对 MLP/Transformer 的独特能力：**记忆长在边里**。
        # 之前三版的所有"递归无效"实验（R093/R094/R096）都只有 delay=1，
        # 等价于一步线性混合，跨步信息一步就衰减完了，所以只能是噪声。
        # 现在把 fr 个槽位分成 Delay 组，第 k 组固定延迟 k+1 步。
        self.Delay = max(1, int(delay_steps))
        self.fr_per = max(1, fr // max(1, self.Delay))
        self.rec_src = T.zeros(cap, self.Delay, self.fr_per, dtype=T.int64, device=self.dev)
        self.w_rec = z(cap, self.Delay, self.fr_per)
        self.rec_mask = T.zeros(cap, self.Delay, self.fr_per, dtype=T.bool, device=self.dev)
        self.n_edges = 0
        self.edge_born = 0
        self.edge_cut = 0
        self.v_rec = o(cap, self.Delay, self.fr_per) * 0.0   # 3D 二阶矩，覆盖上面的 2D
        self.src_sign = o(cap)
        # ── 真 LIF 状态 ──
        self.tau_m = o(cap) * 5.0            # 膜时间常数（tick = token），多尺度 2~30
        self.tau_decay = o(cap) * 0.8        # exp(-1/tau_m)，每步泄漏
        self.ref_len = 1                     # 不应期（tick）
        self.v_scale = o(cap) * 0.02         # 膜电位尺度 EMA（给 theta_spike 稳态定标）
        self.spk_ptr = 0                     # 环形缓冲写指针
        self._ptr_before = 0                 # 本次 forward 开始时的指针（learn 用）
        self.spike_buf = None                # (Delay, B, n) 最近 Delay 步的脉冲
        self.v_mem = None
        self.ref = None
        self.spike = None

        self.f_state = None
        self.s_state = None
        self.h = None
        self.h_prev = None
        self.score_v = None
        self.score = None
        self.pr = None

        self.N = 0
        self.n_slots = 0
        self.birth_counter = 0
        self.events = 0
        self.born_total = 0
        self.dead_total = 0
        self.alive_idx = T.zeros(0, dtype=T.int64, device=self.dev)
        self.tok_cnt = z(self.V)
        self.tok_cnt_raw = z(self.V)
        self.Tglob = float(temp_abs)
        self.lr_theta = lr_theta
        self.temp_abs = float(temp_abs)
        self.norm = norm
        self.gain_jitter = gain_jitter
        self.theta_init = theta_init
        self.w_init_sd = w_init_sd
        self.tau_fixed = tau_fixed
        self.tau_s_mult = tau_s_mult
        self.wd = wd
        self.wd_in = wd_in
        self.wd_rec = wd_rec
        self.beta2 = beta2

    # -------------------------------------------------- 出生 / 死亡

    def _birth(self, n_new, corpus):
        if self.n_slots >= self.cap * 0.98:
            self.compact()
        n_new = int(min(n_new, self.cap - self.n_slots))
        if n_new <= 0:
            return 0
        err, cnt = self.tok_err, self.tok_cnt_raw
        # 误差高、人少的词优先（与 1.0 的生长规则同一条）
        score = (err + 1e-3) / T.sqrt(cnt + 1.0)
        score = score.masked_fill(cnt <= 0, score.max() + 1.0)
        base = T.topk(score, min(n_new, self.V)).indices.cpu().numpy()
        reps = int(math.ceil(n_new / max(1, base.size)))
        targets = np.tile(base, reps)[:n_new]

        self.coord_np = self.coord.detach().cpu().numpy()
        for t in range(n_new):
            col = self.n_slots
            self.n_slots += 1
            v = int(targets[t])
            parent = self.coord_np[col - 1] if col > 0 else np.zeros(2, np.float32)
            c = parent + self.rng.standard_normal(2).astype(np.float32) / math.sqrt(self.cap)
            self.coord[col] = T.from_numpy(c).to(self.dev)
            self.born[col] = self.birth_counter
            self.birth_counter += 1
            self.is_inh[col] = bool(self.use_inh and (self.rng.random() < self.frac_inh))
            self.src_sign[col] = -1.0 if bool(self.is_inh[col]) else 1.0

            tf = (float(self.tau_fixed) if self.tau_fixed > 0
                  else float(np.exp(self.rng.uniform(math.log(1.5), math.log(6.0)))))
            self.tau_f[col] = tf
            self.tau_s[col] = tf * self.tau_s_mult
            # 真 LIF 的膜时间常数：对数均匀 2~30 tick，天然形成多尺度记忆
            tm = float(np.exp(self.rng.uniform(math.log(self.tau_m_lo), math.log(self.tau_m_hi))))
            self.tau_m[col] = tm
            self.tau_decay[col] = math.exp(-1.0 / tm)
            # θ 必须初始化在**驱动电流的实际尺度**上。L2 半径 4、256 维时 u 约 32，
            # 若 θ 从 2 起、每步只挪 1e-4，要 32 万步才追上，期间 softmax 一直饱和。
            # E[u] = mean(x)·Σw，x∈[0,1] 取 0.5，所以 0.5·Σw 就是合适的起点。
            if self.theta_init == "drive":
                self.theta[col] = (float(self.theta[:col][self.alive[:col]].median())
                                   if col > 0 else 0.5 * float(self.w_in[col].sum()))
            else:
                self.theta[col] = 0.0
            self.theta_spike[col] = (float(self.theta_spike[:col][self.alive[:col]].median())
                                     if col > 0 else 0.0)
            self.spk_ema[col] = float(self.spk_target)
            self.gain[col] = float(np.exp(self.rng.normal(0.0, 0.35) * self.gain_jitter))
            f = float(corpus.freq[v])
            self.rate_tgt[col] = min(0.3, max(1e-4, f))
            self.rate_ema[col] = 0.0
            self.mean_p[col] = f
            same = (self.tok[:col] == v) & self.alive[:col]
            self.acc_ema[col] = float(self.acc_ema[:col][same].mean()) if bool(same.any()) else 0.9
            self.u_mean[col] = 0.0
            self.u_scale[col] = float(self.u_scale[:col][self.alive[:col]].median()) if col > 0 else 0.02
            self.age[col] = 0.0

            # 输入：**全部** D 维（不再是随机子集）—— 这是 2.0 与 1.0 最关键的区别
            wv = self.rng.standard_normal(self.D).astype(np.float32) * self.w_init_sd
            self.w_in[col] = T.from_numpy(wv).to(self.dev)

            # 递归边：只接更老的单元，几何衰减；并按轮转铺到不同延迟组。
            # 新单元一出生就带上跨 1..Delay 步的记忆通道 —— 这是"拓扑"的起点。
            m = col
            if m > 0 and self.fr > 0:
                if self.geo:
                    d = np.linalg.norm(self.coord_np[:m] - c, axis=1)
                    pp = np.exp(-d / self.d0)
                    pp = pp / pp.sum() if pp.sum() > 0 else np.ones(m) / m
                    kk = min(self.fr, int((pp > 0).sum()))
                    src = self.rng.choice(m, size=kk, replace=False, p=pp) if kk > 0 else np.zeros(0, np.int64)
                else:
                    src = self.rng.choice(m, size=min(self.fr, m), replace=False)
                # init_degree：出生时只填一部分槽位，把剩下的留给结构期按慢状态重建。
                # 文献03「长出结构 = 总数守恒下重新分配」(Ko 2011)：不该出生即填满。
                # 默认 1.0（复现 3.0.0）；慢状态实验设 <1 才有空槽可分。
                keep = max(1, int(len(src) * self.init_degree)) if len(src) else 0
                src = src[:keep]
                for ii, su in enumerate(src):
                    k = ii % self.Delay                     # 延迟组轮转
                    j = (ii // self.Delay) % self.fr_per   # 组内槽位
                    self.rec_src[col, k, j] = int(su)
                    self.w_rec[col, k, j] = abs(float(self.rng.standard_normal())) * 0.03 + 1e-3
                    self.rec_mask[col, k, j] = True
                    self.n_edges += 1

            self.alive[col] = True
            self.tok[col] = v

        ids = T.arange(self.n_slots - n_new, self.n_slots, device=self.dev)
        # 只在显式要求 l2 时才归一。之前这里无条件调用，
        # 导致 --norm none 时新生儿权重仍被拉到 L2=4（初始本该 0.16），起点差 25 倍。
        if self.norm == "l2":
            self._normalize_rows(ids)
        self.N += n_new
        self.born_total += n_new
        self._refresh()
        return n_new

    def _normalize_rows(self, ids):
        """每单元权重做 L2 归一到一个固定半径。

        L1 归一会把 256 维上的权重摊平，导致"跨样本的电流差"只有 1e-2 量级，
        而 softmax 需要的是 1e1 量级的分数差 —— 这就是 PPL 一直贴着均匀分布的根因。
        半径 = budget，温度固定为 temp_abs，两者都不随输入统计漂移。
        """
        if ids.numel() == 0:
            return
        s = (self.w_in[ids].pow(2).sum(1) + self.w_rec[ids].pow(2).sum(1)).sqrt() + 1e-9
        sc = (self.budget / s).unsqueeze(1)
        self.w_in[ids] *= sc
        self.w_rec[ids] *= sc

    def _refresh(self):
        alive = self.alive[:self.n_slots]
        self.alive_idx = alive.nonzero(as_tuple=True)[0]
        self.Tglob = float(self.temp_abs)
        cnt = T.zeros(self.V, device=self.dev)
        m = alive & (self.tok[:self.n_slots] >= 0)
        if bool(m.any()):
            t = self.tok[:self.n_slots][m]
            cnt.index_add_(0, t, T.ones(t.numel(), device=self.dev))
        self.tok_cnt_raw = cnt                       # 出生打分用：0 表示这个词还没人认
        self.tok_cnt = cnt.clamp(min=1.0)            # 输出平均用：避免除零

    def _death(self, n_die, grace):
        elig = self.alive[:self.n_slots] & (self.age[:self.n_slots] > grace)
        idx = elig.nonzero(as_tuple=True)[0]
        if idx.numel() <= 4:
            return 0
        n_die = int(min(n_die, idx.numel() - 2))
        if n_die <= 0:
            return 0
        acc = self.acc_ema[idx]
        grp = self.tok[idx]
        gsum = T.zeros(self.V, device=self.dev).index_add_(0, grp, acc)
        gcnt = T.zeros(self.V, device=self.dev).index_add_(0, grp, T.ones_like(acc))
        rel = acc - (gsum / gcnt.clamp(min=1))[grp]
        if self.slow_state:
            # 长期适应度低的单元，即使当下 acc 尚可，也应优先淘汰
            # （结构在慢状态之上重建：没用的逐渐死掉）。
            rel = rel - self.slow_mut * 0.5 * (1.0 - self.slow[idx])
        guard = self.tok_cnt[grp] > 1.5
        rel = rel.masked_fill(~guard, 1e9)
        order = rel.argsort()
        order = order[rel[order] < 1e8]
        n_die = min(n_die, int(order.numel()))
        if n_die <= 0:
            return 0
        dead = idx[order[:n_die]]
        self.alive[dead] = False
        self.tok[dead] = -1
        self.w_in[dead] = 0
        cut = int(self.rec_mask[dead].sum())
        self.n_edges -= cut
        self.w_rec[dead] = 0
        self.rec_mask[dead] = False
        self.rec_src[dead] = 0
        self.N -= int(dead.numel())
        self.dead_total += int(dead.numel())
        self._refresh()
        return int(dead.numel())

    def compact(self):
        n = self.n_slots
        alive_full = self.alive.clone()
        alive = self.alive[:n].nonzero(as_tuple=True)[0]
        m = int(alive.numel())
        if m == n:
            return 0
        remap = T.zeros(self.cap, dtype=T.int64, device=self.dev)
        remap[alive] = T.arange(m, device=self.dev)
        for k in ("coord", "born", "alive", "is_inh", "tok", "theta", "gain", "tau_f", "tau_s",
                  "rate_tgt", "rate_ema", "mean_p", "acc_ema", "age", "src_sign",
                  "u_mean", "u_scale", "w_in", "rec_src", "w_rec", "rec_mask",
                  "v_in", "v_rec", "v_th", "theta_spike", "spk_ema", "rec_gain",
                  "tau_m", "tau_decay", "v_scale", "slow"):
            getattr(self, k)[:m] = getattr(self, k)[:n][alive]
        rs = self.rec_src[:m].clone()
        bad = ~alive_full[rs]                     # 指向已死单元的边
        self.rec_src[:m] = remap[rs]
        cut = self.rec_mask[:m] & bad
        self.n_edges -= int(cut.sum())
        self.rec_mask[:m].masked_fill_(cut, False)
        self.w_rec[:m].masked_fill_(bad, 0.0)
        self.alive[m:n] = False
        self.is_inh[m:n] = False
        self.tok[m:n] = -1
        self.n_slots = m
        self._refresh()
        if self.h is not None and self.h.shape[1] != m:
            self.h = self.h[:, :m]
            self.h_prev = self.h_prev[:, :m]
            self.f_state = self.f_state[:, :m]
            self.s_state = self.s_state[:, :m]
            if self.v_mem is not None:
                self.v_mem = self.v_mem[:, :m]
                self.ref = self.ref[:, :m]
                self.spike = self.spike[:, :m]
                self.spike_buf = T.zeros(self.Delay + 1, self.h.shape[0], m, device=self.dev)
                self.spk_ptr = 0
        return n - m

    # -------------------------------------------------- 前向

    # -------------------------------------------------- 结构生长（拓扑会变）
    def synaptogenesis(self, n_new, corpus=None):
        """误差驱动的突触发生：直接在**空槽集合**里按"所认词的残余误差"加权采样，
        源按当前发放活跃度加权（活跃优先连接 = Hebb 的"一起放电连起来"）。
        命中率 100%（旧版随机撞 (k,j) 再过滤，边快满时几乎撞不到空槽，
        冒烟实测 edge_born=48 / edge_cut=6165，结构生长形同虚设）。
        于是拓扑随训练持续重构，而不是出生那刻定死。"""
        n = self.n_slots
        if n <= 1 or n_new <= 0 or self.fr <= 0:
            return 0
        alive = self.alive[:n]
        free = (~self.rec_mask[:n]) & alive.view(n, 1, 1)
        idx = free.reshape(-1).nonzero(as_tuple=True)[0]      # 全局空槽（展平索引）
        if idx.numel() == 0:
            return 0
        tokerr = getattr(self, "tok_err", None)
        units = idx // (self.Delay * self.fr_per)
        if tokerr is not None:
            we = tokerr[self.tok[units].clamp(min=0)]
            w = we - we.min() + 1e-3
        else:
            w = T.ones(idx.numel(), device=self.dev)
        if self.slow_state:
            if self.slow_rank:
                # 排序主导（线C 正解）：免疫 CLONALG 第一步就是「按亲和力排序」。
                # slow 绝对值 97.5% 挤在 <0.02，乘性/绝对值加权都无区分度（_lineC_sampling
                # 实测入选率 0.48~0.52≈随机）；只有按 rank 富集弱者才真分化(0.73~0.99)。
                # rank∈[0,1]：slow 最小(适应度最低)→rank≈0→权重最大。
                su = self.slow[units]
                rank = su.argsort().argsort().float() / max(1, su.numel() - 1)
                w = T.exp(-self.slow_mut * rank)
            else:
                # R098 旧行为（乘性微调，已被证明被 tok_err 淹没），保留作对照/复现
                w = w * (1.0 + self.slow_mut * (1.0 - self.slow[units])).clamp(min=0.1)
        ns = min(int(n_new), int(idx.numel()))
        sel = T.multinomial(w.clamp(min=1e-9), ns, replacement=False)
        chosen = idx[sel]
        u = chosen // (self.Delay * self.fr_per)
        rem = chosen % (self.Delay * self.fr_per)
        k = rem // self.fr_per
        j = rem % self.fr_per
        pool = alive.nonzero(as_tuple=True)[0]
        rate = self.rate_ema[pool].clamp(min=1e-4)
        rate = rate / rate.sum().clamp(min=1e-9)
        src = pool[T.multinomial(rate, ns, replacement=True)]
        self.rec_src[u, k, j] = src
        self.w_rec[u, k, j] = T.rand(ns, device=self.dev) * 0.03 + 1e-3
        self.rec_mask[u, k, j] = True
        self.n_edges += ns
        self.edge_born += ns
        return int(ns)

    def prune_edges(self, frac=0.05):
        """剪掉已衰减到近乎无用的边（|w| < 0.1 x 初始权重）。
        用绝对弱阈值而非"最弱 frac 比例"：旧版按比例剪会把刚长出、
        还在 0.03 量级的新边优先剪掉（冒烟里结构净减少），等于永远长不起来。
        生物学对应：未巩固的突触若长期不增强才被修剪。"""
        n = self.n_slots
        m = self.rec_mask[:n]
        if not bool(m.any()):
            return 0
        aw = self.w_rec[:n].abs()
        cut = m & (aw < 0.003)
        self.rec_mask[:n].masked_fill_(cut, False)
        c = int(cut.sum())
        self.n_edges -= c
        self.edge_cut += c
        return c

    def topology(self):
        """可测量的拓扑指标 —— 用来回答"结构到底长没长出来"。"""
        n = self.n_slots
        if n == 0:
            return {"edges": 0, "avg_indeg": 0.0, "per_delay": [0] * self.Delay}
        alive = self.alive[:n]
        m = self.rec_mask[:n]
        indeg = m.sum(dim=(1, 2)).float()
        per = m.sum(dim=(0, 2)).float()
        return {
            "edges": int(m[alive].sum()),
            "edges_total": self.n_edges,
            "avg_indeg": round(float(indeg[alive].mean()), 2) if bool(alive.any()) else 0.0,
            "max_indeg": int(indeg.max()),
            "per_delay": [int(x) for x in per],
            "edge_born": self.edge_born,
            "edge_cut": self.edge_cut,
            "slow_mean": round(float(self.slow[:n][alive].mean()), 4) if bool(alive.any()) else 0.0,
            "slow_std": round(float(self.slow[:n][alive].std()), 4) if bool(alive.any()) else 0.0,
            # 结构分化判据：把存活单元按 slow 分高低两半，比较平均入度。
            # 「适应度低的多变异」若成立，低 slow 半区入度应更高（拿到更多新边）。
            "indeg_lowslow": round(float(indeg[alive][self.slow[:n][alive] <=
                                          self.slow[:n][alive].median()].mean()), 2)
                              if bool(alive.any()) else 0.0,
            "indeg_highslow": round(float(indeg[alive][self.slow[:n][alive] >
                                           self.slow[:n][alive].median()].mean()), 2)
                               if bool(alive.any()) else 0.0,
            "free_slots": int(((~m) & alive.view(n, 1, 1)).sum()),
        }

    def reset_state(self, B):
        n = self.n_slots
        self.f_state = T.zeros(B, n, device=self.dev)
        self.s_state = T.zeros(B, n, device=self.dev)
        self.h = T.zeros(B, n, device=self.dev)
        self.h_prev = T.zeros(B, n, device=self.dev)
        # 真 LIF：膜电位、不应期、最近 Delay 步的脉冲
        self.v_mem = T.zeros(B, n, device=self.dev)
        self.ref = T.zeros(B, n, device=self.dev)
        self.spike = T.zeros(B, n, device=self.dev)
        self.spike_buf = T.zeros(self.Delay + 1, B, n, device=self.dev)
        self.spk_ptr = 0
        self._ptr_before = 0

    def forward(self, x):
        n = self.n_slots
        B = x.shape[0]
        if self.h is None or self.h.shape != (B, n):
            self.reset_state(B)

        # 输入电流：稠密 (B, D) @ (D, N)
        u_in = x @ self.w_in[:n].t()          # (B, n)

        # ================= 记忆 = 带延迟的递归脉冲 =================
        # spike_buf 有 (Delay+1) 格环形缓冲；写指针 spk_ptr 指向"本步写入格"。
        # 第 k 组边（延迟 k+1 步）读 spike_buf[(ptr-1-k) % (Delay+1)]。
        # 写入格 ptr 恰好不在任何读取格里（读集合 D 个、总 D+1 个），
        # 所以先读后写不会踩踏，learn 阶段也能按同一公式重算出正确的历史脉冲。
        # 这跟之前三版"只有 delay=1"的递归有本质区别：信息能跨 1..Delay 步传，
        # 而不是一步就衰减光 —— 这正是上下文能成立的物理基础。
        self._ptr_before = self.spk_ptr
        u_rec = T.zeros(B, n, device=self.dev)
        if self.fr > 0 and self.n_edges > 0:
            p = self.spk_ptr
            for k in range(self.Delay):
                slot = (p - 1 - k) % (self.Delay + 1)
                buf_k = self.spike_buf[slot]                       # (B, n) 第 k+1 步前的脉冲
                ids = self.rec_src[:n, k, :]                        # (n, fr_per)
                w = (self.w_rec[:n, k, :] * self.src_sign[ids] *
                     self.rec_mask[:n, k, :].float())               # (n, fr_per)
                g = buf_k[:, ids.reshape(-1)].reshape(B, n, self.fr_per)
                u_rec += (g * w.unsqueeze(0)).sum(-1)
            if self.rec_norm:
                # 连续归一化：把递归电流拉到零中心单位方差。
                # 但这会抹掉"几个脉冲同步抵达"的离散语义，等于把 polychrony 糊成连续噪声
                # （R097 对照：delay=1/8/16 一律 +9% 变差，且 delay 越长越差）。
                ur = (u_rec - u_rec.mean(0, keepdim=True)) / u_rec.std(0, keepdim=True).clamp(min=1e-6)
                self.ur = ur
                u = u_in + self.rec_gain[:n].unsqueeze(0) * ur
            else:
                # 不去归一化：原始脉冲电流直接叠加，靠 LIF 阈值做 coincidence detection。
                # Izhikevich polychrony：异步发放的脉冲经补偿延迟在同一 tick 抵达，
                # 电流相加越阈才触发发放 —— 这才是"带延迟的边能形成时空记忆群"的机制。
                # rec_gain 需要放大（脉冲和量级小于 u_in），故初值可配。
                self.ur = u_rec
                u = u_in + self.rec_gain[:n].unsqueeze(0) * u_rec
        else:
            self.ur = None
            u = u_in

        ubar = u.mean(0)
        self.u_mean[:n] = 0.98 * self.u_mean[:n] + 0.02 * ubar
        self.u_scale[:n] = (0.98 * self.u_scale[:n] +
                            0.02 * (u - ubar).abs().mean(0)).clamp_(min=2e-3)

        # 模拟量 drive：给输出打分（连续值，能表达 Zipf 尾部概率；不是脉冲）
        tf = self.tau_f[:n].unsqueeze(0)
        ts = self.tau_s[:n].unsqueeze(0)
        self.f_state += (u - self.f_state) / tf
        self.s_state += (u - self.s_state) / ts
        drive = (self.f_state + self.gamma * self.s_state) * self.gain[:n].unsqueeze(0)
        score = ((drive - self.theta[:n].unsqueeze(0)) / self.Tglob).clamp_(-20.0, 20.0)

        # ================= 脉冲发放：真 LIF =================
        # 轴突传的是脉冲（记忆/时序），读出用连续率（表达幅度）——两条通路分离。
        # 这是仿生且必要的：二值脉冲直接做 softmax 表达不了 1e-4 的稀有词概率。
        if self.use_lif:
            v = self.v_mem * self.tau_decay[:n].unsqueeze(0) + u     # 泄漏积分
            v = v * (self.ref <= 0).float()                           # 不应期内不累积
            spike = (v > self.theta_spike[:n].unsqueeze(0)).float()
            v = T.where(spike > 0, T.zeros_like(v), v)                # 发放后 reset
            self.ref = (self.ref - 1).clamp(min=0)
            self.ref = T.where(spike > 0, T.full_like(self.ref, float(self.ref_len)), self.ref)
            self.v_mem = v
            with T.no_grad():
                self.v_scale[:n] = 0.98 * self.v_scale[:n] + 0.02 * v.abs().mean(0).clamp(min=1e-3)
            self.pr = spike                                            # 诊断兼容
        else:
            pr = T.sigmoid((drive - self.theta_spike[:n].unsqueeze(0)) /
                           (self.u_scale[:n].unsqueeze(0) + 1e-9))
            spike = (T.rand(pr.shape, device=self.dev) < pr).float()
            self.pr = pr

        # 局部抑制（PV 式 k-WTA）：让脉冲态离散、稀疏，才有"地址"
        if self.wta_block > 0:
            nb_ = n // self.wta_block
            if nb_ > 0:
                s_ = score[:, :nb_ * self.wta_block].reshape(B, nb_, self.wta_block)
                thr = s_.topk(self.wta_k, dim=2).values[:, :, -1:]
                hh = (s_ >= thr).float().reshape(B, nb_ * self.wta_block)
                tail = n - nb_ * self.wta_block
                spike = T.cat([hh, T.zeros(B, tail, device=self.dev)], dim=1) if tail else hh
        alivef = self.alive[:n].float().unsqueeze(0)
        spike = spike * alivef
        self.spike = spike
        self.h_prev = self.h
        self.h = spike
        # 写环形缓冲（与所有读取格不冲突）
        self.spike_buf[self.spk_ptr] = spike.detach()
        self.spk_ptr = (self.spk_ptr + 1) % (self.Delay + 1)
        self.score = score * alivef

        # 输出：同一个词的单元取平均（固定 1/n，无参数、不可学），再 softmax 归一
        sc = T.zeros(B, self.V, device=self.dev)
        mask = self.alive[:n] & (self.tok[:n] >= 0)
        if bool(mask.any()):
            sel = mask.nonzero(as_tuple=True)[0]
            sc.index_add_(1, self.tok[sel], score[:, sel])
            sc = sc / self.tok_cnt.unsqueeze(0)
            # 中心化：让"已经有神经元认的词"的分数均值为 0。
            # 否则空词（还没有神经元）的分数恒为 0，而真实词的分数可能是负的（θ 偏大时），
            # softmax 会把概率全给没人认的词 —— R093 冒烟实测：score_max=0、score_min=-20、
            # 于是 PPL 直接掉到均匀分布的水平。
            cov = (self.tok_cnt_raw > 0).float().unsqueeze(0)
            ncov = cov.sum().clamp(min=1.0)
            sc = sc - (sc * cov).sum(1, keepdim=True) / ncov
        self.score_v = sc
        return sc

    # -------------------------------------------------- 局部学习

    def learn(self, tgt_tok, lr=0.05, plastic_b=4):
        n = self.n_slots
        B = self.h.shape[0]
        if B == 0 or n == 0:
            return
        nb = B if plastic_b <= 0 else min(plastic_b, B)
        tok = self.tok[:n]
        valid = (self.alive[:n] & (tok >= 0))
        # 每个单元的误差 = 它认的那个词在 softmax 下的概率 - 它是不是真值。
        # 这仍然是**逐单元局部**量：单元只看自己那一组归一化后的概率和自己的 0/1 标签，
        # 不需要任何从输出往回传的梯度。
        pv = T.softmax(self.score_v[:nb], dim=1)                  # (nb, V)
        p_own = pv.gather(1, tok.clamp(min=0).unsqueeze(0).expand(nb, n))   # (nb, n)
        hit = (tgt_tok[:nb].unsqueeze(1) == tok.unsqueeze(0)).float()
        e = (hit - p_own) * valid.float().unsqueeze(0)
        with T.no_grad():
            acc = (1.0 - (hit - p_own).abs()).mean(0)
            self.acc_ema[:n] = 0.99 * self.acc_ema[:n] + 0.01 * acc
            self.rate_ema[:n] = 0.995 * self.rate_ema[:n] + 0.005 * self.h[:nb].mean(0)
            if self.slow_state:
                # 慢适应度 = 「被需求度」hit.mean(0) 的 long-τ EMA（纯局部）。
                # R098 首版用 acc=(1-|hit-p_own|)，对负类单元恒≈1（p_own≈1e-4），
                # 结果 slow 全爬到 0.99、std 0.028，毫无区分度 → mut 调了等于没调。
                # 改用 hit：该词实际出现频率，Zipf 分布天然分化常见/稀有单元。
                # λ=0.999 → τ≈1000 步，远大于突触(0.99≈100步)，这就是「更慢的那一份」。
                demand = hit.mean(0)
                self.slow[:n] = self.slow_lambda * self.slow[:n] + (1 - self.slow_lambda) * demand

        # 输入突触：可加 delta，按分数的尺度归一。
        # 分数的定义是 s=(u-θ)/u_scale，所以 ds/dw = x/u_scale —— 这是这条局部规则的正确梯度尺度。
        # 之前用乘性 exp(lr·e·x)，而 softmax 下负类的 e 只有 1/V≈1e-4，指数等于 0，等于没学。
        pre = self.x_t[:nb]                                   # (nb, D)
        es = e / self.Tglob
        g = (es.t() @ pre) / nb                                # (n, D)
        v = self.v_in[:n]
        v.mul_(self.beta2).add_(g * g, alpha=1 - self.beta2)
        self.w_in[:n] += lr * g / (v.sqrt() + 1e-8)
        if self.wd_in > 0:
            self.w_in[:n] *= (1.0 - self.wd_in)

        # 递归突触：按延迟分组，每条边只看"自己那一格历史脉冲 x 自己这一份误差"。
        # 完全局部：一个突触不需要知道别的突触、别的时刻发生了什么。
        if self.fr > 0 and self.n_edges > 0:
            p = self._ptr_before
            es = e / self.Tglob
            for k in range(self.Delay):
                slot = (p - 1 - k) % (self.Delay + 1)
                buf_k = self.spike_buf[slot]                          # (B, n)
                ids = self.rec_src[:n, k, :]                           # (n, fr_per)
                pre = buf_k[:nb, ids.reshape(-1)].reshape(nb, n, self.fr_per)
                gg = T.einsum("bn,bnc->nc", es, pre) / nb             # (n, fr_per)
                vr = self.v_rec[:n, k, :]
                vr.mul_(self.beta2).add_(gg * gg, alpha=1 - self.beta2)
                upd = lr * gg / (vr.sqrt() + 1e-8) * self.rec_mask[:n, k, :].float()
                self.w_rec[:n, k, :] += upd
                if self.wd_rec > 0:
                    self.w_rec[:n, k, :] *= (1.0 - self.wd_rec)

        if self.norm == "l2":
            self._normalize_rows(self.alive_idx)

        # 递归增益：dL/dg = e · u_rec_norm，完全局部（只用自己那一路的递归电流）
        if self.fr > 0 and self.ur is not None and self.lr_gain > 0:
            self.rec_gain[:n] += self.lr_gain * (e * self.ur[:nb]).mean(0)

        # 发放阈值稳态：只用自己的发放率，目标是 spk_target。与误差无关。
        with T.no_grad():
            self.spk_ema[:n] = 0.995 * self.spk_ema[:n] + 0.005 * self.h[:nb].mean(0)
            # 对数域控制：发放率误差在对数空间才对称。
            # 线性域里"发太多"(+0.95) 和"发太少"(−0.05) 差 19 倍，
            # 于是一次冲过头就永久卡在 0 发放（R093 冒烟实测 fire=0.0）。
            _e = T.log(self.spk_ema[:n].clamp(min=1e-3)) - math.log(max(1e-6, self.spk_target))
            scl = self.v_scale[:n] if self.use_lif else self.u_scale[:n]
            self.theta_spike[:n] += self.spk_eta * scl * _e
            self.theta_spike[:n].clamp_(-1e4, 1e4)

        # 阈值 = 偏置项，同样用局部二阶矩归一（否则对稀有类步子太小）
        if self.homeo > 0:
            gt = es.mean(0)
            self.v_th[:n].mul_(self.beta2).add_(gt * gt, alpha=1 - self.beta2)
            self.theta[:n] -= self.lr_theta * gt / (self.v_th[:n].sqrt() + 1e-8)

    # -------------------------------------------------- 存取

    def snapshot(self):
        n = self.n_slots
        d = {"cap": self.cap, "n_slots": n, "N": self.N, "D": self.D, "V": self.V,
             "fr": self.fr, "geo": self.geo, "use_inh": self.use_inh, "frac_inh": self.frac_inh,
             "d0": self.d0, "budget": self.budget, "gamma": self.gamma, "homeo": self.homeo,
             "temp": self.temp, "norm": self.norm, "wd": self.wd, "beta2": self.beta2, "birth_counter": self.birth_counter, "events": self.events,
             "born_total": self.born_total, "dead_total": self.dead_total}
        for k in ("alive", "is_inh", "tok", "coord", "born", "theta", "gain", "tau_f", "tau_s",
                  "rate_tgt", "rate_ema", "mean_p", "acc_ema", "u_mean", "u_scale",
                  "w_in", "rec_src", "w_rec", "rec_mask", "v_in", "v_rec", "v_th",
                  "theta_spike", "spk_ema", "rec_gain", "tau_m", "tau_decay", "v_scale", "slow"):
            d[k] = getattr(self, k)[:n].cpu()
        d["n_edges"] = self.n_edges
        d["edge_born"] = self.edge_born
        d["edge_cut"] = self.edge_cut
        d["topology"] = self.topology()
        return d

    def load_state(self, d):
        """把存档灌回大脑。流程里"用死掉的脑子提问"就靠这一步。"""
        n = int(d["n_slots"])
        self.n_slots = n
        self.N = int(d["N"])
        self.birth_counter = int(d["birth_counter"])
        self.events = int(d["events"])
        self.born_total = int(d["born_total"])
        self.dead_total = int(d["dead_total"])
        for k in ("coord", "born", "alive", "is_inh", "tok", "theta", "gain", "tau_f", "tau_s",
                  "rate_tgt", "rate_ema", "mean_p", "acc_ema", "age", "src_sign",
                  "u_mean", "u_scale", "w_in", "rec_src", "w_rec", "rec_mask",
                  "v_in", "v_rec", "v_th", "theta_spike", "spk_ema", "rec_gain",
                  "tau_m", "tau_decay", "v_scale", "slow"):
            if k in d:
                getattr(self, k)[:n] = d[k].to(self.dev)
        self.n_edges = int(d.get("n_edges", 0))
        self.edge_born = int(d.get("edge_born", 0))
        self.edge_cut = int(d.get("edge_cut", 0))
        self._refresh()
        self.Tglob = float(self.temp_abs)

    @staticmethod
    def save(path, brain, extra=None):
        d = brain.snapshot()
        if extra:
            d.update(extra)
        T.save(d, path)


# ---------------------------------------------------------------- 评估

class _Keep:
    def __init__(self, b):
        self.b = b

    def __enter__(self):
        b = self.b
        self.s = None
        if b.h is not None:
            self.s = (b.h.clone(), b.h_prev.clone(), b.f_state.clone(), b.s_state.clone(),
                      b.v_mem.clone() if b.v_mem is not None else None,
                      b.ref.clone() if b.ref is not None else None,
                      b.spike.clone() if b.spike is not None else None,
                      b.spike_buf.clone() if b.spike_buf is not None else None,
                      b.spk_ptr, b._ptr_before)
        return b

    def __exit__(self, *a):
        if self.s is not None:
            b = self.b
            (b.h, b.h_prev, b.f_state, b.s_state, b.v_mem, b.ref, b.spike, b.spike_buf,
             b.spk_ptr, b._ptr_before) = self.s
        return False


def evaluate(brain, corpus, n_batches=50, batch=16, seed=0):
    with _Keep(brain):
        return _eval(brain, corpus, n_batches, batch, seed)


def _eval(brain, corpus, n_batches, batch, seed):
    brain.reset_state(batch)
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, corpus.n - 2, size=(n_batches, batch))
    nll = 0.0
    top1 = top5 = 0
    nsamp = 0
    ar = T.arange(batch, device=brain.dev)
    pmax = -1e9
    pmin = 1e9
    psum = psum2 = 0.0
    for bi in range(n_batches):
        idx = T.from_numpy(idxs[bi].astype(np.int64)).to(brain.dev)
        pred = brain.forward(corpus.inputs(idx))
        tgt = corpus.tok_at(idx, 1)
        p = T.softmax(pred, dim=1).clamp(min=1e-9)
        nll += float(-T.log(p)[ar, tgt].sum())
        top1 += int((p.argmax(1) == tgt).sum())
        top5 += int(p.topk(5, 1).indices.eq(tgt.unsqueeze(1)).any(1).sum())
        nsamp += batch
        pmax = max(pmax, float(pred.max()))
        pmin = min(pmin, float(pred.min()))
        psum += float(pred.sum())
        psum2 += float((pred ** 2).sum())
    nn = float(pred.numel() * n_batches)
    mean = psum / max(1.0, nn)
    std = max(0.0, psum2 / max(1.0, nn) - mean * mean) ** 0.5

    alive = brain.alive[:brain.n_slots]
    tk = brain.tok[:brain.n_slots]
    tc = T.bincount(tk[tk >= 0], minlength=brain.V).float()
    diag = {
        "classes_empty": int((tc == 0).sum()),
        "cnt_min": int(tc.min()), "cnt_max": int(tc.max()),
        "w_norm_mean": round(float(brain.w_in[:brain.n_slots][alive].norm(dim=1).mean()), 4),
        "theta_mean": round(float(brain.theta[:brain.n_slots][alive].mean()), 4),
        "theta_std": round(float(brain.theta[:brain.n_slots][alive].std()), 4),
        "score_max": round(pmax, 3), "score_min": round(pmin, 3),
        "score_mean": round(mean, 4), "score_std": round(std, 4),
        "fire": round(float(brain.h.float().mean()), 4),
        "rec_gain_abs": round(float(brain.rec_gain[:brain.n_slots][alive].abs().mean()), 4),
        "fire_h": round(float(brain.h.float().mean()), 4),
        "spk_tgt": round(float(brain.spk_target), 4),
    }
    print("[diag] " + "  ".join("%s=%s" % (k, v) for k, v in diag.items()), flush=True)
    return {"ppl": math.exp(nll / max(1, nsamp)), "top1": top1 / max(1, nsamp),
            "top5": top5 / max(1, nsamp), "batches": n_batches, "diag": diag}


# ---------------------------------------------------------------- 主流程

def run(args):
    dev = args.device
    T.set_num_threads(4)
    corpus = Corpus(args.art, args.in_mode, args.ctx_k, args.ctx_dim, dev)
    log("输入模式 %s：in_dim=%d（原 %d + 上下文 %d），窗口 %d，词表 %d，语料 %d token" % (
        args.in_mode, corpus.D, corpus.D0, corpus.D - corpus.D0, args.ctx_k, corpus.V, corpus.n))
    brain = Brain(corpus, args.cap, fr=args.fr, frac_inh=args.frac_inh, d0=args.d0,
                  geo=bool(args.geo), inh=bool(args.inh), budget=args.budget,
                  gamma=args.gamma, homeo=args.homeo, temp=args.temp,
                  lr_theta=args.lr_theta, temp_abs=args.temp_abs, norm=args.norm,
                  wd=args.wd, wd_in=args.wd_in, wd_rec=args.wd_rec, beta2=args.beta2, gain_jitter=args.gain_jitter,
                  theta_init=args.theta_init,
                  w_init_sd=args.w_init_sd, tau_fixed=args.tau_fixed, tau_s_mult=args.tau_s_mult,
                  spk_target=args.spk_target, spk_eta=args.spk_eta,
                  lr_gain=args.lr_gain, wta_block=args.wta_block, wta_k=args.wta_k,
                  delay_steps=args.delay_steps, lif=args.lif,
                  tau_m_lo=args.tau_m_lo, tau_m_hi=args.tau_m_hi,
                  syn_rate=args.syn_rate, prune_rate=args.prune_rate, grow2_frac=args.grow2_frac,
                  rec_norm=args.rec_norm, gain_rec_init=args.gain_rec_init,
                  slow_state=args.slow_state, slow_lambda=args.slow_lambda, slow_mut=args.slow_mut,
                  init_degree=args.init_degree, slow_rank=args.slow_rank,
                  seed=args.seed, device=dev)
    assert not T.is_grad_enabled(), "红线：不许有梯度"
    log("y=%d，从 0 开始长" % args.cap)

    brain.tok_err = -(T.log(corpus.freq.clamp(min=1e-6)))
    brain.tok_cnt = T.zeros(corpus.V, device=brain.dev)
    brain.tok_cnt_raw = T.zeros(corpus.V, device=brain.dev)
    brain._birth(1, corpus)
    log("第 1 个神经元已出生（认的词 = %d）" % int(brain.tok[0]))

    B = args.batch
    brain.reset_state(B)
    rng = np.random.default_rng(args.seed)
    pos = rng.integers(0, corpus.n - (B + 2), size=B)
    steps = args.tokens // B
    t0 = time.time()
    curve = []
    err_ref = 0.0

    for t in range(steps):
        idx = T.from_numpy(pos.astype(np.int64)).to(brain.dev)
        x = corpus.inputs(idx)
        tgt = corpus.tok_at(idx, 1)
        brain.x_t = x
        pred = brain.forward(x)

        with T.no_grad():
            pv = T.softmax(pred, dim=1).clamp(min=1e-9)
            ll = -T.log(pv.gather(1, tgt.unsqueeze(1)).squeeze(1))
            # 每个词的残余误差 = 该词的平均对数损失（EMA）。生长按它排序。
            ssum = T.zeros(corpus.V, device=brain.dev).index_add_(0, tgt, ll.detach())
            scnt = T.zeros(corpus.V, device=brain.dev).index_add_(
                0, tgt, T.ones_like(ll.detach()))
            cur = ssum / scnt.clamp(min=1.0)
            # 只更新这一批真正出现过的词；否则 8176 个没出现的词会被 0.99 衰减到 0，
            # 均值塌到 0.036，err_norm 变成 0.005，生长退化成每事件 +1（R091 第一次跑就是这样）。
            seen = scnt > 0
            te = brain.tok_err
            te[seen] = 0.99 * te[seen] + 0.01 * cur[seen]
            brain.tok_err = te

        brain.learn(tgt, lr=args.lr, plastic_b=args.plastic_b)
        brain.age[:brain.n_slots] += 1.0
        pos += 1

        if t > 0 and t % args.grow_every == 0 and brain.events < args.max_events:
            brain.events += 1
            if err_ref <= 0:
                # 固定参照 = unigram 的平均对数损失（模型的"什么都没学到"水平）
                err_ref = max(1e-6, float(corpus.unigram_nll()))
            err_norm = min(1.0, max(0.0, float(brain.tok_err.mean()) / err_ref))
            room = max(0.0, 1.0 - brain.N / brain.cap)
            # ---- 两阶段 ----
            # 阶段1（N < grow2_frac*y）：数量瓶颈，神经元快速 S 形增长，几乎不动结构。
            # 阶段2（N 接近 y）：神经元慢长，主要做突触发生 / 剪枝 —— "复杂拓扑开始出现"。
            phase2 = brain.N >= args.grow2_frac * brain.cap
            n_birth = int(math.ceil(args.grow_rate * max(1, brain.N) * room * err_norm))
            if phase2:
                n_birth = int(math.ceil(n_birth * 0.05))   # 结构期：神经元缓慢增长
            n_birth = min(n_birth, args.max_birth_per_event)
            nb = brain._birth(n_birth, corpus) if n_birth > 0 else 0
            nsyn = ncut = 0
            if phase2 and args.syn_rate > 0:
                nsyn = brain.synaptogenesis(int(args.syn_rate * brain.N), corpus)
            if phase2 and args.prune_rate > 0:
                ncut = brain.prune_edges(args.prune_rate)
            nd = brain._death(int(round(args.death_rate * brain.N * (brain.N / brain.cap))),
                              args.grace)
            if brain.h is not None and brain.h.shape[1] != brain.n_slots:
                brain.reset_state(B)
            topo = brain.topology()
            curve.append({"step": t, "N": brain.N, "born": nb, "died": nd,
                          "edges": topo["edges"], "edge_born": topo["edge_born"],
                          "edge_cut": topo["edge_cut"], "phase": 2 if phase2 else 1,
                          "err": float(brain.tok_err.mean())})
            if brain.events <= 8 or brain.events % 15 == 0:
                log("t=%6d 阶段%d N=%6d (+%d/-%d) 边=%d(生%d/剪%d) 误差=%.3f 用时%.0fs" % (
                    t, 2 if phase2 else 1, brain.N, nb, nd, topo["edges"],
                    topo["edge_born"], topo["edge_cut"],
                    float(brain.tok_err.mean()), time.time() - t0))

        if args.log_every and t > 0 and t % args.log_every == 0:
            m = evaluate(brain, corpus, n_batches=10, batch=B)
            log("t=%6d N=%6d PPL=%.1f top1=%.3f 发放=%.3f" % (
                t, brain.N, m["ppl"], m["top1"], float(brain.pr.mean())))

    m = evaluate(brain, corpus, n_batches=args.eval_batches, batch=B)
    uni = math.exp(corpus.unigram_nll())
    out = {"version": "grow-3.1.0", "in_mode": args.in_mode, "cap_y": args.cap, "N": brain.N,
           "y0_over_y": brain.N / args.cap, "born_total": brain.born_total,
           "dead_total": brain.dead_total, "steps": steps, "tokens": steps * B,
           "ppl": m["ppl"], "top1": m["top1"], "top5": m["top5"],
           "unigram_ppl": uni, "vs_unigram": m["ppl"] / uni,
           "backprop": False, "linear_readout": False,
           "lif": args.lif, "delay_steps": args.delay_steps, "slow_state": args.slow_state,
           "topology": brain.topology(),
           "grad_enabled": bool(T.is_grad_enabled())}
    log(json.dumps(out, ensure_ascii=False))
    if args.out:
        Brain.save(args.out, brain, {"result": out, "curve": curve})
        log("大脑已保存 -> %s" % args.out)
    if args.curve_out:
        json.dump({"curve": curve, "result": out}, open(args.curve_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--art", default=ART)
    p.add_argument("--in-mode", choices=["emb", "code"], default="emb")
    p.add_argument("--ctx-k", type=int, default=0, help="上下文窗口：看前 k 个词（0=不看）")
    p.add_argument("--ctx-dim", type=int, default=64, help="窗口压缩到多少维（固定稀疏投影）")
    p.add_argument("--cap", type=int, default=32768)
    p.add_argument("--fr", type=int, default=64)
    p.add_argument("--frac-inh", type=float, default=0.2)
    p.add_argument("--inh", type=int, default=1)
    p.add_argument("--geo", type=int, default=1)
    p.add_argument("--d0", type=float, default=10.0)
    p.add_argument("--budget", type=float, default=4.0, help="每单元权重的 L2 半径")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--homeo", type=float, default=0.25)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--lr-theta", type=float, default=1.0, help="阈值/偏置的学习率（应远大于权重）")
    p.add_argument("--temp-abs", type=float, default=1.0, help="softmax 的绝对温度（固定，不随输入漂移）")
    p.add_argument("--norm", choices=["none", "l2"], default="none", help="权重归一化方式")
    p.add_argument("--wd", type=float, default=0.0, help="（旧）权重衰减系数")
    p.add_argument("--wd-in", type=float, default=0.0, help="输入突触的每步衰减（0.01 ≈ 时间常数 100 步）")
    p.add_argument("--wd-rec", type=float, default=0.0, help="递归突触的每步衰减")
    p.add_argument("--beta2", type=float, default=0.999, help="二阶矩的衰减率")
    p.add_argument("--gain-jitter", type=float, default=1.0, help="增益的 log-normal 幅度；0=全部为 1")
    p.add_argument("--theta-init", choices=["zero", "drive"], default="zero")
    p.add_argument("--w-init-sd", type=float, default=0.01, help="输入权重的初始化标准差（可正负）")
    p.add_argument("--tau-fixed", type=float, default=0.0, help="固定快时间常数；0=从对数均匀分布抽；1=瞬时")
    p.add_argument("--tau-s-mult", type=float, default=4.5, help="慢时间常数 = 快时间常数 × 此倍数")
    p.add_argument("--spk-target", type=float, default=0.05, help="目标发放率（文献：成熟皮层 5-10%%）")
    p.add_argument("--spk-eta", type=float, default=0.02, help="发放阈值稳态速率")
    p.add_argument("--lr-gain", type=float, default=0.05, help="递归增益的学习率")
    p.add_argument("--wta-block", type=int, default=0, help="块内 k-WTA 的块大小（0=关）")
    p.add_argument("--wta-k", type=int, default=8, help="每块保留几个（局部抑制）")
    p.add_argument("--delay-steps", type=int, default=8, help="递归边的延迟档数 1..D（记忆跨度）")
    p.add_argument("--lif", type=int, default=1, help="1=真 LIF 发放；0=概率发放（对照）")
    p.add_argument("--tau-m-lo", type=float, default=2.0, help="膜时间常数下限")
    p.add_argument("--tau-m-hi", type=float, default=30.0, help="膜时间常数上限")
    p.add_argument("--syn-rate", type=float, default=0.05, help="结构期每事件新长边数 = 此值 x N")
    p.add_argument("--prune-rate", type=float, default=0.05, help="结构期每事件剪掉最弱边的比例")
    p.add_argument("--grow2-frac", type=float, default=0.85, help="N 超过 cap 的此比例后进入结构期")
    p.add_argument("--rec-norm", type=int, default=1, help="1=递归电流连续归一化；0=原始脉冲叠加(coincidence detection)")
    p.add_argument("--gain-rec-init", type=float, default=0.5, help="递归增益初值（rec_norm=0 时建议放大）")
    p.add_argument("--slow-state", type=int, default=0, help="1=启用慢状态驱动结构(文献34)；0=关，逐位复现3.0.0")
    p.add_argument("--slow-lambda", type=float, default=0.999, help="慢状态衰减(时间常数~1000步，远大于突触)")
    p.add_argument("--slow-mut", type=float, default=1.0, help="变异率与适应度反相关强度(免疫CLONALG)")
    p.add_argument("--init-degree", type=float, default=1.0, help="出生时初始填槽比例；<1 留空槽给结构期重建(文献03)")
    p.add_argument("--slow-rank", type=int, default=0, help="1=慢状态按排名主导采样(线C正解,CLONALG)；0=旧乘性微调复现")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--tokens", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--plastic-b", type=int, default=0, help="0=用满 batch")
    p.add_argument("--grow-every", type=int, default=250)
    p.add_argument("--grow-rate", type=float, default=1.0)
    p.add_argument("--death-rate", type=float, default=0.0005)
    p.add_argument("--max-birth-per-event", type=int, default=200000)
    p.add_argument("--max-events", type=int, default=200)
    p.add_argument("--grace", type=float, default=600.0)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--curve-out", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

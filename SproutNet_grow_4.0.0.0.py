"""SproutNet grow 4.0.0 —— 语义恒等码本 + 固定临界储层 + 误差驱动生长

这一版为什么这么改（全部由 R103 的实测判决，不是口味）
----------------------------------------------------
R099→R102 把病因追到「每单元认一个词 → 词表展开 = 可学线性层」。R103 试了它的出路
（每码位一个局部目标 + 固定码本解码），并用 **oracle 条件分布** 做了零 bug 判决：

    oracle trigram 条件分布 q                PPL  616.6  (6.42 nats)
    随机码 + q 的**真实**位边际 → 固定码本解码  PPL 1174.2  (7.07 nats)  ← 损失 1.90×
    随机码 + 完美预测（解码分辨率自检）        PPL    1.00

结论：4096 位的**随机**稀疏码装不下 8192 个词的条件分布，位边际解码结构性损失 0.64 nats。
原因很具体：码 = 纯随机投射 k-WTA，每个位 = 一个随机半空间的约 102 个词，这种"任意集合"
的隶属度本来就没有词身份信息。→ 那条出路判死，**不是学习问题，是码/解码问题**。

所以 4.0.0 的解码改成 **语义恒等码本**：
    词 w 的分数 = 属于 w 的那群神经元的活动均值（固定 1/count，零共享可学投影）
这与"每个词自己的读出群体"是同一件事，但它**不是**可学线性层堆在固定特征上：
每个神经元有自己的输入突触、自己的阈值、自己的误差，误差从不需要跨单元回传。
解码的分辨率损失为 0（R103 自检：完美预测 PPL=1.00）。

第二个判决：上下文。R062 说"递归是毒药"，R097 说"延迟边 +9%"，但那些递归都是**学出来的**
（局部规则驱动的递归权重会自激/发散，破坏回声状态性）。4.0.0 改成**固定临界储层**：
    s_t = (1−α)·s_{t−1} + α·tanh(W_in·x_t + W_rec·s_{t−1}),  α = 1/τ
W_rec 稀疏随机、谱半径 = 1（临界，边沿混沌——这是有生物物理依据的临界态，不是魔数），
**不可学**。τ 就是**上下文时间常数**，可手调（--ctx-tau），并且实测 PPL 随之变化。
结构从规则长出来（随机稀疏接线 + 临界标度），强度由局部规则学 —— 这正是"DNA 只编码
生长规则、结构自发生长"的立场。

身体结构
--------
  记忆器官（先天）  N_res 个固定临界递归单元：结构由生长规则在 t=0 生成，训练中不变
  读出皮层（后天）  从 1 个神经元长到 y：误差驱动的出生/死亡/突触发生/剪枝，S 形曲线
  总神经元 = N_res + N（两个数都打印，不含糊）

守住的红线
----------
  无 BP：全程 no_grad，没有 backward，逐单元误差 (hit − p_own)
  无共享可学线性层：词表展开 = 固定 1/count 池化；没有 h→logits 的共享投影
  SNN：轴突传脉冲（LIF 阈值+泄漏+reset+不应期），读出用连续率表达幅度
  显存：--n-res/--y 决定规模，运行时打印峰值显存；起 CUDA 前检查占用
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
V_DEFAULT = 8192


def log(*a):
    print("[g4]", *a, flush=True)


# ---------------------------------------------------------------- 数据

class Corpus:
    def __init__(self, art=ART, mem="res", win=4, device="cuda", heldout=False):
        self.dev = T.device(device)
        with open(os.path.join(art, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.V = int(self.meta["vocab"])
        self.tr = np.load(os.path.join(art, "tokens.npy"), mmap_mode="r")
        self.he = np.load(os.path.join(art, "heldout_tokens.npy"), mmap_mode="r")
        e = np.load(os.path.join(art, "emb.npy")).astype(np.float32)
        e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        self.emb = T.from_numpy(e).to(self.dev)
        self.D0 = int(self.emb.shape[1])
        self.mem = mem
        self.win = int(win) if mem in ("window", "both") else 0
        cnt = np.bincount(np.asarray(self.tr[:4_000_000], dtype=np.int64),
                          minlength=self.V).astype(np.float64) + 1.0
        self.freq = T.from_numpy((cnt / cnt.sum()).astype(np.float32)).to(self.dev)
        self.logfreq = T.log(self.freq.clamp(min=1e-12))

    def unigram_ppl(self, src=None, n=20000):
        src = self.he if src is None else src
        idx = np.random.default_rng(0).integers(0, src.size - 2, n)
        t = T.from_numpy(np.asarray(src[idx], dtype=np.int64)).to(self.dev)
        return float(math.exp(float(-self.logfreq[t].mean())))


class Reservoir:
    """固定临界递归储层：结构随机稀疏、谱半径 1、泄漏 α=1/τ。参数全部不可学。

    生物学对应：先天接线的递归皮层。临界（ρ=1）是"边沿混沌"，文献里与最大记忆容量/
    动态范围相关；τ 是膜/突触泄漏的时间常数，决定上下文能追溯多远。
    """

    def __init__(self, corpus, n, tau=4.0, fan=32, seed=20261104, device="cuda"):
        self.dev = T.device(device)
        self.N = int(n)
        self.alpha = 1.0 / max(1.0, float(tau))
        self.tau = float(tau)
        g = T.Generator().manual_seed(seed)
        D = corpus.D0
        # 输入：单位方差（x 是单位范数嵌入）
        self.W_in = (T.randn(self.N, D, generator=g) / math.sqrt(D)).to(self.dev)
        # 递归：稀疏 fan-in，行归一后按谱半径 1 缩放
        if self.N > 1:
            fan = min(int(fan), self.N)
            rows = T.arange(self.N).repeat_interleave(fan)
            cols = T.stack([T.randperm(self.N, generator=g)[:fan] for _ in range(self.N)]).reshape(-1)
            vals = T.randn(rows.numel(), generator=g) / math.sqrt(fan)
            W = T.sparse_coo_tensor(T.stack([rows, cols]), vals, (self.N, self.N)).coalesce()
            # 幂迭代估计谱半径（固定随机矩阵的 ρ）
            v = T.randn(self.N, 1, generator=g).to(self.dev)
            Wd = W.to(self.dev)
            for _ in range(30):
                v = T.sparse.mm(Wd, v)
                nv = float(v.norm())
                if nv < 1e-12:
                    break
                v = v / nv
            rho = float(T.sparse.mm(Wd, v).norm())
            self.W_rec = T.sparse_coo_tensor(Wd.indices(), Wd.values() / max(rho, 1e-6),
                                            (self.N, self.N)).coalesce()
        else:
            self.W_rec = None
        self.s = None

    def reset(self, B):
        self.s = T.zeros(B, self.N, device=self.dev)

    def step(self, x):
        """x: (B, D) 当前词嵌入 → 新状态 (B, N)"""
        if self.s is None or self.s.shape[0] != x.shape[0]:
            self.reset(x.shape[0])
        u = x @ self.W_in.t()
        if self.W_rec is not None:
            u = u + T.sparse.mm(self.W_rec, self.s.t()).t()
        self.s = (1 - self.alpha) * self.s + self.alpha * T.tanh(u)
        return self.s


# ---------------------------------------------------------------- 大脑

class Brain:
    def __init__(self, corpus, cap, mem="res", D_feat=512, fr=0, delay_steps=8,
                 frac_inh=0.2, budget=0.0, gamma=1.0, homeo=0.25, temp_abs=1.0,
                 lr_theta=1.0, spk_target=0.05, spk_eta=0.02, lif=1,
                 tau_m_lo=2.0, tau_m_hi=30.0, beta2=0.999, w_init_sd=0.01,
                 geo=1, d0=10.0, seed=0, device="cuda"):
        self.dev = T.device(device)
        self.V = corpus.V
        self.D = int(D_feat)
        self.cap = int(cap)
        self.mem = mem
        self.fr = int(fr)
        self.Delay = max(1, int(delay_steps))
        self.fr_per = max(1, self.fr // max(1, self.Delay)) if self.fr else 0
        self.use_lif = int(lif)
        self.gamma = gamma
        self.homeo = homeo
        self.budget = float(budget)
        self.spk_target = spk_target
        self.spk_eta = spk_eta
        self.lr_theta = lr_theta
        self.Tglob = float(temp_abs)
        self.beta2 = beta2
        self.frac_inh = frac_inh
        self.tau_m_lo = float(tau_m_lo)
        self.w_init_sd = float(w_init_sd)
        self.tau_m_hi = float(tau_m_hi)
        self.geo = int(geo)
        self.d0 = float(d0)
        self.rng = np.random.default_rng(seed + 13)
        self.g = T.Generator(device="cpu").manual_seed(seed + 7)

        z = lambda *s: T.zeros(*s, dtype=T.float32, device=self.dev)
        o = lambda *s: T.ones(*s, dtype=T.float32, device=self.dev)

        self.coord = z(cap, 2)
        self.alive = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.is_inh = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.tok = T.full((cap,), -1, dtype=T.int64, device=self.dev)
        self.theta = z(cap)
        self.gain = o(cap)
        self.tau_f = o(cap)
        self.tau_s = o(cap)
        self.rate_tgt = z(cap)
        self.rate_ema = z(cap)
        self.acc_ema = z(cap) + 0.5
        self.age = z(cap)
        self.w_in = z(cap, self.D)
        self.v_in = z(cap, self.D)
        self.v_th = z(cap)
        self.theta_spike = z(cap)
        self.spk_ema = o(cap) * spk_target
        self.v_scale = o(cap) * 0.02
        self.u_scale = o(cap) * 0.02
        self.u_mean = z(cap)
        self.tau_m = o(cap) * 5.0
        self.tau_decay = o(cap) * 0.8
        self.src_sign = o(cap)
        self.born = T.zeros(cap, dtype=T.int64, device=self.dev)
        self.w_rec = z(cap, self.Delay, self.fr_per) if self.fr_per else None
        self.rec_src = T.zeros(cap, self.Delay, self.fr_per, dtype=T.int64, device=self.dev) if self.fr_per else None
        self.rec_mask = T.zeros(cap, self.Delay, self.fr_per, dtype=T.bool, device=self.dev) if self.fr_per else None
        self.v_rec = z(cap, self.Delay, self.fr_per) if self.fr_per else None
        self.rec_gain = o(cap) * 0.5
        self.n_edges = 0
        self.edge_born = 0
        self.edge_cut = 0

        self.n_slots = 0
        self.N = 0
        self.birth_counter = 0
        self.born_total = 0
        self.dead_total = 0
        self.events = 0
        self.alive_idx = T.zeros(0, dtype=T.int64, device=self.dev)
        self.tok_cnt = z(self.V)
        self.tok_cnt_raw = z(self.V)

        self.f_state = self.s_state = self.h = None
        self.v_mem = self.ref = self.spike = self.spike_buf = None
        self.spk_ptr = 0
        self._ptr_before = 0
        self.ur = None
        self.x_t = None

    # ------------------------------------------------ 出生 / 死亡

    def _birth(self, n_new, corpus):
        if self.n_slots >= self.cap * 0.98:
            compacted = self.compact()
            if self.n_slots >= self.cap:
                return 0
        n_new = int(min(n_new, self.cap - self.n_slots))
        if n_new <= 0:
            return 0
        # 生长位置：按"词的残余误差 / sqrt(已有神经元数)"——误差高、人少的词优先
        score = (self.tok_err + 1e-3) / T.sqrt(self.tok_cnt_raw + 1.0)
        score = score.masked_fill(self.tok_cnt_raw <= 0, score.max() + 1.0)
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
            self.is_inh[col] = bool(self.rng.random() < self.frac_inh)
            self.src_sign[col] = -1.0 if bool(self.is_inh[col]) else 1.0
            tf = float(np.exp(self.rng.uniform(math.log(1.5), math.log(6.0))))
            self.tau_f[col] = tf
            self.tau_s[col] = tf * 4.5
            tm = float(np.exp(self.rng.uniform(math.log(self.tau_m_lo), math.log(self.tau_m_hi))))
            self.tau_m[col] = tm
            self.tau_decay[col] = math.exp(-1.0 / tm)
            self.theta[col] = 0.0
            self.theta_spike[col] = (float(self.theta_spike[:col][self.alive[:col]].median())
                                     if col > 0 else 0.0)
            self.spk_ema[col] = float(self.spk_target)
            self.gain[col] = float(np.exp(self.rng.normal(0.0, 0.35)))
            f = float(corpus.freq[v])
            self.rate_tgt[col] = min(0.3, max(1e-4, f))
            same = (self.tok[:col] == v) & self.alive[:col]
            self.acc_ema[col] = float(self.acc_ema[:col][same].mean()) if bool(same.any()) else 0.5
            self.age[col] = 0.0
            wv = self.rng.standard_normal(self.D).astype(np.float32) * self.w_init_sd
            self.w_in[col] = T.from_numpy(wv).to(self.dev)
            # 递归边（默认关）：只接更老的单元，几何衰减
            if self.fr_per and col > 0:
                if self.geo:
                    d = np.linalg.norm(self.coord_np[:col] - c, axis=1)
                    pp = np.exp(-d / self.d0)
                    pp = pp / pp.sum() if pp.sum() > 0 else np.ones(col) / col
                    kk = min(self.fr, int((pp > 0).sum()))
                    src = self.rng.choice(col, size=kk, replace=False, p=pp) if kk > 0 else np.zeros(0, np.int64)
                else:
                    src = self.rng.choice(col, size=min(self.fr, col), replace=False)
                for ii, su in enumerate(src):
                    k = ii % self.Delay
                    j = (ii // self.Delay) % self.fr_per
                    self.rec_src[col, k, j] = int(su)
                    self.w_rec[col, k, j] = abs(float(self.rng.standard_normal())) * 0.03 + 1e-3
                    self.rec_mask[col, k, j] = True
                    self.n_edges += 1
            self.alive[col] = True
            self.tok[col] = v
        self.N += n_new
        self.born_total += n_new
        if self.budget > 0:
            self._normalize_rows(T.arange(self.n_slots - n_new, self.n_slots, device=self.dev))
        self._refresh()
        return n_new

    def _normalize_rows(self, ids):
        if ids.numel() == 0:
            return
        s = self.w_in[ids].pow(2).sum(1).sqrt() + 1e-9
        self.w_in[ids] *= (self.budget / s).unsqueeze(1)

    def _refresh(self):
        alive = self.alive[:self.n_slots]
        self.alive_idx = alive.nonzero(as_tuple=True)[0]
        cnt = T.zeros(self.V, device=self.dev)
        m = alive & (self.tok[:self.n_slots] >= 0)
        if bool(m.any()):
            t = self.tok[:self.n_slots][m]
            cnt.index_add_(0, t, T.ones(t.numel(), device=self.dev))
        self.tok_cnt_raw = cnt
        self.tok_cnt = cnt.clamp(min=1.0)

    def _death(self, n_die, grace=200.0):
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
        if self.fr_per:
            cut = int(self.rec_mask[dead].sum())
            self.n_edges -= cut
            self.w_rec[dead] = 0
            self.rec_mask[dead] = False
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
        keys = ["coord", "born", "alive", "is_inh", "tok", "theta", "gain", "tau_f", "tau_s",
                "rate_tgt", "rate_ema", "acc_ema", "age", "src_sign", "u_mean", "u_scale",
                "w_in", "v_in", "v_th", "theta_spike", "spk_ema", "rec_gain",
                "tau_m", "tau_decay", "v_scale"]
        if self.fr_per:
            keys += ["rec_src", "w_rec", "rec_mask", "v_rec"]
        for k in keys:
            getattr(self, k)[:m] = getattr(self, k)[:n][alive]
        if self.fr_per:
            rs = self.rec_src[:m].clone()
            bad = ~alive_full[rs]
            self.rec_src[:m] = remap[rs]
            cut = self.rec_mask[:m] & bad
            self.n_edges -= int(cut.sum())
            self.rec_mask[:m].masked_fill_(cut, False)
            self.w_rec[:m].masked_fill_(bad, 0.0)
        self.alive[m:n] = False
        self.tok[m:n] = -1
        self.n_slots = m
        self._refresh()
        if self.h is not None and self.h.shape[1] != m:
            B = self.h.shape[0]
            self.reset_state(B)
        return n - m

    # ------------------------------------------------ 前向

    def reset_state(self, B):
        n = self.n_slots
        z = lambda *s: T.zeros(*s, device=self.dev)
        self.f_state = z(B, n)
        self.s_state = z(B, n)
        self.h = z(B, n)
        self.v_mem = z(B, n)
        self.ref = z(B, n)
        self.spike = z(B, n)
        if self.fr_per:
            self.spike_buf = z(self.Delay + 1, B, n)
        self.spk_ptr = 0
        self._ptr_before = 0

    def forward(self, h_feat):
        """h_feat: (B, D) 已单位化的记忆特征 → 词分数 (B, V)"""
        n = self.n_slots
        B = h_feat.shape[0]
        if self.h is None or self.h.shape != (B, n):
            self.reset_state(B)
        self.x_t = h_feat
        u_in = h_feat @ self.w_in[:n].t()
        if self.fr_per and self.n_edges > 0:
            self._ptr_before = self.spk_ptr
            u_rec = T.zeros(B, n, device=self.dev)
            p = self.spk_ptr
            for k in range(self.Delay):
                slot = (p - 1 - k) % (self.Delay + 1)
                buf_k = self.spike_buf[slot]
                ids = self.rec_src[:n, k, :]
                w = self.w_rec[:n, k, :] * self.src_sign[ids] * self.rec_mask[:n, k, :].float()
                g = buf_k[:, ids.reshape(-1)].reshape(B, n, self.fr_per)
                u_rec += (g * w.unsqueeze(0)).sum(-1)
            self.ur = u_rec
            u = u_in + self.rec_gain[:n].unsqueeze(0) * u_rec
        else:
            self.ur = None
            u = u_in

        ubar = u.mean(0)
        self.u_mean[:n] = 0.98 * self.u_mean[:n] + 0.02 * ubar
        self.u_scale[:n] = (0.98 * self.u_scale[:n] + 0.02 * (u - ubar).abs().mean(0)).clamp_(min=2e-3)

        tf = self.tau_f[:n].unsqueeze(0)
        ts = self.tau_s[:n].unsqueeze(0)
        self.f_state += (u - self.f_state) / tf
        self.s_state += (u - self.s_state) / ts
        drive = (self.f_state + self.gamma * self.s_state) * self.gain[:n].unsqueeze(0)
        score = ((drive - self.theta[:n].unsqueeze(0)) / self.Tglob).clamp_(-30.0, 30.0)

        if self.use_lif:
            v = self.v_mem * self.tau_decay[:n].unsqueeze(0) + u
            v = v * (self.ref <= 0).float()
            spike = (v > self.theta_spike[:n].unsqueeze(0)).float()
            v = T.where(spike > 0, T.zeros_like(v), v)
            self.ref = (self.ref - 1).clamp(min=0)
            self.ref = T.where(spike > 0, T.full_like(self.ref, 1.0), self.ref)
            self.v_mem = v
            self.v_scale[:n] = 0.98 * self.v_scale[:n] + 0.02 * v.abs().mean(0).clamp(min=1e-3)
        else:
            pr = T.sigmoid((drive - self.theta_spike[:n].unsqueeze(0)) /
                           (self.u_scale[:n].unsqueeze(0) + 1e-9))
            spike = (T.rand(pr.shape, device=self.dev) < pr).float()
        alivef = self.alive[:n].float().unsqueeze(0)
        spike = spike * alivef
        self.spike = spike
        self.h = spike
        if self.fr_per:
            self.spike_buf[self.spk_ptr] = spike
            self.spk_ptr = (self.spk_ptr + 1) % (self.Delay + 1)

        # 语义恒等码本：词的分数 = 属于它的神经元分数均值（固定 1/count，零可学参数）
        sc = T.zeros(B, self.V, device=self.dev)
        mask = self.alive[:n] & (self.tok[:n] >= 0)
        if bool(mask.any()):
            sel = mask.nonzero(as_tuple=True)[0]
            sc.index_add_(1, self.tok[sel], score[:, sel])
            sc = sc / self.tok_cnt.unsqueeze(0)
            cov = (self.tok_cnt_raw > 0).float().unsqueeze(0)
            ncov = cov.sum().clamp(min=1.0)
            sc = sc - (sc * cov).sum(1, keepdim=True) / ncov
        return sc

    # ------------------------------------------------ 局部学习

    def learn(self, tgt_tok, lr=0.003, plastic_b=0):
        n = self.n_slots
        if self.h is None or n == 0:
            return {}
        B = self.h.shape[0]
        nb = B if plastic_b <= 0 else min(plastic_b, B)
        tok = self.tok[:n]
        valid = self.alive[:n] & (tok >= 0)
        pv = T.softmax(self.score_v[:nb], dim=1)
        p_own = pv.gather(1, tok.clamp(min=0).unsqueeze(0).expand(nb, n))
        hit = (tgt_tok[:nb].unsqueeze(1) == tok.unsqueeze(0)).float()
        e = (hit - p_own) * valid.float().unsqueeze(0)
        with T.no_grad():
            acc = (1.0 - (hit - p_own).abs()).mean(0)
            self.acc_ema[:n] = 0.99 * self.acc_ema[:n] + 0.01 * acc
            self.rate_ema[:n] = 0.995 * self.rate_ema[:n] + 0.005 * self.h[:nb].mean(0)

        pre = self.x_t[:nb]
        es = e / self.Tglob
        g = (es.t() @ pre) / nb
        v = self.v_in[:n]
        v.mul_(self.beta2).add_(g * g, alpha=1 - self.beta2)
        self.w_in[:n] += lr * g / (v.sqrt() + 1e-8)
        if self.budget > 0:
            self._normalize_rows(self.alive_idx)

        if self.fr_per and self.n_edges > 0:
            p = self._ptr_before
            for k in range(self.Delay):
                slot = (p - 1 - k) % (self.Delay + 1)
                buf_k = self.spike_buf[slot]
                ids = self.rec_src[:n, k, :]
                pre_r = buf_k[:nb, ids.reshape(-1)].reshape(nb, n, self.fr_per)
                gg = T.einsum("bn,bnc->nc", es, pre_r) / nb
                vr = self.v_rec[:n, k, :]
                vr.mul_(self.beta2).add_(gg * gg, alpha=1 - self.beta2)
                self.w_rec[:n, k, :] += lr * gg / (vr.sqrt() + 1e-8) * self.rec_mask[:n, k, :].float()

        with T.no_grad():
            self.spk_ema[:n] = 0.995 * self.spk_ema[:n] + 0.005 * self.h[:nb].mean(0)
            _e = T.log(self.spk_ema[:n].clamp(min=1e-3)) - math.log(max(1e-6, self.spk_target))
            scl = self.v_scale[:n] if self.use_lif else self.u_scale[:n]
            self.theta_spike[:n] += self.spk_eta * scl * _e
            self.theta_spike[:n].clamp_(-1e4, 1e4)
            if self.homeo > 0:
                gt = es.mean(0)
                self.v_th[:n].mul_(self.beta2).add_(gt * gt, alpha=1 - self.beta2)
                self.theta[:n] -= self.lr_theta * gt / (self.v_th[:n].sqrt() + 1e-8)
        return {"mae": float((hit - p_own).abs().mean())}

    # ------------------------------------------------ 突触发生 / 剪枝 / 拓扑

    def synaptogenesis(self, n_new):
        if not self.fr_per or self.n_slots <= 1 or n_new <= 0:
            return 0
        n = self.n_slots
        alive = self.alive[:n]
        free = (~self.rec_mask[:n]) & alive.view(n, 1, 1)
        idx = free.reshape(-1).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return 0
        units = idx // (self.Delay * self.fr_per)
        we = self.tok_err[self.tok[units].clamp(min=0)]
        w = we - we.min() + 1e-3
        ns = min(int(n_new), int(idx.numel()))
        sel = T.multinomial(w.clamp(min=1e-9), ns, replacement=False)
        chosen = idx[sel]
        u = chosen // (self.Delay * self.fr_per)
        rem = chosen % (self.Delay * self.fr_per)
        k, j = rem // self.fr_per, rem % self.fr_per
        pool = alive.nonzero(as_tuple=True)[0]
        rate = self.rate_ema[pool].clamp(min=1e-4)
        src = pool[T.multinomial(rate / rate.sum(), ns, replacement=True)]
        self.rec_src[u, k, j] = src
        self.w_rec[u, k, j] = T.rand(ns, device=self.dev) * 0.03 + 1e-3
        self.rec_mask[u, k, j] = True
        self.n_edges += ns
        self.edge_born += ns
        return int(ns)

    def prune_edges(self):
        if not self.fr_per:
            return 0
        n = self.n_slots
        m = self.rec_mask[:n]
        if not bool(m.any()):
            return 0
        cut = m & (self.w_rec[:n].abs() < 0.003)
        self.rec_mask[:n].masked_fill_(cut, False)
        c = int(cut.sum())
        self.n_edges -= c
        self.edge_cut += c
        return c

    def topology(self):
        n = self.n_slots
        if n == 0:
            return {"edges": 0, "avg_indeg": 0.0, "free_slots": 0}
        alive = self.alive[:n]
        if not self.fr_per:
            return {"edges": 0, "avg_indeg": 0.0, "free_slots": 0}
        m = self.rec_mask[:n]
        indeg = m.sum(dim=(1, 2)).float()
        return {"edges": int(m[alive].sum()), "avg_indeg": round(float(indeg[alive].mean()), 2),
                "max_indeg": int(indeg.max()),
                "edge_born": self.edge_born, "edge_cut": self.edge_cut,
                "free_slots": int(((~m) & alive.view(n, 1, 1)).sum())}

    # ------------------------------------------------ 存取

    def snapshot(self):
        n = self.n_slots
        d = {"cap": self.cap, "n_slots": n, "N": self.N, "D": self.D, "V": self.V,
             "mem": self.mem, "fr": self.fr, "Delay": self.Delay, "fr_per": self.fr_per,
             "birth_counter": self.birth_counter, "born_total": self.born_total,
             "dead_total": self.dead_total, "events": self.events, "Tglob": self.Tglob}
        keys = ["alive", "is_inh", "tok", "coord", "born", "theta", "gain", "tau_f", "tau_s",
                "rate_tgt", "rate_ema", "acc_ema", "age", "src_sign", "u_mean", "u_scale",
                "w_in", "v_in", "v_th", "theta_spike", "spk_ema", "rec_gain",
                "tau_m", "tau_decay", "v_scale", "tok_cnt", "tok_cnt_raw"]
        if self.fr_per:
            keys += ["rec_src", "w_rec", "rec_mask", "v_rec"]
        for k in keys:
            d[k] = getattr(self, k)[:n].cpu()
        d["n_edges"] = self.n_edges
        d["topology"] = self.topology()
        return d

    def load_state(self, d):
        n = int(d["n_slots"])
        self.n_slots = n
        self.N = int(d["N"])
        self.birth_counter = int(d.get("birth_counter", 0))
        self.born_total = int(d.get("born_total", 0))
        self.dead_total = int(d.get("dead_total", 0))
        self.events = int(d.get("events", 0))
        keys = ["alive", "is_inh", "tok", "coord", "born", "theta", "gain", "tau_f", "tau_s",
                "rate_tgt", "rate_ema", "acc_ema", "age", "src_sign", "u_mean", "u_scale",
                "w_in", "v_in", "v_th", "theta_spike", "spk_ema", "rec_gain",
                "tau_m", "tau_decay", "v_scale", "tok_cnt", "tok_cnt_raw"]
        if self.fr_per:
            keys += ["rec_src", "w_rec", "rec_mask", "v_rec"]
        for k in keys:
            if k in d:
                getattr(self, k)[:n] = d[k].to(self.dev)
        self.n_edges = int(d.get("n_edges", 0))
        self._refresh()
        self.h = None


# ---------------------------------------------------------------- 评测 / 生成

def _win_toks(src, pos_np, win):
    """位置数组 → (B, win) token id 矩阵，第 0 列是当前位置的词。"""
    w = max(1, win)
    return np.stack([np.asarray(src[np.asarray(pos_np) - j], dtype=np.int64) for j in range(w)], axis=1)


def _feat(corpus, res, win, toks_np, hnorm=1, proj=None):
    """toks_np: (B, win) token id，第 0 列 = 当前词（注意是 id，不是位置）"""
    toks = [T.from_numpy(np.asarray(toks_np[:, j], dtype=np.int64)).to(corpus.dev)
            for j in range(max(1, win))]
    parts = []
    if win > 0:
        parts.append(T.cat([corpus.emb[t] for t in toks], dim=1))
    if res is not None:
        parts.append(res.step(corpus.emb[toks[0]]))
    h = T.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    if proj is not None:
        h = h @ proj
    if hnorm:
        h = h / (h.norm(dim=1, keepdim=True) + 1e-6)
    return h


class _Keep:
    """评测不污染训练状态（3.1.0 的教训：评测后状态被清，训练流等于断掉）。"""

    def __init__(self, brain, res):
        self.b, self.r = brain, res

    def __enter__(self):
        b = self.b
        self.s = None
        if b.h is not None:
            self.s = (b.h.clone(), b.f_state.clone(), b.s_state.clone(),
                      b.v_mem.clone(), b.ref.clone(), b.spike.clone(),
                      b.spike_buf.clone() if b.fr_per else None, b.spk_ptr, b._ptr_before)
        self.rs = self.r.s.clone() if (self.r is not None and self.r.s is not None) else None
        return self

    def __exit__(self, *a):
        if self.s is not None:
            b = self.b
            (b.h, b.f_state, b.s_state, b.v_mem, b.ref, b.spike,
             b.spike_buf, b.spk_ptr, b._ptr_before) = self.s
        if self.r is not None and self.rs is not None:
            self.r.s = self.rs
        return False


def evaluate(brain, corpus, res, win, n_batches=40, batch=64, seed=0, src=None, hnorm=1,
             proj=None, warm=24):
    src = corpus.he if src is None else src
    rng = np.random.default_rng(seed)
    lo = max(win, 2) + warm + 2
    idxs = rng.integers(lo, src.size - 2, size=(n_batches, batch))
    nll, top1, top5, n = 0.0, 0, 0, 0
    with _Keep(brain, res):
      for bi in range(n_batches):
        idx = idxs[bi]
        if res is not None:
            res.reset(batch)
        brain.reset_state(batch)
        for w in range(warm):                       # 预热：让记忆真的带上历史
            brain.forward(_feat(corpus, res, win, _win_toks(src, idx - warm + w, win), hnorm, proj))
        h = _feat(corpus, res, win, _win_toks(src, idx, win), hnorm, proj)
        pred = brain.forward(h)
        tgt = T.from_numpy(np.asarray(src[idx + 1], dtype=np.int64)).to(corpus.dev)
        p = T.softmax(pred, dim=1).clamp(min=1e-9)
        ar = T.arange(batch, device=corpus.dev)
        nll += float(-T.log(p)[ar, tgt].sum())
        top1 += int((p.argmax(1) == tgt).sum())
        top5 += int(p.topk(5, 1).indices.eq(tgt.unsqueeze(1)).any(1).sum())
        n += batch
    fire = float(brain.h.float().mean()) if brain.h is not None else 0.0
    diag = {"empty_words": int((brain.tok_cnt_raw == 0).sum()),
            "w_norm": round(float(brain.w_in[:brain.n_slots][brain.alive[:brain.n_slots]].norm(dim=1).mean()), 3)
            if brain.N else 0.0,
            "theta": round(float(brain.theta[:brain.n_slots][brain.alive[:brain.n_slots]].mean()), 3)
            if brain.N else 0.0,
            "fire": round(fire, 4)}
    return {"ppl": math.exp(nll / n), "top1": top1 / n, "top5": top5 / n, "diag": diag}


def generate(brain, corpus, res, win, prompt_ids, n_new=60, temp=0.9, topk=0, hnorm=1,
             seed=0, proj=None):
    """死脑子提问：给一段 token，自回归续写。返回新 token 列表。"""
    rng = np.random.default_rng(seed)
    if res is not None:
        res.reset(1)
    brain.reset_state(1)
    buf = list(prompt_ids)
    out = []
    for step in range(n_new):
        h = _feat(corpus, res, win, np.array([[buf[-1 - j] if len(buf) > j else buf[0]
                                               for j in range(max(1, win))]], dtype=np.int64),
                   hnorm, proj)
        sc = brain.forward(h)
        lg = sc[0] / max(1e-6, temp)
        if topk > 0:
            v, i = lg.topk(topk)
            lg = T.full_like(lg, -1e9)
            lg[i] = v
        p = T.softmax(lg, dim=0).cpu().numpy().astype(np.float64)
        p = p / p.sum()
        nxt = int(rng.choice(p.size, p=p))
        out.append(nxt)
        buf.append(nxt)
    return out


# ---------------------------------------------------------------- 主流程

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--art", default=ART)
    p.add_argument("--mem", choices=["window", "res", "both"], default="res")
    p.add_argument("--win", type=int, default=4)
    p.add_argument("--n-res", dest="n_res", type=int, default=4096)
    p.add_argument("--ctx-tau", dest="ctx_tau", type=float, default=4.0,
                   help="上下文时间常数（储层泄漏 α=1/τ）；越大记忆越长，可手调")
    p.add_argument("--y", type=int, default=32768, help="读出皮层神经元上限 y")
    p.add_argument("--fr", type=int, default=0, help="读出层内部递归边数（默认 0=关）")
    p.add_argument("--delay-steps", dest="delay_steps", type=int, default=8)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--lr-theta", dest="lr_theta", type=float, default=1.0)
    p.add_argument("--homeo", type=float, default=0.25)
    p.add_argument("--budget", type=float, default=0.0)
    p.add_argument("--grow-every", dest="grow_every", type=int, default=400)
    p.add_argument("--grow-rate", dest="grow_rate", type=float, default=1.0)
    p.add_argument("--death-rate", dest="death_rate", type=float, default=0.0005)
    p.add_argument("--grow2-frac", dest="grow2_frac", type=float, default=0.85)
    p.add_argument("--syn-rate", dest="syn_rate", type=float, default=0.0)
    p.add_argument("--max-events", dest="max_events", type=int, default=400)
    p.add_argument("--grace", type=float, default=200.0)
    p.add_argument("--log-every", dest="log_every", type=int, default=0)
    p.add_argument("--eval-batches", dest="eval_batches", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--curve-out", dest="curve_out", default="")
    p.add_argument("--load", default="")
    p.add_argument("--sample", default="", help="训练结束后用它当提示续写（逗号分隔的 token id）")
    p.add_argument("--sample-n", dest="sample_n", type=int, default=40)
    p.add_argument("--feat-dim", dest="feat_dim", type=int, default=0,
                   help="把记忆特征固定稀疏投射到这个维度（0=不投射）。y×D 受显存约束，"
                        "要更大 y 就压 D；投射不可学，属输入编码")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        if not T.cuda.is_available():
            raise SystemExit("CUDA 不可用")
        free, total = T.cuda.mem_get_info()
        log("GPU 空闲 %.2f / %.2f GB" % (free / 1e9, total / 1e9))

    corpus = Corpus(args.art, args.mem, args.win, args.device)
    res = None
    if args.mem in ("res", "both"):
        res = Reservoir(corpus, args.n_res, args.ctx_tau, device=args.device)
        log("储层 %d 单元，τ=%.1f（α=%.3f），谱半径=1，不可学" % (res.N, res.tau, res.alpha))
    D_raw = (args.win * corpus.D0 if args.mem in ("window", "both") else 0) + \
            (args.n_res if res is not None else 0)
    proj = None
    D_feat = D_raw
    if args.feat_dim and args.feat_dim < D_raw:
        g = T.Generator().manual_seed(20261105)
        nnz = 16
        P_ = T.zeros(D_raw, args.feat_dim)
        for j in range(args.feat_dim):
            ii = T.randperm(D_raw, generator=g)[:nnz]
            P_[ii, j] = T.randn(nnz, generator=g) / math.sqrt(nnz)
        proj = P_.to(corpus.dev)
        D_feat = args.feat_dim
        log("记忆特征 %d → 固定稀疏投射 → %d 维（不可学，属输入编码）" % (D_raw, D_feat))
    brain = Brain(corpus, args.y, mem=args.mem, D_feat=D_feat, fr=args.fr,
                  delay_steps=args.delay_steps, budget=args.budget, homeo=args.homeo,
                  lr_theta=args.lr_theta, seed=args.seed, device=args.device)
    log("特征维 D=%d；读出皮层 y=%d（从 1 长起）" % (D_feat, args.y))
    if args.load:
        d = T.load(args.load, map_location=args.device)
        brain.load_state(d["brain"] if "brain" in d else d)
        log("已加载 %s（N=%d）" % (args.load, brain.N))
    assert not T.is_grad_enabled(), "红线：不许有梯度"

    src = corpus.tr
    brain.tok_err = -(T.log(corpus.freq.clamp(min=1e-6)))
    brain.tok_cnt = T.zeros(corpus.V, device=brain.dev)
    brain.tok_cnt_raw = T.zeros(corpus.V, device=brain.dev)
    if not args.load:
        brain._birth(1, corpus)
        log("第 1 个读出神经元已出生（词 %d）" % int(brain.tok[0]))

    B = args.batch
    corpus.tr_src = src
    if res is not None:
        res.reset(B)
    brain.reset_state(B)
    rng = np.random.default_rng(args.seed)
    lo = max(args.win, 2) + 1
    pos = rng.integers(lo, src.size - 4, size=B).astype(np.int64)
    steps = max(1, args.tokens // B)
    curve = []
    err_ref = max(1e-6, float(-corpus.logfreq.mean()))
    t0 = time.time()
    nll_acc, n_acc = 0.0, 0

    for t in range(1, steps + 1):
        h = _feat(corpus, res, args.win, _win_toks(src, pos, args.win), 1, proj)
        tgt = T.from_numpy(np.asarray(src[pos + 1], dtype=np.int64)).to(corpus.dev)
        pred = brain.forward(h)
        brain.score_v = pred
        with T.no_grad():
            pv = T.softmax(pred, dim=1).clamp(min=1e-9)
            ll = -T.log(pv.gather(1, tgt.unsqueeze(1)).squeeze(1))
            nll_acc += float(ll.sum())
            n_acc += B
            ssum = T.zeros(corpus.V, device=brain.dev).index_add_(0, tgt, ll)
            scnt = T.zeros(corpus.V, device=brain.dev).index_add_(0, tgt, T.ones_like(ll))
            seen = scnt > 0
            cur = ssum / scnt.clamp(min=1.0)
            brain.tok_err[seen] = 0.99 * brain.tok_err[seen] + 0.01 * cur[seen]
        brain.learn(tgt, lr=args.lr)
        brain.age[:brain.n_slots] += 1.0
        pos += 1
        pos[pos >= src.size - 4] = lo

        if t % args.grow_every == 0 and brain.events < args.max_events:
            brain.events += 1
            err_norm = min(1.0, max(0.0, float(brain.tok_err.mean()) / err_ref))
            room = max(0.0, 1.0 - brain.N / brain.cap)
            phase2 = brain.N >= args.grow2_frac * brain.cap
            n_birth = int(math.ceil(args.grow_rate * max(1, brain.N) * room * err_norm))
            if phase2:
                n_birth = int(math.ceil(n_birth * 0.05))
            n_birth = min(n_birth, args.y)
            nb = brain._birth(n_birth, corpus) if n_birth > 0 else 0
            nsyn = brain.synaptogenesis(int(args.syn_rate * brain.N)) if (phase2 and args.syn_rate) else 0
            nc = brain.prune_edges() if (phase2 and args.syn_rate) else 0
            nd = brain._death(int(round(args.death_rate * brain.N * (brain.N / brain.cap))), args.grace)
            if brain.h is not None and brain.h.shape[1] != brain.n_slots:
                brain.reset_state(B)
            to = brain.topology()
            curve.append({"step": t, "N": brain.N, "born": nb, "died": nd, "phase": 2 if phase2 else 1,
                          "edges": to["edges"], "err": float(brain.tok_err.mean()),
                          "train_ppl": math.exp(nll_acc / max(1, n_acc))})
            nll_acc, n_acc = 0.0, 0
            if brain.events <= 10 or brain.events % 20 == 0:
                log("t=%6d 阶段%d N=%6d (+%d/-%d) 边=%d 误差=%.3f 训练PPL=%.0f 用时%.0fs 显存%.1f/%.1fGB" % (
                    t, 2 if phase2 else 1, brain.N, nb, nd, to["edges"],
                    float(brain.tok_err.mean()), curve[-1]["train_ppl"], time.time() - t0,
                    T.cuda.memory_allocated() / 1e9, T.cuda.memory_reserved() / 1e9))

        if args.log_every and t % args.log_every == 0:
            m = evaluate(brain, corpus, res, args.win, n_batches=8, batch=32, seed=1, proj=proj)
            log("t=%6d N=%6d 留出PPL=%.1f top1=%.3f" % (t, brain.N, m["ppl"], m["top1"]))

    m = evaluate(brain, corpus, res, args.win, n_batches=args.eval_batches, batch=64, proj=proj)
    uni = corpus.unigram_ppl()
    out = {"version": "grow-4.0.0", "mem": args.mem, "ctx_tau": args.ctx_tau,
           "n_res": args.n_res, "D_feat": D_feat, "D_raw": D_raw, "y": args.y, "N_readout": brain.N,
           "N_memory": args.n_res if res is not None else 0,
           "y0_over_y": brain.N / args.y, "born_total": brain.born_total,
           "dead_total": brain.dead_total, "tokens": steps * B,
           "ppl": m["ppl"], "top1": m["top1"], "top5": m["top5"], "diag": m["diag"],
           "unigram_ppl": uni, "vs_unigram": m["ppl"] / uni,
           "topology": brain.topology(), "backprop": False, "shared_linear_readout": False,
           "grad_enabled": bool(T.is_grad_enabled()),
           "seconds": round(time.time() - t0, 1),
           "gpu_peak_GB": round(T.cuda.max_memory_allocated() / 1e9, 2) if T.cuda.is_available() else 0}
    log(json.dumps(out, ensure_ascii=False))

    if args.sample:
        ids = [int(x) for x in args.sample.split(",") if x.strip()]
        gen = generate(brain, corpus, res, args.win, ids, n_new=args.sample_n, temp=0.9, topk=0, proj=proj)
        log("续写 token：%s" % gen)
        out["sample_prompt"] = ids
        out["sample_gen"] = gen

    if args.out:
        T.save({"brain": brain.snapshot(), "reservoir": {"n": args.n_res, "tau": args.ctx_tau}
                if res is not None else None, "result": out, "args": vars(args)}, args.out)
        log("大脑已保存 -> %s" % args.out)
    if args.curve_out:
        json.dump({"curve": curve, "result": out}, open(args.curve_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

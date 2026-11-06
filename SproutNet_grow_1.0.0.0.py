"""SproutNet grow 1.0.0 —— 从第一个神经元开始生长的脉冲网络语言模型

设计约束（用户给的，不可违背）
------------------------------
1. 无反向传播。全程 torch.no_grad()，没有任何 .backward()、没有梯度、没有优化器。
2. 无线性的读出层。输出不是 h→logits 的矩阵乘，而是"每个神经元各自预测一个码位，
   同一位的神经元群取平均"。聚合权重固定为 1/n，不可学、也没有参数。
3. 只用阈值单元（SNN 的离散时间抽象）。
4. 从 N=1 开始长，长到用户给的 y 或更低的 y0。

为什么这样能不靠 BP 学习
------------------------
每个神经元在出生时被分配一个**目标码位** b_i。它要预测的是"下一时刻第 b_i 位是否为 1"。
在 t+1 时刻，这个位的真值就出现在它的突触后（也就是它下游那个细胞的状态），
所以误差 (c_{b_i}(t+1) − h_i(t)) 对它是**局部可得**的 —— 这正对应文献 23 里
"每个神经元有自己的、带符号的教学信号",不需要任何误差向回传播。

更新是乘性的（文献 23/24）：w ← w · exp(η · e_i · pre_j)，数学上禁止变号（Dale 定律），
并按每个神经元固定的 L1 预算归一（文献 26 的两条承重假设之一）。

生长为什么会长出 S 形
--------------------
  出生 = g · N_mature · (1 − N/y) · 归一化的残余误差
  死亡 = d · N · (N/y)
平衡点：N*/y = g·err / (g·err + d)
  · 早期 err 高、N 小 → 出生 >> 死亡 → 快速上升
  · 后期 err 低、N 接近 y → 出生趋零、死亡上升 → 停在 y0 < y
**y0 < y 不是失败，是这个方程的解** —— 用户说的"y 还有富余"正好对应它。

用法
----
  python SproutNet_grow_1.0.0.py --tokens 3000000 --cap 262144 --device cuda
  python SproutNet_grow_1.0.0.py --eval-only brain.pt
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
    print("[grow]", *a, flush=True)


# ---------------------------------------------------------------- 数据与码

class Corpus:
    """token 流 + 稀疏二值码。码是固定的，不学习。"""

    def __init__(self, art=ART, device="cuda"):
        self.dev = T.device(device)
        with open(os.path.join(art, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.sm = int(self.meta["sm"])
        self.k = int(self.meta["k"])
        self.sigma = float(self.meta["sigma"])
        # token 流可能有几亿条：留在内存当 numpy，避免占满显存
        self.tokens_np = np.load(os.path.join(art, "tokens.npy")).astype(np.int64)
        self.tokens = T.from_numpy(self.tokens_np)
        code = np.load(os.path.join(art, "code.npy"))
        self.code = T.from_numpy(code.astype(np.float32)).to(self.dev)   # (V, SM) 0/1
        self.V = self.code.shape[0]
        # 码位频率：决定每个神经元的目标发放率，是数据的属性，不是手填常数
        self.bit_freq = self.code.mean(0).to(self.dev)                    # (SM,)
        self.n = int(self.tokens_np.size)

    def tok_at(self, idx, shift=0):
        """按位置取 token（CPU 索引后再搬到显卡，避免整条 token 流常驻显存）。"""
        i = idx.detach().cpu().numpy() if T.is_tensor(idx) else np.asarray(idx)
        return T.from_numpy(self.tokens_np[i + shift]).to(self.dev)

    def batch_codes(self, idx):
        """idx: (B,) 起始位置，返回 c_t 与 c_{t+1} 的 (B, SM)。"""
        return self.code[self.tok_at(idx)], self.code[self.tok_at(idx, 1)]


# ---------------------------------------------------------------- 大脑

class Brain:
    """全部状态都在这里；容量 cap 是用户给的 y（显存上限）。"""

    def __init__(self, corpus, cap, fi=64, fr=64, frac_inh=0.2, d0=10.0,
                 geo=True, inh=True, syn_k=2, budget=1.0, gamma=1.0, syn_beta=0.25, homeo=0.02, temp=0.3,
                 seed=0, device="cuda"):
        self.dev = T.device(device)
        self.sm = corpus.sm
        self.fi, self.fr = fi, fr
        self.cap = cap
        self.geo = geo
        self.use_inh = inh
        self.frac_inh = frac_inh
        self.d0 = d0
        self.syn_k = syn_k
        self.budget = budget
        self.gamma = gamma
        self.syn_beta = syn_beta
        self.homeo = homeo
        self.temp = temp
        self.g = T.Generator(device="cpu").manual_seed(seed + 7)
        self.rng = np.random.default_rng(seed + 13)

        z = lambda *s, dt=T.float32: T.zeros(*s, dtype=dt, device=self.dev)
        o = lambda *s: T.ones(*s, dtype=T.float32, device=self.dev)

        # ---- 单元（槽位固定；死亡的槽位 h 置零，不回收引用）----
        self.coord = z(cap, 2)
        self.born = z(cap, dt=T.int64)        # 出生序
        self.alive = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.is_inh = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.bit = T.full((cap,), -1, dtype=T.int64, device=self.dev)
        self.theta = z(cap)
        self.gain = o(cap)
        self.tau_f = o(cap)
        self.tau_s = o(cap)
        self.rate_tgt = z(cap)                   # 由目标码位的频率给出（稳态设定点）
        self.rate_ema = z(cap)
        self.acc_ema = z(cap)                    # 预测正确率 EMA，用于死亡判据
        self.u_mean = z(cap)                     # 输入电流的慢均值（每单元自己估）
        self.u_scale = o(cap) * 0.02             # 输入电流的慢波动幅度 = sigmoid 的温度
        self.mean_p = z(cap)                     # 长期平均发放率（只给阈值稳态用）
        self.age = z(cap)

        # ---- 突触 ----
        self.in_idx = T.zeros(cap, fi, dtype=T.int64, device=self.dev)
        self.w_in = z(cap, fi)
        self.rec_idx = T.zeros(cap, fr, dtype=T.int64, device=self.dev)
        self.w_rec = z(cap, fr)
        # 复杂突触：慢内部态（文献 25 Q10：容量 ~ 线性于突触数）
        self.u_in = z(cap, fi)
        self.u_rec = z(cap, fr)
        # 抑制性单元的符号：源是抑制性则贡献为负（Dale 定律，结构上保证）
        self.src_sign = o(cap)

        # ---- 前向状态（按 batch）----
        self.f_state = None
        self.s_state = None
        self.h = None
        self.h_prev = None

        self.N = 0                # 当前存活数
        self.n_slots = 0          # 用过的最深槽位
        self.birth_counter = 0
        self.events = 0
        self.dead_total = 0
        self.born_total = 0

    # ------------------------------------------------------------ 出生

    def _sample_rec_sources(self, col, k, only_older_of, coord):
        """几何接线：p ∝ exp(-d/d0)，且只接比自己老的单元（文献 05：新生优先接前辈）。"""
        m = only_older_of
        if m <= 0:
            return np.zeros(0, dtype=np.int64)
        cand = np.arange(m)
        if self.geo:
            d = np.linalg.norm(self.coord_np[:m] - coord, axis=1)
            p = np.exp(-d / self.d0)
            p[self.born_np[:m] >= self.birth_counter] = 0.0
            s = p.sum()
            if s <= 0:
                p = np.ones(m)
            p = p / p.sum()
            kk = min(k, int((p > 0).sum()))
            if kk <= 0:
                return np.zeros(0, dtype=np.int64)
            return self.rng.choice(m, size=kk, replace=False, p=p)
        return self.rng.choice(m, size=min(k, m), replace=False)

    def _birth(self, n_new, corpus):
        """一次出生事件：挑残余误差最大、覆盖最不足的码位，给它们添神经元。"""
        if self.n_slots >= self.cap * 0.98:
            self.compact()
        n_new = int(min(n_new, self.cap - self.n_slots))
        if n_new <= 0:
            return 0
        err, cnt = self.bit_err, self.bit_cnt
        score = (err + 1e-4) / T.sqrt((cnt + 1).float())      # 误差高、人少的位优先
        score = score.masked_fill(cnt <= 0, score.max() + 1.0)
        prio = torch_topk_idx(score, min(n_new, score.numel()))
        # 新单元的目标位：名额多于码位时循环覆盖（每轮都先补最欠覆盖的位）
        base = prio.cpu().numpy()
        reps = int(math.ceil(n_new / max(1, base.size)))
        targets = np.tile(base, reps)[:n_new]

        self.coord_np = self.coord.detach().cpu().numpy()
        self.born_np = self.born.detach().cpu().numpy()

        new_ids = []
        for t in range(n_new):
            col = self.n_slots
            self.n_slots += 1
            b = int(targets[t])
            parent_coord = self.coord_np[col - 1] if col > 0 else np.zeros(2, np.float32)
            c = parent_coord + self.rng.standard_normal(2).astype(np.float32) * (1.0 / max(1, math.sqrt(self.cap)))
            self.coord[col] = T.from_numpy(c).to(self.dev)
            self.born[col] = self.birth_counter
            self.birth_counter += 1

            # 细胞类型：抑制性按比例随机；抑制性单元也预测码位，但它们的作用是压同组
            self.is_inh[col] = bool(self.use_inh and (self.rng.random() < self.frac_inh))
            self.src_sign[col] = -1.0 if bool(self.is_inh[col]) else 1.0

            # 时间尺度：对数均匀，比例约 1:4.5（文献 25 Q14）
            tf = float(np.exp(self.rng.uniform(math.log(1.5), math.log(6.0))))
            self.tau_f[col] = tf
            self.tau_s[col] = tf * 4.5

            # 阈值：只给一个粗起点，真正的值由稳态自适应决定（见 homeostat）。
            # 理由：文献 22 说稳态设定点在"输入侧"，而输入电流的量级取决于扇入与码密度，
            # 手填一个绝对阈值必然与电流量级错配 —— R090 第一次冒烟就是这样：u≈0.013，θ≈1，
            # 神经元永不发放，残余误差停在"全零预测"的基线上，看着像"学了"其实没学。
            self.theta[col] = float(np.exp(self.rng.normal(0.0, 0.5))) * 0.05
            self.gain[col] = float(np.exp(self.rng.normal(0.0, 0.35)))

            # 目标发放率 = 目标码位的频率（稳态设定点在数据侧，不手填）
            f = float(corpus.bit_freq[b])
            self.rate_tgt[col] = min(0.3, max(1e-3, f))
            self.rate_ema[col] = 0.0
            # 新生儿不知道自己的本事：用同组现存单元的均值作先验，而不是一个固定低值。
            # （固定 0.5 会让每个新生儿一过 grace 就被判为最差并淘汰。）
            same = (self.bit[:col] == b) & self.alive[:col]
            self.acc_ema[col] = float(self.acc_ema[:col][same].mean()) if bool(same.any()) else 0.9
            self.u_mean[col] = 0.0
            self.u_scale[col] = float(self.u_scale[:col][self.alive[:col]].median()) if col > 0 else 0.02
            self.mean_p[col] = float(f)
            self.age[col] = 0.0

            # 输入突触：从码空间随机取 fi 位
            idx = self.rng.integers(0, self.sm, size=self.fi)
            self.in_idx[col] = T.from_numpy(idx.astype(np.int64)).to(self.dev)
            self.w_in[col] = T.from_numpy(
                (np.abs(self.rng.standard_normal(self.fi)) * 0.05).astype(np.float32)).to(self.dev)

            # 递归突触：只接更老的单元，几何衰减
            src = self._sample_rec_sources(col, self.fr, col, c)
            self.rec_idx[col] = 0
            if len(src) > 0:
                self.rec_idx[col, :len(src)] = T.from_numpy(src.astype(np.int64)).to(self.dev)
                self.w_rec[col, :len(src)] = T.from_numpy(
                    (np.abs(self.rng.standard_normal(len(src))) * 0.05).astype(np.float32)).to(self.dev)

            self.alive[col] = True
            self.bit[col] = b
            new_ids.append(col)

        self._normalize_rows(T.from_numpy(np.array(new_ids, dtype=np.int64)).to(self.dev))
        self.N += n_new
        self.born_total += n_new
        self._refresh_counts()
        return n_new

    def _normalize_rows(self, ids):
        """每个神经元 L1 预算恒定（零和）。这是承重约束，不是正则项。"""
        if ids.numel() == 0:
            return
        s = self.w_in[ids].abs().sum(1) + self.w_rec[ids].abs().sum(1) + 1e-9
        sc = self.budget / s
        self.w_in[ids] *= sc.unsqueeze(1)
        self.w_rec[ids] *= sc.unsqueeze(1)

    def _refresh_counts(self):
        alive = self.alive[:self.n_slots]
        self.alive_idx = alive.nonzero(as_tuple=True)[0]
        b = self.bit[:self.n_slots]
        cnt = T.zeros(self.sm, dtype=T.float32, device=self.dev)
        m = alive & (b >= 0)
        if m.any():
            cnt.index_add_(0, b[m], T.ones(int(m.sum()), device=self.dev))
        self.bit_cnt = cnt

    # ------------------------------------------------------------ 死亡

    def _death(self, n_die, grace):
        """按"预测正确率"淘汰。存活判据在文献 23 里是活动+同步，这里用可算的代理：
        这个单元对本位预测的准确率。比同组平均还差的，就是冗余。"""
        elig = self.alive[:self.n_slots] & (self.age[:self.n_slots] > grace)
        idx = elig.nonzero(as_tuple=True)[0]
        if idx.numel() <= 4:
            return 0
        n_die = int(min(n_die, idx.numel() - 2))
        if n_die <= 0:
            return 0
        acc = self.acc_ema[idx]
        # 同组平均：只有低于同组均值足够多的才杀，避免把整组杀光
        grp = self.bit[idx]
        gsum = T.zeros(self.sm, device=self.dev).index_add_(0, grp, acc)
        gcnt = T.zeros(self.sm, device=self.dev).index_add_(0, grp, T.ones_like(acc))
        gmean = gsum / gcnt.clamp(min=1)
        rel = acc - gmean[grp]
        # 某位只剩一个单元时不杀：否则这一位永远没有预测者，误差信号消失
        guard = self.bit_cnt[grp] > 1.5
        rel = rel.masked_fill(~guard, 1e9)
        order = rel.argsort()
        order = order[rel[order] < 1e8]
        n_die = min(n_die, int(order.numel()))
        if n_die <= 0:
            return 0
        dead = idx[order[:n_die]]
        self.alive[dead] = False
        self.bit[dead] = -1
        self.w_in[dead] = 0
        self.w_rec[dead] = 0
        self.u_in[dead] = 0
        self.u_rec[dead] = 0
        self.in_idx[dead] = 0
        self.rec_idx[dead] = 0
        self.N -= int(dead.numel())
        self.dead_total += int(dead.numel())
        self._refresh_counts()
        return int(dead.numel())

    def compact(self):
        """把存活单元压到前面，回收死亡槽位。

        不压缩的话 n_slots 顶到 cap 就再也生不出来
        （R090 实测：N 长到 cap 后出生恒为 0，只剩死亡，N 从 8192 一路掉到 2520）。
        """
        n = self.n_slots
        alive_full = self.alive.clone()          # 压缩前先留一份，下面要靠它判断指向
        alive = self.alive[:n].nonzero(as_tuple=True)[0]
        m = int(alive.numel())
        if m == n:
            return 0
        remap = T.zeros(self.cap, dtype=T.int64, device=self.dev)
        remap[alive] = T.arange(m, device=self.dev)
        for k in ("coord", "born", "alive", "is_inh", "bit", "theta", "gain", "tau_f", "tau_s",
                  "rate_tgt", "rate_ema", "acc_ema", "age", "src_sign",
                  "u_mean", "u_scale", "mean_p",
                  "in_idx", "w_in", "rec_idx", "w_rec", "u_in", "u_rec"):
            getattr(self, k)[:m] = getattr(self, k)[:n][alive]
        ridx = self.rec_idx[:m].clone()
        bad = ~alive_full[ridx]
        self.rec_idx[:m] = remap[ridx]
        self.w_rec[:m][bad] = 0.0
        getattr(self, "alive")[m:n] = False
        getattr(self, "is_inh")[m:n] = False
        self.bit[m:n] = -1
        self.n_slots = m
        self._refresh_counts()
        if self.h is not None and self.h.shape[1] != m:
            self.h = self.h[:, :m]
            self.h_prev = self.h_prev[:, :m]
            self.f_state = self.f_state[:, :m]
            self.s_state = self.s_state[:, :m]
        return n - m

    # ------------------------------------------------------------ 前向

    p = None

    def reset_state(self, B):
        self.f_state = T.zeros(B, self.n_slots, device=self.dev)
        self.s_state = T.zeros(B, self.n_slots, device=self.dev)
        self.h = T.zeros(B, self.n_slots, device=self.dev)
        self.h_prev = T.zeros(B, self.n_slots, device=self.dev)

    def forward(self, c):
        """c: (B, SM)。返回 pred: (B, SM)，即每个码位的预测概率（固定群平均）。"""
        n = self.n_slots
        B = c.shape[0]
        dev = self.dev
        if self.h is None or self.h.shape != (B, n):
            self.reset_state(B)

        # 输入电流（分块，避免 (B,N,fi) 爆显存）
        u = T.zeros(B, n, device=dev)
        CH = max(512, min(32768, 20_000_000 // max(1, B * self.fi)))
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.in_idx[c0:c1]                       # (C, fi)
            w = self.w_in[c0:c1]                           # (C, fi)
            g = c[:, ids.reshape(-1)].reshape(B, c1 - c0, self.fi)
            u[:, c0:c1] += (g * w.unsqueeze(0)).sum(-1)

        # 递归电流：符号由源单元决定（Dale）
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.rec_idx[c0:c1]                      # (C, fr)
            w = self.w_rec[c0:c1] * self.src_sign[ids]     # (C, fr)
            g = self.h_prev[:, ids.reshape(-1)].reshape(B, c1 - c0, self.fr)
            u[:, c0:c1] += (g * w.unsqueeze(0)).sum(-1)

        # 双时间尺度内部状态（每个单元自己的快/慢变量）
        tf = self.tau_f[:n].unsqueeze(0)
        ts = self.tau_s[:n].unsqueeze(0)
        self.f_state += (u - self.f_state) / tf
        self.s_state += (u - self.s_state) / ts

        # 每单元自己估计输入电流的均值与波动幅度，用它当 sigmoid 的温度。
        # 这样就不存在任何"绝对阈值/绝对温度"要手填 —— 对应 AIS 式增益（文献 22/25）。
        ubar = u.mean(0)
        self.u_mean[:n] = 0.98 * self.u_mean[:n] + 0.02 * ubar
        self.u_scale[:n] = 0.98 * self.u_scale[:n] + 0.02 * (u - ubar).abs().mean(0)
        drive = (self.f_state + self.gamma * self.s_state) * self.gain[:n].unsqueeze(0)
        th = self.theta[:n].unsqueeze(0)
        # 温度 = 相对阈值的一个比例。这不是平滑技巧，而是神经元本身的阈值噪声：
        # 没有它，阈值单元的发放率对 theta 是阶跃的，稳态控制必然极限环振荡
        # （R090 第一次跑就是这样：theta 在 1e-4 与 0.05 之间来回，PPL 在 7e4 到 2e6 之间摆）。
        if self.temp > 0:
            p = T.sigmoid((drive - th) / (self.u_scale[:n].unsqueeze(0) + 1e-9))
            h = (T.rand(p.shape, device=dev) < p).float()
        else:
            p = (drive > th).float()
            h = p
        h = h * self.alive[:n].float().unsqueeze(0)
        p = p * self.alive[:n].float().unsqueeze(0)
        self.h_prev = self.h
        self.h = h
        self.p = p

        # 输出：同一位的神经元群取平均（固定权重 1/n，无参数、不可学）。
        # 用发放概率的群平均而不是采样值：群很小时采样噪声会把 PPL 打飞。
        pred = T.zeros(B, self.sm, device=dev)
        mask = self.alive[:n] & (self.bit[:n] >= 0)
        if mask.any():
            sel = mask.nonzero(as_tuple=True)[0]
            pred.index_add_(1, self.bit[sel], p[:, sel])
            inv = (1.0 / self.bit_cnt.clamp(min=1)).unsqueeze(0)
            pred = pred * inv
        return pred

    # ------------------------------------------------------------ 局部学习

    def learn(self, target, lr=0.02, plastic_b=4):
        """target: (B, SM) 下一时刻的码。只用每个单元自己的误差，不回传。
        更新是乘性的 w *= exp(lr * e_i * pre_j)，并按 L1 预算归一。"""
        n = self.n_slots
        h = self.h
        B = h.shape[0]
        if B == 0:
            return
        nb = min(plastic_b, B)

        # 每个单元自己的误差（只取它在 t 时刻的发放 vs 它在 t+1 的目标位）
        bit = self.bit[:n]
        valid = self.alive[:n] & (bit >= 0)
        if not bool(valid.any()):
            return
        p = self.p[:nb]
        tgt = target[:nb].gather(1, bit.clamp(min=0).unsqueeze(0).expand(nb, n))  # (nb, n)
        # 局部 delta：误差 = 目标位 - 本单元此刻的发放概率。
        # 只用本单元自己的活动与它那一位的目标值，没有任何跨层回传 —— 不是反向传播。
        e = (tgt - p) * valid.float().unsqueeze(0)
        # 累计正确率（死亡判据）
        with T.no_grad():
            acc = (1.0 - (tgt - p).abs()).mean(0)
            self.acc_ema[:n] = 0.995 * self.acc_ema[:n] + 0.005 * acc
            self.rate_ema[:n] = 0.995 * self.rate_ema[:n] + 0.005 * h[:nb].mean(0)
            self.mean_p[:n] = 0.999 * self.mean_p[:n] + 0.001 * p.mean(0)

        CH = max(512, min(32768, 20_000_000 // max(1, nb * self.fi)))
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.in_idx[c0:c1]
            pre = self.code_t[:nb][:, ids.reshape(-1)].reshape(nb, c1 - c0, self.fi)
            g = T.einsum("bn,bnc->nc", e[:, c0:c1], pre) / nb
            self.w_in[c0:c1] *= T.exp(lr * g).clamp(0.5, 2.0)
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.rec_idx[c0:c1]
            pre = self.h_prev[:nb, ids.reshape(-1)].reshape(nb, c1 - c0, self.fr)
            g = T.einsum("bn,bnc->nc", e[:, c0:c1], pre) / nb
            self.w_rec[c0:c1] *= T.exp(lr * g).clamp(0.5, 2.0)

        # 稳态（阈值自适应）：活动高于目标就升阈，低于目标就降阈。
        # 这一步完全不看误差标签，只用自身发放率 —— 是"输入侧设定点"的最小实现。
        # 阈值只负责"长期发多少"，不参与"什么时候发"。时间常数比权重慢两个数量级，
        # 两者分工不重叠 —— R090 的单元测试里，θ 和 w 抢同一个 delta 目标时 5/16 个单元发散。
        if self.homeo > 0:
            mp = self.mean_p[:n]
            self.theta[:n] = (self.theta[:n] +
                              self.homeo * self.u_scale[:n] * (mp - self.rate_tgt[:n]))
            self.theta[:n].clamp_(-1e3, 1e3)

        # 复杂突触：快态缓慢渗入慢态
        if self.syn_k > 1:
            a = 0.01
            self.u_in.mul_(1 - a).add_(a * self.w_in)
            self.u_rec.mul_(1 - a).add_(a * self.w_rec)

        self._normalize_rows(self.alive_idx)

    # ------------------------------------------------------------ 统计与存取

    def snapshot(self):
        return {
            "cap": self.cap, "n_slots": self.n_slots, "N": self.N,
            "fi": self.fi, "fr": self.fr, "sm": self.sm,
            "geo": self.geo, "use_inh": self.use_inh, "frac_inh": self.frac_inh,
            "d0": self.d0, "syn_k": self.syn_k, "budget": self.budget,
            "gamma": self.gamma, "syn_beta": self.syn_beta,
            "birth_counter": self.birth_counter, "events": self.events,
            "born_total": self.born_total, "dead_total": self.dead_total,
            "alive": self.alive[:self.n_slots].cpu(),
            "is_inh": self.is_inh[:self.n_slots].cpu(),
            "bit": self.bit[:self.n_slots].cpu(),
            "coord": self.coord[:self.n_slots].cpu(),
            "born": self.born[:self.n_slots].cpu(),
            "theta": self.theta[:self.n_slots].cpu(),
            "gain": self.gain[:self.n_slots].cpu(),
            "tau_f": self.tau_f[:self.n_slots].cpu(),
            "tau_s": self.tau_s[:self.n_slots].cpu(),
            "rate_tgt": self.rate_tgt[:self.n_slots].cpu(),
            "rate_ema": self.rate_ema[:self.n_slots].cpu(),
            "acc_ema": self.acc_ema[:self.n_slots].cpu(),
            "u_mean": self.u_mean[:self.n_slots].cpu(),
            "u_scale": self.u_scale[:self.n_slots].cpu(),
            "mean_p": self.mean_p[:self.n_slots].cpu(),
            "in_idx": self.in_idx[:self.n_slots].cpu(),
            "w_in": self.w_in[:self.n_slots].cpu(),
            "rec_idx": self.rec_idx[:self.n_slots].cpu(),
            "w_rec": self.w_rec[:self.n_slots].cpu(),
            "u_in": self.u_in[:self.n_slots].cpu(),
            "u_rec": self.u_rec[:self.n_slots].cpu(),
        }

    @staticmethod
    def save(path, brain, extra=None):
        d = brain.snapshot()
        if extra:
            d.update(extra)
        T.save(d, path)

    def load_state(self, d):
        n = d["n_slots"]
        self.n_slots = n
        self.N = d["N"]
        self.birth_counter = d["birth_counter"]
        self.events = d["events"]
        self.born_total = d["born_total"]
        self.dead_total = d["dead_total"]
        for k in ("alive", "is_inh", "bit", "coord", "born", "theta", "gain", "tau_f", "tau_s",
                  "rate_tgt", "rate_ema", "acc_ema", "u_mean", "u_scale", "mean_p",
                  "in_idx", "w_in", "rec_idx", "w_rec",
                  "u_in", "u_rec"):
            getattr(self, k)[:n] = d[k].to(self.dev)
        self.src_sign[:n] = T.where(self.is_inh[:n], -1.0, 1.0)
        self._refresh_counts()


def torch_topk_idx(x, k):
    k = min(int(k), int(x.numel()))
    return x.topk(k).indices


# ---------------------------------------------------------------- 评估

class _KeepState:
    """评估会推进网络状态；训练中的探针必须在评估后原样恢复，否则结论不可靠。"""

    def __init__(self, brain):
        self.b = brain

    def __enter__(self):
        b = self.b
        self.saved = None
        if b.h is not None:
            self.saved = (b.h.clone(), b.h_prev.clone(), b.f_state.clone(), b.s_state.clone())
        return b

    def __exit__(self, *a):
        if self.saved is not None:
            b = self.b
            b.h, b.h_prev, b.f_state, b.s_state = self.saved
        return False


def evaluate(brain, corpus, n_batches=50, batch=16, seed=0):
    """核 PPL：p(v) ∝ exp(-||pred - code_v||^2 / 2σ^2)。这是**外部测量**，网络不参与。"""
    with _KeepState(brain):
        return _evaluate_inner(brain, corpus, n_batches, batch, seed)


def _evaluate_inner(brain, corpus, n_batches, batch, seed):
    """两个口径都报：
       (a) 伯努利似然：每个码位是一个独立伯努利，p(v) ∝ Π_b p_b^{c_vb}(1-p_b)^{1-c_vb}
           —— 这是这套"每个单元预测一位"的设计的**精确**似然，不需要核宽。
       (b) 固定核：文献 30 的口径，作为对照。核宽 σ 由码的最近邻距离中位数定。
       两个都是**外部测量**：网络本身从不计算它们。"""
    brain.reset_state(batch)
    C = corpus.code
    eps = 1e-4
    sig2 = corpus.sigma ** 2
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, corpus.n - 2, size=(n_batches, batch))
    nll_b = nll_k = 0.0
    top1 = top5 = 0
    bit_hit = bit_tot = 0
    bit_nll_sum = 0.0
    fire_rate_sum = pred_sum = 0.0
    cnt = 0
    ar = T.arange(batch, device=brain.dev)
    for bi in range(n_batches):
        idx = T.from_numpy(idxs[bi].astype(np.int64)).to(brain.dev)
        c, _ = corpus.batch_codes(idx)
        pred = brain.forward(c)
        tgt_code, _ = corpus.batch_codes(idx + 1)
        tgt = corpus.tok_at(idx, 1)
        p = pred.clamp(eps, 1.0 - eps)
        logit = T.log(p) - T.log1p(-p)
        sc = logit @ C.t()
        nll_b += float(-T.log_softmax(sc, 1)[ar, tgt].sum())
        sc_k = (pred @ C.t()) / sig2
        nll_k += float(-T.log_softmax(sc_k, 1)[ar, tgt].sum())
        fire_rate_sum += float(brain.p.float().mean()) * batch
        pred_sum += float(pred.mean()) * batch
        bit_hit += int((p > 0.5).long().eq(tgt_code.long()).sum())
        bit_nll_sum += float(-(tgt_code * T.log(p) +
                               (1 - tgt_code) * T.log(1 - p)).sum())
        bit_tot += tgt_code.numel()
        top1 += int((sc.argmax(1) == tgt).sum())
        top5 += int(sc.topk(5, 1).indices.eq(tgt.unsqueeze(1)).any(1).sum())
        cnt += batch
    return {
        "ppl_bernoulli": math.exp(nll_b / max(1, cnt)),
        "ppl_kernel": math.exp(nll_k / max(1, cnt)),
        "top1": top1 / max(1, cnt),
        "top5": top5 / max(1, cnt),
        "bit_acc": bit_hit / max(1, bit_tot),
        "bit_nll": bit_nll_sum / max(1, bit_tot),
        "fire_rate": fire_rate_sum / max(1, cnt),
        "pred_mean": pred_sum / max(1, cnt),
        "sigma": corpus.sigma,
        "batches": n_batches,
    }


def bit_marginal_entropy(corpus):
    """只按每个码位自身的边际频率预测时的位级交叉熵（nats/位）。
    这是"完全没有上下文"的零点；位级 NLL 低于它，才说明真的用上了上下文。"""
    eps = 1e-5
    f = corpus.bit_freq.clamp(eps, 1 - eps)
    return float(-(f * T.log(f) + (1 - f) * T.log(1 - f)).mean())


def marginal_baseline(corpus):
    """独立伯努利基线：只按每个码位自己的边际频率预测，没有任何上下文。
    这是"网络到底有没有学到东西"的正确零点（比 token unigram 更严）。"""
    eps = 1e-4
    p = corpus.bit_freq.clamp(eps, 1 - eps)
    logit = T.log(p) - T.log1p(-p)
    sc = (logit @ corpus.code.t()).unsqueeze(0)
    lp = T.log_softmax(sc, 1)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, corpus.n - 2, size=20000)
    tgt = corpus.tok_at(T.from_numpy(idx.astype(np.int64)), 1)
    return float(T.exp(-lp[0].gather(0, tgt).mean())), float((sc[0].argmax() == tgt).float().mean())


def unigram_baseline(corpus, tokens_sample=400000):
    ids = corpus.tokens_np[:tokens_sample]
    cnt = np.bincount(ids, minlength=corpus.V).astype(np.float64) + 1e-6
    p = cnt / cnt.sum()
    ll = np.log(p[ids])
    return float(np.exp(-ll.mean()))


# ---------------------------------------------------------------- 主流程

def run(args):
    T.set_num_threads(4)
    dev = args.device
    if dev == "cuda" and not T.cuda.is_available():
        raise RuntimeError("要求 CUDA 但不可用")
    corpus = Corpus(args.art, dev)
    log("语料 %d token，词表 %d，码 %d 维 / 每 token %d 位活跃，核宽 σ=%.3f" % (
        corpus.n, corpus.V, corpus.sm, corpus.k, corpus.sigma))

    brain = Brain(corpus, args.cap, fi=args.fi, fr=args.fr, frac_inh=args.frac_inh,
                  d0=args.d0, geo=bool(args.geo), inh=bool(args.inh), syn_k=args.syn_k,
                  budget=args.budget, gamma=args.gamma, homeo=args.homeo, temp=args.temp, seed=args.seed, device=dev)
    log("容量上限 y=%d（显存规划值），初始神经元=0，将从第 1 个开始长" % args.cap)
    assert not T.is_grad_enabled(), "红线：不许有梯度"

    # 先长一个种子单元，否则没有输入输出可算
    # 生长信号的初值：把"按边际频率预测"的对数损失作为基线，而不是随手写一个常数
    _bf = corpus.bit_freq.clamp(1e-5, 1 - 1e-5)
    brain.bit_err = -(corpus.bit_freq * T.log(_bf) +
                      (1 - corpus.bit_freq) * T.log(1 - _bf))
    brain.bit_cnt = T.zeros(brain.sm, device=brain.dev)
    brain._birth(1, corpus)
    log("第 1 个神经元已出生（目标码位 %d）" % int(brain.bit[0]))

    B = args.batch
    brain.reset_state(B)
    rng = np.random.default_rng(args.seed)
    curve = []
    t0 = time.time()
    steps = args.tokens // B
    grow_every = args.grow_every
    lr = args.lr

    err_ref = 0.0
    pos = rng.integers(0, corpus.n - (B + 2), size=B)
    for t in range(steps):
        idx = T.from_numpy(pos.astype(np.int64)).to(brain.dev)
        c, c_next = corpus.batch_codes(idx)
        brain.code_t = c
        pred = brain.forward(c)

        # 逐位残余误差（生长信号）
        eps = 1e-5
        err = -(c_next * T.log(pred.clamp(min=eps)) +
                (1 - c_next) * T.log((1 - pred).clamp(min=eps))).mean(0)
        brain.bit_err = 0.99 * brain.bit_err + 0.01 * err

        brain.learn(c_next, lr=lr, plastic_b=args.plastic_b)
        brain.age[:brain.n_slots] += 1.0
        pos += 1

        if t > 0 and t % grow_every == 0 and brain.events < args.max_events:
            brain.events += 1
            # 成熟数：只有长过 grace 的单元才能产生后代（文献 22 规则 1）
            mature = int((brain.age[:brain.n_slots] > args.grace).sum())
            if err_ref <= 0:
                err_ref = max(1e-6, float(brain.bit_err.mean()))
            err_norm = min(1.0, max(0.0, float(brain.bit_err.mean()) / err_ref))
            room = max(0.0, 1.0 - brain.N / brain.cap)
            n_birth = int(math.ceil(args.grow_rate * max(1, brain.N) * room * err_norm))
            n_birth = min(n_birth, args.max_birth_per_event)
            nb = brain._birth(n_birth, corpus) if n_birth > 0 else 0
            n_death = int(round(args.death_rate * brain.N * (brain.N / brain.cap)))
            nd = brain._death(n_death, args.grace)
            if brain.n_slots > 0 and brain.h is not None and brain.h.shape[1] != brain.n_slots:
                brain.reset_state(B)
            curve.append({"step": t, "N": brain.N, "born": nb, "died": nd,
                          "err": float(brain.bit_err.mean())})
            if brain.events % 10 == 0 or t < grow_every * 6:
                log("t=%6d 步 N=%6d (+%d/-%d) 残余误差=%.4f 用时%.0fs" % (
                    t, brain.N, nb, nd, float(brain.bit_err.mean()), time.time() - t0))

        if args.log_every and t > 0 and t % args.log_every == 0:
            m = evaluate(brain, corpus, n_batches=25, batch=B)
            log("t=%6d N=%6d PPL=%.1f 位NLL=%.4f(基线%.4f) 平均发放=%.3f 残差=%.4f" % (
                t, brain.N, m["ppl_bernoulli"], m["bit_nll"],
                bit_marginal_entropy(corpus), m["fire_rate"],
                float(brain.bit_err.mean())))

    log("训练结束：%d 步，N=%d，出生 %d，死亡 %d，用时 %.0fs" % (
        steps, brain.N, brain.born_total, brain.dead_total, time.time() - t0))

    m = evaluate(brain, corpus, n_batches=args.eval_batches, batch=B)
    uni = unigram_baseline(corpus)
    marg, marg_top1 = marginal_baseline(corpus)
    out = {
        "version": "grow-1.0.0", "cap_y": args.cap, "N": brain.N,
        "y0_over_y": brain.N / args.cap, "y_surplus": args.cap - brain.N,
        "born_total": brain.born_total, "dead_total": brain.dead_total,
        "steps": steps, "tokens": steps * B,
        "ppl_bernoulli": m["ppl_bernoulli"], "ppl_kernel": m["ppl_kernel"],
        "top1": m["top1"], "top5": m["top5"], "bit_acc": m["bit_acc"],
        "unigram_ppl": uni, "marginal_ppl": marg, "marginal_top1": marg_top1,
        "vs_marginal": m["ppl_bernoulli"] / marg,
        "bit_nll": m["bit_nll"], "bit_marginal_entropy": bit_marginal_entropy(corpus),
        "bit_nll_gain": bit_marginal_entropy(corpus) - m["bit_nll"],
        "backprop": False, "linear_readout": False,
        "grad_enabled": bool(T.is_grad_enabled()),
    }
    log(json.dumps(json.dumps(out, ensure_ascii=False), ensure_ascii=False))

    if args.out:
        Brain.save(args.out, brain, {"result": out, "curve": curve})
        log("大脑已保存 -> %s" % args.out)
    if args.curve_out:
        with open(args.curve_out, "w", encoding="utf-8") as f:
            json.dump({"curve": curve, "result": out}, f, ensure_ascii=False, indent=1)
        log("生长曲线 -> %s" % args.curve_out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--art", default=ART)
    p.add_argument("--cap", type=int, default=262144, help="y：神经元上限（由 plan_capacity.py 算）")
    p.add_argument("--fi", type=int, default=64)
    p.add_argument("--fr", type=int, default=64)
    p.add_argument("--frac-inh", type=float, default=0.2)
    p.add_argument("--inh", type=int, default=1)
    p.add_argument("--geo", type=int, default=1)
    p.add_argument("--d0", type=float, default=10.0)
    p.add_argument("--syn-k", type=int, default=2)
    p.add_argument("--budget", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--homeo", type=float, default=0.25, help="阈值边际校准速率，0=关")
    p.add_argument("--temp", type=float, default=0.3, help="发放温度（相对阈值）；0=确定阈值")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tokens", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--plastic-b", type=int, default=4)
    p.add_argument("--grow-every", type=int, default=200)
    p.add_argument("--grow-rate", type=float, default=1.0)
    p.add_argument("--death-rate", type=float, default=0.008)
    p.add_argument("--max-birth-per-event", type=int, default=200000)
    p.add_argument("--max-events", type=int, default=200)
    p.add_argument("--grace", type=float, default=450.0)
    p.add_argument("--err-ref", type=float, default=0.12)
    p.add_argument("--log-every", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=40)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--curve-out", default="")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()

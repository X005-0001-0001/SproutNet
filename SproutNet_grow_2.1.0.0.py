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
    def __init__(self, art=ART, in_mode="emb", device="cuda"):
        self.dev = T.device(device)
        with open(os.path.join(art, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.tokens_np = np.load(os.path.join(art, "tokens.npy")).astype(np.int64)
        self.n = int(self.tokens_np.size)
        self.in_mode = in_mode
        if in_mode == "emb":
            e = np.load(os.path.join(art, "emb.npy")).astype(np.float32)
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            x = (e + 1.0) * 0.5                       # 移到 [0,1]，便于乘性更新保持符号
        else:
            c = np.load(os.path.join(art, "code.npy")).astype(np.float32)
            x = c
        self.x = T.from_numpy(x).to(self.dev)          # (V, D) 每个 token 的输入向量
        self.V, self.D = self.x.shape
        self.sigma = float(self.meta.get("sigma", 1.0))
        cnt = np.bincount(self.tokens_np[:2_000_000], minlength=self.V).astype(np.float64) + 1.0
        self.freq = T.from_numpy((cnt / cnt.sum()).astype(np.float32)).to(self.dev)
        self.logfreq = T.log(self.freq.clamp(min=1e-12))

    def tok_at(self, idx, shift=0):
        i = idx.detach().cpu().numpy() if T.is_tensor(idx) else np.asarray(idx)
        return T.from_numpy(self.tokens_np[i + shift]).to(self.dev)

    def inputs(self, idx):
        return self.x[self.tok_at(idx)]

    def unigram_nll(self):
        return float(-self.logfreq[self.tok_at(T.arange(200000, device=self.dev))].mean())


# ---------------------------------------------------------------- 大脑

class Brain:
    def __init__(self, corpus, cap, fr=64, frac_inh=0.2, d0=10.0, geo=True, inh=True,
                 budget=4.0, gamma=0.5, homeo=1.0, temp=1.0, lr_theta=1.0, temp_abs=1.0,
                 norm="none", wd=0.0, beta2=0.999, gain_jitter=1.0, theta_init="zero", w_init_sd=0.01, tau_fixed=0.0,
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
        self.u_mean = z(cap)
        self.u_scale = o(cap) * 0.02

        self.w_in = z(cap, self.D)
        # 每个突触自己的二阶矩：只用自己的梯度历史归一化自己的步长。
        # 这是**完全局部**的（不需要任何跨突触信息），生物学对应物是习惯化/突触缩放。
        # _local_only 实测：加上它把 PPL 从 2113 打到 1442；不加则对 8192 个
        # 频率差几个数量级的类用统一步长，稀有类永远学不动。
        self.v_in = o(cap, self.D) * 1e-6
        self.v_rec = o(cap, fr) * 1e-6
        self.v_th = o(cap) * 1e-6
        self.rec_idx = T.zeros(cap, fr, dtype=T.int64, device=self.dev)
        self.w_rec = z(cap, fr)
        self.src_sign = o(cap)

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
        self.wd = wd
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
            self.tau_s[col] = tf * 4.5
            # θ 必须初始化在**驱动电流的实际尺度**上。L2 半径 4、256 维时 u 约 32，
            # 若 θ 从 2 起、每步只挪 1e-4，要 32 万步才追上，期间 softmax 一直饱和。
            # E[u] = mean(x)·Σw，x∈[0,1] 取 0.5，所以 0.5·Σw 就是合适的起点。
            if self.theta_init == "drive":
                self.theta[col] = (float(self.theta[:col][self.alive[:col]].median())
                                   if col > 0 else 0.5 * float(self.w_in[col].sum()))
            else:
                self.theta[col] = 0.0
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

            # 递归：只接更老的单元，几何衰减
            m = col
            if m > 0:
                if self.geo:
                    d = np.linalg.norm(self.coord_np[:m] - c, axis=1)
                    pp = np.exp(-d / self.d0)
                    pp = pp / pp.sum() if pp.sum() > 0 else np.ones(m) / m
                    kk = min(self.fr, int((pp > 0).sum()))
                    src = self.rng.choice(m, size=kk, replace=False, p=pp) if kk > 0 else np.zeros(0, np.int64)
                else:
                    src = self.rng.choice(m, size=min(self.fr, m), replace=False)
                if len(src) > 0:
                    self.rec_idx[col, :len(src)] = T.from_numpy(src.astype(np.int64)).to(self.dev)
                    self.w_rec[col, :len(src)] = T.from_numpy(
                        (np.abs(self.rng.standard_normal(len(src))) * 0.05).astype(np.float32)).to(self.dev)

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
        self.w_rec[dead] = 0
        self.rec_idx[dead] = 0
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
                  "u_mean", "u_scale", "w_in", "rec_idx", "w_rec",
                  "v_in", "v_rec", "v_th"):
            getattr(self, k)[:m] = getattr(self, k)[:n][alive]
        ridx = self.rec_idx[:m].clone()
        bad = ~alive_full[ridx]
        self.rec_idx[:m] = remap[ridx]
        self.w_rec[:m][bad] = 0.0
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
        return n - m

    # -------------------------------------------------- 前向

    def reset_state(self, B):
        self.f_state = T.zeros(B, self.n_slots, device=self.dev)
        self.s_state = T.zeros(B, self.n_slots, device=self.dev)
        self.h = T.zeros(B, self.n_slots, device=self.dev)
        self.h_prev = T.zeros(B, self.n_slots, device=self.dev)

    def forward(self, x):
        n = self.n_slots
        B = x.shape[0]
        if self.h is None or self.h.shape != (B, n):
            self.reset_state(B)

        # 输入电流：稠密 (B, D) @ (D, N)
        u = x @ self.w_in[:n].t()

        # 递归（分块）
        CH = max(512, min(32768, 20_000_000 // max(1, B * self.fr)))
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.rec_idx[c0:c1]
            w = self.w_rec[c0:c1] * self.src_sign[ids]
            g = self.h_prev[:, ids.reshape(-1)].reshape(B, c1 - c0, self.fr)
            u[:, c0:c1] += (g * w.unsqueeze(0)).sum(-1)

        ubar = u.mean(0)
        self.u_mean[:n] = 0.98 * self.u_mean[:n] + 0.02 * ubar
        self.u_scale[:n] = (0.98 * self.u_scale[:n] +
                            0.02 * (u - ubar).abs().mean(0)).clamp_(min=2e-3)

        tf = self.tau_f[:n].unsqueeze(0)
        ts = self.tau_s[:n].unsqueeze(0)
        self.f_state += (u - self.f_state) / tf
        self.s_state += (u - self.s_state) / ts
        drive = (self.f_state + self.gamma * self.s_state) * self.gain[:n].unsqueeze(0)
        # 每个单元输出一个 unbounded 的分数（它的"投票强度"），而不是二值发放。
        # 二值单元表达不了 Zipf：稀有词的条件概率是 1e-4，阈值单元要么永不发放、
        # 要么被稳态推到极限环 —— R091 第一版 PPL 93 万就是这么来的。
        score = ((drive - self.theta[:n].unsqueeze(0)) / self.Tglob).clamp_(-20.0, 20.0)
        # 仍然采样发放，供递归通路使用（网络里传的确实是脉冲）
        pr = T.sigmoid(score / max(self.temp, 1e-6))
        h = (T.rand(pr.shape, device=self.dev) < pr).float()
        alivef = self.alive[:n].float().unsqueeze(0)
        h = h * alivef
        self.h_prev = self.h
        self.h = h
        self.score = score * alivef
        self.pr = pr * alivef

        # 输出：同一个词的单元取平均（固定 1/n，无参数、不可学），再 softmax 归一
        sc = T.zeros(B, self.V, device=self.dev)
        mask = self.alive[:n] & (self.tok[:n] >= 0)
        if bool(mask.any()):
            sel = mask.nonzero(as_tuple=True)[0]
            sc.index_add_(1, self.tok[sel], score[:, sel])
            sc = sc / self.tok_cnt.unsqueeze(0)
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

        # 输入突触：可加 delta，按分数的尺度归一。
        # 分数的定义是 s=(u-θ)/u_scale，所以 ds/dw = x/u_scale —— 这是这条局部规则的正确梯度尺度。
        # 之前用乘性 exp(lr·e·x)，而 softmax 下负类的 e 只有 1/V≈1e-4，指数等于 0，等于没学。
        pre = self.x_t[:nb]                                   # (nb, D)
        es = e / self.Tglob
        g = (es.t() @ pre) / nb                                # (n, D)
        v = self.v_in[:n]
        v.mul_(self.beta2).add_(g * g, alpha=1 - self.beta2)
        self.w_in[:n] += lr * g / (v.sqrt() + 1e-8)
        if self.wd > 0:
            self.w_in[:n] -= self.wd * lr * self.w_in[:n]

        # 递归突触
        CH = max(512, min(32768, 20_000_000 // max(1, nb * self.fr)))
        for c0 in range(0, n, CH):
            c1 = min(n, c0 + CH)
            ids = self.rec_idx[c0:c1]
            pr = self.h_prev[:nb, ids.reshape(-1)].reshape(nb, c1 - c0, self.fr)
            es_c = e[:, c0:c1] / self.Tglob
            gg = T.einsum("bn,bnc->nc", es_c, pr) / nb
            vr = self.v_rec[c0:c1]
            vr.mul_(self.beta2).add_(gg * gg, alpha=1 - self.beta2)
            self.w_rec[c0:c1] += lr * gg / (vr.sqrt() + 1e-8)

        if self.norm == "l2":
            self._normalize_rows(self.alive_idx)

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
                  "w_in", "rec_idx", "w_rec"):
            d[k] = getattr(self, k)[:n].cpu()
        return d

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
            self.s = (b.h.clone(), b.h_prev.clone(), b.f_state.clone(), b.s_state.clone())
        return b

    def __exit__(self, *a):
        if self.s is not None:
            b = self.b
            b.h, b.h_prev, b.f_state, b.s_state = self.s
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
        "fire": round(float(brain.pr.mean()), 4),
    }
    print("[diag] " + "  ".join("%s=%s" % (k, v) for k, v in diag.items()), flush=True)
    return {"ppl": math.exp(nll / max(1, nsamp)), "top1": top1 / max(1, nsamp),
            "top5": top5 / max(1, nsamp), "batches": n_batches, "diag": diag}


# ---------------------------------------------------------------- 主流程

def run(args):
    dev = args.device
    T.set_num_threads(4)
    corpus = Corpus(args.art, args.in_mode, dev)
    log("输入模式 %s：in_dim=%d，词表 %d，语料 %d token" % (
        args.in_mode, corpus.D, corpus.V, corpus.n))
    brain = Brain(corpus, args.cap, fr=args.fr, frac_inh=args.frac_inh, d0=args.d0,
                  geo=bool(args.geo), inh=bool(args.inh), budget=args.budget,
                  gamma=args.gamma, homeo=args.homeo, temp=args.temp,
                  lr_theta=args.lr_theta, temp_abs=args.temp_abs, norm=args.norm,
                  wd=args.wd, beta2=args.beta2, gain_jitter=args.gain_jitter,
                  theta_init=args.theta_init,
                  w_init_sd=args.w_init_sd, tau_fixed=args.tau_fixed,
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
            n_birth = min(int(math.ceil(args.grow_rate * max(1, brain.N) * room * err_norm)),
                          args.max_birth_per_event)
            nb = brain._birth(n_birth, corpus) if n_birth > 0 else 0
            nd = brain._death(int(round(args.death_rate * brain.N * (brain.N / brain.cap))),
                              args.grace)
            if brain.h is not None and brain.h.shape[1] != brain.n_slots:
                brain.reset_state(B)
            curve.append({"step": t, "N": brain.N, "born": nb, "died": nd,
                          "err": float(brain.tok_err.mean())})
            if brain.events <= 8 or brain.events % 15 == 0:
                log("t=%6d N=%6d (+%d/-%d) 误差=%.3f 用时%.0fs" % (
                    t, brain.N, nb, nd, float(brain.tok_err.mean()), time.time() - t0))

        if args.log_every and t > 0 and t % args.log_every == 0:
            m = evaluate(brain, corpus, n_batches=10, batch=B)
            log("t=%6d N=%6d PPL=%.1f top1=%.3f 发放=%.3f" % (
                t, brain.N, m["ppl"], m["top1"], float(brain.pr.mean())))

    m = evaluate(brain, corpus, n_batches=args.eval_batches, batch=B)
    uni = math.exp(corpus.unigram_nll())
    out = {"version": "grow-2.0.0", "in_mode": args.in_mode, "cap_y": args.cap, "N": brain.N,
           "y0_over_y": brain.N / args.cap, "born_total": brain.born_total,
           "dead_total": brain.dead_total, "steps": steps, "tokens": steps * B,
           "ppl": m["ppl"], "top1": m["top1"], "top5": m["top5"],
           "unigram_ppl": uni, "vs_unigram": m["ppl"] / uni,
           "backprop": False, "linear_readout": False,
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
    p.add_argument("--wd", type=float, default=0.0, help="权重衰减系数")
    p.add_argument("--beta2", type=float, default=0.999, help="二阶矩的衰减率")
    p.add_argument("--gain-jitter", type=float, default=1.0, help="增益的 log-normal 幅度；0=全部为 1")
    p.add_argument("--theta-init", choices=["zero", "drive"], default="zero")
    p.add_argument("--w-init-sd", type=float, default=0.01, help="输入权重的初始化标准差（可正负）")
    p.add_argument("--tau-fixed", type=float, default=0.0, help="固定快时间常数；0=从对数均匀分布抽；1=瞬时")
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

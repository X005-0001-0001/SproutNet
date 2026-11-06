"""SproutNet grow 4.1.0 —— 群体发放率读出 + 固定临界储层 + 误差驱动生长

R103 的三个判决（全部零 bug、可复现，写进 MAP）
--------------------------------------------
① 码位边际 + 固定随机码本解码：**结构性损失 0.64 nats（1.90×）**。
   oracle trigram 条件分布本身 PPL 616.6，把它真实的位边际喂给码本解码 → 1174.2。
   原因：码位 = 随机半空间（约 102 个词），这种"任意集合"的隶属度不含词身份信息。
   → 那条出路判死。解码必须无损，即**语义恒等码本**：每个词自己的细胞群。

② 记忆不是瓶颈。固定临界储层（ρ=1，τ 可调）/ 显式 4 词窗口 / 前馈随机扩张（4096 维），
   同一 token 量下训练 PPL 309 / 307 / 315（τ=6 更差 509）→ 三者打平，且都到 ~300 就饱和。
   → 再堆记忆机制是白费；"上下文"要给的是**可手调且真的有效**的旋钮，不是新的记忆器官。

③ 3.x 的读出在数学上是**仿射的**：score_v = mean_i (drive_i − θ_i)，仿射的均值仍是仿射
   → N 个单元/词 ≡ 1 个单元/词。这解释了 N 从 512 到 16384 收益饱和、脉冲从不影响输出
   （旧存档留出集 fire=0.0002 而 PPL 不变）、约束A 必然崩 13×。
   → 必须给每个单元加**非线性再池化**，"多神经元"才第一次真的有用。

本版读出（唯一一处与 3.x 的本质区别）
-----------------------------------
  每个细胞 i：z_i = w_i·h + b_i,  σ_i = sigmoid(z_i) = 发放概率,  spike_i ~ Bernoulli(σ_i)
  每个词 v：m_v = (1/n_v) Σ_{i∈v} σ_i        ← 群体发放率（固定均值，零可学参数）
            score_v = logit(m_v)              ← 固定单调读出，零参数
  逐单元误差（精确链式法则，完全局部；单元只看自己的 z 和所属词的误差）：
      dL/dz_i = (p_v − y_v) / (m_v(1−m_v)) · σ_i(1−σ_i) / n_v
  初值 b_i = logit(freq_v) → t=0 时 m_v≈freq_v、score=logit(freq_v) → **精确等于一元分布**
  （所以"还没学会"= unigram，不是均匀分布；这也是生长信号的参照点。）

  这不是"可学线性层"：没有 h→logits 的共享投影；每个细胞有自己的突触、自己的阈值；
  误差按词下达、按细胞自己的 σ' 缩放，不需要任何跨单元回传。脉冲是真实的（Bernoulli），
  并且驱动可选的递归边 —— 它现在**在因果链上**。

身体结构
--------
  记忆器官（先天）   N_res 个固定临界递归单元；结构由生长规则在 t=0 生成，训练中不变；τ 可手调
  读出皮层（后天）   从 1 个细胞长到 y；误差驱动出生/死亡，S 形曲线，结构期重排
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch as T

T.set_grad_enabled(False)
hidden_learn = True
hid_teach = None      # (H,) 下行教学信号；None=用随机投影目标（learn 算出来前）
hid_teach_on = False
hid_teach_size = 0    # 隐藏层宽度 H（learn 需要它才能取最后 H 列）
dropout_p = 0.0       # 输入稀疏码的随机失活概率（突触释放失败）
add_scale = 1.0       # 隐藏层拼进读出时的缩放
hid_in_dim = 0        # >0 时隐藏层只看 h 的前这么多维（只看稀疏码）
hid_w_start = 0       # 读出输入里隐藏块从第几维开始（诊断读出层是否在用隐藏块）
cur_toks = None       # 当前 batch 的输入 token（B, win），供在线学码用
lag_dim = 0           # 近窗码占的前多少维（诊断读出层是否在用 lag 上下文）

ART = "_grow_art"


def log(*a):
    print("[g4]", *a, flush=True)


MEM_DEBUG = False


def _mem(tag):
    """仅 --mem-debug 时打印；不改变任何计算。"""
    if not MEM_DEBUG or not T.cuda.is_available():
        return
    T.cuda.synchronize()
    a = T.cuda.memory_allocated() / (1024 ** 3)
    r = T.cuda.memory_reserved() / (1024 ** 3)
    m = T.cuda.max_memory_allocated() / (1024 ** 3)
    log("[mem] %-14s alloc=%.2fG reserved=%.2fG peak=%.2fG" % (tag, a, r, m))


# ---------------------------------------------------------------- 数据

class Corpus:
    def __init__(self, art=ART, mem="res", win=4, device="cuda", dev_tokens=0):
        self.dev = T.device(device)
        with open(os.path.join(art, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.V = int(self.meta["vocab"])
        _all = np.load(os.path.join(art, "tokens.npy"), mmap_mode="r")
        _n = int(_all.size)
        _dev = int(dev_tokens)
        # dev 从训练语料**尾部**切，且永不参与训练；留出集(seg_0003)只用于最后报一次。
        # 用户裁决：超参不得在留出集上选（那等于拟合数据集）。所以必须有一个 dev。
        self.tr = _all[: max(1000, _n - _dev)]
        # 注意：self.dev 已经是 torch.device，dev 语料段必须叫 dev_tok，不能重名
        self.dev_tok = _all[max(1000, _n - _dev): _n] if _dev > 0 else None
        self.he = np.load(os.path.join(art, "heldout_tokens.npy"), mmap_mode="r")
        e = np.load(os.path.join(art, "emb.npy")).astype(np.float32)
        e /= (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        self.emb = T.from_numpy(e).to(self.dev)
        self.D0 = int(self.emb.shape[1])
        # 稀疏分布式码（本语料自训 PPMI-SVD → 固定随机投射 → k-WTA，51/4096 活跃）
        # 这是**输入编码**：数据驱动、无外部语料、二值稀疏（生物皮层/KC 的常态）。
        cp = getattr(self, "code_path", None) or os.path.join(art, "code.npy")
        self.code = None
        if os.path.exists(cp):
            c = np.load(cp).astype(np.float32)
            self.code = T.from_numpy(c).to(self.dev)
            self.SM = int(c.shape[1])
            self.k = int(c.sum(1).mean())
        self.in_code = False
        self.mem, self.win = mem, win
        cnt = np.bincount(np.asarray(self.tr[:4_000_000], dtype=np.int64),
                          minlength=self.V).astype(np.float64) + 1.0
        f = (cnt / cnt.sum()).astype(np.float32)
        self.freq = T.from_numpy(f).to(self.dev)
        # 词的先验对数几率：细胞初值用它 → t=0 精确等于 unigram
        self.logit_prior = torch_logit(f)

    def unigram_ppl(self, n=20000):
        idx = np.random.default_rng(0).integers(0, self.he.size - 2, n)
        t = T.from_numpy(np.asarray(self.he[idx], dtype=np.int64)).to(self.dev)
        return float(math.exp(float(-T.log(self.freq[t]).mean())))


def torch_logit(p_np, eps=1e-5):
    p = np.clip(p_np.astype(np.float64), eps, 1 - eps)
    return T.from_numpy(np.log(p / (1 - p)).astype(np.float32))


class Reservoir:
    """固定临界递归储层（不可学）。ρ=1 临界、稀疏 fan-in、泄漏 α=1/τ。"""

    def __init__(self, corpus, n, tau=4.0, fan=32, seed=20261104, device="cuda"):
        self.dev = T.device(device)
        self.N = int(n)
        self.alpha = 1.0 / max(1.0, float(tau))
        self.tau = float(tau)
        g = T.Generator().manual_seed(seed)
        D = corpus.D0
        self.W_in = (T.randn(self.N, D, generator=g) / math.sqrt(D)).to(self.dev)
        if self.N > 1:
            fan = min(int(fan), self.N)
            rows = T.arange(self.N).repeat_interleave(fan)
            cols = T.stack([T.randperm(self.N, generator=g)[:fan] for _ in range(self.N)]).reshape(-1)
            vals = T.randn(rows.numel(), generator=g) / math.sqrt(fan)
            W = T.sparse_coo_tensor(T.stack([rows, cols]), vals, (self.N, self.N)).coalesce().to(self.dev)
            v = T.randn(self.N, 1, generator=g).to(self.dev)
            for _ in range(30):
                v = T.sparse.mm(W, v)
                nv = float(v.norm())
                if nv < 1e-12:
                    break
                v = v / nv
            rho = float(T.sparse.mm(W, v).norm())
            self.W_rec = T.sparse_coo_tensor(W.indices(), W.values() / max(rho, 1e-6),
                                             (self.N, self.N)).coalesce()
        else:
            self.W_rec = None
        self.s = None

    def reset(self, B):
        self.s = T.zeros(B, self.N, device=self.dev)

    def step(self, x):
        if self.s is None or self.s.shape[0] != x.shape[0]:
            self.reset(x.shape[0])
        u = x @ self.W_in.t()
        if self.W_rec is not None:
            u = u + T.sparse.mm(self.W_rec, self.s.t()).t()
        self.s = (1 - self.alpha) * self.s + self.alpha * T.tanh(u)
        return self.s


# ---------------------------------------------------------------- 大脑

class Brain:
    def __init__(self, corpus, cap, D_feat=1024, fr=0, delay_steps=8, beta2=0.999,
                 w_init=1.0, eta_b=0.02, budget=4.0, rms_eps=1e-8,
                 plast_tau=0.0, half_feat=0, b_noise=0.1, eta_cal=0.01, upd="add", shy=0.0,
                 topk=0, topk_every=25, rfan=0, norm_score=0, unit_l1=0, readout="lse",
                 dend=0, dz_mode="cell", gate_lag=0, n_lag=1, lag_w=0, gate_full=0,
                 gate_soft=0, mod=0, grp=0, grp_pairs="0,1;0,2;1,2", grp_trace=0, trace_h=0,
                 eta_temp=0.0, code_learn=0, code_lr=0.0, mod_grow=0, mod_max=0, gate_ofs=0, plast_norm=0, freeze_readout=0, out_code=None, out_tb=None, out_rep=1.0, apical_mode="none", apical_norm="none", coord_sample=0, coord_mode="all", coord_groups=0, coord_swap=0, chunk_grad=0, rec_birth=0,
                 tier=None,
                 seed=0, device="cuda"):
        self.dev = T.device(device)
        self.V, self.D, self.cap = corpus.V, int(D_feat), int(cap)
        self.SM = int(getattr(corpus, "SM", self.D))
        self.apical_mode = str(apical_mode)
        self.chunk_grad = int(chunk_grad)
        self.rec_birth = int(rec_birth)
        self.apical_norm = str(apical_norm)
        self.coord_sample = int(coord_sample)
        self.coord_mode = str(coord_mode)
        self.coord_swap = int(coord_swap)
        self.h_exp = None
        self.ng = 1
        self._coord_diag_done = False
        self._perms = None
        if self.coord_sample:
            if self.D % self.SM != 0:
                raise SystemExit("--coord-sample 需要 D 能被 SM 整除（当前 D=%d SM=%d）" % (self.D, self.SM))
            self.nblk = self.D // self.SM
            _g = int(coord_groups)
            self.ng = _g if _g > 0 else max(1, self.cap // self.V)   # 0=由容量/词表导出
            log("[R168-C] 坐标采样：G=%d，mode=%s，每细胞仍看 D=%d 维，%d 个块"
                % (self.ng, self.coord_mode, self.D, self.nblk))
        self.u_last = None
        self._apical_diag_done = False
        self.fr, self.Delay = int(fr), max(1, int(delay_steps))
        self.fr_per = max(1, self.fr // max(1, self.Delay)) if self.fr else 0
        self.beta2 = beta2
        self.w_init = float(w_init)
        self.eta_b = float(eta_b)
        self.budget = float(budget)
        self.rms_eps = float(rms_eps)
        self.plast_tau = float(plast_tau)
        self.plast_norm = int(plast_norm)
        self.freeze_readout = int(freeze_readout)
        self.out_code = out_code
        # ---- 出生序 → 时间尺度（D1）----
        self.out_tb = out_tb
        # tier = {"V":词表, "T":档数, "K":带数, "bow":每带宽, "lagdim":窗口块总宽, "b":每档允许的带数}
        self.out_rep = float(out_rep)
        # 设计原则（哲学8/约束C）：
        #   · 档数 T = y // V 是**算出来的**（每个词分到几档），不是我填的；
        #   · 每档的时间视界 = ctx_len ** ((t+1)/T)，是把 ctx_len 对数等分 —— 与带通自身的
        #     对数等距 τ 同一把尺子，没有引入新常数；
        #   · 只挡"超出自己视界"的带通块（窗口块 τ≤win 永远在视界内），不挡别的。
        self.tier = tier or None
        self.tier_idx = None
        self.half_feat = int(half_feat)
        self.b_noise = float(b_noise)
        self.eta_cal = float(eta_cal)
        self.upd = str(upd)
        self.topk = int(topk)
        self.topk_every = int(max(1, topk_every))
        self._tct = 0
        self.rfan = int(rfan)
        self.hid_w_frac = 0.0
        self.w_frac_lag = 0.0
        self.norm_score = int(norm_score)
        self.unit_l1 = int(unit_l1)
        self.readout = str(readout)
        self.dend = int(dend)
        self.dz_mode = str(dz_mode)
        self.gate_lag = int(gate_lag)
        self.n_lag = max(1, int(n_lag))
        self.lag_w = int(lag_w)
        self.gate_full = int(gate_full)
        self.gate_soft = int(gate_soft)
        self.mod = int(mod)
        self.grp = int(grp)
        self.grp_k = float(getattr(corpus, "k", 51) or 51)
        self.eta_temp = float(eta_temp)
        self.mod_grow = int(mod_grow)
        self.mod_max = int(mod_max) if mod_max > 0 else int(max(1, mod))
        self._mg_every = 1
        self.code_learn = int(code_learn)
        if self.code_learn and getattr(corpus, "code", None) is not None:
            self.code = corpus.code                      # 直接引用：原地更新即生效
            self.v_code = T.full(self.code.shape, 1e-6, dtype=T.float32, device=self.dev)
            # k 由码自身的活跃位数决定（不新增常数）：取当前每词非零位数的中位数
            with T.no_grad():
                _nz = (self.code != 0).float().sum(1)
                self.code_k = int(max(1, _nz.median().item()))
            log("在线学码已启用：码 %s，每词保留 k=%d 位（由当前码的活跃位数导出），L2 归一" % (
                tuple(self.code.shape), self.code_k))
        else:
            self.code_learn = 0
        if self.code_learn:
            self.code_grad_ema = T.zeros_like(self.code)
            self.code_ema_decay = 0.9
            self.code_every = 200
            self._code_t = 0
            self.code_lr = float(code_lr)   # 0 = 沿用该步的读出 lr
        self._prod = None
        if self.grp > 0 and self.lag_w > 0 and self.grp <= self.lag_w:
            # 跨 lag 的段对（默认 (0,1),(0,2),(1,2)）
            self.gpairs = [tuple(int(x) for x in t.split(","))
                           for t in str(grp_pairs).split(";") if t.strip()]
            self.gpairs = [t for t in self.gpairs if t[1] < self.n_lag]
            self.theta = T.zeros(len(self.gpairs), cap, self.grp, dtype=T.float32, device=self.dev)
            self.v_th = T.full((len(self.gpairs), cap, self.grp), 1e-6, dtype=T.float32, device=self.dev)
            self.grp_trace = int(grp_trace)
            # 资格迹的时间常数：**不引入新常数**，直接取模型的窗口长度（"迹与上下文窗口同长"）
            _th = int(trace_h) if trace_h else max(1, int(n_lag and 0) or 1) * 0 + max(1, 4)
            self.trace_h = int(trace_h) if trace_h else 4
            self.rho = 1.0 - 1.0 / float(max(1, self.trace_h))
            # 资格迹只放在**特征侧**（每个 batch 流一份）：形状 (P, nb, G) ≈ 10 万个数。
            # 关键：必须是**分流**的。批内平均会把 1024 条无关流的共激活平均成一个常数
            # （≈ 边缘分布），迹就失去上下文特异性、退化成无效项。
            self.e_th = None
        self._m = None
        if self.mod > 0:
            g2 = T.Generator().manual_seed(seed + 13)
            # 共享模块 P: (M, D)，方差保持（每列 O(1/sqrt(D))）⇒ m 的量级 O(1)
            self.P = (T.randn(self.mod, self.D, generator=g2) / math.sqrt(self.D)).to(self.dev)
            # α = 1/M（a_v = 模块激活的均值），β = 0（乘积初值为 0，不打乱先验）
            self.al = T.full((cap, self.mod), 1.0 / self.mod, dtype=T.float32, device=self.dev)
            self.be = T.zeros((cap, self.mod), dtype=T.float32, device=self.dev)
            self.v_P = T.full((self.mod, self.D), 1e-6, dtype=T.float32, device=self.dev)
            self.v_al = T.full((cap, self.mod), 1e-6, dtype=T.float32, device=self.dev)
            self.v_be = T.full((cap, self.mod), 1e-6, dtype=T.float32, device=self.dev)
        if self.gate_lag and self.lag_w > 0:
            # 每个 lag 一组门权重。零初始化 ⇒ g=0.5（均匀缩放，被 w 吸收）。
            # gate_full=1 时门看**全部上下文**（(n_lag, D)）：能表达"lag1 是 X 就别信 lag2"这种
            # **跨 lag 的乘性结构**，而参数量仍只有 n_lag×D（几万），远小于 8192² 的交互权重。
            _gw = self.D if self.gate_full else self.lag_w
            self.gam = T.zeros(self.n_lag, _gw, device=self.dev)
            # 每细胞门偏置 δ（n_lag × cap）：**只有 laggate 读出用**。
            # 有了它，同词的不同细胞可以各自"信不同的 lag" ⇒ 分工。
            # 零初始化 ⇒ 门=0.5 均匀（被 w 吸收），不扰乱出生时的先验。
            self.gofs = T.zeros(self.n_lag, cap, device=self.dev)
            self.v_gofs = T.full((self.n_lag, cap), 1e-6, device=self.dev)
            # 每细胞的门控偏置：让同词的不同细胞可以各自"信不同的 lag"（分工的最小实现）。
            # 零初始化 ⇒ g=0.5 均匀（被 w 吸收），不扰乱出生时的先验；参数只多 n_lag×cap。
            self.gate_ofs = int(gate_ofs)
            if self.gate_ofs:
                self.gofs = T.zeros(self.n_lag, cap, device=self.dev)
                self.v_gofs = T.full((self.n_lag, cap), 1e-6, device=self.dev)
            self.v_gam = T.full((self.n_lag, _gw), 1e-6, device=self.dev)
            self._gl = None
        self._ua = None
        self._ub = None
        self.shy = float(shy)
        self.rng = np.random.default_rng(seed + 13)
        g = T.Generator(device="cpu").manual_seed(seed + 7)
        self.g = g

        z = lambda *s: T.zeros(*s, dtype=T.float32, device=self.dev)
        o = lambda *s: T.ones(*s, dtype=T.float32, device=self.dev)
        self.alive = T.zeros(cap, dtype=T.bool, device=self.dev)
        self.tok = T.full((cap,), -1, dtype=T.int64, device=self.dev)
        self.w = z(cap, self.D)
        self.b = z(cap)
        if self.dend:
            # 两个树突区室的突触（与 w 同维、同初始化尺度 ⇒ 出生时 (a·h)(b·h) 与 std(b) 同量级）
            self.wd1 = T.empty((cap, self.D), dtype=T.float32, device=self.dev)
            self.wd2 = T.empty((cap, self.D), dtype=T.float32, device=self.dev)
            self.v_d1 = T.full((cap, self.D), 1e-6, dtype=T.float32, device=self.dev)
            self.v_d2 = T.full((cap, self.D), 1e-6, dtype=T.float32, device=self.dev)
        # 单次分配：`o(cap,D) * 1e-6` 会先建一个全 1 的 (cap,D) 再乘，等于双份占用
        # （D=10240/cap=36864 时多 1.5GB，reserved 冲到 11.6GB → 大 y 容易卡死）
        self.v_w = T.full((cap, self.D), 1e-6, dtype=T.float32, device=self.dev)
        self.v_b = T.full((cap,), 1e-6, dtype=T.float32, device=self.dev)
        self.acc = z(cap) + 0.5
        self.rate = z(cap) + 0.05
        self.age = z(cap)
        self.sigma = z(cap)
        self.spike = z(cap)
        # 递归边（可选；由脉冲驱动，现在真的在因果链上）
        self.w_rec = z(cap, self.Delay, self.fr_per) if self.fr_per else None
        self.rec_src = T.zeros(cap, self.Delay, self.fr_per, dtype=T.int64, device=self.dev) if self.fr_per else None
        self.rec_mask = T.zeros(cap, self.Delay, self.fr_per, dtype=T.bool, device=self.dev) if self.fr_per else None
        self.v_rec = z(cap, self.Delay, self.fr_per) if self.fr_per else None
        self.rec_gain = o(cap) * 0.5
        self.n_edges = self.edge_born = self.edge_cut = 0
        self.spike_buf = None
        self.spk_ptr = 0
        self._ptr_before = 0

        self.n_slots = self.N = 0
        self.birth_counter = self.born_total = self.dead_total = self.events = 0
        self.tok_cnt = z(self.V)
        self.tok_cnt_raw = z(self.V)
        self.h = None
        self.sig_all = None
        self.m_v = None
        self.tok_err = z(self.V) + 7.0
        self.prior = corpus.freq.clone()          # 每词一个"先验虚拟细胞"
        self.b_logit = corpus.logit_prior.clone().to(self.dev)  # 先验对数几率（b_v），固定统计量
        self.cell_w = None
        self.gain_t = 1.0    # 全局读出增益（置信度匹配控制器）
        # 稀疏扇入：每个读出细胞只连特征里的一小部分（固定接线，不可学）。
        # 为什么：全连接时每个词只有 ~730 个样本却拿了 12288 个突触（过参数化 17 倍）→ 必然在线记忆。
        # 生物界一个神经元只有 ~10^4 个突触、连接是稀疏的，不是全连接。
        self.ridx = None
        if self.rfan > 0 and self.rfan < self.D:
            g = T.Generator().manual_seed(20261112)
            self.ridx = T.randint(0, self.D, (cap, self.rfan), generator=g).to(self.dev)
        self.n0 = 1.0                             # 虚拟细胞的等效个数（贝叶斯收缩，可由 CLI 覆盖）

    # -------------------------------------------- 生长 / 死亡
    def _draw_unit(self, v, corpus):
        col = self.n_slots
        if self.half_feat and self.D >= 4:
            # ③ 分工：单号细胞只看前半特征、双号只看后半（打破同词细胞的对称）
            hh = self.D // 2
            if col % 2 == 0:
                self.w[col, hh:] = 0.0
            else:
                self.w[col, :hh] = 0.0
        # 权重尺度：让 z 的标准差 ≈ w_init（|h|=1 时 z ≈ w·h）
        wv = self.rng.standard_normal(self.D).astype(np.float32) * (self.w_init / math.sqrt(self.D))
        self.w[col] = T.from_numpy(wv).to(self.dev)
        self.b[col] = float(corpus.logit_prior[v]) + float(self.rng.normal(0, self.b_noise))
        self.acc[col] = 0.5
        self.rate[col] = 0.5
        self.age[col] = 0.0
        self.alive[col] = True
        self.tok[col] = v

    def _birth(self, n_new, corpus, target="err"):
        """向量化出生：一次算完，不再逐细胞做小拷贝。

        为什么必须向量化：逐细胞 Python 循环在 y=65536 时会产生 65536×十几个小块 H2D 拷贝，
        把 caching allocator 打碎（实测显存 14GB、步速掉 4 倍）。R104 实测。
        """
        n_new = int(min(n_new, self.cap - self.n_slots - 1))
        if n_new <= 0:
            return 0
        if target == "rand":
            tg = self.rng.integers(0, self.V, size=n_new)
        else:
            score = (self.tok_err + 1e-3) / T.sqrt(self.tok_cnt_raw + 1.0)
            score = score.masked_fill(self.tok_cnt_raw <= 0, score.max() + 1.0)
            base = T.topk(score, min(n_new, self.V)).indices.cpu().numpy()
            reps = int(math.ceil(n_new / max(1, base.size)))
            tg = np.tile(base, reps)[:n_new]
        c0 = self.n_slots
        if self.tier is not None:
            # 出生序即身份（文献 约束C）：第 i 个槽位 → 词 = i % V，档 = i // V。
            # 这样**每一档都把整个词表覆盖一遍**，同一个词自然得到
            # "短上下文专家 + 长上下文专家"；不是每个词只配一个固定视界的细胞。
            tg = ((np.arange(c0, c0 + n_new)) % int(self.tier["V"])).astype(np.int64)
        # **分块出生**：一次翻倍到 32768 个细胞时，(n_new, D) 临时张量在 D=10240 下是 1.3GB，
        # 会把 caching allocator 打碎（实测 reserved 冲到 15.9GB 后卡死）。
        # 每块 4096 个 → 临时张量 ≤168MB，碎片消失。
        CH = 4096
        done = 0
        while done < n_new:
            m = int(min(CH, n_new - done))
            cols = T.arange(c0 + done, c0 + done + m, device=self.dev)
            tgt = T.from_numpy(np.ascontiguousarray(tg[done:done + m], dtype=np.int64)).to(self.dev)
            self.alive[cols] = True
            self.tok[cols] = tgt
            self.b[cols] = self.b_logit[tgt] + T.from_numpy(
                self.rng.normal(0.0, self.b_noise, size=m).astype(np.float32)).to(self.dev)
            self.acc[cols] = 0.5
            self.rate[cols] = 0.5
            self.age[cols] = 0.0
            wv = T.from_numpy(self.rng.standard_normal((m, self.D)).astype(np.float32)).to(self.dev)
            wv.mul_(self.w_init / math.sqrt(self.D))
            if self.half_feat and self.D >= 4:
                hh = self.D // 2
                ev = (cols % 2) == 0
                wv[ev, hh:] = 0.0
                wv[~ev, :hh] = 0.0
            if self.ridx is not None:
                _mm = T.zeros_like(wv)
                _mm.scatter_(1, self.ridx[cols], 1.0)
                wv.mul_(_mm)
            if self.tier is not None:
                # 把超出本档时间视界的带通块清零：这就是"功能分化"本身 ——
                # 早期出生（低档）只能看短 τ，晚期出生（高档）能看长 τ。
                # 只挡带通块；窗口块 τ≤win 永远在视界内，不动。
                _lav = int(self.tier["lagdim"])
                _bow = int(self.tier["bow"])
                _V = int(self.tier["V"])
                _sl = np.arange(c0 + done, c0 + done + m)
                _tt = np.minimum(_sl // _V, len(self.tier["b"]) - 1)
                for _t in np.unique(_tt):
                    _rows = np.where(_tt == _t)[0]
                    _b0 = int(self.tier["b"][int(_t)])
                    _cut = _lav + _b0 * _bow
                    if _cut < self.D:
                        wv[T.from_numpy(np.ascontiguousarray(_rows)).to(self.dev), _cut:] = 0.0
            if self.out_code is not None:
                # 输出矩阵 = 码表本身（固定、零可学参数）：第 v 个词那一行就是它自己的稀疏码
                # 按 tb 展开，再除以"每个码位有几个细胞投票"(H/SM)。不是手填尺度，是重复投票取平均。
                wv = (self.out_code[tgt].float()[:, self.out_tb] / self.out_rep)
            self.w[cols] = wv
            self.v_w[cols] = 1e-6
            if self.fr_per and self.rec_birth:
                # 出生即建 rec 边；源取自本批已出生/正在出生的细胞。
                _hi = c0 + done + m
                self.rec_src[cols] = T.randint(0, _hi, (m, self.Delay, self.fr_per), device=self.dev)
                self.rec_mask[cols] = True
                _ws = self.w_init / math.sqrt(self.D)
                self.w_rec[cols] = T.randn((m, self.Delay, self.fr_per), device=self.dev) * _ws
                self.n_edges += int(m * self.Delay * self.fr_per)
            if self.gate_lag and self.lag_w > 0:
                self.gofs[:, cols] = 0.0
                self.v_gofs[:, cols] = 1e-6
            if self.grp > 0 and getattr(self, "theta", None) is not None:
                self.theta[:, cols] = 0.0
                self.v_th[:, cols] = 1e-6
                # 迹在**流**维度上（不是细胞维度），出生不需要清它
                pass
            if self.mod > 0:
                self.al[cols] = 1.0 / self.mod
                self.be[cols] = 0.0
                self.v_al[cols] = 1e-6
                self.v_be[cols] = 1e-6
            if self.dend:
                # 出生时：区室1 复制基底突触，区室2 **全零** ⇒ 乘积项恰为 0（模型=先验，不引入噪声）。
                # 而零不会造成死锁：b 的梯度 = dz·(a·h)·h ≠ 0（a·h≠0），b 会自己长出来。
                # （曾试过两个区室都方差匹配初始化 → 随机二次型给每个词加 ~1 nat 噪声，PPL 4120 > unigram）
                self.wd1[cols] = wv
                self.wd2[cols] = 0.0
                self.v_d1[cols] = 1e-6
                self.v_d2[cols] = 1e-6
            done += m
        self.n_slots += n_new
        self.birth_counter += n_new
        self.N += n_new
        self.born_total += n_new
        self._refresh()
        return n_new

    def _refresh(self):
        alive = self.alive[:self.n_slots]
        cnt = T.zeros(self.V, device=self.dev)
        m = alive & (self.tok[:self.n_slots] >= 0)
        if bool(m.any()):
            t = self.tok[:self.n_slots][m]
            cnt.index_add_(0, t, T.ones(t.numel(), device=self.dev))
        self.tok_cnt_raw = cnt
        self.tok_cnt = cnt.clamp(min=1.0)

    def turnover(self, k, corpus):
        """成熟期的"死亡 + 新生"：把最差的一批细胞**就地重生**（同一生态位、新随机突触）。

        为什么需要它：`_death` 的守卫要求同词细胞数 >1.5 才允许淘汰，所以 per_word=1 时
        永远死不了 —— 那就不符合"一边长一边死、成熟后调整"的流程了。
        周转不改变 N（也不会让槽位溢出，compact 那条路有 bug 已默认关闭），
        做法是把最差的 k 个细胞重新初始化：新的随机突触、新的兴奋性、年龄归零。
        对应生物学：细胞凋亡 + 同一位置神经发生（新细胞重接进来）。
        """
        n = self.n_slots
        if k <= 0 or n == 0:
            return 0
        alive = self.alive[:n]
        idx = alive.nonzero(as_tuple=True)[0]
        if idx.numel() < 4:
            return 0
        k = int(min(k, idx.numel() // 4))
        acc = self.acc[idx]
        order = acc.argsort()                 # 最不准的先淘汰
        dead = idx[order[:k]]
        tg = self.tok[dead].clone()
        # 就地重生
        self.b[dead] = self.b_logit[tg] + T.from_numpy(
            self.rng.normal(0.0, self.b_noise, size=int(dead.numel())).astype(np.float32)).to(self.dev)
        wv = T.from_numpy(self.rng.standard_normal((int(dead.numel()), self.D)).astype(np.float32)).to(self.dev)
        wv.mul_(self.w_init / math.sqrt(self.D))
        if self.half_feat and self.D >= 4:
            hh = self.D // 2
            ev = (dead % 2) == 0
            wv[ev, hh:] = 0.0
            wv[~ev, :hh] = 0.0
        self.w[dead] = wv
        self.v_w[dead] = 1e-6
        self.acc[dead] = 0.5
        self.age[dead] = 0.0
        self.dead_total += k
        self.born_total += k
        self.edge_born += 0
        return k

    def _death(self, n_die, grace=200.0):
        elig = self.alive[:self.n_slots] & (self.age[:self.n_slots] > grace)
        idx = elig.nonzero(as_tuple=True)[0]
        if idx.numel() <= 4:
            return 0
        n_die = int(min(n_die, idx.numel() - 2))
        if n_die <= 0:
            return 0
        acc = self.acc[idx]
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
        self.w[dead] = 0
        self.b[dead] = 0
        if self.fr_per:
            self.n_edges -= int(self.rec_mask[dead].sum())
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
        keys = ["alive", "tok", "w", "b", "v_w", "v_b", "acc", "rate", "age", "sigma", "spike", "rec_gain"]
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
            self.reset_state(self.h.shape[0])
        return n - m

    # -------------------------------------------- 前向
    def reset_state(self, B):
        n = self.n_slots
        self.h = T.zeros(B, n, device=self.dev)
        self.sigma = T.zeros(B, n, device=self.dev)
        self.spike = T.zeros(B, n, device=self.dev)
        self.spike_buf = T.zeros(self.Delay + 1, B, n, device=self.dev) if self.fr_per else None
        self.spk_ptr = 0
        self._ptr_before = 0

    def _forward_laggate(self, h_feat):
        """逐 lag 的 per-cell 加权：u_i = Σ_l σ(γ_l·h_l + δ_{l,i}) · (h_l·w_{l,i})。"""
        n = self.n_slots
        B = h_feat.shape[0]
        _W = self.w[:n]
        if getattr(self, "gam", None) is not None:
            if self.gate_full:
                _zs = h_feat @ self.gam.t()
            else:
                _hl0 = h_feat[:, : self.n_lag * self.lag_w].view(B, self.n_lag, self.lag_w)
                _zs = T.einsum("blw,lw->bl", _hl0, self.gam)
        else:
            _zs = T.zeros((B, self.n_lag), device=self.dev)
        _fl, _gl = [], []
        u = self.b[:n].unsqueeze(0)
        for _l in range(self.n_lag):
            _hl = h_feat[:, _l * self.lag_w:(_l + 1) * self.lag_w]
            _f = _hl @ _W[:, _l * self.lag_w:(_l + 1) * self.lag_w].t()
            _g = T.sigmoid(_zs[:, _l].unsqueeze(1) + self.gofs[_l, :n].unsqueeze(0))
            u = u + _g * _f
            _fl.append(_f)
            _gl.append(_g)
        self._fl, self._gl2 = _fl, _gl
        mask = self.alive[:n]
        uu = u.masked_fill(~mask.unsqueeze(0), float("-inf"))
        tokn = self.tok[:n]
        idx = tokn.clamp(min=0).unsqueeze(0).expand(B, n)
        mx = T.full((B, self.V), float("-inf"), device=self.dev).scatter_reduce_(
            1, idx, uu, reduce="amax", include_self=True)
        M = T.maximum(mx, self.b_logit.unsqueeze(0))
        zc = uu - M.gather(1, idx)
        ex = T.exp(zc) * mask.float().unsqueeze(0)
        suma = T.zeros(B, self.V, device=self.dev).index_add_(1, tokn[mask], ex[:, mask])
        suma = suma + T.exp(self.b_logit.unsqueeze(0) - M)
        self.cell_w = (ex, suma, idx)
        return (M + T.log(suma)) * self.gain_t

    def _forward_mix(self, h_feat):
        """回退混合树：score_v = LSE_k( log λ_k + c_{k,v} ),  c_k = b + Σ_{l≤k} f_l。

        与加性打分的本质区别：它能在**多个上下文长度的分布之间做混合**（KN 的形式），
        因此可以表达"这个上下文只该信一元 / 该信到 lag2"。深度 k = 用到第 k 个 lag，
        即**深度是结构**（第 k 级就是第 k 个 lag），而不是手搭的层数。
        数值有界：LSE 的输入是有限分数、λ 归一 ⇒ 不需要任何增益或夹取。
        """
        n = self.n_slots
        B = h_feat.shape[0]
        # 门（= 级权重 λ 的 logits；没有门控时用均匀 λ）
        if self.gate_lag and getattr(self, "gam", None) is not None:
            if self.gate_full:
                _z = h_feat @ self.gam.t()
            else:
                _hl0 = h_feat[:, : self.n_lag * self.lag_w].view(B, self.n_lag, self.lag_w)
                _z = T.einsum("blw,lw->bl", _hl0, self.gam)
            if getattr(self, "gate_ofs", 0):
                _z = _z + self.gofs[:, :n].t()
            self._gl = T.sigmoid(_z) if not self.gate_soft else T.softmax(_z, dim=1)
            lam = self._gl / self._gl.sum(1, keepdim=True).clamp(min=T.finfo(T.float32).tiny)
        else:
            self._gl = None
            lam = T.full((B, self.n_lag), 1.0 / self.n_lag, device=self.dev)
        # 逐级累加 c_k
        _W = self.w[:n]
        _cum = self.b[:n].unsqueeze(0)
        _cs = []
        for _l in range(self.n_lag):
            _hl = h_feat[:, _l * self.lag_w:(_l + 1) * self.lag_w]
            if self._gl is not None:
                _hl = _hl * self._gl[:, _l].unsqueeze(1)
            _cum = _cum + _hl @ _W[:, _l * self.lag_w:(_l + 1) * self.lag_w].t()
            _cs.append(_cum)
        a = T.stack(_cs, 0) + T.log(lam.clamp(min=T.finfo(T.float32).tiny)).t().unsqueeze(2)
        # 按词聚合（同词细胞取 max，有界、无 epsilon）
        K = self.n_lag
        mask = self.alive[:n]
        tokn = self.tok[:n]
        idx = tokn.clamp(min=0).unsqueeze(0).expand(K * B, n)
        am = a.reshape(K * B, n).masked_fill(~mask.unsqueeze(0), float("-inf"))
        mxw = T.full((K * B, self.V), float("-inf"), device=self.dev).scatter_reduce_(
            1, idx, am, reduce="amax", include_self=True).view(K, B, self.V)
        Mk = mxw.amax(0)
        # 关键：没有细胞的词，`mxw` 全是 -inf，`-inf - (-inf) = NaN`。
        # 所以先把 -inf 的基准换成 0，再用 sw>0 判断哪些词有细胞；没有的停在先验 b_v。
        _ok = T.isfinite(Mk)
        Mk_s = T.where(_ok, Mk, T.zeros_like(Mk))
        ew = T.exp(mxw - Mk_s.unsqueeze(0))              # 未覆盖处 exp(-inf)=0
        sw = ew.sum(0)
        _sw_s = T.where(_ok, sw, T.ones_like(sw))
        self._lw = ew / _sw_s.unsqueeze(0)               # (K,B,V) 级权重（未覆盖处为 0）
        # 每个细胞的块权重 w̄_l = Σ_{k≥l} w_k（按该细胞自己的词取，再对 batch 平均）
        _idx2 = tokn.clamp(min=0).unsqueeze(0).unsqueeze(0).expand(K, B, n)
        _wk = self._lw.gather(2, _idx2)                  # (K,B,n)
        self._mix_scale = _wk.flip(0).cumsum(0).flip(0).mean(1)      # (K,n)
        self.cell_w = None
        sc_m = T.where(_ok, Mk_s + T.log(sw.clamp(min=T.finfo(T.float32).tiny)),
                       self.b_logit.unsqueeze(0))
        return sc_m * self.gain_t

    def forward(self, h_feat):
        n = self.n_slots
        B = h_feat.shape[0]
        if self.h is None or self.h.shape != (B, n):
            self.reset_state(B)
        if self.norm_score:
            self.gain_t = 1.0          # 归一化后不再需要任何全局增益（禁止"先炸再乘小"）
        self._ptr_before = self.spk_ptr
        self._mix_scale = None
        self._fl = None
        if self.readout == "mix" and self.n_lag > 1 and self.lag_w > 0:
            return self._forward_mix(h_feat)
        if self.readout == "laggate" and self.n_lag > 1 and self.lag_w > 0:
            return self._forward_laggate(h_feat)
        if self.gate_lag and self.lag_w > 0:
            # 每 lag 一个标量门（增益调制）：g_l = σ(γ_l·h_l)，然后把该 lag 的输入乘上它。
            # 这一步让分数可以表达"对这个上下文，只该信某个 lag" —— 纯加性模型做不到。
            _B = h_feat.shape[0]
            _hl = h_feat[:, : self.n_lag * self.lag_w].view(_B, self.n_lag, self.lag_w)
            if self.gate_full:
                _z = h_feat @ self.gam.t()                     # (B, n_lag)
            else:
                _z = T.einsum("blw,lw->bl", _hl, self.gam)     # (B, n_lag)
            if getattr(self, "gate_ofs", 0):
                _z = _z + self.gofs[:, :n].t()                 # (n,n_lag)->(B,n_lag) 每细胞自己的偏置
            if self.gate_soft:
                # lag 间 softmax：Σ_l g_l = 1 ⇒ "信任预算守恒"，门在 lag 之间**做选择**而不是各自加和。
                # 这正是 Kneser–Ney 混合（score = Σ_l λ_l f_l，Σλ=1）的形式；
                # 且 softmax 有界、可导，不会爆炸，也不需要任何温度常数。
                _g = T.softmax(_z, dim=1)
            else:
                _g = T.sigmoid(_z)
            self._gl = _g
            _hg = h_feat.clone()
            _hg[:, : self.n_lag * self.lag_w] = (_hl * _g.unsqueeze(2)).reshape(
                _B, self.n_lag * self.lag_w)
            h_feat = _hg
        if self.coord_sample and self.ng > 1:
            # R168-C：不同细胞组看不同排列的输入块。块级 roll 是确定性的、连续访存，
            # 不产生 (B,H,fan) 随机 gather，也不新增可学权重。
            _hb = h_feat.view(B, self.nblk, self.SM)
            _hs = []
            if self.coord_mode in ("rand", "randall") and self._perms is None:
                # self.rng 在 __init__ 后段才建立，所以这里惰性生成；仍用已有 rng，不引入新种子。
                # rand：第 0 组恒等，其余随机；randall：每组都独立随机。
                self._perms = ([np.arange(self.nblk)] if self.coord_mode == "rand" else [])
                _need = self.ng if self.coord_mode == "randall" else self.ng - 1
                for _ in range(_need):
                    self._perms.append(self.rng.permutation(self.nblk))
            for g in range(self.ng):
                _pg = (self.ng - 1 - g) if self.coord_swap else g
                if self.coord_mode == "lag":
                    _lag = T.roll(_hb[:, :self.n_lag], shifts=_pg, dims=1)
                    _hs.append(T.cat([_lag, _hb[:, self.n_lag:]], dim=1).reshape(B, self.D))
                elif self.coord_mode == "band":
                    _band = T.roll(_hb[:, self.n_lag:], shifts=_pg, dims=1)
                    _hs.append(T.cat([_hb[:, :self.n_lag], _band], dim=1).reshape(B, self.D))
                elif self.coord_mode in ("rand", "randall"):
                    _p = T.from_numpy(self._perms[_pg].astype(np.int64)).to(self.dev)
                    _hs.append(_hb[:, _p, :].reshape(B, self.D))
                else:
                    _hs.append(T.roll(_hb, shifts=_pg, dims=1).reshape(B, self.D))
            self.h_exp = T.cat(_hs, dim=1)
            if not self._coord_diag_done:
                print("[R168-C] G=%d mode=%s swap=%d 各组的输入块确实不同：|h0-h1|均值=%.4f，块数=%d"
                      % (self.ng, self.coord_mode, self.coord_swap,
                         float((self.h_exp[:, :self.D] - self.h_exp[:, self.D:2 * self.D]).abs().mean()),
                         self.nblk), flush=True)
                self._coord_diag_done = True
            u = T.empty((B, n), device=self.dev, dtype=h_feat.dtype)
            for _g in range(self.ng):
                _s = (_g * n) // self.ng
                _e = ((_g + 1) * n) // self.ng
                u[:, _s:_e] = self.h_exp[:, _g * self.D:(_g + 1) * self.D] @ self.w[_s:_e].t()
            u = u + self.b[:n].unsqueeze(0)
        else:
            self.h_exp = None
            u = h_feat @ self.w[:n].t() + self.b[:n].unsqueeze(0)
        if self.grp > 0 and getattr(self, "theta", None) is not None:
            # 段级跨 lag 重合：s_{l,g} ∈[0,1]，prod_{ab,g} = s_a,g · s_b,g ∈[0,1]
            _Bg = h_feat.shape[0]
            _wg = self.lag_w // self.grp
            _hl2 = h_feat[:, : self.n_lag * self.lag_w].view(_Bg, self.n_lag, self.grp, _wg)
            # 按**期望段和**归一：每段平均有 k/G 个码位活跃（k = 码的活跃位数，是数据量、非常数），
            # 归一后典型段和 = 1 ⇒ prod 是 O(1)，与加性项同量级。
            # （曾按段长 wg 归一 → 段和 ≈1.6/32=0.05、prod≈0.0025，比加性项小 400 倍，交互等于没有。）
            _ss = _hl2.sum(3) / max(1e-12, self.grp_k / float(self.grp))   # (B, n_lag, G)
            _prods = []
            for _pi, (_pa, _pb) in enumerate(self.gpairs):
                _prods.append(_ss[:, _pa, :] * _ss[:, _pb, :])   # (B, G)
            _pr = T.stack(_prods, 0)                             # (P, B, G)
            self._prod = _pr
            # u_v += Σ_{p,g} θ_{v,p,g} prod_{p,g}
            for _pi in range(_pr.shape[0]):
                u = u + _pr[_pi] @ self.theta[_pi, :n].t()
        if self.mod > 0:
            # 共享模块张成的交互：m = h·P^T；a_v = α_v·m；b_v = β_v·m；u += a_v·b_v
            self._m = h_feat @ self.P.t()                      # (B, M)
            _a = self.al[:n] @ self._m.t()                      # (n, B)
            _b = self.be[:n] @ self._m.t()                      # (n, B)
            u = u + (_a * _b).t()
        if self.dend:
            # 两个树突区室各自求和，再在胞体相乘（重合检测）。
            # 展开 = Σ_ij a_i b_j h_i h_j ⇒ 能表达"lag1 的 u × lag2 的 w"这类跨 lag 乘积，
            # 而这是加性读出（f0+f1+f2…）恒等表达不出的（R142）。
            self._ua = h_feat @ self.wd1[:n].t()
            self._ub = h_feat @ self.wd2[:n].t()
            u = u + self._ua * self._ub
        if self.norm_score == 1:
            # 突触缩放（Turrigiano）：只归一 w·h 这一项，b 保持绝对（词先验锚不该被缩放）。
            _wn = self.w[:n].norm(dim=1).unsqueeze(0)
            u = (h_feat @ self.w[:n].t()) / _wn.where(_wn > 0, T.ones_like(_wn)) \
                + self.b[:n].unsqueeze(0)
        elif self.norm_score >= 2:
            # 除法归一化（Carandini–Heeger）：除以**群体驱动的标准差**。
            # u 整体放大 c 倍 → sd 也放大 c 倍 ⇒ 归一化后的驱动对 |w| 尺度完全不变，
            # 所以不需要任何全局增益，也不存在"先炸再乘小"。sd=0（所有细胞相同，无信息）时令其为 0，无 epsilon。
            #
            # 刻度：把归一化后的驱动乘到 **std(b)**（先验 b 在词表上的标准差，是可测的数据量）。
            # 自洽含义："上下文证据的跨度 = 先验自身的跨度" ⇒ 上下文可以完全推翻先验。
            # 若只归一到 std=1，先验会盖过证据，模型退化为一元模型（PPL ≈ unigram）。
            _al = self.alive[:n]
            if bool(_al.any()):
                _ua = u[:, _al]
                _mu = _ua.mean(dim=1, keepdim=True)
                _sd = _ua.std(dim=1, keepdim=True, unbiased=False)
                _scale = float(self.b_logit.std()) if self.b_logit.numel() > 1 else 1.0
                _ok = _sd > 0
                _z = (u - _mu) / _sd.where(_ok, T.ones_like(_sd))
                u = T.where(_ok, _z * _scale, T.zeros_like(u)) + self.b[:n].unsqueeze(0)
        if self.fr_per and self.n_edges > 0:
            u_rec = T.zeros(B, n, device=self.dev)
            p = self.spk_ptr
            for k in range(self.Delay):
                slot = (p - 1 - k) % (self.Delay + 1)
                ids = self.rec_src[:n, k, :]
                wgt = self.w_rec[:n, k, :] * self.rec_mask[:n, k, :].float()
                g_ = self.spike_buf[slot][:, ids.reshape(-1)].reshape(B, n, self.fr_per)
                u_rec += (g_ * wgt.unsqueeze(0)).sum(-1)
            u = u + self.rec_gain[:n].unsqueeze(0) * u_rec
        sig = T.sigmoid(u)
        alivef = self.alive[:n].float().unsqueeze(0)
        sig = sig * alivef
        self.sigma = sig
        # 脉冲：Bernoulli(σ)。率读出用 σ；脉冲驱动递归边（真在因果链上）
        spike = (T.rand(sig.shape, device=self.dev) < sig).float() * alivef
        self.h = spike
        self.spike = spike
        if self.fr_per:
            self.spike_buf[self.spk_ptr] = spike
            self.spk_ptr = (self.spk_ptr + 1) % (self.Delay + 1)
        # ---- 固定读出：对每词细胞做 LSE（soft-max）池化 ----
        # score_v = log( (Σ_{i∈v} e^{z_i} + n0·e^{b_v}) / (n_v + n0) )
        #   生物学对应：**首脉冲延迟 / rank-order 编码**（Thorpe）——哪个细胞驱动最强，
        #   它的延迟最短，词就归它。LSE 是"最先发放者胜出"的光滑版。
        # 为什么不是 σ 率池化：R103 实测 dscore/dσ = 1/(m(1−m)) 在罕见词上 ~1e4，
        # 敏感度爆炸（PPL 2736→14718）；LSE 输出就在对数几率单位上，尺度天然正确。
        # 先验虚拟细胞（z 恒 = b_v = logit(freq_v)）保证：没细胞的词停在先验上（一元），
        # t=0 时所有 z_i = b_v → score = b_v → **精确等于 unigram**。
        # ---------------------------------------------------------------- 读出
        # --readout poisson（默认，首脉冲/泊松率叠加）：
        #     λ_v = Σ_{i∈v} e^{u_i}  +  e^{b_v}          （同词细胞率 + 一个虚拟细胞的率）
        #     score_v = log λ_v                          （= −log 到达时刻，最先到阈值者胜）
        #   · 虚拟细胞**恰好一个**：`e^{b_v}` 的权重恒为 1 ⇒ 原来的平滑常数 n0 消失（不需要调）。
        #   · 数值：取 M_v = max(max_i u_i, b_v)，则所有指数 ≤ 0 ⇒ exp ≤ 1，
        #     和 ≤ n_v + 1，**不可能溢出**，因此不需要 clamp、也不需要 -1e30 哨兵。
        #   · 有界：u 被 RMS 球约束在 ±√(活跃输入数) ⇒ score ≤ 上界，**永不可能爆炸**。
        #   · 稀疏性：这同时就是"同词细胞的脉冲汇到一个桶、谁先倒空谁胜"的精确形式。
        mask = self.alive[:n]
        uu = u.masked_fill(~mask.unsqueeze(0), float("-inf"))
        tokn = self.tok[:n]
        idx = tokn.clamp(min=0).unsqueeze(0).expand(B, n)
        mx = T.full((B, self.V), float("-inf"), device=self.dev).scatter_reduce_(
            1, idx, uu, reduce="amax", include_self=True)
        M = T.maximum(mx, self.b_logit.unsqueeze(0))     # 至少含虚拟细胞自身的率
        zc = uu - M.gather(1, idx)                        # ≤ 0，exp ≤ 1
        ex = T.exp(zc) * mask.float().unsqueeze(0)        # 死细胞贡献 0（exp(-inf)=0）
        suma = T.zeros(B, self.V, device=self.dev).index_add_(
            1, tokn[mask], ex[:, mask])
        if self.readout != "linear":
            # 虚拟先验细胞（只在 lse/poisson 里用）
            suma = suma + T.exp(self.b_logit.unsqueeze(0) - M)
        self.cell_w = (ex, suma, idx)
        self.u_last = u.detach()          # 顶端教学流要用：胞体电位（不参与梯度）
        if self.readout == "linear":
            # 纯 softmax：score_v = log Σ_{i∈v} e^{u_i}（单词/词时恰为 u_v = w_v·h + b_v）
            # 为什么不加虚拟先验细胞：词表已 100% 覆盖（words_uncovered=0），而 `+e^{b_v}` 会让
            # dz = (p−y)·e^{u}/λ 对"模型认为不可能的正确词"趋近 0（softmax 梯度消失），
            # 这正是模型只达到 bigram 水平的原因之一。
            sc = T.where(suma > 0, M + T.log(suma.clamp(min=T.finfo(T.float32).tiny)),
                         self.b_logit.unsqueeze(0))
        elif self.readout == "lse":
            # 旧形式（为复现历史数字保留）：按 (n_v + n0) 归一，n0 是手填的平滑常数。
            sc = M + T.log((suma / (self.tok_cnt + self.n0).unsqueeze(0)))
        else:
            # 泊松率叠加：不除以计数（log 之和本身就是正确的量纲），也不需要 n0。
            sc = M + T.log(suma)
        return sc * self.gain_t

    # -------------------------------------------- 局部学习
    def _mod_grow_step(self):
        """模块数由数据决定：死亡（无细胞认领）+ 分裂（误差最高的模块，排位判据）。"""
        n = self.n_slots
        if self.mod <= 0 or n < 64 or self.be is None:
            return
        with T.no_grad():
            be = self.be[:n].abs()
            prim = be.argmax(1)                                  # (n,) 每个细胞的主模块
            err = self.tok_err[self.tok[:n].clamp(min=0)]         # (n,) 该细胞的词级误差
            cnt = T.zeros(self.mod, device=self.dev).index_add_(
                0, prim, T.ones(n, device=self.dev))
            # ---- 死亡：没有细胞认领的模块直接删掉（零阈值）
            keep = (cnt > 0).nonzero(as_tuple=True)[0]
            if keep.numel() >= 2 and keep.numel() < self.mod:
                self.P = self.P[keep].contiguous()
                self.al = self.al[:, keep].contiguous()
                self.be = self.be[:, keep].contiguous()
                self.v_P = self.v_P[keep].contiguous()
                self.v_al = self.v_al[:, keep].contiguous()
                self.v_be = self.v_be[:, keep].contiguous()
                self.mod = int(keep.numel())
                log("[mod-grow] 死亡 → P=%d al=%d be=%d mod=%d" % (
                    self.P.shape[0], self.al.shape[1], self.be.shape[1], self.mod))
                cnt = cnt[keep]
                prim = self.be[:n].abs().argmax(1)
            # ---- 分裂：误差最高的模块（排位判据）；新模块用"随机插值"生成，无尺度常数
            if self.mod < max(self.mod + 1, self.mod_max):
                esum = T.zeros(self.mod, device=self.dev).index_add_(0, prim, err)
                emean = esum / cnt.clamp(min=1)
                if float(emean.max()) > float(err.mean()):
                    m = int(emean.argmax())
                    k = int(T.randint(0, self.mod, (1,), device=self.dev))
                    u = float(T.rand(1, device=self.dev))
                    newP = (self.P[m] + u * (self.P[k] - self.P[m])).unsqueeze(0)
                    self.P = T.cat([self.P, newP], 0)
                    self.v_P = T.cat([self.v_P, T.full((1, self.D), 1e-6, device=self.dev)], 0)
                    self.al = T.cat([self.al, (self.al[:, m] + u * (self.al[:, k] - self.al[:, m])
                                               ).unsqueeze(1)], 1)
                    self.be = T.cat([self.be, (self.be[:, m] + u * (self.be[:, k] - self.be[:, m])
                                               ).unsqueeze(1)], 1)
                    self.v_al = T.cat([self.v_al, T.full((self.cap, 1), 1e-6, device=self.dev)], 1)
                    self.v_be = T.cat([self.v_be, T.full((self.cap, 1), 1e-6, device=self.dev)], 1)
                    self.mod = int(self.P.shape[0])
                    self._mod_births = getattr(self, "_mod_births", 0) + 1
                    log("[mod-grow] 分裂 → P=%d al=%d be=%d mod=%d" % (
                        self.P.shape[0], self.al.shape[1], self.be.shape[1], self.mod))

    def _chunkable(self):
        return (self.chunk_grad and self.upd == "add" and not self.freeze_readout
                and self.plast_tau <= 0 and self.tier is None and not self.dend
                and self.ridx is None and self.mod <= 0 and self._fl is None
                and self._mix_scale is None and self.unit_l1 <= 0
                and self.topk <= 0 and self.norm_score != 1 and self.apical_mode == "none")

    def _apply_chunked(self, dz, tgt, nb, lr):
        """按行分块更新 w/v_w；与整块 gW 路径数学等价，仅降低临时显存。"""
        n = self.n_slots
        h = self.h_feat[:nb]
        chunk = max(1, (8 * 1024 * 1024) // max(1, self.D))   # 8M float ≈ 32MB/块
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            if self.coord_sample and self.ng > 1 and self.h_exp is not None:
                # 坐标采样时每个细胞组看的是 h_exp 的不同块，梯度必须按同一分组取。
                g = T.empty((e - s, self.D), device=self.dev, dtype=h.dtype)
                for _g in range(self.ng):
                    _gs = (_g * n) // self.ng
                    _ge = ((_g + 1) * n) // self.ng
                    _lo, _hi = max(s, _gs), min(e, _ge)
                    if _lo < _hi:
                        g[_lo - s:_hi - s] = (dz[:, _lo:_hi].t()
                                              @ self.h_exp[:nb, _g * self.D:(_g + 1) * self.D])
            else:
                g = dz[:, s:e].t() @ h             # (e-s, D)
            g.div_(nb)
            vw = self.v_w[s:e]
            vw.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            den = vw.sqrt()
            den.add_(self.rms_eps)
            g.div_(den)
            self.w[s:e].add_(g, alpha=-lr)
            if self.shy > 0:
                self.w[s:e].mul_(1.0 - self.shy)
            if self.budget > 0:
                nrm = self.w[s:e].pow(2).sum(1).sqrt()
                sc = (self.budget / nrm.where(nrm > 0, T.ones_like(nrm))).clamp(max=1.0)
                self.w[s:e] *= sc.unsqueeze(1)
            del g, vw, den
        with T.no_grad():
            hit = (tgt[:nb].unsqueeze(1) == self.tok[:n].unsqueeze(0)).float()
            sig = self.sigma[:nb]
            acc = 1.0 - (sig - hit).abs()
            self.acc[:n] = 0.99 * self.acc[:n] + 0.01 * acc.mean(0)
            self.rate[:n] = 0.995 * self.rate[:n] + 0.005 * sig.mean(0)
            self.b[:n] += self.eta_b * (T.log(self.rate[:n].clamp(min=1e-6))
                                        - T.log(self.prior[self.tok[:n].clamp(min=0)].clamp(min=1e-6)))
        return {}

    def learn(self, tgt, lr=0.003, plastic_b=0, code_next=None):
        n = self.n_slots
        if self.h is None or n == 0:
            return {}
        if self.h.shape[1] != n or self.sigma.shape[1] != n:
            self.reset_state(self.h.shape[0])
            return {}
        B = self.h.shape[0]
        nb = B if plastic_b <= 0 else min(plastic_b, B)
        p = T.softmax(self.logits[:nb], dim=1)
        oh = T.zeros_like(p)
        oh.scatter_(1, tgt[:nb].unsqueeze(1), 1.0)
        # 置信度匹配：让"模型自报的平均置信度"= "实际命中率"。
        # 这是校准的定义，不是手填温度：目标量由两个可观测量在线给出（都不需要额外标签）。
        # R104 实测：所有配置都出现"排名涨、PPL 崩"，即有效增益单调变大 → 过自信。
        with T.no_grad():
            conf = float(p.max(dim=1).values.mean())
            acc = float((p.argmax(1) == tgt[:nb]).float().mean())
            if self.eta_temp > 0 and self.norm_score == 0:
                # 温度由**训练似然**的梯度在线拟合（不是拟合"置信度=命中率"这个代理判据）：
                #   NLL = −log softmax(g·s)_y  ⇒  ∂NLL/∂log g = −g·(s_y − E_p[s])
                # 若正确答案的分数高于平均 ⇒ 变尖，否则变平。完全局部、不需要额外标签。
                # 换掉旧判据的原因：实测置信度匹配会**过冲**（收敛到 g=1.57，而 PPL 最优在 g=1.25），
                # 反而让 PPL 差约 28%。
                _g = self.gain_t
                _s0 = self.logits[:nb] / _g
                _sy = _s0.gather(1, tgt[:nb].unsqueeze(1)).squeeze(1)
                _ey = (p * _s0).sum(1)
                _step = self.eta_temp * float((_sy - _ey).mean())
                self.gain_t = min(1e3, max(1e-3, _g * math.exp(_step)))
            elif conf > 0 and acc > 0 and self.eta_cal > 0 and self.norm_score == 0:
                self.gain_t *= math.exp(-self.eta_cal * (math.log(conf) - math.log(acc)))
                self.gain_t = min(10.0, max(0.01, self.gain_t))
        e = (p - oh)                                            # (nb, V) 逐词误差
        tok = self.tok[:n]
        e_u = e.gather(1, tok.clamp(min=0).unsqueeze(0).expand(nb, n))
        # dL/dz_i = (p_v − y_v) · softmax_{j∈v}(z_j)  —— **只有赢家拿梯度**（自动分工）
        if self.cell_w is not None:
            _w0, _w1, _ = self.cell_w
        else:
            _w0 = _w1 = None            # mix 模式：教学信号直接用词级误差，不需要 cellw
        _idx = tok.clamp(min=0).unsqueeze(0).expand(nb, n)
        if self._mix_scale is not None:
            # Σ_k w_k = 1 ⇒ 词级误差直接作教学信号（不再乘 cellw）
            dz = e_u * self.gain_t
        elif self.dz_mode == "word":
            # 教学信号 = **词级预测误差** (p_v − y_v)，直接广播给该词的所有细胞。
            # 为什么与 cellw 分开：cellw = e^{u}/λ 是"先验地板"带来的词内归一，它属于**前向的稳态机制**；
            # 若把它也乘进学习信号，就会把"被压低的正确词"的修正压到 ~0（softmax 梯度消失），
            # 那正是模型只到 bigram 水平的原因之一。
            # 生物对应：调质/教学信号是词级的（多巴胺式广播），地板是另一套（稳态）机制，二者不该混。
            dz = e_u * self.gain_t
        else:
            # dscore/du_i = e^{u_i} / λ_v（词内 softmax）
            cellw = _w0[:nb] / _w1[:nb].gather(1, _idx)
            dz = e_u * cellw * self.gain_t
        alivef = self.alive[:n].float().unsqueeze(0)
        dz = dz * alivef
        if self._chunkable():
            return self._apply_chunked(dz, tgt, nb, lr)
        if self.apical_mode != "none":
            # ---- 顶端教学流（B1+D1 冒烟原型）----
            # 不新增隐藏层、不新增可学权重、不使用 W^T；教学信号逐样本、逐细胞、带符号，
            # 只作为局部调制因子乘在已有的逐细胞误差 dz 上。
            with T.no_grad():
                if self.apical_mode == "bit":
                    if code_next is None:
                        raise SystemExit("--apical-mode bit 需要把下一 token 的码传进 learn")
                    _bits = (T.arange(n, device=self.dev) % self.SM)
                    _cn = code_next[:nb, _bits].float()
                    t = 2.0 * _cn - 1.0          # 二值码 → 有符号；2/1 来自码本身，不是拟合常数
                else:
                    # y_v − p_v：符号由"该细胞所属的词是否为正确词"决定；按词内细胞数分摊
                    _ev = (oh[:nb] - p[:nb]).gather(
                        1, self.tok[:n].clamp(min=0).unsqueeze(0).expand(nb, n))
                    _cnt = T.zeros(self.V, device=self.dev).index_add_(
                        0, self.tok[:n].clamp(min=0), T.ones(n, device=self.dev)).clamp(min=1.0)
                    t = _ev / _cnt.gather(0, self.tok[:n].clamp(min=0)).unsqueeze(0)
                if not bool(T.isfinite(t).all()):
                    raise SystemExit("[NaN/Inf] 顶端教学信号非有限 → 立即停止")
                if self.apical_norm == "sat":
                    t = t / (1.0 + t.abs())          # 受体饱和；1 是恒等基线，不是拟合常数
                elif self.apical_norm == "rms":
                    t = (t - t.mean()) / t.std().clamp(min=self.rms_eps)
                if not bool(T.isfinite(t).all()):
                    raise SystemExit("[NaN/Inf] 归一化后的顶端教学信号非有限 → 立即停止")
                if not self._apical_diag_done:
                    print("[apical] mode=%s norm=%s mean=%.4f std=%.4f min=%.4f max=%.4f pos=%.3f neg=%.3f"
                          % (self.apical_mode, self.apical_norm, float(t.mean()), float(t.std()),
                             float(t.min()), float(t.max()), float((t > 0).float().mean()),
                             float((t < 0).float().mean())), flush=True)
                    self._apical_diag_done = True
            dz = dz * (1.0 + t)             # 1 是恒等基线；不夹取、不缩放
        # 每个细胞自己的突触、自己的二阶矩（完全局部）
        # 必须拆成两步：写成 `(dz.T @ h) / nb` 会先算矩阵乘再复制一份 (n,D) 张量，
        # 在 y=32768/D=14336 时多占 2.1GB（实测峰值 15.6GB 触到 16G 上限）。
        # 候选 1：读出误差经"隐藏细胞自己的传出突触"回传 = 下行教学信号。
        # 用的是细胞自己的突触（不是别人的转置），与 Nature 2026 顶端树突的图像一致；
        # 数学上等价于一层 BP，属灰区，默认关（--hid-teach 1 打开）。
        global hid_teach, hid_teach_on, hid_teach_size
        if hid_teach_on and hid_teach_size > 0 and self.D >= hid_teach_size:
            Wh = self.w[:n, self.D - hid_teach_size:]          # (n,H) 连到隐藏细胞的权重
            hid_teach = (Wh.t() @ dz.mean(0)).detach()         # (H,)
        if getattr(self, "mod_grow", 0) and self.mod > 0:
            self._mg_t = getattr(self, "_mg_t", 0) + 1
            if (self._mg_t % max(1, self._mg_every)) == 0:
                self._mod_grow_step()
        if self.code_learn and cur_toks is not None and self.lag_w > 0:
            # ∂L/∂code[w] = Σ_b dz_b · w_l（每个细胞只用自己那块突触回传）
            _Wc = self.w[:n]
            _gc = T.zeros_like(self.code)
            for _l in range(min(self.n_lag, cur_toks.shape[1])):
                _sl = slice(_l * self.lag_w, (_l + 1) * self.lag_w)
                _g = dz @ _Wc[:, _sl]                      # (nb, lag_w) = 误差经第 l 块回传
                _ids = T.from_numpy(np.ascontiguousarray(cur_toks[:nb, _l], dtype=np.int64)).to(self.dev)
                _gc.index_add_(0, _ids, _g)
            self.code_grad_ema.mul_(self.code_ema_decay).add_(_gc, alpha=1 - self.code_ema_decay)
            _step = self.code_grad_ema
            self.v_code.mul_(self.beta2).addcmul_(_step, _step, value=1 - self.beta2)
            _dv = self.v_code.sqrt()
            _dv.add_(self.rms_eps)
            # 码的学习率按**扇出**归一：改一个码位会影响该词的全部细胞，
            # 而改一个突触只影响 1 个 ⇒ 码的步长必须按 1/√n 缩小（这是标度问题，不是调参）。
            _clr = self.code_lr if self.code_lr > 0 else lr / math.sqrt(max(1, n))
            self.code.add_(_step / _dv, alpha=-_clr)
            del _dv
            self._code_t += 1
            if (self._code_t % self.code_every) == 0:
                with T.no_grad():
                    # k-WTA：只保留绝对值最大的 k 位（结构约束，等价于出生码的稀疏度）
                    _thr = self.code.abs().kthvalue(self.code.shape[1] - self.code_k + 1,
                                                    dim=1, keepdim=True).values
                    self.code.mul_((self.code.abs() >= _thr).to(self.code.dtype))
                    # 每词 L2 归一：尺度有界
                    _nr = self.code.norm(dim=1, keepdim=True)
                    self.code.div_(_nr.where(_nr > 0, T.ones_like(_nr)))
        if self.coord_sample and self.ng > 1 and self.h_exp is not None:
            gW = T.zeros((n, self.D), dtype=T.float32, device=self.dev)
            for _g in range(self.ng):
                _s = (_g * n) // self.ng
                _e = ((_g + 1) * n) // self.ng
                gW[_s:_e] = dz[:, _s:_e].t() @ self.h_exp[:nb, _g * self.D:(_g + 1) * self.D]
        elif getattr(self, "_fl", None) is not None and self.readout == "laggate":
            # 第 l 块的有效权重就是 g_{l,i}（每细胞不同）
            gW = T.zeros((n, self.D), dtype=T.float32, device=self.dev)
            for _l in range(self.n_lag):
                _sl = slice(_l * self.lag_w, (_l + 1) * self.lag_w)
                gW[:, _sl] = (dz * self._gl2[_l]).t() @ self.h_feat[:nb, _sl]
        elif self._mix_scale is not None:
            # 块权重 w̄_l 是**每个细胞**的（行方向），所以缩放 dz 而不是 h，逐块做 matmul。
            gW = T.zeros((n, self.D), dtype=T.float32, device=self.dev)
            for _l in range(self.n_lag):
                _sl = slice(_l * self.lag_w, (_l + 1) * self.lag_w)
                gW[:, _sl] = (dz * self._mix_scale[_l].unsqueeze(0)).t() @ self.h_feat[:nb, _sl]
        else:
            gW = dz.t() @ self.h_feat[:nb]
        gW.div_(nb)
        if self.mod > 0 and self._m is not None:
            # 三因子：dL/dα_vm = dz_v (β_v·m) m_m；dL/dβ_vm = dz_v (α_v·m) m_m
            # dL/dP = Σ_v dz_v [(β_v·m) α_vm + (α_v·m) β_vm] h  → 用 (B,M) 的中间量算，成本 O(n·M + M·D)
            _m = self._m[:nb]                                   # (nb, M)
            _a = (self.al[:n] @ _m.t()).t()                     # (nb, n)
            _b = (self.be[:n] @ _m.t()).t()                     # (nb, n)
            _A = dz * _b                                        # (nb, n)
            _Bv = dz * _a
            _ga = (_A.t() @ _m) / nb                            # (n, M)
            _gb = (_Bv.t() @ _m) / nb
            _G = (_A @ self.al[:n] + _Bv @ self.be[:n]) / nb     # (nb, M)
            _gP = _G.t() @ self.h_feat[:nb]                     # (M, D)
            for _par, _g, _v in ((self.al, _ga, self.v_al),
                                 (self.be, _gb, self.v_be)):
                _v[:n].mul_(self.beta2).addcmul_(_g, _g, value=1 - self.beta2)
                _dn = _v[:n].sqrt(); _dn.add_(self.rms_eps)
                _par[:n].add_(_g / _dn, alpha=-lr)
                del _dn
            self.v_P.mul_(self.beta2).addcmul_(_gP, _gP, value=1 - self.beta2)
            _dp = self.v_P.sqrt(); _dp.add_(self.rms_eps)
            self.P.add_(_gP / _dp, alpha=-lr)
            del _dp, _gP, _G, _ga, _gb, _A, _Bv, _a, _b
        if self.grp > 0 and self._prod is not None and getattr(self, "theta", None) is not None:
            _dzs = dz.sum(0) / nb                                 # (n,) 调制信号（词级误差）
            for _pi in range(self._prod.shape[0]):
                if self.grp_trace:
                    # 三因子规则（e-prop）：迹在**每条流**上累积该段的共激活，
                    # 再由当前词级误差门控 —— 一次误差能强化**一整段历史**的共激活，
                    # 交互权重的样本效率提高 ~trace_h 倍（R161 表明样本量正是瓶颈）。
                    if self.e_th is None or self.e_th.shape[1] != nb:
                        self.e_th = T.zeros(self._prod.shape[0], nb, self.grp,
                                            dtype=T.float32, device=self.dev)
                    self.e_th[_pi].mul_(self.rho).add_(self._prod[_pi][:nb])
                    _gth = (dz.t() @ self.e_th[_pi]) / nb             # (n, G)
                else:
                    _gth = (dz.t() @ self._prod[_pi][:nb]) / nb
                self.v_th[_pi, :n].mul_(self.beta2).addcmul_(_gth, _gth, value=1 - self.beta2)
                _dn = self.v_th[_pi, :n].sqrt(); _dn.add_(self.rms_eps)
                self.theta[_pi, :n].add_(_gth / _dn, alpha=-lr)
                del _dn, _gth
        if getattr(self, "mod_grow", 0) and self.mod > 0:
            self._mg_t = getattr(self, "_mg_t", 0) + 1
            if (self._mg_t % max(1, self._mg_every)) == 0:
                self._mod_grow_step()
        if self.code_learn and cur_toks is not None and self.lag_w > 0:
            # ∂L/∂code[w] = Σ_b dz_b · w_l（每个细胞只用自己那块突触回传）
            _Wc = self.w[:n]
            _gc = T.zeros_like(self.code)
            for _l in range(min(self.n_lag, cur_toks.shape[1])):
                _sl = slice(_l * self.lag_w, (_l + 1) * self.lag_w)
                _g = dz @ _Wc[:, _sl]                      # (nb, lag_w) = 误差经第 l 块回传
                _ids = T.from_numpy(np.ascontiguousarray(cur_toks[:nb, _l], dtype=np.int64)).to(self.dev)
                _gc.index_add_(0, _ids, _g)
            self.code_grad_ema.mul_(self.code_ema_decay).add_(_gc, alpha=1 - self.code_ema_decay)
            _step = self.code_grad_ema
            self.v_code.mul_(self.beta2).addcmul_(_step, _step, value=1 - self.beta2)
            _dv = self.v_code.sqrt()
            _dv.add_(self.rms_eps)
            # 码的学习率按**扇出**归一：改一个码位会影响该词的全部细胞，
            # 而改一个突触只影响 1 个 ⇒ 码的步长必须按 1/√n 缩小（这是标度问题，不是调参）。
            _clr = self.code_lr if self.code_lr > 0 else lr / math.sqrt(max(1, n))
            self.code.add_(_step / _dv, alpha=-_clr)
            del _dv
            self._code_t += 1
            if (self._code_t % self.code_every) == 0:
                with T.no_grad():
                    # k-WTA：只保留绝对值最大的 k 位（结构约束，等价于出生码的稀疏度）
                    _thr = self.code.abs().kthvalue(self.code.shape[1] - self.code_k + 1,
                                                    dim=1, keepdim=True).values
                    self.code.mul_((self.code.abs() >= _thr).to(self.code.dtype))
                    # 每词 L2 归一：尺度有界
                    _nr = self.code.norm(dim=1, keepdim=True)
                    self.code.div_(_nr.where(_nr > 0, T.ones_like(_nr)))
        if getattr(self, "_fl", None) is not None and self.readout == "laggate":
            # dL/dg = dz·f，再过 sigmoid 导数 ⇒ γ（共享）与 δ（每细胞）的梯度
            for _l in range(self.n_lag):
                _dgv = dz * self._fl[_l]
                _c2 = _dgv * self._gl2[_l] * (1.0 - self._gl2[_l])
                _gb = _c2.mean(0)
                self.v_gofs[_l, :n].mul_(self.beta2).addcmul_(_gb, _gb, value=1 - self.beta2)
                _db = self.v_gofs[_l, :n].sqrt()
                _db.add_(self.rms_eps)
                self.gofs[_l, :n].add_(_gb / _db, alpha=-lr)
                if getattr(self, "gam", None) is not None:
                    if self.gate_full:
                        _xin2 = self.h_feat[:nb]
                    else:
                        _xin2 = self.h_feat[:nb, _l * self.lag_w:(_l + 1) * self.lag_w]
                    # γ_l 是**共享**标量组：先把所有细胞的 ∂L/∂z 求和，再乘 h（否则维度对不上）
                    _gv2 = (_c2.sum(1).unsqueeze(1) * _xin2).sum(0) / nb
                    self.v_gam[_l].mul_(self.beta2).addcmul_(_gv2, _gv2, value=1 - self.beta2)
                    _dn2 = self.v_gam[_l].sqrt()
                    _dn2.add_(self.rms_eps)
                    self.gam[_l].add_(_gv2 / _dn2, alpha=-lr)
                    del _dn2, _gv2
                del _db, _gb, _c2, _dgv
        if self.gate_lag and self.lag_w > 0 and self._gl is not None:
            # dL/dg_l = Σ_v dz_v (w_{v,l}·h_l) = h_l · (dz^T w_l)
            _W = self.w[:n]
            for _l in range(self.n_lag):
                _wl = _W[:, _l * self.lag_w:(_l + 1) * self.lag_w]        # (n, lag_w)
                _dg = dz @ _wl                                            # (nb, lag_w) = Σ_v dz_v w_{v,l}
                _hl = self.h_feat[:nb, _l * self.lag_w:(_l + 1) * self.lag_w]
                _dl = (_dg * _hl).sum(1)                                  # (nb,)
                if self.gate_soft:
                    continue          # softmax 的雅可比需要跨 lag 求和，统一在循环外算（见下）
                _c = _dl * self._gl[:nb, _l] * (1.0 - self._gl[:nb, _l])  # 过 sigmoid 的导数
                _xin = self.h_feat[:nb] if self.gate_full else _hl         # 门看什么，梯度就乘什么
                if getattr(self, "gate_ofs", 0):
                    # 偏置是每细胞一个标量 ⇒ 梯度就是 _c 的批均值
                    _gb = _c.mean(0)
                    self.v_gofs[_l, :n].mul_(self.beta2).addcmul_(_gb, _gb, value=1 - self.beta2)
                    _db = self.v_gofs[_l, :n].sqrt(); _db.add_(self.rms_eps)
                    self.gofs[_l, :n].add_(_gb / _db, alpha=-lr)
                    del _db, _gb
                _gvec = (_c.unsqueeze(1) * _xin).sum(0) / nb
                self.v_gam[_l].mul_(self.beta2).addcmul_(_gvec, _gvec, value=1 - self.beta2)
                _den = self.v_gam[_l].sqrt(); _den.add_(self.rms_eps)
                self.gam[_l].add_(_gvec / _den, alpha=-lr)
                del _den, _gvec, _dl, _dg
        if self.gate_lag and self.gate_soft and self.lag_w > 0 and self._gl is not None:
            # dL/dz_m = g_m (dL/dg_m − Σ_l g_l dL/dg_l)   （softmax 的雅可比）
            _W = self.w[:n]
            _D = []
            for _l in range(self.n_lag):
                _wl = _W[:, _l * self.lag_w:(_l + 1) * self.lag_w]
                _dg = dz @ _wl
                _hl = self.h_feat[:nb, _l * self.lag_w:(_l + 1) * self.lag_w]
                _D.append((_dg * _hl).sum(1))          # (nb,) = dL/dg_l
            _Dm = T.stack(_D, 1)                        # (nb, n_lag)
            _g = self._gl[:nb]
            _dzm = _g * (_Dm - (_Dm * _g).sum(1, keepdim=True))
            for _l in range(self.n_lag):
                # 门看什么，梯度就乘什么（gate_full 时是全部上下文）
                _hl = self.h_feat[:nb, _l * self.lag_w:(_l + 1) * self.lag_w]
                _xin = self.h_feat[:nb] if self.gate_full else _hl
                if getattr(self, "gate_ofs", 0):
                    _gb2 = _dzm[:, _l].mean(0)
                    self.v_gofs[_l, :n].mul_(self.beta2).addcmul_(_gb2, _gb2, value=1 - self.beta2)
                    _db2 = self.v_gofs[_l, :n].sqrt(); _db2.add_(self.rms_eps)
                    self.gofs[_l, :n].add_(_gb2 / _db2, alpha=-lr)
                    del _db2, _gb2
                _gvec = (_dzm[:, _l].unsqueeze(1) * _xin).sum(0) / nb
                self.v_gam[_l].mul_(self.beta2).addcmul_(_gvec, _gvec, value=1 - self.beta2)
                _den = self.v_gam[_l].sqrt(); _den.add_(self.rms_eps)
                self.gam[_l].add_(_gvec / _den, alpha=-lr)
                del _den, _gvec
            del _Dm, _dzm, _D
        if self.dend and self._ua is not None:
            # 区室 a 的梯度用区室 b 的电位，反之亦然 —— 都在同一细胞内，属局部量。
            _g1 = (dz * self._ub[:nb]).t() @ self.h_feat[:nb]
            _g1.div_(nb)
            _g2 = (dz * self._ua[:nb]).t() @ self.h_feat[:nb]
            _g2.div_(nb)
            self.v_d1[:n].mul_(self.beta2).addcmul_(_g1, _g1, value=1 - self.beta2)
            self.v_d2[:n].mul_(self.beta2).addcmul_(_g2, _g2, value=1 - self.beta2)
            _d1 = self.v_d1[:n].sqrt(); _d1.add_(self.rms_eps)
            _d2 = self.v_d2[:n].sqrt(); _d2.add_(self.rms_eps)
            self.wd1[:n].add_(_g1 / _d1, alpha=-lr)
            self.wd2[:n].add_(_g2 / _d2, alpha=-lr)
            del _d1, _d2, _g1, _g2
        if self.ridx is not None:
            # 只保留该细胞自己的扇入：建一个 (n,D) 稀疏掩码太贵，改用散射清零
            _m = T.zeros_like(gW)
            _m.scatter_(1, self.ridx[:n], 1.0)
            gW.mul_(_m)
        vw = self.v_w[:n]
        # 原地更新二阶矩 + 原地归一化：临时张量从 5 个降到 1 个。
        # 之前 y=65536/D=4096 时 w/v/gW/step/sqrt 各 2.1GB → 15.8GB 撞墙抖动，
        # 说明"大 y 跑不动"先是内存写法问题，不是原理问题。
        vw.mul_(self.beta2).addcmul_(gW, gW, value=1 - self.beta2)
        # 符号：e = p − oh **就是** dL/dscore 的方向，所以必须是梯度下降（−）。
        # （3.x 用 e = hit − p 配 +=；换成 p − oh 后忘了改符号 → 全部在训练成错的，
        #   实测训练 PPL 从 unigram 2737 一路涨到 2856 且误差单调上升。R104 踩过这个坑。）
        # 每突触自适应步长 + **逐步夹紧到 ±1**：
        # 稀疏码下罕见词的梯度只有 ~1e-9，`v` 衰减到 1e-20 后 sqrt(v)=1e-10 被 eps=1e-8 压住，
        # 步长变成 lr·g/1e-8 → 后期爆掉（实测误差 7.71→8.82、训练 PPL 385→822，与 lr 无关）。
        # 夹紧后每个突触每步最多走 lr，等价于"符号步"，累积爆不掉（也是标准的 per-element clip）。
        # 试过把每步步长夹紧到 ±1：更差（t=12500 训练 PPL 1641 vs 385）。说明"大步长"本身是
        # 学习机制的一部分（罕见词的稀疏大梯度），不是纯粹的噪声；夹紧把它一起掐掉了。已回滚。
        plast = None
        if self.plast_tau > 0:
            # ① 可塑性随细胞年龄衰减（关键期）：1/(1+age/τ)。τ 取 run 长度的量级，
            # 是尺度相对的，不是对着数据拟合出来的常数。
            _p = 1.0 / (1.0 + self.age[:n] / self.plast_tau)
            if self.plast_norm:
                # **稳态归一**：除以**在世细胞**的平均可塑性，让总可塑性守恒。
                # 为什么必须归一：不归一时 1/(1+age/τ) 会把整个网络的学习率一起压到 0，
                # 长跑就等于"训练到一半自己冻结"（τ 远小于步数时）。归一时只保留**相对**差异 ——
                # 年轻的细胞更可塑，老细胞让位，但群体的总步长不变。
                # 生物对应：乘性突触缩放 / 稳态设定点（文献04），不是新机制。
                _m = self.alive[:n]
                _pm = (_p * _m.float()).sum() / _m.float().sum().clamp(min=1.0)
                _p = _p / _pm.clamp(min=1e-6)
            plast = _p.unsqueeze(1)
        den = vw.sqrt()
        den.add_(self.rms_eps)
        gW.div_(den)
        del den
        if self.freeze_readout:
            # 约束A 检验：把**读出**冻住，只让别的部分学。
            # 做法是把梯度清零（等价于 lr=0），不夹取、不缩放 —— 冻结就是冻结。
            gW.zero_()
        if plast is not None:
            gW.mul_(plast)
        if self.tier is not None:
            # 结构与"出生序=身份"一致：第 t 档只允许看前 b[t] 个带通块，
            # 其余列的**梯度直接清零**（等价于永久剪掉，但不额外占显存）。
            # 槽位是连续的 ⇒ 每档的行集合是一段连续区间，切片赋值即可，没有额外开销。
            _V = int(self.tier["V"]); _lav = int(self.tier["lagdim"]); _bow = int(self.tier["bow"])
            for _t, _b0 in enumerate(self.tier["b"]):
                _r0 = _t * _V
                _r1 = n if _t == len(self.tier["b"]) - 1 else min((_t + 1) * _V, n)
                if _r0 >= _r1:
                    continue
                _cut = _lav + int(_b0) * _bow
                if _cut < self.D:
                    gW[_r0:_r1, _cut:] = 0.0
        if self.upd == "eg":
            # ③ 指数化梯度（乘性更新）+ 符号锁定 + 睡眠式乘性下调
            #   · 乘性 → 权重自然变成 log-normal 重尾（实测皮层：单个细胞内就是 log-normal）
            #   · sign(w) 锁定 → 突触不变号，满足 Dale 定律（兴奋/抑制不互换）
            #   · (1−η_shy) 是按强度比例的乘性下调 = 突触稳态假说的睡眠重整
            # 文献 23 第二节：这样训练出的网络对修剪更鲁棒、更会忽略无关输入，性能不输梯度下降。
            # 注意：**不要**额外加确定性衰减。EG 的指数期望为 0，叠加一个 −η 衰减
            # 会让 |w| 单调衰减到 0（实测 12500 步后 ×4.5e-5，训练 PPL 3559 vs 基线 1946）。
            # EG 本身靠梯度方向收敛：梯度与 w 同号 → |w| 减小，异号 → |w| 增大。
            e = (-lr * gW * T.sign(self.w[:n])).clamp(-0.05, 0.05)
            self.w[:n].mul_(T.exp(e))
            if self.shy > 0:
                self.w[:n].mul_(1.0 - self.shy)
        else:
            self.w[:n].add_(gW, alpha=-lr)
            if self.shy > 0:
                # 睡眠式乘性下调（SHY，突触稳态假说）：与强度成比例的衰减。
                # 为什么要它：budget=0 时 |w| 单调增长（实测续跑 4 段：4.08→5.24→9.41→12.74），
                # 与 dev PPL 崩塌（775→579→858→1614）完全同步 —— 这是长期训练唯一的结构性失稳。
                # 加法更新让 |w| 随机游走式增长，乘性衰减正好提供那个平衡力（λ 与增长率同量级即可）。
                self.w[:n].mul_(1.0 - self.shy)
        # 总突触强度守恒（突触缩放/稳态）：学习只重排突触的相对分配，不改变总强度。
        # 不加这一条会正反馈塌陷：w 无界增长 → σ 饱和 → 过度自信 → 误差更大 → 更增长
        # （R103 实测 w_norm 540→1205，PPL 4404→5216，per_word=8 时更糟）。
        # 幅度由发育设定，与 3.x 的 budget=4.0 保持同一量级。
            # 结构掩码等价实现：把不属于自己那一半的梯度清掉（不占显存）
            hh = self.D // 2
            gW = gW.clone()
            gW[0::2, hh:] = 0.0
            gW[1::2, :hh] = 0.0
        if self.budget > 0 and self.dend:
            # 三个向量各自约束在 RMS=1 的球内（同一单位约定，不是三个不同的数）
            for _vv in (self.w, self.wd1, self.wd2):
                _nr = _vv[:n].pow(2).sum(1).sqrt()
                _sc = (self.budget / _nr.where(_nr > 0, T.ones_like(_nr))).clamp(max=1.0)
                _vv[:n] *= _sc.unsqueeze(1)
        if self.unit_l1:
            # 真正的 L1 球面投影：w ← w / Σ|w|。零行保持 0（用 where 而非 epsilon）。
            # 与"budget 只往下夹"不同：这是双向约束，w 不会长到需要事后补偿，也不会缩到消失。
            _l1 = self.w[:n].abs().sum(dim=1, keepdim=True)
            self.w[:n] = self.w[:n] / _l1.where(_l1 > 0, T.ones_like(_l1))
        elif self.budget > 0:
            ids = self.alive[:n].nonzero(as_tuple=True)[0]
            if ids.numel():
                nrm = self.w[:n].pow(2).sum(1).sqrt()
                sc = (self.budget / nrm.where(nrm > 0, T.ones_like(nrm))).clamp(max=1.0)
                self.w[:n] *= sc.unsqueeze(1)
        # 突触稀疏化：每个细胞只保留最强的 K 个突触，其余置零。
        # 为什么：1 亿参数去记 6M token 必然在线记忆（训练 PPL 125 vs dev 580，差 4.6 倍），
        # 而 budget / 剪枝 / 可塑性衰减 / dropout 都拦不住 → 唯一对症的是**限制每个细胞的存储容量**。
        # 生物依据（文献 23 第二节）：单个细胞内的突触强度是 log-normal，只有少量强突触承载计算，
        # 弱突触 shrinking 到零 —— 这正是"容量受限"在生物里的实现方式。
        if self.topk > 0 and self.topk < self.D:
            self._tct += 1
            if (self._tct % self.topk_every) == 0:
                wv = self.w[:n]
                thr = wv.abs().kthvalue(self.D - self.topk + 1, dim=1, keepdim=True).values
                wv.mul_((wv.abs() >= thr).to(wv.dtype))
        if lag_dim and lag_dim > 0 and self.n_slots > 0:
            # 读出层到底有没有在"看"前几个 lag 的稀疏码？权重能量的分布是最直接的证据。
            _et = self.w[:n].pow(2).sum().clamp(min=1e-12)
            self.w_frac_lag = float(self.w[:n, :lag_dim].pow(2).sum() / _et)
        if hid_w_start and hid_w_start > 0 and self.n_slots > 0:
            # 读出层到底有没有在用隐藏块？权重能量分布是最直接的证据。
            _e = self.w[:n].pow(2).sum().clamp(min=1e-12)
            self.hid_w_frac = float(self.w[:n, hid_w_start:].pow(2).sum() / _e)
        # b 不跟误差走，而是**内禀可塑性**：对数域把每个细胞的平均发放率稳到词先验 freq_v。
        # 理由（R103 实测）：b 跟误差走会塌——标量上 RMSprop 等价 ±lr 随机游走，
        # 罕见词先验被推 5 nats，PPL 2736→14718；b 完全固定又会 σ 饱和塌成单词。
        # 发放率稳态是控制回路（有不动点、不随机游走），生物对应 = 阈值适应/内禀可塑性；
        # 同时也对上历史结论「θ 只该管边际」。区分能力由 w 的**方向**带来（w 幅度不受限，
        # b 把均值拉回先验 → σ 的方差随 |w| 增长，这才是"能变自信"的正确来源）。
        # （b 的稳态更新在下面的 no_grad 块里，与 rate 的 EMA 同步）
        if self.fr_per and self.n_edges > 0:
            pprev = self._ptr_before
            for k in range(self.Delay):
                slot = (pprev - 1 - k) % (self.Delay + 1)
                ids = self.rec_src[:n, k, :]
                pre = self.spike_buf[slot][:nb, ids.reshape(-1)].reshape(nb, n, self.fr_per)
                gg = T.einsum("bn,bnc->nc", dz, pre) / nb
                vr = self.v_rec[:n, k, :]
                vr.mul_(self.beta2).add_(gg * gg, alpha=1 - self.beta2)
                self.w_rec[:n, k, :] += lr * gg / (vr.sqrt() + 1e-8) * self.rec_mask[:n, k, :].float()
        with T.no_grad():
            hit = (tgt[:nb].unsqueeze(1) == tok.unsqueeze(0)).float()
            sig = self.sigma[:nb]                       # 供存活判据（acc）用，不参与读出
            acc = 1.0 - (sig - hit).abs()
            self.acc[:n] = 0.99 * self.acc[:n] + 0.01 * acc.mean(0)
            self.rate[:n] = 0.995 * self.rate[:n] + 0.005 * sig.mean(0)
            self.b[:n] += self.eta_b * (T.log(self.rate[:n].clamp(min=1e-6))
                                       - T.log(self.prior[tok.clamp(min=0)].clamp(min=1e-6)))
        return {}

    # -------------------------------------------- 拓扑
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
        chosen = idx[T.multinomial(w.clamp(min=1e-9), ns, replacement=False)]
        u = chosen // (self.Delay * self.fr_per)
        rem = chosen % (self.Delay * self.fr_per)
        k, j = rem // self.fr_per, rem % self.fr_per
        pool = alive.nonzero(as_tuple=True)[0]
        rate = self.rate[pool].clamp(min=1e-4)
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
            return {"edges": 0, "avg_indeg": 0.0, "free_slots": 0, "per_word": 0.0,
                    "cnt_min": 0, "cnt_max": 0, "words_uncovered": int(self.V)}
        base = {"per_word": round(self.N / max(1, int((self.tok_cnt_raw > 0).sum())), 2),
                "cnt_min": int(self.tok_cnt_raw.min()), "cnt_max": int(self.tok_cnt_raw.max()),
                "words_uncovered": int((self.tok_cnt_raw == 0).sum())}
        alive = self.alive[:n]
        if not self.fr_per:
            return {**base, "edges": 0, "avg_indeg": 0.0, "free_slots": 0}
        m = self.rec_mask[:n]
        indeg = m.sum(dim=(1, 2)).float()
        return {**base, "edges": int(m[alive].sum()),
                "avg_indeg": round(float(indeg[alive].mean()), 2) if bool(alive.any()) else 0.0,
                "max_indeg": int(indeg.max()), "edge_born": self.edge_born,
                "edge_cut": self.edge_cut,
                "free_slots": int(((~m) & alive.view(n, 1, 1)).sum())}

    # -------------------------------------------- 存取
    def snapshot(self):
        n = self.n_slots
        d = {"cap": self.cap, "n_slots": n, "N": self.N, "D": self.D, "V": self.V,
             "fr": self.fr, "Delay": self.Delay, "fr_per": self.fr_per,
             "birth_counter": self.birth_counter, "born_total": self.born_total,
             "dead_total": self.dead_total, "events": self.events}
        keys = ["alive", "tok", "w", "b", "v_w", "v_b", "acc", "rate", "age", "rec_gain",
                "tok_cnt", "tok_cnt_raw"]
        if self.fr_per:
            keys += ["rec_src", "w_rec", "rec_mask", "v_rec"]
        for k in keys:
            d[k] = getattr(self, k)[:n].cpu()
        d["tok_err"] = self.tok_err.cpu()
        d["gain_t"] = self.gain_t
        d["n_edges"] = self.n_edges
        d["topology"] = self.topology()
        return d

    def load_state(self, d):
        n = int(d["n_slots"])
        self.n_slots = n
        self.N = int(d["N"])
        for k in ("birth_counter", "born_total", "dead_total", "events"):
            setattr(self, k, int(d.get(k, 0)))
        keys = ["alive", "tok", "w", "b", "v_w", "v_b", "acc", "rate", "age", "rec_gain",
                "tok_cnt", "tok_cnt_raw"]
        if self.fr_per:
            keys += ["rec_src", "w_rec", "rec_mask", "v_rec"]
        for k in keys:
            if k in d:
                getattr(self, k)[:n] = d[k].to(self.dev)
        if "tok_err" in d:
            self.tok_err = d["tok_err"].to(self.dev)
        self.gain_t = float(d.get("gain_t", 1.0))
        self.n_edges = int(d.get("n_edges", 0))
        self._refresh()
        self.h = None


# ---------------------------------------------------------------- 特征 / 评测 / 生成

def _win_toks(src, pos_np, win):
    w = max(1, win)
    return np.stack([np.asarray(src[np.asarray(pos_np) - j], dtype=np.int64) for j in range(w)], axis=1)


def _feat(corpus, res, win, toks_np, proj=None, bank=None, hidden=None, code_next=None):
    toks = [T.from_numpy(np.ascontiguousarray(toks_np[:, j], dtype=np.int64)).to(corpus.dev)
            for j in range(max(1, win))]
    parts = []
    if win > 0:
        tab = corpus.code if corpus.in_code else corpus.emb
        parts.append(T.cat([tab[t] for t in toks], dim=1))
    if res is not None:
        parts.append(res.step(corpus.emb[toks[0]]))
    if bank is not None:
        # 长上下文摘要：喂当前 token 的码。若设了 code_b（诊断用），带通用它、窗口仍用 code，
        # 从而可以把"窗口输入编码"与"长上下文编码"分开检验。
        _cb = getattr(corpus, "code_b", None)
        if _cb is not None:
            ct = _cb[toks[0]]
        else:
            ct = corpus.code[toks[0]] if corpus.code is not None else corpus.emb[toks[0]]
        parts.append(bank.step(ct))
    _pt1 = getattr(corpus, "pair_t1", None)
    if _pt1 is not None:
        # 二阶合取块（可多对）：每对 (i,j) 给一路 z = T1[x_{t-i}] * T2[x_{t-j}] → k-WTA。
        # 固定代数绑定、零可学参数；只依赖那两个位置的 token ⇒ 是**表征**，不是读出的乘性项。
        _Bp = toks[0].shape[0]
        _kk = min(int(getattr(corpus, "pair_k", 1)), _pt1.shape[1])
        for (_pi, _pj) in corpus.pair_pairs:
            if len(toks) > max(_pi, _pj):
                _z = _pt1[toks[_pi]] * corpus.pair_t2[toks[_pj]]
            else:
                _z = T.zeros((_Bp, _pt1.shape[1]), device=_pt1.device)
            _vi, _ii = T.topk(_z, _kk, dim=1)
            _pb = T.zeros_like(_z)
            _pb.scatter_(1, _ii, 1.0)
            parts.append(_pb)
    h = T.cat(parts, dim=1) if len(parts) > 1 else parts[0]
    global hidden_learn, dropout_p
    if dropout_p > 0 and hidden_learn:
        # 输入 dropout = 突触传递的随机失效（实测皮层释放概率 0.1~0.5，这是真实存在的噪声）。
        # 为什么需要：长期训练会记忆训练流（训练 PPL 125 vs dev 580，差 4.6 倍），
        # 且固定拓扑/钉住 w_norm 都拦不住 → 只能靠正则化。
        h = h * (T.rand_like(h) > dropout_p).to(h.dtype)
    if hidden is not None:
        # 隐藏层看"近窗 + 摘要"的拼接特征，输出高发放率脉冲活动，再拼回读出层
        cn = code_next if code_next is not None else (
            corpus.code[toks[0]] if corpus.code is not None else corpus.emb[toks[0]])
        # 级联池化：bands 逐级被消费，replace 模式下它们从读出输入里被移除
        # 只把前 hid_in_dim 维喂给隐藏层（= 只看稀疏窗口码，不看稠密带通）。
        # 为什么：随机扇入混 4 个 lag + 6 个 band，每个细胞是 128 维随机混合，形不成干净的
        # "跨 lag 联合检测器"；而线性读出**无法表示 A 与 B 同时出现**（只能相加）。
        # 稀疏码只有 204/4096 活跃，展开到 6144 个细胞（920 活跃）是**扩张**，不丢信息。
        _hin = h.detach() if hid_in_dim <= 0 else h.detach()[:, :hid_in_dim]
        if isinstance(hidden, Cascade):
            _acts = hidden.step(_hin, cn, learn=hidden_learn)
            _hs = T.cat(_acts, dim=1)
            if hidden.mode == "replace":
                # 只保留 band 之前的原始特征（近窗码），band 全部由级联重建
                h = T.cat([h[:, :hidden.band0], _hs], dim=1)
            elif hidden.mode == "only":
                h = _hs
            else:
                h = T.cat([h, _hs / (_hs.norm(dim=1, keepdim=True) + 1e-6)], dim=1)
            hs = _hs
            hidden.resid_chain = [0.0] * len(_acts)
        else:
            # 支持层堆叠：逐层串联，每层用自己的局部目标（自监督时不看下游误差）
            _layers = hidden if isinstance(hidden, (list, tuple)) else [hidden]
            _x = _hin
            _sigs = []
            _resid_chain = []
            for _lyr in _layers:
                _s, _x = _lyr.step(_x, cn, learn=hidden_learn)   # _x 变成残差
                _sigs.append(_s)
                _resid_chain.append(_lyr.resid_var)
            hidden = _layers[-1]
            # 读出用**所有层活动的拼接**：哪个层"没用"（残差已被前层解释完），它的 sig 自然接近常数，
            # 权重学成 0 → 有效层数由数据决定，而不是由 --hid-layers 预设。
            hs = T.cat(_sigs, dim=1) if len(_sigs) > 1 else _sigs[0]
            hidden.resid_chain = _resid_chain
        if hidden.mode == "gate":
            # 候选 2：门控而非加性注入。以 1 为中心（减去稳态发放率），经固定投影回 D_in 维，
            # 期望上不改变特征，只有偏离稳态才调制 → 不摊薄权重预算、D 不变。
            g = (hs - hidden.target) @ hidden.G
            h = h * (1.0 + g)
        elif hidden.mode == "only":
            # 隐藏层是**唯一通路**：读出层看不到原始特征，信息必须流经拓扑。
            # 为什么必须做这个对照：并联旁支时（add 模式）读出层可以直接忽略隐藏层，
            # 实测"结构拓扑 vs 随机拓扑"的 dev PPL 逐位相同（783.9/671.3/741.6 vs 783.9/671.3/741.5）——
            # 那不是"拓扑没用"，而是"没有信息必须经过它"。
            # **不要** L2 归一化：归一化会把范数压成 1，而原始特征（204 个 1）范数是 sqrt(204)≈14，
            # 信号被压小 14 倍 → 读出层学不动（实测 PPL 2321 贴着 unigram 2736）。
            # 隐藏层 25% 发放率时范数 = sqrt(1024)=32，与原始特征同量级，直接用它。
            h = hs
        elif hidden.mode != "replace":
            # 加性注入。**不要**再归一化到范数 1：原始特征范数约 25（204 个 1 + 密集带通），
            # 把隐藏层压成范数 1 等于把它稀释 25 倍，读出层自然学成 0 ——
            # 这就是"并联旁支完全无差异"（783.9 逐位相同）的另一个技术原因。
            # 现在按 add_scale 缩放，默认 1.0（隐藏层范数 ≈ sqrt(H·target) ≈ 30，与原始特征同量级）。
            h = T.cat([h, hs * float(add_scale)], dim=1)
    if getattr(corpus, "logodds_out", 0):
        # 伯努利似然输出：logit_v = c_v · logit(a) + 常数，a = 隐藏层发放率。
        # **恒等式**：logit(sigmoid((u-θ)/T)) ≡ (u-θ)/T ⇒ 直接用前激活，绝不会 ±inf。
        # （上一版对已饱和的 σ 取 log，t=167 就 NaN 了 —— 被守卫抓住，没污染任何结论。）
        if not hasattr(hidden, "u_last"):
            raise SystemExit("--logodds 需要隐藏层暴露前激活 u_last")
        h = (hidden.u_last - hidden.theta.unsqueeze(0)) / hidden.temp
    if proj is not None:
        h = h @ proj
    if corpus.in_code:
        # 稀疏二值码的模长本身就是有意义的尺度（sqrt(k·win)），归一化会把驱动压掉 14 倍。
        return h
    return h / (h.norm(dim=1, keepdim=True) + 1e-6)      # 稠密特征才做分裂式归一化


class CtxBank:
    """多时间尺度带通滤波器组：把"任意长上下文"压成固定维度的特征。

    为什么必须这样
    --------------
    把每个历史 token 的码都拼进特征，D = 上下文长度 × SM，512 个 token 就是 52 万维，
    读出层 y×D 直接爆掉。所以"上下文可调到 2k+"**不能**靠拼接，只能靠对历史的压缩摘要。

    做法（全部固定、无参数、无学习）
    -------------------------------
    维护 K 个泄漏积分器 a_k（一阶低通，时间常数 τ_k 从 1 到 ctx_len 对数等距）：
        a_k ← (1−1/τ_k)·a_k + (1/τ_k)·code_t
    相邻两个低通相减就是**带通**：
        b_k = a_k − a_{k+1}      （b_{K-1} = a_{K-1}）
    b_k 大致对应"lag 落在 [τ_k, τ_{k+1}) 那一带的历史码"。再把每个 b_k 用**固定随机投射**
    压到 d 维（不可学，属输入编码）。于是：

        h_ctx = [b_0·P_0, b_1·P_1, ..., b_{K-1}·P_{K-1}]   →  维度 = K·d，与 ctx_len 无关

    生物学对应：皮层神经元的时间尺度是**每个细胞自己的属性**，绝大多数呈双成分（快+慢），
    且跨区域有一致的尺度层级（文献 23 第七节）。多个泄漏常数并行 = 多尺度适应。
    显存代价：K×(B×SM)，与 ctx_len 无关 → **上下文想调到多大就多大**。
    """

    def __init__(self, corpus, ctx_len, K=10, d=128, seed=20261110, device="cuda",
                 sm_override=0):
        self.dev = T.device(device)
        self.SM = int(sm_override) if sm_override else int(corpus.SM)
        self.ctx_len = float(max(1.0, ctx_len))
        self.K = int(K)
        self.d = int(d)
        # 时间常数：1 → ctx_len 对数等距
        taus = np.exp(np.linspace(0.0, math.log(self.ctx_len), self.K)).astype(np.float32)
        self.taus = taus
        self.alpha = (1.0 / np.maximum(taus, 1.0)).astype(np.float32)
        g = T.Generator().manual_seed(seed)
        # d<=0 时不压缩：直接把该尺度的码当特征（读出层才能学到"这一带里出现过哪些 token"）。
        # 压缩到 128 维会把码的内容扔掉 —— 512 token 的码平均本就接近词频分布，压完几乎没信息，
        # 实测 ctx=512 与 ctx=4 打平（618.4 vs 619.6）且 top1 更差。所以默认不压缩。
        self.P = None
        if self.d > 0:
            self.P = [ (T.randn(self.SM, self.d, generator=g) / math.sqrt(self.SM)).to(self.dev)
                       for _ in range(self.K) ]
        self.a = None

    def reset(self, B):
        self.a = T.zeros(self.K, B, self.SM, device=self.dev)

    def step(self, code_t):
        if self.a is None or self.a.shape[1] != code_t.shape[0]:
            self.reset(code_t.shape[0])
        if not hasattr(self, '_al'):
            self._al = T.from_numpy(self.alpha).to(self.dev).view(self.K, 1, 1)
        al = self._al
        # 原地低通：a_k ← a_k + (1/τ_k)(code − a_k)
        self.a.add_((code_t.unsqueeze(0) - self.a) * al)
        outs = []
        for k in range(self.K):
            b = self.a[k] - self.a[k + 1] if k + 1 < self.K else self.a[k]
            outs.append(b @ self.P[k] if self.P is not None else b)
        return T.cat(outs, dim=1)

    @property
    def out_dim(self):
        return self.K * (self.d if self.d > 0 else self.SM)





class Cascade:
    """级联池化：层 = 逐级扩大的时间感受野 + 逐级减半的细胞数。

    为什么是"池化"而不是"残差"（R119 已量化否掉残差流）
    --------------------------------------------------
    我们的输入是稀疏二值码，方差均匀摊在 1024 个码位上（前 200 个主成分只解释 35%，
    有效维度 113/10240）。"减掉已解释的部分"每层只能啃零头（实测残差 0.0373/0.0380/0.0371/0.0377）。
    池化相反：**把细节扔掉、只留不变的部分** —— 这才是真正降方差的操作。
    生物对应感受野逐级增大（V1→V2→V4→IT：每级看更大范围、单元更少）。

    结构（K 个尺度带，band 在 h 里的偏移是 band0）
    ---------------------------------------------
        L0: band[0:b1]              → H0 个细胞（细时间尺度）
        L1: pool(L0) + band[b1:b2]  → H1（中）
        L2: pool(L1) + band[b2:]    → H2（粗）
    层间 pool 是**固定随机投射**（不可学），所以"结构"本身不引入可学参数。

    replace 模式：被消费掉的 band 从读出输入里移除 → 信息**必须**流经级联。
    这是唯一能让"拓扑"承重的设置（并联旁支时读出层可以直接忽略它，实测逐位相同）。
    """

    def __init__(self, layers, bounds, mode="replace", band0=0, band1=0, seed=20261113):
        self.layers = list(layers)          # list[HiddenLayer]
        self.bounds = list(bounds)          # 每层的 band 区间（band 内坐标）
        self.mode = str(mode)
        self.band0 = int(band0)
        self.band1 = int(band1)
        self.n_split = 0
        self.resid_chain = []
        self.resid_var = 0.0
        self._seed = int(seed)
        self.rebuild_pools()

    @property
    def H(self):
        return sum(l.H for l in self.layers)

    @property
    def target(self):
        return self.layers[0].target

    @property
    def dev(self):
        return self.layers[0].dev

    def rebuild_pools(self):
        """层间固定池化矩阵（不可学）：(H_{k-1}, H_k)。"""
        g = T.Generator().manual_seed(self._seed)
        self.pools = [None]
        for k in range(1, len(self.layers)):
            a, b = self.layers[k - 1].H, self.layers[k].H
            self.pools.append((T.randn(a, b, generator=g) / math.sqrt(a)).to(self.layers[k].dev))

    def _input_parts(self, k, h, prev):
        parts = []
        if prev is not None:
            parts.append(prev @ self.pools[k])
        b0, b1 = self.bounds[k]
        parts.append(h[:, self.band0 + b0: self.band0 + b1])
        return T.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def step(self, h, cn, learn=True):
        """返回 list[每层活动 (B,H_k)]。"""
        acts, prev = [], None
        for k, lyr in enumerate(self.layers):
            inp = self._input_parts(k, h, prev)
            a, _ = lyr.step(inp, cn, learn=learn)
            acts.append(a)
            prev = a
        return acts

    def block_energy(self, k):
        """第 k 层的权重能量在各输入块上的分布 → (H_k, n_blocks)。

        输入块 = [pool(上一层)] + [这一层的每个 band]。局部量：只用本层自己的突触与接线。
        用来判断"这一层是否尺度混杂"（自发生长/分裂的判据）。
        """
        lyr = self.layers[k]
        off = self.layers[k - 1].H if k > 0 else 0
        b0, b1 = self.bounds[k]
        step = getattr(lyr, "_band_step", 1)
        blk = []
        if k > 0:
            blk.append((0, off))
        for x in range(b0, b1, step):
            blk.append((off + (x - b0), off + (min(b1, x + step) - b0)))
        E = []
        for (a, b) in blk:
            if b <= a:
                continue
            m = ((lyr.src >= a) & (lyr.src < b)).to(lyr.w.dtype)
            E.append((lyr.w.pow(2) * m).sum(1))
        if not E:
            return T.zeros(lyr.H, 1, device=lyr.dev), blk
        return T.stack(E, 1), blk

    def mixed_frac(self, k):
        """1 − 主导块能量占比的均值。越高越"尺度混杂"（该分裂）。"""
        E, _ = self.block_energy(k)
        tot = E.sum(1).clamp(min=1e-12)
        dom = (E / tot.unsqueeze(1)).max(1).values
        return float(1.0 - dom.mean())

    def layer_codes(self):
        """每层的"结构身份"：一个短的十六进制指纹，用来检查续跑时结构是否一致。"""
        out = []
        for l in self.layers:
            out.append("%d:%d" % (l.H, int(l.src.sum() % 99991)))
        return "|".join(out)


class HiddenLayer:
    """脉冲隐藏层：拓扑的宿主，也是"脉冲真正进入因果链"的地方。

    为什么需要它
    ------------
    单层读出里，细胞的发放率被设计成 ≈ 词频（1e-4），几乎不发放；接递归边也传不出东西。
    所以拓扑必须先有一层**高发放**的细胞，它们之间才谈得上连线。

    结构与学习（全部局部，无 BP）
    ----------------------------
    每个隐藏细胞 i：
        输入 = 当前特征 x 的一个**固定稀疏扇入**（F 根线，接线由生长规则给出，不可学）
        递归 = 上一拍其它隐藏细胞的发放，经**稀疏接线**（这才是拓扑的边）
        发放 σ_i = sigmoid((u_i − θ_i) / T)；θ_i 由稳态拉到目标发放率 target（约 5%）
        目标 t_i = sigmoid(下一 token 的码 · v_i)，v_i 是**固定随机投影**
        学习 Δw_i ∝ (t_i − σ_i)·σ_i(1−σ_i)·x_inputs   ← 只用自己的目标与自己的输入
        递归边学习 Δw_rec ∝ (t_i − σ_i)·σ_i(1−σ_i)·发放_src  ← 也只用局部量

    生物学对应：预测编码（每个细胞预测自己那份未来输入）、多尺度适应、内禀可塑性(稳态发放率)、
    睡眠式乘性下调。递归边 = 细胞间真实连线；剪枝 + 突触发生让它**随时间变化**（拓扑会"长"）。

    注意：目标投影 v_i 是固定的，所以不需要任何外部标签；下一 token 的码是流里本来就读得到的。
    """

    def __init__(self, D_in, SM, H=2048, fan=32, rec_fan=8, target=0.05,
                 temp=1.0, eta=0.002, eta_th=0.02, eta_rec=0.001,
                 prune_th=0.002, syn_new=0, seed=20261111, mode="add",
                 topk=0, topk_every=25, geo=0, geo_knb=32, geo_mod=40, geo_wout=0.1,
                 rec_scale=1.0, dale=0, inh_frac=0.2, inh_gain=3.0, self_sup=0,
                 target_mode="proj", autotune=0, device="cuda"):
        self.dev = T.device(device)
        self.D_in, self.SM, self.H = int(D_in), int(SM), int(H)
        self.target, self.temp = float(target), float(temp)
        self.eta, self.eta_th, self.eta_rec = float(eta), float(eta_th), float(eta_rec)
        self.prune_th, self.syn_new = float(prune_th), int(syn_new)
        self.mode = str(mode)
        self.tmode = str(target_mode)
        self.autotune = int(autotune)
        self.topk = int(topk)
        self.topk_every = int(max(1, topk_every))
        self.geo_knb = float(geo_knb)
        self.geo_mod = float(geo_mod)
        self.geo_wout = float(geo_wout)
        g = T.Generator().manual_seed(seed)
        # 固定稀疏扇入（结构规则，不学）
        self.src = T.randint(0, self.D_in, (self.H, fan), generator=g).to(self.dev)
        self.w = (T.randn(self.H, fan, generator=g) / math.sqrt(fan)).to(self.dev)
        self.v_w = T.full((self.H, fan), 1e-6, device=self.dev)
        # 固定目标：proj = 随机方向（旧）；bit = 每个细胞负责一个具体码位
        if self.tmode == "bit":
            # 每个细胞盯住一个码位："下个 token 的第 j 位会不会亮"。
            # 这个问题依赖上下文组合，线性读出表示不了 → 隐藏层才有独占信息。
            self.tb = (T.arange(self.H, device=self.dev) % self.SM)
            self.v = None
        else:
            self.tb = None
            self.v = (T.randn(self.H, self.SM, generator=g) / math.sqrt(self.SM)).to(self.dev)
        # ---------------------------------------------------------------- 接线生成
        # 两条路：geo=0 全局随机（旧的、无结构）；geo=1 按"生成规则"接（位置梯度 + 模块原型）。
        #
        # 为什么必须有 geo：全局随机接线**既不可重复也没有结构**（文献 13 第一条已写明），
        # 拿它去测"拓扑有没有用"逻辑上无效 —— 那是用无结构的对象否证结构的价值。
        #
        # geo=1 的三条规则，全部来自 2026 年的实验证据：
        #   ① 位置梯度（ chemoaffinity / 分子梯度）：连接概率 ∝ exp(−距离/λ)。
        #      证据：PNAS 2026，763 个基因在 213 个脑区的表达梯度预测连接 R=0.88（只看距离是 0.70）；
        #      中科院 2026，灵长类 Pr–Al 双梯度轴在出生时(P0)就有雏形，出生后"锐化"。
        #   ② 模块原型（ground plans）：模块内密、模块间稀。
        #      证据：密歇根 2026（Nature），果蝇 cerebrum 8000+ 种神经元 → <200 个结构原型，
        #      比例约 1/40；两套基因分别定"粗原型"和"原型内的精细连接"。
        #   ③ λ 由"邻域内细胞数 k_nb"反解，而不是手填 —— 这样它随 H 自动缩放，
        #      "特征连接距离 ~ 若干个平均间距"这个比例关系保持不变（文献 21：9–12 胞体半径）。
        self.geo = int(geo)
        self.pos = None
        self.mid = None
        if self.geo:
            _L = float(self.H) ** (1.0 / 3.0)             # 立方体边长：每单位体积一个细胞
            self.pos = (T.rand(self.H, 3, generator=g) * _L).to(self.dev)
            _M = max(1, int(round(self.H / max(2.0, float(geo_mod)))))   # 模块数
            _mside = _L / max(1.0, _M ** (1.0 / 3.0))                     # 模块边长
            _mi = (self.pos / _mside).floor().long().clamp(max=int(round(geo_mod ** (1.0 / 3.0))))
            self.mid = (_mi[:, 0] * 1000 + _mi[:, 1]) * 1000 + _mi[:, 2]
            # λ 反解：球内期望细胞数 = k_nb
            _lam = float(max(1e-6, (float(geo_knb) * _L ** 3 / (self.H * 4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)))
            _d = T.cdist(self.pos, self.pos)
            _w = T.exp(-_d / _lam)
            _same = (self.mid.unsqueeze(1) == self.mid.unsqueeze(0)).float()
            _w = _w * (_same + (1.0 - _same) * float(geo_wout))
            _w.fill_diagonal_(0.0)
            # 无放回按概率采样 rec_fan 条出边：Gumbel top-k
            _u = T.rand(self.H, self.H, generator=g).clamp(1e-9, 1 - 1e-9).to(self.dev)
            _lg = T.log(_w.clamp_min(1e-12)) - T.log(-T.log(_u))
            self.rsrc = T.topk(_lg, rec_fan, dim=1).indices.to(self.dev)
            self._geo_lam, self._geo_L, self._geo_M = _lam, _L, _M
        else:
            # 递归接线：每个细胞 rec_fan 条入边（拓扑的边）
            self.rsrc = T.randint(0, self.H, (self.H, rec_fan), generator=g).to(self.dev)
        # rec_scale：递归边相对前馈的强度。默认 0.1 时 rec 只有前馈的 ~10% → 拓扑被淹没（实测无任何影响）
        # Dale 定律：一个细胞要么全兴奋、要么全抑制（出边同号）。
        # 为什么必须有抑制：只靠"抬高阈值"把发放率压到 25%，会把所有细胞推到 sigmoid≈0.25
        # 的过渡区 → 4096 个输出几乎是同一个常数，方差被压没 → 读出层学不动
        # （实测训练 PPL 2523，贴 unigram 2736）。真正的 25% 稀疏必须靠**竞争**实现，
        # 而竞争电路就是抑制性递归边 —— 拓扑在这里才第一次决定计算。皮层约 20% 是抑制性中间神经元。
        self.dale = int(dale)
        self.inh_frac = float(inh_frac)
        self.inh_gain = float(inh_gain)
        _b = (T.randn(self.H, rec_fan, generator=g) / math.sqrt(rec_fan)).to(self.dev) * (0.1 * float(rec_scale))
        if self.dale:
            _is_inh = (T.arange(self.H, device=self.dev) % 100) < int(round(100 * self.inh_frac))
            # **符号只带 ±1**，强度单独给。之前把 -inh_gain 塞进符号，学习时 rw.abs()*rsign
            # 会每步把幅度翻倍（0.96→1.93→3.86→7.71→15.4），几千步后溢出成 NaN。
            self.rsign = T.where(_is_inh.unsqueeze(1), -1.0, 1.0).expand(self.H, rec_fan).contiguous()
            _amp = T.where(_is_inh.unsqueeze(1), float(self.inh_gain), 1.0).expand(self.H, rec_fan)
            _b = _b.abs() * _amp * self.rsign
        else:
            self.rsign = None
        self.rw = _b
        self.rmask = T.ones(self.H, rec_fan, dtype=T.bool, device=self.dev)
        self.v_rec = T.full((self.H, rec_fan), 1e-6, device=self.dev)
        # 门控模式下把 H 个发放率映射回 D_in 维（固定、不可学、无稀疏开销）
        self.G = (T.randn(self.H, self.D_in, generator=g) / math.sqrt(self.H)).to(self.dev) * 0.1
        # 层级预测编码：每层自己的目标是"预测我下一拍会收到什么输入"（固定随机投影降维），
        # 完全不需要下游误差 —— 这是打破"隐藏层等下游、下游等隐藏层"死锁的关键。
        # 生物依据：皮层每一级都有自己的预测目标（预测编码）；语言通路本身是多级串联
        # （听觉皮层→Wernicke→弓状束→Broca→运动皮层），不是一层。
        self.self_sup = int(self_sup)
        # 稀疏编码 / 匹配追踪（Olshausen–Field）：每个单元只解释输入的一部分，
        # **残差留给下一个单元**。注意这不是"预测编码"—— 那个依据被 Westerberg 2024 否掉了
        # （早期皮层是适应、不是预测）。匹配追踪在 V1 简单细胞上有强证据。
        # W_rec：从本层活动重建输入；层间传 x − sig·W_rec。
        self.W_rec = (T.randn(self.H, self.D_in, generator=g) / math.sqrt(self.H)).to(self.dev) * 0.1
        self.v_rec_w = T.full((self.H, self.D_in), 1e-6, device=self.dev)
        self.resid_var = 0.0
        self.V = (T.randn(self.D_in, self.H, generator=g) / math.sqrt(self.D_in)).to(self.dev)
        self.inp_prev = None
        self.sig_prev = None
        self.theta = T.zeros(self.H, device=self.dev)
        self.rate = T.full((self.H,), self.target, device=self.dev)
        self.prev = T.zeros(self.H, device=self.dev)
        self.sig = T.zeros(self.H, device=self.dev)
        self.edge_born = self.edge_cut = 0

    def reset(self):
        self.prev = T.zeros(self.H, device=self.dev)
        self.sig = T.zeros(self.H, device=self.dev)

    @property
    def n_edges(self):
        return int(self.rmask.sum())

    def step(self, x, code_next, learn=True):
        """x:(B,D_in) 当前特征, code_next:(B,SM) 下一 token 的码。返回 (B,H) 发放率。"""
        B = x.shape[0]
        _fan = int(self.w.shape[1])
        # **分块前向**：`x[:, self.src]` 会真的物化 (B,H,fan)。
        # H=8192、fan=2048、B=256 时那是 17GB —— 这就是过去"深度无用"全部结论的真凶：
        # 实现把 fan 卡在 ~32，隐藏层容量只有读出层的 0.2%。
        # 这里按 H 分块，峰值内存从 O(B·H·fan) 降到 O(B·chunk·fan)。
        _budget = 96 * 1024 * 1024            # 每块 ~96M 元素（float32 ≈ 384MB）
        _chunk = max(1, _budget // max(1, B * _fan))
        if _chunk >= self.H:
            inp = x[:, self.src]                                  # 小 fan：旧路径，逐位不变
            u = (inp * self.w.unsqueeze(0)).sum(-1)               # (B,H)
        else:
            inp = None
            u = T.empty((B, self.H), device=self.dev, dtype=self.w.dtype)
            for _s in range(0, self.H, _chunk):
                _e = min(self.H, _s + _chunk)
                _inp = x[:, self.src[_s:_e]]
                u[:, _s:_e] = (_inp * self.w[_s:_e].unsqueeze(0)).sum(-1)
        if self.n_edges:
            rec = (self.prev[self.rsrc] * self.rw * self.rmask).sum(-1)   # (H,)
            u = u + rec.unsqueeze(0)
        if getattr(self, "autotune", 0) and not getattr(self, "_tuned", False):
            # **出生即校准**：θ 初值 0、temp 初值 1 ⇒ sigmoid(0)=0.5，所有细胞一出生 50% 发放，
            # 前激活 logit 量级不受控 ⇒ 固定输出矩阵下 softmax 一出生就饱和、梯度锁死。
            # 这里用**第一次前向的实测分布**定 θ 与 temp：
            #   temp = std(u)                （让 log-odds 一开始就是 O(1)：方差匹配，与 w_init 同一原则）
            #   θ    = u 的 (1−target) 分位 （让初始发放率就等于目标发放率，不手填偏移）
            with T.no_grad():
                _uf = u.reshape(-1)
                self.temp = float(_uf.std().clamp(min=1e-6))
                self.theta.fill_(float(T.quantile(_uf, 1.0 - float(self.target))))
            self._tuned = True
            print("[hid] 出生校准：temp=%.4f  θ=%.4f（目标发放率 %.3f）"
                  % (self.temp, float(self.theta[0]), self.target), flush=True)
        self.u_last = u.detach()
        sig = T.sigmoid((u - self.theta.unsqueeze(0)) / self.temp)
        self.prev = sig.detach().mean(0)
        self.sig = sig.detach().mean(0)
        if learn:
            global hid_teach
            if self.self_sup:
                # 自监督：用 t-1 拍的活动去预测 t 拍的输入（延迟一拍，完全局部，零额外前向）
                if self.inp_prev is not None and self.sig_prev is not None:
                    if self.tb is not None:
                        _tgt = T.sigmoid(6.0 * (code_next[:, self.tb] - 0.5))
                    else:
                        _tgt = T.sigmoid((x @ self.V) * 3.0)
                    _e = (_tgt - self.sig_prev) * self.sig_prev * (1 - self.sig_prev)
                    _dw = T.einsum("bh,bhf->hf", _e, self.inp_prev) / B
                    self.v_w.mul_(0.999).addcmul_(_dw, _dw, value=0.001)
                    self.w.add_(_dw / (self.v_w.sqrt() + 1e-8), alpha=-self.eta)
                if inp is None:
                    raise SystemExit("--hid-selfsup 与 fan>%d 不兼容（需要存 (B,H,fan)）" % _chunk)
                self.inp_prev = inp.detach()
                self.sig_prev = sig.detach()
                e = T.zeros_like(sig)
            elif hid_teach is not None and hid_teach.numel() == self.H:
                # 下行教学信号：teach_i = Σ_j W_{j←i} · 下游误差_j（用自己的传出突触）
                e = hid_teach.to(sig.device).unsqueeze(0) * sig * (1 - sig)
            else:
                if self.tb is not None:
                    # +3 if 该位亮 else -3：对比度足够，且完全是局部可观测量
                    tgt = T.sigmoid(6.0 * (code_next[:, self.tb] - 0.5))
                else:
                    tgt = T.sigmoid((code_next @ self.v.t()) * 3.0)
                e = (tgt - sig) * sig * (1 - sig)                # (B,H) 局部误差
            if not self.self_sup:
                if inp is not None:
                    dw = T.einsum("bh,bhf->hf", e, inp) / B
                else:
                    dw = T.empty_like(self.w)
                    for _s in range(0, self.H, _chunk):
                        _e = min(self.H, _s + _chunk)
                        _inp = x[:, self.src[_s:_e]]
                        dw[_s:_e] = T.einsum("bh,bhf->hf", e[:, _s:_e], _inp) / B
                self.v_w.mul_(0.999).addcmul_(dw, dw, value=0.001)
                self.w.add_(dw / (self.v_w.sqrt() + 1e-8), alpha=-self.eta)
            # 递归边：也只用局部误差与发放
            if self.n_edges:
                pre = self.prev[self.rsrc]                    # (H,rec_fan)
                gr = (e.mean(0).unsqueeze(1) * pre) / B
                self.v_rec.mul_(0.999).addcmul_(gr, gr, value=0.001)
                self.rw.add_(gr / (self.v_rec.sqrt() + 1e-8), alpha=-self.eta_rec)
                if self.rsign is not None:
                    # Dale：学完把符号锁回去（只让强度变，不让兴奋变抑制）。rsign 只含 ±1。
                    self.rw = self.rw.abs() * self.rsign
                self.rw *= self.rmask
            # 内禀可塑性：把平均发放率稳到 target
            self.rate.mul_(0.995).add_(sig.mean(0), alpha=0.005)
            # 符号：rate 高于 target 时要**抬高**阈值让它少发放（之前写成 sub_ → 正反馈到饱和
            # rate=1.000、|theta|=180，隐藏层输出全是常数，这是上一轮 PPL 崩到 1747 的真因）
            self.theta.add_(self.eta_th * (T.log(self.rate.clamp(min=1e-6))
                                           - math.log(self.target)))
            # 睡眠式乘性下调（SHY）：与强度成比例
            self.w.mul_(1.0 - 1e-6)
            self.rw.mul_(1.0 - 1e-6)
            self._topology_step()
        # 匹配追踪：本层解释掉 sig·W_rec，残差交给下一层
        _recon = sig @ self.W_rec
        _resid = x - _recon
        self.resid_var = float(_resid.pow(2).mean())
        if learn:
            _gw = T.einsum("bh,bd->hd", sig, _resid) / B      # 只减重建误差：完全局部
            self.v_rec_w.mul_(0.999).addcmul_(_gw, _gw, value=0.001)
            self.W_rec.add_(_gw / (self.v_rec_w.sqrt() + 1e-8), alpha=self.eta)
        return sig, _resid

    def _topology_step(self):
        """剪掉弱边 + 随机补新边 → 边数随时间变化（拓扑真的在长/在调整）。"""
        if self.prune_th > 0:
            cut = self.rmask & (self.rw.abs() < self.prune_th)
            c = int(cut.sum())
            if c:
                self.rmask.masked_fill_(cut, False)
                self.rw.masked_fill_(cut, 0.0)
                self.edge_cut += c
        if self.syn_new > 0:
            pool = (~self.rmask).nonzero(as_tuple=False)
            if pool.numel():
                k = min(self.syn_new, pool.shape[0])
                pick = pool[T.randint(0, pool.shape[0], (k,), device=self.dev)]
                self.rmask[pick[:, 0], pick[:, 1]] = True
                self.rw[pick[:, 0], pick[:, 1]] = (
                    T.randn(k, device=self.dev) / math.sqrt(self.rsrc.shape[1])) * 0.05
                self.edge_born += k


class _Keep:
    def __init__(self, brain, res, bank=None):
        self.b, self.r, self.bk = brain, res, bank

    def __enter__(self):
        b = self.b
        self.s = None
        if b.h is not None:
            self.s = (b.h.clone(), b.sigma.clone(), b.spike.clone(),
                      b.spike_buf.clone() if b.fr_per else None, b.spk_ptr, b._ptr_before)
        self.rs = self.r.s.clone() if (self.r is not None and self.r.s is not None) else None
        self.bs = self.bk.a.clone() if (self.bk is not None and self.bk.a is not None) else None
        return self

    def __exit__(self, *a):
        if self.s is not None:
            b = self.b
            b.h, b.sigma, b.spike, b.spike_buf, b.spk_ptr, b._ptr_before = self.s
        if self.r is not None and self.rs is not None:
            self.r.s = self.rs
        if self.bk is not None and self.bs is not None:
            self.bk.a = self.bs
        return False


def evaluate_long(brain, corpus, res, bank, win, n_eval=2048, batch=64, stride=16,
                  seed=0, src=None, proj=None, hid=None):
    """连续多流评测：每条流连续走 warm + stride*n 步，丢掉前 warm 步。

    长上下文模型**必须**这样评：慢时间常数（τ 到 ctx_len）的滤波器要预热 ~ctx_len 步才建得起来，
    随机位置各喂 24 步的话，慢带全是 0，等于把上下文关掉了。
    """
    global hidden_learn
    hidden_learn = False
    src = corpus.he if src is None else src
    warm = int(min(8192, bank.ctx_len)) if bank is not None else 24
    n_per = max(1, n_eval // batch)
    L = warm + stride * n_per + 2
    rng = np.random.default_rng(seed)
    lo = max(win, 2) + 2
    hi = max(lo + 1, src.size - L - 4)
    pos = rng.integers(lo, hi, size=batch).astype(np.int64)
    if res is not None:
        res.reset(batch)
    if bank is not None:
        bank.reset(batch)
    brain.reset_state(batch)
    nll, top1, top5, n = 0.0, 0, 0, 0
    _nlls = []                     # 逐位置 NLL（用于跨模型配对比较，消除位置抽样噪声）
    ar = T.arange(batch, device=corpus.dev)
    with _Keep(brain, res, bank):
        for t in range(L):
            _cn = (corpus.code[T.from_numpy(np.ascontiguousarray(
                np.asarray(src[pos + 1], dtype=np.int64))).to(corpus.dev)]
                   if bank is not None or hid is not None else None)
            h = _feat(corpus, res, win, _win_toks(src, pos, win), proj, bank, hid, _cn)
            sc = brain.forward(h)
            if t >= warm and (t - warm) % stride == 0 and (t - warm) // stride < n_per:
                tgt = T.from_numpy(np.ascontiguousarray(
                    np.asarray(src[pos + 1], dtype=np.int64))).to(corpus.dev)
                p = T.softmax(sc, dim=1).clamp(min=1e-9)
                _pv = -T.log(p)[ar, tgt]
                nll += float(_pv.sum())
                _nlls.append(_pv.detach().cpu())
                top1 += int((sc.argmax(1) == tgt).sum())
                top5 += int(sc.topk(5, 1).indices.eq(tgt.unsqueeze(1)).any(1).sum())
                n += batch
            pos += 1
            pos[pos >= src.size - 2] = lo
    to = brain.topology()
    _nll_vec = (np.concatenate([x.numpy() for x in _nlls]) if _nlls else np.zeros(0))
    return {"ppl": math.exp(nll / max(1, n)), "top1": top1 / max(1, n),
            "top5": top5 / max(1, n), "_nll": _nll_vec,
            "diag": {"per_word": to["per_word"], "words_uncovered": to["words_uncovered"],
                     "fire": round(float(brain.spike.float().mean()), 4) if brain.spike is not None else 0.0,
                     "w_norm": round(float(brain.w[:brain.n_slots][brain.alive[:brain.n_slots]]
                                           .norm(dim=1).mean()), 3) if brain.N else 0.0}}


def evaluate(brain, corpus, res, win, n_batches=60, batch=64, seed=0, src=None, proj=None, warm=24,
             bank=None, hid=None):
    global hidden_learn
    hidden_learn = False
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
            for w in range(warm):
                brain.forward(_feat(corpus, res, win, _win_toks(src, idx - warm + w, win), proj, bank, hid,
                                    corpus.code[T.from_numpy(np.ascontiguousarray(np.asarray(
                                        src[idx - warm + w + 1], dtype=np.int64))).to(corpus.dev)]))
            pred = brain.forward(_feat(corpus, res, win, _win_toks(src, idx, win), proj, bank, hid,
                                        corpus.code[T.from_numpy(np.ascontiguousarray(np.asarray(
                                            src[idx + 1], dtype=np.int64))).to(corpus.dev)]))
            tgt = T.from_numpy(np.asarray(src[idx + 1], dtype=np.int64)).to(corpus.dev)
            p = T.softmax(pred, dim=1).clamp(min=1e-9)
            ar = T.arange(batch, device=corpus.dev)
            nll += float(-T.log(p)[ar, tgt].sum())
            top1 += int((p.argmax(1) == tgt).sum())
            top5 += int(p.topk(5, 1).indices.eq(tgt.unsqueeze(1)).any(1).sum())
            n += batch
    to = brain.topology()
    return {"ppl": math.exp(nll / n), "top1": top1 / n, "top5": top5 / n,
            "diag": {"per_word": to["per_word"], "words_uncovered": to["words_uncovered"],
                     "fire": round(float(brain.spike.float().mean()), 4) if brain.spike is not None else 0.0,
                     "w_norm": round(float(brain.w[:brain.n_slots][brain.alive[:brain.n_slots]]
                                           .norm(dim=1).mean()), 3) if brain.N else 0.0}}


def generate(brain, corpus, res, win, prompt_ids, n_new=60, temp=0.9, topk=0, proj=None,
             seed=0, bank=None):
    rng = np.random.default_rng(seed)
    if res is not None:
        res.reset(1)
    if bank is not None:
        bank.reset(1)
    brain.reset_state(1)
    buf = list(prompt_ids)
    warm = int(min(1024, bank.ctx_len)) if bank is not None else 0
    if warm > 0:                      # 预热上下文摘要（慢带需要 ctx_len 步才建得起来）
        seq = [buf[0]] * max(0, warm - len(buf)) + list(buf)
        for i in range(len(seq)):
            cur = seq[max(0, i - max(1, win) + 1):i + 1][::-1]
            cur = cur + [seq[0]] * (max(1, win) - len(cur))
            tk0 = np.array([cur], dtype=np.int64)
            _feat(corpus, res, win, tk0, proj, bank)
    out = []
    for _ in range(n_new):
        tk = np.array([[buf[-1 - j] if len(buf) > j else buf[0] for j in range(max(1, win))]],
                      dtype=np.int64)
        h = _feat(corpus, res, win, tk, proj, bank)
        sc = brain.forward(h)[0] / max(1e-6, temp)
        if topk > 0:
            v, i = sc.topk(topk)
            sc = T.full_like(sc, -1e9)
            sc[i] = v
        p = T.softmax(sc, dim=0).cpu().numpy().astype(np.float64)
        p = p / p.sum()
        nxt = int(rng.choice(p.size, p=p))
        out.append(nxt)
        buf.append(nxt)
    return out




# ---------------------------------------------------------------- 在线学习问答

def learn_text(brain, corpus, res, bank, ids, epochs, lr, win, proj=None):
    """在线学习：对一段文本做 teacher-forced 的逐 token 更新（局部规则，无 BP）。

    返回学习前后这段文本自己的 NLL —— 这是"在线学习是否真的学到了"的唯一判据。
    """
    if len(ids) < 2:
        return None, None
    w0 = max(1, win)

    def one_pass(learn):
        if res is not None:
            res.reset(1)
        if bank is not None:
            bank.reset(1)
        brain.reset_state(1)
        nll = 0.0
        n = 0
        for i in range(len(ids) - 1):
            cur = [ids[i - j] if i - j >= 0 else ids[0] for j in range(w0)]
            h = _feat(corpus, res, win, np.array([cur], dtype=np.int64), proj, bank)
            pred = brain.forward(h)
            tgt = T.tensor([ids[i + 1]], dtype=T.int64, device=corpus.dev)
            p = T.softmax(pred, dim=1).clamp(min=1e-9)
            nll += float(-T.log(p[0, tgt[0]]))
            n += 1
            if learn:
                brain.logits = pred
                brain.h_feat = h
                brain.learn(tgt, lr=lr)
        return nll / max(1, n)

    before = one_pass(False)
    for _ in range(max(1, int(epochs))):
        one_pass(True)
    after = one_pass(False)
    return before, after


def chat_loop(brain, corpus, res, bank, args, proj=None):
    """死脑子提问 + 在线学习问答。"""
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.art, "bpe.json"))
    except Exception as e:
        print("[chat] 读不到分词器（%s），只能按 token id 输入" % e)
        tok = None
    print("=" * 60)
    print("在线问答：直接输入文字 → 它接着写；/learn 前一段会被学习；/temp x 调温度；/quit 退出")
    print("  --chat-learn 控制每段文本学几轮（当前 %d），--chat-lr 控制学习率（当前 %.4g）"
          % (args.chat_learn, args.chat_lr))
    print("=" * 60)
    temp = 0.9
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line.startswith("/temp"):
            try:
                temp = float(line.split()[1])
            except Exception:
                pass
            print("  温度 = %.2f" % temp)
            continue
        if tok is None:
            try:
                ids = [int(x) for x in line.replace(",", " ").split()]
            except Exception:
                print("  这行不是 token id 序列")
                continue
        else:
            ids = tok.encode(line).ids
        if len(ids) < 2:
            print("  太短")
            continue
        if args.chat_learn > 0:
            b, a = learn_text(brain, corpus, res, bank, ids, args.chat_learn, args.chat_lr,
                              args.win, proj)
            print("  [在线学习] 这段文本自己的 NLL：%.3f → %.3f nats（PPL %.0f → %.0f）"
                  % (b, a, math.exp(b), math.exp(a)))
        out = generate(brain, corpus, res, args.win, ids, n_new=args.chat_len, temp=temp,
                       topk=0, proj=proj, bank=bank)
        if tok is not None:
            print("它> " + line + tok.decode(out))
        else:
            print("它> " + line + " | " + " ".join(map(str, out)))


# ---------------------------------------------------------------- 主流程


def plan_capacity(vram_gb, ctx_len, ctx_band, ctx_dim, sm, win, y=None, fill=0.85):
    """按显存算 x（上下文）与 y（神经元数）。

    显存 ≈ cap × D × 16 B，其中
        cap = y + max(4096, y/8)          （槽位只追加，留 1/8 余量给周转）
        D   = win×SM + K×(ctx_dim>0 ? ctx_dim : SM)
        24 B = w + v_w + gW + den 四个 (n,D) 张量 + allocator 碎片余量
              （实测 16 B 偏乐观：y=73784/D=10240 时峰值 16.36GB 触到 16GB 上限）
    给定 y 就报"最多能支撑多长上下文"，给定 ctx 就报"最多能放多少神经元"。
    """
    D = win * sm + ctx_band * (ctx_dim if ctx_dim > 0 else sm)
    budget = vram_gb * 1e9 * fill
    slots = budget / (D * 24.0)
    y_max = int(slots / 1.125)
    if y is None:
        return {"D": D, "y_max": y_max, "cap": int(y_max * 1.125)}
    # 反之：给定 y，算能支撑的最大 D，再反推 ctx
    slots = y * 1.125
    d_max = budget / (slots * 24.0)
    k_max = max(1, int((d_max - win * sm) / (ctx_dim if ctx_dim > 0 else sm)))
    return {"D": D, "y_max": y_max, "cap": int(y * 1.125), "d_max": int(d_max), "k_max": k_max}


def main():
    global hidden_learn
    hidden_learn = True
    p = argparse.ArgumentParser()
    p.add_argument("--art", default=ART)
    p.add_argument("--mem", choices=["window", "res", "both"], default="window")
    p.add_argument("--chunk-grad", dest="chunk_grad", type=int, default=0,
                   help="1=读出梯度按行分块计算并就地更新，避免整块 (n,D) gW 峰值；仅简单加性路径")
    p.add_argument("--mem-debug", dest="mem_debug", type=int, default=0,
                   help="1=打印 init/forward/learn 的 CUDA 分配与峰值，不改变计算")
    p.add_argument("--win", type=int, default=4)
    p.add_argument("--n-res", dest="n_res", type=int, default=1536)
    p.add_argument("--ctx-tau", dest="ctx_tau", type=float, default=4.0)
    p.add_argument("--feat-dim", dest="feat_dim", type=int, default=0)
    p.add_argument("--y", type=int, default=16384, help="读出皮层细胞上限 y")
    p.add_argument("--fr", type=int, default=0, help="读出层递归边数（脉冲驱动；0=关）")
    p.add_argument("--delay-steps", dest="delay_steps", type=int, default=8)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.003,
                   help="峰值学习率；按 1/√(t/T) 衰减（标准 schedule，不是手调魔数）")
    p.add_argument("--plast-tau", dest="plast_tau", type=float, default=0.0,
                   help="① 细胞可塑性随年龄衰减的时间常数（步）；0=关。建议取 tokens/batch/8")
    p.add_argument("--out-code", dest="out_code", type=int, default=0,
                   help="1=输出矩阵固定为码表（零可学参数，约束A 由构造满足），隐含冻结读出")
    p.add_argument("--logodds", dest="logodds", type=int, default=0,
                   help="1=特征取 log-odds（伯努利似然 logit_v=c_v·logit(a)+常数），零尺度常数")
    p.add_argument("--apical-mode", dest="apical_mode", choices=["none", "bit", "causal"], default="none",
                   help="顶端教学流形态：none=关；bit=下一 token 的稀疏码位目标（分支A）；"
                        "causal=按词因果角色投递 (y_v-p_v)（分支B）。只作局部调制，不新增隐藏层/权重")
    p.add_argument("--coord-swap", dest="coord_swap", type=int, default=0,
                   help="1=R168-C 频率 confound 对照：交换两组的块排列（高频半区随机、低频半区保留）")
    p.add_argument("--coord-groups", dest="coord_groups", type=int, default=0,
                   help="R168-C 的组数：0=由 cap//V 自动导出；>0=显式组数（用于细粒度坐标采样）")
    p.add_argument("--coord-mode", dest="coord_mode", choices=["all", "lag", "band", "rand", "randall"], default="all",
                   help="R168-C 的坐标多样性来源：all=全部块循环移位；lag=只转 lag 块；"
                        "band=只转带通块；rand=组间固定随机块序（种子来自已有 rng）")
    p.add_argument("--coord-sample", dest="coord_sample", type=int, default=0,
                   help="1=R168-C：按 SM 切块、不同细胞组看不同块级循环移位（结构化稀疏，"
                        "无随机 gather、无新手填常数、不新增可学权重）")
    p.add_argument("--apical-norm", dest="apical_norm", choices=["none", "sat", "rms"], default="none",
                   help="顶端教学流的量纲：none=原值；sat=t/(1+|t|)（受体饱和，1 为恒等基线）；"
                        "rms=按本 batch 的均值/标准差标准化（方差匹配，沿用 w_init 原则）。均无手填常数")
    p.add_argument("--freeze-readout", dest="freeze_readout", type=int, default=0,
                   help="1=冻结读出权重（梯度清零）。检验文献20 约束A：冻住出口后，计算还能否由内部接线承载")
    p.add_argument("--pair-bind", dest="pair_bind", type=int, default=0,
                   help="1=加一路二阶合取码：z=T1[x_{t-1}]*T2[x_{t-2}] 再 k-WTA，拼在特征末尾。"
                        "T1/T2 是固定随机投影表（零可学参数、k_pair=k）。"
                        "依据文献37：人脑字母对由独立稀疏群编码，组合住在表征里而不是读出权重里")
    p.add_argument("--pair-lags", dest="pair_lags", default="1,2",
                   help="要绑定的 lag 对，分号分隔，如 \"1,2\" 或 \"1,2;1,3;2,3\"。每个对给一路合取块（各宽 SM）")
    p.add_argument("--tier-ctx", dest="tier_ctx", type=int, default=0,
                   help="1=出生序决定时间视界：第 t 档只看到 ctx_len^((t+1)/T) 以内的带通，"
                        "T=y//V 档。用途：让晚出生的细胞**功能上**不同（长上下文专家），"
                        "而不是只改学习速率。槽位 i → 词 = i%%V、档 = i//V")
    p.add_argument("--plast-norm", dest="plast_norm", type=int, default=0,
                   help="1=把年龄依赖可塑性按在世细胞均值归一（总可塑性守恒，只保留相对差异）。"
                        "不归一时 1/(1+age/τ) 会把整网学习率一起压到 0，长跑会自己冻结")
    p.add_argument("--upd", choices=["add", "eg"], default="add",
                   help="权重更新：add=加法(默认)；eg=指数化梯度(乘性+符号锁定+重尾)")
    p.add_argument("--shy", type=float, default=0.0,
                   help="睡眠式乘性下调速率；0=自动取学习率")
    p.add_argument("--code-learn", dest="code_learn", type=int, default=0,
                   help="1=在线学码：用同一条局部规则更新输入码（k-WTA + L2 归一保稀疏/有界）")
    p.add_argument("--code-lr", dest="code_lr", type=float, default=0.0,
                   help="码的学习率；0=与读出 lr 相同")
    p.add_argument("--eta-temp", dest="eta_temp", type=float, default=0.0,
                   help=">0 时用**训练似然的梯度**在线拟合温度（∂NLL/∂log g = −g(s_y − E_p[s])），"
                        "替代置信度匹配判据（后者实测过冲、损失约 28% PPL）")
    p.add_argument("--eta-cal", dest="eta_cal", type=float, default=0.01,
                   help="置信度匹配控制器的速率；0=关（会重现 PPL 随训练变差）")
    p.add_argument("--b-noise", dest="b_noise", type=float, default=0.1,
                   help="出生时每个细胞的兴奋性抖动（打破同词细胞对称；>0.3 才真正分化）")
    p.add_argument("--half-feat", dest="half_feat", type=int, default=0,
                   help="③ 相邻细胞各用一半特征维（感受野分工，打破同词对称）")
    p.add_argument("--freeze-at", dest="freeze_at", type=int, default=0,
                   help="到这一步之后停止学习（诊断：区分退化来自更新规则还是数据流）")
    p.add_argument("--rms-eps", dest="rms_eps", type=float, default=1e-8,
                   help="RMSprop 的 eps；稀疏码下小梯度元素会被它放大，默认 1e-8 与历史一致")
    p.add_argument("--eta-b", dest="eta_b", type=float, default=0.02,
                   help="内禀可塑性速率（对数域发放率稳态）")
    p.add_argument("--w-init", dest="w_init", type=float, default=1.0)
    p.add_argument("--budget", type=float, default=4.0,
                   help="每细胞输入突触的 L2 半径（总强度守恒；0=关闭，会正反馈塌陷）")
    p.add_argument("--grow-every", dest="grow_every", type=int, default=250)
    p.add_argument("--grow-rate", dest="grow_rate", type=float, default=1.0)
    p.add_argument("--death-rate", dest="death_rate", type=float, default=0.0003)
    p.add_argument("--turnover-rate", dest="turnover_rate", type=float, default=0.0,
                   help="成熟期周转率：每事件把最差的这一比例的细胞就地重生（死亡+新生）")
    p.add_argument("--vram", type=float, default=16.0,
                   help="可用显存(GB)。配合 --auto-y 自动算 y；总会打印容量表")
    p.add_argument("--auto-y", dest="auto_y", action="store_true",
                   help="按 --vram 自动定 y（覆盖 --y）")
    p.add_argument("--vram-fill", dest="vram_fill", type=float, default=0.85,
                   help="显存占用目标比例（留余量给碎片）")
    p.add_argument("--hidden", type=int, default=0, help="脉冲隐藏层细胞数 H（0=关）；拓扑的宿主")
    p.add_argument("--hid-teach", dest="hid_teach", type=int, default=0,
                   help="1=隐藏层用下行教学信号(读出误差经自己的传出突触回传)；0=用随机投影目标")
    p.add_argument("--nll-out", dest="nll_out", default="",
                   help="把最终评测的逐位置 NLL 存成 .npy（用于跨模型配对比较）")
    p.add_argument("--eval-temp", dest="eval_temp", type=float, default=0.0,
                   help="评测时用温度 T（设 gain_t=1/T），用于量校准余量；0=用存档里的增益")
    p.add_argument("--eval-n", dest="eval_n", type=int, default=0,                   help="最终评测的位置数（0=按 eval_batches 推）")
    p.add_argument("--kill-mode", dest="kill_mode", choices=["rand", "one"], default="rand",
                   help="rand=随机杀 kill_frac 比例的细胞（看整体退化）；"
                        "one=**每个多细胞词只杀掉其中一个**（看'剩一个还行不行'）——"
                        "用来区分同词细胞是**冗余**还是**互补**："
                        "只剩一个时的 PPL 若≈单细胞模型，说明每个都能独立撑住（互补/集成）；"
                        "若明显更差，说明它们分工了、必须成对才好")
    p.add_argument("--kill-frac", dest="kill_frac", type=float, default=0.0,
                   help="评测前随机杀掉这个比例的读出细胞（鲁棒性判据，文献20 约束B）。"
                        "损伤种子固定=12345，可复现；只用于 --load 的存档")
    p.add_argument("--hid-lag-pair", dest="hid_lag_pair", default="",
                   help="把隐藏层扇入限制在两个 lag 块上（如 \"1,2\"）⇒ 重合检测器，这是三元语法所必需的乘法交互")
    p.add_argument("--hid-casc", dest="hid_casc", type=int, default=0,
                   help="级联池化层数 L（0=关）：bands 按尺度分给 L 层，感受野逐级扩大、细胞数逐级减半")
    p.add_argument("--hid-casc-r", dest="hid_casc_r", type=float, default=0.6,
                   help="级联各层大小的公比（默认 0.6：逐级减半略慢）")
    p.add_argument("--hid-layers", dest="hid_layers", type=int, default=1,
                   help="串联几层（>1 时自动改用每层自监督目标，破解双向死锁）")
    p.add_argument("--hid-selfsup", dest="hid_selfsup", type=int, default=0,
                   help="1=每层自监督（预测自己下一拍收到的输入），不看下游误差")
    p.add_argument("--hid-mode", dest="hid_mode", choices=["add", "gate", "only", "replace"], default="add",
                   help="add=拼接进特征；gate=乘到特征上（以稳态为中心，D 不变）")
    p.add_argument("--hid-fan", dest="hid_fan", type=int, default=32, help="隐藏细胞输入扇入")
    p.add_argument("--hid-rec", dest="hid_rec", type=int, default=8, help="隐藏细胞递归入边数")
    p.add_argument("--hid-target", dest="hid_target", type=float, default=0.05, help="隐藏层目标发放率")
    p.add_argument("--hid-syn", dest="hid_syn", type=int, default=0, help="每步新补的递归边数")
    p.add_argument("--norm-score", dest="norm_score", type=int, default=0,
                   help="0=不归一（需靠 gain 补偿，禁止）；1=按突触 L2 强度归一；"
                        "2=除以群体驱动标准差（除法归一化，尺度不变、无需增益）")
    p.add_argument("--dz-mode", dest="dz_mode", choices=["cell", "word"], default="cell",
                   help="word=教学信号取词级误差 (p−y)（不乘先验地板带来的词内归一），cell=旧（乘 e^u/λ）")
    p.add_argument("--grp", dest="grp", type=int, default=0,
                   help="G>0：码位分 G 段，在**跨 lag 的同段**之间做重合交互（参数 O(n·G)，这份数据学得动）")
    p.add_argument("--grp-trace", dest="grp_trace", type=int, default=0,
                   help="1=交互权重 θ 用**跨 token 资格迹**（三因子规则）：一次误差能强化一整段历史的共激活")
    p.add_argument("--trace-h", dest="trace_h", type=int, default=0,
                   help="迹的时间常数（token）；0=取 win（迹与上下文窗口同长，不引入新常数）")
    p.add_argument("--grp-pairs", dest="grp_pairs", default="0,1;0,2;1,2",
                   help="要交互的 lag 对，分号分隔")
    p.add_argument("--mod", dest="mod", type=int, default=0,
                   help="M>0 时启用共享模块张成的树突交互（u += (α·P h)(β·P h)）："
                        "交互权重 O(M·D) 而非 O(SM²)，对应皮层结构原型(约1/40)")
    p.add_argument("--gate-lag", dest="gate_lag", type=int, default=0,
                   help="1=每个 lag 一个上下文相关的标量门（增益调制）⇒ 可表达 KN 式的上下文回退")
    p.add_argument("--n-lag", dest="n_lag", type=int, default=1, help="门的个数（=近窗 lag 数）")
    p.add_argument("--gate-soft", dest="gate_soft", type=int, default=0,
                   help="1=lag 间用 softmax（Σg=1，信任预算守恒=KN 混合形式），0=各自独立 sigmoid")
    p.add_argument("--gate-full", dest="gate_full", type=int, default=0,
                   help="1=门看全部上下文（可表达跨 lag 的乘性结构，参数仍只有 n_lag×D）")
    p.add_argument("--lag-w", dest="lag_w", type=int, default=0, help="每个 lag 块的宽度（=码维 SM）")
    p.add_argument("--dend", dest="dend", type=int, default=0,
                   help="1=读出细胞两个树突区室相乘（u = w·h + (a·h)(b·h)）⇒ 重合检测，补上加性模型缺的乘积项")
    p.add_argument("--readout", dest="readout",
                   choices=["linear", "poisson", "lse", "mix", "laggate"],
                   default="linear",
                   help="poisson=首脉冲/泊松率叠加 log(Σe^u + e^{b_v})（无 n0、无 clamp、天然有界）；"
                        "lse=旧形式（按 n_v+n0 归一，仅用于复现历史数字）")
    p.add_argument("--unit-rms", dest="unit_rms", type=int, default=0,
                   help="1=把权重约束在 RMS 突触权重=1 的球面上（半径 √D，单位约定、非调参）："
                        "u=w·h 天然落在 ±√(活跃输入数) 的量级，永远不需要全局增益")
    p.add_argument("--unit-l1", dest="unit_l1", type=int, default=0,
                   help="1=权重投影到 L1 单位球（Σ|w|=1，真正的双向约束，不是单向夹）")
    p.add_argument("--rfan", dest="rfan", type=int, default=0,
                   help="每个读出细胞的输入扇入数（固定稀疏接线）；0=全连接")
    p.add_argument("--topk", dest="topk", type=int, default=0,
                   help="每个读出细胞只保留绝对值最大的 K 个突触（0=不限）；限制记忆容量、产生重尾稀疏")
    p.add_argument("--topk-every", dest="topk_every", type=int, default=25,
                   help="每多少步做一次 Top-K 稀疏化")
    p.add_argument("--hid-autotune", dest="hid_autotune", type=int, default=0,
                   help="1=第一次前向时按实测 u 的分布定 θ（取 1−target 分位）与 temp（取 std），"
                        "使初始发放率=目标、log-odds 起始 O(1)。方差匹配，不引入新常数")
    p.add_argument("--hid-target-mode", dest="hid_target_mode", choices=["proj", "bit"],
                   default="proj",
                   help="proj=预测下一码的随机投影（等价于学边际，与线性读出冗余）；bit=每个细胞盯住一个具体码位")
    p.add_argument("--hid-in", dest="hid_in", choices=["all", "code"], default="all",
                   help="code=隐藏层只看稀疏窗口码（不吞稠密带通），可形成跨 lag 联合检测器且不丢信息")
    p.add_argument("--hid-add-scale", dest="hid_add_scale", type=float, default=1.0,
                   help="隐藏层输出拼进读出的缩放（1=不缩放；之前归一化到范数1 相当于 1/25）")
    p.add_argument("--dropout", dest="dropout", type=float, default=0.0,
                   help="输入特征的随机失活概率（突触释放失败），只在训练时生效")
    p.add_argument("--hid-geo", dest="hid_geo", type=int, default=0,
                   help="1=按生成规则接（位置梯度+模块原型）；0=全局随机（旧）")
    p.add_argument("--hid-knb", dest="hid_knb", type=float, default=32,
                   help="位置梯度：一个细胞的特征邻域内期望有多少个细胞（反解 λ）")
    p.add_argument("--hid-mod", dest="hid_mod", type=float, default=40,
                   help="模块原型：每多少个细胞一个模块（文献：果蝇 8000 种→200 个原型 ≈ 1/40）")
    p.add_argument("--hid-dale", dest="hid_dale", type=int, default=0,
                   help="1=按 Dale 定律给递归边符号（含抑制性 → 横向竞争）")
    p.add_argument("--hid-inh-frac", dest="hid_inh_frac", type=float, default=0.2,
                   help="抑制性细胞比例（皮层约 0.2）")
    p.add_argument("--hid-inh-gain", dest="hid_inh_gain", type=float, default=3.0,
                   help="抑制性边相对兴奋性边的强度倍数")
    p.add_argument("--hid-rec-scale", dest="hid_rec_scale", type=float, default=1.0,
                   help="递归边相对前馈的强度倍数（1=原值 0.1；10≈与前馈同量级）")
    p.add_argument("--hid-eta-rec", dest="hid_eta_rec", type=float, default=0.001,
                   help="递归边学习率")
    p.add_argument("--hid-wout", dest="hid_wout", type=float, default=0.1,
                   help="模块间连接的相对概率（模块内=1）")
    p.add_argument("--hid-prune", dest="hid_prune", type=float, default=0.002,
                   help="递归边剪枝阈值（|w|<阈值 就断开）；0=不剪（固定拓扑）")
    p.add_argument("--ctx-len", dest="ctx_len", type=int, default=512,
                   help="上下文长度（token）：1..任意大。多尺度带通压缩，显存与它无关")
    p.add_argument("--ctx-band", dest="ctx_band", type=int, default=10,
                   help="尺度层数 K（τ 在 1..ctx_len 上对数等距）")
    p.add_argument("--ctx-dim", dest="ctx_dim", type=int, default=0,
                   help="每个尺度带压缩到的维度 d；0=不压缩直接用码（推荐）")
    p.add_argument("--n0", type=float, default=0.1,
                   help="先验虚拟细胞的等效个数（t=0 时无论取多少都精确等于 unigram）")
    p.add_argument("--code-win", dest="code_win", default="",
                   help="诊断用：窗口输入的码文件（默认同 --code-path）。可给 one-hot 恒等码以量"
                        "输入编码的信息上限，而带通仍用 --code-path 的稀疏码")
    p.add_argument("--code-path", dest="code_path", default="",
                   help="换一份稀疏码（扫 SM/共享位用）")
    p.add_argument("--in-code", dest="in_code", action="store_true",
                   help="输入用稀疏分布式码（SM×k 二值）而不是 256 维稠密嵌入")
    p.add_argument("--init-n", dest="init_n", type=int, default=0,
                   help="t=0 就预置这么多细胞（对照组 A：一开始就给 N；默认 0=从 1 长起）")
    p.add_argument("--grow-target", dest="grow_target", choices=["err", "rand"], default="err",
                   help="出生位置：err=高误差词优先（默认）；rand=随机（对照组 C）")
    p.add_argument("--grow2-frac", dest="grow2_frac", type=float, default=0.9)
    p.add_argument("--phase2-on-y", dest="phase2_on_y", type=int, default=0,
                   help="1=phase2 以目标 y 为门槛（否则沿用旧 cap 门槛；旧门槛 cap>y 永不触发）")
    p.add_argument("--rec-birth", dest="rec_birth", type=int, default=0,
                   help="1=fr>0 时出生即建立初始递归边（否则只在 synaptogenesis 里建，phase2 未开时为空）")
    p.add_argument("--syn-rate", dest="syn_rate", type=float, default=0.0)
    p.add_argument("--max-events", dest="max_events", type=int, default=800)
    p.add_argument("--grace", type=float, default=200.0)
    p.add_argument("--log-every", dest="log_every", type=int, default=0)
    p.add_argument("--eval-batches", dest="eval_batches", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    p.add_argument("--save-every", dest="save_every", type=int, default=0,
                   help="每多少步存一次（配合 --load 续跑，绕开长跑卡死的环境问题）")
    p.add_argument("--lr-offset", dest="lr_offset", type=int, default=0,
                   help="续跑时把 lr schedule 的起点往后挪，避免每段都重置 lr")
    p.add_argument("--curve-out", dest="curve_out", default="")
    p.add_argument("--load", default="")
    p.add_argument("--sample", default="")
    p.add_argument("--sample-n", dest="sample_n", type=int, default=40)
    p.add_argument("--dev-tokens", dest="dev_tokens", type=int, default=0,
                   help="从训练语料尾部切这么多 token 作 dev（超参只能在这上面选，不得用留出集）")
    p.add_argument("--eval-src", dest="eval_src", choices=["heldout", "dev"], default="heldout",
                   help="周期评测用哪一段：dev=调参用；heldout=只最后报一次")
    p.add_argument("--chat", action="store_true", help="进入在线问答（死脑子提问 + 在线学习）")
    p.add_argument("--chat-learn", dest="chat_learn", type=int, default=3,
                   help="每段输入学几轮（0=纯离线提问）")
    p.add_argument("--chat-lr", dest="chat_lr", type=float, default=0.0008,
                   help="在线学习的学习率")
    p.add_argument("--chat-len", dest="chat_len", type=int, default=40,
                   help="每次续写多少 token")
    p.add_argument("--eval-only", dest="eval_only", action="store_true",
                   help="只评测/生成（配合 --load），不训练")
    args = p.parse_args()
    global MEM_DEBUG
    MEM_DEBUG = bool(args.mem_debug)

    if args.device.startswith("cuda"):
        if not T.cuda.is_available():
            raise SystemExit("CUDA 不可用")
        free, total = T.cuda.mem_get_info()
        log("GPU 空闲 %.2f / %.2f GB" % (free / 1e9, total / 1e9))

    corpus = Corpus(args.art, args.mem, args.win, args.device,
                   dev_tokens=args.dev_tokens)
    if getattr(args, "code_win", ""):
        # 先把原始码留作带通用，再让窗口用新的码
        c0 = np.load(args.code_path).astype(np.float32) if args.code_path else None
        cw = np.load(args.code_win).astype(np.float32)
        corpus.code = T.from_numpy(cw).to(corpus.dev)
        corpus.SM = int(cw.shape[1])
        corpus.in_code = True
        if c0 is not None:
            corpus.code_b = T.from_numpy(c0).to(corpus.dev)
        log("诊断：窗口码=%s（SM=%d），带通码=%s（SM=%d）" % (
            os.path.basename(args.code_win), cw.shape[1],
            os.path.basename(args.code_path) if args.code_path else "emb",
            c0.shape[1] if c0 is not None else corpus.D0))
    elif args.code_path:
        c = np.load(args.code_path).astype(np.float32)
        corpus.code = T.from_numpy(c).to(corpus.dev)
        corpus.SM, corpus.k = int(c.shape[1]), int(round(float(c.sum(1).mean())))
        log("换成稀疏码 %s（SM=%d k=%d，期望共享位 k²/SM=%.1f）" % (
            args.code_path, corpus.SM, corpus.k, corpus.k ** 2 / corpus.SM))
    res = None
    if args.mem in ("res", "both"):
        res = Reservoir(corpus, args.n_res, args.ctx_tau, device=args.device)
        log("储层 %d 单元，τ=%.1f（α=%.3f），谱半径=1，不可学" % (res.N, res.tau, res.alpha))
    if args.in_code:
        if corpus.code is None:
            raise SystemExit("没有 code.npy，无法 --in-code")
        corpus.in_code = True
        log("输入编码 = 稀疏分布式码 SM=%d k=%d（本语料自训，固定不可学）" % (corpus.SM, corpus.k))
        # --lag-w 必须等于码宽，否则所有"按 lag 分块"的机制（门控/段对/在线学码）会静默失效。
        # 这是本项目踩过的坑：开关开着、机制其实没建起来，结果被当成"机制无效"。
        if args.gate_lag or args.grp or args.code_learn or args.readout in ("mix", "laggate"):
            if args.lag_w != corpus.SM:
                raise SystemExit(
                    "--lag-w 必须等于码宽 SM=%d（当前 %d）：否则门控/段对/学码会静默失效"
                    % (corpus.SM, args.lag_w))
    D_in = corpus.SM if args.in_code else corpus.D0
    # ---- 容量规划：把"16G 显存"翻译成 x（上下文）与 y（神经元） ----
    _sm = corpus.SM if corpus.code is not None else corpus.D0
    _plan = plan_capacity(args.vram, args.ctx_len, args.ctx_band, args.ctx_dim, _sm, args.win,
                          y=args.y, fill=args.vram_fill)
    log("容量：显存 %.0fGB（目标 %.0f%%）D=%d，y 上限 %d；按 y=%d 算，最多 %d 个尺度带"
        % (args.vram, 100 * args.vram_fill, _plan["D"], _plan["y_max"], args.y,
           _plan.get("k_max", -1)))
    if args.auto_y:
        args.y = max(1024, _plan["y_max"])
        log("--auto-y：按显存把 y 定为 %d" % args.y)
    # 硬闸：y 超过这个 D 允许的上限时**直接退出**，不许"先跑起来再爆显存"。
    # 事故：SM=8192 时 D=114688、y 上限只有 4391，我照旧传 --y 8192，
    # 生长到 t≈3250 后继续往上长，直接把 16GB 显存打爆（2026-11-04）。
    if args.y > _plan["y_max"]:
        raise SystemExit(
            "y=%d 超出当前 D=%d 的显存上限 y_max=%d。三种改法：降 --y、"
            "降 --ctx-band（砍带通块）、或降码宽 SM。不要硬跑。"
            % (args.y, _plan["D"], _plan["y_max"]))

    bank = None
    if args.ctx_len and args.ctx_len > 0:
        _cb = getattr(corpus, "code_b", None)
        bank = CtxBank(corpus, args.ctx_len, args.ctx_band, args.ctx_dim, device=args.device,
                       sm_override=(int(_cb.shape[1]) if _cb is not None else 0))
        log("上下文 = %d token：%d 个对数等距尺度 tau=%.0f..%.0f，每带 %s；摘要维度 %d，"
            "显存与 ctx_len 无关" % (
                args.ctx_len, bank.K, bank.taus[0], bank.taus[-1],
                "不压缩（直接用码）" if args.ctx_dim <= 0 else ("压到 %d 维" % args.ctx_dim),
                bank.out_dim))
    global hid_teach, hid_teach_on, hid_teach_size, dropout_p, add_scale
    hid_teach_on = bool(args.hid_teach)
    dropout_p = float(args.dropout)
    add_scale = float(args.hid_add_scale)
    global hid_in_dim, hid_w_start, lag_dim
    hid_in_dim = (args.win * D_in) if getattr(args, "hid_in", "all") == "code" else 0
    hid_teach_size = 0
    hid = None
    if args.hidden > 0:
        _din = args.win * D_in + (bank.out_dim if bank is not None else 0)
        _din_h = (args.win * D_in) if getattr(args, "hid_in", "all") == "code" else _din
        _band0 = args.win * D_in
        if args.hid_casc > 0:
            # ---------------- 级联池化：bands 按尺度分给 L 层 ----------------
            _K = bank.K if bank is not None else 0
            _bow = (bank.out_dim // _K) if _K else 0
            _L = int(args.hid_casc)
            _r = float(args.hid_casc_r)
            _w = [_r ** i for i in range(_L)]
            _sw = sum(_w)
            _tot = _K * _bow
            _Hs = [int(_tot * x / _sw) for x in _w]
            _Hs[-1] += _tot - sum(_Hs)
            # band 分组（按层大小比例，保证每层至少一个 band）
            _bounds, _acc, _prev = [], 0.0, 0
            for _i in range(_L):
                _acc += _w[_i]
                _e = min(_K, max(_prev + 1, int(round(_K * _acc / _sw))))
                _bounds.append((_prev * _bow, _e * _bow))
                _prev = _e
            _bounds[-1] = (_bounds[-1][0], _tot)
            _hl_list = []
            for _i in range(_L):
                _b0, _b1 = _bounds[_i]
                # 层间池化把 H_{i-1} **降到** H_i（这就是"池化"），所以第 i 层输入宽 = H_i + band 宽
                _d = (_Hs[_i] if _i > 0 else 0) + (_b1 - _b0)
                _lyr = HiddenLayer(
                    _d, corpus.SM, H=_Hs[_i], fan=args.hid_fan,
                    rec_fan=args.hid_rec, target=args.hid_target,
                    syn_new=args.hid_syn, prune_th=args.hid_prune,
                    geo=args.hid_geo, geo_knb=args.hid_knb, geo_mod=args.hid_mod,
                    geo_wout=args.hid_wout, rec_scale=args.hid_rec_scale,
                    dale=args.hid_dale, inh_frac=args.hid_inh_frac, inh_gain=args.hid_inh_gain,
                    eta_rec=args.hid_eta_rec,
                    self_sup=(1 if args.hid_selfsup or _L > 1 else 0),
                    target_mode=args.hid_target_mode, autotune=args.hid_autotune,
                    mode=args.hid_mode, device=args.device)
                _lyr._band_step = _bow          # 用于"尺度混杂"诊断
                _hl_list.append(_lyr)
            hid = Cascade(_hl_list, _bounds, mode=args.hid_mode,
                          band0=_band0, band1=_band0 + _tot)
            log("级联池化：%d 层，每层 H=%s，band 切分=%s，模式=%s；"
                "被消费的 %d 维 band 从读出输入移除 → 信息必须流经级联" % (
                    _L, _Hs, _bounds, args.hid_mode, _tot))
        else:
          _hids = []
          # 匹配追踪：每层解释的是**同一个输入空间**里的残差，所以每层的 D_in 都等于原始特征维，
          # 不是上一层的 H。（之前写成 _prev_dim = args.hidden → 第二层维度就对不上了。）
          _prev_dim = _din_h
          for _li in range(max(1, int(args.hid_layers))):
            _hids.append(HiddenLayer(
                _prev_dim, corpus.SM, H=args.hidden, fan=args.hid_fan,
                rec_fan=args.hid_rec, target=args.hid_target,
                syn_new=args.hid_syn, prune_th=args.hid_prune,
                geo=args.hid_geo, geo_knb=args.hid_knb, geo_mod=args.hid_mod,
                geo_wout=args.hid_wout, rec_scale=args.hid_rec_scale,
                dale=args.hid_dale, inh_frac=args.hid_inh_frac, inh_gain=args.hid_inh_gain,
                eta_rec=args.hid_eta_rec,
                self_sup=(1 if (args.hid_selfsup or args.hid_layers > 1) else 0),
                target_mode=args.hid_target_mode, autotune=args.hid_autotune,
                mode=args.hid_mode, device=args.device))
          hid = _hids if len(_hids) > 1 else _hids[0]
        if hid_teach_on:
            hid_teach_size = hid.H        # 用 None 起步：learn 第一拍之后才有信号
        # ---- 重合检测（coincidence detection）：把隐藏层扇入限制在两个不同 lag 的码上 ----
        # 为什么必需：读出对 [code_t, code_{t-1}, ...] 是**线性**的，分数是各 lag 的**加法**，
        # 加法模型表达不出"u 在 lag1 且 w 在 lag2"（那是乘积）⇒ 上限只到 bigram。
        # 生物学依据：树突的乘法交互（NMDA 尖峰、缝隙连接、远端/近端区室半独立计算）——
        # 单个神经元可以是"重合检测器"，这是"神经元不止是简单单元"的具体形式。
        if getattr(args, "hid_lag_pair", ""):
            _pa, _pb = [int(x) for x in args.hid_lag_pair.split(",")]
            _SM = corpus.SM
            _layers_ = hid.layers if isinstance(hid, Cascade) else (hid if isinstance(hid, list) else [hid])
            for _lyr in _layers_:
                if _lyr.D_in < (_pb + 1) * _SM:
                    continue
                _fan = _lyr.src.shape[1]
                _half = _fan // 2
                _g = T.Generator().manual_seed(7777 + _pa * 31 + _pb)
                _s = _lyr.src.clone()
                _s[:, :_half] = T.randint(_pa * _SM, (_pa + 1) * _SM, (_lyr.H, _half),
                                          generator=_g).to(_lyr.dev)
                _s[:, _half:] = T.randint(_pb * _SM, (_pb + 1) * _SM, (_lyr.H, _fan - _half),
                                          generator=_g).to(_lyr.dev)
                _lyr.src = _s
            log("隐藏层扇入限制在 lag%d × lag%d 的码上（重合检测）：每个细胞只看这两个 lag 的输入" % (_pa, _pb))
        _h0 = hid.layers[0] if isinstance(hid, Cascade) else (hid[0] if isinstance(hid, list) else hid)
        if not isinstance(hid, Cascade):
            log("脉冲隐藏层：%d 层 × H=%d，目标发放率 %.0f%%，递归边 %d（每细胞 %d 条）；%s"
                "边会剪枝/补生 → 拓扑随时间变化" % (
                    max(1, args.hid_layers), _h0.H, 100 * _h0.target, _h0.n_edges, args.hid_rec,
                    "目标=下行教学信号(灰区) / " if hid_teach_on else "目标=随机投影预测 / "))
    if isinstance(hid, Cascade):
        # replace: band 被级联取代 → D = band 之前的原始特征 + 级联总宽
        if args.hid_mode == "replace":
            D_hid = hid.H - (hid.band1 - hid.band0)
        elif args.hid_mode == "add":
            D_hid = hid.H
        else:
            D_hid = -1          # 占位，下面覆盖
    elif args.hid_mode == "add":
        D_hid = args.hidden
    elif args.hid_mode == "gate":
        D_hid = 0
    else:                       # only：D = H
        D_hid = 0
    D_bank = bank.out_dim if bank is not None else 0
    # ---- D1：出生序 → 时间视界（分档）----
    _tier = None
    if args.tier_ctx:
        if bank is None or bank.K <= 0:
            raise SystemExit("--tier-ctx 需要 --ctx-len > 0（没有带通就没有时间尺度可分）")
        _V = int(corpus.V)
        _T = max(1, int(args.y) // _V)
        _K = int(bank.K)
        _bow = int(bank.out_dim // _K)
        _lagd = int(args.win * D_in)
        _ctx = float(args.ctx_len)
        _taus = [float(t) for t in bank.taus]
        # 每档的时间视界 = 把 ctx_len 对数等分（与带通自己的对数 τ 同一把尺子，无新常数）
        _hor = [_ctx ** ((i + 1) / _T) for i in range(_T)]
        _b = [int(sum(1 for t in _taus if t <= h + 1e-9)) for h in _hor]
        _tier = {"V": _V, "T": _T, "K": _K, "bow": _bow, "lagdim": _lagd, "b": _b}
        log("时间尺度分档：%d 档（每档覆盖整个词表一次），视界=%s，允许带数=%s（共 %d 带）"
            % (_T, [round(h, 1) for h in _hor], _b, _K))
    # ---- A1：二阶合取码（固定随机投影 + 逐位乘 + k-WTA），零可学参数 ----
    _pair_extra = 0
    if args.pair_bind:
        if args.tier_ctx:
            raise SystemExit("--pair-bind 与 --tier-ctx 不能同时开：分档掩码会把末尾的 pair 块挡掉")
        if corpus.code is None:
            raise SystemExit("--pair-bind 需要 --in-code（要码才能做合取）")
        _sm = int(corpus.SM)
        _gg = T.Generator(device="cpu").manual_seed(int(args.seed) + 991)
        _p1 = (T.randn(_sm, _sm, generator=_gg) / math.sqrt(_sm)).to(corpus.dev)
        _p2 = (T.randn(_sm, _sm, generator=_gg) / math.sqrt(_sm)).to(corpus.dev)
        with T.no_grad():
            corpus.pair_t1 = corpus.code.float() @ _p1
            corpus.pair_t2 = corpus.code.float() @ _p2
        corpus.pair_k = int(corpus.k)
        del _p1, _p2
        # 解析绑定的 lag 对（默认 "1,2" = 只绑 (t-1,t-2)，与历史 A1 一致）
        _pl = str(getattr(args, "pair_lags", "1,2")).split(";")
        _pairs = []
        for _t in _pl:
            _t = _t.strip()
            if not _t:
                continue
            _a, _b = [int(x) for x in _t.split(",")]
            _pairs.append((int(_a), int(_b)))
        if not _pairs:
            raise SystemExit("--pair-lags 为空")
        for (_a, _b) in _pairs:
            if max(_a, _b) >= int(args.win):
                raise SystemExit("--pair-lags 里的位置 %d,%d 超出 --win=%d" % (_a, _b, args.win))
        corpus.pair_pairs = _pairs
        _pair_extra = _sm * len(_pairs)
        log("二阶合取码：%d 对绑定 %s，z=T1[x_{t-i}]*T2[x_{t-j}] -> k-WTA(k=%d)，"
            "宽度 %d/对，零可学参数" % (len(_pairs), _pairs, corpus.pair_k, _sm))
    D_raw = (args.win * D_in if args.mem in ("window", "both") else 0) + \
            (args.n_res if res is not None else 0) + D_bank + D_hid + _pair_extra
    if args.pair_bind and int(D_raw) > int(_plan["D"]):
        # 容量规划用的 D 不含 pair 块 ⇒ y_max 被高估。按真实 D 等比收紧后再判一次。
        _ymax2 = int(_plan["y_max"] * float(_plan["D"]) / float(D_raw))
        if args.y > _ymax2:
            raise SystemExit("y=%d 超出加宽后 D=%d 的显存上限 y_max≈%d（pair 块让 D 从 %d 增到 %d）"
                             % (args.y, D_raw, _ymax2, _plan["D"], D_raw))
        log("容量复核（含 pair 块）：D=%d，y 上限≈%d，y=%d 通过" % (D_raw, _ymax2, args.y))
    if isinstance(hid, Cascade):
        if args.hid_mode == "only":
            D_raw = hid.H
        elif args.hid_mode == "replace":
            D_raw = hid.band0 + hid.H
    elif args.hid_mode == "only":
        # 读出拼所有层活动 → D = 层数 × H
        D_raw = args.hidden * max(1, int(args.hid_layers))
    lag_dim = int(args.win * D_in) if args.mem in ("window", "both") else 0
    # 隐藏块在读出输入里的起点 = 原始特征宽度（add 模式把它们拼在后面）
    if hid is not None and args.hid_mode == "add":
        hid_w_start = int(args.win * D_in + (args.n_res if res is not None else 0) + D_bank)
    else:
        hid_w_start = 0
    proj, D_feat = None, D_raw
    if getattr(args, "unit_rms", 0):
        # ① 约束：RMS 突触权重 = 1 ⇒ ||w||₂ = √D（单位约定，半径由输入维数决定，不是调出来的数）。
        #    于是 u = w·h ≤ √D·||h||，且 |w| 不随训练增长 ⇒ 从源头消灭"先炸再乘一个数变小"。
        args.budget = float(math.sqrt(D_feat))
        # ② 初始化：让**出生时**上下文驱动的 std = 先验 b 的 std（两者都是可测量量）。
        #    为什么：随机初始化若让驱动 std=√k（比先验跨度还大），出生时模型就比 unigram 差，
        #    局部规则得先花几千步把权重缩回来（实测 PPL 6517 > unigram）。这叫"出生即失配"。
        #    方差匹配后：出生时 score ≈ b（就是先验），学习只负责把有信息的方向长出来。
        _sb = float(corpus.logit_prior.std())
        _kact = max(1, int(getattr(corpus, "k", 0) or 51) * max(1, args.win))
        _sig = _sb / math.sqrt(_kact)
        args.w_init = float(_sig * math.sqrt(D_feat))
        log("单位球 + 方差匹配初始化：std(b)=%.3f，活跃输入≈%d → 每突触 σ=%.4f，w_init=%.1f，球半径=√D=%.1f" % (
            _sb, _kact, _sig, args.w_init, math.sqrt(D_feat)))
    if args.feat_dim and args.feat_dim < D_raw:
        g = T.Generator().manual_seed(20261105)
        nnz = 16
        P_ = T.zeros(D_raw, args.feat_dim)
        for j in range(args.feat_dim):
            ii = T.randperm(D_raw, generator=g)[:nnz]
            P_[ii, j] = T.randn(nnz, generator=g) / math.sqrt(nnz)
        proj, D_feat = P_.to(corpus.dev), args.feat_dim
        log("记忆特征 %d → 固定稀疏投射 → %d 维（不可学，属输入编码）" % (D_raw, D_feat))
    # 先读存档，按存档里的 cap / D 构脑。否则维度对不上 → 加载直接失败
    # （这是"训练完能当模型用"的前提，之前 --load 会因为 cap 不够而报错）。
    _ck = None
    if args.load:
        _ck = T.load(args.load, map_location=args.device, weights_only=False)
        _bs = _ck["brain"] if "brain" in _ck else _ck
        _cap = int(_bs["cap"])
        if int(_bs["D"]) != D_feat:
            log("注意：存档 D=%d ≠ 当前 D=%d，按存档来（上下文/尺度参数应以存档为准）"
                % (int(_bs["D"]), D_feat))
            D_feat = int(_bs["D"])
        _n0 = float((_ck.get("args") or {}).get("n0", args.n0))
    else:
        _cap = args.y + max(4096, args.y // 8)
        _n0 = float(args.n0)
    # ---- 输出矩阵 = 码表（约束A 由构造满足）----
    _out_code, _out_tb, _out_rep = None, None, 1.0
    corpus.logodds_out = int(args.logodds)
    if args.out_code:
        if args.hid_mode != "only" or hid is None:
            raise SystemExit("--out-code 需要 --hidden H --hid-mode only（否则特征里混着别的块，码表对不上）")
        if corpus.code is None:
            raise SystemExit("--out-code 需要 --in-code")
        _H = int(hid.H)
        _sm = int(corpus.SM)
        # 第 j 个隐藏细胞盯住码位 j % SM（与 HiddenLayer.tb 同一约定）
        _out_tb = (T.arange(_H, device=corpus.dev) % _sm)
        _out_rep = float(_H) / float(_sm)          # 每个码位有几个细胞投票
        _out_code = corpus.code
        args.freeze_readout = 1                     # 出口不学
        if not args.logodds:
            raise SystemExit("--out-code 需要 --logodds 1（伯努利似然输出，否则量级不对）")
        log("输出矩阵=码表：%d 个词 × %d 维（每码位 %g 个细胞投票），零可学参数、出口冻结"
            % (int(corpus.code.shape[0]), _H, _out_rep))
    brain = Brain(corpus, _cap, D_feat=D_feat, fr=args.fr, delay_steps=args.delay_steps,
                  eta_b=args.eta_b, w_init=args.w_init, budget=args.budget, rms_eps=args.rms_eps,
                  plast_tau=args.plast_tau, plast_norm=args.plast_norm, freeze_readout=args.freeze_readout, out_code=_out_code, out_tb=_out_tb, out_rep=_out_rep,
                  apical_mode=args.apical_mode, apical_norm=args.apical_norm,
                  coord_sample=args.coord_sample, coord_mode=args.coord_mode,
                  coord_groups=args.coord_groups, coord_swap=args.coord_swap,
                  chunk_grad=args.chunk_grad, rec_birth=args.rec_birth, tier=_tier,
                  half_feat=args.half_feat, b_noise=args.b_noise,
                  eta_cal=args.eta_cal, upd=args.upd, shy=args.shy,
                  topk=args.topk, topk_every=args.topk_every, rfan=args.rfan,
                  norm_score=args.norm_score, unit_l1=args.unit_l1, readout=args.readout,
                  dend=args.dend, dz_mode=args.dz_mode,
                  gate_lag=args.gate_lag, n_lag=args.n_lag, lag_w=args.lag_w,
                  gate_full=args.gate_full, gate_soft=args.gate_soft, mod=args.mod,
                  grp=args.grp, grp_pairs=args.grp_pairs,
                  grp_trace=args.grp_trace, trace_h=args.trace_h or args.win,
                  eta_temp=args.eta_temp, code_learn=args.code_learn, code_lr=args.code_lr,
                  seed=args.seed, device=args.device)
    brain.n0 = _n0
    if str(args.device).startswith("cuda"):
        T.cuda.empty_cache()
    log("特征维 D=%d；读出皮层 y 上限 %d（从 1 个细胞长起）" % (D_feat, args.y))
    if _ck is not None:
        brain.load_state(_ck["brain"] if "brain" in _ck else _ck)
        # 隐藏层状态也要恢复（θ / 稳态发放率 / 上一拍发放 / 递归边）。
        # 少了它，每次 --load 隐藏层都重新长，而读出层已适应旧的 → 实测 dev PPL 775→580→856→1610。
        if "gate" in _ck and getattr(brain, "gam", None) is not None:
            brain.gam = _ck["gate"]["gam"].to(brain.dev)
            brain.v_gam = _ck["gate"]["vgam"].to(brain.dev)
            log("已恢复门状态：每 lag 门权重的均值=%.4f" % float(brain.gam.mean()))
        if "mod" in _ck and brain.mod > 0:
            _mm = _ck["mod"]
            brain.P = _mm["P"].to(brain.dev)
            brain.al = _mm["al"].to(brain.dev)
            brain.be = _mm["be"].to(brain.dev)
            brain.v_P = _mm["vP"].to(brain.dev)
            brain.v_al = _mm["val"].to(brain.dev)
            brain.v_be = _mm["vbe"].to(brain.dev)
            log("已恢复交互模块状态：|α|均值=%.3f，|β|均值=%.3f" % (
                float(brain.al[:brain.n_slots].norm(dim=1).mean()),
                float(brain.be[:brain.n_slots].norm(dim=1).mean())))
        elif brain.mod > 0 and _ck is not None:
            log("警告：存档里没有交互模块状态，加载后交互项会被重置")
        if "hid" in _ck and hid is not None and not isinstance(hid, Cascade):
            for _k, _v in _ck["hid"].items():
                setattr(hid, _k, _v.to(hid.dev))
            log("已恢复隐藏层状态：|theta|=%.2f，稳态发放率=%.3f，边=%d" % (
                float(hid.theta.abs().mean()), float(hid.rate.mean()), hid.n_edges))
        elif hid is not None:
            log("警告：存档里没有隐藏层状态，续跑会让隐藏层重新长（与读出层失配）")
        log("已加载 %s（N=%d，n0=%.2f，gain=%.3f）" % (args.load, brain.N, brain.n0, brain.gain_t))

    # ---- 鲁棒性判据（文献20 约束B，A 级；本项目第一次做） ----
    # 随机杀掉一部分读出细胞，看 PPL 掉多少。生物网络应该掉得少；
    # 掉得多的意思是"每个词只有那一根柱子撑着"，功能没有冗余 —— 也就是结构没有承载计算。
    # 固定损伤种子：同一次损伤必须可复现（哲学15）。
    if args.kill_mode == "one":
        # 每个词只留一个细胞：按**槽位奇偶**交替保留，确定、可复现、与 freq 无关。
        with T.no_grad():
            _n = brain.n_slots
            _al = brain.alive[:_n]
            _tk = brain.tok[:_n]
            # 每个词的细胞按槽位排序，杀掉第 2,4,6... 个（即每对里后面的那个）
            _df = T.zeros(_n, dtype=T.long, device=brain.dev)
            _ord = T.argsort(_tk * _n + T.arange(_n, device=brain.dev))
            # 用一个简单办法：按词分组后的序号（向量化：同词位置累加）
            _sorted_tok = _tk[_ord]
            _same = T.zeros(_n, dtype=T.long, device=brain.dev)
            _same[1:] = (_sorted_tok[1:] == _sorted_tok[:-1]).long()
            # 前缀和得到组内序号
            _grp = T.zeros(_n, dtype=T.long, device=brain.dev)
            _run = T.zeros((), dtype=T.long, device=brain.dev)
            # 用 cumsum 的差分技巧避免 Python 循环
            _cs = T.cumsum(_same, 0)
            _start = T.zeros(_n, dtype=T.long, device=brain.dev)
            # 每个"组起点"的 cumsum 值
            _first = T.ones(_n, dtype=T.bool, device=brain.dev)
            _first[1:] = _same[1:] == 0
            _base = T.zeros(_n, dtype=T.long, device=brain.dev)
            _base[_first] = _cs[_first]
            _grp = _cs - _base
            _kill_sorted = (_grp % 2 == 1)
            _kill = _ord[_kill_sorted & _al[_ord]]
            brain.alive[_kill] = False
        log("互补/冗余判据：每个多细胞词杀掉一半（保留奇数号），共杀 %d 个，剩 %d 个活"
            % (int(_kill.numel()), int(brain.alive[: brain.n_slots].sum())))
    if args.kill_frac > 0:
        with T.no_grad():
            _idx = T.nonzero(brain.alive[: brain.n_slots]).squeeze(1)
            _tot = int(_idx.numel())
            _k = int(round(float(args.kill_frac) * _tot))
            _g = np.random.default_rng(12345)
            # 用固定**置换**取前缀，而不是 choice：这样不同 kill_frac 的损伤是**嵌套**的
            # （10% ⊂ 20% ⊂ 30%），才能画出一条真正的损伤曲线；choice 做不到嵌套。
            _perm = _g.permutation(_idx.cpu().numpy())
            _pick = _perm[:_k]
            brain.alive[T.from_numpy(np.ascontiguousarray(_pick)).to(brain.dev)] = False
        log("鲁棒性损伤：随机杀掉 %d/%d 个细胞（%.0f%%，种子固定=12345）"
            % (_k, _tot, 100 * float(args.kill_frac)))

    src = corpus.tr
    if not args.load:
        if args.init_n > 0:
            brain._birth(max(1, args.init_n), corpus)
            log("对照组 A：t=0 直接给 %d 个细胞（不靠生长）；槽位上限 %d" % (brain.N, brain.cap))
        else:
            brain._birth(1, corpus)
            log("从 1 个细胞长起（词 %d）；槽位上限 %d（=2y+1024，只追加）" % (
                int(brain.tok[0]), brain.cap))
    B = args.batch
    if res is not None:
        res.reset(B)
    brain.reset_state(B)
    rng = np.random.default_rng(args.seed)
    lo = max(args.win, 2) + warm_default()
    pos = rng.integers(lo, src.size - 4, size=B).astype(np.int64)
    steps = 0 if args.eval_only else max(1, args.tokens // B)
    _mem("after-init")
    if T.cuda.is_available():
        T.cuda.reset_peak_memory_stats()

    err_ref = max(1e-6, corpus.unigram_ppl() * 0 + 7.7)      # 一元熵的量级（生长参照）
    curve, t0 = [], time.time()
    acc_nll, acc_n = 0.0, 0

    for t in range(1, steps + 1):
        hidden_learn = True
        _need_cn = (hid is not None) or (args.apical_mode == "bit")
        _cn = corpus.code[T.from_numpy(np.ascontiguousarray(
            np.asarray(src[pos + 1], dtype=np.int64))).to(corpus.dev)] if _need_cn else None
        _tk_cur = _win_toks(src, pos, args.win)
        globals()["cur_toks"] = _tk_cur
        h = _feat(corpus, res, args.win, _tk_cur, proj, bank, hid, _cn)
        tgt = T.from_numpy(np.asarray(src[pos + 1], dtype=np.int64)).to(corpus.dev)
        pred = brain.forward(h)
        brain.logits = pred
        brain.h_feat = h
        if t <= 2:
            _mem("t%d-forward" % t)
        with T.no_grad():
            pv = T.softmax(pred, dim=1).clamp(min=1e-9)
            ll = -T.log(pv.gather(1, tgt.unsqueeze(1)).squeeze(1))
            # ---- 稳定守卫 ①：损失必须有限。硬约束是"内部有一处 NaN 就不可用"，
            # 所以这里**直接停**，不许带着 NaN 继续算出一堆能被我当结论记录的数字。
            _llf = float(ll.sum())
            if not math.isfinite(_llf):
                raise SystemExit("[NaN/Inf] t=%d 损失非有限（%.6g）→ 立即停止，本跑不可用" % (t, _llf))
            acc_nll += _llf
            acc_n += B
            ssum = T.zeros(corpus.V, device=brain.dev).index_add_(0, tgt, ll)
            scnt = T.zeros(corpus.V, device=brain.dev).index_add_(0, tgt, T.ones_like(ll))
            seen = scnt > 0
            brain.tok_err[seen] = 0.99 * brain.tok_err[seen] + 0.01 * (ssum / scnt.clamp(min=1.0))[seen]
        if not (args.freeze_at and t > args.freeze_at):
            _tt = t + args.lr_offset
        brain.learn(tgt, lr=args.lr / math.sqrt(1.0 + _tt / (0.5 * (steps + args.lr_offset))),
                    code_next=_cn)
        if t <= 2:
            _mem("t%d-learn" % t)
        if t == 1 and str(args.device).startswith("cuda"):
            # 第一次 learn 的 gW 是最大临时块；训练稳定后归还缓存，避免 reserved 长期占满。
            T.cuda.empty_cache()
        brain.age[:brain.n_slots] += 1.0
        # ---- 稳定守卫 ②：权重/打分每 250 步体检一次（比每步查便宜，又不至于漏掉）。
        # 只查"是否有限"，不夹取、不缩放 —— 出问题就停，不做任何掩盖。
        if t % 250 == 0:
            with T.no_grad():
                if not bool(T.isfinite(brain.w[:brain.n_slots]).all()) or \
                   not bool(T.isfinite(pred).all()):
                    raise SystemExit("[NaN/Inf] t=%d 权重或打分出现非有限值 → 立即停止" % t)
        pos += 1
        pos[pos >= src.size - 4] = lo

        if t % args.grow_every == 0 and brain.events < args.max_events:
            brain.events += 1
            err_norm = min(1.0, max(0.0, float(brain.tok_err.mean()) / err_ref))
            room = max(0.0, 1.0 - brain.N / args.y)
            _p2base = args.y if args.phase2_on_y else brain.cap
            phase2 = brain.N >= args.grow2_frac * _p2base
            n_birth = int(math.ceil(args.grow_rate * max(1, brain.N) * room * err_norm))
            if phase2:
                n_birth = int(math.ceil(n_birth * 0.05))
            nb = brain._birth(min(n_birth, args.y), corpus, args.grow_target) if n_birth > 0 else 0
            if nb and str(args.device).startswith("cuda"):
                # 出生会分配 (n_new,D) 大块；事件后归还，压住 allocator 碎片
                # （实测长跑在 t≈62500 因 reserved 冲到 17GB 卡死）
                T.cuda.empty_cache()
            nsyn = brain.synaptogenesis(int(args.syn_rate * brain.N)) if (phase2 and args.syn_rate) else 0
            ncut = brain.prune_edges() if (phase2 and args.syn_rate) else 0
            nd = brain._death(int(round(args.death_rate * brain.N * (brain.N / args.y))), args.grace)
            nt = brain.turnover(int(round(args.turnover_rate * brain.N)), corpus) if args.turnover_rate else 0
            if brain.h is not None and brain.h.shape[1] != brain.n_slots:
                brain.reset_state(B)
            to = brain.topology()
            curve.append({"step": t, "N": brain.N, "born": nb, "died": nd, "turnover": nt,
                          "phase": 2 if phase2 else 1, "edges": to["edges"],
                          "err": float(brain.tok_err.mean()),
                          "train_ppl": math.exp(acc_nll / max(1, acc_n))})
            acc_nll, acc_n = 0.0, 0
            if brain.events <= 12 or brain.events % 25 == 0:
                _ma = T.cuda.memory_allocated() / 1e9
                _mr = T.cuda.memory_reserved() / 1e9
                log("t=%6d 阶段%d N=%6d (+%d/-%d) 每词%.1f 误差=%.2f 训练PPL=%.0f 用时%.0fs 显存%.1f/%.1fGB" % (
                    t, 2 if phase2 else 1, brain.N, nb, nd, to["per_word"],
                    float(brain.tok_err.mean()), curve[-1]["train_ppl"], time.time() - t0, _ma, _mr))
        if args.save_every and args.out and t % args.save_every == 0:
            _tmp = args.out + ".part"
            _pp = {"version": "grow-4.1.0", "args": vars(args), "brain": brain.snapshot()}
            if brain.gate_lag and brain.lag_w > 0:
                # 门的状态必须进档：少了它，加载后所有门回到 0（sigmoid ⇒ 均匀 0.5），
                # 与训练好的模型不一致 —— 会让"训练完当模型用"和续跑都失真。
                _pp["gate"] = {"gam": brain.gam.detach().cpu(), "vgam": brain.v_gam.detach().cpu()}
            if brain.mod > 0:
                # 交互模块状态必须进档（否则加载后交互项被静默重置为初始值）
                _pp["mod"] = {"P": brain.P.detach().cpu(), "al": brain.al.detach().cpu(),
                              "be": brain.be.detach().cpu(),
                              "vP": brain.v_P.detach().cpu(), "val": brain.v_al.detach().cpu(),
                              "vbe": brain.v_be.detach().cpu()}
            if hid is not None and not isinstance(hid, (Cascade, list, tuple)):
                _pp["hid"] = {k: getattr(hid, k).detach().cpu() for k in
                              ("w", "v_w", "theta", "rate", "prev", "rw", "rmask", "v_rec")}
            T.save(_pp, _tmp)
            os.replace(_tmp, args.out)
        if args.log_every and t % args.log_every == 0:
            # 周期评测的样本量必须够：PPL 的标准误 ≈ 4.5/sqrt(N) nats，256 样本时 ~30% 不可用
            _src = corpus.dev_tok if (args.eval_src == "dev" and corpus.dev_tok is not None) else None
            if bank is not None:
                m = evaluate_long(brain, corpus, res, bank, args.win,
                                  n_eval=max(512, args.eval_batches * 16), batch=32, seed=1,
                                  proj=proj, hid=hid, src=_src)
            else:
                m = evaluate(brain, corpus, res, args.win, n_batches=max(20, args.eval_batches // 3),
                             batch=64, seed=1, proj=proj, hid=hid, src=_src)
            m["_src"] = "dev" if _src is not None else "heldout"
            log("t=%6d N=%6d 留出PPL=%.1f top1=%.3f 发放=%.3f" % (
                t, brain.N, m["ppl"], m["top1"], m["diag"]["fire"]))

    if str(args.device).startswith("cuda"):
        T.cuda.empty_cache()
    if args.eval_temp and args.eval_temp > 0:
        brain.gain_t = 1.0 / float(args.eval_temp)
        log("评测温度 T=%.3f（gain_t=%.4f）" % (args.eval_temp, brain.gain_t))
    if bank is not None:
        m = evaluate_long(brain, corpus, res, bank, args.win,
                          n_eval=(args.eval_n or max(1024, args.eval_batches * 32)),
                          batch=64, seed=0, proj=proj, hid=hid)
    else:
        m = evaluate(brain, corpus, res, args.win, n_batches=args.eval_batches, batch=64, proj=proj, hid=hid)
    uni = corpus.unigram_ppl()
    # 隐藏层可能是多层的 list → 先汇总成标量，供结果字典使用
    if isinstance(hid, Cascade):
        _hl = list(hid.layers)
    elif isinstance(hid, (list, tuple)):
        _hl = list(hid)
    else:
        _hl = [hid] if hid is not None else []
    _h_edges = sum(h.n_edges for h in _hl)
    _h_born = sum(h.edge_born for h in _hl)
    _h_cut = sum(h.edge_cut for h in _hl)
    _h_rate = sum(float(h.rate.mean()) for h in _hl) / max(1, len(_hl))
    _h_theta = sum(float(h.theta.abs().mean()) for h in _hl) / max(1, len(_hl))
    _hg = [h for h in _hl if getattr(h, "geo", 0)]
    _h_mods = sum(h._geo_M for h in _hg)
    _h_modin = (sum(float((h.mid[h.rsrc[:, 0]] == h.mid).float().mean()) for h in _hg)
                / max(1, len(_hg)))
    _h_dist = (sum(float(T.cdist(h.pos, h.pos)[T.arange(h.H, device=h.dev), h.rsrc[:, 0]].mean()
                        / h._geo_L) for h in _hg) / max(1, len(_hg)))
    _wv = brain.w[:brain.n_slots][brain.alive[:brain.n_slots]]
    w_abs = _wv.abs() if _wv.numel() else T.zeros(1, device=args.device)
    if args.nll_out and "_nll" in m:
        np.save(args.nll_out, m.pop("_nll"))
    elif "_nll" in m:
        m.pop("_nll")
    out = {"version": "grow-4.1.0", "mem": args.mem, "ctx_tau": args.ctx_tau, "win": args.win,
           "n_res": args.n_res, "ctx_len": args.ctx_len, "D_bank": D_bank, "hidden": args.hidden,
           "hidden_edges": _h_edges,
           "w_lognorm_sigma": float(T.log10(w_abs.clamp(min=1e-9)).std()) if brain.N else 0.0,
           "w_nearzero_frac": float((w_abs < 0.01 * float(w_abs.mean())).float().mean()) if brain.N else 0.0,
           "w_pos_frac": float((brain.w[:brain.n_slots][brain.alive[:brain.n_slots]] > 0).float().mean()) if brain.N else 0.0,
           "upd": args.upd,
           "hidden_rate_mean": _h_rate,
           "hidden_rate_std": 0.0,
           "hidden_theta_abs": _h_theta,
           "hidden_geo": args.hid_geo,
           "hidden_mods": _h_mods,
           "hid_casc": (len(hid.layers) if isinstance(hid, Cascade) else 0),
           "hid_w_frac": round(float(brain.hid_w_frac), 4),
           "w_frac_lag": round(float(brain.w_frac_lag), 4),
           "mod_M": int(getattr(brain, "mod", 0)),
           "mod_births": int(getattr(brain, "_mod_births", 0)),
           "grp_theta_norm": (round(float(brain.theta[:, :brain.n_slots].norm(dim=(0, 2)).mean()), 4)
                            if getattr(brain, "grp", 0) else 0.0),
           "mod_alpha_norm": (round(float(brain.al[:brain.n_slots].norm(dim=1).mean()), 4) if brain.mod else 0.0),
           "mod_beta_norm": (round(float(brain.be[:brain.n_slots].norm(dim=1).mean()), 4) if brain.mod else 0.0),
           "dend2_norm": (round(float(brain.wd2[:brain.n_slots].norm(dim=1).mean()), 4)
                          if brain.dend else 0.0),
           "dend1_norm": (round(float(brain.wd1[:brain.n_slots].norm(dim=1).mean()), 4)
                          if brain.dend else 0.0),
           "casc_bands": ([list(b) for b in hid.bounds] if isinstance(hid, Cascade) else []),
           "casc_mixed": ([round(hid.mixed_frac(i), 3) for i in range(len(hid.layers))]
                          if isinstance(hid, Cascade) else []),
           "resid_var_per_layer": (getattr(hid, "resid_chain", None) if not isinstance(hid, list) else getattr(hid[-1], "resid_chain", None)),
           "topo_mod_in_frac": _h_modin,
           "fire_sum": (float(brain.sigma[:, brain.alive[:brain.n_slots]].sum(1).mean())
                        if brain.N else 0.0),
           "fire_max": (float(brain.sigma[:, brain.alive[:brain.n_slots]].max(1).values.mean())
                        if brain.N else 0.0),
           "fire_med": (float(brain.sigma[:, brain.alive[:brain.n_slots]].median())
                        if brain.N else 0.0),
           "topo_dist_mean": _h_dist,
           "hidden_edge_born": _h_born,
           "hidden_edge_cut": _h_cut,
           "D_feat": D_feat, "D_raw": D_raw, "y": args.y,
           "N_readout": brain.N, "N_memory": args.n_res if res is not None else 0,
           "y0_over_y": brain.N / args.y, "born_total": brain.born_total,
           "dead_total": brain.dead_total, "tokens": steps * B,
           "ppl": m["ppl"], "top1": m["top1"], "top5": m["top5"], "diag": m["diag"],
           "unigram_ppl": uni, "vs_unigram": m["ppl"] / uni,
           "topology": brain.topology(), "backprop": False, "shared_linear_readout": False,
           "grad_enabled": bool(T.is_grad_enabled()), "seconds": round(time.time() - t0, 1),
           "gpu_peak_GB": round(T.cuda.max_memory_allocated() / 1e9, 2) if T.cuda.is_available() else 0}
    log(json.dumps(out, ensure_ascii=False))
    if args.chat:
        chat_loop(brain, corpus, res, bank, args, proj)
        return
    if args.sample:
        ids = [int(x) for x in args.sample.split(",") if x.strip()]
        gen = generate(brain, corpus, res, args.win, ids, n_new=args.sample_n, temp=0.9, topk=0,
                       proj=proj, bank=bank)
        log("续写 token：%s" % gen)
        out["sample_prompt"], out["sample_gen"] = ids, gen
    if args.out:
        _pp = {"brain": brain.snapshot(),
               "reservoir": {"n": args.n_res, "tau": args.ctx_tau} if res is not None else None,
               "result": out, "args": vars(args)}
        if brain.gate_lag and brain.lag_w > 0:
            _pp["gate"] = {"gam": brain.gam.detach().cpu(), "vgam": brain.v_gam.detach().cpu()}
        if brain.mod > 0:
            _pp["mod"] = {"P": brain.P.detach().cpu(), "al": brain.al.detach().cpu(),
                          "be": brain.be.detach().cpu(),
                          "vP": brain.v_P.detach().cpu(), "val": brain.v_al.detach().cpu(),
                          "vbe": brain.v_be.detach().cpu()}
        if hid is not None and not isinstance(hid, (Cascade, list, tuple)):
            _pp["hid"] = {k: getattr(hid, k).detach().cpu() for k in
                          ("w", "v_w", "theta", "rate", "prev", "rw", "rmask", "v_rec")}
        T.save(_pp, args.out)
        log("大脑已保存 -> %s" % args.out)
    if args.curve_out:
        json.dump({"curve": curve, "result": out}, open(args.curve_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


def warm_default():
    return 2


if __name__ == "__main__":
    main()

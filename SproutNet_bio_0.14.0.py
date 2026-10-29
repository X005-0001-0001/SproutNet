# -*- coding: utf-8 -*-
"""
SproutNet_bio_0.1.0 —— 整链路仿生原型（CPU 快速版）

要验证的假设（用户提出的"整条信息流动链路都要改"）：

  现在 cortex 的链路是
      BPE 词号 → one-hot 注入 → LIF 隐藏层 → softmax 分类器(V 类, teacher=下一个词)
  这条链路有三处不是仿生的：
      ① 编码：感受器输入的是"符号身份"，不是"预测误差"。生物感受器做感觉适应，
         常见刺激几乎不引起响应 —— one-hot 让常见词与罕见词驱动强度相同，信息量被抹平。
      ② 解码：皮层没有分类器。它预测的是"下一时刻的感觉输入"。
      ③ 没有闭环：预测值从未回到输入端去扣减。

  这一版把链路改成
      e_t   = r_t − r̂_t                     感受器层只把"意外"送出去（感觉适应）
      h_t   = spike(W_in·e_t + U·h_{t-1})   阈值非线性 + 群体稀疏
      r̂_{t+1} = normalize(relu(W_out·h_t)) 预测下一时刻的感受器模式（非负、和为 1 → 本身就是分布）
      W_out += η·(r_{t+1} − r̂_{t+1}) ⊗ h_t   每条输出突触只用自己那一路的误差（局部）

  PPL 直接用 −log r̂[target]，不需要额外分类器，也没有 teacher（"下一个词"就是自然到来的输入）。

对照开关 B2_ENC=onehot|pred：只改"输入是否扣掉预测"，其余完全相同。
"""
import math
import os
import time

import numpy as np
import torch

DEV = torch.device(os.environ.get('B2_DEV', 'cpu'))
torch.set_num_threads(int(os.environ.get('B2_THREADS', '16')))
if DEV.type == 'cuda':
    torch.cuda.init()
torch.manual_seed(int(os.environ.get('B2_SEED', '0')))
np.random.seed(int(os.environ.get('B2_SEED', '0')))

ROOT = os.path.dirname(os.path.abspath(__file__))
V = 10000
N = int(os.environ.get('B2_N', '0'))   # 0 = 由 G7 规则（数据量/SPP/V）决定，不手填
HIST = int(os.environ.get('B2_HIST', '0'))   # 1 = 读出看 h_t 与 h_{t-1} 拼接
DEND = int(os.environ.get('B2_DEND', '0'))   # 1 = 树突分段非线性（段内饱和，段间求和）
SEG = int(os.environ.get('B2_SEG', '4'))     # 每个神经元的树突段数（解剖值 ~4）
B = int(os.environ.get('B2_B', '32'))
P0 = 0.1                          # 实测皮层连接概率
TARGET_RATE = float(os.environ.get('B2_RATE', '0.05'))   # 群体发放率（实测皮层 5%，这里可调以测选择性）
TOKENS = int(os.environ.get('B2_TOKENS', '200000'))
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '3'))
EPOCHS = int(os.environ.get('B2_EPOCHS', '1'))   # 多遍扫同一数据（在线单 pass 对 10000 类远不够）
ENC = os.environ.get('B2_ENC', 'pred')          # pred | onehot
LR_SCALE = float(os.environ.get('B2_LR', '1.0'))
MAX_RSS_GB = float(os.environ.get('B2_MAX_RSS_GB', '8.0'))
NOFEAT = os.environ.get('B2_NOFEAT', '0') == '1'   # 自检：冻结特征，只学 bias
NOBIAS = os.environ.get('B2_NOBIAS', '0') == '1'   # 诊断：冻结偏置，只用特征
# NLMS：按当前输入能量归一化步长。恒定步长的 LMS 只保证不发散、不保证收敛（实测 trainPPL 在 711↔4159 振荡），
# NLMS 的更新是非扩张的（η̄ ≤ 1 时），这是无超参的稳定化方案。
NLMS = os.environ.get('B2_NLMS', '0') == '1'
# 动量：平滑高曲率方向上的振荡（文献标准 β=0.9，不是针对本任务调的）
MOM = float(os.environ.get('B2_MOM', '0.0'))
# C 组重测（MAP.md 第 5 节队列 1）：局部可塑规则此前都在 η_b/θ bug 下测过，需在干净基座重测
RULE = os.environ.get('B2_RULE', 'none')   # none | oja | wta | stdp | three
RULE_ETA = float(os.environ.get('B2_RULE_ETA', '0.01'))
# 接线拓扑：random = 均匀随机（旧做法）；edr = 指数距离规则
#   实测：皮层连接概率 p(d)=c·exp(−λd)，λ≈0.3/mm，平均间距 26.6mm → λ·d̄ ≈ 8
#   这里"距离"用表征空间的化学亲和（bit 的位置 vs 神经元出生的位置）
WIRE_TOPO = os.environ.get('B2_TOPO', 'random')
# 生长（哲学 2/8/11）：神经元不是一次填满，而是从 N0 起按「误差停滞」分批出生。
GROW = os.environ.get('B2_GROW', '0') == '1'
N0 = int(os.environ.get('B2_N0', '64'))
STEP = int(os.environ.get('B2_STEP', '64'))
# G7：容量不能靠扫描 val PPL 选（哲学22）。用「每个参数至少 SPP 个样本」这条统计原则反推上限：
#   N_cap = min(TOKENS / (SPP * V), 8*N0 上限保护)
# 这是规则，不是调参：数据量决定容量。
SPP = float(os.environ.get('B2_SPP', '10'))
# N_cap 只由「数据量 / (SPP×V)」决定；N=0 表示"容量完全交给规则"，N>0 表示显式上限
N_CAP = int(max(4, int(TOKENS / (SPP * V)) if TOKENS > 0 else 4))
if N <= 0:
    N = N_CAP
GROW_WIN = int(os.environ.get('B2_GWIN', '2000'))
# 时间深度（哲学12）：一个 token 让网络迭代 T 步再读出。
# 深度来自"处理时间"而不是手填层数（哲学11）。皮层一次前馈扫描 ~50-100ms，跨多个突触站。
TSTEPS = int(os.environ.get('B2_TSTEPS', '1'))
# 慢状态（NMDA 型持续活动）：递归连接读到的是慢突触变量 s，而不是瞬时发放 h。
# τ 由 TSTEPS 给出（s 在 T 步内积分完），不额外手填时间常数。
SLOW = os.environ.get('B2_SLOW', '0') == '1'
# B7：读出稀疏化（每个词只读 k 个神经元）。目的不是"更仿生"，而是把参数量降下来，
# 提高「每个权重分到的样本数」——G3 指出这是当前唯一没被数据打死的方向。
SPOUT = int(os.environ.get('B2_SPOUT', '0'))   # 0 = 稠密读出；>0 = 每词接 SPOUT 个神经元
# A4：高维稀疏 SDR 输入编码。one-hot 之间没有相似度；皮层感觉编码是"相似输入→重叠模式"。
# 做法：把 64 维共现 embedding 投影到 M 位，每个 token 取 top-k 位（保相似度 + 稀疏）。
SDR = int(os.environ.get('B2_SDR', '0'))
SM = int(os.environ.get('B2_SM', '1024'))          # SDR 位数
SDR_K = float(os.environ.get('B2_SDR_K', '0.05'))  # 每个 token 激活比例

_psutil = None
try:
    import psutil as _psutil
except ImportError:
    pass


# ---------- 数据 ----------
def load_ids():
    import glob
    # B2_CACHE 可选语料：默认 1712 万（_pretrain_300k）；big=5025 万（_pretrain_1m，dataset_out/seg_0001 前 100 万行）
    want = os.environ.get('B2_CACHE', '')
    if want:
        cand = [os.path.join(ROOT, '_sprout_build', want)]
    else:
        cand = sorted(glob.glob(os.path.join(ROOT, '_sprout_build', 'tokcache_325b036129_10000.npz')))
        if not cand:
            cand = sorted(glob.glob(os.path.join(ROOT, '_sprout_build', 'tokcache_*10000.npz')))
    ids = np.load(cand[0])['ids']
    print('[data] %s  %d tokens' % (os.path.basename(cand[0]), len(ids)), flush=True)
    n_val = max(20000, len(ids) // 50)
    return ids[:-n_val].astype(np.int64), ids[-n_val:].astype(np.int64)


# ---------- 模型 ----------
class Bio:
    def __init__(self):
        g = torch.Generator().manual_seed(int(os.environ.get('B2_SEED', '0')) * 10000 + 7)
        # 稀疏随机投射（固定掩码，不学习）：生物学上这些是发育期长好的、之后基本稳定的连接
        if DEND:
            # 每个树突段独立一套输入：列号 = n*SEG + k，k 是段号
            self.WCOL = N * SEG
            self.W_in = torch.randn(V, self.WCOL, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * V)))
            self.W_in *= (torch.rand(V, self.WCOL, generator=g) < P0).float()
        else:
            self.WCOL = N
            if SDR:
                # A4：高维稀疏 SDR 输入。one-hot 之间无相似度；这里把 64 维共现 embedding
                # 投影到 SM 位后取 top-k，得到「相似 token → 重叠模式」的稀疏编码。
                _emb = torch.from_numpy(np.load(os.path.join(
                    ROOT, '_sprout_build', 'token_clusters.npz'))['emb']).float()
                _emb = _emb / _emb.norm(dim=1, keepdim=True).clamp(min=1e-9)
                _R = torch.randn(_emb.shape[1], SM, generator=g)
                _P = _emb @ _R
                _k = max(1, int(SDR_K * SM))
                self.code = torch.zeros(V, SM)
                self.code.scatter_(1, _P.topk(_k, dim=1).indices, 1.0)
                self.MIN = SM
                self.W_in = torch.randn(SM, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * SM)))
                self.W_in *= (torch.rand(SM, N, generator=g) < P0).float()
                print('[sdr] M=%d k=%d active=%.1f' % (SM, _k, float(self.code.sum(1).mean())), flush=True)
            else:
                self.MIN = V
                self.W_in = torch.randn(V, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * V)))
                self.W_in *= (torch.rand(V, N, generator=g) < P0).float()
        if WIRE_TOPO == 'edr' and SDR:
            # 每个 bit 的位置 = 激活它的那些 token 的 embedding 均值（数据决定，不手填）
            _cnt = self.code.sum(dim=0, keepdim=True).t().clamp(min=1.0)      # (M,1)
            _P = (self.code.t() @ _emb) / _cnt
            _P = _P / _P.norm(dim=1, keepdim=True).clamp(min=1e-9)
            # 每个神经元"出生"在某个 token 的表征位置上（化学亲和标记）
            _g = torch.Generator().manual_seed(int(os.environ.get('B2_SEED', '0')) * 10000 + 53)
            _mark = _emb[torch.randint(0, V, (N,), generator=_g)]
            _mark = _mark / _mark.norm(dim=1, keepdim=True).clamp(min=1e-9)
            _d = 1.0 - _P @ _mark.t()                                          # (M,N) 余弦距离
            _d = _d / _d.mean().clamp(min=1e-9)                                # 归一化到均值 1
            _lam = float(os.environ.get('B2_LAM', '8.0'))                       # λ·d̄（实测 0.3×26.6≈8）
            _logits = -_lam * _d
            _k = max(1, int(P0 * SM))
            mask = torch.zeros(SM, N)
            for _j in range(N):
                sel = torch.multinomial(torch.softmax(_logits[:, _j], 0), _k, replacement=False)
                mask[sel, _j] = 1.0
            self.W_in = self.W_in * mask
            print('[topo] EDR 接线：λ·d̄=%.1f  fan_in=%d  平均连接数=%.0f' % (
                _lam, _k, float(mask.sum())), flush=True)
        # 固定稀疏掩码：可塑更新只能在既有连接上进行，否则"长出来的接线"会被抹掉变成全连接（违反哲学11）
        self.mask_in = (self.W_in != 0).float()
        # C5 迹式 STDP 需要逐突触的 pre/post 迹
        self.tr_pre = torch.zeros_like(self.W_in)
        self.tr_post = torch.zeros_like(self.W_in)
        self.U = torch.randn(N, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * N)))
        self.U *= (torch.rand(N, N, generator=g) < P0).float()
        # 输出：预测下一时刻感受器模式。初始极小 → r̂ 近似均匀
        self.NF = N * (2 if HIST else 1)      # 读出特征维（HIST=1 时拼接 h_{t-1}）
        self.h_prev = torch.zeros(B, N)
        self.s_slow = torch.zeros(B, N)
        self.W_out = torch.zeros(self.NF, V)
        # 输出偏置：没有它模型连「词频」这个最基本的基线都表达不出来
        #（cortex 一直有 self.b，这个原型漏了 → PPL 卡在接近均匀）
        self.b = torch.zeros(V)
        if SPOUT > 0:
            g2 = torch.Generator().manual_seed(int(os.environ.get('B2_SEED', '0')) * 10000 + 91)
            # 固定随机稀疏读出掩码（接线由发育决定、之后基本不变）
            self.mask_out = torch.zeros(self.NF, V)
            for v in range(V):
                idx = torch.randperm(self.NF, generator=g2)[:SPOUT]
                self.mask_out[idx, v] = 1.0
            self.eta = LR_SCALE / (SPOUT * TARGET_RATE)   # 稀疏读出的 fan_in 是 SPOUT
        else:
            self.mask_out = None
        # 显式 N > 0 时优先用显式值；只有 N=0（由规则决定）时才用 N_CAP
        _cap = N_CAP if int(os.environ.get('B2_N', '0')) <= 0 else N
        self.n_cap = max(4, min(_cap, N))
        self.n_alive = min(N0, self.n_cap) if GROW else self.n_cap
        if not GROW:
            self.n_cap_used = self.n_alive
        self.alive = torch.zeros(N, device=DEV)
        self.alive[:self.n_alive] = 1.0
        self.theta = torch.full((N,), 0.01)
        self.h = torch.zeros(B, N)
        self.r_hat = torch.zeros(B, V)          # 上一步对"当前时刻"的预测
        self.clock = 0
        # LMS 步长：稳定条件 η ≤ 1/(fan_in · E[h²]) = 1/(N · TARGET_RATE)
        self.eta = LR_SCALE / (self.NF * TARGET_RATE)
        self.dw_stat = 0.0
        self.avg_on = os.environ.get('B2_AVG', '1') == '1'
        self.avg_W = torch.zeros_like(self.W_out)
        self.avg_b = torch.zeros_like(self.b)
        self.avg_n = 0
        self.vel_W = torch.zeros_like(self.W_out) if MOM > 0 else None
        # 统一搬到 DEV（GPU 可用时用它，否则 CPU）
        for _k, _v in list(self.__dict__.items()):
            if torch.is_tensor(_v):
                setattr(self, _k, _v.to(DEV))

    def step(self, x, learn=True):
        """x: (B,) 当前时刻的 token"""
        ar = torch.arange(B, device=DEV)
        # 感受器层：r 是 one-hot。误差 = r − r̂（只在 x 处有 +1−r̂，其余 −r̂）
        if SDR:
            e = self.code[x]                                 # (B, SM) 稀疏 SDR
        elif ENC == 'pred':
            e = -self.r_hat.clone()
            e[ar, x] += 1.0
        else:
            e = torch.zeros(B, V, device=DEV)
            e[ar, x] = 1.0
        self.e_last = e
        # 隐藏层：输入误差 + 内部递归
        if DEND:
            # 段内线性求和 → 段内饱和非线性 → 胞体求和（无额外手填权重，段间等权）
            ds = e @ self.W_in                                    # (B, N*SEG)
            ds = ds.view(B, N, SEG)
            sq = ds * ds
            seg_act = sq / (1.0 + sq)                             # NMDA 型：超线性→饱和
            d = seg_act.sum(dim=2) / math.sqrt(SEG) + self.h @ self.U
        else:
            d = e @ self.W_in + self.h @ self.U
        if SLOW:
            # s_{t+1} = (1-α)s_t + α h_t，α = 1/TSTEPS；递归驱动改用 s
            _alpha = 1.0 / max(1, TSTEPS)
            for _ in range(max(1, TSTEPS) - 1):
                self.s_slow = (1 - _alpha) * self.s_slow + _alpha * self.h
                if learn:
                    _al = self.n_alive
                    self.theta[:_al] = torch.quantile(d.detach()[:, :_al], 1.0 - TARGET_RATE,
                                                      dim=0).clamp(min=1e-6)
                _h = (d > self.theta.unsqueeze(0)).float() * self.alive.unsqueeze(0)
                d = e @ self.W_in + self.s_slow @ self.U
        if TSTEPS > 1 and not SLOW:
            # 输入驱动算一次，之后让递归动力学自己走 T-1 步（信号在网络上多传几站）
            _drive_in = e @ self.W_in
            for _ in range(TSTEPS - 1):
                if learn:
                    _al = self.n_alive
                    self.theta[:_al] = torch.quantile(d.detach()[:, :_al], 1.0 - TARGET_RATE,
                                                      dim=0).clamp(min=1e-6)
                _h = (d > self.theta.unsqueeze(0)).float() * self.alive.unsqueeze(0)
                d = _drive_in + _h @ self.U
                if DEND:
                    break
        if learn:
            # 分位数阈值：锁定群体发放率 = TARGET_RATE（无魔数，直接是实测值）
            # 只在**已出生的**神经元上求分位（生长时未出生列不参与）
            _al = self.n_alive
            self.theta[:_al] = torch.quantile(d.detach()[:, :_al], 1.0 - TARGET_RATE,
                                              dim=0).clamp(min=1e-6)
        # 未出生的神经元必须恒为 0（否则生长等于空操作，实测会与固定 N 完全相同）
        h = (d > self.theta.unsqueeze(0)).float() * self.alive.unsqueeze(0)
        if RULE == 'wta' and learn:
            _k = max(1, int(TARGET_RATE * self.n_alive_dummy())) if False else max(1, int(TARGET_RATE * N))
            _tv, _ti = torch.topk(h, _k, dim=1)
            h = torch.zeros_like(h).scatter_(1, _ti, _tv)
        self.h_prev = self.h
        self.h = h
        # 输出：预测下一时刻的感受器模式（非负、归一化 → 本身就是分布）
        # 预测的感受器模式 → 归一化成分布。
        # 注意：relu+归一化 在这个参数化下不稳定（实测 trainPPL 1643→109294 发散），
        # 因为 (target−r̂) 不再是 loss 对 logits 的梯度。这里改用 softmax，
        # 它只是「把预测的模式变成一个分布」的数值约定，误差来源仍是
        # 「预测下一时刻的感受器模式」，不是 V 类分类的先验。
        hf = torch.cat([h, self.h_prev], dim=1) if HIST else h
        logits = hf @ self.W_out + self.b
        r_hat_next = torch.softmax(logits, dim=1)
        self.hoff = hf
        return r_hat_next

    def learn_out(self, r_hat_next, y):
        """局部误差学习：每条输出突触只用「目标那一路的误差」和「自己的 pre」。"""
        ar = torch.arange(B, device=DEV)
        target = torch.zeros(B, V, device=DEV)
        target[ar, y] = 1.0
        err = target - r_hat_next
        if RULE == 'three':
            # 相位性调质：惊讶度(NLL) 减去慢平均；只在显著偏离时才非零
            _nll = -torch.log(r_hat_next[torch.arange(B, device=DEV), y].clamp(min=1e-12)).mean()
            if not hasattr(self, 'nll_slow'):
                self.nll_slow = _nll.detach()
            _dev = _nll - self.nll_slow
            self.mod = _dev.clamp(-3.0, 3.0)
            self.nll_slow = 0.99 * self.nll_slow + 0.01 * _nll.detach()
        if NLMS:
            # η_t = η̄ / ‖h‖²：输入能量大时自动缩小步长（非扩张更新）
            _E = float((self.hoff * self.hoff).sum(dim=1).mean()) + 1e-8
            dw = (self.hoff.t() @ err) / (B * _E)
            if not NOFEAT:
                if self.mask_out is not None:
                    dw = dw * self.mask_out
                if self.vel_W is not None:
                    self.vel_W.mul_(MOM).add_(dw)
                    dw = self.vel_W
                self.W_out += float(LR_SCALE) * dw
        else:
            dw = self.hoff.t() @ err / B   # (NF,B)@(B,V) -> (NF,V)，与 W_out 同形
            if not NOFEAT:
                if self.mask_out is not None:
                    dw = dw * self.mask_out
                if self.vel_W is not None:
                    self.vel_W.mul_(MOM).add_(dw)
                    dw = self.vel_W
                self.W_out += self.eta * dw
        if RULE == 'three':
            # 三因子规则（Bi & Poo 1998 的时序 + Schultz 1997 的相位性多巴胺）
            # Δw_ij = A · pre_i · (post_j − post̅_j) · m
            #   pre_i      输入活动
            #   post−post̅  中央化后活动（高于自身平均→LTP，低于→LTD，等价于帧内竞争）
            #   m          全局标量：惊讶度对慢平均的偏离（相位性调质，不是持续水平）
            _e = self.e_last.float()
            _h = self.h.float()
            _post_mean = self.post_ema if hasattr(self, 'post_ema') else _h.mean(dim=0)
            _post_mean = 0.99 * _post_mean + 0.01 * _h.mean(dim=0).detach()
            self.post_ema = _post_mean
            _m = self.mod if hasattr(self, 'mod') else torch.tensor(0.0)
            _dW = (_e.t() @ (_h - _post_mean.unsqueeze(0))) * _m / B
            self.W_in += RULE_ETA * _dW * self.mask_in
        if RULE == 'stdp':
            # Δw = A·(pre_tr·post − pre·post_tr)，乘 mask 只在既有连接上更新（Bi & Poo 1998 形式）
            _e = self.e_last.float()
            _h = self.h.float()
            _pre = _e.mean(dim=0)                    # (MIN,) 每条输入的平均活动
            _post = _h.mean(dim=0)                   # (N,)
            self.tr_pre.mul_(0.9).add_(_pre.unsqueeze(1).expand_as(self.tr_pre))
            self.tr_post.mul_(0.9).add_(_post.unsqueeze(0).expand_as(self.tr_post))
            _dW = (self.tr_pre * _post.unsqueeze(0) - _pre.unsqueeze(1) * self.tr_post)
            self.W_in += RULE_ETA * _dW * self.mask_in
        if RULE == 'oja':
            # Oja 规则：Δw = η·post·(pre − post·w)，只更新既有连接（乘 mask）
            _e = self.e_last
            _h = self.h
            _dW = (_e.t() @ _h / B) - self.W_in * (_h * _h).mean(dim=0).unsqueeze(0)
            self.W_in += RULE_ETA * _dW * self.mask_in
        # 偏置：LMS 稳定性要求 η ≤ 1/输入能量；偏置的"输入"恒为 1，所以 η_b = 1。
        # 原来用权重的 η = 1/(NF·rate)，慢了 25 倍 → 稀有词的偏置根本来不及收敛（实测 bias-only 5344 vs 数数 2706）。
        if not NOBIAS:
            self.b += float(LR_SCALE) * err.mean(dim=0)
        if GROW and self.n_alive < self.n_cap and self.clock > 0 and self.clock % GROW_WIN == 0:
            # 误差停滞 → 出生一批（"长满即停"由上限 N 决定，不由手填的层数决定）
            _recent = float(getattr(self, '_err_win', torch.tensor(1.0)))
            if not hasattr(self, '_err_prev'):
                self._err_prev = _recent
            if _recent > 0.995 * self._err_prev:
                _new = min(self.n_cap, self.n_alive + STEP)
                self.alive[self.n_alive:_new] = 1.0
                print('  [grow] n_alive %d -> %d at clock=%d (err=%.1f)' % (
                    self.n_alive, _new, self.clock, _recent), flush=True)
                self.n_alive = _new
            self._err_prev = _recent
        self._err_win = torch.tensor(float((-torch.log(r_hat_next[torch.arange(B, device=DEV), y]
                                                        .clamp(min=1e-12))).mean()))
        if self.avg_on:
            self.avg_n += 1
            _f = 1.0 / self.avg_n
            self.avg_W.add_(self.W_out - self.avg_W, alpha=_f)
            self.avg_b.add_(self.b - self.avg_b, alpha=_f)
        if self.clock % 200 == 0:
            self.dw_stat = float(dw.abs().mean())


def eval_ppl(m, ids, recs):
    m.h.zero_()
    m.h_prev.zero_()
    if hasattr(m, 's_slow'):
        m.s_slow.zero_()
    m.r_hat.zero_()
    nll, n = 0.0, 0
    seg = len(ids) // B
    mat = torch.from_numpy(ids[:seg * B].reshape(seg, B)).to(DEV)   # (时间, B)：mat[t] 是 t 时刻的 B 个 token
    with torch.no_grad():
        cur = mat[0]
        for t in range(seg - 1):
            nxt = mat[t + 1]
            r_hat = m.step(cur, learn=False)
            if t < recs * 200:
                p = r_hat[torch.arange(B), nxt].clamp(min=1e-12)
                nll -= float(torch.log(p).mean())
                n += 1
            m.r_hat = r_hat
            cur = nxt
    return math.exp(nll / max(n, 1)), n


def diag_corr(m, ids, n=20000):
    """拓扑诊断：固定接线、喂 n 个 token，看神经元响应的两两相关（含有效维数）。"""
    seg = n // B
    mat = torch.from_numpy(ids[:seg * B].reshape(seg, B)).to(DEV)
    H = []
    m.h.zero_(); m.r_hat.zero_()
    with torch.no_grad():
        cur = mat[0]
        for t in range(seg - 1):
            m.step(cur, learn=False)
            H.append(m.h.clone())
            cur = mat[t + 1]
    H = torch.cat(H, dim=0)
    Hc = H - H.mean(dim=0, keepdim=True)
    sd = Hc.std(dim=0).clamp(min=1e-9)
    C = (Hc / sd).t() @ (Hc / sd) / H.shape[0]
    N_ = C.shape[0]
    off = (C.sum() - C.diagonal().sum()) / (N_ * (N_ - 1))
    ev = torch.linalg.eigvalsh(Hc.t() @ Hc / H.shape[0]).clamp(min=0)
    p_ = ev / ev.sum().clamp(min=1e-12)
    eff = float(torch.exp(-(p_ * torch.log(p_ + 1e-12)).sum()))   # 有效维数（参与比）
    print('[corr] 平均|相关|=%.4f  最大相关=%.3f  有效维数=%.1f / %d' % (
        float(off), float(C.fill_diagonal_(0).max()), eff, N_), flush=True)
    return H


def main():
    ids_tr, ids_va = load_ids()
    ids_tr = ids_tr[:TOKENS]
    print('[bio] V=%d N_env=%d N_cap=%d SPP=%.0f tokens=%d' % (
        V, N, N_CAP, SPP, len(ids_tr)), flush=True)
    m = Bio()
    print('[bio] eta=%.3e  W_in nnz=%d  U nnz=%d' % (
        m.eta, int((m.W_in != 0).sum()), int((m.U != 0).sum())), flush=True)

    if os.environ.get('B2_DIAG') == '1':
        diag_corr(m, ids_tr)
        return
    seg = len(ids_tr) // B
    mat = torch.from_numpy(ids_tr[:seg * B].reshape(seg, B)).to(DEV)
    t0 = time.time()
    for ep in range(EPOCHS):
      cur = mat[0]
      m.h.zero_()
      m.h_prev.zero_()
      m.s_slow.zero_()
      m.r_hat.zero_()
      for t in range(seg - 1):
        nxt = mat[t + 1]
        r_hat = m.step(cur, learn=True)
        m.learn_out(r_hat, nxt)
        m.r_hat = r_hat
        cur = nxt
        m.clock += 1
        if (t + 1) % 500 == 0:
            with torch.no_grad():
                p = r_hat[torch.arange(B), nxt].clamp(min=1e-12)
                _tr = math.exp(-float(torch.log(p).mean()))
            print('  ep=%d t=%d %.0f tok/s h_rate=%.4f |dw|=%.2e trainPPL=%.1f' % (
                ep, t + 1, (t + 1) * B / (time.time() - t0), float(m.h.mean()), m.dw_stat, _tr), flush=True)
            if _psutil is not None:
                rss = _psutil.Process().memory_info().rss / 2 ** 30
                if rss > MAX_RSS_GB:
                    print('[mem-guard] RSS %.2f > %.2f' % (rss, MAX_RSS_GB), flush=True)
                    raise SystemExit('RSS budget exceeded')

    ppl, n = eval_ppl(m, ids_va, EVAL_RECS)
    if getattr(m, 'avg_on', False) and m.avg_n > 0:
        _w, _b = m.W_out.clone(), m.b.clone()
        m.W_out.copy_(m.avg_W); m.b.copy_(m.avg_b)
        ppl_avg, _ = eval_ppl(m, ids_va, EVAL_RECS)
        m.W_out.copy_(_w); m.b.copy_(_b)
        print('[avg] VAL PPL raw=%.1f  polyak=%.1f' % (ppl, ppl_avg), flush=True)
    print('[bio] ENC=%s | VAL PPL %.1f (n=%d) | %.0f tok/s total' % (
        ENC, ppl, n, seg * B / (time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()

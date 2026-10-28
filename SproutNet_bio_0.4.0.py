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

torch.set_num_threads(int(os.environ.get('B2_THREADS', '16')))
torch.manual_seed(int(os.environ.get('B2_SEED', '0')))
np.random.seed(int(os.environ.get('B2_SEED', '0')))

ROOT = os.path.dirname(os.path.abspath(__file__))
V = 10000
N = int(os.environ.get('B2_N', '512'))
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

_psutil = None
try:
    import psutil as _psutil
except ImportError:
    pass


# ---------- 数据 ----------
def load_ids():
    import glob
    cand = sorted(glob.glob(os.path.join(ROOT, '_sprout_build', 'tokcache_325b036129_10000.npz')))
    if not cand:
        cand = sorted(glob.glob(os.path.join(ROOT, '_sprout_build', 'tokcache_*10000.npz')))
    ids = np.load(cand[0])['ids']
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
            self.W_in = torch.randn(V, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * V)))
            self.W_in *= (torch.rand(V, N, generator=g) < P0).float()
        self.U = torch.randn(N, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * N)))
        self.U *= (torch.rand(N, N, generator=g) < P0).float()
        # 输出：预测下一时刻感受器模式。初始极小 → r̂ 近似均匀
        self.NF = N * (2 if HIST else 1)      # 读出特征维（HIST=1 时拼接 h_{t-1}）
        self.h_prev = torch.zeros(B, N)
        self.W_out = torch.zeros(self.NF, V)
        # 输出偏置：没有它模型连「词频」这个最基本的基线都表达不出来
        #（cortex 一直有 self.b，这个原型漏了 → PPL 卡在接近均匀）
        self.b = torch.zeros(V)
        self.theta = torch.full((N,), 0.01)
        self.h = torch.zeros(B, N)
        self.r_hat = torch.zeros(B, V)          # 上一步对"当前时刻"的预测
        self.clock = 0
        # LMS 步长：稳定条件 η ≤ 1/(fan_in · E[h²]) = 1/(N · TARGET_RATE)
        self.eta = LR_SCALE / (self.NF * TARGET_RATE)
        self.dw_stat = 0.0

    def step(self, x, learn=True):
        """x: (B,) 当前时刻的 token"""
        ar = torch.arange(B)
        # 感受器层：r 是 one-hot。误差 = r − r̂（只在 x 处有 +1−r̂，其余 −r̂）
        if ENC == 'pred':
            e = -self.r_hat.clone()
            e[ar, x] += 1.0
        else:
            e = torch.zeros(B, V)
            e[ar, x] = 1.0
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
        if learn:
            # 分位数阈值：锁定群体发放率 = TARGET_RATE（无魔数，直接是实测值）
            self.theta = torch.quantile(d.detach(), 1.0 - TARGET_RATE, dim=0).clamp(min=1e-6)
        h = (d > self.theta.unsqueeze(0)).float()
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
        ar = torch.arange(B)
        target = torch.zeros(B, V)
        target[ar, y] = 1.0
        err = target - r_hat_next
        dw = self.hoff.t() @ err / B   # (NF,B)@(B,V) -> (NF,V)，与 W_out 同形
        if not NOFEAT:
            self.W_out += self.eta * dw
        # 偏置：LMS 稳定性要求 η ≤ 1/输入能量；偏置的"输入"恒为 1，所以 η_b = 1。
        # 原来用权重的 η = 1/(NF·rate)，慢了 25 倍 → 稀有词的偏置根本来不及收敛（实测 bias-only 5344 vs 数数 2706）。
        self.b += float(LR_SCALE) * err.mean(dim=0)
        if self.clock % 200 == 0:
            self.dw_stat = float(dw.abs().mean())


def eval_ppl(m, ids, recs):
    m.h.zero_()
    m.h_prev.zero_()      # 原来漏了这一行：HIST 实验的 eval 会带着训练的残状态
    m.r_hat.zero_()
    nll, n = 0.0, 0
    seg = len(ids) // B
    mat = ids[:seg * B].reshape(seg, B)   # (时间, B)：mat[t] 是 t 时刻的 B 个 token
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


def main():
    ids_tr, ids_va = load_ids()
    ids_tr = ids_tr[:TOKENS]
    print('[bio] V=%d N=%d B=%d enc=%s tokens=%d eta=%.3e' % (V, N, B, ENC, len(ids_tr), 0.0), flush=True)
    m = Bio()
    print('[bio] eta=%.3e  W_in nnz=%d  U nnz=%d' % (
        m.eta, int((m.W_in != 0).sum()), int((m.U != 0).sum())), flush=True)

    seg = len(ids_tr) // B
    mat = ids_tr[:seg * B].reshape(seg, B)
    t0 = time.time()
    for ep in range(EPOCHS):
      cur = mat[0]
      m.h.zero_()
      m.h_prev.zero_()
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
    print('[bio] ENC=%s | VAL PPL %.1f (n=%d) | %.0f tok/s total' % (
        ENC, ppl, n, seg * B / (time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
SproutNet_bio_0.2.0 —— 整链路仿生原型 v2（CPU 快速版）

与 v1 的区别（这是用户提出「整条信息流动链路都要改」的落点）：

  v1 的链路仍是
      one-hot 词号 → 隐藏层 → V 类 softmax 分类器
  三处不仿生：
    ① 感受器是 one-hot 符号。**符号之间没有相似度**（token 1234 与 1235 毫无关系），
       随机投射之后无法泛化 → 每个 (上下文, 词) 对都必须单独见一次 → 数据饥渴。
       皮层的编码是**保相似度**的。
    ② 解码是 V 类分类器（大脑里没有分类器）。
    ③ 没有闭环：预测值从未回到输入端被扣减。

  v2 的链路
      r_t   = emb[x_t]                        感觉表面 = 保相似度的连续编码（64 维）
      e_t   = r_t − r̂_t                        感觉适应：只有「意外」进入皮层
      h_t   = spike(W_in·e_t + U·h_{t-1})      阈值非线性 + 群体稀疏（5%）
      r̂_{t+1} = W_pred·h_t                     预测下一时刻的感觉模式（64 维，不是词表）
      p(v)  ∝ exp(−‖r̂_{t+1} − emb[v]‖² / 2σ²)  解码 = 「我的预测最像哪个已知模式」
      σ²    = 滑动平均的预测误差方差（自适应，不手填）
      W_pred += η·(r_{t+1} − r̂_{t+1}) ⊗ h_t     每条突触只用自己那一路的误差

  **没有分类器，没有 teacher（下一个词就是自然到来的感觉输入），解码器不含任何可学参数。**

开关：
  B2_SENS = embed | onehot     感觉表面用保相似度编码还是 one-hot
  B2_PRED = 1 | 0              输入是否扣掉自己的预测（感觉适应）
"""
import math
import os
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('B2_THREADS', '16')))
SEED = int(os.environ.get('B2_SEED', '0'))
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = os.path.dirname(os.path.abspath(__file__))
V = 10000
N = int(os.environ.get('B2_N', '512'))
B = int(os.environ.get('B2_B', '32'))
P0 = 0.1
TARGET_RATE = 0.05
TOKENS = int(os.environ.get('B2_TOKENS', '1000000'))
EVAL_RECS = int(os.environ.get('B2_EVAL_RECS', '3'))
SENS = os.environ.get('B2_SENS', 'embed')     # embed | onehot
PRED = os.environ.get('B2_PRED', '1') == '1'
LR = float(os.environ.get('B2_LR', '1.0'))
MAX_RSS_GB = float(os.environ.get('B2_MAX_RSS_GB', '8.0'))

_psutil = None
try:
    import psutil as _psutil
except ImportError:
    pass


def load_data():
    import glob
    f = os.path.join(ROOT, '_sprout_build', 'tokcache_325b036129_10000.npz')
    if not os.path.exists(f):
        f = sorted(glob.glob(os.path.join(ROOT, '_sprout_build', 'tokcache_*10000.npz')))[0]
    ids = np.load(f)['ids']
    emb = torch.from_numpy(np.load(os.path.join(ROOT, '_sprout_build', 'token_clusters.npz'))['emb'])
    emb = emb / emb.norm(dim=1, keepdim=True).clamp(min=1e-9)     # 单位化：距离才有意义
    n_val = max(20000, len(ids) // 50)
    return ids[:-n_val].astype(np.int64), ids[-n_val:].astype(np.int64), emb


class Bio:
    def __init__(self, emb):
        g = torch.Generator().manual_seed(SEED * 10000 + 7)
        self.emb = emb                       # (V, D) 固定的感觉表面（不学习）
        D = emb.shape[1]
        self.D = D
        if SENS == 'embed':
            fan_in = max(1, int(P0 * D))
            self.W_in = torch.randn(D, N, generator=g) * (1.0 / math.sqrt(fan_in))
            self.W_in *= (torch.rand(D, N, generator=g) < P0).float()
        else:
            # one-hot 对照：D 维输入 = V 维稀疏（用 (V,N) 投射实现）
            self.W_in_oh = torch.randn(V, N, generator=g) * (1.0 / math.sqrt(max(1.0, P0 * V)))
            self.W_in_oh *= (torch.rand(V, N, generator=g) < P0).float()
        fan_u = max(1, int(P0 * N))
        self.U = torch.randn(N, N, generator=g) * (1.0 / math.sqrt(fan_u))
        self.U *= (torch.rand(N, N, generator=g) < P0).float()
        self.W_pred = torch.zeros(N, D)      # 预测下一时刻的感觉模式
        self.LIN = os.environ.get('B2_LIN', '0') == '1'
        self.Wm = torch.zeros(D, D)          # 诊断：跳过隐藏层的线性预测器
        self.theta = torch.zeros(N)
        self.h = torch.zeros(B, N)
        self.r_hat = torch.zeros(B, D)       # 上一步对「当前时刻」的预测
        self.var = torch.ones(())            # σ²：预测误差的滑动方差（自适应）
        self.clock = 0
        self.eta = LR / (N * TARGET_RATE)
        self.dw_stat = 0.0

    def sense(self, x):
        """把 token 变成感觉模式。one-hot 对照时返回 index，embed 时返回向量。"""
        if SENS == 'embed':
            return self.emb[x]
        return x

    def step(self, x, learn=True):
        ar = torch.arange(B)
        r = self.sense(x)
        if SENS == 'onehot':
            # one-hot 对照：符号之间没有相似度，也无法表达「部分预测」，
            # 所以 PRED 在这一路上恒等于关（显式写出来，避免静默退化）。
            e_oh = torch.zeros(B, V)
            e_oh[ar, x] = 1.0
            d = e_oh @ self.W_in_oh + self.h @ self.U
        else:
            e = r - self.r_hat if PRED else r
            d = e @ self.W_in + self.h @ self.U
        if learn:
            self.theta = torch.quantile(d.detach(), 1.0 - TARGET_RATE, dim=0).clamp(min=1e-9)
        h = (d > self.theta.unsqueeze(0)).float()
        self.h = h
        # 预测下一时刻的感觉模式
        if self.LIN:
            self.r_now = r
            return None
        r_hat_next = h @ self.W_pred
        return r_hat_next

    def decode(self, r_hat_next):
        if self.LIN:
            r_hat_next = self.r_hat_next_lin
        """解码：预测最像哪个已知感觉模式。无可学参数，用自适应的 σ²。"""
        # ‖r̂−emb‖² = ‖r̂‖² − 2 r̂·embᵀ + ‖emb‖²；用内积避免 (B,V,D) 三维张量
        d2 = (r_hat_next.pow(2).sum(dim=1).unsqueeze(1)
              - 2.0 * (r_hat_next @ self.emb.t())
              + self.emb.pow(2).sum(dim=1).unsqueeze(0))
        logits = -d2 / (2.0 * self.var.clamp(min=1e-6))
        return torch.softmax(logits, dim=1)

    def learn(self, r_hat_next, r_next):
        if self.LIN:
            # 诊断通路：r̂_{t+1} = r_t @ Wm，ΔWm = η·r_tᵀ·err / B（同样是局部误差）
            r_hat = self.r_now @ self.Wm
            err = r_next - r_hat
            self.Wm += (LR / max(1, self.D)) * (self.r_now.t() @ err / B)
            with torch.no_grad():
                mse = float(err.pow(2).sum(dim=1).mean())
                self.var.mul_(0.99).add_(torch.tensor(0.01 * mse))
            self.r_hat_next_lin = r_hat
            return
        err = r_next - r_hat_next                       # (B,D) 局部误差
        dw = self.h.t() @ err / B                       # (N,B)@(B,D) -> (N,D)
        self.W_pred += self.eta * dw
        # σ² 用预测误差的滑动均方（自适应，不手填）
        with torch.no_grad():
            mse = float(err.pow(2).sum(dim=1).mean())
            self.var.mul_(0.99).add_(torch.tensor(0.01 * mse))
        if self.clock % 200 == 0:
            self.dw_stat = float(dw.abs().mean())


def eval_ppl(m, ids, recs):
    m.h.zero_()
    m.r_hat.zero_()
    nll, n = 0.0, 0
    seg = len(ids) // B
    mat = ids[:seg * B].reshape(seg, B)
    with torch.no_grad():
        cur = mat[0]
        for t in range(seg - 1):
            nxt = mat[t + 1]
            r_hat = m.step(cur, learn=False)
            if m.LIN:
                r_hat = m.r_hat_next_lin
            p = m.decode(r_hat)
            if t < recs * 100:
                nll -= float(torch.log(p[torch.arange(B), nxt].clamp(min=1e-12)).mean())
                n += 1
            m.r_hat = m.sense(nxt)
            cur = nxt
    return math.exp(nll / max(n, 1)), n


def main():
    ids_tr, ids_va, emb = load_data()
    ids_tr = ids_tr[:TOKENS]
    D = emb.shape[1]
    assert SENS == 'embed', ('0.2.0 只支持 embed 感觉表面；'
                             'one-hot 对照请用 SproutNet_bio_0.1.0.py（B2_ENC=onehot）')
    m = Bio(emb)
    print('[bio2] V=%d N=%d B=%d D=%d sens=%s pred=%s tokens=%d eta=%.3e'
          % (V, N, B, D, SENS, PRED, len(ids_tr), m.eta), flush=True)

    seg = len(ids_tr) // B
    mat = ids_tr[:seg * B].reshape(seg, B)
    t0 = time.time()
    cur = mat[0]
    for t in range(seg - 1):
        nxt = mat[t + 1]
        r_hat = m.step(cur, learn=True)
        m.learn(r_hat, m.sense(nxt))
        if m.LIN:
            r_hat = m.r_hat_next_lin
        m.r_hat = m.sense(nxt)
        cur = nxt
        m.clock += 1
        if (t + 1) % 500 == 0:
            with torch.no_grad():
                p = m.decode(r_hat)
                tr = math.exp(-float(torch.log(p[torch.arange(B), nxt].clamp(min=1e-12)).mean()))
            _lp = torch.log(p.clamp(min=1e-30))
            print('  t=%d %.0f tok/s h=%.4f |dw|=%.2e var=%.4f trainPPL=%.1f '
                  '|Wp|=%.3f |rh|=%.3f logit_std=%.4f acc=%.3f' % (
                      t + 1, (t + 1) * B / (time.time() - t0), float(m.h.mean()),
                      m.dw_stat, float(m.var), tr,
                      float(m.W_pred.abs().mean()), float(r_hat.norm(dim=1).mean()),
                      float(_lp.std(dim=1).mean()),
                      float((p.argmax(dim=1) == nxt).float().mean())), flush=True)
            if _psutil is not None:
                rss = _psutil.Process().memory_info().rss / 2 ** 30
                if rss > MAX_RSS_GB:
                    print('[mem-guard] RSS %.2f > %.2f' % (rss, MAX_RSS_GB), flush=True)
                    raise SystemExit('RSS budget exceeded')

    ppl, n = eval_ppl(m, ids_va, EVAL_RECS)
    print('[bio2] sens=%s pred=%s | VAL PPL %.1f (n=%d) | %.0f tok/s total' % (
        SENS, PRED, ppl, n, seg * B / (time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()

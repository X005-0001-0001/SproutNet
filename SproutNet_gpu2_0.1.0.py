# ================================================================================================
# SproutNet · gpu2 分支 · 0.1.0  —— backoff 0.1.0 内核的 GPU(torch/CUDA) 移植
#
# 目的：让同一个大脑吃到全量数据（CPU 约 5 小时/轮；GPU 目标 10-30 分钟/轮），
#   并保留全部机制：内存预算（SPROUT_MEM_MB）、共激活生长、使用率排名淘汰、
#   回退平滑（γ=实测 miss 率）、局部 delta 学习、无 BP。
#
# 移植约定：
#   - 数值状态用 float32 放显存；掩码缓存为索引张量（numel() 无同步）；
#   - 生长/淘汰的随机决策仍用 numpy RNG（CPU），结构写入为显存张量；
#   - 检查点仍为 npz（可被 CPU 分支读取，反之亦然）；
#   - 与 CPU 版差异：RNG 序列不同、float32 精度，PPL 允许 ±1% 偏差。
# ================================================================================================
import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('base', os.path.join(ROOT, 'SproutNet_backoff_0.1.0.py'))
base = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ['gpu2_base']
_spec.loader.exec_module(base)
sys.argv = _argv

SPROUT_BRANCH = 'gpu2'
SPROUT_VERSION = '0.1.0'
VOCAB_SIZE = base.VOCAB_SIZE
GP = dict(base.SPROUT_GP)
CAP_INT = base.CAP_INT
CAP_SYN = base.CAP_SYN
CAP_COINC = base.CAP_COINC
MEM_BYTES = base._MEM_BYTES
INT_BYTES = base._INT_BYTES
SYN_BYTES = base._SYN_BYTES
COINC_BYTES = base._COINC_BYTES
CHECKPOINT = 'brain_gpu2.pt'
LOG_FILE = 'training_log_gpu2.txt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DT = torch.float32
LOG_EVERY_TOKENS = base.LOG_EVERY_TOKENS if hasattr(base, 'LOG_EVERY_TOKENS') else 2000


def _log(msg):
    line = str(msg)
    print(line, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


class GpuBrain:
    def __init__(self, V, seed=1):
        gp = GP
        self.gp = gp
        self.V = V
        self.N = 2 * V + CAP_INT
        self.rng = np.random.RandomState(seed)
        self.dev = DEV
        d = torch.device(DEV)
        self.d = d
        self.tg = torch.Generator(device=d)
        self.tg.manual_seed(seed)
        z = lambda n: torch.zeros(n, device=d, dtype=DT)
        self.lam_vec = torch.empty(self.N, device=d, dtype=DT)
        self.lam_vec[:V] = gp['lam']
        self.lam_vec[V:2 * V] = gp['lam_slow']
        self.lam_vec[2 * V:] = gp['lam']
        self.v = z(self.N)
        self.u = z(self.N)
        self.ap = z(self.N)
        self.av = z(self.N)
        self.var = z(self.N)
        self.atp = torch.ones(self.N, device=d, dtype=DT)
        self.u_util = z(self.N)
        self.gamma = 1.0
        self.int_alive = torch.zeros(self.N, dtype=torch.bool, device=d)
        self.free_slots = list(range(2 * V, self.N))      # 低位优先分配：hw 保持紧凑（replay 才真的小）
        self.b = z(V)
        self.b_clip = max(2.0, math.log(V))
        self.syn_cap = CAP_SYN
        self.s_i = torch.zeros(self.syn_cap, dtype=torch.long, device=d)
        self.s_j = torch.zeros(self.syn_cap, dtype=torch.long, device=d)
        self.s_w = z(self.syn_cap)
        self.s_alive = torch.zeros(self.syn_cap, dtype=torch.bool, device=d)
        self.s_usage = z(self.syn_cap)
        self.s_pre = z(self.syn_cap)
        self.n_syn = 0
        self.syn_free = []
        self.coinc_cap = CAP_COINC
        self.fan = max(1, min(gp['birth_rec_in'], V))
        self.c_i = torch.zeros(self.coinc_cap, dtype=torch.long, device=d)
        self.c_j = torch.zeros(self.coinc_cap, dtype=torch.long, device=d)
        self.c_rows = torch.zeros((self.coinc_cap, self.fan), dtype=torch.long, device=d)
        self.c_w = torch.zeros((self.coinc_cap, self.fan), device=d, dtype=DT)
        self.c_usage = z(self.coinc_cap)
        self.c_alive = torch.zeros(self.coinc_cap, dtype=torch.bool, device=d)
        self.n_coinc = 0
        self.coinc_free = []
        self.e_s = z(self.syn_cap)
        self.e_c = torch.zeros((self.coinc_cap, self.fan), device=d, dtype=DT)
        self.elig_lam = float(gp['lam_slow'])
        self.elig_alpha = 1.0 - self.elig_lam
        self.scar_syn = set()
        self.syn_keys = set()
        self.coinc_keys = set()
        self._idx_dirty = True
        self._mro = None
        self._mii = None
        self._mintro = None
        self._mc = None
        self._refresh_age = 99
        self.recent = []
        self._since_replay = 0
        self.err = torch.ones(V, device=d, dtype=DT)
        self.err_slow = torch.ones(V, device=d, dtype=DT)
        self.hab = z(V)
        self.nov = 1.0
        self.nov_slow = 1.0
        self.G = gp['neuro_cost'] * (1.0 + math.log(2 * V) / math.log(100.0))
        self.clock = 0
        self.spines = 0
        self.births = 0
        self.deaths = 0
        self.max_n = 2 * V
        self.hw = 2 * V
        self.last_e = z(V)
        self.prev_tok = None
        self._nov_t = torch.tensor(1.0, device=d)
        self._nov_slow_t = torch.tensor(1.0, device=d)
        self._cost = torch.zeros((), device=d)
        self._nll = torch.zeros((), device=d)
        self._nacc = torch.zeros((), device=d, dtype=torch.long)
        self._hit = torch.zeros((), device=d, dtype=torch.long)
        self._cache_age = 10 ** 9
        self._err_np = None
        self._hab_np = None
        self._ap_np = None
        self._av_np = None
        self._alive_np = None

    def _stats_np(self, fresh=False):
        if fresh or self._err_np is None or self._cache_age > 8:
            self._err_np = self.err.detach().cpu().numpy()
            self._hab_np = self.hab.detach().cpu().numpy()
            self._alive_np = self.int_alive.detach().cpu().numpy()
            self._cache_age = 0
        return self._err_np, self._hab_np

    def _activity_np(self):
        self._ap_np = self.ap.detach().cpu().numpy()
        self._av_np = self.av.detach().cpu().numpy()
        return self._ap_np, self._av_np

    def train_stats(self):
        n = int(self._nacc.item())
        if n == 0:
            return float('inf'), 0, 0.0
        ppl = math.exp(float(self._nll.item()) / n)
        acc = float(self._hit.item()) / n
        self._nll.zero_()
        self._nacc.zero_()
        self._hit.zero_()
        return ppl, n, acc

    # ---------- helpers ----------
    def _refresh(self, force=False):
        # 代价高（nonzero 触发同步）：最多每 4 步重建一次；结构变更时由 kill_int 强制。
        if not self._idx_dirty:
            return
        if not force and self._refresh_age < 4:
            self._refresh_age += 1
            return
        self._mro = torch.nonzero(self.s_alive & (self.s_j < self.V), as_tuple=False).flatten()
        self._mii = torch.nonzero(self.s_alive & (self.s_j >= 2 * self.V), as_tuple=False).flatten()
        self._mintro = torch.nonzero(self.s_alive & (self.s_j < self.V) & (self.s_i >= 2 * self.V),
                                     as_tuple=False).flatten()
        self._mc = torch.nonzero(self.c_alive, as_tuple=False).flatten()
        self._idx_dirty = False
        self._refresh_age = 0

    def _bytes_used(self):
        return (int(self.int_alive.sum().item()) * INT_BYTES
                + int(self.s_alive.sum().item()) * SYN_BYTES
                + int(self.c_alive.sum().item()) * COINC_BYTES)

    def _add_syn(self, i, j, w):
        if (i, j) in self.syn_keys:
            return False
        if self.syn_free:
            idx = self.syn_free.pop()
        elif self.n_syn < self.syn_cap:
            idx = self.n_syn
            self.n_syn += 1
        else:
            alive = torch.nonzero(self.s_alive, as_tuple=False).flatten()
            if alive.numel() == 0:
                return False
            idx = int(alive[int(torch.argmin(self.s_usage[alive]).item())].item())
            self.syn_keys.discard((int(self.s_i[idx].item()), int(self.s_j[idx].item())))
        self.s_i[idx] = i
        self.s_j[idx] = j
        self.s_w[idx] = w
        self.s_alive[idx] = True
        self.s_usage[idx] = 0.0
        self.s_pre[idx] = 0.0
        self.syn_keys.add((i, j))
        self._idx_dirty = True
        return True

    def _add_coinc(self, i, j, rows, w):
        key = (i, j)
        if key in self.coinc_keys:
            return False
        if self.coinc_free:
            idx = self.coinc_free.pop()
        elif self.n_coinc < self.coinc_cap:
            idx = self.n_coinc
            self.n_coinc += 1
        else:
            alive = torch.nonzero(self.c_alive, as_tuple=False).flatten()
            if alive.numel() == 0:
                return False
            idx = int(alive[int(torch.argmin(self.c_usage[alive]).item())].item())
            self.coinc_keys.discard((int(self.c_i[idx].item()), int(self.c_j[idx].item())))
        self.c_i[idx] = i
        self.c_j[idx] = j
        self.c_rows[idx] = torch.as_tensor(rows, dtype=torch.long, device=self.d)
        self.c_w[idx] = w
        self.c_usage[idx] = 0.0
        self.c_alive[idx] = True
        self.coinc_keys.add(key)
        return True

    def new_int(self):
        if not self.free_slots:
            return -1
        nid = self.free_slots.pop(0)
        self.int_alive[nid] = True
        self.v[nid] = 0.0
        self.u[nid] = 0.0
        self.ap[nid] = 0.0
        self.av[nid] = 0.0
        self.atp[nid] = 1.0
        self.max_n = max(self.max_n, int(self.int_alive.sum().item()) + 2 * self.V)
        self.hw = max(self.hw, nid + 1)
        return nid

    def kill_int(self, nid):
        self.int_alive[nid] = False
        self.v[nid] = 0.0
        self.u[nid] = 0.0
        self.ap[nid] = 0.0
        self.av[nid] = 0.0
        self.free_slots.append(nid)
        ms = torch.nonzero(self.s_alive & ((self.s_i == nid) | (self.s_j == nid)), as_tuple=False).flatten()
        if ms.numel():
            for idx in ms.tolist():
                self.syn_keys.discard((int(self.s_i[idx].item()), int(self.s_j[idx].item())))
            self.s_alive[ms] = False
            self.syn_free.extend(int(x) for x in ms.tolist())
        mcc = torch.nonzero(self.c_alive & ((self.c_i == nid) | (self.c_j == nid)), as_tuple=False).flatten()
        if mcc.numel():
            for idx in mcc.tolist():
                self.coinc_keys.discard((int(self.c_i[idx].item()), int(self.c_j[idx].item())))
            self.c_alive[mcc] = False
            self.coinc_free.extend(int(x) for x in mcc.tolist())
        self._idx_dirty = True
        self._refresh_age = 99
        self.deaths += 1

    def _admit(self, need_bytes, protect=-1):
        if self._bytes_used() + need_bytes <= MEM_BYTES:
            return True
        guard = 0
        while guard < 64:
            guard += 1
            alive = torch.nonzero(self.int_alive, as_tuple=False).flatten()
            if alive.numel() == 0:
                return False
            vals = self.u_util[alive].clone()
            if protect >= 0:
                vals[alive == protect] = float('inf')
            nid = int(alive[int(torch.argmin(vals).item())].item())
            self.kill_int(nid)
            if self._bytes_used() + need_bytes <= MEM_BYTES:
                return True
        return False

    def reset_state(self):
        self.v.zero_()
        self.u.zero_()
        self.ap.zero_()
        self.prev_tok = None

    # ---------- core ----------
    def logits_from(self, ap):
        self._refresh()
        l = self.b.clone()
        mro = self._mro
        if mro.numel():
            an = ap / (torch.sqrt(self.var) + 0.1)
            l.index_add_(0, self.s_j[mro], self.s_w[mro] * an[self.s_i[mro]])
        mc = self._mc
        if mc.numel():
            fp = ap[self.c_i[mc]] * ap[self.c_j[mc]]
            rows = self.c_rows[mc].reshape(-1)
            vals = (fp[:, None] * self.c_w[mc]).reshape(-1)
            l.index_add_(0, rows, vals)
        q = torch.softmax(l, dim=0)
        u = torch.softmax(self.b, dim=0)
        m = (1.0 - self.gamma) * q + self.gamma * u
        return torch.log(m + 1e-12)

    def step(self, x, logits, learn):
        gp = self.gp
        self.clock += 1
        if learn and self.prev_tok is not None:
            hit = (int(self.prev_tok), int(x)) in self.syn_keys
            self.gamma += gp['e_ema'] * ((0.0 if hit else 1.0) - self.gamma)
        if learn and self.prev_tok is not None and self.prev_tok != x:
            key = (int(self.prev_tok), int(x))
            if key not in self.syn_keys and key not in self.scar_syn:
                sgn = 1.0 if float(self.last_e[x].item()) >= 0.0 else -1.0
                if self._add_syn(key[0], key[1], gp['spine_w0'] * sgn):
                    self.spines += 1
        if logits is not None:
            p = torch.softmax(logits, dim=0)
            e = -p
            e[x] += 1.0
            self.last_e = e
            if learn:
                self.hab[x] = self.elig_lam * self.hab[x] + self.elig_alpha * e[x]
                self._nll += -torch.log(torch.clamp(p[x], min=1e-12))
                self._nacc += 1
                self._hit += (torch.argmax(p) == x).long()
                self.plasticity(e)
        self._refresh()
        mii = self._mii
        if mii.numel():
            rec = torch.zeros(self.N, device=self.d, dtype=DT)
            rec.index_add_(0, self.s_j[mii], self.s_w[mii] * self.ap[self.s_i[mii]])
        else:
            rec = torch.zeros(self.N, device=self.d, dtype=DT)
        noise = torch.randn(self.N, device=self.d, dtype=DT, generator=self.tg) * gp['sigma']
        self.v = self.lam_vec * self.v + rec - gp['g_adapt'] * self.u + noise
        self.v[x] += gp['c_in']
        self.v[x + self.V] += gp['c_in']
        self.v += gp['c_basal'] * (1.0 - self.lam_vec) * self.int_alive
        a = torch.clamp(self.v, 0.0, 1.0)
        self.u += gp['tau_adapt'] * (a - self.u)
        self.av += 0.01 * (a - self.av)
        self.var += 0.01 * ((a - self.av) ** 2 - self.var)
        self.ap = a
        self.prev_tok = int(x)
        return self.logits_from(a)

    def plasticity(self, e):
        gp = self.gp
        self._refresh()
        mro, mii, mintro = self._mro, self._mii, self._mintro
        es = 1.0 / (1.0 + self.clock / gp['eta_age_tau'])
        nov = (e * e).sum()
        rpe = torch.tanh(gp['rpe_gain'] * (self._nov_slow_t - nov))
        self._nov_t += gp['nov_ema'] * (nov - self._nov_t)
        self._nov_slow_t += gp['nov_slow'] * (nov - self._nov_slow_t)
        blame = torch.zeros(self.N, device=self.d, dtype=DT)
        if mro.numel():
            blame.index_add_(0, self.s_i[mro], self.s_w[mro] * e[self.s_j[mro]])
        mc = self._mc
        if mc.numel():
            ci = self.c_i[mc]
            cj = self.c_j[mc]
            api = self.ap[ci]
            apj = self.ap[cj]
            contrib = self.c_w[mc] * e[self.c_rows[mc]]
            ci_rep = ci.repeat_interleave(self.fan)
            cj_rep = cj.repeat_interleave(self.fan)
            blame.index_add_(0, ci_rep, (contrib * apj[:, None]).reshape(-1))
            blame.index_add_(0, cj_rep, (contrib * api[:, None]).reshape(-1))
        an = self.ap / (torch.sqrt(self.var) + 0.1)
        lam_e = self.elig_lam
        alp_e = self.elig_alpha
        if mro.numel():
            prod = an[self.s_i[mro]] * e[self.s_j[mro]]
            self.e_s[mro] = lam_e * self.e_s[mro] + alp_e * prod
            d = gp['eta_r'] * es * self.e_s[mro]
            self.s_w[mro] = torch.clamp(self.s_w[mro] + d, -gp['w_clip'], gp['w_clip'])
            self._cost += d.abs().sum()
        if mintro.numel():
            pre_r = self.ap[self.s_i[mintro]]
            post_r = self.ap[self.s_j[mintro]]
            self.s_usage[mintro] = 0.98 * self.s_usage[mintro] + pre_r * post_r
            self.s_pre[mintro] = 0.98 * self.s_pre[mintro] + pre_r
            weak = (self.s_usage[mintro] < gp['w_res']) & (self.s_pre[mintro] >= gp['w_res'])
            w_idx = mintro[weak]
            if w_idx.numel():
                self.s_w[w_idx] *= (1.0 - gp['wd'])
        if mc.numel():
            fpn = an[self.c_i[mc]] * an[self.c_j[mc]]
            prod = fpn[:, None] * e[self.c_rows[mc]]
            self.e_c[mc] = lam_e * self.e_c[mc] + alp_e * prod
            d = gp['eta_coinc'] * es * self.e_c[mc]
            self.c_w[mc] = torch.clamp(self.c_w[mc] + d, -gp['w_clip'], gp['w_clip'])
            self._cost += d.abs().sum()
        if mii.numel():
            pre = self.ap[self.s_i[mii]]
            post = self.ap[self.s_j[mii]]
            mod = torch.tanh(gp['blame_gain'] * blame[self.s_j[mii]]) + 0.3 * rpe
            prod = pre * post * mod
            self.e_s[mii] = lam_e * self.e_s[mii] + alp_e * prod
            d = gp['eta_i'] * es * self.e_s[mii]
            self.s_w[mii] += d
            self._cost += d.abs().sum()
            self.s_usage[mii] = 0.98 * self.s_usage[mii] + pre * post
            self.s_pre[mii] = 0.98 * self.s_pre[mii] + pre
            weak = (self.s_usage[mii] < gp['w_res']) & (self.s_pre[mii] >= gp['w_res'])
            w_idx = mii[weak]
            if w_idx.numel():
                self.s_w[w_idx] *= (1.0 - gp['wd'])
        self.b += gp['eta_b'] * es * e
        self.b.clamp_(-self.b_clip, self.b_clip)
        if mc.numel():
            self.c_usage[mc] = 0.99 * self.c_usage[mc] + 0.01 * self.ap[self.c_i[mc]] * self.ap[self.c_j[mc]]
        # (代谢成本在 energy() 中合并同步，避免每步 .item())
        self.err += gp['e_ema'] * (e.abs() - self.err)
        self.err_slow += gp['e_ema'] * 0.1 * (e.abs() - self.err_slow)

    def energy(self):
        gp = self.gp
        if self.clock % 8 != 0:
            return
        self._refresh()
        sf = 1.0 + math.log(1.0 + self.V) / math.log(25.0) if hasattr(self, 'V') else 1.0
        self.nov = float(self._nov_t.item())
        inc = gp['pool_in'] * sf * (1.0 + gp['stress_gain'] * self.nov / (1.0 + self.nov))
        self.G -= gp['m_plast'] * float(self._cost.item()) * 8.0
        self._cost.zero_()
        self.G += inc
        n_int = int(self.int_alive.sum().item())
        spend = gp['m_receptor'] * 2 * sf + gp['m_neuron'] * n_int + gp['m_spike'] * float(self.ap.sum().item())
        mro = self._mro
        if mro.numel():
            spend += gp['m_syn'] * int(self.s_alive.sum().item())
            spend += gp['m_syn_w'] * float((self.s_w[mro].abs() * self.ap[self.s_i[mro]]).sum().item())
        mc = self._mc
        if mc.numel():
            spend += gp['m_syn'] * mc.numel()
            spend += gp['m_syn_w'] * float(
                (self.c_w[mc].abs().sum(dim=1) * (self.ap[self.c_i[mc]] * self.ap[self.c_j[mc]])).sum().item())
        self.G -= spend
        if self.G > 0.0:
            self.G -= gp['g_leak'] * self.G
        rmask = torch.arange(2 * self.V, device=self.d)
        self.atp[rmask] = torch.minimum(torch.ones(2 * self.V, device=self.d, dtype=DT),
                                        self.atp[rmask] * gp['atp_decay'] + gp['use_gain'] * self.ap[rmask]
                                        + gp['receptor_supply'])
        ms_use = torch.nonzero(self.s_alive & (self.s_i >= 2 * self.V) & (self.s_j < self.V),
                               as_tuple=False).flatten()
        use_inc = torch.zeros(self.N, device=self.d, dtype=DT)
        if ms_use.numel():
            use_inc.index_add_(0, self.s_i[ms_use], self.s_usage[ms_use])
        ii = torch.nonzero(self.int_alive, as_tuple=False).flatten()
        if ii.numel():
            self.atp[ii] = (self.atp[ii] * gp['atp_decay'] + gp['int_basal'] - gp['m_int_upkeep']
                            + 0.5 * gp['use_gain'] * self.ap[ii] + gp['util_gain'] * use_inc[ii])
            self.u_util[ii] = self.u_util[ii] * gp['atp_decay'] + use_inc[ii]
        if self.G < 0.0:
            self.G = 0.0

    def growth(self):
        gp = self.gp
        err_np, hab_np = self._stats_np()
        self._cache_age += 1
        wts = err_np * err_np
        tok = int(self.rng.choice(self.V, p=wts / wts.sum()))
        hab_np = self.hab.detach().cpu().numpy()
        q_x = 1.0 - float(hab_np[tok])
        qbar = max(1e-6, 1.0 - float(hab_np.mean()))
        demand = (1.0 + max(-1.0, min(1.0, (qbar - q_x) / (qbar + q_x + 1e-12)))) * 0.5
        r_sp = gp['g_syn'] * demand
        if self.rng.rand() < 1.0 - math.exp(-r_sp):
            sgn = 1.0 if float(self.last_e[tok].item()) >= 0.0 else -1.0
            ap_np, av_np = self._activity_np()
            alive_np = self._alive_np
            score = np.where(alive_np, ap_np - av_np, 0.0)
            score[:self.V] = ap_np[:self.V] - av_np[:self.V]
            score[self.V:2 * self.V] = ap_np[self.V:2 * self.V] - av_np[self.V:2 * self.V]
            score[tok] = -1.0
            roll = self.rng.rand()
            if roll < gp['g_dend'] and alive_np.any():
                alive_int = np.nonzero(alive_np)[0]
                wt = ap_np[alive_int] + 1e-6
                tgt_i = int(self.rng.choice(alive_int, p=wt / wt.sum()))
                cand = np.argsort(-score)[:16]
                for c in cand:
                    j = int(c)
                    if score[j] <= 0 or j == tgt_i:
                        continue
                    if (j, tgt_i) in self.scar_syn or (j, tgt_i) in self.syn_keys:
                        continue
                    w = float(self.rng.uniform(gp['birth_rec_lo'], gp['birth_rec_hi'])) * sgn
                    if self._add_syn(j, tgt_i, w):
                        self.G -= gp['spine_cost']
                        self.spines += 1
                        break
            elif roll < gp['g_dend'] + (1.0 - gp['g_dend']) * gp['coinc_frac']:
                alive_ch = np.concatenate([np.arange(self.V), np.arange(self.V, 2 * self.V),
                                           np.nonzero(alive_np)[0]])
                act = ap_np[alive_ch] + 1e-6
                act = act / act.sum()
                for att in range(4):
                    if att < 2:
                        i = int(self.rng.choice(alive_ch, p=act))
                        j = int(self.rng.choice(alive_ch, p=act))
                    else:
                        i = int(self.rng.choice(alive_ch))
                        j = int(self.rng.choice(alive_ch))
                    if i == j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in self.coinc_keys:
                        continue
                    rows = [tok] + [int(self.rng.randint(self.V)) for _ in range(self.fan - 1)]
                    if self._add_coinc(key[0], key[1], rows, gp['coinc_w0'] * sgn):
                        self.G -= gp['spine_cost']
                        self.spines += 1
                        break
            else:
                best = -1
                best_s = 0.0
                prefer_int = self.rng.rand() < 0.5
                for nid in np.nonzero(score > 0)[0]:
                    is_int = bool(alive_np[nid])
                    if prefer_int != is_int:
                        continue
                    if (int(nid), tok) in self.scar_syn or (int(nid), tok) in self.syn_keys:
                        continue
                    if score[nid] > best_s:
                        best_s = score[nid]
                        best = int(nid)
                if best >= 0:
                    if self._add_syn(best, tok, gp['spine_w0'] * sgn):
                        self.G -= gp['spine_cost']
                        self.spines += 1
        n_cost = gp['neuro_cost'] * (1.0 + math.log(2 * self.V) / math.log(100.0))
        r_nb = gp['g_neuro'] * demand
        if self.rng.rand() < 1.0 - math.exp(-r_nb) and self._admit(INT_BYTES + 16.0 * SYN_BYTES):
            nid = self.new_int()
            if nid >= 0:
                lam_i = float(self.rng.uniform(0.10, 0.90))
                self.lam_vec[nid] = lam_i
                gap = float(10.0 ** self.rng.uniform(-2.0, -0.22))
                w_self = max(0.02, (1.0 - gap) - lam_i)
                self._add_syn(nid, nid, w_self)
                order = self.rng.permutation(self.V)
                fan_in = max(1, min(gp['birth_n_in'], max(1, self.V // 6)))
                for t in order[:fan_in]:
                    w = float(self.rng.uniform(gp['birth_in_lo'], gp['birth_in_hi']))
                    if self.rng.rand() < 0.5:
                        w = -w
                    self._add_syn(int(t), nid, w)
                alive_int = np.nonzero(self.int_alive.detach().cpu().numpy())[0]
                alive_rec = np.concatenate([np.arange(self.V), np.arange(self.V, 2 * self.V)])
                pool = np.concatenate([alive_int, alive_rec])
                if len(pool):
                    picks = self.rng.permutation(len(pool))[:gp['birth_rec_in']]
                    for pi in picks:
                        j = int(pool[pi])
                        if j == nid:
                            continue
                        w = float(self.rng.uniform(gp['birth_rec_lo'], gp['birth_rec_hi']))
                        if self.rng.rand() < 0.5:
                            w = -w
                        self._add_syn(j, nid, w)
                top = np.argsort(-err_np)[:2]
                for t in top:
                    w = gp['out_init'] if float(self.last_e[int(t)].item()) >= 0 else -gp['out_init']
                    self._add_syn(nid, int(t), w)
                self.G -= n_cost
                self.births += 1

    def replay(self):
        gp = self.gp
        B = len(self.recent)
        if B == 0:
            return 0
        keep = min(int(r[0].numel()) for r in self.recent)
        Ap = torch.stack([r[0][:keep] for r in self.recent])
        xsn = np.array([r[1] for r in self.recent])
        es = 1.0 / (1.0 + self.clock / gp['eta_age_tau'])
        self._refresh()
        mro = self._mro[self.s_i[self._mro] < keep]
        mc = torch.nonzero(self.c_alive & (self.c_i < keep) & (self.c_j < keep), as_tuple=False).flatten()
        An = Ap / (torch.sqrt(self.var[:keep]) + 0.1)
        bs = gp['replay_batch']
        order = np.arange(B)
        ci = self.c_i[mc] if mc.numel() else None
        cj = self.c_j[mc] if mc.numel() else None
        cr = self.c_rows[mc] if mc.numel() else None
        W = self.c_w[mc].clone() if mc.numel() else None
        sweeps = 0
        for _ in range(gp['replay_sweeps']):
            self.rng.shuffle(order)
            for s0 in range(0, B, bs):
                idx = torch.as_tensor(order[s0:s0 + bs], dtype=torch.long, device=self.d)
                A = Ap[idx]
                X = torch.as_tensor(xsn[order[s0:s0 + bs]], dtype=torch.long, device=self.d)
                L = self.b.unsqueeze(0).repeat(len(idx), 1)
                if mro.numel():
                    vals = A[:, self.s_i[mro]] * self.s_w[mro]
                    L.scatter_add_(1, self.s_j[mro].unsqueeze(0).expand(len(idx), -1), vals)
                if mc.numel():
                    F = A[:, ci] * A[:, cj]
                    for f in range(self.fan):
                        L.scatter_add_(1, cr[:, f].unsqueeze(0).expand(len(idx), -1), F * W[:, f].unsqueeze(0))
                P = torch.softmax(L, dim=1)
                T = torch.zeros((len(idx), self.V), device=self.d, dtype=DT)
                T[torch.arange(len(idx), device=self.d), X] = 1.0
                E = T - P
                Anb = An[idx]
                if mro.numel():
                    d = gp['eta_r'] * es * (Anb[:, self.s_i[mro]] * E[:, self.s_j[mro]]).mean(dim=0)
                    self.s_w[mro] = torch.clamp(self.s_w[mro] + d, -gp['w_clip'], gp['w_clip'])
                if mc.numel():
                    Fn = Anb[:, ci] * Anb[:, cj]
                    for f in range(self.fan):
                        dW = gp['eta_coinc'] * es * (Fn * E[:, cr[:, f]]).mean(dim=0)
                        W[:, f] = torch.clamp(W[:, f] + dW, -gp['w_clip'], gp['w_clip'])
                self.b += gp['eta_b'] * es * E.mean(dim=0)
                sweeps += 1
        if mc.numel():
            self.c_w[mc] = W
        return sweeps

    def run_tokens(self, tokens, learn=False, mask=None, score=True, reset=False):
        if reset:
            self.reset_state()
        nll = 0.0
        n = 0
        hits = 0
        logits = None
        for i, x in enumerate(tokens):
            m = True if mask is None else bool(mask[i])
            if score and logits is not None and m:
                p = torch.softmax(logits, dim=0)
                px = float(p[x].item())
                nll += -math.log(max(px, 1e-12))
                n += 1
                if int(torch.argmax(p).item()) == x:
                    hits += 1
            learn_here = learn and m
            if learn_here and logits is not None:
                self.recent.append((self.ap[:self.hw].clone(), int(x)))
                if len(self.recent) > self.gp['replay_len']:
                    self.recent.pop(0)
            logits = self.step(int(x), logits, learn_here)
            if learn_here:
                self.energy()
                self.growth()
                self._since_replay += 1
                if (self.gp['micro_sleep'] and self.gp['replay_len'] > 0
                        and self._since_replay >= self.gp['replay_len'] and self.recent):
                    self._since_replay = 0
                    self.replay()
        ppl = math.exp(nll / n) if n else float('inf')
        return ppl, n, hits

    def save(self, path):
        def cpu(t):
            return t.detach().cpu().numpy()
        np.savez(path + '.npz',
                 V=self.V, N=self.N, b=cpu(self.b), lam_vec=cpu(self.lam_vec),
                 v=cpu(self.v), u=cpu(self.u), ap=cpu(self.ap), av=cpu(self.av), var=cpu(self.var),
                 atp=cpu(self.atp), u_util=cpu(self.u_util), gamma=self.gamma,
                 int_alive=cpu(self.int_alive), s_i=cpu(self.s_i), s_j=cpu(self.s_j),
                 s_w=cpu(self.s_w), s_alive=cpu(self.s_alive), s_usage=cpu(self.s_usage),
                 s_pre=cpu(self.s_pre), e_s=cpu(self.e_s), n_syn=self.n_syn,
                 c_i=cpu(self.c_i), c_j=cpu(self.c_j), c_rows=cpu(self.c_rows), c_w=cpu(self.c_w),
                 c_usage=cpu(self.c_usage), c_alive=cpu(self.c_alive), e_c=cpu(self.e_c),
                 n_coinc=self.n_coinc, err=cpu(self.err), err_slow=cpu(self.err_slow),
                 hab=cpu(self.hab), last_e=cpu(self.last_e), nov=self.nov, nov_slow=self.nov_slow,
                 G=self.G, clock=self.clock, spines=self.spines, births=self.births,
                 deaths=self.deaths, max_n=self.max_n, hw=self.hw, fan=self.fan)

    def load(self, path):
        z = np.load(path + '.npz' if not path.endswith('.npz') else path, allow_pickle=False)

        def t(name):
            return torch.as_tensor(z[name], device=self.d, dtype=DT)

        self.b[:] = t('b')
        self.lam_vec[:] = t('lam_vec')
        self.v[:] = t('v')
        self.u[:] = t('u')
        self.ap[:] = t('ap')
        self.av[:] = t('av')
        self.var[:] = t('var')
        self.atp[:] = t('atp')
        if 'u_util' in z.files:
            self.u_util[:] = t('u_util')
        if 'gamma' in z.files:
            self.gamma = float(z['gamma'])
        self.int_alive[:] = torch.as_tensor(z['int_alive'], device=self.d)
        self.s_i[:] = torch.as_tensor(z['s_i'], device=self.d)
        self.s_j[:] = torch.as_tensor(z['s_j'], device=self.d)
        self.s_w[:] = t('s_w')
        self.s_alive[:] = torch.as_tensor(z['s_alive'], device=self.d)
        self.s_usage[:] = t('s_usage')
        self.s_pre[:] = t('s_pre')
        self.e_s[:] = t('e_s')
        self.n_syn = int(z['n_syn'])
        self.c_i[:] = torch.as_tensor(z['c_i'], device=self.d)
        self.c_j[:] = torch.as_tensor(z['c_j'], device=self.d)
        self.c_rows[:] = torch.as_tensor(z['c_rows'], device=self.d)
        self.c_w[:] = t('c_w')
        self.c_usage[:] = t('c_usage')
        self.c_alive[:] = torch.as_tensor(z['c_alive'], device=self.d)
        self.e_c[:] = t('e_c')
        self.n_coinc = int(z['n_coinc'])
        self.err[:] = t('err')
        self.err_slow[:] = t('err_slow')
        self.hab[:] = t('hab')
        self.last_e[:] = t('last_e')
        self.nov = float(z['nov'])
        self.nov_slow = float(z['nov_slow'])
        self.G = float(z['G'])
        self.clock = int(z['clock'])
        self.spines = int(z['spines'])
        self.births = int(z['births'])
        self.deaths = int(z['deaths'])
        self.max_n = int(z['max_n'])
        self.hw = int(z['hw']) if 'hw' in z.files else self.max_n
        self.fan = int(z['fan']) if 'fan' in z.files else self.fan
        self._idx_dirty = True
        self.syn_keys = set(zip(z['s_i'][z['s_alive']].tolist(), z['s_j'][z['s_alive']].tolist()))
        self.coinc_keys = set(zip(z['c_i'][z['c_alive']].tolist(), z['c_j'][z['c_alive']].tolist()))
        self.free_slots = [nid for nid in range(self.N - 1, 2 * self.V - 1, -1) if not self.int_alive[nid].item()]
        self.syn_free = [i for i in range(self.n_syn) if not self.s_alive[i].item()]
        self.coinc_free = [i for i in range(self.n_coinc) if not self.c_alive[i].item()]
        self.recent = []
        self._since_replay = 0
        self.prev_tok = None


def evaluate_val(brain, tokenizer, val_records, max_records=None, is_txt=False):
    total_nll = 0.0
    total_n = 0
    hits = 0
    used = 0
    for _ln, line in val_records:
        if max_records is not None and used >= max_records:
            break
        r = base.render_sample(tokenizer, line, is_txt)
        if r is None:
            continue
        ids, mask = r
        if not mask.any():
            continue
        ppl, n, h = brain.run_tokens(ids, learn=False, mask=mask, score=True, reset=True)
        if n > 0:
            total_nll += math.log(ppl) * n
            total_n += n
            hits += h
        used += 1
    if total_n == 0:
        return float('inf'), 0, 0.0
    return math.exp(total_nll / total_n), total_n, hits / total_n


def main():
    args = sys.argv[1:]
    data_path = 'dsr1_v5_0001.jsonl'
    epochs = 10
    tokens_per_epoch = 0
    records = 500
    ckpt = CHECKPOINT
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--train' and i + 1 < len(args):
            data_path = args[i + 1]
            i += 2
        elif a == '--epochs' and i + 1 < len(args):
            epochs = int(args[i + 1])
            i += 2
        elif a == '--tokens' and i + 1 < len(args):
            tokens_per_epoch = int(args[i + 1])
            i += 2
        elif a == '--records' and i + 1 < len(args):
            records = int(args[i + 1])
            i += 2
        elif a == '--ckpt' and i + 1 < len(args):
            ckpt = args[i + 1]
            i += 2
        else:
            i += 1
    is_txt = data_path.lower().endswith('.txt')
    base.DATA_RECORDS_LIMIT = records
    tokenizer = base.FastBPE(VOCAB_SIZE)
    if not tokenizer.load(os.path.join(ROOT, base.TOKENIZER_CHECKPOINT)):
        raise SystemExit('tokenizer missing')
    lines = base.load_lines(os.path.join(ROOT, data_path))
    train_recs, val_recs = base.split_records(lines)
    _log('%s · %s · %s | --train %s epochs=%d tokens/ep=%d records=%d | DEV=%s'
         % (base.SPROUT_PROJECT, SPROUT_BRANCH, SPROUT_VERSION, data_path, epochs, tokens_per_epoch, records, DEV))
    _log('数据集 %s: 共 %d 条 | 训练 %d 条 | 验证 %d 条' % (data_path, len(lines), len(train_recs), len(val_recs)))
    brain = GpuBrain(tokenizer.vocab_size)
    if os.path.exists(ckpt + '.npz'):
        brain.load(ckpt)
        _log('已加载 %s，继续训练' % ckpt)
    for ep in range(epochs):
        t0 = time.time()
        order = np.arange(len(train_recs))
        np.random.RandomState(1000 + ep).shuffle(order)
        seen = 0
        last_log = 0
        while seen < (tokens_per_epoch if tokens_per_epoch > 0 else 10 ** 18):
            stop = False
            for oi in order:
                r = base.render_sample(tokenizer, train_recs[oi][1], is_txt)
                if r is None:
                    continue
                ids, mask = r
                _ppl, n, _acc = brain.run_tokens(ids, learn=True, mask=mask, score=False, reset=True)
                seen += len(ids)
                if seen - last_log >= LOG_EVERY_TOKENS:
                    last_log = seen
                    tppl, tn, tacc = brain.train_stats()
                    _log('[ep %3d] 已学 %6d tok | trainPPL %.3f acc %.2f | N %d+%d | syn %d coinc %d | 出生 %d 死亡 %d | G %.2f | %4.0f tok/s'
                         % (ep + 1, seen, tppl, tacc,
                            2 * tokenizer.vocab_size, int(brain.int_alive.sum().item()),
                            int(brain.s_alive.sum().item()), int(brain.c_alive.sum().item()),
                            brain.births, brain.deaths, brain.G, seen / max(1e-9, time.time() - t0)))
                if tokens_per_epoch > 0 and seen >= tokens_per_epoch:
                    stop = True
                    break
            if stop or tokens_per_epoch <= 0:
                break
        val_ppl, val_n, val_acc = evaluate_val(brain, tokenizer, val_recs, is_txt=is_txt)
        brain.save(ckpt)
        _log('[ep %3d] 完成 | VAL PPL %.3f acc %.3f (n=%d) | N %d+%d | syn %d coinc %d | 出生 %d 死亡 %d | %.1f min'
             % (ep + 1, val_ppl, val_acc, val_n, 2 * tokenizer.vocab_size,
                int(brain.int_alive.sum().item()), int(brain.s_alive.sum().item()),
                int(brain.c_alive.sum().item()), brain.births, brain.deaths,
                (time.time() - t0) / 60.0))


if __name__ == '__main__':
    main()

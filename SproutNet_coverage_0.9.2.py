# ================================================================================================
# SproutNet · coverage 分支 · 0.9.2（从 0.9.0 分叉，只改一处：bias 步长与突触步长对齐）
# 诊断依据（2026-10-25）：
#   12 万 token 训完，bias 的 |b| 中位数只有 0.10（b_clip=9.21 允许到 ±9）。
#   一阶表的 log-odds 根本没学出来，地基是塌的。
#   原因：eta_b=0.02 而 eta_r=0.25。bias 相当于"输入恒为 1 的突触"，
#   其单步更新 Δb = eta_b × e = 0.02e；而 a≈0.30 的突触是 0.25×0.30×e = 0.075e。
#   输入更大（1 > 0.3）、需要表达的范围更大（9.2 > 3），步长却小 3.75 倍。
#   本版把 eta_b 提到 0.25，与 eta_r 同一尺度——消除不一致，不是新增魔数。
# 不改：生长规则、能量账本、通道结构、coverage 机制。
# ================================================================================================
# ================================================================================================
# SproutNet · coverage 分支 · 0.9.0（从 0.3.0 分叉，只改一处：读出权重范围与 b_clip 对齐）
# 负结果记录（同日）：先试过"bigram 边改挂 slow 通道（λ=0.85）"——2 万 token 下
#   PPL 2603.9 vs 基线 2325.0，更差。原因：slow 通道在下一步保留的是 prev 与更早
#   token 的加权混合（0.85/0.72/0.61…），幅度大了但纯度没了，查 bigram 表全是串扰。
#   fast（λ=0.3）虽然只有 30% 幅度，但 2 步前就衰减到 0.09，表示更干净。此路不通。
# 本版改动（唯一一处）：
#   0.2.0 起 b_clip = max(2, lnV) ≈ 9.2，允许 bias 表达 unigram 的完整 log-odds；
#   但突触 w_clip 一直停在 3.0。于是同一张一阶表：bias 能到 ±9.2，边只能到 ±3.0。
#   而 fast 通道 a≈0.30，单条边最大 logit 贡献 = 0.30×3.0 = 0.90（≈2.5× 概率比）；
#   真实中文 bigram 的条件概率比常在 10-100×（需 2.3-4.6 logit）。
#   把 w_clip 提到 9.0，与 b_clip 同一表达范围——不是新魔数，是消除一处不一致。
# 不改：学习率、阈值、生长规则、能量账本、通道结构。
# ================================================================================================
# ================================================================================================
# SproutNet · coverage 分支 · 0.3.0（从 0.2.0 分叉：只做性能等价重构）
# 每步只在 [0,hw) 上运算（hw=2V+已分配内部槽）——空槽不再参与状态/塑性/读出。
# 数学等价（空槽状态恒为 0），速度 ~2-3×；其余机制与 0.2.0 完全一致。
# 主题：共激活生长——0.1.0 把覆盖率从 0.5% 提到 13.4%（突触 6.4k→52k），
#   分桶诊断证明突触质量已超平滑 bigram（有覆盖 PPL 10.4 vs 26.8；无覆盖 3125 vs 13655），
#   缺失的纯粹是覆盖率（13.4% vs bigram 的 47%）。
#   根因：生长用"采样高误差 token"的方式建突触，而不是给刚观测到的真实 bigram (prev→x) 建。
#   0.2.0：共激活即生长——观测到 (prev→x) 且无突触 → 立即建一条（由共激活事件驱动，
#   非采样猜测；容量仍由内存预算约束，满员按使用率排名淘汰）。
# 主题：覆盖——诊断发现 val 只有 10% 的 token 有 bigram 突触（训练 bigram 类型 132,076 vs
#   活着突触 6,367 = 0.5% 覆盖）。根因两处：①每 8 步按 |w|<0.001 绝对阈值杀突触
#   （稀有 bigram 还没等到第二次出现就被杀）；②被杀后 scar_syn 永久禁止再生。
#   0.1.0：删除绝对阈值剪枝与 scar；突触/配对单元改为"内存满员时按使用率排名淘汰"。
#   内存预算沿用 budget 的 SPROUT_MEM_MB 锚。预测：覆盖率上来后 PPL 应接近/超过
#   平滑 bigram（731）。
# 探索主题：把"能活多少神经元"从手填能量账本里解放出来——预算 = 你给的内存。
#
#   批评与修正（来自对 0.2.x 的复盘）：0.2.x 里神经元能不能活由我手填的
#   收支常数（G0/pool_in/neuro_cost/crisis 抽税/ATP 费率）决定——想要大脑袋就得
#   手调这些常数，这等于"规模是参数调出来的"。这在 LLM 原生环境里是错的锚：
#   这台机器没有"食物/液冷/遗传屎山"，它只有【内存】这一条真实的硬约束。
#
#   0.1.0 重设计（因地制宜）：
#     ①预算 = 环境变量 SPROUT_MEM_MB（默认 64MB）→ 按真实字节占用换算各结构容量：
#         内部神经元 ~96B/槽、突触 ~32B/条、配对单元 ~208B/单元；
#     ②货币 = 实测使用率 u_util（use_inc 的慢 EMA，沿用 atp_decay 时标）——
#         零手填尺度、零新常数；
#     ③生长尝试只看需求 demand（hab 驱动）——不再被手填能量 G 节流；
#     ④准入 = 字节级：内存够 → 直接生；内存满 → 淘汰 u_util 排名最低的神经元
#         （排名比较，无量纲无阈值），腾出空间后再生；
#     ⑤删除：ATP 饿死、危机抽税（(1.05−av) 加权）、crisis 常数参与生存。
#         atp/G 仅保留为诊断观测，不参与生死。
#   结果：N* = f(你给的内存, 数据的需求, 实测使用率的相对排名)——调大内存或喂更多
#   真实数据，N* 自动变大；不再需要我改任何"消耗/收入"常数。其余内核继承 economy 0.2.1
#   （其本身继承 elite 0.8.0 的毕业机制）。
#   问题（0.1.0 及以前）：神经元 ATP 收入里用 out_abs=Σ|w|（连接幅度）——幅度大但
#     无用的神经元照样领工资，N* 由固定收入/成本比率决定，而非数据生态位数。
#
#   0.2.0（负结果，如实记录）：用 |Σ_j w_j×e_j|（对齐惊讶）当贡献——量纲不匹配：
#     e 对非观测 token ≈ p ≈ 1e-4，导致贡献比 |w| 小 3 个数量级，收入盖不住维持费，
#     所有神经元出生后 1-2k 步内全部饿死（ints→0）——过度选择 = 死光。教训：
#     贡献统计量的时间稀疏性/量纲必须与每步开支账本匹配。
#
#   0.2.1 修正：改用"读出使用率"当贡献——成功投递才算钱：
#     use_inc[n] = Σ_{n 的读出突触（n→token）} s_usage[突触]
#     其中 s_usage = EMA(前级活跃×后级活跃)（与 0.8.0 毕业的维护规则共用同一统计）。
#     语义：读出端的突触真正"送达"了活跃输出才有使用量——大权重但不投递 = 白拿钱。
#     量纲：使用率与其 EMA 常数天然处在与 ATP 收支同一尺度（10^-3~10^-1），
#     无需任何新常数；util_gain 沿用。
#   配套：本版同时吸收 0.8.0 毕业的读出突触维护块（s_usage/s_pre/w_res 衰减），
#     这样"维护率"与"工资"用的就是同一个成功投递统计——无用神经元被双杀：
#     工资归零（饿死）+ 读出突触衰减（失去立足点）。
# 0.1.0 探索主题：把"数量均衡依赖隐藏魔数"从根上拆掉（第一步：单位归一化）。
#   问题（0.6.0 及以前）：
#     ①生长率的半饱和常数 0.05 是隐藏的"能量单位魔数"（G/(0.05+G)）；
#     ②出生有硬门槛 G > n_cost（阈值，违背红线）；
#     ③初始预算 G0=5.0 与其它常数无推导关系。
#   0.1.0 修改（全部无量纲化，无新常数）：
#     ①r_sp = g_syn × G/(spine_cost+G) × demand —— 半饱和点=一个棘突的价格（天然单位）；
#        r_nb = g_neuro × G/(n_cost+G) × demand —— 半饱和点=一个神经元的出生价；
#     ②删除 G>n_cost 硬门槛——平滑速率已内在地处理（G≪价格时速率≈0）；
#     ③G0 = n_cost（初始预算 = 恰好一个神经元出生价，其余靠每步收入）。
#   效果：绝对能量单位不再影响动力学形状；数量均衡只由"比率"（成本/收入）与数据决定。
#   下一步（0.2.0）：贡献收入——每个神经元/突触的存活由其对预测的贡献支付（N* 由数据结构决定）。
# 0.6.0 任务特化实验支持（不改训练数学）：
#   ①.txt 语料按"空行分段"成块（保留代码/文本的多行结构），整段预测（mask 全 1）。
#   ②--ckpt 覆盖检查点路径：同一分支可并行维护多个脑（问答脑 / 代码脑互不污染）。
#   命名约定：SproutNet_<分支>_<版本>.py；每条分支独立 CHECKPOINT/LOG_FILE，可并行跑。
# 0.5.1 拓扑指标扩展（只增观测，不改训练数学）：
#   --topology 新增：入/出度分位数、基尼系数、重尾斜率 α、内部图（int→int）聚类与
#   随机重连基线（小世界性）、互环/自环、读出耦合投影聚类（共享内部源的 token 行是否成簇）。
#   并修正突触分类标签（区分 感受器→token / 内部→token / 树突 / 自环）。
# 0.5.0 内部神经元唤醒（修复 0.4.1 拓扑快照暴露的"树突饥饿"）：
#   现象：96 个内部神经元中 75 个树突入度为 0——出生接线的树突是"均匀随机"选受体，
#         多为低频 token，永不与下游共激活 → 旧"未使用衰减"（usage<0.05 每步 ×0.99）
#         在 ~700 步内把权重磨到 0 剪掉 → 神经元哑火，只剩空转的输出突触。
#   修复①自发驱动（生物直抄）：内部神经元每步获得与激活尺度无关的基线驱动，
#         稳态激活 = c_basal = 1% c_in（生物：自发发放 ~1 Hz vs 最大 ~100 Hz）。
#         意义：神经元有基线活动 → 共激活统计不再归零 → 可被学习塑形。
#   修复②树突早死修复：未使用衰减改为"先被激活过足够多次（s_pre ≥ w_res）才判定无用"
#         （新增 s_pre = 每突触的前级活动 EMA，复用 w_res 尺度，无新魔数）。
#         意义：随机的罕见输入不会被冤枉剪掉；真正"前级常来但不共激活"的错误接线才剪。
# 0.4.1 观测与审计（不改训练数学）：
#   ①--topology：生长拓扑快照——内部神经元出/入度分布与枢纽、lam 时间尺度分带、
#     突触/配对对 token 行的覆盖率（用于与生物连接组统计对照，支撑"任务特化结构"验证）。
#   ②docs/parameters.md：SPROUT_GP 参数来源审计（生物直抄 / 时间尺度与单位推导 / 经验值
#     + 逐项推导计划），兑现"不要魔数"路线图条目。
# 0.4.0 自适应生长（让"生长多少"由动力学涌现，而非手调常数）：
#   ①n_cost 去掉 ints/20 线性加价——出生价格恒定为 neuro_cost×(1+ln(2V)/ln100)；
#     数量的自限改由"维持成本 × 数量 vs 能量收入"的均衡涌现（贵在养，不在生）。
#   ②生长配额按"相对学习亏欠"分配：hab[x]=EMA(1-p[x]) 记录每个目标 token 的持续惊讶；
#     q=1-hab 是每个 token 的当前预测质量。demand=(1+clip((q̄-q)/(q̄+q)))/2（单位归一化映射）：
#     学得比平均好 → 0（停止生长）；落后越多 → 越接近 1（配额流向学习落后的目标）。
#     全新鲜脑时 q≈q̄ → demand=0.5（自然起步，不饿死也不爆长）。
#   ③cap_int 96→1024：仅是数组内存上界，不是生长规则；实际数量由能量均衡自限。
#   ④CPU-only 协议：语料缩到 500 条。
# 0.3.0 学习机制升级（无新增魔数）：
#   ①资格迹（eligibility trace）三因子规则的迹形式：E ← λ·E + (1-λ)·pre·post，
#     w += η·E·调制信号。λ = lam_slow（沿用既有慢记忆时间尺度，不是新魔数；1-λ 为单位
#     归一化 EMA——单次共现的总冲量与瞬时规则相同，但更新被时间平滑、跨多步生效）。
#     资格迹为瞬态状态，不写入检查点。重放（replay）仍走瞬时批量规则（不污染迹）。
#   ②语料条数上限 DATA_RECORDS_LIMIT（默认 1000，用户指定；--records N 可调）。
# 0.2.2 修复（实锤 bug，解释"PPL 一直在 2000 附近"的一大半原因）：np.savez 会给不带 .npz
#   后缀的路径静默追加 .npz —— 存到 brain.pt.npz，而载入检查的是 brain.pt → 检查点从未被
#   载入过，**每次训练运行都是从零开始**（此前所有数字实为"重学第 1~2 遍"的成绩）。
#   修复：_ckpt_file() 统一规范化存/取文件名；存→取闭环已用两次连续运行验证。
# 0.2.1 修复：--epochs 命令行参数之前不生效（train_sprout 的形参默认值在定义时就绑定了
#   模块初始 EPOCHS=40，_cli 里 global EPOCHS 赋值改不动已绑定的默认值）——显式传参。
# 0.2.0 学习能力修复（对照基线实测后的定向改动；同协议实测 unigram PPL=1780/acc=0.050，
#   bigram PPL=1152/acc=0.158，而 0.1.x 模型 PPL=3391/acc=0.04 —— 连 unigram 都没到）：
#   ①bias 表达范围 b_clip=max(2, lnV)：原 ±2 截断在 V=10000 下无法表达 unigram 分布；
#   ②关联容量 cap_syn 3072→16384：突触是 bigram/上下文关联的唯一通路，容量=覆盖率；
#   ③时间尺度 lam_slow 0.85→0.95、内部神经元 lam~U(0.10,0.90)：上下文窗口 → 20+ 步；
#   ④学习率平衡 eta_r 0.08→0.25 / eta_coinc 0.22→0.35 / eta_b 0.05→0.02、w_clip 3.0；
#   ⑤配对出生 2/4 定向（两个活跃上下文通道 × 误差目标行）+ 2/4 均匀（覆盖探索），
#      coinc_frac 0.5→0.35（把生长配额更多让给单通道关联=bigram 通路）。
#   注：lam_slow/内部 lam 范围变更后，延迟回忆探针需复验（后续步骤）。
# 0.1.2 修复：①神经元出生成本 n_cost = neuro_cost×(1 + ints/20 + ln(2V)/ln(100))（0.1.1 的
#           log2 公式在 V=10000 仍为 20 > G≈17，出生被堵死——本版约 3.15，G 可负担）；
#           ②新增 m_int_upkeep：内部神经元 ATP 维护开销，不活跃者自然凋亡（凋亡红线复活）；
#           ③热点路径 np.add.at 合并/向量化（重合项 logits/blame/更新），降低随 coinc 增长的速度衰减；
#           ④突触/重合死亡槽位回收（syn_free/coinc_free），修复长跑容量池被死槽耗尽的问题。
# 0.1.1 修复：能量预算回归物质约束——收入端 max(1,V/25) 的 V 线性放大因子改为有界亚线性 sf(V)；
#           受体支出同因子（防止只砍收入的赤字死亡螺旋）；新增代谢泄漏 g_leak（多余能量的持续
#           耗散 = 自然软上限，非阈值）；神经元出生成本由 (2V+ints)/20 改为 log 亚线性。
# 单文件项目：外壳沿用 3120012elite.py 体系（菜单/日志/v5 schema/FastBPE/题库/拒绝采样/P4 切分），
#             内核替换为 SproutNet 生长脑（能量预算 + 局部规则 + 速率化生长 + 重合突触 + 睡眠重放）。
# 内核红线（不可退让）：无反向传播 / 无两次前向对比 / 无外部监督注入 / 无工程外挂 / 零阈值。
# 版本约定：文件名 = 项目名_分支_x.y.z.py；每次改动保留旧文件（复制→改→改名），靠文件保存历史。
# 内核来源：evo_grow/v6_torch/brain.py（v6.3 规则）。运行期依赖：numpy + tokenizers（无 torch）。
# GPU 说明：内核为 CPU/numpy 在线流式；GPU 加速的正确姿势是「批量并行多序列 + 重放矩阵运算」，
#           不改变架构与规则（设备无关），将作为后续 torch 端口分支实现。
# ================================================================================================

SPROUT_PROJECT = 'SproutNet'
SPROUT_BRANCH = 'coverage'
SPROUT_VERSION = '0.9.2'

# ================================================================================================
# 【顶部统一控制区】—— 所有超参数与开关集中于此
# ================================================================================================

# ---- A. 数据与 v5 schema ----
NORMALIZE_MODE = 'none'          # 'none'=原样 | 'nfkc'=只做 NFKC（以词表元数据为准）
BPE_MIN_FREQUENCY = 2            # BPE 最小词频（调大=词表更小更干净）
TRAIN_ON_REASONING = True        # assistant 的 reasoning_content 是否纳入训练
VAL_BUCKET = 10                  # 验证集抽法：内容哈希 % VAL_BUCKET == 0 → 验证集（10 = 10%）
                                 # 防泄露：训练集自动剔除与验证集内容一致的记录（内容指纹去重）
MAX_SEQ_LEN = 2048               # 单样本 token 上限（超出从结尾往回保留 user/assistant 对）
TRAIN_TOKENS_PER_EPOCH = 0       # 每 epoch 训练 token 预算，0 = 不限
RESET_STATE_PER_RECORD = True    # 每条记录开始前清膜电位/适应（权重与结构保留，防跨记录串扰）
LOG_EVERY_TOKENS = 2000          # 训练日志粒度（token）

# ---- B. 分词器 ----
VOCAB_SIZE = 10000               # BPE 词表大小（内核每步开销随 V 线性增长，可酌情调小）
TOKENIZER_CHECKPOINT = 'tokenizer_incremental.bin'

# ---- C. 内核超参（SproutNet 生长脑）----
import os as _os
# budget 0.1.0：预算不再是我手填的能量收支，而是【你给它的内存】。
#   SPROUT_MEM_MB = 环境变量：直接给大脑一份真实内存预算（默认 64MB，测试规模）。
#   各结构容量由真实字节占用换算（不拍脑袋）：内部神经元 ~96B/槽、突触 ~32B/条、
#   配对单元 ~208B/单元（fan=6）——按三等分折算槽位，运行期再做字节级准入。
_MEM_MB = float(_os.environ.get('SPROUT_MEM_MB', '64'))
_MEM_BYTES = _MEM_MB * 1024.0 * 1024.0
_INT_BYTES = 96.0                # v,u,ap,av,atp,lam,u_util,nov?等 N 数组的每槽摊分
_SYN_BYTES = 32.0                # s_i,s_j,s_w,s_usage
_COINC_BYTES = 208.0             # c_i,c_j + c_rows/c_w/c_av/c_var（fan=6 → 16+48*4）
CAP_INT = int(_MEM_BYTES * 0.34 / _INT_BYTES)
CAP_SYN = int(_MEM_BYTES * 0.33 / _SYN_BYTES)
CAP_COINC = int(_MEM_BYTES * 0.33 / _COINC_BYTES)
_BEST_PPL = float('inf')
FREEZE_CAP = _os.environ.get('SPROUT_FREEZE_CAP', '0') == '1'   # A/B：满员冻结结构 vs 淘汰
SPROUT_GP = dict(
    # 动力学
    lam=0.35, lam_slow=0.95, c_in=1.0, sigma=0.02,
    c_basal=0.01,                # 内部神经元自发驱动：稳态激活=1% c_in（生物自发发放率 ~1% 最大）
    # 局部学习率（Hebbian/LMS 近似，无梯度回传）
    eta_r=0.25, eta_coinc=0.35, eta_i=0.02, eta_b=0.25,
    # 惊讶与奖励预测误差（局部信号）
    e_ema=0.02, nov_ema=0.02, nov_slow=0.002, rpe_gain=4.0, blame_gain=4.0,
    # 能量预算（ATP 收入/支出，生长与生存的物质约束；单位：每步收入为基准，G0 由 n_cost 推导）
    pool_in=0.060, stress_gain=2.0, g_leak=0.02,
    g_adapt=0.60, tau_adapt=0.05,
    m_neuron=0.0010, m_receptor=0.0001, m_spike=0.0006, m_syn=0.00002, m_syn_w=0.0001,
    m_plast=0.002, atp_decay=0.997, use_gain=0.01, util_gain=0.10, int_basal=0.001,
    m_int_upkeep=0.0012, receptor_supply=0.004,
    # 生长（速率化：P(event)=1-exp(-rate)，无阈值；预算越足越可能长）
    g_syn=0.15, g_neuro=0.001, g_dend=0.35,
    spine_cost=0.10, spine_w0=0.20, coinc_frac=0.35, coinc_w0=0.15,
    neuro_cost=1.0, eta_age_tau=200000.0,
    birth_in_lo=0.5, birth_in_hi=1.2, birth_n_in=5, birth_rec_in=6,
    birth_rec_lo=0.15, birth_rec_hi=0.5,
    self_loop_lo=0.45, self_loop_hi=0.65, out_init=0.05,
    # 修剪与稳态（突触缩放、用率门控凋亡）
    w_res=0.001, wd=0.010, w_clip=9.0, crisis=3.0, dream_noise=0.15,
    # 睡眠重放（微睡眠间隔=缓冲容量；批次化序列重放）
    replay_len=1024, replay_sweeps=1, replay_batch=32, micro_sleep=True,
    cap_syn=CAP_SYN, cap_coinc=CAP_COINC, cap_int=CAP_INT,
)

# ---- D. 训练与巩固 ----
EPOCHS = 40                      # 菜单 1 / --train 默认训练轮数
EPOCH_END_REPLAY = True          # 每 epoch 结束做一次额外巩固
CHECKPOINT = 'brain_cov92.pt'  # 脑检查点（npz 格式；分支独立命名）
LOG_FILE = 'training_log_coverage92.txt'
DATA_RECORDS_LIMIT = 500         # 语料条数上限（0=全量；CPU-only 协议本轮取 500 条）


def _ckpt_file(p):
    # np.savez 会给没有 .npz 后缀的路径静默追加 .npz，导致存/取文件名不一致（0.2.2 前每次运行都从零开始）
    return p if p.lower().endswith('.npz') else p + '.npz'

# ---- E. 生成 ----
TEMPERATURE = 0.7                # 采样温度（1.0=原分布）
TOP_P = 0.9                      # nucleus 采样
GEN_MAX_TOKENS = 256             # 单次生成上限
CHAT_MAX_HISTORY_TOKENS = 2048   # 对话历史 token 上限

# ---- F. 规则奖励（拒绝采样 / 试错巩固 / 组内相对重放）----
RL_PROMPT_SUFFIX = "\n\n请自由推理。最后必须单独写一行：最终答案：你的答案"
RL_SYSTEM_PROMPT = ''
RS_SAMPLES_PER_Q = 8             # 拒绝采样：每题采样次数
RL_SAMPLES_PER_Q = 8             # 试错巩固：每题尝试次数
GRPO_GROUP = 8                   # 组内相对重放：每组采样数
REWARD_REPLAY_BONUS = 2          # 奖励事件额外巩固遍数
P4_SEG_MAX_TOKENS = 2048         # P4 切分阈值

# ================================================================================================
# 依赖导入
# ================================================================================================
import json
import math
import os
import re
import sys
import time
import random
import hashlib
import unicodedata
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Union

import numpy as np

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    HAS_TOKENIZERS = True
except ImportError:
    raise ImportError("请安装 tokenizers 库: pip install tokenizers")

random.seed(42)
np.random.seed(42)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

_VOCAB_NORMALIZE_MODE = None

_LOG_FILE = None

def init_log_file():
    global _LOG_FILE
    if _LOG_FILE is None:
        _LOG_FILE = open(LOG_FILE, "a", encoding='utf-8')
        _LOG_FILE.write(f"\n--- 新会话开始于 {datetime.now()} ---\n")
        _LOG_FILE.flush()

_LOG_FLUSH_COUNTER = 0
_LOG_FLUSH_INTERVAL = 50

def log_and_print(*args, **kwargs):
    import sys
    global _LOG_FLUSH_COUNTER
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    msg = sep.join(str(arg) for arg in args) + end
    sys.stdout.write(msg)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg)
        _LOG_FLUSH_COUNTER += 1
        if _LOG_FLUSH_COUNTER >= _LOG_FLUSH_INTERVAL:
            _LOG_FILE.flush()
            _LOG_FLUSH_COUNTER = 0

def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None

def _active_normalize_mode() -> str:
    """归一化模式一律以词表元数据为准；v5 不保留 NFKC+lower 历史模式。"""
    return _VOCAB_NORMALIZE_MODE if _VOCAB_NORMALIZE_MODE is not None else NORMALIZE_MODE


def normalize_text(text: str) -> str:
    if _active_normalize_mode() == 'nfkc':
        return unicodedata.normalize('NFKC', text)
    return text

# ============================================================
# v5 数据格式层（messages / reasoning_content / system / tools）
# ------------------------------------------------------------
# 与旧 conversations 格式逐 token 等价：没有 system/tools/tool_calls 时，
#   render_messages() 渲染出的文本与旧实现拼接的字符串完全一致（已全语料对拍）。
# 不新增任何特殊 token：工具调用用明文 JSON 协议承载（见 _render_tool_calls）。
# ============================================================

def _messages_from_record(rec):
    """取出一行数据的 messages（v5 唯一格式；旧 conversations 不再兼容，遇到就报错）。"""
    if isinstance(rec, list):
        msgs = rec
    else:
        msgs = rec.get('messages')
        if msgs is None:
            raise ValueError("数据缺少 messages 字段：v5 不保留旧 conversations 兼容，"
                             "请先用 _to_v5.py 把语料/题库转成 {'messages': [...]}")
    for _m in msgs:
        _r = _m.get('role')
        if _r not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError(f"非法 role {_r!r}：只允许 system/user/assistant/tool")
        if _m.get('content') is None and not _m.get('tool_calls'):
            raise ValueError("消息既没有 content 也没有 tool_calls")
    return [dict(m) for m in msgs]


def _render_tools(tools):
    """工具定义 → 明文段（不新增 token）。"""
    if not tools:
        return ''
    return '<|user|>[工具]' + json.dumps(tools, ensure_ascii=False) + '<|end|>'


def _render_tool_calls(tcs):
    """assistant 的工具调用 → 明文 JSON（与 content 同段，一起算 loss）。"""
    if not tcs:
        return ''
    return json.dumps(tcs, ensure_ascii=False)


def render_messages(messages, tools=None, add_generation_prompt=False):
    """渲染 messages → (text, spans)。

    spans = [(start, end), ...]，是需要计算 loss 的字符区间（assistant 段）。
    规则与旧实现一致：assistant 的 reasoning_content(可选)/content/tool_calls + 回合结束符 + <|eos|> 参与训练；
    system / user / tool 段不训练。每段单独 normalize_text，保证 spans 与最终文本严格对齐。
    """
    parts = []
    spans = []
    pos = 0

    def emit(s, train=False):
        nonlocal pos
        n = normalize_text(s)
        if train:
            spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n)

    emit('<|bos|>')
    if tools:
        emit(_render_tools(tools))
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        content = '' if content is None else str(content)
        if role == 'system':
            emit('<|user|>[系统]' + content + '<|end|>')
        elif role == 'user':
            emit('<|user|>' + content + '<|end|>')
        elif role == 'tool':
            tid = msg.get('tool_call_id')
            emit('<|user|>[工具结果' + ((' id=' + str(tid)) if tid else '') + ']' + content + '<|end|>')
        elif role == 'assistant':
            emit('<|assistant|>')
            seg = []
            rc = msg.get('reasoning_content')
            if TRAIN_ON_REASONING and rc:
                seg.append(str(rc))
            seg.append(content)
            tc = _render_tool_calls(msg.get('tool_calls'))
            if tc:
                seg.append(tc)
            emit(''.join(seg) + '<|end|>', train=True)
    if add_generation_prompt:
        emit('<|assistant|>')       # 生成提示：不给 <|eos|>（旧实现同样没有）
    else:
        emit('<|eos|>', train=True)  # 文档模式：补 <|eos|> 并纳入训练（与旧实现逐 token 一致）
    return ''.join(parts), spans


def _encode_with_offsets(tokenizer, text: str):
    """编码并返回字符级 offsets。text 必须已是"按段归一化后"的最终字符串。"""
    if tokenizer._tokenizer is None:
        raise RuntimeError("Tokenizer 未训练或未加载")
    enc = tokenizer._tokenizer.encode(text)
    ids = [tid if tid < tokenizer._vocab_size_cache else tokenizer.unk_token_id for tid in enc.ids]
    return ids, list(enc.offsets)


def _encode_batch_with_offsets(tokenizer, texts):
    """批量编码并返回 [(ids, offsets), ...]；与逐条 _encode_with_offsets 逐 token 等价。

    仅把「一次 encode 一条」换成「一次 encode_batch 一批」，用于加速数据加载。
    encode_batch 对同一文本返回的 Encoding 与 encode 相同（ids/offsets 一致）。
    """
    if tokenizer._tokenizer is None:
        raise RuntimeError("Tokenizer 未训练或未加载")
    encs = tokenizer._tokenizer.encode_batch(list(texts))
    vcap = tokenizer._vocab_size_cache
    unk = tokenizer.unk_token_id
    out = []
    for enc in encs:
        out.append(([tid if tid < vcap else unk for tid in enc.ids], list(enc.offsets)))
    return out


def _split_reasoning(text: str):
    """拆出 thinking 段与正文。

    ⚠️ 当前 v5 的 render_messages **不注入任何 thinking 标签**（reasoning_content 直接拼 content），
    因此模型生成的 response 通常不含标签，本函数走"整段算正文" 的 no-op 分支。
    这是**预期行为**——本函数为未来扩展预留，不是 active 逻辑。

    支持两种标签对（依次尝试，用 re.search 允许标签出现在中间）：
       - think ... （开标签字面 think，闭标签字面 /think，均用尖括号包起来）
       - thinking ... （开标签字面 thinking，闭标签字面 /thinking，均用尖括号包起来）

    无标签 → 返回 ('', text.strip())：整段算正文，剥首尾空白。
    有标签 → 返回 (reasoning.strip(), 剩余文本.strip())。
    """
    if not text:
        return '', ''
    for _open, _close in (
        ('<' + 'think' + '>', '<' + '/think' + '>'),
        ('<' + 'thinking' + '>', '<' + '/thinking' + '>'),
    ):
        m = re.search(
            re.escape(_open) + r'(.*?)' + re.escape(_close),
            text, flags=re.S,
        )
        if m:
            reasoning = m.group(1).strip()
            body = (text[:m.start()] + text[m.end():]).strip()
            return reasoning, body
    return '', text.strip()


def _truncate_pairs(messages, tokenizer, max_seq_len):
    """从结尾往回取相邻 (user, assistant) 对（含 id_cache 优化）；开头的 system 段始终保留。

    旧语义只认 (user, assistant) 对，会把 system 整段丢掉 —— 于是出现"render_messages 支持
    system、训练数据里却永远看不到 system"的不一致（tools 却因为单独传参而保留）。
    这里先把开头连续的 system 段原样摘出，并把其 token 数计入预算。
    """
    id_cache = {}
    head = []
    head_tokens = 0
    start = 0
    while start < len(messages) and messages[start].get('role') == 'system':
        m = messages[start]
        c = str(m.get('content') or '')
        ids = id_cache.get(c)
        if ids is None:
            ids = tokenizer.encode(c)
            id_cache[c] = ids
        head.append(m)
        head_tokens += len(ids) + 2
        start += 1
    kept = []
    count = head_tokens
    i = len(messages) - 1
    while i >= start + 1:
        if messages[i].get('role') == 'assistant' and messages[i - 1].get('role') == 'user':
            pair = [messages[i - 1], messages[i]]
            pair_tokens = 0
            for turn in pair:
                c = str(turn.get('content') or '')
                ids = id_cache.get(c)
                if ids is None:
                    ids = tokenizer.encode(c)
                    id_cache[c] = ids
                pair_tokens += len(ids) + 2
                if TRAIN_ON_REASONING and turn.get('role') == 'assistant':
                    rc = turn.get('reasoning_content')
                    if rc:
                        pair_tokens += len(tokenizer.encode(str(rc)))
                if turn.get('role') == 'assistant':
                    tc = _render_tool_calls(turn.get('tool_calls'))
                    if tc:
                        pair_tokens += len(tokenizer.encode(tc))
            kept.insert(0, pair)
            count += pair_tokens
            if count + 2 >= max_seq_len:
                break
            i -= 2
        else:
            i -= 1
    return head + [t for p in kept for t in p]


def _has_assistant_target(messages):
    """截断后至少要有 1 条 assistant 消息，才有可训练目标。

    否则 render_messages 的唯一训练目标只剩结尾的 <|eos|>，模型会学到"user 之后立刻结束序列"，
    污染生成行为。此类残缺样本（如只有 user、或只有 system+user）应直接跳过。
    """
    return any(m.get('role') == 'assistant' for m in messages)


class StreamDecoder:
    """流式反解码（tokenizers 0.23.1 没有 DecodeStream）。

    安全前缀法：每次 push 后重新 decode 整个缓冲，只输出"已稳定"的新增部分；
    末尾若是 U+FFFD（字节序列未完成）则不输出，等下一个 token 补齐。
    逐 token 直接 decode 在 byte-level 词表下会碎字，必须用这个。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buf = []
        self.emitted = ''

    def _stable(self):
        cur = self.tokenizer.decode(self.buf)
        if cur.endswith('\ufffd'):
            cur = cur[:-1]
        return cur

    def push(self, token_id):
        self.buf.append(token_id)
        cur = self._stable()
        if not cur.startswith(self.emitted):
            return ''
        out = cur[len(self.emitted):]
        self.emitted = cur
        return out

    def flush(self):
        cur = self.tokenizer.decode(self.buf)
        out = cur[len(self.emitted):] if cur.startswith(self.emitted) else ''
        self.emitted = cur
        return out

# ------------------------------------------------------------
# 高性能 BPE 分词器
# ------------------------------------------------------------
SPECIAL_TOKENS_LIST = ['<|bos|>', '<|eos|>', '<|pad|>', '<|unk|>', '<|user|>', '<|assistant|>', '<|end|>']

class FastBPE:
    def __init__(self, initial_vocab_size: int = VOCAB_SIZE):
        self.initial_vocab_size = initial_vocab_size
        self.special_tokens_list = SPECIAL_TOKENS_LIST.copy()
        self._tokenizer = None
        self.pad_token_id = None
        self.unk_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None
        self.user_token_id = None
        self.assistant_token_id = None
        self.end_token_id = None
        self._vocab_size_cache = None
        # v5 只有一种风格（ByteLevel + 不做 NFKC/lower）；换词表一律重新 train()，不兼容旧词表文件
        self._pre_style = 'byte_level'
        self._decoder_style = 'byte_level'
        self._normalize_mode = NORMALIZE_MODE

    def train(self, texts: List[str], verbose: bool = False):
        """训练 BPE 词表（v5 只有一种风格：ByteLevel）。

        ByteLevel 预分词 + ByteLevel 解码：保空格/换行/大小写/全角标点；
        initial_alphabet 把 256 个字节全部写进词表 -> 结构上不可能出现 <|unk|>（不再吞字）。
        """
        global _VOCAB_NORMALIZE_MODE
        tokenizer = Tokenizer(BPE(unk_token='<|unk|>'))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        trainer = BpeTrainer(
            vocab_size=self.initial_vocab_size,
            special_tokens=self.special_tokens_list,
            min_frequency=BPE_MIN_FREQUENCY,
            show_progress=verbose,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(list(texts), trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()
        self._tokenizer = tokenizer
        self._pre_style = 'byte_level'
        self._decoder_style = 'byte_level'
        self._normalize_mode = NORMALIZE_MODE
        _VOCAB_NORMALIZE_MODE = self._normalize_mode
        self._update_special_ids()

    def _update_special_ids(self):
        if self._tokenizer is None:
            return
        vocab = self._tokenizer.get_vocab()
        self._vocab_size_cache = self._tokenizer.get_vocab_size()
        self.bos_token_id = vocab.get('<|bos|>', 0)
        self.eos_token_id = vocab.get('<|eos|>', 1)
        self.pad_token_id = vocab.get('<|pad|>', 2)
        self.unk_token_id = vocab.get('<|unk|>', 3)
        self.user_token_id = vocab.get('<|user|>', 4)
        self.assistant_token_id = vocab.get('<|assistant|>', 5)
        self.end_token_id = vocab.get('<|end|>', 6)

    def encode(self, text: str) -> List[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        norm_text = normalize_text(text)
        ids = self._tokenizer.encode(norm_text).ids
        return [tid if tid < self._vocab_size_cache else self.unk_token_id for tid in ids]

    def decode(self, ids: List[int]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer 未训练或未加载")
        # 修复 Bug 5：仅使用 decode 的 skip_special_tokens，移除手动过滤及 replace
        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        return text

    def save(self, path: str):
        if self._tokenizer is None:
            log_and_print("警告：tokenizer未初始化，无法保存")
            raise RuntimeError("Tokenizer 未初始化，保存失败")
        try:
            dir_name, file_name = os.path.split(path)
            file_base = os.path.splitext(file_name)[0]
            tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
            self._tokenizer.save(tokenizer_path)
            # 新增 3 个键（老键原样保留）；旧代码读这份文件时取不到新键 -> 自动退回历史路径
            state = {
                'initial_vocab_size': self.initial_vocab_size,
                'special_tokens_list': self.special_tokens_list,
                'normalize_mode': self._normalize_mode,
                'pre_tokenizer': self._pre_style,
                'decoder': self._decoder_style,
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception as e:
            # 修复 Bug 7：抛出异常，不再静默
            raise RuntimeError(f"保存 tokenizer 失败: {e}") from e

    def load(self, path: str):
        global _VOCAB_NORMALIZE_MODE
        if not os.path.exists(path):
            log_and_print(f"警告: 文件 {path} 不存在，无法加载 tokenizer")
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.initial_vocab_size = data.get('initial_vocab_size', VOCAB_SIZE)
            self.special_tokens_list = data.get('special_tokens_list', SPECIAL_TOKENS_LIST.copy())
        except Exception as e:
            log_and_print(f"加载 tokenizer 元数据失败 ({path}): {e}")
            return False
        # v5 不保留旧词表兼容：缺元数据直接报错（避免拿错词表/旧路径训练）
        _missing = [k for k in ('pre_tokenizer', 'decoder', 'normalize_mode') if k not in data]
        if _missing:
            raise RuntimeError(f"词表元数据缺少 {_missing}：v5 不保留旧词表兼容，请用 FastBPE.train() 重新训练词表")
        self._pre_style = data['pre_tokenizer']
        self._decoder_style = data['decoder']
        self._normalize_mode = data['normalize_mode']

        dir_name, file_name = os.path.split(path)
        file_base = os.path.splitext(file_name)[0]
        tokenizer_path = os.path.join(dir_name, f"{file_base}_tokenizer.json")
        if os.path.exists(tokenizer_path):
            try:
                self._tokenizer = Tokenizer.from_file(tokenizer_path)
                # v5 只有一种风格
                self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
                self._tokenizer.decoder = decoders.ByteLevel()
                _VOCAB_NORMALIZE_MODE = self._normalize_mode
                self._update_special_ids()
                log_and_print(f"成功从 {tokenizer_path} 加载 tokenizer（style={self._pre_style}/{self._decoder_style}, normalize={self._normalize_mode}）")
                return True
            except Exception as e:
                log_and_print(f"加载 tokenizer.json 失败: {e}")
                return False
        else:
            log_and_print(f"未找到 {tokenizer_path}，无法加载 tokenizer")
            return False

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return len(self.special_tokens_list)
        if self._vocab_size_cache is None:
            self._vocab_size_cache = self._tokenizer.get_vocab_size()
        return self._vocab_size_cache

# ================================================================================================
# SproutNet 内核：能量预算驱动的自生长神经网
# 局部规则 / 无 BP / 零阈值 / 速率化事件 / 生长即学习 / 无用者凋亡
# 移植自 evo_grow/v6_torch/brain.py（v6.3），新增：逐 token 掩码学习、按记录复位、
# 温度/top-p 采样、npz 检查点、统计接口。架构规则与旧版逐项一致。
# ================================================================================================
class SproutBrain:
    def __init__(self, V, gp=None, seed=1):
        self.gp = dict(SPROUT_GP)
        if _os.environ.get('SPROUT_NO_GROWTH', '0') == '1':
            self.gp['g_syn'] = 0.0
            self.gp['coinc_frac'] = 0.0   # A/B：冻结结构，只调权重
        if gp:
            self.gp.update(gp)
        gp = self.gp
        self.V = V
        self.rng = np.random.RandomState(seed)
        self.N = 2 * V + gp['cap_int']
        self.fast = np.arange(V)
        self.slow = np.arange(V) + V
        self.lam_vec = np.empty(self.N)
        self.lam_vec[self.fast] = gp['lam']
        self.lam_vec[self.slow] = gp['lam_slow']
        self.lam_vec[2 * V:] = gp['lam']
        self.v = np.zeros(self.N)
        self.u = np.zeros(self.N)
        self.ap = np.zeros(self.N)
        self.av = np.zeros(self.N)
        self.var = np.zeros(self.N)
        self.atp = np.ones(self.N)
        self.u_util = np.zeros(self.N)     # budget 0.1.0：实测使用率 EMA（淘汰排名的唯一依据）
        self.int_alive = np.zeros(self.N, dtype=bool)
        self.free_slots = list(range(2 * V, self.N))      # 低位优先：hw 紧凑，replay 条目才真的小
        self.b = np.zeros(V)
        self.b_clip = max(2.0, math.log(V))   # bias 表达范围随词表规模自然扩展（unigram 需 ±lnV）
        self.syn_cap = gp['cap_syn']
        self.s_i = np.zeros(self.syn_cap, dtype=np.int32)
        self.s_j = np.zeros(self.syn_cap, dtype=np.int32)
        self.s_w = np.zeros(self.syn_cap)
        self.s_alive = np.zeros(self.syn_cap, dtype=bool)
        self.s_usage = np.zeros(self.syn_cap)
        self.s_pre = np.zeros(self.syn_cap)
        self.n_syn = 0
        self.syn_free = []
        self.coinc_cap = gp['cap_coinc']
        self.fan = max(1, min(gp['birth_rec_in'], V))
        self.c_i = np.zeros(self.coinc_cap, dtype=np.int32)
        self.c_j = np.zeros(self.coinc_cap, dtype=np.int32)
        self.c_rows = np.zeros((self.coinc_cap, self.fan), dtype=np.int32)
        self.c_w = np.zeros((self.coinc_cap, self.fan))
        self.c_usage = np.zeros(self.coinc_cap)
        self.c_alive = np.zeros(self.coinc_cap, dtype=bool)
        self.n_coinc = 0
        self.coinc_free = []
        self.e_s = np.zeros(self.syn_cap)
        self.e_c = np.zeros((self.coinc_cap, self.fan))
        self.elig_lam = float(gp['lam_slow'])
        self.elig_alpha = 1.0 - self.elig_lam
        self.scar_syn = set()
        self.syn_keys = set()
        self.coinc_keys = set()
        self._m_dirty = True
        self._mro = None
        self._mii = None
        self.recent = []
        self._since_replay = 0
        self.err = np.ones(V)
        self.err_slow = np.ones(V)
        self.hab = np.zeros(V)
        self.nov = 1.0
        self.nov_slow = 1.0
        self.G = gp['neuro_cost'] * (1.0 + math.log(2 * V) / math.log(100.0))   # G0 = n_cost（一个神经元出生价，派生）
        self.clock = 0
        self.spines = 0
        self.births = 0
        self.deaths = 0
        self.max_n = 2 * V
        self.hw = 2 * V                    # 高水位线：replay 只存 [0,hw)，避免 310MB/token 的灾难
        self.last_e = np.zeros(V)
        self.prev_tok = None               # coverage 0.2.0：上一个 token（共激活生长用）

    # ---------- helpers ----------
    def _masks(self):
        if self._m_dirty:
            self._mro = self.s_alive & (self.s_j < self.V) & (self.s_i < self.hw)
            self._mii = self.s_alive & (self.s_j >= 2 * self.V) & (self.s_j < self.hw)
            self._m_dirty = False
        return self._mro, self._mii

    def _add_syn(self, i, j, w):
        if (i, j) in self.syn_keys:
            return False
        if self.syn_free:
            idx = self.syn_free.pop()
        elif self.n_syn < self.syn_cap:
            idx = self.n_syn
            self.n_syn += 1
        else:
            # coverage 0.1.0：满员 → 淘汰使用率最低的突触（按字节比较，无绝对阈值、无 scar）
            # 0.3.0 A/B：SPROUT_FREEZE_CAP=1 时满员即冻结结构（只调权重，不换人）
            if FREEZE_CAP:
                return False
            alive_syn = np.nonzero(self.s_alive)[0]
            if alive_syn.size == 0:
                return False
            idx = int(alive_syn[int(np.argmin(self.s_usage[alive_syn]))])
            self.syn_keys.discard((int(self.s_i[idx]), int(self.s_j[idx])))
        self.s_i[idx] = i
        self.s_j[idx] = j
        self.s_w[idx] = w
        self.s_alive[idx] = True
        self.s_usage[idx] = 0.0
        self.s_pre[idx] = 0.0
        self.syn_keys.add((i, j))
        self._m_dirty = True
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
            # coverage 0.1.0：满员 → 淘汰使用率最低的配对单元（同上）
            alive_c = np.nonzero(self.c_alive)[0]
            if alive_c.size == 0:
                return False
            idx = int(alive_c[int(np.argmin(self.c_usage[alive_c]))])
            self.coinc_keys.discard((int(self.c_i[idx]), int(self.c_j[idx])))
        self.c_i[idx] = i
        self.c_j[idx] = j
        self.c_rows[idx, :] = rows
        self.c_w[idx, :] = w
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
        self.max_n = max(self.max_n, int(self.int_alive.sum()) + 2 * self.V)
        self.hw = max(self.hw, nid + 1)
        self._m_dirty = True
        return nid

    def kill_int(self, nid):
        self.int_alive[nid] = False
        self.v[nid] = self.u[nid] = self.ap[nid] = self.av[nid] = 0.0
        self.free_slots.append(nid)
        ms = self.s_alive & ((self.s_i == nid) | (self.s_j == nid))
        if ms.any():
            ms_idx = np.nonzero(ms)[0]
            for idx in ms_idx:
                self.syn_keys.discard((int(self.s_i[idx]), int(self.s_j[idx])))
            self.s_alive[ms] = False
            self.syn_free.extend(int(x) for x in ms_idx)
        mcc = self.c_alive & ((self.c_i == nid) | (self.c_j == nid))
        if mcc.any():
            mcc_idx = np.nonzero(mcc)[0]
            for idx in mcc_idx:
                self.coinc_keys.discard((int(self.c_i[idx]), int(self.c_j[idx])))
            self.c_alive[mcc] = False
            self.coinc_free.extend(int(x) for x in mcc_idx)
        self._m_dirty = True
        self.deaths += 1

    def reset_state(self):
        """按记录复位：清膜电位/适应/活动（权重、结构、稳态统计保留）。"""
        self.v[:] = 0.0
        self.u[:] = 0.0
        self.ap[:] = 0.0
        self.prev_tok = None

    # ---------- core step ----------
    def logits_from(self, ap):
        l = self.b.copy()
        mro, _ = self._masks()
        if mro.any():
            l += np.bincount(self.s_j[mro], weights=self.s_w[mro] * ap[self.s_i[mro]], minlength=self.V)
        mc = self.c_alive
        if mc.any():
            fp = ap[self.c_i[mc]] * ap[self.c_j[mc]]
            rows = self.c_rows[mc]
            wv = self.c_w[mc]
            for f in range(self.fan):
                l += np.bincount(rows[:, f], weights=fp * wv[:, f], minlength=self.V)
        return l

    def step(self, x, logits, learn):
        gp = self.gp
        self.clock += 1
        # coverage 0.2.0：共激活生长——刚观测到 (prev_tok → x) 且无突触时立即建一条。
        # 结构由共激活事件驱动（非采样猜测），容量由内存预算约束（满员按使用率淘汰）。
        # 0.9.0：源节点改用 slow[prev]（索引 +V）。快通道 λ=0.3 让"上一个 token"在下一步
        #   只剩 30%，单条边最大贡献被锁在 0.9 logit；slow 通道 λ=0.85 保留 85%，
        #   同样的 w_clip 能表达 2.8× 的 logit 幅度，才可能学成真正的一阶表。
        if learn and self.prev_tok is not None and self.prev_tok != x:
            key = (int(self.prev_tok), int(x))
            if key not in self.syn_keys and key not in self.scar_syn:
                sgn = 1.0 if self.last_e[x] >= 0.0 else -1.0
                if self._add_syn(key[0], key[1], gp['spine_w0'] * sgn):
                    self.spines += 1
        if logits is not None:
            z = logits - logits.max()
            p = np.exp(z)
            p /= p.sum()
            e = -p
            e[x] += 1.0
            self.last_e = e
            if learn:
                self.hab[x] = self.elig_lam * self.hab[x] + self.elig_alpha * float(e[x])
                self.plasticity(e)
        hw = self.hw
        # 0.3.0：状态只在 [0,hw) 上更新（空槽恒为 0，等价但省 ~3× 内存带宽）。
        self.v[:hw] = (self.lam_vec[:hw] * self.v[:hw] - gp['g_adapt'] * self.u[:hw]
                       + self.rng.randn(hw) * gp['sigma'])
        self.v[x] += gp['c_in']
        self.v[x + self.V] += gp['c_in']
        self.v[:hw] += gp['c_basal'] * (1.0 - self.lam_vec[:hw]) * self.int_alive[:hw]
        _, mii = self._masks()
        if mii.any():
            self.v[:hw] += np.bincount(self.s_j[mii], weights=self.s_w[mii] * self.ap[self.s_i[mii]], minlength=hw)
        a = np.clip(self.v[:hw], 0.0, 1.0)
        self.u[:hw] += gp['tau_adapt'] * (a - self.u[:hw])
        self.av[:hw] += 0.01 * (a - self.av[:hw])
        self.var[:hw] += 0.01 * ((a - self.av[:hw]) ** 2 - self.var[:hw])
        self.ap[:hw] = a
        self.prev_tok = int(x)
        return self.logits_from(a)

    def plasticity(self, e):
        gp = self.gp
        es = 1.0 / (1.0 + self.clock / gp['eta_age_tau'])
        nov = float(e @ e)
        rpe = math.tanh(gp['rpe_gain'] * (self.nov_slow - nov))
        self.nov += gp['nov_ema'] * (nov - self.nov)
        self.nov_slow += gp['nov_slow'] * (nov - self.nov_slow)
        mro, mii = self._masks()
        hw = self.hw
        blame = np.zeros(hw)
        if mro.any():
            blame += np.bincount(self.s_i[mro], weights=self.s_w[mro] * e[self.s_j[mro]], minlength=hw)
        mc = self.c_alive
        if mc.any():
            ci = self.c_i[mc]
            cj = self.c_j[mc]
            api = self.ap[ci]
            apj = self.ap[cj]
            contrib = self.c_w[mc] * e[self.c_rows[mc]]
            ci_rep = np.repeat(ci, self.fan)
            cj_rep = np.repeat(cj, self.fan)
            blame += np.bincount(ci_rep, weights=(contrib * apj[:, None]).ravel(), minlength=hw)
            blame += np.bincount(cj_rep, weights=(contrib * api[:, None]).ravel(), minlength=hw)
        cost = 0.0
        an = self.ap[:hw] / (np.sqrt(self.var[:hw]) + 0.1)
        lam_e = self.elig_lam
        alp_e = self.elig_alpha
        if mro.any():
            prod = an[self.s_i[mro]] * e[self.s_j[mro]]
            self.e_s[mro] = lam_e * self.e_s[mro] + alp_e * prod
            d = gp['eta_r'] * es * self.e_s[mro]
            self.s_w[mro] = np.clip(self.s_w[mro] + d, -gp['w_clip'], gp['w_clip'])
            cost += np.abs(d).sum()
        m_intro = mro & (self.s_i >= 2 * self.V)   # 0.8.0 毕业：内部→token 读出突触纳入统一维护规则
        if m_intro.any():
            pre_r = self.ap[self.s_i[m_intro]]
            post_r = self.ap[self.s_j[m_intro]]
            self.s_usage[m_intro] = 0.98 * self.s_usage[m_intro] + pre_r * post_r
            self.s_pre[m_intro] = 0.98 * self.s_pre[m_intro] + pre_r
            weak_r = m_intro.copy()
            weak_r[m_intro] = (self.s_usage[m_intro] < gp['w_res']) & (self.s_pre[m_intro] >= gp['w_res'])
            self.s_w[weak_r] *= (1.0 - gp['wd'])
        if mc.any():
            fpn = an[self.c_i[mc]] * an[self.c_j[mc]]
            prod = fpn[:, None] * e[self.c_rows[mc]]
            self.e_c[mc] = lam_e * self.e_c[mc] + alp_e * prod
            d = gp['eta_coinc'] * es * self.e_c[mc]
            self.c_w[mc] = np.clip(self.c_w[mc] + d, -gp['w_clip'], gp['w_clip'])
            cost += np.abs(d).sum()
        if mii.any():
            pre = self.ap[self.s_i[mii]]
            post = self.ap[self.s_j[mii]]
            mod = np.tanh(gp['blame_gain'] * blame[self.s_j[mii]]) + 0.3 * rpe
            prod = pre * post * mod
            self.e_s[mii] = lam_e * self.e_s[mii] + alp_e * prod
            d = gp['eta_i'] * es * self.e_s[mii]
            self.s_w[mii] += d
            cost += np.abs(d).sum()
            self.s_usage[mii] = 0.98 * self.s_usage[mii] + pre * post
            self.s_pre[mii] = 0.98 * self.s_pre[mii] + pre
            weak = mii.copy()
            weak[mii] = (self.s_usage[mii] < gp['w_res']) & (self.s_pre[mii] >= gp['w_res'])
            self.s_w[weak] *= (1.0 - gp['wd'])
        self.b += gp['eta_b'] * es * e
        np.clip(self.b, -self.b_clip, self.b_clip, out=self.b)
        # coverage 0.1.0：删除"|w|<w_res 绝对阈值剪枝 + scar 永久禁止再生"。
        # 结构死亡只由内存满员时的排名淘汰负责（_add_syn/_add_coinc），权重照常衰减。
        if mc.any():
            self.c_usage[mc] = 0.99 * self.c_usage[mc] + 0.01 * self.ap[self.c_i[mc]] * self.ap[self.c_j[mc]]
            # coverage 0.1.0：删除绝对阈值剪枝（同突触）
        self.G -= gp['m_plast'] * float(cost)
        self.err += gp['e_ema'] * (np.abs(e) - self.err)
        self.err_slow += gp['e_ema'] * 0.1 * (np.abs(e) - self.err_slow)

    def energy(self):
        gp = self.gp
        sf = 1.0 + math.log(1.0 + self.V) / math.log(25.0)   # 有界亚线性规模因子（V=96≈2.4，V=10000≈3.9）
        inc = gp['pool_in'] * sf * (1.0 + gp['stress_gain'] * self.nov / (1.0 + self.nov))
        self.G += inc
        spend = gp['m_receptor'] * 2 * sf + gp['m_neuron'] * int(self.int_alive.sum()) + gp['m_spike'] * float(self.ap.sum())
        m = self.s_alive
        if m.any():
            spend += gp['m_syn'] * int(m.sum()) + gp['m_syn_w'] * float((np.abs(self.s_w[m]) * self.ap[self.s_i[m]]).sum())
        mc = self.c_alive
        if mc.any():
            spend += gp['m_syn'] * int(mc.sum()) + gp['m_syn_w'] * float(
                (np.abs(self.c_w[mc]).sum(axis=1) * (self.ap[self.c_i[mc]] * self.ap[self.c_j[mc]])).sum())
        self.G -= spend
        if self.G > 0.0:
            self.G -= gp['g_leak'] * self.G   # 代谢泄漏：多余能量随池规模持续耗散（自然软上限）
        hw = self.hw
        rmask = np.arange(hw) < 2 * self.V
        atp = self.atp[:hw]
        atp[rmask] = np.minimum(1.0, atp[rmask] * gp['atp_decay'] + gp['use_gain'] * self.ap[:hw][rmask] + gp['receptor_supply'])
        ms_use = self.s_alive & (self.s_i >= 2 * self.V) & (self.s_j < self.V) & (self.s_i < hw)   # 内部→token 读出突触
        if ms_use.any():
            use_inc = np.bincount(self.s_i[ms_use], weights=self.s_usage[ms_use], minlength=hw)
        else:
            use_inc = np.zeros(hw)
        ii = self.int_alive[:hw]
        atp[ii] = (atp[ii] * gp['atp_decay'] + gp['int_basal'] - gp['m_int_upkeep']
                   + 0.5 * gp['use_gain'] * self.ap[:hw][ii]
                   + gp['util_gain'] * use_inc[ii])
        # ---- budget 0.1.0：生存不再由手填的 ATP 收支决定 ----
        # 收入统计 u_util = 使用率（读出突触 s_usage 之和）的慢 EMA——纯实测、无手填尺度；
        # 死亡只发生在【内存满员 + 更高需求的出生要进来】时：按 u_util 排名淘汰最低者。
        self.u_util[:hw] = self.u_util[:hw] * gp['atp_decay'] + use_inc
        if self.G < 0.0:
            self.G = 0.0

    def _bytes_used(self):
        return (float(self.int_alive.sum()) * _INT_BYTES
                + float(self.s_alive.sum()) * _SYN_BYTES
                + float(self.c_alive.sum()) * _COINC_BYTES)

    def _admit(self, need_bytes, protect=-1):
        """内存准入：够就直接放行；不够就淘汰 u_util 最低的神经元（换人），腾出为止。"""
        if self._bytes_used() + need_bytes <= _MEM_BYTES:
            return True
        guard = 0
        while guard < 64:
            guard += 1
            alive = np.nonzero(self.int_alive)[0]
            if alive.size == 0:
                return False
            vals = self.u_util[alive].copy()
            if protect >= 0:
                vals[alive == protect] = np.inf
            nid = int(alive[int(np.argmin(vals))])
            self.kill_int(nid)
            if self._bytes_used() + need_bytes <= _MEM_BYTES:
                return True
        return False

    def growth(self):
        gp = self.gp
        wts = self.err * self.err
        tok = int(self.rng.choice(self.V, p=wts / wts.sum()))
        q_x = 1.0 - float(self.hab[tok])
        qbar = max(1e-6, 1.0 - float(self.hab.mean()))
        demand = (1.0 + max(-1.0, min(1.0, (qbar - q_x) / (qbar + q_x + 1e-12)))) * 0.5
        # budget 0.1.0：出生尝试只看需求（不再由手填能量 G 节流）；内存准入在下面把关。
        r_sp = gp['g_syn'] * demand
        mem_ok_syn = True    # coverage: 满员时由 _add_syn 排名淘汰，不再挡出生
        mem_ok_coinc = True  # coverage: 同突触
        if self.rng.rand() < 1.0 - math.exp(-r_sp):
            sgn = 1.0 if self.last_e[tok] >= 0.0 else -1.0
            shw = self.hw
            score = np.where(self.int_alive[:shw], self.ap[:shw] - self.av[:shw], 0.0)
            score[self.fast] = self.ap[self.fast] - self.av[self.fast]
            score[self.slow] = self.ap[self.slow] - self.av[self.slow]
            score[self.fast[tok]] = -1.0
            roll = self.rng.rand()
            if roll < gp['g_dend'] and self.int_alive.any():
                alive_int = np.nonzero(self.int_alive[:self.hw])[0]
                wt = self.ap[alive_int] + 1e-6
                tgt_i = int(self.rng.choice(alive_int, p=wt / wt.sum()))
                cand = np.argsort(-score)[:16] if mem_ok_syn else []
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
                alive_ch = np.concatenate([self.fast, self.slow, np.nonzero(self.int_alive[:self.hw])[0]])
                act = self.ap[alive_ch] + 1e-6
                act = act / act.sum()
                for att in range(4 if mem_ok_coinc else 0):
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
                    is_int = bool(self.int_alive[nid])
                    if prefer_int != is_int:
                        continue
                    if (int(nid), tok) in self.scar_syn or (int(nid), tok) in self.syn_keys:
                        continue
                    if score[nid] > best_s:
                        best_s = score[nid]
                        best = int(nid)
                if best >= 0 and mem_ok_syn:
                    if self._add_syn(best, tok, gp['spine_w0'] * sgn):
                        self.G -= gp['spine_cost']
                        self.spines += 1
        n_cost = gp['neuro_cost'] * (1.0 + math.log(2 * self.V) / math.log(100.0))
        r_nb = gp['g_neuro'] * demand    # budget：只看需求，不看手填能量
        if self.rng.rand() < 1.0 - math.exp(-r_nb) and self._admit(_INT_BYTES + 16.0 * _SYN_BYTES):
            nid = self.new_int()
            if nid >= 0:
                lam_i = float(self.rng.uniform(0.10, 0.90))
                self.lam_vec[nid] = lam_i
                gap = float(10.0 ** self.rng.uniform(-2.0, -0.22))
                w_self = max(0.02, (1.0 - gap) - lam_i)
                self._add_syn(nid, nid, w_self)
                order = self.rng.permutation(self.V)
                fan_in = max(1, min(gp['birth_n_in'], max(1, self.V // 6)))
                for tok in order[:fan_in]:
                    w = float(self.rng.uniform(gp['birth_in_lo'], gp['birth_in_hi']))
                    if self.rng.rand() < 0.5:
                        w = -w
                    self._add_syn(int(tok), nid, w)
                alive_int = np.nonzero(self.int_alive[:self.hw])[0]
                alive_rec = np.concatenate([self.fast, self.slow])
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
                top = np.argsort(-self.err)[:2]
                for tok in top:
                    w = gp['out_init'] if self.last_e[tok] >= 0 else -gp['out_init']
                    self._add_syn(nid, int(tok), w)
                self.G -= n_cost
                self.births += 1

    # ---------- 睡眠重放（序列化小批次巩固） ----------
    def replay(self):
        gp = self.gp
        B = len(self.recent)
        if B == 0:
            return 0
        keep = min(len(r[0]) for r in self.recent)
        Ap = np.stack([r[0][:keep] for r in self.recent])
        xs = np.array([r[1] for r in self.recent])
        es = 1.0 / (1.0 + self.clock / gp['eta_age_tau'])
        mro = self.s_alive & (self.s_j < self.V) & (self.s_i < keep)
        mc = self.c_alive & (self.c_i < keep) & (self.c_j < keep)
        den = (np.sqrt(self.var[:keep]).astype(np.float32) + np.float32(0.1))  # 分块时再除，峰值少一份 (B,hw) 数组
        bs = gp['replay_batch']
        # 0.3.0：分块更小省内存；按 (块数×轮数) 补偿，保持每次 replay 的总更新量不变
        sc = 32.0 * bs / (gp['replay_sweeps'] * B)
        order = np.arange(B)
        ci = self.c_i[mc]
        cj = self.c_j[mc]
        cr = self.c_rows[mc]
        W = self.c_w[mc].copy() if len(ci) else None
        sweeps = 0
        for _ in range(gp['replay_sweeps']):
            self.rng.shuffle(order)
            for s in range(0, B, bs):
                idx = order[s:s + bs]
                A = Ap[idx]
                X = xs[idx]
                L = np.tile(self.b, (len(idx), 1))
                if mro.any():
                    vals = A[:, self.s_i[mro]] * self.s_w[mro]
                    sj = self.s_j[mro]
                    L += np.array([np.bincount(sj, weights=vals[r], minlength=self.V)
                                   for r in range(len(idx))])
                if len(ci):
                    F = (A[:, ci] * A[:, cj]).astype(np.float64)
                    for f in range(self.fan):
                        cf = cr[:, f]
                        Wf = W[:, f]
                        L += np.array([np.bincount(cf, weights=F[r] * Wf, minlength=self.V)
                                       for r in range(len(idx))])
                L -= L.max(axis=1, keepdims=True)
                P = np.exp(L)
                P /= P.sum(axis=1, keepdims=True)
                T = np.zeros((len(idx), self.V))
                T[np.arange(len(idx)), X] = 1.0
                E = T - P
                Anb = Ap[idx] / den
                if mro.any():
                    d = sc * gp['eta_r'] * es * (Anb[:, self.s_i[mro]] * E[:, self.s_j[mro]]).mean(axis=0)
                    self.s_w[mro] = np.clip(self.s_w[mro] + d, -gp['w_clip'], gp['w_clip'])
                if len(ci):
                    Fn = (Anb[:, ci] * Anb[:, cj]).astype(np.float64)
                    for f in range(self.fan):
                        dW = sc * gp['eta_coinc'] * es * (Fn * E[:, cr[:, f]]).mean(axis=0)
                        np.clip(W[:, f] + dW, -gp['w_clip'], gp['w_clip'], out=W[:, f])
                self.b += sc * gp['eta_b'] * es * E.mean(axis=0)
            sweeps += 1
        if len(ci):
            self.c_w[mc] = W
        return sweeps

    # ---------- 流式（掩码感知） ----------
    def run_tokens(self, tokens, learn=False, mask=None, score=True, reset=False):
        """逐 token 流式处理。

        mask[i]=True 表示第 i 个 token 产生学习信号（问答语料的 assistant 段）；
        上下文（user/system）只推进状态、不学习、不进重放、不参与生长。
        验证时只统计 mask 位的 NLL（防泄露口径）。
        """
        if reset and RESET_STATE_PER_RECORD:
            self.reset_state()
        nll = 0.0
        n = 0
        hits = 0
        logits = None
        for i, x in enumerate(tokens):
            m = True if mask is None else bool(mask[i])
            if score and logits is not None and m:
                z = logits - logits.max()
                p = np.exp(z)
                s = p.sum()
                nll += -math.log(max(float(p[x]) / float(s), 1e-12))
                n += 1
                if p.argmax() == x:
                    hits += 1
            learn_here = learn and m
            if learn_here and logits is not None:
                self.recent.append((self.ap[:self.hw].astype(np.float32), int(x)))
                if len(self.recent) > self.gp['replay_len']:
                    self.recent.pop(0)
            logits = self.step(int(x), logits, learn_here)
            if learn_here:
                if self.clock % 8 == 0:
                    self.energy()   # budget 框架下 ATP/G 仅诊断：每 8 步更新一次（0.3.0 提速）
                self.growth()
                self._since_replay += 1
                if (self.gp['micro_sleep'] and self.gp['replay_len'] > 0
                        and self._since_replay >= self.gp['replay_len'] and self.recent):
                    self._since_replay = 0
                    self.replay()
        ppl = math.exp(nll / n) if n else float('inf')
        return ppl, n, hits

    # ---------- 生成 ----------
    def generate(self, seed_ids, steps, rng=None, temperature=1.0, top_p=1.0, stream=None, stop_ids=()):
        saved = self.gp['sigma']
        self.gp['sigma'] = saved * self.gp['dream_noise']
        logits = None
        for tok in seed_ids:
            logits = self.step(int(tok), logits, False)
        out = []
        temp = max(float(temperature), 1e-6)
        for _ in range(steps):
            z = logits / temp
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            if top_p < 1.0:
                order = np.argsort(-p)
                csum = np.cumsum(p[order])
                k = int(np.searchsorted(csum, top_p) + 1)
                keep = order[:k]
                q = np.zeros_like(p)
                q[keep] = p[keep]
                s = q.sum()
                p = q / s if s > 0 else p
            y = int(p.argmax()) if rng is None else int(np.searchsorted(np.cumsum(p), rng.random()))
            out.append(y)
            if stream is not None:
                stream(y)
            logits = self.step(y, logits, False)
            if stop_ids and y in stop_ids:
                break
        self.gp['sigma'] = saved
        return out

    # ---------- 检查点 ----------
    def save(self, path):
        syn_keys_arr = np.array(sorted(self.syn_keys), dtype=np.int32) if self.syn_keys else np.zeros((0, 2), np.int32)
        coinc_keys_arr = np.array(sorted(self.coinc_keys), dtype=np.int32) if self.coinc_keys else np.zeros((0, 2), np.int32)
        np.savez(
            _ckpt_file(path),
            V=self.V, N=self.N, lam_vec=self.lam_vec,
            int_alive=self.int_alive, b=self.b,
            s_i=self.s_i, s_j=self.s_j, s_w=self.s_w, s_alive=self.s_alive, s_usage=self.s_usage, s_pre=self.s_pre, n_syn=self.n_syn,
            c_i=self.c_i, c_j=self.c_j, c_rows=self.c_rows, c_w=self.c_w, c_alive=self.c_alive,
            c_usage=self.c_usage, n_coinc=self.n_coinc,
            syn_keys=syn_keys_arr, coinc_keys=coinc_keys_arr,
            clock=self.clock, spines=self.spines, births=self.births, deaths=self.deaths, max_n=self.max_n, hw=self.hw,
            err=self.err, err_slow=self.err_slow, hab=self.hab, nov=self.nov, nov_slow=self.nov_slow, G=self.G,
            av=self.av, var=self.var, atp=self.atp, u_util=self.u_util,
            gp_json=json.dumps(self.gp),
        )

    def load(self, path, seed=1):
        z = np.load(_ckpt_file(path), allow_pickle=False)
        V = int(z['V'])
        if V != self.V:
            raise RuntimeError(f"检查点词表规模 {V} 与当前 tokenizer {self.V} 不一致：请使用相同词表")
        self.clock = int(z['clock'])
        self.spines = int(z['spines'])
        self.births = int(z['births'])
        self.deaths = int(z['deaths'])
        self.max_n = int(z['max_n'])
        self.G = float(z['G'])
        self.nov = float(z['nov'])
        self.nov_slow = float(z['nov_slow'])
        self.err[:] = z['err']
        self.err_slow[:] = z['err_slow']
        if 'hab' in z.files:
            self.hab[:] = z['hab']
        self.av[:] = z['av']
        self.var[:] = z['var']
        self.atp[:] = z['atp']
        if 'u_util' in z.files:
            self.u_util[:] = z['u_util']
        self.lam_vec[:] = z['lam_vec']
        self.int_alive[:] = z['int_alive']
        self.hw = max(2 * V, int(np.nonzero(self.int_alive)[0].max()) + 1) if self.int_alive.any() else 2 * V
        self.b[:] = z['b']
        self.s_i[:] = z['s_i']
        self.s_j[:] = z['s_j']
        self.s_w[:] = z['s_w']
        self.s_alive[:] = z['s_alive']
        self.s_usage[:] = z['s_usage']
        if 's_pre' in z.files:
            self.s_pre[:] = z['s_pre']
        self.n_syn = int(z['n_syn'])
        self.c_i[:] = z['c_i']
        self.c_j[:] = z['c_j']
        self.c_rows[:] = z['c_rows']
        self.c_w[:] = z['c_w']
        self.c_alive[:] = z['c_alive']
        self.c_usage[:] = z['c_usage']
        self.n_coinc = int(z['n_coinc'])
        sk = z['syn_keys']
        self.syn_keys = set((int(a), int(b)) for a, b in sk)
        ck = z['coinc_keys']
        self.coinc_keys = set((int(a), int(b)) for a, b in ck)
        self.free_slots = [nid for nid in range(2 * self.V, self.N) if not self.int_alive[nid]]
        self.syn_free = [i for i in range(self.n_syn) if not self.s_alive[i]]
        self._compact()

    def _compact(self):
        """0.3.0：把内部神经元重映射到 [2V, 2V+k) 紧凑区间（旧检查点的高位槽 → hw 可收缩，速度恢复）。
        突触/配对准入按同映射改写；引用已死内部神经元的突触直接退役。"""
        V = self.V
        ids = np.nonzero(self.int_alive)[0]
        if ids.size == 0:
            self.hw = 2 * V
            return
        mp = np.full(self.N, -1, dtype=np.int64)
        mp[ids] = np.arange(2 * V, 2 * V + ids.size)
        for arr in (self.s_i, self.s_j, self.c_i, self.c_j):
            m = arr >= 2 * V
            arr[m] = np.where(mp[arr[m]] >= 0, mp[arr[m]], arr[m])
        dead_s = (self.s_alive & (self.s_i >= 2 * V) & (mp[self.s_i] < 0)) | \
                 (self.s_alive & (self.s_j >= 2 * V) & (mp[self.s_j] < 0))
        if dead_s.any():
            self.s_alive[dead_s] = False
            for idx in np.nonzero(dead_s)[0]:
                self.syn_keys.discard((int(self.s_i[idx]), int(self.s_j[idx])))
            self.syn_free = [i for i in range(self.n_syn) if not self.s_alive[i]]
        dead_c = ((self.c_alive & (self.c_i >= 2 * V) & (mp[self.c_i] < 0))
                  | (self.c_alive & (self.c_j >= 2 * V) & (mp[self.c_j] < 0)))
        if dead_c.any():
            self.c_alive[dead_c] = False
            for idx in np.nonzero(dead_c)[0]:
                self.coinc_keys.discard((int(self.c_i[idx]), int(self.c_j[idx])))
            self.coinc_free = [i for i in range(self.n_coinc) if not self.c_alive[i]]
        for name in ('v', 'u', 'ap', 'av', 'var', 'atp', 'lam_vec', 'int_alive', 'u_util'):
            old = getattr(self, name)
            new = np.zeros_like(old)
            new[:2 * V] = old[:2 * V]
            new[2 * V:2 * V + ids.size] = old[ids]
            setattr(self, name, new)
        self.hw = 2 * V + int(ids.size)
        self.free_slots = [nid for nid in range(2 * V, self.N) if not self.int_alive[nid]]
        self._m_dirty = True
        self.coinc_free = [i for i in range(self.n_coinc) if not self.c_alive[i]]
        self.v[:] = 0.0
        self.u[:] = 0.0
        self.ap[:] = 0.0
        self.recent = []
        self._since_replay = 0
        self._m_dirty = True

    def stats(self):
        return dict(
            V=self.V, N=self.N, ints=int(self.int_alive.sum()), max_ints=self.max_n - 2 * self.V,
            syn=int(self.s_alive.sum()), coinc=int(self.c_alive.sum()), fan=self.fan,
            G=self.G, clock=self.clock, births=self.births, deaths=self.deaths,
            spines=self.spines, cap_int=self.gp['cap_int'], cap_syn=self.gp['cap_syn'],
            cap_coinc=self.gp['cap_coinc'],
        )

    def topology_report(self):
        """生长拓扑快照：度分布 / 枢纽 / 覆盖 / 时间尺度分带（用于与生物连接组统计对照）。"""
        V = self.V
        int_ids = np.nonzero(self.int_alive)[0]
        sa = self.s_alive
        si = self.s_i[sa].astype(np.int64)
        sj = self.s_j[sa].astype(np.int64)
        n_syn_alive = int(sa.sum())
        to_tok = int((sj < V).sum())
        rec_to_tok = int(((sj < V) & (si < 2 * V)).sum())
        int_to_tok = int(((sj < V) & (si >= 2 * V)).sum())
        to_int = int((sj >= 2 * V).sum())
        self_loops = int((si == sj).sum())
        indeg_tok = np.bincount(sj[sj < V], minlength=V) if to_tok else np.zeros(V, dtype=np.int64)
        covered_tok = int((indeg_tok > 0).sum())
        outdeg_src = np.bincount(si, minlength=self.N)
        int_out = outdeg_src[int_ids] if len(int_ids) else np.zeros(0, dtype=np.int64)
        indeg_int_all = np.bincount(sj[sj >= 2 * V], minlength=self.N) if to_int else np.zeros(self.N, dtype=np.int64)
        int_in = indeg_int_all[int_ids] if len(int_ids) else np.zeros(0, dtype=np.int64)
        mc = self.c_alive
        ci = self.c_i[mc]; cj = self.c_j[mc]
        rows_unique = np.unique(self.c_rows[mc].ravel()) if int(mc.sum()) else np.zeros(0, dtype=np.int64)
        covered_pair = int(len(rows_unique))
        covered_union = int(len(np.union1d(np.nonzero(indeg_tok)[0], rows_unique)))
        rr = int(((ci < 2 * V) & (cj < 2 * V)).sum())
        wi = int(mc.sum()) - rr
        lam_ints = self.lam_vec[int_ids] if len(int_ids) else np.zeros(0)
        bands = [int(((lam_ints >= a) & (lam_ints < b)).sum()) for a, b in ((0.0, 0.35), (0.35, 0.7), (0.7, 1.01))]
        top_rows = np.argsort(-indeg_tok)[:10]
        top_hubs = ' '.join(f"t{int(t)}:{int(indeg_tok[t])}" for t in top_rows if indeg_tok[t] > 0)
        med = lambda a: (int(np.median(a)) if len(a) else 0)
        mx = lambda a: (int(a.max()) if len(a) else 0)
        print('===== 生长拓扑快照 =====')
        print(f"步数 clock={self.clock} | 能量 G={self.G:.2f} | 总出生 {self.births} / 死亡 {self.deaths}")
        print(f"内部神经元: 存活 {len(int_ids)}（内存上界 {self.gp['cap_int']}）| lam 分带 快[0,0.35)={bands[0]} 中[0.35,0.7)={bands[1]} 慢[0.7,1)={bands[2]}")
        if len(int_ids):
            print(f"  出度(→token行/其它): 中位 {med(int_out)} 最大 {mx(int_out)} | 有出突触的 {int((int_out > 0).sum())} 个")
            print(f"  入度(树突输入): 中位 {med(int_in)} 最大 {mx(int_in)} | 有树突输入的 {int((int_in > 0).sum())} 个")
        print(f"突触（存活 {n_syn_alive}）: 感受器→token行 {rec_to_tok} | 内部→token行 {int_to_tok} | →内部(树突) {to_int} | 自环 {self_loops}")
        print(f"  token 行入度覆盖: {covered_tok}/{V} ({100.0 * covered_tok / V:.1f}%) | 枢纽: {top_hubs or '-'}")
        print(f"配对角（存活 {int(mc.sum())}）: 感受器×感受器 {rr} | 含内部 {wi}")
        print(f"  配对覆盖 token 行: {covered_pair}/{V} ({100.0 * covered_pair / V:.1f}%)")
        print(f"总覆盖（突触 ∪ 配对）: {covered_union}/{V} ({100.0 * covered_union / V:.1f}%)")
        def _gini(a):
            a = np.asarray(a, dtype=float)
            if len(a) < 2 or a.mean() <= 0:
                return 0.0
            return float(np.abs(np.subtract.outer(a, a)).mean() / (2.0 * a.mean()))
        def _pct(a, q):
            return int(np.percentile(a, q)) if len(a) else 0
        def _tail_alpha(deg):
            nz = np.sort(deg[deg > 0])[::-1]
            if len(nz) < 20:
                return 0.0
            k = min(200, len(nz))
            x = np.log(np.maximum(nz[:k].astype(float), 1.0))
            y = np.log(np.arange(1, k + 1, dtype=float))
            return float(-np.polyfit(x, y, 1)[0])
        nz_tok = indeg_tok[indeg_tok > 0]
        print(f"token 行入度（有输入 {covered_tok}）: p50={_pct(nz_tok, 50)} p90={_pct(nz_tok, 90)} p99={_pct(nz_tok, 99)} max={int(indeg_tok.max())} | 基尼={_gini(nz_tok):.2f} | 尾斜率 α≈{_tail_alpha(indeg_tok):.2f}")
        if len(int_ids):
            print(f"内部神经元 出度: p50={_pct(int_out,50)} p90={_pct(int_out,90)} max={mx(int_out)} 基尼={_gini(int_out):.2f} | 入度: p50={_pct(int_in,50)} p90={_pct(int_in,90)} max={mx(int_in)} 基尼={_gini(int_in):.2f}")
            idx_map = -np.ones(self.N, dtype=np.int64)
            idx_map[int_ids] = np.arange(len(int_ids))
            m_int = (si >= 2 * V) & (sj >= 2 * V)
            a_src = idx_map[si[m_int]]
            a_tgt = idx_map[sj[m_int]]
            ok = (a_src >= 0) & (a_tgt >= 0)
            a_src = a_src[ok]; a_tgt = a_tgt[ok]
            non_self = a_src != a_tgt
            A = np.zeros((len(int_ids), len(int_ids)))
            A[a_src[non_self], a_tgt[non_self]] = 1.0
            deg_i = A.sum(1)
            tri = (A @ A * A).sum(1)
            okd = deg_i > 1
            C = float((tri[okd] / (deg_i[okd] * (deg_i[okd] - 1))).mean()) if okd.any() else 0.0
            nulls = []
            rng_t = np.random.RandomState(7)
            for _ in range(20):
                tgt2 = rng_t.permutation(a_tgt)
                nss = a_src != tgt2
                B = np.zeros_like(A)
                B[a_src[nss], tgt2[nss]] = 1.0
                d2 = B.sum(1); tr2 = (B @ B * B).sum(1); ok2 = d2 > 1
                nulls.append(float((tr2[ok2] / (d2[ok2] * (d2[ok2] - 1))).mean()) if ok2.any() else 0.0)
            C0 = float(np.mean(nulls))
            mutual = int((A * A.T).sum() // 2)
            print(f"内部图（int→int，{int(non_self.sum())} 条边）: 自环 {int((a_src == a_tgt).sum())} | 互环 {mutual} | 聚类 C={C:.3f} vs 目标随机基线 {C0:.3f}（比值 {(C / C0 if C0 > 0 else 0):.1f}×）")
            msp = (si >= 2 * V) & (sj < V)
            r_ids = sj[msp]; c_ids = idx_map[si[msp]]
            okr = c_ids >= 0
            r_ids = r_ids[okr]; c_ids = c_ids[okr]
            urows = np.unique(r_ids)
            if len(urows) >= 3 and len(int_ids) >= 2:
                rmap = -np.ones(V, dtype=np.int64)
                rmap[urows] = np.arange(len(urows))
                M = np.zeros((len(urows), len(int_ids)))
                M[rmap[r_ids], c_ids] = 1.0
                Co = M @ M.T
                np.fill_diagonal(Co, 0.0)
                E = (Co > 0).astype(float)
                d_r = E.sum(1)
                t_r = (E @ E * E).sum(1)
                okr2 = d_r > 1
                C_r = float((t_r[okr2] / (d_r[okr2] * (d_r[okr2] - 1))).mean()) if okr2.any() else 0.0
                reps = 10 if len(urows) <= 800 else 3
                nulls_r = []
                for _ in range(reps):
                    c2 = rng_t.permutation(c_ids)
                    M2 = np.zeros_like(M)
                    M2[rmap[r_ids], c2] = 1.0
                    Co2 = M2 @ M2.T
                    np.fill_diagonal(Co2, 0.0)
                    E2 = (Co2 > 0).astype(float)
                    d2r = E2.sum(1)
                    t2r = (E2 @ E2 * E2).sum(1)
                    ok2r = d2r > 1
                    nulls_r.append(float((t2r[ok2r] / (d2r[ok2r] * (d2r[ok2r] - 1))).mean()) if ok2r.any() else 0.0)
                C_r0 = float(np.mean(nulls_r))
                print(f"读出耦合投影（{len(urows)} 个 token 行 × 共享内部源）: 聚类 C={C_r:.3f} vs 源随机基线 {C_r0:.3f}（比值 {(C_r / C_r0 if C_r0 > 0 else 0):.1f}×）")
        print("参考（生物连接组）: 度分布重尾 α≈2–3、枢纽存在、聚类显著高于随机（小世界特征）、自环/互环少量存在")

# ================================================================================================
# 数据层：记录流 + 训练/验证划分（防泄露）+ v5 渲染 → (ids, mask)
# ================================================================================================
def load_lines(path):
    if path.lower().endswith('.txt'):
        # 纯文本/代码语料：按空行分段成块，保留多行结构（比逐行更有上下文）
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
        if DATA_RECORDS_LIMIT:
            blocks = blocks[:DATA_RECORDS_LIMIT]
        return blocks
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                if DATA_RECORDS_LIMIT and len(lines) >= DATA_RECORDS_LIMIT:
                    break
    return lines


def _content_key(line):
    """记录内容的归一化指纹（用于训练/验证集合的重复内容剔除，防泄露）。"""
    try:
        rec = json.loads(line)
        msgs = _messages_from_record(rec)
        parts = []
        for m in msgs:
            parts.append(str(m.get('role', '')))
            parts.append(str(m.get('content') or ''))
            if m.get('reasoning_content'):
                parts.append(str(m['reasoning_content']))
        key = '\u0001'.join(parts)
    except Exception:
        key = line
    return hashlib.md5(normalize_text(key).encode('utf-8', 'ignore')).hexdigest()


def split_records(lines, val_bucket=VAL_BUCKET):
    """确定性划分：内容哈希 % val_bucket == 0 → 验证集；训练集剔除与验证集内容重复的记录。"""
    train, val = [], []
    val_keys = set()
    for line_no, line in enumerate(lines, 1):
        key = _content_key(line)
        if val_bucket > 1 and int(key[:10], 16) % val_bucket == 0:
            val.append((line_no, line))
            val_keys.add(key)
    for line_no, line in enumerate(lines, 1):
        key = _content_key(line)
        if val_bucket > 1 and int(key[:10], 16) % val_bucket == 0:
            continue
        if key in val_keys:
            continue
        train.append((line_no, line))
    return train, val


def _mask_from_spans(ids, offs, spans):
    mask = np.zeros(len(ids), dtype=bool)
    if spans:
        si = 0
        nspans = len(spans)
        for i, (o0, o1) in enumerate(offs):
            while si < nspans and spans[si][1] <= o0:
                si += 1
            if si < nspans and o0 >= spans[si][0] and o1 <= spans[si][1]:
                mask[i] = True
    return mask


def render_record(tokenizer, line, max_seq_len=MAX_SEQ_LEN):
    """一行 JSONL 记录 → (ids, mask)。mask[i]=True 表示该 token 属于 assistant 段（学习目标）。"""
    try:
        rec = json.loads(line)
        msgs = _messages_from_record(rec)
    except Exception:
        return None
    if max_seq_len:
        msgs = _truncate_pairs(msgs, tokenizer, max_seq_len)
    if not _has_assistant_target(msgs):
        return None
    text, spans = render_messages(msgs)
    ids, offs = _encode_with_offsets(tokenizer, text)
    if not ids:
        return None
    return ids, _mask_from_spans(ids, offs, spans)


def render_raw_text(tokenizer, line):
    """纯文本行 → (ids, mask=全 1)。用于 .txt 语料（整段预测）。"""
    text = normalize_text(line)
    ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
    if len(ids) <= 1:
        return None
    mask = np.ones(len(ids), dtype=bool)
    return ids, mask


def render_sample(tokenizer, line, is_txt):
    if is_txt:
        return render_raw_text(tokenizer, line)
    return render_record(tokenizer, line)

# ================================================================================================
# 验证与训练
# ================================================================================================
def evaluate_val(brain, tokenizer, val_records, is_txt=False, max_records=None):
    """验证集：逐记录复位状态、只在 mask 位统计 PPL（问答语料=只算 assistant 段）。"""
    total_nll = 0.0
    total_n = 0
    hits = 0
    used = 0
    for line_no, line in val_records:
        if max_records is not None and used >= max_records:
            break
        r = render_sample(tokenizer, line, is_txt)
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


def train_sprout(brain, tokenizer, data_path, epochs=EPOCHS, save_path=CHECKPOINT):
    lines = load_lines(data_path)
    is_txt = data_path.lower().endswith('.txt')
    train_recs, val_recs = split_records(lines)
    log_and_print(f"数据集 {data_path}: 共 {len(lines)} 条 | 训练 {len(train_recs)} 条 | 验证 {len(val_recs)} 条"
                  f"（内容哈希 %{VAL_BUCKET} 划分，已剔除跨集重复内容；问答语料只对 assistant 段计算 loss/学习）")
    if not train_recs:
        log_and_print("训练集为空，终止。")
        return None
    rng = random.Random(1234)
    val_ppl = float('inf')
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_recs)))
        rng.shuffle(order)
        t0 = time.time()
        tok_count = 0
        tok_win = 0
        step_count = 0
        nll_acc = 0.0
        n_acc = 0
        hit_acc = 0
        skipped = 0
        for oi in order:
            line_no, line = train_recs[oi]
            r = render_sample(tokenizer, line, is_txt)
            if r is None:
                skipped += 1
                continue
            ids, mask = r
            if not mask.any():
                skipped += 1
                continue
            ppl, n, hits = brain.run_tokens(ids, learn=True, mask=mask, score=True, reset=True)
            if n > 0:
                nll_acc += math.log(ppl) * n
                n_acc += n
                hit_acc += hits
            tok_count += len(ids)
            tok_win += len(ids)
            step_count += int(mask.sum())
            if LOG_EVERY_TOKENS and step_count >= LOG_EVERY_TOKENS:
                dt = time.time() - t0
                st = brain.stats()
                log_and_print('[ep %2d] 已学 %6d tok | trainPPL %7.3f acc %.2f | N %d+%d | syn %d coinc %d | '
                              '出生 %d 死亡 %d | G %8.1f | %5.0f tok/s' % (
                                  epoch, step_count, math.exp(nll_acc / max(n_acc, 1)), hit_acc / max(n_acc, 1),
                                  st['V'] * 2, st['ints'],                                   st['syn'], st['coinc'], st['births'], st['deaths'], st['G'],
                                  tok_win / max(dt, 1e-6)))
                step_count = 0
                tok_win = 0
                nll_acc = 0.0
                n_acc = 0
                hit_acc = 0
                t0 = time.time()
            if TRAIN_TOKENS_PER_EPOCH and tok_count >= TRAIN_TOKENS_PER_EPOCH:
                break
        if EPOCH_END_REPLAY and brain.recent:
            brain.replay()
        val_ppl, val_n, val_acc = evaluate_val(brain, tokenizer, val_recs, is_txt=is_txt)
        st = brain.stats()
        log_and_print('[ep %2d] 完成 | 跳过 %d 条 | VAL PPL %8.3f acc %.2f (n=%d) | N %d+%d | syn %d coinc %d | '
                      '出生 %d 死亡 %d' % (
                          epoch, skipped, val_ppl, val_acc, val_n, st['V'] * 2, st['ints'], st['syn'], st['coinc'],
                          st['births'], st['deaths']))
        if save_path:
            brain.save(save_path)
            log_and_print(f"检查点已保存: {save_path}")
            best_path = save_path.replace('.pt', '_best.pt') if '.pt' in save_path else save_path + '.best'
            global _BEST_PPL
            if val_ppl < _BEST_PPL:
                _BEST_PPL = val_ppl
                brain.save(best_path)
                log_and_print(f"★ 新最佳 VAL PPL {val_ppl:.3f}，快照已保存: {best_path}")
    return val_ppl

# ================================================================================================
# 对话（外壳沿用；生成由生长脑流式采样）
# ================================================================================================
def chat_loop(brain, tokenizer, mode='jsonl'):
    log_and_print("输入 exit 退出，reset 重置上下文。")
    if mode == 'txt':
        context_ids = []
        while True:
            user_input = input("\n输入: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                context_ids = []
                log_and_print("上下文已重置。")
                continue
            context_ids.extend(tokenizer.encode(user_input))
            context_ids = context_ids[-CHAT_MAX_HISTORY_TOKENS:]
            brain.reset_state()
            _sdec = StreamDecoder(tokenizer)
            gen_rng = random.Random()
            gen_ids = brain.generate(context_ids, GEN_MAX_TOKENS, rng=gen_rng,
                                     temperature=TEMPERATURE, top_p=TOP_P,
                                     stream=lambda t: print(_sdec.push(t), end="", flush=True),
                                     stop_ids=(tokenizer.eos_token_id, tokenizer.end_token_id))
            tail = _sdec.flush()
            if tail:
                print(tail, end="", flush=True)
            context_ids.extend(gen_ids)
            context_ids = context_ids[-CHAT_MAX_HISTORY_TOKENS:]
            print()
    else:
        history = []
        while True:
            user_input = input("\n用户: ")
            if user_input.lower() == 'exit':
                break
            if user_input.lower() == 'reset':
                history = []
                log_and_print("上下文已重置。")
                continue
            history.append({'role': 'user', 'content': user_input})
            text, _spans = render_messages(history, add_generation_prompt=True)
            ids, _offs = _encode_with_offsets(tokenizer, text)
            ids = ids[-CHAT_MAX_HISTORY_TOKENS:]
            brain.reset_state()
            print("助手: ", end="", flush=True)
            _sdec = StreamDecoder(tokenizer)
            gen_rng = random.Random()
            gen_ids = brain.generate(ids, GEN_MAX_TOKENS, rng=gen_rng,
                                     temperature=TEMPERATURE, top_p=TOP_P,
                                     stream=lambda t: print(_sdec.push(t), end="", flush=True),
                                     stop_ids=(tokenizer.eos_token_id, tokenizer.end_token_id))
            tail = _sdec.flush()
            if tail:
                print(tail, end="", flush=True)
            print()
            response = tokenizer.decode(gen_ids)
            history.append({'role': 'assistant', 'content': response})
            if len(history) > 12:
                history = history[-12:]

# ================================================================================================
# 规则奖励适配：生成 / 奖励调制重放（试错巩固、组内相对重放）
# 全部为局部规则 + 睡眠重放的组合——没有梯度回传，没有外部监督注入（奖励只调制已有经历的重放强度）。
# ================================================================================================
def generate_one(brain, tokenizer, prompt_ids, max_new_tokens):
    brain.reset_state()
    gen_rng = random.Random()
    gen_ids = brain.generate(prompt_ids, max_new_tokens, rng=gen_rng,
                             temperature=TEMPERATURE, top_p=TOP_P,
                             stop_ids=(tokenizer.eos_token_id, tokenizer.end_token_id))
    return gen_ids, tokenizer.decode(gen_ids)


def learn_episode(brain, tokenizer, prompt, response, bonus_sweeps=0):
    """奖励事件巩固：把 (prompt → response) 整段重新入网学习；正确经历获得额外巩固遍数。"""
    msgs = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    text, spans = render_messages(msgs)
    ids, offs = _encode_with_offsets(tokenizer, text)
    mask = _mask_from_spans(ids, offs, spans)
    if not mask.any():
        return
    brain.run_tokens(ids, learn=True, mask=mask, score=False, reset=True)
    for _ in range(max(0, bonus_sweeps)):
        brain.run_tokens(ids, learn=True, mask=mask, score=False, reset=True)
    if brain.recent:
        brain.replay()


def rejection_sampling(brain, tokenizer, problem_path, output_path, samples_per_question, max_new_tokens):
    if os.path.abspath(problem_path) == os.path.abspath(output_path):
        raise ValueError("输出文件不能覆盖输入题库。")
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    kept = 0
    tried = 0
    seen = set()
    with open(output_path, 'w', encoding='utf-8') as out:
        for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
            prompt = ultra_make_prompt(problem)
            pids = ultra_prompt_ids(tokenizer, prompt)
            local_kept = 0
            for _ in range(samples_per_question):
                _gen_ids, response = generate_one(brain, tokenizer, pids, max_new_tokens)
                tried += 1
                reward, predicted = ultra_rule_reward(response, problem)
                key = (problem['line_no'], response)
                if reward > 0 and key not in seen:
                    ultra_write_sft_line(out, prompt, response, {
                        'reward': float(reward), 'predicted': predicted, 'from': 'rs',
                        'line_no': problem['line_no']})
                    seen.add(key)
                    kept += 1
                    local_kept += 1
            log_and_print(f"[RS {index}/{total_problems}] 正确保留 {local_kept}/{samples_per_question}")
    log_and_print(f"拒绝采样完成：尝试 {tried} 个回答，保留 {kept} 个正确回答，输出至 {output_path}")


def trial_replay_training(brain, tokenizer, problem_path, output_path, samples_per_question, max_new_tokens, bonus_sweeps):
    """试错巩固（菜单 8）：正确回答 → 写入 SFT + 奖励调制重放；错误回答 → 不加固。"""
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    solved = 0
    tried = 0
    with open(output_path, 'a', encoding='utf-8') as out:
        for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
            prompt = ultra_make_prompt(problem)
            pids = ultra_prompt_ids(tokenizer, prompt)
            local_ok = 0
            for _ in range(samples_per_question):
                _gen_ids, response = generate_one(brain, tokenizer, pids, max_new_tokens)
                tried += 1
                reward, predicted = ultra_rule_reward(response, problem)
                if reward > 0:
                    local_ok += 1
                    ultra_write_sft_line(out, prompt, response, {
                        'reward': float(reward), 'predicted': predicted, 'from': 'trial_rl',
                        'line_no': problem['line_no']})
                    learn_episode(brain, tokenizer, prompt, response, bonus_sweeps=bonus_sweeps)
            if local_ok > 0:
                solved += 1
            log_and_print(f"[试错 {index}/{total_problems}] 本题正确 {local_ok}/{samples_per_question} | 已解 {solved} 题")
    if brain.recent:
        brain.replay()
    log_and_print(f"试错巩固完成：尝试 {tried} 次，解出 {solved}/{total_problems} 题。强化样本写入 {output_path}")


def relative_replay_training(brain, tokenizer, problem_path, output_path, group_size, max_new_tokens, bonus_sweeps):
    """组内相对重放（菜单 9，GRPO 的免梯度对应版）：同题采样一组，只对最优且正确者做强化重放。"""
    total_problems = ultra_count_problems(problem_path)
    if total_problems == 0:
        log_and_print("题库没有可用题目。")
        return
    solved = 0
    with open(output_path, 'a', encoding='utf-8') as out:
        for index, problem in enumerate(ultra_iter_problems(problem_path), 1):
            prompt = ultra_make_prompt(problem)
            pids = ultra_prompt_ids(tokenizer, prompt)
            group = []
            for _ in range(group_size):
                _gen_ids, response = generate_one(brain, tokenizer, pids, max_new_tokens)
                reward, predicted = ultra_rule_reward(response, problem)
                group.append((reward, response, predicted))
            group.sort(key=lambda t: -t[0])
            best_reward, best_response, best_predicted = group[0]
            if best_reward > 0:
                solved += 1
                ultra_write_sft_line(out, prompt, best_response, {
                    'reward': float(best_reward), 'predicted': best_predicted, 'from': 'relative',
                    'group_size': group_size, 'line_no': problem['line_no']})
                learn_episode(brain, tokenizer, prompt, best_response, bonus_sweeps=bonus_sweeps)
            n_correct = sum(1 for g in group if g[0] > 0)
            log_and_print(f"[相对 {index}/{total_problems}] 组内正确 {n_correct}/{group_size} | "
                          f"最优强化: {'是' if best_reward > 0 else '否'} | 已解 {solved} 题")
    if brain.recent:
        brain.replay()
    log_and_print(f"组内相对重放完成：已解 {solved}/{total_problems} 题。样本写入 {output_path}")

def select_training_file():
    """扫描当前目录下 .txt 和 .jsonl 文件，让用户选择数据集文件。"""
    files = []
    for f in sorted(os.listdir('.')):
        if f.startswith('.'):
            continue
        if f.endswith('.txt') or f.endswith('.jsonl'):
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            files.append((f, size))
    if not files:
        log_and_print("当前目录下未找到 .txt 或 .jsonl 文件。")
        return None, None
    print("\n可用数据集文件：")
    for i, (name, size) in enumerate(files, 1):
        if size >= 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {i}. {name} ({size_str})")
    while True:
        raw = input("请输入文件序号: ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("输入无效，请输入数字。")
            continue
        if idx < 1 or idx > len(files):
            print(f"序号超出范围，请输入 1-{len(files)}。")
            continue
        selected, _ = files[idx - 1]
        ext = os.path.splitext(selected)[1].lower()
        log_and_print(f"已选择文件: {selected}")
        return selected, ext

def ultra_read_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的整数。")


def ultra_read_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"请输入不小于 {minimum} 的数字。")


def ultra_select_problem_file():
    files = []
    for name in sorted(os.listdir('.')):
        if name.startswith('.') or not name.endswith('.jsonl'):
            continue
        try:
            size = os.path.getsize(name)
        except OSError:
            size = 0
        files.append((name, size))
    if not files:
        log_and_print("当前目录没有可用的 .jsonl 题库。")
        return None
    print("\n可用 JSONL 题库：")
    for i, (name, size) in enumerate(files, 1):
        print(f"  {i}. {name} ({size / 1024:.1f} KB)")
    while True:
        raw = input("请输入题库序号: ").strip()
        try:
            index = int(raw)
            if 1 <= index <= len(files):
                return files[index - 1][0]
        except ValueError:
            pass
        print(f"请输入 1-{len(files)}。")


def ultra_iter_problems(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                log_and_print(f"跳过题库第 {line_no} 行：JSON 解析失败: {e}")
                continue
            msgs = _messages_from_record(item)
            question = '\n'.join(str(m.get('content', '')) for m in msgs
                                 if m.get('role') in ('user', 'system'))
            answers = item.get('answers', item.get('answer'))
            if not question.strip() or answers is None:
                log_and_print(f"跳过题库第 {line_no} 行：v5 题库需要 messages 与 answer/answers。")
                continue
            if not isinstance(answers, list):
                answers = [answers]
            answers = [str(answer) for answer in answers if str(answer).strip()]
            if not answers:
                log_and_print(f"跳过题库第 {line_no} 行：没有可用答案。")
                continue
            yield {
                'question': question,
                'messages': msgs or None,
                'answers': answers,
                'type': str(item.get('type', 'auto')).lower(),
                'tools': item.get('tools'),
                'line_no': line_no,
            }


def ultra_count_problems(path: str) -> int:
    n = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ultra_iter_problems_shuffled(path: str, buffer_size: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    buf = []
    for p in ultra_iter_problems(path):
        buf.append(p)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            while buf:
                yield buf.pop()
    if buf:
        rng.shuffle(buf)
        while buf:
            yield buf.pop()


def ultra_infer_answer_type(problem) -> str:
    import re
    answer_type = problem.get('type', 'auto')
    if answer_type != 'auto':
        return answer_type
    answers = [answer.strip() for answer in problem['answers']]
    if answers and all(re.fullmatch(r'[A-Za-z]', answer) for answer in answers):
        return 'choice'
    if answers and all(re.fullmatch(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', answer.replace(',', ''))
                       for answer in answers):
        return 'number'
    return 'text'


def ultra_normalize_answer(value: str, answer_type: str) -> str:
    import re
    import unicodedata
    from decimal import Decimal, InvalidOperation
    from fractions import Fraction

    text = unicodedata.normalize('NFKC', str(value)).strip()
    if answer_type == 'choice':
        choices = re.findall(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])', text)
        return choices[-1].upper() if choices else text.upper()
    if answer_type == 'number':
        matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?', text.replace(',', ''))
        if matches:
            raw = matches[-1]
            try:
                if '/' in raw:
                    left, right = raw.split('/', 1)
                    value_fraction = Fraction(Decimal(left)) / Fraction(Decimal(right))
                else:
                    value_fraction = Fraction(Decimal(raw))
                return f"{value_fraction.numerator}/{value_fraction.denominator}"
            except (InvalidOperation, ZeroDivisionError):
                pass
    text = re.sub(r'\s+', '', text.lower())
    return text.strip('。.!！?？,，;；:：')


def ultra_extract_final_answer(response: str) -> str:
    import re
    boxed = re.findall(r'\\boxed\s*\{([^{}]+)\}', response)
    if boxed:
        return boxed[-1].strip()
    marked = re.findall(r'(?:最终答案|答案|final\s+answer)\s*[:：]\s*([^\r\n]+)', response,
                        flags=re.IGNORECASE)
    if marked:
        return marked[-1].strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ''


def ultra_rule_reward(response: str, problem):
    _reasoning, body = _split_reasoning(response)      # 只看正文，思考内容不参与判分
    predicted = ultra_extract_final_answer(body)
    answer_type = ultra_infer_answer_type(problem)
    predicted_normalized = ultra_normalize_answer(predicted, answer_type)
    accepted = {ultra_normalize_answer(answer, answer_type) for answer in problem['answers']}
    reward = 1.0 if predicted_normalized in accepted else 0.0
    return reward, predicted


def ultra_make_prompt(problem) -> str:
    return f"{problem['question']}{RL_PROMPT_SUFFIX}"


def ultra_prompt_ids(tokenizer: FastBPE, prompt: str):
    msgs = []
    if RL_SYSTEM_PROMPT.strip():
        msgs.append({'role': 'system', 'content': RL_SYSTEM_PROMPT})
    msgs.append({'role': 'user', 'content': prompt})
    text, _spans = render_messages(msgs, add_generation_prompt=True)
    ids, _offs = _encode_with_offsets(tokenizer, text)
    return ids

def ultra_write_sft_line(f, prompt: str, response: str, meta: dict = None, reasoning_content: str = ''):
    """写一条 SFT 样本（v5：messages + meta，可被本文件菜单 1 直接读回）。"""
    assistant = {'role': 'assistant', 'content': response}
    if reasoning_content:
        assistant['reasoning_content'] = reasoning_content
    record = {'messages': [{'role': 'user', 'content': prompt}, assistant]}
    if meta:
        record['meta'] = dict(meta)
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    f.flush()

def p4_segment_dataset(input_path, output_path, tokenizer, threshold):
    """P4: 超长对话切分。

    规则：
      1. 解析每条对话的 conversations（多轮）
      2. 按 (user, assistant) 对切分
      3. 单对 token 长度 <= threshold → 保留为独立样本
      4. 单对 token 长度 >  threshold → 丢弃该对（不截断，不学半句）
      5. 一条多轮对话的所有对都超长 → 整条丢弃
      6. 不截断、不重训 tokenizer（只用现有词表切）

    返回统计 dict。
    """
    stats = {
        'total_lines': 0,
        'kept_samples': 0,
        'dropped_pairs': 0,
        'dropped_lines': 0,
        'kept_lines': 0,
    }

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats['total_lines'] += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            convs = _messages_from_record(data)
            if not convs:
                continue

            # 开头的连续 system 段始终保留（与训练路径 _truncate_pairs 语义一致）：
            # 计入 token 预算后拼到每条切分样本前，避免切分后 system 整段丢失。
            head = []
            head_tokens = 0
            start = 0
            while start < len(convs) and convs[start].get('role') == 'system':
                m = convs[start]
                ids = tokenizer.encode(str(m.get('content') or ''))
                head.append(m)
                head_tokens += len(ids) + 2
                start += 1

            # 按 (user, assistant) 对切分
            pairs = []
            i = 0
            while i < len(convs) - 1:
                if convs[i].get('role') == 'user' and convs[i + 1].get('role') == 'assistant':
                    pairs.append((convs[i], convs[i + 1]))
                    i += 2
                else:
                    i += 1

            if not pairs:
                continue

            line_kept = 0
            for pair in pairs:
                pair_tokens = head_tokens
                for turn in pair:
                    content = turn.get('content', '')
                    ids = tokenizer.encode(content)
                    pair_tokens += len(ids) + 2      # 近似：role token + end token 开销
                    if TRAIN_ON_REASONING and turn.get('role') == 'assistant':
                        rc = turn.get('reasoning_content')
                        if rc:
                            pair_tokens += len(tokenizer.encode(str(rc)))
                    if turn.get('role') == 'assistant':
                        tc = _render_tool_calls(turn.get('tool_calls'))
                        if tc:
                            pair_tokens += len(tokenizer.encode(tc))
                # 与真实样本长度对齐：input_ids = tokens[:-1]（tokens = [BOS] + 各轮 + [EOS]），
                # 故真实长度 = Σ(len(ids)+2) + 1。不加这 1 会让边界样本多留一条。
                pair_tokens += 1

                if pair_tokens <= threshold:
                    record = {'messages': head + list(pair)}
                    fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats['kept_samples'] += 1
                    line_kept += 1
                else:
                    stats['dropped_pairs'] += 1

            if line_kept > 0:
                stats['kept_lines'] += 1
            else:
                stats['dropped_lines'] += 1

    log_and_print("=== P4 数据切分统计 ===")
    log_and_print(f"原行数: {stats['total_lines']}")
    log_and_print(f"保留样本数: {stats['kept_samples']}")
    log_and_print(f"保留的原行数: {stats['kept_lines']}")
    log_and_print(f"完全丢弃的行数: {stats['dropped_lines']}")
    log_and_print(f"丢弃的对话对数: {stats['dropped_pairs']}")
    log_and_print(f"输出文件: {output_path}")
    log_and_print(f"阈值: {threshold} tokens")
    return stats

def p4_segment_dataset_menu():
    """主菜单 11：数据预处理，切分超长对话。"""
    input_path = input("输入 jsonl 文件（默认 dialogues.jsonl）: ").strip() or 'dialogues.jsonl'
    if not os.path.exists(input_path):
        log_and_print(f"文件不存在: {input_path}")
        return

    default_out = input_path.replace('.jsonl', '_seg.jsonl')
    output_path = input("输出 jsonl 文件（默认 %s）: " % default_out).strip() or default_out
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        log_and_print("输入输出不能是同一文件（避免覆盖）")
        return

    try:
        threshold_str = input(f"单样本 token 上限（默认 {P4_SEG_MAX_TOKENS}）: ").strip()
        threshold = int(threshold_str) if threshold_str else P4_SEG_MAX_TOKENS
    except ValueError:
        threshold = P4_SEG_MAX_TOKENS

    tokenizer = FastBPE(VOCAB_SIZE)
    if os.path.exists("tokenizer_incremental.bin"):
        if not tokenizer.load("tokenizer_incremental.bin"):
            log_and_print("加载 tokenizer 失败")
            return
    else:
        log_and_print("未找到 tokenizer_incremental.bin，无法切分")
        return

    p4_segment_dataset(input_path, output_path, tokenizer, threshold)

# ================================================================================================
# 主菜单 + 命令行入口（--smoke 自检 / --train 无头训练）
# ================================================================================================
def _ensure_brain(state, tokenizer):
    if state['brain'] is None:
        V = tokenizer.vocab_size if tokenizer._tokenizer is not None else VOCAB_SIZE
        state['brain'] = SproutBrain(V, seed=1)
        log_and_print(f"初始化新生长脑：V={V}, N={state['brain'].N}, 内部神经元容量={CAP_INT}（实际数量由能量均衡涌现）")
        if not state['loaded'] and os.path.exists(_ckpt_file(CHECKPOINT)):
            try:
                state['brain'].load(CHECKPOINT)
                state['loaded'] = True
                log_and_print(f"已加载 {CHECKPOINT}")
            except Exception as e:
                log_and_print(f"加载 {CHECKPOINT} 失败: {e}（使用当前内存中的脑）")
    return state['brain']


def _train_tokenizer_from(data_path, tokenizer):
    log_and_print("准备训练新分词器（ByteLevel BPE）...")
    texts = []
    if data_path.lower().endswith('.jsonl'):
        for line in load_lines(data_path):
            try:
                rec = json.loads(line)
                msgs = _messages_from_record(rec)
            except Exception:
                continue
            for turn in msgs:
                texts.append(normalize_text(str(turn.get('content') or '')))
                rc = turn.get('reasoning_content')
                if TRAIN_ON_REASONING and rc:
                    texts.append(normalize_text(str(rc)))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(normalize_text(line))
    if not texts:
        texts = ['']
    char_set = set(''.join(texts))
    log_and_print(f"语料含 {len(char_set)} 个不同字符，VOCAB_SIZE={VOCAB_SIZE}")
    tokenizer.train(texts, verbose=False)
    tokenizer.save(TOKENIZER_CHECKPOINT)
    return tokenizer


def main():
    global TEMPERATURE, TOP_P
    init_log_file()
    log_and_print(f"{SPROUT_PROJECT} · {SPROUT_BRANCH} · {SPROUT_VERSION} | 内核：能量预算生长脑（局部规则/无 BP/零阈值）")
    tokenizer = FastBPE(VOCAB_SIZE)
    if os.path.exists(TOKENIZER_CHECKPOINT):
        if not tokenizer.load(TOKENIZER_CHECKPOINT):
            log_and_print("词表加载失败（旧格式？），将在首次训练时重训。")
    else:
        log_and_print(f"未找到 {TOKENIZER_CHECKPOINT}，将在首次训练时从所选数据训练词表。")

    state = {'brain': None, 'loaded': False}

    while True:
        print(f"\n=== {SPROUT_PROJECT}（{SPROUT_BRANCH} {SPROUT_VERSION}）训练与对话系统菜单 ===")
        print("1. 训练模型（继续训练）")
        print("2. 对话模式")
        print("3. 保存模型")
        print("4. 加载模型")
        print("5. 调整生成参数")
        print("6. 查看模型统计信息")
        print("7. 规则验证拒绝采样：生成正确回答 SFT 数据")
        print("8. 基础规则奖励 RL：单回答试错巩固（奖励调制重放）")
        print("9. 组内相对重放：同题分组相对强化")
        print("10. 退出")
        print("11. 数据预处理：超长对话切分（多轮切单轮 / 单轮超长丢弃）")
        choice = input("请选择: ").strip()

        if choice == '1':
            data_path, file_ext = select_training_file()
            if data_path is None:
                continue
            if tokenizer._tokenizer is None:
                _train_tokenizer_from(data_path, tokenizer)
            brain = _ensure_brain(state, tokenizer)
            if input("开始训练？(y/n): ").lower() == 'y':
                train_sprout(brain, tokenizer, data_path)
        elif choice == '2':
            if tokenizer._tokenizer is None:
                log_and_print("尚未训练/加载分词器，无法对话。")
                continue
            brain = _ensure_brain(state, tokenizer)
            chat_mode = input("对话格式 (jsonl) 还是纯文本续写 (txt)？[jsonl]: ").strip().lower()
            if chat_mode not in ("jsonl", "txt"):
                chat_mode = "jsonl"
            chat_loop(brain, tokenizer, mode=chat_mode)
        elif choice == '3':
            brain = _ensure_brain(state, tokenizer)
            try:
                brain.save(CHECKPOINT)
                log_and_print(f"已保存至 {CHECKPOINT}")
            except Exception as e:
                log_and_print(f"保存失败: {e}")
        elif choice == '4':
            brain = _ensure_brain(state, tokenizer)
            try:
                brain.load(CHECKPOINT)
                state['loaded'] = True
                log_and_print("模型加载成功。")
            except Exception as e:
                log_and_print(f"加载失败: {e}")
        elif choice == '5':
            try:
                TEMPERATURE = float(input(f"temperature (0.1-1.5) [当前 {TEMPERATURE:.2f}]: ") or TEMPERATURE)
                TOP_P = float(input(f"top_p (0.0-1.0) [当前 {TOP_P:.2f}]: ") or TOP_P)
            except ValueError:
                log_and_print("输入无效，保持原值")
        elif choice == '6':
            brain = _ensure_brain(state, tokenizer)
            st = brain.stats()
            log_and_print(f"词表规模: {st['V']} | 神经元槽位: {st['N']}（感受器 {2 * st['V']} + 内部 {st['cap_int']}）")
            log_and_print(f"存活内部神经元: {st['ints']}（峰值 {st['max_ints']}，红线上限 {st['cap_int']}）")
            log_and_print(f"突触: {st['syn']}/{st['cap_syn']} | 重合配对: {st['coinc']}/{st['cap_coinc']}（扇出 {st['fan']}）")
            log_and_print(f"出生 {st['births']} | 死亡 {st['deaths']} | 生长事件 {st['spines']} | 时钟 {st['clock']}")
            log_and_print(f"能量池 G: {st['G']:.1f} | 生成参数: T={TEMPERATURE:.2f} top_p={TOP_P:.2f}")
            log_and_print("架构：能量预算驱动自生长神经网（多尺度水库 + 重合二次特征 + 睡眠重放；局部规则、无 BP）")
        elif choice == '7':
            brain = _ensure_brain(state, tokenizer)
            problem_path = ultra_select_problem_file()
            if problem_path is None:
                continue
            out_path = input("输出 SFT 文件（默认 rs_sft.jsonl）: ").strip() or 'rs_sft.jsonl'
            n = ultra_read_int("每题采样次数", RS_SAMPLES_PER_Q)
            m = ultra_read_int("最大生成 token", GEN_MAX_TOKENS)
            rejection_sampling(brain, tokenizer, problem_path, out_path, n, m)
        elif choice == '8':
            brain = _ensure_brain(state, tokenizer)
            problem_path = ultra_select_problem_file()
            if problem_path is None:
                continue
            out_path = input("输出强化样本文件（默认 trial_rl.jsonl）: ").strip() or 'trial_rl.jsonl'
            n = ultra_read_int("每题尝试次数", RL_SAMPLES_PER_Q)
            m = ultra_read_int("最大生成 token", GEN_MAX_TOKENS)
            k = ultra_read_int("奖励事件额外巩固遍数", REWARD_REPLAY_BONUS, minimum=0)
            trial_replay_training(brain, tokenizer, problem_path, out_path, n, m, k)
        elif choice == '9':
            brain = _ensure_brain(state, tokenizer)
            problem_path = ultra_select_problem_file()
            if problem_path is None:
                continue
            out_path = input("输出强化样本文件（默认 relative_rl.jsonl）: ").strip() or 'relative_rl.jsonl'
            g = ultra_read_int("组内采样数", GRPO_GROUP)
            m = ultra_read_int("最大生成 token", GEN_MAX_TOKENS)
            k = ultra_read_int("胜者额外巩固遍数", REWARD_REPLAY_BONUS, minimum=0)
            relative_replay_training(brain, tokenizer, problem_path, out_path, g, m, k)
        elif choice == '11':
            p4_segment_dataset_menu()
        elif choice == '10':
            break
        else:
            log_and_print("无效选择，请重新输入")
    close_log_file()


def smoke_test(data_path):
    """--smoke：非交互快速自检（tokenizer → 掩码训练 → 验证 → 生成）。"""
    init_log_file()
    log_and_print(f"[SMOKE] {SPROUT_PROJECT} {SPROUT_BRANCH} {SPROUT_VERSION}")
    lines = load_lines(data_path)
    tok_lines = lines
    tokenizer = FastBPE(VOCAB_SIZE)
    if os.path.exists(TOKENIZER_CHECKPOINT):
        tokenizer.load(TOKENIZER_CHECKPOINT)
    if tokenizer._tokenizer is None:
        texts = []
        for line in tok_lines:
            try:
                msgs = _messages_from_record(json.loads(line))
            except Exception:
                continue
            for turn in msgs:
                texts.append(normalize_text(str(turn.get('content') or '')))
        tokenizer.train(texts or [''], verbose=False)
        tokenizer.save(TOKENIZER_CHECKPOINT)
    log_and_print(f"[SMOKE] 词表规模 {tokenizer.vocab_size}")
    r = render_record(tokenizer, lines[0])
    if r is not None:
        ids, mask = r
        log_and_print(f"[SMOKE] 记录1: tokens={len(ids)} | assistant(mask) tokens={int(mask.sum())} | 前12 ids={ids[:12]}")
    brain = SproutBrain(tokenizer.vocab_size, seed=1)
    train_recs = [(i + 1, l) for i, l in enumerate(lines[:24])]
    log_and_print(f"[SMOKE] 小规模训练 {len(train_recs)} 条 × 3 epoch ...")
    for ep in range(1, 4):
        nll = 0.0
        n = 0
        hits = 0
        for line_no, line in train_recs:
            rr = render_record(tokenizer, line)
            if rr is None:
                continue
            ids, mask = rr
            if not mask.any():
                continue
            ppl, nn, h = brain.run_tokens(ids, learn=True, mask=mask, score=True, reset=True)
            if nn:
                nll += math.log(ppl) * nn
                n += nn
                hits += h
        brain.replay()
        st = brain.stats()
        log_and_print('[SMOKE ep %d] trainPPL %.3f acc %.2f | N %d+%d | syn %d coinc %d | 出生 %d | G %.1f' % (
            ep, math.exp(nll / max(n, 1)), hits / max(n, 1), 2 * st['V'], st['ints'],
            st['syn'], st['coinc'], st['births'], st['G']))
    val_recs = [(i + 25, l) for i, l in enumerate(lines[24:36])]
    vppl, vn, vacc = evaluate_val(brain, tokenizer, val_recs)
    log_and_print(f"[SMOKE] VAL PPL {vppl:.3f} acc {vacc:.2f} (n={vn})")
    prompt_msgs = [{'role': 'user', 'content': '你好，请介绍一下你自己。'}]
    text, _s = render_messages(prompt_msgs, add_generation_prompt=True)
    ids, _offs = _encode_with_offsets(tokenizer, text)
    gen_ids = brain.generate(ids, 64, rng=random.Random(7), temperature=0.7, top_p=0.9)
    log_and_print(f"[SMOKE] 生成示例: {tokenizer.decode(gen_ids)[:200]!r}")
    log_and_print("[SMOKE] 通过。")


def _cli():
    global EPOCHS, TRAIN_TOKENS_PER_EPOCH, VAL_BUCKET, DATA_RECORDS_LIMIT, CHECKPOINT
    args = sys.argv[1:]
    if '--smoke' in args:
        data_path = 'dsr1_v5_0001.jsonl'
        for i, a in enumerate(args):
            if a == '--smoke' and i + 1 < len(args) and not args[i + 1].startswith('--'):
                data_path = args[i + 1]
        if not os.path.exists(data_path):
            print(f"找不到数据文件: {data_path}")
            return
        smoke_test(data_path)
        return
    if '--topology' in args:
        for i, a in enumerate(args):
            if a == '--ckpt' and i + 1 < len(args):
                CHECKPOINT = args[i + 1]
        tokenizer = FastBPE(VOCAB_SIZE)
        if os.path.exists(TOKENIZER_CHECKPOINT):
            tokenizer.load(TOKENIZER_CHECKPOINT)
        if tokenizer._tokenizer is None:
            print('未找到分词器，无法统计拓扑')
            return
        brain = SproutBrain(tokenizer.vocab_size, seed=1)
        if not os.path.exists(_ckpt_file(CHECKPOINT)):
            print(f"未找到检查点 {CHECKPOINT}")
            return
        try:
            brain.load(CHECKPOINT)
        except Exception as e:
            print(f"加载 {CHECKPOINT} 失败: {e}")
            return
        print(f"已加载 {CHECKPOINT}，统计生长拓扑：")
        brain.topology_report()
        return
    if '--train' in args:
        data_path = None
        epochs = EPOCHS
        tokens = TRAIN_TOKENS_PER_EPOCH
        for i, a in enumerate(args):
            if a == '--train' and i + 1 < len(args):
                data_path = args[i + 1]
            if a == '--epochs' and i + 1 < len(args):
                epochs = int(args[i + 1])
            if a == '--tokens' and i + 1 < len(args):
                tokens = int(args[i + 1])
            if a == '--val-bucket' and i + 1 < len(args):
                VAL_BUCKET = int(args[i + 1])
            if a == '--records' and i + 1 < len(args):
                DATA_RECORDS_LIMIT = int(args[i + 1])
            if a == '--ckpt' and i + 1 < len(args):
                CHECKPOINT = args[i + 1]
        if not data_path or not os.path.exists(data_path):
            print(f"用法: python {os.path.basename(__file__)} --train <file> [--epochs N] [--tokens N] [--val-bucket N]")
            return
        EPOCHS = epochs
        TRAIN_TOKENS_PER_EPOCH = tokens
        init_log_file()
        log_and_print(f"{SPROUT_PROJECT} · {SPROUT_BRANCH} · {SPROUT_VERSION} | --train {data_path} epochs={EPOCHS} tokens/ep={TRAIN_TOKENS_PER_EPOCH} records={DATA_RECORDS_LIMIT}")
        tokenizer = FastBPE(VOCAB_SIZE)
        if os.path.exists(TOKENIZER_CHECKPOINT):
            tokenizer.load(TOKENIZER_CHECKPOINT)
        if tokenizer._tokenizer is None:
            _train_tokenizer_from(data_path, tokenizer)
        brain = SproutBrain(tokenizer.vocab_size, seed=1)
        if os.path.exists(_ckpt_file(CHECKPOINT)):
            try:
                brain.load(CHECKPOINT)
                log_and_print(f"已加载 {CHECKPOINT}，继续训练")
            except Exception as e:
                log_and_print(f"加载 {CHECKPOINT} 失败: {e}（丢弃半加载状态，重置为全新脑）")
                brain = SproutBrain(tokenizer.vocab_size, seed=1)
        train_sprout(brain, tokenizer, data_path, epochs=EPOCHS, save_path=CHECKPOINT)
        gen_msgs = [{'role': 'user', 'content': '你好，请介绍一下你自己。'}]
        text, _s = render_messages(gen_msgs, add_generation_prompt=True)
        ids, _offs = _encode_with_offsets(tokenizer, text)
        gen_ids = brain.generate(ids, 128, rng=random.Random(7), temperature=TEMPERATURE, top_p=TOP_P)
        log_and_print(f"生成示例: {tokenizer.decode(gen_ids)[:300]!r}")
        close_log_file()
        return
    main()


if __name__ == "__main__":
    _cli()

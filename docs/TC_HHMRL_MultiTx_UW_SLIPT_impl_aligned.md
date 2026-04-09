# 面向多发射端水下 SLIPT 的任务条件约束分层元安全混合强化学习方法

## 1 引言与问题定位
本文面向链路级水下光无线同步信息与能量传输（SLIPT）场景，研究的是多发射端异构光链路中的结构选择、连续资源分配与安全约束控制问题，而不是网络层多节点路由问题。系统由一个接收/能量收集端和三个候选发射端组成，其中一路为宽束 Anchor 发射端，两路为窄束 Boost 发射端。本文关注的核心问题是：在水体衰减、失准、热压力和时变扰动共同存在的条件下，如何通过分层强化学习、执行侧安全投影和任务条件适应机制，实现频谱效率（SE）、能量采集（EH）和安全稳定性的联合优化。

本文采用 Physics-Aware Link-Level Model，即在链路级别融合几何扩散、水体衰减、失准、突发扰动、器件异构与热约束因素的物理感知代理建模。建模重点是验证“固定异构多发射端结构 + 分层控制 + 长期约束 + 执行侧安全投影 + 任务条件适应”的控制框架，而不是建立器件级热光耦合、严格接收机非线性和全物理参数标定的高精度模型。

### 1.1 任务分布上的总体优化目标
记所有可学习参数为
\[
\Theta = \{\psi,\phi_1,\phi_2,\theta,\varphi\},
\]
其中 \(\psi\) 表示上层 DQN 参数，\(\phi_1,\phi_2\) 表示下层 twin-critic 参数，\(\theta\) 表示下层 actor 参数，\(\varphi\) 表示上下文编码与任务预测相关参数。设任务 \(\omega\) 从任务分布 \(p(\omega)\) 中采样，单个 episode 长度为 \(H\)，则本文的总体优化目标写为
\[
\max_{\Theta}\;
\mathbb{E}_{\omega\sim p(\omega)}
\left[
\mathbb{E}_{\pi_\Theta}
\left[
\sum_{t=0}^{H-1} r_t(\omega)
\right]
\right]
\]
\[
\text{s.t.}\quad
\mathbb{E}_{\omega\sim p(\omega)}
\left[
\mathbb{E}_{\pi_\Theta}
\left[
\sum_{t=0}^{H-1} c_{k,t}(\omega)
\right]
\right]
\le b_k,\qquad k=1,\dots,K,
\]
其中 \(c_{k,t}\) 为第 \(k\) 个瞬时约束分量，\(b_k\) 为对应的长期约束预算，本文取 \(K=4\)，分别对应 QoS 约束与三个发射端的热约束。训练阶段通过向量拉格朗日对偶层将上述长期约束转化为可优化的惩罚项。

### 1.2 状态、动作、奖励与约束的统一定义
为便于后续公式推导，本文统一采用表 1 中的记号。

| 记号 | 含义 | 维度/取值 |
|---|---|---|
| \(s_t\) | 环境观测状态 | \(20\) 维 |
| \(z_t\) | 任务条件潜变量 | \(8\) 维 |
| \(a_t^{U,\mathrm{raw}}\) | 上层 raw 宏动作 | \(u_t^{\mathrm{raw}}\in\{0,1,2,3\},\; m_t^{\mathrm{raw}}\in\{0,1,2\}\) |
| \(a_t^{U,\mathrm{exec}}\) | 上层执行宏动作 | \(u_t^{\mathrm{exec}}, m_t^{\mathrm{exec}}\) |
| \(a_t^{L,\mathrm{raw}}\) | 下层 raw 连续动作 | \([\hat I_{0,t},\hat I_{1,t},\hat I_{2,t},\hat\rho_t,\hat\tau_t]\) |
| \(a_t^{L,\mathrm{exec}}\) | 下层执行连续动作 | \([I_{0,t},I_{1,t},I_{2,t},\rho_t^{\mathrm{exec}},\tau_t^{\mathrm{exec}}]\) |
| \(r_t\) | 单步环境奖励 | 标量 |
| \(c_t\) | 瞬时约束向量 | \([c_t^{\mathrm{qos}}, c_t^{T_A}, c_t^{T_{B1}}, c_t^{T_{B2}}]^\top\) |
| \(\tilde r_t\) | 拉格朗日塑形奖励 | \(r_t-\lambda^\top c_t\) |

## 2 固定异构多发射端结构
### 2.1 发射端集合
定义发射端集合为
\[
\mathcal{T}=\{0,1,2\},
\]
其中：
- 发射端 0：宽束 Anchor，固定实现为 LED；
- 发射端 1：窄束 Boost1，固定实现为 LD；
- 发射端 2：窄束 Boost2，固定实现为 LD。

因此本文采用的是“发射端固定异构”建模，而不是在同一发射端内部再对 LED/LD 比例进行软混合。Anchor 负责保底覆盖和鲁棒连接，Boost 负责在链路条件较好时提供增强增益。

### 2.2 名义使能与实际执行
上层策略只显式决定两路 Boost 的名义使能组合以及 SLIPT 结构模式，Anchor 在名义层持续参与。定义名义 Boost 组合
\[
u_t \in \{[1,0,0],[1,1,0],[1,0,1],[1,1,1]\},
\]
分别对应：
- Anchor only；
- Anchor + Boost1；
- Anchor + Boost2；
- Anchor + Boost1 + Boost2。

需要强调的是，Anchor 的“持续参与”指的是名义结构设计偏好，而不是任何时刻都强制要求其实际电流严格大于零。当热保护触发时，执行侧安全层允许包括 Anchor 在内的任一路发射端的实际电流被平滑压降到接近零。

## 3 Physics-Aware Link-Level Model：LED–LD 异构链路模型
### 3.1 非相干光强叠加
设第 \(i\) 路发射端在时刻 \(t\) 的执行电流为 \(I_{i,t}\)，等效器件增益为 \(\eta_i\)，链路增益为 \(h_{i,t}\)，则各发射端的接收贡献写为
\[
P_{i,t}=I_{i,t}\,\eta_i\,h_{i,t}, \quad i\in\mathcal{T}.
\]
总接收光功率采用非相干光强叠加近似：
\[
P_t^{\mathrm{rx}} = \sum_{i\in\mathcal{T}} P_{i,t}.
\]
在实现中，上式被显式写成逐发射端求和的形式：
- Anchor LED 贡献一项；
- Boost1 LD 贡献一项；
- Boost2 LD 贡献一项；
- 总接收功率为三项求和。

### 3.2 LED / LD 异构差异
为反映老师提出的 Hybrid LED–LD 场景，LED 与 LD 的链路模型参数不完全相同。第 \(i\) 路链路增益近似写为
\[
h_{i,t}=e_i\,g_i(d_i)\,a_i(\zeta_t,d_i)\,m_i(\epsilon_{i,t})\,b_i(\xi_{i,t}),
\]
其中：
- \(e_i\in\{0,1\}\) 表示该发射端是否在硬件层启用；
- \(g_i(d_i)\) 表示由几何距离决定的基准扩散项；
- \(\zeta_t\) 表示时变有效水体衰减系数；
- \(a_i(\zeta_t,d_i)\) 表示由水体衰减和距离共同决定的损耗项；
- \(m_i(\epsilon_{i,t})\) 表示失准惩罚项；
- \(b_i(\xi_{i,t})\) 表示 burst / 突发扰动造成的附加衰减。

与本文采用的实现一致，上述各项可进一步写为
\[
g_i(d_i)=\left(\frac{d_0}{d_i+\varepsilon}\right)^2,
\qquad
a_i(\zeta_t,d_i)=\exp\!\left(-\zeta_t\,\alpha_i\,d_i\right),
\]
\[
m_i(\epsilon_{i,t})=\exp\!\left(-\frac{1}{2}\left(\frac{\epsilon_{i,t}}{\sigma_i}\right)^2\right),
\qquad
b_i(\xi_{i,t})=\exp(-\beta_i \xi_{i,t}),
\]
其中 \(\alpha_i,\sigma_i,\beta_i\) 分别由 LED/LD 的衰减尺度、失准尺度和 burst 敏感系数决定。

LED 与 LD 的差异通过下列器件相关系数体现：
- 衰减系数：LED 与 LD 采用不同的 attenuation factor；
- 失准尺度：LED 对失准更耐受，LD 对失准更敏感；
- 光电转换效率：\(\eta_{\mathrm{LED}}\) 与 \(\eta_{\mathrm{LD}}\) 不同；
- 噪声系数：LED / LD 对有效信号引起的附加噪声权重不同；
- 热系数：LED / LD 的发热强度不同；
- burst 敏感系数：LED / LD 对突发恶化的脆弱程度不同。

因此，本文中的“宽束/窄束”并不是纯抽象概念，而是明确落实到了“Anchor=LED、Boost=LD”的器件异构层面。

在此基础上，本文定义信息侧有效信号和能量侧输入分别为
\[
P_t^{\mathrm{info}} = \sum_{i\in\mathcal{T}} \omega_i^{\mathrm{SE}} P_{i,t},
\qquad
P_t^{\mathrm{eh}} = \sum_{i\in\mathcal{T}} \omega_i^{\mathrm{EH}} P_{i,t},
\]
噪声功率与瞬时信噪比写为
\[
N_t = N_0 + \nu_{\mathrm{LED}}|P_t^{\mathrm{LED}}| + \nu_{\mathrm{LD}}|P_t^{\mathrm{LD}}| + \varepsilon_t^{(n)},
\]
\[
\mathrm{SNR}_t = \max\!\left(\frac{P_t^{\mathrm{info}}}{\max(N_t,\varepsilon)}, \varepsilon\right),
\]
其中 \(P_t^{\mathrm{LED}}\) 与 \(P_t^{\mathrm{LD}}\) 分别表示 LED 与 LD 贡献信号之和。

## 4 SLIPT 模式与接收端控制变量
### 4.1 模式集合
定义上层模式变量
\[
m_t \in \{0,1,2\},
\]
分别表示：
- \(m_t=0\)：PS（Power Splitting）；
- \(m_t=1\)：TS（Time Switching）；
- \(m_t=2\)：HY（Hybrid）。

### 4.2 连续接收端控制变量
下层连续动作中包含接收端参数 \(\rho_t\) 与 \(\tau_t\)，其物理语义如下：
- \(\rho_t\in[0,1]\) 表示功率分流比例；
- \(\tau_t\in[0,1]\) 表示时间切换中的信息时隙占比。

为了消除无效自由度，执行侧 Safety Layer 对 \(\rho_t\) 与 \(\tau_t\) 做模式感知投影：
- 当 \(m_t=0\)（PS）时，固定 \(\tau_t^{\mathrm{exec}}=1\)，仅保留 \(\rho_t\) 作为有效自由度；
- 当 \(m_t=1\)（TS）时，固定 \(\rho_t^{\mathrm{exec}}=0\)，仅保留 \(\tau_t\) 作为有效自由度；
- 当 \(m_t=2\)（HY）时，\(\rho_t\) 与 \(\tau_t\) 共同参与控制。

从物理意义上看，PS 模式对应功率域分流，TS 模式对应时间域切换，而 HY 模式对应功率域与时间域的耦合联合控制。因此，模式切换不是简单的奖励标签切换，而是接收端物理工作机制的切换。

### 4.3 SE / EH 计算
根据信息分流与能量分流语义，定义：
- 若 \(m_t=0\)（PS），则
\[
s_t^{\mathrm{info}} = 1-\rho_t^{\mathrm{exec}}, \qquad s_t^{\mathrm{eh}}=\rho_t^{\mathrm{exec}}.
\]
- 若 \(m_t=1\)（TS），则
\[
s_t^{\mathrm{info}} = \tau_t^{\mathrm{exec}}, \qquad s_t^{\mathrm{eh}}=1-\tau_t^{\mathrm{exec}}.
\]
- 若 \(m_t=2\)（HY），则
\[
s_t^{\mathrm{info}} = \tau_t^{\mathrm{exec}}(1-\rho_t^{\mathrm{exec}}), \qquad s_t^{\mathrm{eh}}=\rho_t^{\mathrm{exec}}.
\]

设模式增益分别为 \(g_m^{\mathrm{SE}}\) 与 \(g_m^{\mathrm{EH}}\)，则 QoS 相关速率项与 EH 相关采集项分别为
\[
R_t^{\mathrm{qos}} = g_{m_t}^{\mathrm{SE}}\, s_t^{\mathrm{info}}\,\log_2(1+\mathrm{SNR}_t),
\]
\[
E_t^{\mathrm{harv}} = g_{m_t}^{\mathrm{EH}}\, s_t^{\mathrm{eh}}\,P_t^{\mathrm{eh}}.
\]
最终进入奖励的 SE 与 EH 项写为
\[
R_t^{\mathrm{SE}} = w_{\mathrm{SE}} R_t^{\mathrm{qos}}, \qquad
R_t^{\mathrm{EH}} = w_{\mathrm{EH}} E_t^{\mathrm{harv}}.
\]

## 5 独立温度动态与状态空间
### 5.1 独立温度动态
每个发射端维护独立温度状态 \(T_{i,t}\)。令环境温度为 \(T_t^{\mathrm{amb}}\)，则一阶热动态写为
\[
T_{i,t+1}=(1-\gamma_t)T_{i,t}+\gamma_t T_t^{\mathrm{amb}}+\delta_t\kappa_i I_{i,t}^2,
\]
其中：
- \(\gamma_t\) 为热回落系数；
- \(\delta_t\) 为发热驱动系数；
- \(\kappa_i\) 为与器件类型有关的热系数，LED 与 LD 取值不同。

### 5.2 状态定义
本文采用的观测状态写为
\[
s_t = [\tilde h_t,\tilde \epsilon_t,\tilde T_t,\tilde T_t^{\mathrm{amb}},\tilde I_{t-1},\rho_{t-1}^{\mathrm{exec}},\tau_{t-1}^{\mathrm{exec}},m_{t-1}^{\mathrm{exec}},u_{t-1}^{\mathrm{exec}},d_{t-1},\tilde R_{t-1}^{\mathrm{qos}},\tilde E_{t-1}^{\mathrm{harv}}],
\]
其中：
- \(\tilde h_t\)：逐发射端归一化链路质量；
- \(\tilde \epsilon_t\)：逐发射端归一化失准；
- \(\tilde T_t\)：逐发射端温度相对环境温度的归一化量；
- \(\tilde T_t^{\mathrm{amb}}\)：环境温度归一化量；
- \(\tilde I_{t-1}\)：上一时刻执行电流归一化量；
- \(m_{t-1}^{\mathrm{exec}}\)：上一时刻执行模式；
- \(u_{t-1}^{\mathrm{exec}}\)：上一时刻执行 Boost 组合；
- \(d_{t-1}\)：上一执行 Boost 组合已经连续保持的步数；
- \(\tilde R_{t-1}^{\mathrm{qos}}\)：上一时刻实际 QoS 速率指标归一化量；
- \(\tilde E_{t-1}^{\mathrm{harv}}\)：上一时刻 EH 指标归一化量。

其中 \(d_{t-1}\) 表示上一执行 Boost 组合的连续保持步数，而不是模式 \(m_t\) 的保持步数；这一设计与后续仅对 Boost 组合施加最小驻留时间约束的执行逻辑保持一致。

## 6 分层动作空间
### 6.1 上层离散动作
上层离散动作由 Boost 组合和 SLIPT 模式组成：
\[
a_t^{U,\mathrm{raw}} = (u_t^{\mathrm{raw}}, m_t^{\mathrm{raw}}),
\]
其中 \(u_t^{\mathrm{raw}}\in\{0,1,2,3\}\)，\(m_t^{\mathrm{raw}}\in\{0,1,2\}\)。因此 raw 宏动作共有 12 个候选。

由于执行侧存在最小驻留时间约束，上层提出的 raw Boost 组合不一定立即执行。令经过驻留时间冻结后的执行 Boost 组合为
\[
u_t^{\mathrm{exec}} = \Pi_{\mathrm{dwell}}(u_t^{\mathrm{raw}}),
\]
则执行宏动作为
\[
a_t^{U,\mathrm{exec}}=(u_t^{\mathrm{exec}},m_t^{\mathrm{raw}}).
\]
训练时，上层 DQN 的回放、价值学习和目标值构造都以执行宏动作语义 \(a_t^{U,\mathrm{exec}}\) 为准，而不是以 raw 提议为准。

### 6.2 下层连续动作
下层连续原始动作记为
\[
a_t^{L,\mathrm{raw}}=[\hat I_{0,t},\hat I_{1,t},\hat I_{2,t},\hat\rho_t,\hat\tau_t].
\]
经 Safety Layer 投影后得到真正执行的连续动作
\[
a_t^{L,\mathrm{exec}}=[I_{0,t},I_{1,t},I_{2,t},\rho_t^{\mathrm{exec}},\tau_t^{\mathrm{exec}}].
\]
在本文方法中，下层 SAC 的 actor 和 critic 显式接收当前执行宏动作的 one-hot 条件：
- Boost one-hot(4)
- Mode one-hot(3)
因此下层连续控制器直接感知“当前究竟执行的是哪一种结构语义”。

## 7 EH–SE–稳定性联合优化与约束
### 7.1 奖励函数
本文采用的单步奖励写为
\[
r_t = R_t^{\mathrm{SE}} + R_t^{\mathrm{EH}} + r_t^{\mathrm{margin}} - w_c c_t^{\Sigma} - w_p\|I_t\|_2^2 - w_s\Delta a_t^2 - p_t^{\mathrm{switch}},
\]
其中：
- \(c_t^{\Sigma}=\mathbf{1}^\top c_t\) 表示所有约束分量的总瞬时违反量；
- \(r_t^{\mathrm{margin}}\) 为热安全裕度奖励；
- \(\Delta a_t^2\) 表示电流与 \(\rho/\tau\) 的动作平滑项；
- \(p_t^{\mathrm{switch}}\) 表示模式切换和 Boost 组合切换惩罚。

其中，上述三项进一步写为
\[
r_t^{\mathrm{margin}} = w_m \cdot \mathrm{clip}\!\left(\frac{T_{\mathrm{safe}}-\max_i T_{i,t}}{T_{\mathrm{safe}}},0,1\right),
\]
\[
\Delta a_t^2 = \frac{1}{|\mathcal{T}|}\sum_{i\in\mathcal{T}}
\left(\frac{I_{i,t}-I_{i,t-1}}{I_i^{\max}}\right)^2
\;+\;
\frac{1}{2}\Big[(\rho_t^{\mathrm{exec}}-\rho_{t-1}^{\mathrm{exec}})^2+(\tau_t^{\mathrm{exec}}-\tau_{t-1}^{\mathrm{exec}})^2\Big],
\]
\[
p_t^{\mathrm{switch}} = w_m^{\mathrm{sw}}\mathbf{1}[m_t\neq m_{t-1}] + w_u^{\mathrm{sw}}\mathbf{1}[u_t\neq u_{t-1}].
\]

需要说明的是，上式中的动作平滑项属于通用奖励结构的一部分，但在本文当前报告的主要实验配置中，其权重 \(w_s\) 默认取 0，因此该项在实验结果中通常不实际生效，除非另行说明。

### 7.2 约束向量
本文的瞬时约束向量定义为
\[
c_t = [c_t^{\mathrm{qos}}, c_t^{T_A}, c_t^{T_{B1}}, c_t^{T_{B2}}]^\top,
\]
其中
\[
c_t^{\mathrm{qos}} = [R_{\min}^{\mathrm{qos}} - R_t^{\mathrm{qos}}]_+,
\]
\[
c_t^{T_i} = [T_{i,t}-T_{\mathrm{safe}}]_+.
\]
因此：
- 第一个约束分量对应 QoS 速率不足；
- 后三个约束分量分别对应 Anchor、Boost1、Boost2 的温度超限量。

## 8 共享拉格朗日对偶层（长期约束层）
为实现任务分布平均意义下的长期约束控制，本文引入向量对偶变量
\[
\lambda = [\lambda_{\mathrm{qos}},\lambda_{T_A},\lambda_{T_{B1}},\lambda_{T_{B2}}]^\top, \qquad \lambda \succeq 0.
\]

对偶变量更新采用逐分量投影梯度上升：
\[
\lambda_k \leftarrow \Pi_{[0,\lambda_k^{\max}]}
\left(
\lambda_k + \eta_k(\bar c_k - b_k)
\right),
\]
其中：
- \(\bar c_k\) 为 support 集或任务分布平均下第 \(k\) 个约束分量的经验均值；
- \(b_k\) 为对应的长期约束预算；
- \(\eta_k\) 为对偶学习率。

在强化学习更新中，长期约束通过拉格朗日惩罚进入训练目标：
\[
\tilde r_t = r_t - \lambda^\top c_t.
\]
在训练过程中，value/policy 更新使用塑形后的 \(\tilde r_t\)，而上下文编码器仍使用原始环境回报 \(r_t\) 与约束反馈 \(c_t\)。这样可以使潜变量 \(z\) 更偏向于表征任务本身，而不是被拉格朗日惩罚项过度主导。

## 9 执行侧 Safety Layer（即时安全层）
Safety Layer 负责将 \((a_t^{U,\mathrm{raw}}, a_t^{L,\mathrm{raw}})\) 映射为满足即时可行性要求的执行动作。

### 9.1 Boost 组合驻留时间约束
若上一执行 Boost 组合尚未保持满最小驻留时间，则冻结 Boost 组合；否则允许切换。记该操作为
\[
u_t^{\mathrm{exec}} = \Pi_{\mathrm{dwell}}(u_t^{\mathrm{raw}}).
\]
本文仅对 Boost 组合施加驻留时间约束，不对模式 \(m_t\) 额外施加冻结。

### 9.2 连续动作安全投影
对下层原始动作，Safety Layer 按以下顺序执行：
1. 对原始电流做 sigmoid 映射并乘以每路电流上限；
2. 结合执行 Boost 组合，对非激活通道施加平滑掩码；
3. 对总电流施加平滑母线投影；
4. 利用一步温度预测执行热感知平滑降额；
5. 对 \(\rho_t\) 与 \(\tau_t\) 做模式感知投影。

温度预测为
\[
\hat T_{i,t+1} = (1-\gamma_t)T_{i,t}+\gamma_t T_t^{\mathrm{amb}}+\delta_t\kappa_i I_{i,t}^2.
\]
对执行 Boost 组合的平滑掩码写为
\[
\tilde m_i(u_t^{\mathrm{exec}})= e_i\Bigl(\mu + (1-\mu)\,m_i(u_t^{\mathrm{exec}})\Bigr),
\]
其中 \(\mu\) 为 `mask_floor`，\(e_i\) 为硬件使能标志，\(m_i(\cdot)\in\{0,1\}\) 为名义 Boost 组合掩码。

对总电流的平滑母线投影可写为
\[
s_t^{\mathrm{bus}}=(1-g_t)+g_t\frac{I_{\mathrm{bus}}^{\max}}{I_t^{\Sigma}+\mathrm{softplus}(I_t^{\Sigma}-I_{\mathrm{bus}}^{\max})+\varepsilon},
\]
\[
g_t=\sigma\!\left(\beta\left(I_t^{\Sigma}-I_{\mathrm{bus}}^{\max}\right)\right),
\qquad
I_t^{\Sigma}=\sum_{i\in\mathcal{T}} I_{i,t},
\]
然后将各路电流统一乘以 \(s_t^{\mathrm{bus}}\)。

热降额采用平滑投影而不是硬裁剪，形式为
\[
I_{i,t}^{\mathrm{safe}} = I_{i,t}\,\sigma\bigl(\alpha_s(T_{\mathrm{safe}}-\hat T_{i,t+1})\bigr)
\,\sigma\bigl(\alpha_c(T_{\mathrm{cut}}-\hat T_{i,t+1})\bigr),
\]
其中 \(\sigma(\cdot)\) 为 sigmoid 函数，\(\alpha_s,\alpha_c\) 分别控制 soft safe 区与 cutoff 区的平滑程度。这样可以避免“超过阈值直接置零”带来的非平滑性和梯度劣化问题，更符合老师提出的 smooth projection / soft clipping 思路。

## 10 上下文潜变量与任务条件建模
本文使用 GRU 上下文编码器和高斯潜变量 \(z_t\) 表征当前任务。代码中的单步上下文条目显式写为
\[
x_\tau=
\big[
 s_\tau,\;
 a_\tau^{U,\mathrm{exec}},\;
 a_\tau^{L,\mathrm{exec}},\;
 r_\tau^{\mathrm{raw}},\;
 c_\tau
\big],
\]
其中：
- \(s_\tau\) 为观测；
- \(a_\tau^{U,\mathrm{exec}}\) 为执行宏动作编码，在实现中具体为 Boost one-hot(4) 与 Mode one-hot(3) 的拼接；
- \(a_\tau^{L,\mathrm{exec}}\) 为执行连续动作；
- \(r_\tau^{\mathrm{raw}}\) 为原始环境回报；
- \(c_\tau\) 为完整约束向量反馈。

因此上下文序列写为
\[
\chi_t = [x_\tau]_{\tau \le t}.
\]

上下文编码器输出的是对角高斯后验，而不是一般满协方差高斯：
\[
q_\phi(z\mid \chi_t)=\mathcal{N}\!\big(\mu_\phi(\chi_t),\mathrm{diag}(\sigma_\phi^2(\chi_t))\big).
\]
为增强潜变量的任务识别能力，上下文编码器还通过辅助预测头回归任务参数摘要，例如：
- 水体衰减系数；
- 失准标准差；
- 环境温度；
- 热回落系数 \(\gamma\)；
- 发热驱动系数 \(\delta\)；
- QoS 最低速率门限。

## 11 分层强化学习结构
### 11.1 上层 DQN
上层使用 DQN 负责结构选择。时间尺度分离下，每经过 \(K_h\) 个环境步，上层才重新提出一次 raw 宏动作；在中间步，下层连续控制器继续在固定的执行宏结构语义下工作。

本文的上层决策机制满足以下特征：
- 上层网络显式输出 12 维离散价值；
- 训练时回放中记录的是执行宏动作索引 \(a_t^{U,\mathrm{exec}}\)；
- 在贪心选择分支中，会先枚举 12 个 raw 宏动作在当前驻留记忆下对应的执行索引，再按照映射后的执行价值完成 greedy 选择；
- target value 的最大化也基于下一时刻 raw\(\rightarrow\)exec 映射后的执行宏动作语义完成。

对应的上层 SMDP 型 TD 目标可写为
\[
y_t^{U}=R_t^{U}+\gamma^{H_t}(1-d_t)\max_{a^{U,\mathrm{raw}}}
Q_{\bar\psi}\!\left(s_{t+H_t},z_{t+H_t},\Pi_{\mathrm{exec}}(a^{U,\mathrm{raw}})\right),
\]
其中 \(H_t\) 为该宏动作实际持续的环境步数，且 target 所使用的 next state 是宏动作结束后的状态 \(s_{t+H_t}\)，而不是单步意义下的 \(s_{t+1}\)。

上层 DQN 损失采用 Huber 损失：
\[
\mathcal{L}_{U}=
\mathbb{E}\Big[
\ell_{\mathrm{Huber}}\big(
Q_{\psi}(s_t,z_t,a_t^{U,\mathrm{exec}}),
y_t^{U}
\big)
\Big].
\]
这里的离散动作标签是执行宏动作索引 \(a_t^{U,\mathrm{exec}}\)，而不是 raw 提议索引。

### 11.2 下层 SAC
下层使用 twin-critic SAC 控制连续动作。critic 与 actor 都接收：
- 当前观测 \(s_t\)；
- 任务潜变量 \(z_t\)；
- 当前执行宏动作的 one-hot 条件。

critic 的训练使用真正进入环境的执行动作 \(a_t^{L,\mathrm{exec}}\)，actor 则通过可微 Safety Layer 投影，把原始动作样本映射为执行动作后再计算策略损失。这保证了训练目标与真实环境转移保持一致。

设下层策略输出原始连续动作样本
\[
\hat a_{t+1}^{L,\mathrm{raw}} \sim \pi_\theta(\cdot \mid s_{t+1},z_{t+1},a_{t+1}^{U,\mathrm{exec}}),
\]
经 Safety Layer 投影后得到
\[
a_{t+1}^{L,\mathrm{exec}}=\Pi_{\mathrm{safe}}\!\left(\hat a_{t+1}^{L,\mathrm{raw}};a_{t+1}^{U,\mathrm{exec}},T_{t+1}\right).
\]
则 twin-critic 的目标写为
\[
y_t^{L}=\tilde r_t+\gamma(1-d_t)\left[
\min_{k\in\{1,2\}}Q_{\bar\phi_k}(s_{t+1},z_{t+1},a_{t+1}^{L,\mathrm{exec}})
-\alpha \log \pi_\theta(\hat a_{t+1}^{L,\mathrm{raw}}\mid s_{t+1},z_{t+1},a_{t+1}^{U,\mathrm{exec}})
\right].
\]
对应的 critic 损失为
\[
\mathcal{L}_{Q}=
\mathbb{E}\Big[
\big(Q_{\phi_1}(s_t,z_t,a_t^{L,\mathrm{exec}})-y_t^{L}\big)^2
+\big(Q_{\phi_2}(s_t,z_t,a_t^{L,\mathrm{exec}})-y_t^{L}\big)^2
\Big].
\]

策略损失写为
\[
\mathcal{L}_{\pi}=
\mathbb{E}\Big[
\alpha \log \pi_\theta(\hat a_t^{L,\mathrm{raw}}\mid s_t,z_t,a_t^{U,\mathrm{exec}})
-\min_{k\in\{1,2\}}Q_{\phi_k}(s_t,z_t,a_t^{L,\mathrm{exec}})
\Big],
\]
其中
\[
a_t^{L,\mathrm{exec}}=\Pi_{\mathrm{safe}}\!\left(\hat a_t^{L,\mathrm{raw}};a_t^{U,\mathrm{exec}},T_t\right).
\]

本文默认使用固定或调度式 \(\alpha\)；同时保留了可选的自动温度调节接口。

## 12 显式内环—外环元学习机制
为了使“元学习”不仅停留在潜变量条件化层面，本文将训练显式划分为任务内内环和任务间外环两个层次。

### 12.1 内环（task-level adaptation）
对每个任务 \(\mathcal{T}_j\)，从共享初始化参数 \(\Theta\) 出发，在 support episodes 上执行若干轮任务内适应，得到任务适应后参数
\[
\Theta'_j = \mathrm{InnerUpdate}(\Theta; \mathcal{D}_j^{\mathrm{sup}}).
\]
在这一阶段：
- replay buffer 与 upper replay 采用任务内局部缓冲；
- support episodes 共享同一个任务上下文历史；
- support 阶段先在共享对偶初始化 \(\lambda\) 下完成任务内适应；
- 随后根据 support 集约束均值更新任务级向量对偶变量 \(\lambda'_j\)；
- query 阶段继续在 \(\lambda'_j\) 下推进任务参数更新。

### 12.2 外环（shared initialization update）
在 query episodes 上继续沿任务适应方向推进，并将 query 后的任务参数 \(\Theta''_j\) 作为外环更新目标。本文采用 Reptile 风格的一阶元更新：
\[
\Theta \leftarrow \Theta + \beta \cdot \frac{1}{N}\sum_{j=1}^{N}(\Theta''_j - \Theta),
\]
其中 \(\beta\) 为外环步长。

对偶变量初始化也采用同样的任务间插值思路。设 support 后任务级对偶变量为 \(\lambda'_j\)，则共享对偶初始化更新为
\[
\lambda \leftarrow \lambda + \beta \cdot \frac{1}{N}\sum_{j=1}^{N}(\lambda'_j - \lambda).
\]
因此，本文的元学习性并不依赖于二阶梯度，而是体现在“跨任务共享初始化 + support 快速适应 + query 驱动外环更新 + 共享初始化插值更新”的闭环机制之中。换言之，本文采用的是显式 inner/outer 结构下的一阶元更新方案，而不是严格的二阶 MAML。

## 13 训练与回放中的 raw / exec 处理
由于 Safety Layer 可能冻结 Boost 组合、平滑缩放电流并对 \(\rho/\tau\) 做模式感知投影，因此 raw 动作与 exec 动作一般不完全相同。本文遵循以下训练与回放原则：
- replay buffer 同时保留原始连续动作和执行连续动作；
- 上层 replay 以执行宏动作索引作为离散动作标签；
- 上层同时保留 raw 宏动作与下一时刻 raw\(\rightarrow\)exec 映射，用于执行语义下的价值选择与目标构造；
- 下层 critic / target 一律按执行连续动作更新；
- 上下文编码器使用的是执行动作、原始环境回报和完整约束向量反馈。

这种 raw / exec 分离处理保证了：
1. 学习目标与环境真实转移一致；
2. 执行侧安全修正不会被误当成环境噪声；
3. 论文中的“原始提议—执行动作”语义能够在代码里真正闭环。

## 14 方法流程总结
综合起来，本文方法的完整流程如下：
1. 环境给出当前观测、温度状态和任务条件；
2. 上下文编码器根据历史执行轨迹推断潜变量 \(z_t\)；
3. 上层 DQN 提出 raw 宏动作；
4. Safety Layer 先根据最小驻留时间约束得到执行 Boost 组合；
5. 下层 SAC 在当前执行宏动作语义下输出连续原始动作；
6. Safety Layer 对电流、母线电流、热风险以及 \(\rho/\tau\) 进行平滑投影，得到执行动作；
7. 环境基于逐发射端 LED/LD 异构贡献求和计算接收信号、SNR、QoS 速率、EH、温度演化和约束分量；
8. 向量对偶层根据长期平均约束违反量更新多维拉格朗日乘子；
9. 内环在 support 集上完成任务内适应，并根据 support 约束均值形成任务级对偶变量；
10. query 集继续推动任务参数更新；
11. 外环将共享初始化参数与共享对偶初始化向 query 后的任务适应结果做插值更新。

从物理意义上看，当链路差、失准大或窄束 LD 热风险较高时，策略更倾向于回落到仅保留 Anchor 的结构；当链路改善且热裕度允许时，则逐步启用一条或两条 Boost，并由下层连续控制器决定三路电流和接收端参数，从而实现“Anchor 保底 + Boost 增强 + 安全平滑执行”的整体设计目标。

### 14.1 训练阶段伪代码
可将本文方法的训练过程概括为如下伪代码：

1. 初始化共享参数 \(\Theta\)、共享对偶初始化 \(\lambda\)、任务采样器与 replay buffer。
2. 对每个 meta-iteration：
3. 根据 curriculum 阶段设置任务采样范围，并采样任务集合 \(\{\mathcal{T}_j\}_{j=1}^{N}\)。
4. 保存共享初始化参数 \(\Theta\) 与共享对偶初始化 \(\lambda\)。
5. 对每个任务 \(\mathcal{T}_j\)：
6. 恢复共享初始化参数与共享对偶初始化，清空任务内 replay buffer。
7. 在 support episodes 上与环境在线交互：
8. 上下文编码器推断 \(z_t\)。
9. 上层 DQN 提出 raw 宏动作，并通过驻留约束映射到执行宏动作。
10. 下层 SAC 在执行宏动作语义下输出原始连续动作。
11. Safety Layer 执行平滑投影，环境返回 reward、cost vector 和下一状态。
12. 用执行动作更新下层 replay，用执行宏动作更新上层 replay，并进行任务内学习。
13. 统计 support 集约束均值，更新任务级对偶变量 \(\lambda'_j\)。
14. 在 query episodes 上继续在线交互与任务内更新。
15. 保存 query 后的任务适应参数 \(\Theta''_j\) 和任务级对偶状态。
16. 对所有任务完成后，执行外环插值更新：
\[
\Theta \leftarrow \Theta + \beta \cdot \frac{1}{N}\sum_j(\Theta''_j-\Theta),
\]
\[
\lambda \leftarrow \lambda + \beta \cdot \frac{1}{N}\sum_j(\lambda'_j-\lambda).
\]
17. 记录 support/query 指标并按计划保存 checkpoint。

### 14.2 在线执行阶段
在线执行时的控制链路可概括为：

1. 接收环境观测 \(s_t\) 与当前温度状态；
2. 基于历史执行轨迹推断任务潜变量 \(z_t\)；
3. 上层提出 raw 宏动作；
4. Safety Layer 根据最小驻留时间得到执行 Boost 组合；
5. 下层在当前执行宏动作条件下输出原始连续动作；
6. Safety Layer 进行模式感知与热感知的平滑投影；
7. 环境执行 \(a_t^{L,\mathrm{exec}}\)，并更新链路、温度与约束状态；
8. 将执行轨迹反馈到上下文编码器，进入下一步决策。

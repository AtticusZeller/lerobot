# Visualize Dataset 数据分析指南

本文档系统化解释三大面板（Dataset Statistics / Filtering / Action Insights）中所有指标的含义、判定方法、推荐阈值与理论依据，服务于 LeRobot 数据集质量评估与清洗工作。

---

## Part 1 · 指标含义速查

### 1.1 Dataset Statistics（数据集统计）

#### Overview Cards

| 指标 | 含义 |
|---|---|
| Robot Type | 机器人型号（来自 `meta/info.json` 的 `robot_type` ） |
| Dataset Version | 数据集格式版本（v2.0 / v2.1 / v3.0） |
| Tasks | 任务标注数（subtask 总数） |
| Total Frames | 全数据集总帧数 |
| Total Episodes | 总回合数 |
| FPS | 录制采样帧率 |
| Total Recording Time | 总录制时长，计算方式： `Total Frames / FPS` |
| Camera Resolutions | 每个摄像头的 `width × height` 分辨率 |

#### Episode Lengths（回合时长统计，单位：秒）

| 指标 | 含义 |
|---|---|
| Shortest | 最短回合时长 |
| Longest | 最长回合时长 |
| Mean | 回合时长均值 |
| Median | 回合时长中位数 |
| Std Dev | 回合时长标准差 |
| Episode Length Distribution | 回合时长分布直方图，X 轴为时长区间，Y 轴为回合数 |

---

### 1.2 Filtering（清洗过滤面板）

| 区块 / 指标 | 含义 |
|---|---|
| Flagged Episodes | 已标记的回合 ID 列表，可一键复制或生成 `lerobot-edit-dataset` 删除命令 |
| Episode Length Filter | 双滑块设定时长范围 `[min, max]` ，范围外的回合实时统计，可批量标记 |
| Lowest-Movement · `totalMovement` | 进度条右侧的数字。每帧平均关节速度幅值： `mean(‖Δa_t‖₂)` ，即所有关节角速度向量的 L2 范数按帧取均值，反映整体运动速度大小。数值偏小说明整体运动不明显——常见于抓取失败后重试、只有夹爪动了几下而手臂关节几乎没移动的回合，因为夹爪位移贡献的 L2 很小，拉低了整体均值 |
| Lowest-Movement 进度条颜色 | 相对最大运动量的比例：🔴 <15% / 🟡 <40% / 🟢 ≥40% |
| Action Velocity（共用） | 见 §1.3 第 ③ 项 |

---

### 1.3 Action Insights（动作洞察）

> 支持两种视角切换：**Current Episode**（当前单回合）/ **All Episodes**（全数据集采样，最多 500 回合）。

#### ① Action Autocorrelation（动作自相关）

| 指标 | 含义 |
|---|---|
| 横轴 Lag（步） | 计算自相关时的时间间隔，括号内为对应秒数 |
| 纵轴 ACF | 自相关系数：1 = 完全相关，0 = 独立，<0 = 反向相关 |
| 0.5 阈值虚线 | 跌破此线的 lag 被视为"自然 chunk 边界" |
| Suggested chunk length | 各维度跌破 0.5 的 lag 的中位数，即推荐的 `chunk_size` |

#### ② State–Action Temporal Alignment（状态-动作时序对齐）

| 指标 | 含义 |
|---|---|
| max 曲线（橙） | 所有匹配维度对的 cross-correlation 最大包络 |
| mean 曲线（灰） | 均值包络 |
| min 曲线（蓝） | 最小包络 |
| 横轴 Lag（步） | >0 表示状态滞后动作；<0 表示动作具预测性 |
| Mean control delay | mean 曲线峰值对应的 lag，即"平均控制延迟"（步数与秒数） |
| Individual peak range | 各维度峰值 lag 的最小~最大范围，反映不同关节的延迟差异 |
| r（相关系数） | 峰值对应的 cross-correlation 强度，r > 0.6 表示 action-state 因果关系明确 |
| Matched pairs | 通过后缀匹配成功配对的 action–state 维度数 |

#### ③ Action Velocity (Δa) — Smoothness Proxy（动作平滑度）

| 指标 | 含义 |
|---|---|
| Δa 直方图 | 每个动作维度的**动作速度分布**（Action Velocity Distribution）。 `Δa = a(t+1) − a(t)` 即相邻帧间的关节角速度（rad/frame 或 m/frame），分布越集中在零附近 = 关节运动越平滑匀速；分布越宽或有长尾 = 有急停急启或抖动 |
| σ (std) | 归一化标准差（除以电机量程），衡量整体抖动程度 |
| `\|Δ\|max` | 归一化最大单帧变化，衡量极端跳变 |
| 进度条颜色 | 相对最大 σ 的比例：🟢 <40% / 🟡 <70% / 🔴 ≥70% |
| Tag: `inactive` | p95 `\|Δa\|` < 0.1% 量程，该维度视为静止 |
| Tag: `discrete` | 唯一值数 ≤4，该维度视为离散信号（如夹爪二值开关） |
| Overall Verdict | 整体平滑度评级：**Smooth** / **Moderate** / **Jerky** / **N/A** |
| Most Jerky Episodes 列表 | 全数据集按 mean `\|Δa\|` 降序排列的 jerky 回合 |

> `inactive` 与 `discrete` 维度不参与 Overall Verdict 计算。

#### ④ Demonstrator Speed Variance（演示者速度方差）

| 指标 | 含义 |
|---|---|
| Speed 分布直方图 | 每个回合的 mean `‖Δa‖` （即执行速度）的跨回合分布 |
| Mean | 平均执行速度 |
| Median | 速度中位数 |
| Std | 速度标准差 |
| **CV**（变异系数） | `std / mean` ，归一化的速度离散度，核心判定指标 |
| Verdict | Consistent / Moderate variance / High variance |

#### ⑤ Cross-Episode Action Variance Heatmap（跨回合动作方差热图）

| 指标 | 含义 |
|---|---|
| 横轴 | 回合相对进度（0–100%，归一化时间分 bin） |
| 纵轴 | 各动作维度 |
| 颜色 | 该维度在该时间点跨回合的方差（暖色 = 高方差，冷色 = 低方差） |
| numEpisodes | 参与计算的采样回合数（最多 500） |

---

## Part 2 · 判定标准、推荐阈值与理论依据

### 2.1 回合时长（Episode Lengths）→ 识别录制异常

**判定方法**

| 现象 | 判读 | 处理 |
|---|---|---|
| Mean ≈ Median | 分布近似正态，时长一致性好 | 正常 |
| Mean >> Median | 少数过长回合拉高均值（长尾） | 检查过长回合：机器人卡住或操作员停顿 |
| Std Dev / Mean > 0.5 | 时长高度不一致 | 核查任务规范，统一操作节奏 |
| 出现明显双峰 | 混有两类不同时长任务 | 分数据集分开训练 |

**推荐阈值**

* SO-101 类抓取/放置任务：单回合建议 **15–60 s**
* **< 5 s**：通常录废（遥操员尚未启动就结束），直接删
* **> 120 s**：通常异常（机器人卡住、演示者迟疑），看视频确认后删

**理论依据**

LeRobot 训练时 `EpisodeAwareSampler` 按回合采样，时长方差大会导致 batch 内序列长度差异，增大 padding 开销和梯度方差，降低训练效率。

---

### 2.2 运动量（Lowest-Movement）→ 识别静止/无效回合

**判定标准**

| 进度条颜色 | totalMovement / max | 判读 | 处理 |
|---|---|---|---|
| 🔴 红 | < 15% | 几乎静止，机器人没动 | **直接 flag，删除** |
| 🟡 黄 | 15–40% | 运动量偏低 | 看视频确认，可能是 idle 任务 |
| 🟢 绿 | ≥ 40% | 正常运动量 | 保留 |

**推荐做法**

* 默认检查 Top 10 最低运动回合，红色直接 flag，黄色结合视频判断
* 静止/无效回合给策略提供"不要动"的信号，严重拖累成功率（输出趋零）

---

### 2.3 Action Autocorrelation → 决定 `chunk_size`

**判定方法**

* 找 ACF 曲线首次跌破 **0.5** 的 lag，该 lag 之后动作基本独立，open-loop 执行收益递减
* **Suggested chunk length** = 所有维度该 lag 的中位数 → 直接用作训练的 `chunk_size`

**推荐阈值**

| Suggested chunk | 解读 | 训练配置建议 |
|---|---|---|
| < 10 步 | 动作快速变化、独立性强 | `chunk_size = 8–16` |
| 10–30 步 | 中等耦合 | `chunk_size = 20–40` |
| > 30 步 | 动作高度连续 | `chunk_size = 50–100` |

**应该遵循的指标**

* **优先看 All Episodes 模式**：单回合噪声大，全数据集采样的 Suggested chunk 更稳健
* 若夹爪维度 lag 极短（2–5 步）而关节维度 lag 长（30+ 步），按主要关节设 chunk；夹爪可用离散建模
* 切勿直接用默认值（如固定 chunk=100）而不看此指标

**理论依据**

> Zhang et al. 2025 — *On the Theory of Imitation Learning with Action Chunking*. [arXiv:2507.09061](https://arxiv.org/abs/2507.09061)
>
> **Theorem 1**：最优 chunk 长度与系统稳定性常数（$L_\pi$、$C_\pi$）对数相关。自相关下降速度直接反映动作序列的记忆长度，ACF 跌破 0.5 的 lag 给出了"理论上 open-loop 执行不再有收益"的边界。

> ACT — Zhao et al. 2023 [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)：action chunking 的 chunk 长度本质是"未来多少步可以不依赖新观测预测"，自相关给出了这个边界的数据驱动估计。

---

### 2.4 State–Action Temporal Alignment → 诊断控制延迟

**判定方法**

| Mean control delay | 判读 | 处理 |
|---|---|---|
| 0 步 | 动作-状态完全对齐 | 无需额外处理 ✅ |
| 1–2 步 | 轻微延迟，可接受 | 酌情处理 |
| 3–5 步 | 明显延迟 | 考虑训练时把 `action[t]` 对齐 `state[t+lag]` |
| > 5 步 | 严重延迟 | **必须** chunk + RTC，否则 closed-loop 震荡 |

**应该遵循的指标**

* mean peak **r > 0.6**：action-state 因果关系明确，延迟估计可信
* r < 0.3：数据质量差或 state/action 维度不对应，需检查数据集 feature 映射
* max 与 min 峰值 lag 差 > 5 步：各维度延迟不一，需分维度对齐

**录制机制与内置延迟的来源**

LeRobot 录制循环在每帧内的执行顺序：

```
① obs  = robot.get_observation()     → 读编码器 → state[t]（follower arm 当前实际位置）
② act  = teleop.get_action()         → 读 leader arm → action[t]（目标位置指令）
③ robot.send_action(action[t])       → 写 Goal_Position，电机开始移动
④ dataset.add_frame({state[t], action[t]})   ← 同帧存入 Parquet
```

`state[t]` 和 `action[t]` 被存在同一帧，但物理含义不同：
* `state[t]` = 电机**此刻实际在哪**（编码器读数）
* `action[t]` = **此刻发出的目标位置**，电机要到 `t+lag` 帧才能到达

因此数据集中天然存在： `action[t] ≈ state[t + lag]`

**理想情况（lag = 1）**：servo 一个控制周期内完成响应，训练对齐良好。

**延迟较大时（lag = L）对训练的影响**：

```
训练样本：(state[t], action[t : t+chunk_size])

state[t] 在物理上是 action[t-L] 执行完毕后的结果
→ 模型看到的观测值，对应的是 L 步前发出的指令已经完成的状态
→ chunk 里的前 L 步（action[t : t+L]）是"追赶"动作，
  真正有意义的新运动从 action[t+L] 才开始

有效 chunk 长度 ≈ chunk_size − lag
```

如需修正，可在训练时将 action label 前移 L 步：

```python
# 未对齐（lag=L 时）
X: state[t],   Y: action[t : t+chunk_size]

# 对齐后
X: state[t],   Y: action[t-L : t-L+chunk_size]
# 使 Y[0] = action[t-L] ≈ state[t]，action 序列与观测物理对齐
```

**理论依据**

> ACT [Zhao et al. 2023](https://arxiv.org/abs/2304.13705)：chunk 是补偿控制延迟的核心机制，延迟大则需要更长 chunk 来平滑误差。
>
> Real-Time Chunking (RTC) [Black et al. 2025, arXiv:2506.07339](https://arxiv.org/abs/2506.07339)：显式建模 inference 延迟下的 chunk 重叠执行策略。
>
> Training-Time RTC [Black et al. 2025, arXiv:2512.05964](https://arxiv.org/abs/2512.05964)：训练时注入延迟噪声，提升真机部署鲁棒性。

---

### 2.5 Action Velocity → 评估动作平滑度 & 清洗 jerky 回合

**单维度判定**

| σ / max σ | 颜色 | 含义 | chunk 建议 |
|---|---|---|---|
| < 40% | 🟢 smooth | 平滑、可预测 | 可用长 chunk |
| 40–70% | 🟡 moderate | 中等抖动 | 中等 chunk |
| ≥ 70% | 🔴 jerky | 严重抖动，可能存在数据噪声 | 必须短 chunk |

**整体 Verdict 判定逻辑**

| Verdict | 条件 | 训练建议 |
|---|---|---|
| **Smooth** | ≥60% 活跃维度 smooth，且无非夹爪 jerky 维度 | 长 chunk，可直接训练 |
| **Moderate** | 非夹爪 jerky 维度 ≤2 个，且 ≥30% 维度 smooth | 中等 chunk，考虑清洗 |
| **Jerky** | 其余情况 | 短 chunk，先清洗 Most Jerky Episodes |

**应该遵循的指标**

* **Most Jerky Episodes Top 5–10%**：强烈建议 flag 删除（遥操手抖、传感器跳变、机器人卡顿）
* **σ > 0.05（非夹爪维度）**：通常是数据质量问题，看视频确认
* 夹爪 jerky 是正常的（二值开/关），verdict 自动忽略，无需处理

**理论依据**

> Zhang et al. 2025 [arXiv:2507.09061](https://arxiv.org/abs/2507.09061)
>
> - **Assumption 3.1（Lipschitz 常数 $L_\pi$）**：策略输出对输入扰动的敏感度
> - **Assumption 4.1（平滑度 $C_\pi$）**：动作序列的连续性
>
> 两者直接出现在 imitation learning 的 **compounding error bound** 中：$\sigma$ 越大代表 $L_\pi \cdot C_\pi$ 越大，误差在时序上累积越快，最终导致策略在真机上发散。

---

### 2.6 Demonstrator Speed Variance → 决定是否做速度归一化

**判定标准**

| CV (std / mean) | Verdict | 判读 | 行动 |
|---|---|---|---|
| < 0.2 | 🟢 Consistent | 速度一致，分布集中 | 无需归一化，可直接训练 |
| 0.2–0.4 | 🟡 Moderate variance | 有速度差异 | 建议归一化后再训练 |
| ≥ 0.4 | 🔴 High variance | 速度差异大 | **必须**做速度归一化 |

**应该遵循的指标**

* **多人录制数据集必看**：不同演示者速度差异会造成"假性多模态"，单模态策略（MSE）会输出均值速度，导致动作迟缓或振荡
* 单人录制的数据集 CV 通常 < 0.2，可跳过
* 速度归一化实现：对每个回合做时间重采样，使 mean `‖Δa‖` 落在统一目标区间

**理论依据**

> AGI-Bot 2025 — *Is Diversity All You Need? Velocity Normalization for VLA Fine-tuning*
>
> 实验表明对多人演示数据做速度归一化，fine-tune 成功率从 ~40% 提升至 ~70%（任务相关）。速度差异是任务无关因素，策略学的是"做什么"而非"多快做"，归一化可消除这一干扰。

---

### 2.7 Cross-Episode Action Variance Heatmap → 数据多样性诊断 & 策略选型

**判定方法**

| 色调 | 含义 | 对策 |
|---|---|---|
| 全冷（低方差） | 演示高度一致，单模态 | MSE / ACT 已够用；⚠️ 可能 coverage 不足，真机泛化差 |
| 全暖（高方差） | 多模态演示 | **必须**用生成式策略（diffusion / flow-matching / VLA） |
| 局部高方差峰 | 某时段存在策略歧义（如"抓 A 还是 B"） | 正常，属于任务内在多样性 |
| 某维度全程低方差 | 该维度从未被充分激活 | 检查 coverage：任务是否需要该维度？需补录 |

**应该遵循的指标**

选择策略时参考全图色调：

| 热图特征 | 推荐策略 |
|---|---|
| 大面积冷色 | ACT / DiffusionPolicy（MSE 模式）均可 |
| 大面积暖色 | SmolVLA / π0 / DiffusionPolicy（diffusion 模式）|
| 冷暖混合 | 生成式策略更稳健 |

**理论依据**

> Zhang et al. 2025 §4 (coverage discussion) [arXiv:2507.09061](https://arxiv.org/abs/2507.09061)
>
> 低方差区域若训练分布与测试分布偏移（如起始姿态略有不同），缺乏"恢复路径"会导致 compounding error 加剧。高方差区域需要生成式策略对多模态建模，否则单模态回归会取均值导致行为崩溃。

---

## Part 3 · 推荐数据清洗工作流

按以下顺序操作，系统化完成数据集清洗：

```
Step 1  Dataset Statistics
        ├── 看 Total Frames / Episodes / FPS 是否符合预期
        └── 看 Episode Length 直方图，初步判断分布形态

Step 2  Filtering · Episode Length Filter
        ├── 设定 [P5, P95] 范围
        └── 视频确认后 flag 离群回合

Step 3  Filtering · Lowest-Movement Episodes
        ├── 红色（<15%）直接 flag
        └── 黄色（15–40%）看视频确认

Step 4  Action Insights · Action Velocity → Most Jerky Episodes
        └── flag Top 5–10% jerky 回合（排除夹爪维度）

Step 5  执行清洗（复制侧边栏生成的 lerobot-edit-dataset 命令）
        ├── 原地删除：--repo_id <repo>
        └── 保留原始：--new_repo_id <repo>_filtered（推荐）

Step 6  切到 All Episodes 模式，完成训练配置诊断
        ├── Speed Variance CV   → 是否做速度归一化
        ├── Suggested chunk     → 设置 chunk_size
        ├── Alignment peak lag  → 是否做 action-state 时序偏移
        └── Variance Heatmap    → 选择单模态还是生成式策略

Step 7  重新可视化清洗后的数据集，确认指标改善
```

---

## 参考文献

| 论文 | 关联指标 |
|---|---|
| Zhang et al. 2025 — *On the Theory of Imitation Learning with Action Chunking*. [arXiv:2507.09061](https://arxiv.org/abs/2507.09061) | Autocorrelation (chunk_size)、Action Velocity (compounding error)、Variance Heatmap (coverage) |
| Zhao et al. 2023 — *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)*. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) | Autocorrelation、Temporal Alignment |
| Black et al. 2025a — *Real-Time Chunking (RTC)*. [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) | Temporal Alignment (control delay) |
| Black et al. 2025b — *Training-Time RTC*. [arXiv:2512.05964](https://arxiv.org/abs/2512.05964) | Temporal Alignment (delay compensation) |
| AGI-Bot 2025 — *Is Diversity All You Need? Velocity Normalization for VLA Fine-tuning* | Demonstrator Speed Variance |

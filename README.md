## 滑块验证 人工轨迹生成模型

### 一、 数据集的格式设计 (Feature Engineering)

要训练一个好模型，原始数据的清洗和特征提取比模型结构更重要。

#### 1. 采集原始数据
你需要编写一个简单的网页或小程序，人工手动去滑（建议用鼠标和手机触摸屏两种方式分别采集，因为鼠标轨迹和手指轨迹特征不同）。
采集的数据点应包含：
$$ P_i = (x_i, y_i, t_i) $$
其中 $t_i$ 是时间戳（毫秒级）。

#### 2. 数据预处理与格式化
直接把 $(x, y)$ 丢进模型效果通常不好，因为滑块的目标距离是变化的。你需要让模型学习**“运动趋势”**而不是**“绝对坐标”**。

建议的数据格式如下：

**输入 (Input / Condition):**
*   **Target Distance ($D$)**: 滑块需要移动的总距离（终点x - 起点x）。
*   *(可选)* **Vertical Offset**: 如果滑块有上下浮动，也可以输入y轴落差。

**输出/训练目标 (Sequence):**
不要训练绝对坐标，要训练**增量（Delta）**。
对于轨迹序列 $S = [p_0, p_1, ..., p_n]$，转换为增量序列：
$$ \Delta x_i = x_i - x_{i-1} $$
$$ \Delta y_i = y_i - y_{i-1} $$
$$ \Delta t_i = t_i - t_{i-1} $$

**最终数据集样本 (Sample):**
*   **Condition:** $[D_{target}]$ (归一化后的目标距离)
*   **Sequence:** $[[\Delta x_1, \Delta y_1, \Delta t_1], [\Delta x_2, \Delta y_2, \Delta t_2], ...]$

**注意点：**
1.  **归一化 (Normalization):** 将位移和时间归一化到 $[-1, 1]$ 或 $[0, 1]$ 区间，有利于模型收敛。
2.  **定长处理 (Padding):** 轨迹长度不一，需要设定一个最大长度（如 100个点），不足的补零，过长的截断。
3.  **对齐:** 确保所有的轨迹起始点都是从 $(0,0)$ 逻辑开始。

---

### 二、 深度学习算法选择

对于这种**生成序列数据**并通过**真假判别**的任务，最适合的架构是 **GAN (生成对抗网络)**，具体来说是 **Sequence GAN** 或 **LSTM-GAN**。

为什么选择 GAN 而不是简单的 LSTM/Transformer 预测？
*   **MSE Loss 的缺陷:** 如果只用 LSTM 做简单的回归预测（MSE Loss），模型倾向于生成“平均值”。即多条人手抖动的轨迹被平均后，变成了一条非常光滑的曲线。在反爬策略看来，光滑=机器。
*   **GAN 的优势:** GAN 包含一个生成器（Generator）和一个判别器（Discriminator）。判别器的作用**正如滑块验证码后台的风控系统**。通过对抗训练，Generator 生成的轨迹不仅符合物理规律，还保留了人类特有的“噪声”和“不规则性”。

#### 推荐模型架构：Conditional WGAN-GP (基于 LSTM 或 1D-CNN)

我们需要根据给定的距离生成轨迹，所以必须是 **Conditional (条件)** GAN。

#### 1. 生成器 (Generator)
*   **输入:** 随机噪声向量 $z$ (Latent vector) + 目标距离条件 $c$ (Condition)。
*   **结构:**
    *   将 $z$ 和 $c$ 此连接 (Concatenate)。
    *   经过全连接层 (Dense) 映射维度。
    *   接入 **LSTM / GRU 层** 或 **1D-Transposed Convolution** (反卷积) 层。
    *   **输出:** 序列形状 `(Batch_Size, Max_Len, 3)`，对应 $(\Delta x, \Delta y, \Delta t)$。
*   **目标:** 生成的序列累加后的总距离 $\approx$ 目标距离 $c$，且骗过判别器。

#### 2. 判别器 (Discriminator)
*   **输入:** 一条轨迹序列 `(Batch_Size, Max_Len, 3)` + 对应的目标距离条件 $c$。
*   **结构:**
    *   **1D-CNN** (推荐) 或 LSTM。1D-CNN 在提取轨迹局部特征（如微小的抖动模式）上非常有效。
    *   最后接入全连接层输出一个 Score (真/假)。
*   **目标:** 区分由于人工采集的真实轨迹和生成器生成的假轨迹。

#### 3. 损失函数的设计 (Critical)
为了保证生成的轨迹不仅像人，而且能准确滑到终点，Loss 需要组合：

$$ Loss_G = Loss_{Adversarial} + \lambda \cdot Loss_{Geometry} $$

1.  **对抗损失 ($Loss_{Adversarial}$):** 使用 WGAN-GP (Wasserstein GAN with Gradient Penalty) 的损失函数，训练极不稳定，WGAN-GP 能很好解决。
2.  **几何约束损失 ($Loss_{Geometry}$):**
    *   **终点误差:** 生成轨迹 $\sum \Delta x$ 必须等于目标距离 $D$。
    *   **物理合理性:** 惩罚过大的加速度（防止瞬移）。
    *   **Y轴漂移:** 惩罚过大的 $y$ 轴偏移（除非是特定需要上下滑动的验证码）。

---

### 三、 具体的训练与推理流程

#### 1. 训练阶段
1.  从人工数据集中采样一批真实轨迹 $(real\_seq, condition)$。
2.  生成随机噪声 $z$，结合 $condition$，通过 Generator 生成 $fake\_seq$。
3.  **训练判别器:** 让判别器识别 $real\_seq$ 为真，$fake\_seq$ 为假。
4.  **训练生成器:**
    *   固定判别器参数。
    *   输入噪声和条件，生成假轨迹。
    *   将假轨迹输入判别器，目标是让判别器打高分。
    *   同时计算几何损失（终点是否对齐），将误差回传。

#### 2. 推理阶段 (生成轨迹)
1.  获取当前滑块缺口的距离 $D_{target}$。
2.  构造输入条件 $c = Normalize(D_{target})$。
3.  构造随机噪声 $z$ (每次生成的随机噪声不同，轨迹就不同，完美解决风控的所谓“重放攻击”检测)。
4.  Generator 输出 $\Delta$ 序列。
5.  **反归一化 & 还原:**
    *   将 $\Delta x, \Delta y, \Delta t$ 还原为绝对数值。
    *   $x_t = x_{start} + \sum_{0}^{t} \Delta x_i$
    *   对最后几个点进行微调（Post-processing），强制修正终点坐标使其精确实配缺口（可以使用简单的插值法修正最后5%的路径）。
6.  使用 `Selenium` / `Puppeteer` / `Playwright` 执行轨迹。

---

### 四、 额外的技巧 (Tips)

1.  **过冲 (Overshoot) 现象:** 人类滑滑块通常会滑过头一点点，然后往回拉；或者在接近终点时明显减速。确保你的训练集中包含这种样本。如果模型学不到，可以在推理结束后，手动在轨迹末尾拼接一小段“回退”的操作。
2.  **Y轴抖动:** 即使是水平滑块，鼠标也会有轻微的上下抖动。千万不要生成 $y$ 轴完全为 0 的直线。
3.  **主要使用 RNN 还是 Transformer?** 对于这种短序列（通常<200个点），**LSTM** 或 **GRU** 往往比 Transformer 更容易训练且推理更快。
4.  **混合密度网络 (MDN):** 如果觉得 GAN 太难训练，可以尝试 **LSTM + MDN (Mixture Density Network)**。MDN 不输出具体的下一个点，而是输出下一个点可能出现的概率分布（高斯混合分布）。推理时从分布中采样。这天生带有随机性，且比 GAN 容易收敛。

### 总结
*   **数据格式:** 增量序列 $[(\Delta x, \Delta y, \Delta t), \dots]$。
*   **核心算法:** **Conditional WGAN-GP** (基于 LSTM 或 1D-CNN)。
*   **关键点:** 引入几何约束 Loss 确保滑块到达终点，利用噪声 $z$ 保证每次轨迹的唯一性。

---

### 五、 本项目使用说明

#### 1. 轨迹采集页面
*   打开 `web/index.html`（用浏览器直接打开或本地静态服务）。
*   拖动滑块完成验证，每次成功会记录一条轨迹；可多次滑动后点击 **「批量导出 JSON」** 下载 `trajectory_dataset.json`。
*   将导出的 JSON 按需放入：
    *   **训练集:** 把单条或批量 JSON 放入 `dataset/train/`（可多文件）。
    *   **测试集:** 放入 `dataset/test/`。
    *   **对比用:** 在采集页点击 **「导出单条 (compare 用)」** 得到 `index.json`，放入 `dataset/compare/index.json`。

#### 2. 训练
```bash
# 安装依赖后执行（需先有 dataset/train/*.json）
uv run python train.py --data-dir dataset --epochs 200 --out checkpoints
```
模型会保存为 `checkpoints/wgan.pt`。

#### 3. 对比绘图
```bash
uv run python main.py
```
会读取 `dataset/compare/index.json`（人工轨迹），并用当前模型生成同目标距离的轨迹（或使用已存在的 `dataset/compare/model.json`），绘制对比图并保存为 `dataset/compare/comparison.png`，包括：
*   位移-时间 (X vs T)
*   速度-时间 (Velocity vs Time)
*   加速度-时间 (Acceleration vs Time)
*   平均速度 vs 轨迹长度 (Avg Speed vs Distance)
*   抖动/方差 (Jitter vs Time)
*   加速度分布 (Acceleration Distribution)

#### 4. 人工行为校验（可选）
```bash
uv run python validate.py
```
会从 `dataset/train` 统计人工轨迹的核心指标（总时长、速度、加速度、抖动、Y 范围、回撤等），生成 `dataset/compare/human_stats.json`。若存在 `dataset/compare/model.json`，会对模型轨迹做一次通过/不通过判定。也可在代码中调用 `validate.compute_trajectory_metrics(points)` 与 `validate.check_trajectory_pass(points, human_stats)` 做推理后自检。
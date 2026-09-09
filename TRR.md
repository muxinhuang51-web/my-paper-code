Temporal Relational Ranking for Stock Prediction\
1809.09441

A股市场整体行业波动相关性远远强于美股

Kalman Filters
 
Autoregressive Models
# 核心痛点
随机过程方法的缺点：
* 随机过程不是一个好的假设
* 指标的挑选是专业化的，难以完成的
* 推理复杂性是随指标指数增长的

传统方法缺点：
* 价格回归是不重要的，涨跌的分类问题也是不重要的，学习的目标应该聚焦于排序
* LSTM或RNN只关注一条序列
* 图神经网络传统方法是随时间静态的，而股票市场是动态的
# 核心架构和调整
## TRR流程总览

```mermaid
flowchart TD
	A[过去S天股票特征 X_i^t] --> B[LSTM]
	B --> C[单股票时序嵌入 e_i^t]
	C --> D[TGC关系聚合]
	D --> E[关系增强嵌入 \bar e_i^t]
	C --> F[拼接]
	E --> F
	F --> G[全连接层 FC]
	G --> H[下一日排序分数 \hat r_i^{t+1}]
	H --> I[联合损失: 回归 + 排序]
	H --> J[按 Top1/Top5/Top10 选股]
	J --> K[t+1 卖出并统计收益]
```

LSTM输出时序
用 LSTM 读入每只股票输出时序嵌入                      

# LSTM公式流程（TRR）
输入是时间序列，模型需要记住短期波动和中期趋势。

只看单只股票时，最自然的做法是递归状态更新。

门控机制用来控制保留什么、忘掉什么，防止长序列梯度退化。
## 1) 输入序列构造
对第 i 只股票，在交易日 t 的输入为过去 S 天的特征序列：

$$
\mathbf{X}_i^t=[\mathbf{x}_{t-S+1},\ldots,\mathbf{x}_t]\in\mathbb{R}^{S\times D}
$$

文中日频特征可理解为 5 维：归一化收盘价、MA5、MA10、MA20、MA30。

### 特征序列怎么构造（从原始数据到 \(\mathbf{X}_i^t\)）
1. 取单只股票 i 的日频收盘价序列 \(\{p_i^\tau\}\)。
2. 对每个交易日 \(\tau\) 计算日特征向量：

$$
\mathbf{x}_{\tau}=
[\text{norm\_close}_{\tau},\text{MA5}_{\tau},\text{MA10}_{\tau},\text{MA20}_{\tau},\text{MA30}_{\tau}]\in\mathbb{R}^{D}
$$

其中 \(D=5\)，移动均线仅使用 \(\tau\) 及之前的历史价格计算。

3. 在预测时点 t 做长度为 S 的时间滑窗，按时间顺序拼接最近 S 天特征：

$$
\mathbf{X}_i^t=[\mathbf{x}_{t-S+1},\ldots,\mathbf{x}_t]\in\mathbb{R}^{S\times D}
$$

4. 用下一天真实收益作为监督标签：

$$
r_i^{t+1}=\frac{p_i^{t+1}-p_i^t}{p_i^t}
$$

因此训练样本是 \((\mathbf{X}_i^t, r_i^{t+1})\)，随 t 滚动生成全样本集。

## 2) 标准LSTM单元更新
对序列中的每个时刻 \(\tau\)，LSTM按以下公式更新：

$$
\mathbf{z}_{\tau}=\tanh(\mathbf{W}_z\mathbf{x}_{\tau}+\mathbf{Q}_z\mathbf{h}_{\tau-1}+\mathbf{b}_z)
$$

$$
\mathbf{i}_{\tau}=\sigma(\mathbf{W}_i\mathbf{x}_{\tau}+\mathbf{Q}_i\mathbf{h}_{\tau-1}+\mathbf{b}_i),\quad
\mathbf{f}_{\tau}=\sigma(\mathbf{W}_f\mathbf{x}_{\tau}+\mathbf{Q}_f\mathbf{h}_{\tau-1}+\mathbf{b}_f)
$$
仓位惯性和信息采纳
$$
\mathbf{c}_{\tau}=\mathbf{f}_{\tau}\odot\mathbf{c}_{\tau-1}+\mathbf{i}_{\tau}\odot\mathbf{z}_{\tau}
$$

$$
\mathbf{o}_{\tau}=\sigma(\mathbf{W}_o\mathbf{x}_{\tau}+\mathbf{Q}_o\mathbf{h}_{\tau-1}+\mathbf{b}_o),\quad
\mathbf{h}_{\tau}=\mathbf{o}_{\tau}\odot\tanh(\mathbf{c}_{\tau})
$$
## 3) 序列嵌入输出
TRR将最后一个隐藏状态作为股票 i 在时刻 t 的序列嵌入：

$$
\mathbf{e}_i^t=\mathbf{h}_i^t
$$

把全部股票堆叠为：

$$
\mathbf{E}_t=[\mathbf{e}_1^t,\ldots,\mathbf{e}_N^t]^\top\in\mathbb{R}^{N\times U}
$$

其中 N 是股票数，U 是LSTM隐藏维度。

## 4) 维度流
- 单只股票输入：\(S\times D\)
- 经过LSTM后每步隐藏态：\(U\)
- 取最后步得到该股票嵌入：\(U\)
- 全市场拼接后：\(N\times U\)

这一步输出 \(\mathbf{E}_t\) 会送入后续 TGC（Temporal Graph Convolution）做关系感知修正，再进入预测层。

# Temporal Graph Convolution
## 1) 关系输入定义
在时刻 t，除了 LSTM 产出的时序嵌入 \(\mathbf{E}_t\in\mathbb{R}^{N\times U}\)，
还需要股票关系张量：

$$
\mathcal{A}\in\mathbb{R}^{N\times N\times K}
$$

其中 \(\mathbf{a}_{ji}\in\mathbb{R}^{K}\) 表示股票 j 到 i 的多热关系向量，K 是关系类型数。

注释（多热关系向量）：
- 多热（multi-hot）表示一个向量中可以同时有多个 1。
- 对于股票对 \((j,i)\)，若第 k 类关系存在，则 \(a_{ji}^{(k)}=1\)，否则为 0。
- 由于两家公司可能同时满足多种关系（例如同产业 + 股权关系），所以不是 one-hot，而是 multi-hot。
- 这些关系来自外部结构化数据抽取（如行业分类关系、Wikidata 公司关系），并编码为 \(\mathbf{a}_{ji}\)。
- 在 TGC 中先用 \(\mathrm{sum}(\mathbf{a}_{ji})>0\) 判断是否连边，再把 \(\mathbf{a}_{ji}\) 输入关系强度函数 \(g(\cdot)\) 计算传播权重。

## 2) TGC的核心目标
把每只股票 i 的邻居股票 j 的信息聚合到 i 上，得到关系增强后的嵌入：

$$
\bar{\mathbf{e}}_i^t=\sum_{j:\,\mathrm{sum}(\mathbf{a}_{ji})>0}\frac{g(\mathbf{a}_{ji},\mathbf{e}_i^t,\mathbf{e}_j^t)}{d_j}\mathbf{e}_j^t
$$

解释：
- \(\mathrm{sum}(\mathbf{a}_{ji})>0\)：i 和 j 至少存在一种关系；
- \(d_j\)：归一化项（可理解为邻接规模）；
- \(g(\cdot)\)：关系强度函数，决定 j 对 i 的影响权重。

## 3) 两种关系强度建模（当下有效权重）
### 3.1 显式建模（RSR_E）

$$
g(\mathbf{a}_{ji},\mathbf{e}_i^t,\mathbf{e}_j^t)=
(\mathbf{e}_i^t)^\top\mathbf{e}_j^t\cdot\phi(\mathbf{w}^\top\mathbf{a}_{ji}+b)
$$

含义：
- 第一项是当前时刻两只股票的相似度；
- 第二项是关系类型重要性；
- 两者相乘得到时间敏感的关系传播强度。

### 3.2 隐式建模（RSR_I）

$$
g(\mathbf{a}_{ji},\mathbf{e}_i^t,\mathbf{e}_j^t)=
\phi\left(\mathbf{w}^\top[\mathbf{e}_i^{t\top},\mathbf{e}_j^{t\top},\mathbf{a}_{ji}^{\top}]^\top+b\right)
$$

含义：把股票对嵌入和关系向量拼接后交给一层非线性映射，
由模型自动学习复杂交互。

## Graph-based Learning
图学习的核心目标是：让相连实体的预测结果在图上尽量平滑，同时保留任务本身的监督信号。

### 1) 总目标函数

$$
\Gamma = \Omega + \lambda \Phi
$$

注释：
- \(\Gamma\)：整体优化目标；
- \(\Omega\)：任务相关损失，用来衡量预测值和真实标签之间的误差；
- \(\Phi\)：图正则项，用来约束图上相邻节点的预测不要差异过大；
- \(\lambda\)：平衡系数，控制任务损失和图平滑强度的权重。

### 2) 图平滑正则项

$$
\Phi = \sum_{i=1}^{N}\sum_{j=1}^{N} g(\mathbf{x}_i,\mathbf{x}_j)
\left\lVert
\frac{f(\mathbf{x}_i)}{\sqrt{D_{ii}}}-\frac{f(\mathbf{x}_j)}{\sqrt{D_{jj}}}
\right\rVert^2
$$

注释：
- \(\mathbf{x}_i\)：第 \(i\) 个实体的输入特征；
- \(f(\mathbf{x}_i)\)：模型对第 \(i\) 个实体输出的预测；
- \(g(\mathbf{x}_i,\mathbf{x}_j)\)：实体 \(i\) 和 \(j\) 之间的相似度或边权；
- \(D_{ii}=\sum_{j=1}^{N} g(\mathbf{x}_i,\mathbf{x}_j)\)：节点 \(i\) 的度；
- 归一化后的差值越小，表示相似节点的预测越一致。

### 3) 矩阵形式

$$
\mathcal{G}=\mathrm{trace}(\mathbf{Y}\mathbf{L}\mathbf{Y}^\top)
$$

$$
\mathbf{L}=\mathbf{D}^{-1/2}(\mathbf{D}-\mathbf{A})\mathbf{D}^{-1/2}
$$

注释：
- \(\mathcal{G}\)：图正则化项的矩阵写法；
- \(\mathbf{Y}=[\hat y_1,\hat y_2,\ldots,\hat y_N]\)：模型对所有节点的预测；
- \(\mathbf{A}\)：邻接矩阵，\(A_{ij}=g(\mathbf{x}_i,\mathbf{x}_j)\)；
- \(\mathbf{D}\)：度矩阵，对角线元素为各节点的度；
- \(\mathbf{L}\)：归一化图拉普拉斯矩阵；
- \(\mathrm{trace}(\cdot)\)：迹运算，用来把成对平滑项压缩成矩阵形式。

### 4) 金融含义

- \(\Omega\) 对应“单个股票预测要尽量准”，也就是基本的 alpha 预测能力。
- \(\Phi\) 对应“关系接近的股票，其收益或排序结果也应接近”，反映行业联动、供应链传导和风格共振。
- \(\lambda\) 越大，模型越依赖图结构；越小，模型越偏向单资产自身信号。

在金融里，这个思想等价于：如果两只股票在产业链、行业或控制关系上足够接近，那么它们的收益信号不应完全独立，模型应把这种联动作为先验纳入预测。

## 4) 预测层
将原始时序嵌入 \(\mathbf{e}_i^t\) 与关系嵌入 \(\bar{\mathbf{e}}_i^t\) 拼接后，
输入全连接层得到下一日排序分数：

$$
\hat{\mathbf{r}}^{t+1}=[\hat r_1^{t+1},\ldots,\hat r_N^{t+1}]^\top
$$

分数越高表示预期下一日收益率越高，排序越靠前。

# 联合损失函数（排序导向）
论文使用点式回归 + 对式排序的联合目标：

$$
\ell(\hat{\mathbf{r}}^{t+1},\mathbf{r}^{t+1})=
\|\hat{\mathbf{r}}^{t+1}-\mathbf{r}^{t+1}\|_2^2
+\alpha\sum_{i=1}^{N}\sum_{j=1}^{N}
\max\left(0,-(\hat r_i^{t+1}-\hat r_j^{t+1})(r_i^{t+1}-r_j^{t+1})\right)
$$

其中：
- \(\mathbf{r}^{t+1}\)：真实下一日收益率（1-day return ratio）；
- 第一项：拟合收益率绝对数值；
- 第二项：强制预测排序与真实排序一致；
- \(\alpha\)：两项权重平衡系数。

注释（\(\mathbf{r}\) 和 \(\hat{\mathbf{r}}\) 的来源）：
- \(r_i^{t+1}\) 是由真实价格计算出来的标签，定义为 \((p_i^{t+1}-p_i^t)/p_i^t\)，其中 \(p_i^t\) 是第 t 天收盘价，\(p_i^{t+1}\) 是下一天收盘价。
- \(\hat r_i^{t+1}\) 是模型输出的预测分数，由“LSTM 时序嵌入 + TGC 关系嵌入 + 全连接层”得到。
- 训练时同时使用这两个量：一方面让预测值尽量接近真实收益率，另一方面强制预测排序和真实排序一致。

# 完整训练与推理流程
## 训练阶段（每个交易日 t）
1. 构造每只股票过去 S 天输入 \(\mathbf{X}_i^t\)。
2. 共享参数的 LSTM 编码得到 \(\mathbf{E}_t\)。
3. 使用关系张量 \(\mathcal A\) 通过 TGC 得到 \(\bar{\mathbf{E}}_t\)。
4. 拼接 \(\mathbf{E}_t\) 与 \(\bar{\mathbf{E}}_t\)，经 FC 输出 \(\hat{\mathbf{r}}^{t+1}\)。
5. 用联合损失反向传播，更新 LSTM、TGC、FC 参数。

## 推理/交易阶段
1. 在收盘 t 时输入最近 S 天数据，预测 \(\hat{\mathbf{r}}^{t+1}\)。
2. 对股票按分数降序，得到候选买入列表。
3. 按 Top1/Top5/Top10 策略在 t 买入，t+1 收盘卖出。

# 一句话总结
TRR 的关键不是只用 LSTM 做时序预测，而是把 LSTM 的单股票时间信息，
通过 TGC 转成时间敏感的跨股票关系传播，再用排序损失直接对齐投资决策目标。

# 数据来源及其处理
1. 时序价格数据
2. 行业关系（是否在同一行业）
3. Wiki关系（上下游、股权、共同产品等）
## 语句处理得到的关系：
如果存在一条语句以 i 和 j 分别作为主语和宾语，公司 i 与 j 之间存在一阶关系。如果公司 i 和 j 的语句共享相同的宾语，则它们具有二阶关系

### Table 4（关系数据统计）怎么来的
这张表来自原论文 Section 4.2（Stock Relation Data）的统计结果：
- Sector-Industry Relation：基于 NASDAQ 官方公司分类层级（sector/industry）抽取“同产业”关系。
- Wiki Relation：基于 Wikidata（2018-01-05 dump）抽取公司一阶/二阶关系，再映射到股票对。
- Relation Types#：在该市场中实际出现的关系类型数量。
- Relation Ratio (Pairwise)：在全部股票对里，至少存在一种该类关系的股票对占比。

| Market | Sector-Industry Relation Types# | Sector-Industry Relation Ratio (Pairwise) | Wiki Relation Types# | Wiki Relation Ratio (Pairwise) |
| --- | ---: | ---: | ---: | ---: |
| NASDAQ | 112 | 5.00% | 42 | 0.21% |
| NYSE | 130 | 9.37% | 32 | 0.30% |

一句话解读：行业关系比 Wiki 关系稠密得多（约 5%-9% vs 0.2%-0.3%），Wiki 关系更稀疏但语义更细。

## 数据筛选与时间切分（原文关键口径）
### 1) 股票池筛选规则
- 原始收集区间：2013-01-02 到 2017-12-08。
- 初始股票数：NASDAQ 3274，NYSE 3163。
- 过滤条件：
	- 自 2013-01-02 起，交易覆盖率超过 98%；
	- 全区间内股价从未低于 5 美元（剔除 penny stocks）。
- 最终用于实验：NASDAQ 1026，NYSE 1737。

### 2) 时序特征与标签
- 频率：日频。
- 标签：1-day return ratio，
	\[
	r_i^{t+1}=\frac{p_i^{t+1}-p_i^t}{p_i^t}
	\]
- 特征：归一化收盘价 + MA5/MA10/MA20/MA30。

### 3) 时间切分（防止时间泄漏）
- 训练集：2013-2015（756个交易日）
- 验证集：2016（252个交易日）
- 测试集：2017-01-03 到 2017-12-08（237个交易日）

## 实验设置与评估协议（原文 Section 5.1）
### 1) 回测协议（daily buy-hold-sell）
- t 日收盘：模型预测 t+1 的排序并买入。
- t+1 日收盘：卖出上一日买入标的。
- 评估策略：Top1 / Top5 / Top10。

### 2) 回测假设
- 每天投入固定金额（消除资金路径依赖，便于公平比较）。
- 买卖都按收盘价成交（假设流动性充足）。
- 忽略交易成本。

### 3) 指标体系
- MSE：收益率数值误差。
- MRR：排名质量。
- IRR：累计投资收益率（论文主指标）。

## 对比基线与问题设置（RQ1/RQ2/RQ3）
### 1) 基线方法
- 回归范式：SFM、LSTM。
- 排序范式：Rank_LSTM（无关系层）。
- 图方法：GBR（图正则）、GCN（静态图卷积）。
- 本文方法：RSR_E（显式TGC）、RSR_I（隐式TGC）。

### 2) 三个研究问题
- RQ1：排序建模是否优于回归/分类思路。
- RQ2：关系信息是否有用，TGC 是否优于传统图方法。
- RQ3：Top1/Top5/Top10 不同交易策略下表现如何。

## 关键实验结论（原文结论压缩）
### 1) RQ1：排序目标有效
- Rank_LSTM 相比回归基线在 IRR 上整体更优，说明“对收益排序”比“仅回归价格”更贴近交易目标。

### 2) RQ2：关系信息有效，TGC优于静态图
- 在 NYSE（更稳定市场）中，关系建模收益更明显。
- RSR_E / RSR_I 通常优于 GCN / GBR，支持“关系强度随时间变化”的建模必要性。
- Industry 关系在 NASDAQ（更高波动）上不总是有效，关系类型与市场风格要匹配。

### 3) RQ3：交易策略差异
- 在关系有效设置下，常见现象是 Top1 > Top5 > Top10（收益上）。
- 但 Top1 波动与回撤风险更高，实务需结合风险约束。

## 复现参数与实现细节（缺失但重要）
### 1) 训练实现
- SFM 使用原作者实现；其余模型基于 TensorFlow。
- 调参目标以 IRR 为主。

### 2) 超参数搜索空间
- 对 LSTM/Rank/GCN/RSR：
	- 序列长度 \(S\in\{2,4,8,16\}\)
	- 隐藏维度 \(U\in\{16,32,64,128\}\)
	- 排序损失权重 \(\alpha\in\{0.1,1,10\}\)
- 对 GBR：图正则权重 \(\lambda\in\{0.1,1,10\}\)

### 3) 指标波动与稳定性
- 论文强调 IRR 对“极少数大涨日”敏感，轻微排名变化会引起大收益差。
- 建议复现时增加滚动窗口、多随机种子、分位数统计，而不只看均值。

## 论文边界与实盘落地风险
### 1) 数据与市场状态边界
- 测试区间偏牛市，跨周期泛化能力未充分验证。

### 2) 交易假设偏理想化
- 忽略手续费、冲击成本、滑点、停牌约束，会高估可实现收益。

### 3) 归一化口径的潜在泄漏风险
- 原文使用全区间最大值归一化价格，实务建议改为仅基于训练窗口统计量，避免未来信息泄漏。

### 4) 可改进方向
- 在目标函数中显式加入风险项（波动、回撤、换手惩罚）。
- 引入替代数据（新闻/社媒）并做时效性与泄漏隔离。
- 在熊市与震荡期做分市场状态回测，检验稳健性。
# 基于Gflow的自动化因子挖掘
arxiv.org/pdf/2509.25055 

![SAGE Overview](asset/SAGE%20overviwe.png)

## 传统自动化方法RL的缺点：
* 奖励稀疏问题：将奖励放在最终盈亏上，而夏普和回撤等约束需要一段较长时间，于是面临信用分配问题。现代方法只是将工程上几乎不可训，变成可以不能稳定得到超过baseline的因子。
* 强化学习数学表达式往往面临语意不足问题：公式里虽然有 $r_t$、$\pi_\theta$、$A_t$ 这类符号，但它们和真实交易语义之间的映射常常不够明确，比如“收益”到底包含未实现盈亏还是只算已实现盈亏，“风险”到底对应波动率、回撤还是仓位暴露，“动作”到底是调仓、下单还是执行算法。于是不同论文可能写出形式相似的目标函数，但优化的实际含义并不一致，导致模型看起来在学 RL，实质上却是在拟合一个语义不清的代理目标，复现和落地都容易偏离真实交易目标。
* 现代多因子模型追求多样化，低相关的alpha,而RL却会底层上追求单一奖励，最后往往收敛到单一期望最优
这三个缺陷会在我们后面的方法中得到解决

## 本文创新点：
* 引入GFlownet架构：逐步构建对象的一种方法
* 关系图卷积网络的编码器：从收集信息到构建节点。
* 多样化奖励结构设计

## 原理讲解：
* pipeline:寻找一种函数，将前t天的数据映射为指标。
1. GFLOWnet: 概率生成策略，按照奖励分布生成概率分布，然后再用这个概率分布采样alpha,而这个阿尔法是一个s0-s1的状态转移轨迹。
2. 状态s是一个语法树，s0是空状态，sn是末状态。sn具有完整轨迹，可以直接用于评分。
3. 语法树结构：普通的alpha结构是一个公式,按照一维序列读取难以区分输入特征，窗口参数，算子等结构。而使用树结构便于破解这一难题，子节点是ochl等参数和时间变量等，树边是计算依赖关系，根节点是顶层运算符，使用树结构的表示能力，我们得以区分公式的每一个结构，从而进行建模。
4. 为防止模型不生成叶子节点，为了奖励无限嵌套，我们设计强制停止机制。停止概率=节点数量/允许的最大节点量。
5. 损失函数：轨迹损失

$$
\mathcal{L}_{TB}(\tau) = \left(\log Z_{\theta} + \sum_{t=1}^{n} \log P_{F}(s_{t}\mid s_{t-1};\theta) - \log R(s_{n}) - \sum_{t=1}^{n} \log P_{B}(s_{t-1}\mid s_{t};\theta)\right)^{2}
$$
6. 编码器RCGN：一种处理不同边关系的方法。
对于每一层的节点特征向量hvl我们用如下公式进行更新,r在这里我们表示某一种关系，通过不同的矩阵Wr
$$
h_v^{(l)} = \mathrm{ReLU}\left(\sum_{r\in\mathcal{R}} \sum_{u\in\mathcal{N}_r(v)} \frac{1}{c_{v,r}} W_r^{(l)} h_u^{(l-1)} + W_0^{(l)} h_v^{(l-1)}\right),
$$
$$
e_{\alpha} = \mathrm{MaxPooling}\left(\{h_v^{(L)}\}_{v\in\mathcal{V}_{\alpha}}\right).
$$
是整个图的嵌入向量，代表整个图特征的综合关系，这里同时也给出了一种空间结构的表征。

7. 结构感知和关系感知：我们希望结构图长得很像的图生成的嵌入向量也很像

**行为距离与自适应权重**

$$
\begin{aligned}
&\text{设 } Z_i\in\mathbb{R}^{D\times N}\text{ 为因子 } \alpha_i \text{ 的时间序列向量， } Z_i(d)\in\mathbb{R}^N \text{ 为第 } d \text{ 日的输出。 定义行为距离：}\\
&d_{\mathrm{behav}}(\alpha_i,\alpha_j)=\frac{1}{D}\sum_{d=1}^D\|Z_i(d)-Z_j(d)\|^2 \tag{8}\\[6pt]
&w_{ij}=\frac{\exp\big(-\|e_{\alpha_i}-e_{\alpha_j}\|^2\big)}{\sum_{k\in\mathcal{N}_K(\alpha_i)}\exp\big(-\|e_{\alpha_i}-e_{\alpha_k}\|^2\big)},\quad j\in\mathcal{N}_K(\alpha_i) \tag{9}\\[6pt]
&R_{SA}(\alpha_i)=\exp\Big(-\sum_{j\in\mathcal{N}_K(\alpha_i)} w_{ij}\cdot d_{\mathrm{behav}}(\alpha_i,\alpha_j)\Big) \tag{10}
\end{aligned}
$$

其中 $\mathcal{N}_K(\alpha_i)$ 表示 $\alpha_i$ 的 $K$-近邻。

## 关键的奖励函数：三奖励加权和
* 终端性能奖励，作为关键指标使用IC（信息系数）

$$
R_{IC}(\alpha)=IC(\alpha,y)=\left|\mathbb{E}_d\left[\frac{\mathrm{Cov}(\alpha(X_d),y_d)}{\sqrt{\mathrm{Var}(\alpha(X_d))\cdot \mathrm{Var}(y_d)}}\right]\right|.
$$

* 结构感知奖励：我们希望alpha的结构和他的行为对齐，这里的结构使用了嵌入向量表征。
* 新颖性奖励

$$
R_{NOV}(\alpha)=1-\max_{\alpha'\in\mathcal{F}_{\mathrm{known}}}\left|IC(\alpha,\alpha')\right|.
$$
* 时序依赖的三奖励汇总：

$$
R(\alpha,T)=R_{IC}(\alpha)+\lambda(T)R_{SA}(\alpha)+\eta(T)R_{NOV}(\alpha),
$$
这里把终端性能、结构感知和新颖性三个奖励加权求和。

$$
\lambda(T)=\left(1-\frac{T}{T_{\mathrm{anneal}}}\right)\cdot\lambda_{\max},
$$
这个$\lambda(T)$会随时间减小，用来逐步降低结构感知奖励的权重。

$$
\eta(T)=\left(1-\frac{t}{T_{\mathrm{anneal}}}\right)\cdot\eta_{\max}.
$$

这个$\eta(T)$表示新颖性奖励的退火权重，训练过程中也会逐步下降。
$$
\mathcal{L}_{\mathrm{ENT}}=-\mathbb{E}_{\tau\sim P_F(\tau;\theta)}\left[\sum_{t=0}^{n-1} H\left(\pi_\theta(\cdot\mid s_t)\right)\right],
$$

这里的熵项鼓励策略在动作选择上保留更多探索空间。
$$
\mathcal{L}_{\mathrm{final}}=\mathbb{E}_{\tau\sim P_F(\tau;\theta)}\left[\mathcal{L}_{\mathrm{TB}}(\tau)\right]+\beta\cdot\mathcal{L}_{\mathrm{ENT}},
$$

最终目标是在轨迹损失基础上加入熵正则，$\beta$ 控制探索强度。
## 总结：
注意我们这里并没有固定alpha因子，而是构建了随时间变化的组合，近期有效的因子会随时间进行加权组合。这样我们不仅抛弃了时间变化后的冗余，也保留了相当的可解释性。
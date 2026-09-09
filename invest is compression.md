# 投资即压缩
https://arxiv.org/pdf/2604.10758
## 通用投资组合和通用压缩相关
## 三个问题：
金钱，熵，散度
唯一方法是最小化散度，散度是一种未知的矩阵或者摩擦，
为什么是压缩？
简单追求期望收益最大容易导致破产，于是我们需要某种非线性函数平衡风险与回报。
这里我们想要避免引入任意成本函数，而是将其比作引入外部价值并附着于信息论之上。
对数最优策略，为什么是对数的
不仅是渐进最优而且是博弈最优

## 凯利公式
f是投资比例，r是赢一次后资产翻为r倍
以概率 $p$ 赢得时，财富变为：
$$(1) \quad V_1 = V_0(fr + (1-f))$$

以概率 $(1-p)$ 输时，财富变为：
$$(2) \quad V_1 = V_0(1-f)$$

经过 $n$ 步后，如果赢了 $n_1$ 次，输了 $n_0$ 次，财富为：
$$(3) \quad V_n = V_0(1-f)^{n_0}(fr + (1-f))^{n_1}$$
因此单位步的平均对数增长率为：
$$g(f) = \lim_{n \to \infty} \frac{\log\left(\frac{V_n}{V_0}\right)}{n} = p\log(fr + (1-f)) + (1-p)\log(1-f)$$
我们期望收益的对数最优

$$
(6) \quad f^* = \frac{pr - 1}{r - 1}
$$

这是著名的“edge/odds” 凯利解，它最大化增长。把它代回增长方程：

$$
(7) \quad g^* = p\log(pr) + (1-p)\log\left(\frac{(1-p)r}{r-1}\right)
$$

如果市场概率是 $q$，并且赔率是公平的，那么 $q\cdot r = 1$，即 $r = \frac{1}{q}$。代入方程：

$$
(8) \quad g^* = p\log\left(\frac{p}{q}\right) + (1-p)\log\left(\frac{1-p}{1-q}\right)
$$

这就是我们的分布 $p$ 和市场分布 $q$ 之间的 KL 散度 $D_{\mathrm{KL}}(P\|Q)$。

## 考虑赛马游戏
$$
g(W) = \sum_{i=1}^{N} p_i \log(r_i w_i)
$$

$$
= \sum_{i=1}^{N} p_i \log(r_i) + \sum_{i=1}^{N} p_i \log(w_i)
$$

$$
= \sum_{i=1}^{N} p_i \log(r_i) + \sum_{i=1}^{N} p_i \log\left(\frac{p_i w_i}{p_i}\right)
$$

$$
= \sum_{i=1}^{N} p_i \log(r_i) + \sum_{i=1}^{N} p_i \log(p_i) + \sum_{i=1}^{N} p_i \log\left(\frac{w_i}{p_i}\right)
$$

$$
= \sum_{i=1}^{N} p_i \log(r_i) - \sum_{i=1}^{N} p_i \log\left(\frac{1}{p_i}\right) - \sum_{i=1}^{N} p_i \log\left(\frac{p_i}{w_i}\right)
$$

$$
= \sum_{i=1}^{N} p_i \log(r_i) - \mathrm{H}(P) - \mathrm{D}_{\mathrm{KL}}(P\|W)
$$
W是权重向量，注意到前两项与W无关，而第三项散度是非负项，只需取它为零

无关回报分布，只有关于概率



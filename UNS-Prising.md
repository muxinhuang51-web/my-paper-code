2601.00593
# what
machine learning in asset pricing

# performance
夏普提升
不优化组合权重，仅限于美股成熟市场（超额收益源于做空，A股做不了）

# why
point predictions have uncertainty

# theory
* predictive uncertainty as a first-class object
*

# experiment design(data features)



# how
machine learning
* linear models
* dimension-rreduction egressions
* tree-based methods
* deep neural networks
add prediction intervals to point predictions
* distribution-free
* residual -based
* estimated from dirction
多头上界空头下界
$$ 
PI_{i,t}(\alpha) = \left[ \hat{\mu}_{i,t} - \hat{q}_{i,t}(\alpha),\ \hat{\mu}_{i,t} + \hat{q}_{i,t}(\alpha) \right] 
$$

可以把它理解为：
* 多头更偏向“上界更高”的资产
* 空头更偏向“下界更低”的资产
* 目标不是优化权重，而是改进排序信号

## 论文结论
* 在美国股票样本上，相比只用点预测排序，这种不确定性调整后的排序能提升组合表现。
* 收益改善主要来自波动率下降，而不只是均值抬升。
* 即使不确定性信息不完整、甚至有一定 misspecification，方法仍然有效。
* 对更灵活的机器学习模型，收益提升更明显。
* 这种改进主要来自资产层面的预测不确定性，而不是时间维度或整体市场层面的不确定性。

## 总结
* 排序更稳
* long-short
* 本质上是在做带不确定性的 long-short ranking。

## 实务含义
* 适合能做空、且横截面选股有效的市场。
* 在短卖约束强的市场里，原文的收益形式可能不容易完全复制。

## 可复用关键词
machine learning, asset pricing, uncertainty-adjusted sorting, prediction interval, long-short portfolio, asset-specific uncertainty

## 结论的证明流程

### 实验设计
**数据层面**
* 美国股票市场，横截面选股有效的市场（理论上需要能做空）
* 对多个时期取样，检验时间稳定性

**对照组设置**
* | Baseline | 点预测排序构建 Long-Short 组合 |
* | 提议方法 | 不确定性调整的预测区间排序构建同等规模的 Long-Short 组合 |
* | 共同点 | 都是排序信号改变，不优化权重 |

### 关键实验结果验证

**第一步：主要表现对比**
* 计算两种排序方法下的超额收益和 Sharpe ratio
* 预期：不确定性调整排序显著更优
* 收益改善机制：主要来自波动率下降（risk reduction），而不是均值提升

**第二步：跨模型一致性**
* 在线性回归、树模型、深度神经网络上分别测试
* 预期：对更灵活（柔性更强）的 ML 模型，收益提升更明显
* 理由：复杂模型的预测误差分布更复杂，不确定性调整的价值更大

**第三步：Robustness 检验**
* 测试不完整或错误指定的不确定性信息下，方法是否仍有效
* 预期：即使不确定性估计不完全，只要相对排序正确，方法仍然有效
* 说明方法的鲁棒性强，不依赖精确的不确定性量化

### Identification：证明改进来自资产层面的不确定性

**问题**：不确定性可能来自三个层面
* Asset-specific：某资产本身难以预测
* Time-series：某时期全市场都难以预测  
* Aggregate：整体市场水平的预测难度

**论文的证明方法**：
1. **方差分解**：将预测不确定性分解为资产层、时间、残差等分量，验证**资产层占主要比例**
2. **交叉验证**：观察改进是否在不同资产间差异化（若只因时间因素，改进应相同；若资产层驱动，改进应差异大）
3. **假设检验**：用平均化的时间不确定性替代资产具体值，结果恶化，证明资产具体性很重要

**结论**：✅ 改进主要来自资产层面的预测不确定性

### 核心推论
* 这是**横截面相对优化**，而非规避市场系统风险
* 说明不同资产的可预测性差异存在，且这种差异被原方法忽视了
* 充分利用了模型给出的"置信度"差异


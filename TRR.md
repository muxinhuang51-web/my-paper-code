Temporal Relational Ranking for Stock Prediction\
1809.09441
# 核心痛点
价格回归是不重要的，涨跌的分类问题也是不重要的，学习的目标应该聚焦于排序
LSTM或RNN只关注一条序列
图神经网络传统方法是随时间静态的，而股票市场是动态的
# 核心架构和调整
LSTM输出时序
用 LSTM 读入每只股票输出时序嵌入
                                
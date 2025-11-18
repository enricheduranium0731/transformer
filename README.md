Transformer深度量化交易模型
项目概述
这是一个基于Transformer架构的深度量化交易模型，专门用于加密货币市场的实时交易决策。项目结合了传统技术指标分析和深度学习预测，实现多时间框架的自动化交易策略。

核心特性
🚀 技术特点
多时间框架分析: 支持15分钟到1日级别的多周期数据融合

Transformer架构: 使用自注意力机制捕捉市场长期依赖关系

技术指标集成: MACD、KDJ、RSI、布林带等20+技术指标

实时交易: 支持Binance交易所的实盘交易

风险控制: 内置止损止盈、仓位管理机制

1. 训练模式
bash
python transformer_trader.py {symbol} {data_years} train

2. 实盘交易模式
bash
python transformer_trader.py {symbol} {data_years} real

特征工程
输入特征 (68维)
4小时级别特征: OHLCV、MACD、KDJ、RSI、布林带、均线(5,10,30,50,100)

日线级别特征: 同上技术指标

形态特征: 吞没形态、影线识别、通道突破

位置特征: 相对高低点位置、超买超卖状态

数据预处理
标准化处理 (StandardScaler)

60时间步序列窗口

多时间框架数据对齐

交易策略
入场条件
多头信号:

价格变化预测 > 阈值 且 方向预测为上涨

通过传统技术指标验证

满足风险控制条件

空头信号:

价格变化预测 < -阈值 且 方向预测为下跌

通过传统技术指标验证

满足风险控制条件

出场条件
止损: 价格触及预设止损位

止盈: 价格触及预设止盈位

模型信号: 模型生成反向交易信号

文件说明
主要文件
transformer_trader.py: 主程序文件，包含数据获取、特征工程、模型训练和交易逻辑

DealTrader.py: 交易执行和订单管理模块

products--transformer.txt: 交易品种配置文件

数据文件
features/features-{symbol}.csv: 特征数据存储

transformer-models/: 训练好的模型和标准化器存储目录


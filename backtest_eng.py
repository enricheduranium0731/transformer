#!/usr/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
import os
import sys
import torch
import joblib
import common_eng
from features_eng import featuresThread
from data_eng import dataThread
from transformer_crypto_trading import ImprovedCryptoTransformer
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
label=sys.argv[1]
dataYear=float(sys.argv[2])   
run_type=sys.argv[3]

# =============== 1. 数据准备模块 ===============
class DataLoader:
    """高效数据加载器"""
    def __init__(self, symbol, data_dir="./test_data"):
        self.symbol = symbol
        self.data_dir = data_dir
    
    def load_feature_data(self, days=365):
        """加载特征数据"""
        file_path = f"{self.data_dir}/features-{self.symbol.lower()}.csv"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print(f"未找到特征文件，正在生成: {file_path}")
            thread1 = dataThread()
            common_eng.dict_symbol=thread1.initK(run_type,self.symbol, int(dataYear))
            thread2 = featuresThread()
            df= thread2.genFeatures(self.symbol, int(dataYear * 365 * 6 * 4),run_type)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def generate_features(self, days):
        """生成技术指标特征（简化版）"""
        thread1 = dataThread()
        thread1.initK(run_type,self.symbol, int(dataYear))
        thread2 = featuresThread()
        df = thread2.genFeatures(self.symbol, int(dataYear * 365 * 6 * 4),run_type)
        return df

# =============== 2. 回测引擎 ===============
class BacktestEngine:
    """完整的回测引擎"""
    
    def __init__(self, model, scaler, symbol, 
                 initial_capital=10000, commission=0.001, 
                 slippage=0.0005, threshold=0.01):
        """
        初始化回测引擎
        
        参数:
            model: 训练好的PyTorch模型
            scaler: 标准化器
            symbol: 交易对
            initial_capital: 初始资金
            commission: 手续费率 (0.1%)
            slippage: 滑点 (0.05%)
            threshold: 信号阈值
        """
        self.model = model
        self.scaler = scaler
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.threshold = threshold
        
        # 交易记录
        self.trades = []
        self.equity_curve = []
        
        # 策略参数
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.05  # 调整止盈为5%
        self.leverage = 5
        self.risk_pct = 0.01
        
    def prepare_sequences(self, df, seq_length=60):
        """准备模型输入序列"""
        # 特征列（排除时间戳和目标列）
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'future_return', 'direction']]
        
        # 标准化
        scaled_data = self.scaler.transform(df[feature_cols])
        
        X = []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i+seq_length])
        
        return np.array(X), df.iloc[seq_length:].reset_index(drop=True)
    
    def generate_signals(self, X, df):
        """生成交易信号"""
        self.model.eval()
        signals = []
        predictions = []
        
        with torch.no_grad():
            for i in tqdm(range(len(X)), desc="生成信号"):
                seq = torch.FloatTensor(X[i]).unsqueeze(0)
                price_change, dir_prob = self.model(seq)
                
                predictions.append({
                    'price_change': price_change.item(),
                    'direction': torch.argmax(dir_prob).item(),
                    'prob': torch.softmax(dir_prob, dim=1)[0].tolist()
                })
                
                # 信号逻辑
                pred = predictions[-1]
                if pred['price_change'] > self.threshold and pred['direction'] == 1:
                    signals.append('LONG')
                elif pred['price_change'] < -self.threshold and pred['direction'] == 0:
                    signals.append('SHORT')
                else:
                    signals.append('HOLD')
        
        # 添加到DataFrame
        df = df.copy().iloc[:len(signals)]
        df['signal'] = signals
        df['pred_price_change'] = [p['price_change'] for p in predictions]
        df['pred_direction'] = [p['direction'] for p in predictions]
        
        return df
    
    def simulate_trades(self, df):
        """模拟交易执行"""
        capital = self.initial_capital
        position = None
        entry_price = None
        
        for idx, row in df.iterrows():
            current_price = row['closeh4']
            signal = row['signal']
            
            # 检查持仓退出条件
            if position is not None:
                pnl = (current_price - entry_price) / entry_price
                if position == 'SHORT':
                    pnl = -pnl
                
                exit_signal = False
                reason = None
                
                # 止损/止盈（简化版）
                if pnl <= -self.stop_loss_pct:
                    exit_signal = True
                    reason = 'stop_loss'
                elif pnl >= self.take_profit_pct:
                    exit_signal = True
                    reason = 'take_profit'
                
                if exit_signal:
                    # 平仓
                    capital *= (1 + pnl * self.leverage * (1 - self.commission))
                    self.trades.append({
                        'entry_time': row['timestamp'],
                        'exit_time': row['timestamp'],
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl * self.leverage,
                        'reason': reason,
                        'equity': capital
                    })
                    position = None
            
            # 开新仓
            if position is None and signal != 'HOLD':
                # 计算滑点后的入场价
                slippage_factor = 1 + self.slippage if signal == 'LONG' else 1 - self.slippage
                position = signal
                entry_price = current_price * slippage_factor
            
            # 记录每日权益
            equity = capital
            if position is not None:
                unrealized_pnl = (current_price - entry_price) / entry_price
                if position == 'SHORT':
                    unrealized_pnl = -unrealized_pnl
                equity += capital * self.risk_pct * self.leverage * unrealized_pnl
            
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': equity,
                'position': position,
                'signal': signal
            })
        
        return pd.DataFrame(self.equity_curve)
    
    def calculate_performance(self, equity_df):
        """计算绩效指标"""
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # 总收益率
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 年化收益率（假设每日数据）
        days = (equity_df['timestamp'].iloc[-1] - equity_df['timestamp'].iloc[0]).days
        annual_return = (1 + total_return/100) ** (365/days) - 1 if days > 0 else 0
        
        # 夏普比率
        returns = equity_df['returns'].dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                              trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else np.inf
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        metrics = {
            '总收益率': f"{total_return:.2f}%",
            '年化收益率': f"{annual_return*100:.2f}%",
            '夏普比率': f"{sharpe:.2f}",
            '最大回撤': f"{max_drawdown:.2f}%",
            '胜率': f"{win_rate:.2f}%",
            '盈亏比': f"{abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A",
            '盈利因子': f"{profit_factor:.2f}",
            '交易次数': len(trades_df),
            '最终权益': f"${equity_df['equity'].iloc[-1]:.2f}"
        }
        
        return metrics, equity_df, trades_df
    
    def plot_results(self, equity_df, trades_df):
        """可视化回测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. 权益曲线
        axes[0].plot(equity_df['timestamp'], equity_df['equity'], 
                    label='策略权益', linewidth=2, color='#1f77b4')
        axes[0].axhline(y=self.initial_capital, color='red', 
                       linestyle='--', alpha=0.7, label='初始资金')
        axes[0].set_title(f'{self.symbol} 回测权益曲线', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('资金 (USDT)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 回撤曲线
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
        axes[1].fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                            alpha=0.5, color='red', label='回撤')
        axes[1].set_title('回撤分析', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('回撤 (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 交易分布
        if len(trades_df) > 0:
            trades_df['pnl_pct'] = trades_df['pnl'] * 100
            long_trades = trades_df[trades_df['position'] == 'LONG']
            short_trades = trades_df[trades_df['position'] == 'SHORT']
            
            axes[2].scatter(long_trades.index, long_trades['pnl_pct'], 
                           marker='^', color='green', s=80, label='做多', alpha=0.7)
            axes[2].scatter(short_trades.index, short_trades['pnl_pct'], 
                           marker='v', color='red', s=80, label='做空', alpha=0.7)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].set_title('单笔交易盈亏分布', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('交易序号')
            axes[2].set_ylabel('盈亏 (%)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest_{self.symbol}_{datetime.now().strftime("%Y%m%d")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run(self, df):
        print(f"开始回测: {self.symbol}")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"回测周期: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"{'='*50}\n")
        
        # 1. 准备数据
        print("步骤1: 准备序列数据...")
        X, df_seq = self.prepare_sequences(df)
        
        # 2. 生成信号
        print("\n步骤2: 生成交易信号...")
        df_signals = self.generate_signals(X, df_seq)
        
        # 3. 模拟交易
        print("\n步骤3: 模拟交易...")
        equity_df = self.simulate_trades(df_signals)
        
        # 4. 计算绩效
        print("\n步骤4: 计算绩效指标...")
        metrics, equity_df, trades_df = self.calculate_performance(equity_df)
        
        # 5. 打印结果
        print("\n" + "="*50)
        print("回测结果")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:>12}: {value}")
        
        # 6. 可视化
        print("\n步骤5: 生成图表...")
        self.plot_results(equity_df, trades_df)
        
        # 7. 保存结果
        equity_df.to_csv(f'equity_{self.symbol}.csv', index=False)
        trades_df.to_csv(f'trades_{self.symbol}.csv', index=False)
        
        return metrics
        
def main():
    CONFIG = {
        'symbol': label,
        'model_path': './transformer-models/transformer_model-'+label.lower()+'.pth',
        'scaler_path': './transformer-models/scaler-'+label.lower()+'.pkl',
        'initial_capital': 10000,
        'commission': 0.0005,  # 0.05%
        'slippage': 0.0001,    # 0.01%
        'threshold': 0.01
    }
    
    # 1. 加载模型
    print("加载Transformer模型...")
    checkpoint = torch.load(CONFIG['model_path'], map_location='cpu')

    # 检查保存的 embedding 维度
    embedding_weight = checkpoint['embedding.0.weight']
    actual_input_dim = embedding_weight.shape[1]

    print(f"保存的模型输入维度: {actual_input_dim}, 实际数据维度: {actual_input_dim}")    
    model = ImprovedCryptoTransformer(input_dim=actual_input_dim, d_model=128, nhead=8, num_layers=4)  # 根据您的特征数调整
    model.load_state_dict(torch.load(CONFIG['model_path']))
    model.eval()
    
    # 2. 加载标准化器
    print("加载标准化器...")
    scaler = joblib.load(CONFIG['scaler_path'])
    
    # 3. 加载数据
    print("加载市场数据...")
    data_loader = DataLoader(CONFIG['symbol'])
    df = data_loader.load_feature_data(days=180*2)  # 使用180天数据
    
    # 4. 创建回测引擎
    engine = BacktestEngine(
        model=model,
        scaler=scaler,
        symbol=CONFIG['symbol'],
        initial_capital=CONFIG['initial_capital'],
        commission=CONFIG['commission'],
        slippage=CONFIG['slippage'],
        threshold=CONFIG['threshold']
    )
    
    # 5. 运行回测
    metrics = engine.run(df)    
    return metrics

if __name__ == "__main__":
    os.makedirs('./transformer-models', exist_ok=True)
    os.makedirs('./features', exist_ok=True)
    
    results = main()
    
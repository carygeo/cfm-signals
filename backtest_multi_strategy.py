#!/usr/bin/env python3
"""
Multi-Strategy Backtester for CFM
Tests multiple strategies and parameter combinations to find optimal approach.

Built for Cary by Abel 🦞
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTPUT_DIR = Path(__file__).parent / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_historical_data(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical data from CoinGecko"""
    print(f"📥 Fetching {days} days of {coin_id} data...")
    
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    resp = requests.get(url, timeout=30)
    data = resp.json()
    
    prices = data.get('prices', [])
    
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['open'] = df['close'].shift(1)
    df['high'] = df['close'].rolling(2).max()
    df['low'] = df['close'].rolling(2).min()
    df.dropna(inplace=True)
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Rate of change
    df['roc'] = df['close'].pct_change(10) * 100
    
    return df

def backtest_strategy(df: pd.DataFrame, strategy: str, leverage: float = 4.0,
                      stop_loss: float = 0.03, take_profit: float = 0.06) -> dict:
    """
    Backtest various strategies
    """
    capital = 100.0
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    equity = [capital]
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        price = row['close']
        date = df.index[i]
        
        # Exit check
        if position != 0:
            if position == 1:
                pnl = (price - entry_price) / entry_price
            else:
                pnl = (entry_price - price) / entry_price
            
            if pnl <= -stop_loss or pnl >= take_profit:
                capital = capital * (1 + pnl * leverage)
                trades.append({
                    'entry': entry_date, 'exit': date, 
                    'dir': 'L' if position == 1 else 'S',
                    'entry_p': entry_price, 'exit_p': price,
                    'pnl': pnl * leverage * 100,
                    'win': pnl > 0
                })
                position = 0
        
        equity.append(capital if position == 0 else capital * (1 + pnl * leverage))
        
        # Entry signals by strategy
        if position == 0:
            signal = get_signal(strategy, row, prev, df.iloc[i-5:i])
            if signal == 1:
                position = 1
                entry_price = price
                entry_date = date
            elif signal == -1:
                position = -1
                entry_price = price
                entry_date = date
    
    # Close open position
    if position != 0:
        price = df.iloc[-1]['close']
        pnl = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
        capital = capital * (1 + pnl * leverage)
        trades.append({
            'entry': entry_date, 'exit': df.index[-1],
            'dir': 'L' if position == 1 else 'S',
            'entry_p': entry_price, 'exit_p': price,
            'pnl': pnl * leverage * 100,
            'win': pnl > 0
        })
    
    wins = sum(1 for t in trades if t['win'])
    total = len(trades)
    
    return {
        'strategy': strategy,
        'return': (capital - 100) / 100 * 100,
        'trades': total,
        'wins': wins,
        'win_rate': wins/total*100 if total > 0 else 0,
        'max_dd': min(0, min([(e/max(equity[:i+1])-1)*100 for i, e in enumerate(equity)])) if equity else 0,
        'trade_log': trades,
        'equity': equity
    }

def get_signal(strategy: str, row, prev, recent) -> int:
    """Get signal for a strategy (1=long, -1=short, 0=none)"""
    
    if strategy == 'RSI_EXTREME':
        # Simple RSI oversold/overbought
        if row['rsi'] < 30:
            return 1
        elif row['rsi'] > 70:
            return -1
            
    elif strategy == 'RSI_MODERATE':
        # Less extreme RSI levels
        if row['rsi'] < 40 and prev['rsi'] >= 40:  # RSI crosses below 40
            return 1
        elif row['rsi'] > 60 and prev['rsi'] <= 60:
            return -1
            
    elif strategy == 'SMA_CROSS':
        # SMA 10/20 crossover
        if row['sma_10'] > row['sma_20'] and prev['sma_10'] <= prev['sma_20']:
            return 1
        elif row['sma_10'] < row['sma_20'] and prev['sma_10'] >= prev['sma_20']:
            return -1
            
    elif strategy == 'MACD_CROSS':
        # MACD crosses signal
        if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            return 1
        elif row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            return -1
            
    elif strategy == 'MACD_ZERO':
        # MACD crosses zero
        if row['macd'] > 0 and prev['macd'] <= 0:
            return 1
        elif row['macd'] < 0 and prev['macd'] >= 0:
            return -1
            
    elif strategy == 'BB_BOUNCE':
        # Bollinger band bounce
        if row['close'] < row['bb_lower']:
            return 1
        elif row['close'] > row['bb_upper']:
            return -1
            
    elif strategy == 'TREND_RSI':
        # Trend following with RSI filter
        trend_up = row['sma_20'] > row['sma_50']
        trend_down = row['sma_20'] < row['sma_50']
        if trend_up and row['rsi'] < 45:
            return 1
        elif trend_down and row['rsi'] > 55:
            return -1
            
    elif strategy == 'MOMENTUM':
        # Rate of change momentum
        if row['roc'] > 5 and prev['roc'] <= 5:
            return 1
        elif row['roc'] < -5 and prev['roc'] >= -5:
            return -1
            
    return 0

def run_comprehensive_backtest():
    """Run all strategies across all assets"""
    
    print("=" * 70)
    print("🎯 CFM Multi-Strategy Backtester")
    print("=" * 70)
    
    assets = {
        'ethereum': 'ETH',
        'bitcoin': 'BTC',
    }
    
    strategies = [
        'RSI_EXTREME',
        'RSI_MODERATE', 
        'SMA_CROSS',
        'MACD_CROSS',
        'MACD_ZERO',
        'BB_BOUNCE',
        'TREND_RSI',
        'MOMENTUM'
    ]
    
    all_results = []
    
    for coin_id, symbol in assets.items():
        df = fetch_historical_data(coin_id, days=180)
        df = calculate_indicators(df)
        
        print(f"\n{'='*50}")
        print(f"📊 {symbol} Strategy Comparison")
        print('='*50)
        
        for strat in strategies:
            result = backtest_strategy(df, strat, leverage=4.0, stop_loss=0.03, take_profit=0.06)
            result['asset'] = symbol
            all_results.append(result)
            
            status = "✅" if result['return'] > 0 else "❌"
            print(f"{status} {strat:15} | Return: {result['return']:+7.1f}% | Trades: {result['trades']:3} | WR: {result['win_rate']:5.1f}% | MaxDD: {result['max_dd']:6.1f}%")
    
    # Find best strategies
    print("\n" + "=" * 70)
    print("🏆 TOP STRATEGIES")
    print("=" * 70)
    
    sorted_results = sorted(all_results, key=lambda x: x['return'], reverse=True)
    
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1}. {r['asset']:4} + {r['strategy']:15} | {r['return']:+7.1f}% | {r['trades']} trades | {r['win_rate']:.0f}% WR")
    
    # Generate comparison plot
    create_comparison_plot(all_results, strategies, list(assets.values()))
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'results': [
            {
                'asset': r['asset'],
                'strategy': r['strategy'],
                'return_pct': round(r['return'], 2),
                'trades': r['trades'],
                'win_rate': round(r['win_rate'], 1),
                'max_drawdown': round(r['max_dd'], 1)
            }
            for r in sorted_results
        ]
    }
    
    with open(OUTPUT_DIR / 'multi_strategy_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return sorted_results

def create_comparison_plot(results, strategies, assets):
    """Create visual comparison of all strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CFM Strategy Comparison - 6 Month Backtest (4x Leverage)', fontsize=14, fontweight='bold')
    
    # 1. Returns heatmap
    ax1 = axes[0, 0]
    returns_matrix = []
    for asset in assets:
        row = [next((r['return'] for r in results if r['asset']==asset and r['strategy']==s), 0) for s in strategies]
        returns_matrix.append(row)
    
    im = ax1.imshow(returns_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(len(assets)))
    ax1.set_yticklabels(assets)
    ax1.set_title('Return % by Strategy & Asset')
    plt.colorbar(im, ax=ax1, label='Return %')
    
    # Add text annotations
    for i in range(len(assets)):
        for j in range(len(strategies)):
            val = returns_matrix[i][j]
            color = 'white' if abs(val) > 25 else 'black'
            ax1.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=8)
    
    # 2. Trade count
    ax2 = axes[0, 1]
    for asset in assets:
        trades = [next((r['trades'] for r in results if r['asset']==asset and r['strategy']==s), 0) for s in strategies]
        ax2.plot(strategies, trades, marker='o', label=asset)
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('Trade Frequency')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Win rate
    ax3 = axes[1, 0]
    for asset in assets:
        wr = [next((r['win_rate'] for r in results if r['asset']==asset and r['strategy']==s), 0) for s in strategies]
        ax3.plot(strategies, wr, marker='s', label=asset)
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Win Rate %')
    ax3.set_title('Win Rate by Strategy')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # 4. Best performers bar chart
    ax4 = axes[1, 1]
    sorted_r = sorted(results, key=lambda x: x['return'], reverse=True)[:8]
    labels = [f"{r['asset']}\n{r['strategy']}" for r in sorted_r]
    values = [r['return'] for r in sorted_r]
    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax4.bar(range(len(labels)), values, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel('Return %')
    ax4.set_title('Top 8 Strategy/Asset Combinations')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    filename = OUTPUT_DIR / 'multi_strategy_comparison.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Comparison plot saved: {filename}")

if __name__ == "__main__":
    run_comprehensive_backtest()

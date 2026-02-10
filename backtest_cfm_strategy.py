#!/usr/bin/env python3
"""
CFM Strategy Backtester with Visualization
Proves out strategies with real historical data and generates plots.

Built for Cary by Abel 🦞
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import json

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "backtest_results"
OUTPUT_DIR.mkdir(exist_ok=True)

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_historical_data(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Fetch historical OHLC data from CoinGecko"""
    print(f"📥 Fetching {days} days of {coin_id} data...")
    
    url = f"{COINGECKO_API}/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"
    resp = requests.get(url, timeout=30)
    data = resp.json()
    
    prices = data.get('prices', [])
    
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Approximate OHLC from close prices (CoinGecko limitation)
    df['open'] = df['close'].shift(1)
    df['high'] = df['close'] * 1.02  # Approximate
    df['low'] = df['close'] * 0.98
    df.dropna(inplace=True)
    
    print(f"   Got {len(df)} daily candles")
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators"""
    # SMAs
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
    
    # ATR for volatility
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Support/Resistance (rolling)
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    
    return df

def backtest_combined_rsi_strategy(df: pd.DataFrame, 
                                    leverage: float = 4.0,
                                    stop_loss_pct: float = 0.03,
                                    take_profit_pct: float = 0.06,
                                    rsi_oversold: int = 35,
                                    rsi_overbought: int = 65) -> dict:
    """
    Backtest the COMBINED + RSI strategy optimized for CFM
    
    Entry LONG: Trend up (SMA20 > SMA50) + RSI < oversold + MACD turning up
    Entry SHORT: Trend down (SMA20 < SMA50) + RSI > overbought + MACD turning down
    """
    
    initial_capital = 100.0  # Normalized
    capital = initial_capital
    position = 0  # 1 = long, -1 = short, 0 = flat
    entry_price = 0
    
    trades = []
    equity_curve = []
    
    for i in range(50, len(df)):  # Start after indicators warm up
        row = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]
        price = row['close']
        
        # Track equity
        if position != 0:
            if position == 1:
                pnl_pct = (price - entry_price) / entry_price * leverage
            else:
                pnl_pct = (entry_price - price) / entry_price * leverage
            current_equity = capital * (1 + pnl_pct)
        else:
            current_equity = capital
        
        equity_curve.append({
            'date': date,
            'equity': current_equity,
            'price': price,
            'position': position
        })
        
        # Check exit conditions first
        if position != 0:
            if position == 1:
                pnl_pct = (price - entry_price) / entry_price
                # Stop loss or take profit
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    realized_pnl = pnl_pct * leverage
                    capital = capital * (1 + realized_pnl)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_leveraged': realized_pnl * 100,
                        'result': 'WIN' if pnl_pct > 0 else 'LOSS'
                    })
                    position = 0
                    
            elif position == -1:
                pnl_pct = (entry_price - price) / entry_price
                if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                    realized_pnl = pnl_pct * leverage
                    capital = capital * (1 + realized_pnl)
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': pnl_pct * 100,
                        'pnl_leveraged': realized_pnl * 100,
                        'result': 'WIN' if pnl_pct > 0 else 'LOSS'
                    })
                    position = 0
        
        # Check entry conditions
        if position == 0:
            trend_up = row['sma_20'] > row['sma_50']
            trend_down = row['sma_20'] < row['sma_50']
            rsi_low = row['rsi'] < rsi_oversold
            rsi_high = row['rsi'] > rsi_overbought
            macd_turning_up = row['macd_hist'] > prev['macd_hist'] and row['macd_hist'] > -0.5
            macd_turning_down = row['macd_hist'] < prev['macd_hist'] and row['macd_hist'] < 0.5
            
            # LONG signal
            if trend_up and rsi_low and macd_turning_up:
                position = 1
                entry_price = price
                entry_date = date
                
            # SHORT signal
            elif trend_down and rsi_high and macd_turning_down:
                position = -1
                entry_price = price
                entry_date = date
    
    # Close any open position at end
    if position != 0:
        price = df.iloc[-1]['close']
        if position == 1:
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price
        realized_pnl = pnl_pct * leverage
        capital = capital * (1 + realized_pnl)
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'direction': 'LONG' if position == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': price,
            'pnl_pct': pnl_pct * 100,
            'pnl_leveraged': realized_pnl * 100,
            'result': 'WIN' if pnl_pct > 0 else 'LOSS'
        })
    
    # Calculate metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df['drawdown'] = equity_df['equity'] / equity_df['equity'].cummax() - 1
    max_drawdown = equity_df['drawdown'].min() * 100
    
    wins = len([t for t in trades if t['result'] == 'WIN'])
    total = len(trades)
    win_rate = wins / total * 100 if total > 0 else 0
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return_pct': total_return,
        'total_trades': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': win_rate,
        'max_drawdown_pct': max_drawdown,
        'trades': trades,
        'equity_curve': equity_df,
        'leverage': leverage,
        'stop_loss': stop_loss_pct,
        'take_profit': take_profit_pct
    }

def plot_backtest_results(df: pd.DataFrame, results: dict, asset: str, strategy_name: str):
    """Generate comprehensive backtest visualization"""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1, 2]})
    fig.suptitle(f'{asset.upper()} - {strategy_name}\n{results["leverage"]}x Leverage | SL: {results["stop_loss"]*100:.0f}% | TP: {results["take_profit"]*100:.0f}%', 
                 fontsize=14, fontweight='bold')
    
    equity_df = results['equity_curve']
    trades = results['trades']
    
    # 1. Price chart with trades
    ax1 = axes[0]
    ax1.plot(df.index[-len(equity_df):], equity_df['price'], label='Price', color='gray', alpha=0.7)
    ax1.plot(df.index[-len(equity_df):], df['sma_20'].iloc[-len(equity_df):], label='SMA 20', color='blue', alpha=0.5)
    ax1.plot(df.index[-len(equity_df):], df['sma_50'].iloc[-len(equity_df):], label='SMA 50', color='orange', alpha=0.5)
    
    # Plot trade markers
    for trade in trades:
        color = 'green' if trade['result'] == 'WIN' else 'red'
        marker = '^' if trade['direction'] == 'LONG' else 'v'
        ax1.scatter(trade['entry_date'], trade['entry_price'], color=color, marker=marker, s=100, zorder=5)
        ax1.scatter(trade['exit_date'], trade['exit_price'], color=color, marker='x', s=100, zorder=5)
        ax1.plot([trade['entry_date'], trade['exit_date']], 
                [trade['entry_price'], trade['exit_price']], 
                color=color, linestyle='--', alpha=0.5)
    
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.set_title('Price Action & Trades')
    ax1.grid(True, alpha=0.3)
    
    # 2. RSI
    ax2 = axes[1]
    rsi_data = df['rsi'].iloc[-len(equity_df):]
    ax2.plot(df.index[-len(equity_df):], rsi_data, color='purple')
    ax2.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Oversold')
    ax2.axhline(y=65, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax2.fill_between(df.index[-len(equity_df):], 35, rsi_data.where(rsi_data < 35), color='green', alpha=0.3)
    ax2.fill_between(df.index[-len(equity_df):], 65, rsi_data.where(rsi_data > 65), color='red', alpha=0.3)
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. MACD
    ax3 = axes[2]
    macd_data = df['macd_hist'].iloc[-len(equity_df):]
    colors = ['green' if x >= 0 else 'red' for x in macd_data]
    ax3.bar(df.index[-len(equity_df):], macd_data, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_ylabel('MACD Histogram')
    ax3.grid(True, alpha=0.3)
    
    # 4. Equity curve
    ax4 = axes[3]
    ax4.fill_between(equity_df['date'], 100, equity_df['equity'], 
                     where=(equity_df['equity'] >= 100), color='green', alpha=0.3)
    ax4.fill_between(equity_df['date'], 100, equity_df['equity'], 
                     where=(equity_df['equity'] < 100), color='red', alpha=0.3)
    ax4.plot(equity_df['date'], equity_df['equity'], color='blue', linewidth=2, label='Portfolio')
    ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Starting Capital')
    
    # Add key metrics annotation
    metrics_text = (
        f"Return: {results['total_return_pct']:.1f}%\n"
        f"Max DD: {results['max_drawdown_pct']:.1f}%\n"
        f"Trades: {results['total_trades']}\n"
        f"Win Rate: {results['win_rate']:.1f}%"
    )
    ax4.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.set_xlabel('Date')
    ax4.legend(loc='upper right')
    ax4.set_title('Equity Curve')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = OUTPUT_DIR / f"{asset}_{strategy_name.replace(' ', '_')}_backtest.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plot saved: {filename}")
    return filename

def plot_comparison(all_results: dict):
    """Create comparison chart of all strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CFM Strategy Comparison - 6 Month Backtest', fontsize=14, fontweight='bold')
    
    assets = list(all_results.keys())
    returns = [all_results[a]['total_return_pct'] for a in assets]
    drawdowns = [abs(all_results[a]['max_drawdown_pct']) for a in assets]
    win_rates = [all_results[a]['win_rate'] for a in assets]
    trades = [all_results[a]['total_trades'] for a in assets]
    
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    # 1. Returns comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(assets, returns, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Total Return by Asset')
    for bar, val in zip(bars, returns):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom' if val > 0 else 'top')
    
    # 2. Max Drawdown
    ax2 = axes[0, 1]
    ax2.bar(assets, drawdowns, color='red', alpha=0.7)
    ax2.set_ylabel('Max Drawdown (%)')
    ax2.set_title('Maximum Drawdown by Asset')
    
    # 3. Win Rate
    ax3 = axes[1, 0]
    ax3.bar(assets, win_rates, color='blue', alpha=0.7)
    ax3.axhline(y=50, color='black', linestyle='--', alpha=0.3)
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate by Asset')
    ax3.set_ylim(0, 100)
    
    # 4. Number of Trades
    ax4 = axes[1, 1]
    ax4.bar(assets, trades, color='purple', alpha=0.7)
    ax4.set_ylabel('Number of Trades')
    ax4.set_title('Trade Frequency by Asset')
    
    plt.tight_layout()
    
    filename = OUTPUT_DIR / "strategy_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved: {filename}")
    return filename

def main():
    """Run backtests for all CFM-available assets"""
    
    print("=" * 60)
    print("🎯 CFM Strategy Backtester")
    print("=" * 60)
    
    # Assets to test (CFM available + comparison)
    assets = {
        'ethereum': 'ETH',
        'bitcoin': 'BTC',
        'avalanche-2': 'AVAX',  # For comparison (not on CFM)
        'near': 'NEAR',          # For comparison
    }
    
    all_results = {}
    
    for coin_id, symbol in assets.items():
        print(f"\n{'='*40}")
        print(f"Testing {symbol}...")
        print('='*40)
        
        # Fetch data
        df = fetch_historical_data(coin_id, days=180)  # 6 months
        df = calculate_indicators(df)
        
        # Run backtest
        results = backtest_combined_rsi_strategy(
            df,
            leverage=4.0,  # CFM leverage
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            rsi_oversold=35,
            rsi_overbought=65
        )
        
        all_results[symbol] = results
        
        # Print results
        print(f"\n📈 {symbol} Results:")
        print(f"   Return: {results['total_return_pct']:.1f}%")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.1f}%")
        print(f"   Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        
        if results['trades']:
            print(f"\n   Trade Log:")
            for t in results['trades'][-5:]:  # Last 5 trades
                print(f"   {t['direction']} | Entry: ${t['entry_price']:.2f} → Exit: ${t['exit_price']:.2f} | {t['pnl_leveraged']:.1f}% | {t['result']}")
        
        # Generate plot
        plot_backtest_results(df, results, symbol, "COMBINED_RSI_4x")
    
    # Comparison plot
    plot_comparison(all_results)
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'COMBINED_RSI',
        'leverage': 4.0,
        'stop_loss': 0.03,
        'take_profit': 0.06,
        'results': {
            symbol: {
                'return_pct': r['total_return_pct'],
                'max_drawdown_pct': r['max_drawdown_pct'],
                'total_trades': r['total_trades'],
                'win_rate': r['win_rate']
            }
            for symbol, r in all_results.items()
        }
    }
    
    with open(OUTPUT_DIR / 'backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Backtest Complete!")
    print(f"   Results saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    return all_results

if __name__ == "__main__":
    main()

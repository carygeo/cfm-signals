#!/usr/bin/env python3
"""
Hourly CFM Strategy Backtester
- Tests multiple strategies every hour
- Compares ALL strategies vs Buy & Hold
- Saves timestamped plots
- Maintains cumulative comparison table

Built for Cary by Abel 🦞
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import json
import csv

OUTPUT_DIR = Path(__file__).parent / "backtest_results"
PLOTS_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Strategy comparison table
STRATEGY_TABLE = OUTPUT_DIR / "strategy_comparison.csv"

COINGECKO_API = "https://api.coingecko.com/api/v3"

def fetch_historical_data(coin_id: str, days: int = 180) -> pd.DataFrame:
    """Fetch historical data from CoinGecko"""
    import time
    print(f"📥 Fetching {days} days of {coin_id} data...")
    
    # Small delay to avoid rate limiting
    time.sleep(1.5)
    
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

def calculate_buy_hold_return(df: pd.DataFrame, leverage: float = 1.0) -> float:
    """Calculate buy & hold return over the period"""
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    raw_return = (end_price - start_price) / start_price * 100
    return raw_return * leverage

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
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
    
    # ATR for volatility
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df

def get_signal(strategy: str, row, prev, recent) -> int:
    """Get signal for a strategy (1=long, -1=short, 0=none)"""
    
    if strategy == 'RSI_EXTREME':
        if row['rsi'] < 30:
            return 1
        elif row['rsi'] > 70:
            return -1
            
    elif strategy == 'RSI_MODERATE':
        if row['rsi'] < 40 and prev['rsi'] >= 40:
            return 1
        elif row['rsi'] > 60 and prev['rsi'] <= 60:
            return -1
            
    elif strategy == 'RSI_REVERSAL':
        # RSI bouncing from oversold
        if prev['rsi'] < 30 and row['rsi'] > 30:
            return 1
        elif prev['rsi'] > 70 and row['rsi'] < 70:
            return -1
            
    elif strategy == 'SMA_CROSS':
        if row['sma_10'] > row['sma_20'] and prev['sma_10'] <= prev['sma_20']:
            return 1
        elif row['sma_10'] < row['sma_20'] and prev['sma_10'] >= prev['sma_20']:
            return -1
            
    elif strategy == 'SMA_TREND':
        # Trade with longer term trend
        if row['close'] > row['sma_50'] and row['sma_10'] > row['sma_20']:
            return 1
        elif row['close'] < row['sma_50'] and row['sma_10'] < row['sma_20']:
            return -1
            
    elif strategy == 'MACD_CROSS':
        if row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            return 1
        elif row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            return -1
            
    elif strategy == 'MACD_ZERO':
        if row['macd'] > 0 and prev['macd'] <= 0:
            return 1
        elif row['macd'] < 0 and prev['macd'] >= 0:
            return -1
            
    elif strategy == 'MACD_HISTOGRAM':
        # Histogram momentum shift
        if row['macd_hist'] > 0 and prev['macd_hist'] <= 0:
            return 1
        elif row['macd_hist'] < 0 and prev['macd_hist'] >= 0:
            return -1
            
    elif strategy == 'BB_BOUNCE':
        if row['close'] < row['bb_lower']:
            return 1
        elif row['close'] > row['bb_upper']:
            return -1
            
    elif strategy == 'BB_BREAKOUT':
        # Breakout strategy
        if row['close'] > row['bb_upper'] and prev['close'] <= prev['bb_upper']:
            return 1
        elif row['close'] < row['bb_lower'] and prev['close'] >= prev['bb_lower']:
            return -1
            
    elif strategy == 'TREND_RSI':
        trend_up = row['sma_20'] > row['sma_50']
        trend_down = row['sma_20'] < row['sma_50']
        if trend_up and row['rsi'] < 45:
            return 1
        elif trend_down and row['rsi'] > 55:
            return -1
            
    elif strategy == 'MOMENTUM':
        if row['roc'] > 5 and prev['roc'] <= 5:
            return 1
        elif row['roc'] < -5 and prev['roc'] >= -5:
            return -1
            
    elif strategy == 'MOMENTUM_STRONG':
        if row['roc'] > 10 and prev['roc'] <= 10:
            return 1
        elif row['roc'] < -10 and prev['roc'] >= -10:
            return -1
            
    elif strategy == 'COMBO_MACD_RSI':
        # MACD cross + RSI filter
        macd_bull = row['macd'] > row['macd_signal'] and prev['macd'] <= prev['macd_signal']
        macd_bear = row['macd'] < row['macd_signal'] and prev['macd'] >= prev['macd_signal']
        if macd_bull and row['rsi'] < 60:
            return 1
        elif macd_bear and row['rsi'] > 40:
            return -1
            
    elif strategy == 'COMBO_BB_RSI':
        # BB bounce + RSI confirmation
        if row['close'] < row['bb_lower'] and row['rsi'] < 40:
            return 1
        elif row['close'] > row['bb_upper'] and row['rsi'] > 60:
            return -1
            
    elif strategy == 'MEAN_REVERT':
        # Mean reversion from SMA
        deviation = (row['close'] - row['sma_20']) / row['sma_20'] * 100
        if deviation < -3:
            return 1
        elif deviation > 3:
            return -1
            
    return 0

def backtest_strategy(df: pd.DataFrame, strategy: str, leverage: float = 4.0,
                      stop_loss: float = 0.03, take_profit: float = 0.06) -> dict:
    """Backtest a strategy"""
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
        
        pnl = 0
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
        
        current_equity = capital if position == 0 else capital * (1 + pnl * leverage)
        equity.append(current_equity)
        
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
    
    # Calculate max drawdown
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = (e - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
    
    return {
        'strategy': strategy,
        'return': (capital - 100),
        'trades': total,
        'wins': wins,
        'win_rate': wins/total*100 if total > 0 else 0,
        'max_dd': max_dd,
        'trade_log': trades,
        'equity': equity
    }

def update_strategy_table(results: list, buy_hold: dict, run_time: str):
    """Update cumulative strategy comparison table"""
    
    # Load existing data
    existing = []
    if STRATEGY_TABLE.exists():
        with open(STRATEGY_TABLE, 'r') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    
    # Add new results
    for r in results:
        bh = buy_hold.get(r['asset'], 0)
        vs_bh = r['return'] - bh
        
        existing.append({
            'timestamp': run_time,
            'asset': r['asset'],
            'strategy': r['strategy'],
            'return_pct': round(r['return'], 2),
            'buy_hold_pct': round(bh, 2),
            'vs_buy_hold': round(vs_bh, 2),
            'trades': r['trades'],
            'win_rate': round(r['win_rate'], 1),
            'max_drawdown': round(r['max_dd'], 1),
            'beats_hold': 'YES' if vs_bh > 0 else 'NO'
        })
    
    # Keep last 1000 rows (avoid file bloat)
    existing = existing[-1000:]
    
    # Write back
    with open(STRATEGY_TABLE, 'w', newline='') as f:
        fieldnames = ['timestamp', 'asset', 'strategy', 'return_pct', 'buy_hold_pct', 
                      'vs_buy_hold', 'trades', 'win_rate', 'max_drawdown', 'beats_hold']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    
    print(f"📊 Updated strategy table: {STRATEGY_TABLE}")

def create_equity_plots(top_results: list, asset_dfs: dict, buy_hold: dict, run_time: str):
    """Create individual equity curve plots for top strategies"""
    
    if not top_results:
        return
    
    timestamp = run_time.replace(':', '-').replace(' ', '_')
    equity_dir = PLOTS_DIR / 'equity_curves'
    equity_dir.mkdir(exist_ok=True)
    
    for i, result in enumerate(top_results):
        if 'equity' not in result or not result['equity']:
            continue
            
        asset = result['asset']
        strategy = result['strategy']
        equity = result['equity']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(equity, label=f'{strategy} (Return: {result["return"]:+.1f}%)', 
                color='green' if result['return'] > 0 else 'red', linewidth=2)
        
        # Add buy & hold reference line
        bh = buy_hold.get(asset, 0)
        bh_line = [100 * (1 + bh/100 * i/len(equity)) for i in range(len(equity))]
        ax.plot(bh_line, label=f'Buy & Hold ({bh:+.1f}%)', 
                color='blue', linestyle='--', alpha=0.7)
        
        # Formatting
        ax.axhline(y=100, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{asset} - {strategy}\n{run_time}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add stats box
        stats_text = f"Trades: {result['trades']}\nWin Rate: {result['win_rate']:.0f}%\nMax DD: {result['max_dd']:.1f}%"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save with timestamp and rank
        filename = equity_dir / f'{timestamp}_rank{i+1}_{asset}_{strategy}.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Equity curves saved: {equity_dir}")


def create_plots(results: list, strategies: list, assets: list, buy_hold: dict, run_time: str):
    """Create and save comparison plots"""
    
    if not results:
        print("⚠️ No results to plot")
        return None
    
    timestamp = run_time.replace(':', '-').replace(' ', '_')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'CFM Strategy Comparison - {run_time}\n4x Leverage vs Buy & Hold', 
                 fontsize=14, fontweight='bold')
    
    # 1. Top performers bar chart (instead of cluttered multi-asset bars)
    ax1 = axes[0, 0]
    
    # Get top 15 by excess return
    top_excess = sorted(results, key=lambda x: x['return'] - buy_hold.get(x['asset'], 0), reverse=True)[:15]
    labels = [f"{r['asset']}-{r['strategy'][:8]}" for r in top_excess]
    returns = [r['return'] for r in top_excess]
    bh_lines = [buy_hold.get(r['asset'], 0) for r in top_excess]
    colors = ['green' if r > 0 else 'red' for r in returns]
    
    x = np.arange(len(labels))
    ax1.bar(x, returns, color=colors, alpha=0.7, label='Strategy Return')
    ax1.scatter(x, bh_lines, color='blue', marker='_', s=200, linewidths=3, label='Buy & Hold', zorder=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Return %')
    ax1.set_title('Top 15 Strategies vs Buy & Hold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax1.set_ylabel('Return %')
    ax1.set_title('Strategy Returns vs Buy & Hold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Win Rate comparison
    ax2 = axes[0, 1]
    for asset in assets:
        wr = [next((r['win_rate'] for r in results if r['asset']==asset and r['strategy']==s), 0) for s in strategies]
        ax2.plot(strategies, wr, marker='o', label=asset, linewidth=2, markersize=8)
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% baseline')
    ax2.set_ylabel('Win Rate %')
    ax2.set_title('Strategy Win Rates')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # 3. Strategies that BEAT buy & hold
    ax3 = axes[1, 0]
    beat_bh = []
    for r in results:
        bh = buy_hold.get(r['asset'], 0)
        beat_bh.append({
            'label': f"{r['asset']}\n{r['strategy']}",
            'excess': r['return'] - bh,
            'strat_return': r['return'],
            'bh_return': bh
        })
    beat_bh = sorted(beat_bh, key=lambda x: x['excess'], reverse=True)[:12]
    
    labels = [b['label'] for b in beat_bh]
    excess = [b['excess'] for b in beat_bh]
    colors = ['green' if e > 0 else 'red' for e in excess]
    
    bars = ax3.barh(range(len(labels)), excess, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel('Excess Return vs Buy & Hold (%)')
    ax3.set_title('Best Strategies vs Buy & Hold')
    ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, excess)):
        ax3.text(val + (1 if val >= 0 else -1), i, f'{val:+.1f}%', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    
    # 4. Summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary
    total_strats = len(results)
    beat_count = sum(1 for r in results if r['return'] > buy_hold.get(r['asset'], 0))
    best = max(results, key=lambda x: x['return'])
    worst = min(results, key=lambda x: x['return'])
    
    summary_text = f"""
📊 SUMMARY - {run_time}
{'='*40}

Total Strategies Tested: {total_strats}
Strategies Beating Buy & Hold: {beat_count}/{total_strats} ({beat_count/total_strats*100:.0f}%)

🏆 BEST PERFORMER:
   {best['asset']} + {best['strategy']}
   Return: {best['return']:+.1f}% vs B&H: {buy_hold.get(best['asset'], 0):+.1f}%
   Excess: {best['return'] - buy_hold.get(best['asset'], 0):+.1f}%

📉 WORST PERFORMER:
   {worst['asset']} + {worst['strategy']}
   Return: {worst['return']:+.1f}% vs B&H: {buy_hold.get(worst['asset'], 0):+.1f}%

💰 BUY & HOLD RETURNS:
   ETH: {buy_hold.get('ETH', 0):+.1f}%
   BTC: {buy_hold.get('BTC', 0):+.1f}%

⚠️ Leverage: 4x | Stop Loss: 3% | Take Profit: 6%
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save with timestamp
    filename = PLOTS_DIR / f'strategy_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save as "latest"
    latest = OUTPUT_DIR / 'latest_comparison.png'
    plt.figure(figsize=(16, 12))
    fig.savefig(latest, dpi=150, bbox_inches='tight')
    
    print(f"📊 Plots saved: {filename}")
    return filename

def run_hourly_backtest():
    """Run comprehensive hourly backtest"""
    
    run_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    print("=" * 70)
    print(f"🎯 CFM Hourly Strategy Backtester - {run_time}")
    print("=" * 70)
    
    # Expanded asset list for CFM derivatives research
    # Core majors that CoinGecko reliably supports
    assets = {
        'bitcoin': 'BTC',
        'ethereum': 'ETH',
        'solana': 'SOL',
        'dogecoin': 'DOGE',
        'litecoin': 'LTC',
        'ripple': 'XRP',
        'binancecoin': 'BNB',
        'matic-network': 'MATIC',
    }
    
    strategies = [
        'RSI_EXTREME',
        'RSI_MODERATE',
        'RSI_REVERSAL',
        'SMA_CROSS',
        'SMA_TREND',
        'MACD_CROSS',
        'MACD_ZERO',
        'MACD_HISTOGRAM',
        'BB_BOUNCE',
        'BB_BREAKOUT',
        'TREND_RSI',
        'MOMENTUM',
        'MOMENTUM_STRONG',
        'COMBO_MACD_RSI',
        'COMBO_BB_RSI',
        'MEAN_REVERT'
    ]
    
    all_results = []
    buy_hold = {}
    asset_dfs = {}
    
    for coin_id, symbol in assets.items():
        try:
            df = fetch_historical_data(coin_id, days=180)
            if df.empty or len(df) < 60:
                print(f"⚠️ Skipping {symbol} - insufficient data")
                continue
            df = calculate_indicators(df)
            asset_dfs[symbol] = df
            
            # Calculate buy & hold return
            bh_return = calculate_buy_hold_return(df, leverage=1.0)  # No leverage for B&H
            buy_hold[symbol] = bh_return
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")
            continue
        
        print(f"\n{'='*50}")
        print(f"📊 {symbol} | Buy & Hold: {bh_return:+.1f}%")
        print('='*50)
        
        for strat in strategies:
            result = backtest_strategy(df, strat, leverage=4.0, stop_loss=0.03, take_profit=0.06)
            result['asset'] = symbol
            all_results.append(result)
            
            vs_bh = result['return'] - bh_return
            status = "✅" if vs_bh > 0 else "❌"
            print(f"{status} {strat:18} | Return: {result['return']:+7.1f}% | vs B&H: {vs_bh:+6.1f}% | Trades: {result['trades']:3} | WR: {result['win_rate']:5.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("🏆 TOP STRATEGIES (Beating Buy & Hold)")
    print("=" * 70)
    
    sorted_results = sorted(all_results, key=lambda x: x['return'] - buy_hold.get(x['asset'], 0), reverse=True)
    
    for i, r in enumerate(sorted_results[:10]):
        bh = buy_hold.get(r['asset'], 0)
        excess = r['return'] - bh
        print(f"{i+1}. {r['asset']:4} + {r['strategy']:18} | {r['return']:+7.1f}% | B&H: {bh:+.1f}% | Excess: {excess:+6.1f}%")
    
    # Create plots
    plot_file = create_plots(all_results, strategies, list(assets.values()), buy_hold, run_time)
    
    # Update strategy table
    update_strategy_table(all_results, buy_hold, run_time)
    
    # Save JSON results - both timestamped and latest
    summary = {
        'timestamp': run_time,
        'buy_hold': buy_hold,
        'assets_tested': list(buy_hold.keys()),
        'total_strategies': len(all_results),
        'beat_buy_hold': sum(1 for r in all_results if r['return'] > buy_hold.get(r['asset'], 0)),
        'results': [
            {
                'asset': r['asset'],
                'strategy': r['strategy'],
                'return_pct': round(r['return'], 2),
                'vs_buy_hold': round(r['return'] - buy_hold.get(r['asset'], 0), 2),
                'trades': r['trades'],
                'wins': r['wins'],
                'win_rate': round(r['win_rate'], 1),
                'max_drawdown': round(r['max_dd'], 1),
                'equity_curve': r.get('equity', [])[-20:] if r.get('equity') else []  # Last 20 points
            }
            for r in sorted_results
        ]
    }
    
    # Save timestamped JSON (historical archive)
    timestamp_str = run_time.replace(':', '-').replace(' ', '_')
    json_dir = OUTPUT_DIR / 'json_history'
    json_dir.mkdir(exist_ok=True)
    with open(json_dir / f'results_{timestamp_str}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save latest (for quick access)
    with open(OUTPUT_DIR / 'multi_strategy_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate equity curve plots for top 5 strategies
    create_equity_plots(sorted_results[:5], asset_dfs, buy_hold, run_time)
    
    # Log completion
    print(f"\n✅ Backtest complete. Plot: {plot_file}")
    
    # Auto-push to GitHub
    push_to_github(run_time)
    
    return sorted_results, buy_hold


def push_to_github(run_time: str):
    """Commit and push results to GitHub"""
    import subprocess
    
    try:
        repo_dir = Path(__file__).parent
        
        # Add all changes
        subprocess.run(['git', 'add', '-A'], cwd=repo_dir, capture_output=True)
        
        # Commit with timestamp
        commit_msg = f"Hourly research: {run_time}"
        result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            cwd=repo_dir, capture_output=True, text=True
        )
        
        if 'nothing to commit' in result.stdout or 'nothing to commit' in result.stderr:
            print("📤 No changes to push")
            return
        
        # Push to GitHub
        subprocess.run(['git', 'push'], cwd=repo_dir, capture_output=True)
        print(f"📤 Pushed to GitHub: {commit_msg}")
        
    except Exception as e:
        print(f"⚠️ GitHub push failed: {e}")


if __name__ == "__main__":
    run_hourly_backtest()

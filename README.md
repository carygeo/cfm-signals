# CFM Signals - Coinbase Derivatives Trading System

> **Built for Cary by Abel 🦞**
> **Style:** Swing trading (2-5 signals/week)
> **Hours:** 6am - 9pm EST
> **Platform:** Coinbase Financial Markets (manual execution)

---

## 🎯 Core Principles

### 1. NEVER GET LIQUIDATED
- Liquidation = total loss of position margin
- Always use stop losses BEFORE liquidation price
- Max risk per trade: 3% of account (with $89 = ~$2.67)

### 2. QUALITY OVER QUANTITY
- Wait for A+ setups only
- Multiple confirmations required
- No FOMO trades

### 3. RISK:REWARD MINIMUM 2:1
- Only take trades where potential profit ≥ 2x potential loss
- Example: Risk $2 to make $4+

---

## 📊 Strategy: Multi-Timeframe Swing

### Entry Conditions (ALL required):
1. **Daily trend** - Trading WITH the trend, not against
2. **4H structure** - Clear support/resistance level
3. **1H confirmation** - RSI oversold (<30) for longs, overbought (>70) for shorts
4. **Volume** - Above average on setup candle

### Exit Rules:
- **Stop Loss:** Always set, max 3% from entry
- **Take Profit:** Minimum 2:1 R:R, scale out at 1.5:1
- **Time Stop:** If trade goes nowhere for 48h, reassess

---

## 💰 Position Sizing (Critical)

### With $89 Account:

| Contract | Margin Req | Max Position | Risk at 3% SL |
|----------|------------|--------------|---------------|
| ETH Nano | ~$50 | 1 contract | ~$6.30 |
| BTC Nano | ~$165 | 0 (need funds) | N/A |

### Leverage Reality Check:
- ETH Nano at 4.2x leverage
- Entry $2,100, Liquidation ~$2,408 (14.7% move against)
- **Our stop loss: 2-3%** = closes WELL before liquidation

---

## 📱 Signal Format

```
🟢 ETH LONG SIGNAL

📍 ENTRY ZONE: $2,050 - $2,060
🛑 STOP LOSS: $2,010 (-2%)  
🎯 TAKE PROFIT 1: $2,120 (+3%) - Close 50%
🎯 TAKE PROFIT 2: $2,160 (+5%) - Close remaining

⚖️ Risk: $4.00 | Reward: $8-12 | R:R: 2-3:1
⏰ Valid for: 2 hours
📊 Confidence: HIGH

💡 REASON:
- Daily uptrend intact
- 4H bounce off $2,040 support (tested 3x)
- 1H RSI at 28 (oversold)
- Volume spike on support touch

📋 EXECUTION STEPS:
1. Go to CFM → ETH 27 FEB 26
2. Click "Buy | Long"  
3. Set Amount: 1 contract
4. Use LIMIT order at $2,055 (mid-zone)
5. Click "Buy"
6. Immediately add TP/SL:
   - Stop Loss: $2,010
   - Take Profit: $2,120
   - Amount: 1
7. Confirm position in Positions tab
```

---

## 🔴 What Makes Me NOT Signal

- Choppy/sideways market (no clear trend)
- Major news event imminent (FOMC, CPI, etc.)
- Weekend low liquidity (Sat night - Sun morning)
- Already have open position (1 at a time with this account size)
- Setup is "okay" but not "great"

---

## 📈 Performance Tracking

Every signal logged with:
- Entry price (actual)
- Exit price (actual)  
- P&L in $ and %
- Hold time
- Notes on execution

Weekly review every Sunday.

---

## ⚠️ Risk Warnings

1. **Futures are leveraged** - losses magnified
2. **Manual execution** - slippage possible
3. **24/7 market** - can move overnight
4. **This is real money** - trade only what you can lose

---

## 🛠️ Technical Setup

- Market data: CoinGecko API (free)
- Analysis: Python + pandas + ta-lib
- Alerts: Telegram via Clawdbot
- Monitoring: Cron every 30 min (6am-9pm)

---

*System built 2026-02-09. Let's make money.* 🦞

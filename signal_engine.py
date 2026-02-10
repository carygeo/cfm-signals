#!/usr/bin/env python3
"""
CFM Signal Engine - Coinbase Derivatives Trading Signals
Built for Cary by Abel 🦞

Generates high-quality swing trade signals for manual execution on CFM.
"""

import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List
import os

# Configuration
COINGECKO_API = "https://api.coingecko.com/api/v3"
BINANCE_API = "https://api.binance.com/api/v3"

ASSETS = {
    "ETH": {
        "coingecko_id": "ethereum",
        "binance_symbol": "ETHUSDT",
        "cfm_contract": "ETH 27 FEB 26",
        "contract_size": 0.1,
        "margin_required": 50,
        "leverage": 4.2,
    },
    "BTC": {
        "coingecko_id": "bitcoin", 
        "binance_symbol": "BTCUSDT",
        "cfm_contract": "BTC 27 FEB 26",
        "contract_size": 0.01,
        "margin_required": 165,
        "leverage": 4.3,
    }
}

# Risk parameters
MAX_RISK_PERCENT = 3.0  # Max 3% stop loss
MIN_RR_RATIO = 2.0  # Minimum 2:1 reward:risk
ACCOUNT_BALANCE = 89.0  # Current balance

@dataclass
class MarketData:
    symbol: str
    price: float
    change_24h: float
    high_24h: float
    low_24h: float
    volume_24h: float
    timestamp: datetime

@dataclass  
class Signal:
    direction: str  # "LONG" or "SHORT"
    asset: str
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_amount: float
    reward_amount: float
    rr_ratio: float
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    reasons: List[str]
    valid_until: datetime

class CFMSignalEngine:
    def __init__(self):
        self.last_signal_time = None
        self.open_position = None
        self.state_file = os.path.expanduser("~/clawd/cfm_signals/state.json")
        self.load_state()
    
    def load_state(self):
        """Load persistent state"""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.last_signal_time = state.get('last_signal_time')
                self.open_position = state.get('open_position')
        except FileNotFoundError:
            pass
    
    def save_state(self):
        """Save persistent state"""
        state = {
            'last_signal_time': self.last_signal_time,
            'open_position': self.open_position,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_price_data(self, symbol: str = "ETH") -> Optional[MarketData]:
        """Fetch current price data from CoinGecko"""
        try:
            asset = ASSETS.get(symbol)
            if not asset:
                return None
            
            # Get ticker data from CoinGecko
            url = f"{COINGECKO_API}/simple/price?ids={asset['coingecko_id']}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            
            coin_data = data.get(asset['coingecko_id'], {})
            price = coin_data.get('usd', 0)
            change_24h = coin_data.get('usd_24h_change', 0)
            
            # Get OHLC for high/low (use klines instead)
            return MarketData(
                symbol=symbol,
                price=float(price),
                change_24h=float(change_24h),
                high_24h=float(price * 1.02),  # Approximate
                low_24h=float(price * 0.98),   # Approximate
                volume_24h=float(coin_data.get('usd_24h_vol', 0)),
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error fetching price data: {e}")
            return None
    
    def get_klines(self, symbol: str = "ETH", interval: str = "4h", limit: int = 100) -> List[dict]:
        """Fetch OHLCV data from CoinGecko"""
        try:
            asset = ASSETS.get(symbol)
            if not asset:
                return []
            
            # CoinGecko market_chart - get last 14 days for 4h candles
            days = 14 if interval == "4h" else 7
            url = f"{COINGECKO_API}/coins/{asset['coingecko_id']}/market_chart?vs_currency=usd&days={days}"
            resp = requests.get(url, timeout=15)
            data = resp.json()
            
            prices = data.get('prices', [])
            if not prices:
                return []
            
            # Convert to kline format (approximate - CoinGecko gives price points, not OHLC)
            # Group into 4-hour or 1-hour buckets
            bucket_hours = 4 if interval == "4h" else 1
            bucket_ms = bucket_hours * 60 * 60 * 1000
            
            klines = []
            current_bucket = []
            bucket_start = None
            
            for price_point in prices:
                ts, price = price_point[0], price_point[1]
                
                if bucket_start is None:
                    bucket_start = ts
                
                if ts - bucket_start < bucket_ms:
                    current_bucket.append(price)
                else:
                    if current_bucket:
                        klines.append({
                            'timestamp': datetime.fromtimestamp(bucket_start / 1000),
                            'open': current_bucket[0],
                            'high': max(current_bucket),
                            'low': min(current_bucket),
                            'close': current_bucket[-1],
                            'volume': 0  # CoinGecko doesn't give volume per candle
                        })
                    bucket_start = ts
                    current_bucket = [price]
            
            # Add final bucket
            if current_bucket:
                klines.append({
                    'timestamp': datetime.fromtimestamp(bucket_start / 1000),
                    'open': current_bucket[0],
                    'high': max(current_bucket),
                    'low': min(current_bucket),
                    'close': current_bucket[-1],
                    'volume': 0
                })
            
            return klines[-limit:] if len(klines) > limit else klines
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return []
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0  # Neutral if not enough data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def find_support_resistance(self, klines: List[dict], lookback: int = 20) -> dict:
        """Find key support and resistance levels"""
        if len(klines) < lookback:
            return {'support': 0, 'resistance': 0}
        
        recent = klines[-lookback:]
        lows = [k['low'] for k in recent]
        highs = [k['high'] for k in recent]
        
        # Simple: use recent swing low/high
        support = min(lows)
        resistance = max(highs)
        
        return {
            'support': support,
            'resistance': resistance,
            'range': resistance - support
        }
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period
    
    def detect_trend(self, klines: List[dict]) -> str:
        """Detect overall trend using SMAs"""
        if len(klines) < 50:
            return "NEUTRAL"
        
        closes = [k['close'] for k in klines]
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        current_price = closes[-1]
        
        if current_price > sma_20 > sma_50:
            return "BULLISH"
        elif current_price < sma_20 < sma_50:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def analyze_for_signal(self, symbol: str = "ETH") -> Optional[Signal]:
        """Main analysis function - looks for trade setups"""
        
        # Get market data
        market = self.get_price_data(symbol)
        if not market:
            return None
        
        klines_4h = self.get_klines(symbol, "4h", 100)
        klines_1h = self.get_klines(symbol, "1h", 50)
        
        if not klines_4h or not klines_1h:
            return None
        
        # Calculate indicators
        closes_4h = [k['close'] for k in klines_4h]
        closes_1h = [k['close'] for k in klines_1h]
        
        rsi_4h = self.calculate_rsi(closes_4h)
        rsi_1h = self.calculate_rsi(closes_1h)
        trend = self.detect_trend(klines_4h)
        sr_levels = self.find_support_resistance(klines_4h)
        
        current_price = market.price
        reasons = []
        
        # Check for LONG setup
        if self._check_long_setup(current_price, rsi_1h, rsi_4h, trend, sr_levels, market, reasons):
            return self._create_long_signal(symbol, current_price, sr_levels, reasons)
        
        # Check for SHORT setup
        if self._check_short_setup(current_price, rsi_1h, rsi_4h, trend, sr_levels, market, reasons):
            return self._create_short_signal(symbol, current_price, sr_levels, reasons)
        
        return None
    
    def _check_long_setup(self, price, rsi_1h, rsi_4h, trend, sr, market, reasons) -> bool:
        """Check if conditions are met for a LONG signal"""
        score = 0
        
        # 1. Trend alignment (not shorting into uptrend)
        if trend == "BULLISH":
            score += 2
            reasons.append(f"Daily trend BULLISH (SMA20 > SMA50)")
        elif trend == "NEUTRAL":
            score += 1
            reasons.append("Trend neutral - acceptable for bounce play")
        else:
            return False  # Don't long in bearish trend
        
        # 2. RSI oversold
        if rsi_1h < 35:
            score += 2
            reasons.append(f"1H RSI oversold at {rsi_1h:.1f}")
        elif rsi_1h < 45:
            score += 1
            reasons.append(f"1H RSI showing weakness at {rsi_1h:.1f}")
        
        # 3. Near support
        support_distance = (price - sr['support']) / price * 100
        if support_distance < 2:
            score += 2
            reasons.append(f"Price near support ${sr['support']:.2f} ({support_distance:.1f}% away)")
        elif support_distance < 4:
            score += 1
            reasons.append(f"Price approaching support ${sr['support']:.2f}")
        
        # 4. Volume confirmation
        if market.change_24h < -3:
            score += 1
            reasons.append(f"Oversold on 24h ({market.change_24h:.1f}%), bounce likely")
        
        # Need score >= 5 for signal
        return score >= 5
    
    def _check_short_setup(self, price, rsi_1h, rsi_4h, trend, sr, market, reasons) -> bool:
        """Check if conditions are met for a SHORT signal"""
        score = 0
        
        # 1. Trend alignment
        if trend == "BEARISH":
            score += 2
            reasons.append(f"Daily trend BEARISH (SMA20 < SMA50)")
        elif trend == "NEUTRAL":
            score += 1
            reasons.append("Trend neutral - acceptable for rejection play")
        else:
            return False  # Don't short in bullish trend
        
        # 2. RSI overbought
        if rsi_1h > 65:
            score += 2
            reasons.append(f"1H RSI overbought at {rsi_1h:.1f}")
        elif rsi_1h > 55:
            score += 1
            reasons.append(f"1H RSI elevated at {rsi_1h:.1f}")
        
        # 3. Near resistance
        resistance_distance = (sr['resistance'] - price) / price * 100
        if resistance_distance < 2:
            score += 2
            reasons.append(f"Price near resistance ${sr['resistance']:.2f} ({resistance_distance:.1f}% away)")
        elif resistance_distance < 4:
            score += 1
            reasons.append(f"Price approaching resistance ${sr['resistance']:.2f}")
        
        # 4. Momentum confirmation
        if market.change_24h > 3:
            score += 1
            reasons.append(f"Overextended on 24h (+{market.change_24h:.1f}%), pullback likely")
        
        return score >= 5
    
    def _create_long_signal(self, symbol, price, sr, reasons) -> Signal:
        """Create a LONG signal with proper levels"""
        asset = ASSETS[symbol]
        
        # Entry zone: current price to slightly below
        entry_low = round(price * 0.998, 2)
        entry_high = round(price * 1.002, 2)
        
        # Stop loss: 2-3% below entry, above support
        stop_loss = round(max(price * 0.97, sr['support'] * 0.995), 2)
        
        # Take profits
        risk = price - stop_loss
        take_profit_1 = round(price + (risk * 1.5), 2)  # 1.5:1
        take_profit_2 = round(price + (risk * 2.5), 2)  # 2.5:1
        
        # Calculate R:R
        risk_amount = (asset['contract_size'] * price) * ((price - stop_loss) / price)
        reward_amount = (asset['contract_size'] * price) * ((take_profit_1 - price) / price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return Signal(
            direction="LONG",
            asset=symbol,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_amount=round(risk_amount, 2),
            reward_amount=round(reward_amount, 2),
            rr_ratio=round(rr_ratio, 2),
            confidence="HIGH" if len(reasons) >= 4 else "MEDIUM",
            reasons=reasons,
            valid_until=datetime.now() + timedelta(hours=2)
        )
    
    def _create_short_signal(self, symbol, price, sr, reasons) -> Signal:
        """Create a SHORT signal with proper levels"""
        asset = ASSETS[symbol]
        
        # Entry zone
        entry_low = round(price * 0.998, 2)
        entry_high = round(price * 1.002, 2)
        
        # Stop loss: 2-3% above entry, below resistance
        stop_loss = round(min(price * 1.03, sr['resistance'] * 1.005), 2)
        
        # Take profits (going down)
        risk = stop_loss - price
        take_profit_1 = round(price - (risk * 1.5), 2)
        take_profit_2 = round(price - (risk * 2.5), 2)
        
        # Calculate R:R
        risk_amount = (asset['contract_size'] * price) * ((stop_loss - price) / price)
        reward_amount = (asset['contract_size'] * price) * ((price - take_profit_1) / price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return Signal(
            direction="SHORT",
            asset=symbol,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_amount=round(risk_amount, 2),
            reward_amount=round(reward_amount, 2),
            rr_ratio=round(rr_ratio, 2),
            confidence="HIGH" if len(reasons) >= 4 else "MEDIUM",
            reasons=reasons,
            valid_until=datetime.now() + timedelta(hours=2)
        )
    
    def format_signal_message(self, signal: Signal) -> str:
        """Format signal for Telegram"""
        emoji = "🟢" if signal.direction == "LONG" else "🔴"
        action = "Buy | Long" if signal.direction == "LONG" else "Sell | Short"
        
        asset_info = ASSETS[signal.asset]
        contract = asset_info['cfm_contract']
        
        reasons_text = "\n".join([f"• {r}" for r in signal.reasons])
        
        msg = f"""{emoji} {signal.asset} {signal.direction} SIGNAL

📍 ENTRY ZONE: ${signal.entry_low:,.2f} - ${signal.entry_high:,.2f}
🛑 STOP LOSS: ${signal.stop_loss:,.2f}
🎯 TP1 (close 50%): ${signal.take_profit_1:,.2f}
🎯 TP2 (close rest): ${signal.take_profit_2:,.2f}

⚖️ Risk: ${signal.risk_amount:.2f} | Reward: ${signal.reward_amount:.2f}+ | R:R: {signal.rr_ratio}:1
⏰ Valid: {signal.valid_until.strftime('%I:%M %p')} EST
📊 Confidence: {signal.confidence}

💡 REASONS:
{reasons_text}

━━━━━━━━━━━━━━━━━━━━━

📋 EXECUTION STEPS:
1. Go to CFM → {contract}
2. Click "{action}"
3. Order type: LIMIT
4. Price: ${(signal.entry_low + signal.entry_high) / 2:,.2f}
5. Amount: 1 contract
6. Click the {action.split()[0]} button
7. IMMEDIATELY add TP/SL:
   • Stop Loss: ${signal.stop_loss:,.2f}
   • Take Profit: ${signal.take_profit_1:,.2f}
   • Amount: 1
8. Verify in Positions tab

⚠️ Don't skip the stop loss!"""
        
        return msg
    
    def run_scan(self) -> Optional[str]:
        """Run a market scan and return signal message if found"""
        # Check if we already have open position
        if self.open_position:
            return None  # Don't signal if position open
        
        # Only scan ETH for now (BTC requires more margin)
        signal = self.analyze_for_signal("ETH")
        
        if signal and signal.rr_ratio >= MIN_RR_RATIO:
            self.last_signal_time = datetime.now().isoformat()
            self.save_state()
            return self.format_signal_message(signal)
        
        return None


def main():
    """Run signal scan"""
    engine = CFMSignalEngine()
    
    print(f"🔍 CFM Signal Scan - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("-" * 50)
    
    # Get current market state
    market = engine.get_price_data("ETH")
    if market:
        print(f"ETH: ${market.price:,.2f} ({market.change_24h:+.2f}% 24h)")
        
        # Get indicators
        klines = engine.get_klines("ETH", "4h", 100)
        if klines:
            closes = [k['close'] for k in klines]
            rsi = engine.calculate_rsi(closes)
            trend = engine.detect_trend(klines)
            sr = engine.find_support_resistance(klines)
            
            print(f"Trend: {trend}")
            print(f"RSI 4H: {rsi:.1f}")
            print(f"Support: ${sr['support']:,.2f}")
            print(f"Resistance: ${sr['resistance']:,.2f}")
    
    print("-" * 50)
    
    # Check for signal
    signal_msg = engine.run_scan()
    
    if signal_msg:
        print("\n🚨 SIGNAL DETECTED!")
        print(signal_msg)
        return signal_msg
    else:
        print("No signal - conditions not met")
        return None


if __name__ == "__main__":
    main()

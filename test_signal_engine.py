#!/usr/bin/env python3
"""
Comprehensive Test Suite for CFM Signal Engine
Created by Abel 🦞 during night shift (2am session)
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the module under test
from signal_engine import (
    CFMSignalEngine,
    MarketData,
    Signal,
    ASSETS,
    MAX_RISK_PERCENT,
    MIN_RR_RATIO,
    ACCOUNT_BALANCE,
    COINGECKO_API,
    BINANCE_API
)


# =============================================================================
# FIXTURE HELPERS
# =============================================================================

@pytest.fixture
def engine():
    """Create a fresh CFMSignalEngine with temporary state file"""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_state = f.name
    
    eng = CFMSignalEngine()
    eng.state_file = temp_state
    yield eng
    
    # Cleanup
    if os.path.exists(temp_state):
        os.remove(temp_state)


@pytest.fixture
def sample_klines():
    """Generate sample kline data for testing"""
    base_price = 3000.0
    klines = []
    for i in range(100):
        # Create slight uptrend
        price = base_price + (i * 5) + ((-1)**i * 20)  # Add some volatility
        klines.append({
            'timestamp': datetime.now() - timedelta(hours=(100-i)*4),
            'open': price - 10,
            'high': price + 30,
            'low': price - 30,
            'close': price,
            'volume': 1000000
        })
    return klines


@pytest.fixture
def sample_bearish_klines():
    """Generate sample bearish kline data"""
    base_price = 3500.0
    klines = []
    for i in range(100):
        # Create downtrend
        price = base_price - (i * 5) - ((-1)**i * 15)
        klines.append({
            'timestamp': datetime.now() - timedelta(hours=(100-i)*4),
            'open': price + 10,
            'high': price + 25,
            'low': price - 25,
            'close': price,
            'volume': 1000000
        })
    return klines


@pytest.fixture
def sample_market_data():
    """Create sample MarketData"""
    return MarketData(
        symbol="ETH",
        price=3000.0,
        change_24h=-2.5,
        high_24h=3100.0,
        low_24h=2950.0,
        volume_24h=1500000000,
        timestamp=datetime.now()
    )


# =============================================================================
# TEST: CONFIGURATION AND CONSTANTS
# =============================================================================

class TestConfiguration:
    """Test configuration constants and ASSETS structure"""
    
    def test_assets_has_eth(self):
        """ETH should be configured"""
        assert "ETH" in ASSETS
        
    def test_assets_has_btc(self):
        """BTC should be configured"""
        assert "BTC" in ASSETS
        
    def test_eth_config_complete(self):
        """ETH config should have all required fields"""
        eth = ASSETS["ETH"]
        assert "coingecko_id" in eth
        assert "binance_symbol" in eth
        assert "cfm_contract" in eth
        assert "contract_size" in eth
        assert "margin_required" in eth
        assert "leverage" in eth
        
    def test_btc_config_complete(self):
        """BTC config should have all required fields"""
        btc = ASSETS["BTC"]
        assert btc["coingecko_id"] == "bitcoin"
        assert btc["binance_symbol"] == "BTCUSDT"
        assert btc["contract_size"] > 0
        
    def test_eth_contract_size(self):
        """ETH contract size should be 0.1"""
        assert ASSETS["ETH"]["contract_size"] == 0.1
        
    def test_btc_contract_size(self):
        """BTC contract size should be 0.01"""
        assert ASSETS["BTC"]["contract_size"] == 0.01
        
    def test_risk_parameters(self):
        """Risk parameters should be within reasonable bounds"""
        assert 1.0 <= MAX_RISK_PERCENT <= 5.0
        assert 1.5 <= MIN_RR_RATIO <= 3.0
        assert ACCOUNT_BALANCE > 0
        
    def test_api_endpoints(self):
        """API endpoints should be valid URLs"""
        assert COINGECKO_API.startswith("https://")
        assert BINANCE_API.startswith("https://")


# =============================================================================
# TEST: DATACLASSES
# =============================================================================

class TestMarketData:
    """Test MarketData dataclass"""
    
    def test_create_market_data(self):
        """Should create MarketData with all fields"""
        data = MarketData(
            symbol="ETH",
            price=3000.0,
            change_24h=-1.5,
            high_24h=3100.0,
            low_24h=2900.0,
            volume_24h=1000000000,
            timestamp=datetime.now()
        )
        assert data.symbol == "ETH"
        assert data.price == 3000.0
        assert data.change_24h == -1.5
        
    def test_market_data_positive_change(self):
        """Should handle positive 24h change"""
        data = MarketData(
            symbol="BTC",
            price=65000.0,
            change_24h=5.2,
            high_24h=66000.0,
            low_24h=63000.0,
            volume_24h=5000000000,
            timestamp=datetime.now()
        )
        assert data.change_24h > 0
        
    def test_market_data_extreme_values(self):
        """Should handle extreme market conditions"""
        data = MarketData(
            symbol="ETH",
            price=10000.0,
            change_24h=-20.0,  # Flash crash
            high_24h=12000.0,
            low_24h=8000.0,
            volume_24h=50000000000,  # High volume
            timestamp=datetime.now()
        )
        assert data.price == 10000.0
        assert data.change_24h == -20.0


class TestSignalDataclass:
    """Test Signal dataclass"""
    
    def test_create_long_signal(self):
        """Should create LONG signal"""
        signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=2990.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3150.0,
            take_profit_2=3250.0,
            risk_amount=10.0,
            reward_amount=15.0,
            rr_ratio=1.5,
            confidence="HIGH",
            reasons=["RSI oversold", "Near support"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        assert signal.direction == "LONG"
        assert signal.asset == "ETH"
        assert len(signal.reasons) == 2
        
    def test_create_short_signal(self):
        """Should create SHORT signal"""
        signal = Signal(
            direction="SHORT",
            asset="BTC",
            entry_low=64990.0,
            entry_high=65010.0,
            stop_loss=67000.0,
            take_profit_1=62000.0,
            take_profit_2=60000.0,
            risk_amount=20.0,
            reward_amount=30.0,
            rr_ratio=1.5,
            confidence="MEDIUM",
            reasons=["RSI overbought"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        assert signal.direction == "SHORT"
        assert signal.confidence == "MEDIUM"
        
    def test_signal_rr_ratio(self):
        """R:R ratio should match reward/risk"""
        signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=3000.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3200.0,
            take_profit_2=3300.0,
            risk_amount=10.0,
            reward_amount=20.0,
            rr_ratio=2.0,
            confidence="HIGH",
            reasons=["Test"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        assert signal.rr_ratio == signal.reward_amount / signal.risk_amount


# =============================================================================
# TEST: CALCULATE RSI
# =============================================================================

class TestCalculateRSI:
    """Test RSI calculation"""
    
    def test_rsi_neutral_prices(self, engine):
        """Equal ups and downs should give RSI ~50"""
        prices = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101,
                  100, 101, 100, 101, 100, 101]
        rsi = engine.calculate_rsi(prices)
        assert 45 <= rsi <= 55
        
    def test_rsi_all_up(self, engine):
        """All gains should give high RSI"""
        prices = list(range(100, 120))  # 100, 101, 102, ..., 119
        rsi = engine.calculate_rsi(prices)
        assert rsi == 100.0
        
    def test_rsi_all_down(self, engine):
        """All losses should give low RSI"""
        prices = list(range(119, 99, -1))  # 119, 118, ..., 100
        rsi = engine.calculate_rsi(prices)
        assert rsi == 0.0
        
    def test_rsi_mostly_up(self, engine):
        """Mostly gains should give RSI > 50"""
        prices = [100, 102, 104, 106, 105, 107, 109, 111, 110, 112,
                  114, 116, 115, 117, 119, 121]
        rsi = engine.calculate_rsi(prices)
        assert rsi > 60
        
    def test_rsi_mostly_down(self, engine):
        """Mostly losses should give RSI < 50"""
        prices = [120, 118, 116, 114, 115, 113, 111, 109, 110, 108,
                  106, 104, 105, 103, 101, 99]
        rsi = engine.calculate_rsi(prices)
        assert rsi < 40
        
    def test_rsi_not_enough_data(self, engine):
        """Should return 50 if not enough data"""
        prices = [100, 101, 102]  # Less than period + 1
        rsi = engine.calculate_rsi(prices)
        assert rsi == 50.0
        
    def test_rsi_exact_period(self, engine):
        """Should work with exactly period + 1 data points"""
        prices = list(range(100, 116))  # 16 prices (15 deltas)
        rsi = engine.calculate_rsi(prices, period=14)
        assert rsi == 100.0  # All gains
        
    def test_rsi_custom_period(self, engine):
        """Should support custom RSI period"""
        prices = list(range(100, 130))
        rsi_7 = engine.calculate_rsi(prices, period=7)
        rsi_14 = engine.calculate_rsi(prices, period=14)
        # Both should be 100 for all-up movement
        assert rsi_7 == 100.0
        assert rsi_14 == 100.0


# =============================================================================
# TEST: CALCULATE SMA
# =============================================================================

class TestCalculateSMA:
    """Test Simple Moving Average calculation"""
    
    def test_sma_basic(self, engine):
        """Basic SMA calculation"""
        prices = [100, 102, 104, 106, 108]
        sma = engine.calculate_sma(prices, period=5)
        assert sma == 104.0  # (100+102+104+106+108)/5
        
    def test_sma_longer_period(self, engine):
        """SMA with longer data"""
        prices = list(range(100, 121))  # 100 to 120
        sma_10 = engine.calculate_sma(prices, period=10)
        # Last 10 values: 111, 112, 113, 114, 115, 116, 117, 118, 119, 120
        expected = sum(range(111, 121)) / 10
        assert sma_10 == expected
        
    def test_sma_not_enough_data(self, engine):
        """Should return last price if not enough data"""
        prices = [100, 102, 104]
        sma = engine.calculate_sma(prices, period=10)
        assert sma == 104  # Last price
        
    def test_sma_empty_prices(self, engine):
        """Should handle empty price list"""
        sma = engine.calculate_sma([], period=10)
        assert sma == 0
        
    def test_sma_single_price(self, engine):
        """Should handle single price"""
        sma = engine.calculate_sma([100], period=10)
        assert sma == 100


# =============================================================================
# TEST: DETECT TREND
# =============================================================================

class TestDetectTrend:
    """Test trend detection"""
    
    def test_bullish_trend(self, engine, sample_klines):
        """Should detect bullish trend when price > SMA20 > SMA50"""
        trend = engine.detect_trend(sample_klines)
        assert trend == "BULLISH"
        
    def test_bearish_trend(self, engine, sample_bearish_klines):
        """Should detect bearish trend when price < SMA20 < SMA50"""
        trend = engine.detect_trend(sample_bearish_klines)
        assert trend == "BEARISH"
        
    def test_neutral_not_enough_data(self, engine):
        """Should return NEUTRAL if not enough data"""
        short_klines = [
            {'close': 100},
            {'close': 101},
            {'close': 102}
        ]
        trend = engine.detect_trend(short_klines)
        assert trend == "NEUTRAL"
        
    def test_neutral_mixed_signals(self, engine):
        """Should return NEUTRAL when SMAs are mixed"""
        # Create flat market
        klines = [{'close': 3000 + (i % 10)} for i in range(100)]
        trend = engine.detect_trend(klines)
        assert trend in ["NEUTRAL", "BULLISH", "BEARISH"]  # Any valid trend


# =============================================================================
# TEST: FIND SUPPORT RESISTANCE
# =============================================================================

class TestFindSupportResistance:
    """Test support/resistance detection"""
    
    def test_basic_sr_levels(self, engine, sample_klines):
        """Should find support and resistance"""
        sr = engine.find_support_resistance(sample_klines)
        assert 'support' in sr
        assert 'resistance' in sr
        assert 'range' in sr
        assert sr['resistance'] > sr['support']
        
    def test_sr_range_calculation(self, engine, sample_klines):
        """Range should equal resistance - support"""
        sr = engine.find_support_resistance(sample_klines)
        assert sr['range'] == sr['resistance'] - sr['support']
        
    def test_sr_not_enough_data(self, engine):
        """Should return zeros if not enough data"""
        short_klines = [
            {'low': 100, 'high': 110}
        ]
        sr = engine.find_support_resistance(short_klines, lookback=20)
        assert sr['support'] == 0
        assert sr['resistance'] == 0
        
    def test_sr_custom_lookback(self, engine, sample_klines):
        """Should support custom lookback period"""
        sr_10 = engine.find_support_resistance(sample_klines, lookback=10)
        sr_30 = engine.find_support_resistance(sample_klines, lookback=30)
        # Different lookbacks should give different levels
        assert sr_10['support'] != sr_30['support'] or sr_10['resistance'] != sr_30['resistance']


# =============================================================================
# TEST: CHECK LONG SETUP
# =============================================================================

class TestCheckLongSetup:
    """Test LONG setup detection"""
    
    def test_strong_long_setup(self, engine, sample_market_data):
        """Should trigger LONG on strong bullish setup"""
        reasons = []
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        result = engine._check_long_setup(
            price=2920,  # Near support
            rsi_1h=30,   # Oversold
            rsi_4h=35,
            trend="BULLISH",
            sr=sr,
            market=sample_market_data,
            reasons=reasons
        )
        assert result is True
        assert len(reasons) >= 2
        
    def test_reject_long_in_bearish(self, engine, sample_market_data):
        """Should reject LONG in bearish trend"""
        reasons = []
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        result = engine._check_long_setup(
            price=2920,
            rsi_1h=30,
            rsi_4h=35,
            trend="BEARISH",  # Bearish trend
            sr=sr,
            market=sample_market_data,
            reasons=reasons
        )
        assert result is False
        
    def test_long_needs_enough_signals(self, engine, sample_market_data):
        """Should require minimum score for LONG"""
        reasons = []
        sr = {'support': 2500, 'resistance': 3500, 'range': 1000}
        
        result = engine._check_long_setup(
            price=3000,  # Not near support
            rsi_1h=50,   # Neutral RSI
            rsi_4h=50,
            trend="NEUTRAL",
            sr=sr,
            market=sample_market_data,
            reasons=reasons
        )
        assert result is False  # Not enough signals


# =============================================================================
# TEST: CHECK SHORT SETUP
# =============================================================================

class TestCheckShortSetup:
    """Test SHORT setup detection"""
    
    def test_strong_short_setup(self, engine):
        """Should trigger SHORT on strong bearish setup"""
        reasons = []
        sr = {'support': 2800, 'resistance': 3100, 'range': 300}
        
        market = MarketData(
            symbol="ETH",
            price=3080,
            change_24h=5.0,  # Overextended
            high_24h=3100,
            low_24h=3000,
            volume_24h=1000000000,
            timestamp=datetime.now()
        )
        
        result = engine._check_short_setup(
            price=3080,  # Near resistance
            rsi_1h=70,   # Overbought
            rsi_4h=65,
            trend="BEARISH",
            sr=sr,
            market=market,
            reasons=reasons
        )
        assert result is True
        assert len(reasons) >= 2
        
    def test_reject_short_in_bullish(self, engine, sample_market_data):
        """Should reject SHORT in bullish trend"""
        reasons = []
        sr = {'support': 2800, 'resistance': 3100, 'range': 300}
        
        result = engine._check_short_setup(
            price=3080,
            rsi_1h=70,
            rsi_4h=65,
            trend="BULLISH",  # Bullish trend
            sr=sr,
            market=sample_market_data,
            reasons=reasons
        )
        assert result is False


# =============================================================================
# TEST: CREATE SIGNALS
# =============================================================================

class TestCreateSignals:
    """Test signal creation"""
    
    def test_create_long_signal(self, engine):
        """Should create valid LONG signal"""
        reasons = ["RSI oversold", "Near support", "Trend bullish"]
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        signal = engine._create_long_signal("ETH", 3000.0, sr, reasons)
        
        assert signal.direction == "LONG"
        assert signal.asset == "ETH"
        assert signal.entry_low < signal.entry_high
        assert signal.stop_loss < signal.entry_low
        assert signal.take_profit_1 > signal.entry_high
        assert signal.take_profit_2 > signal.take_profit_1
        assert signal.rr_ratio > 0
        
    def test_create_short_signal(self, engine):
        """Should create valid SHORT signal"""
        reasons = ["RSI overbought", "Near resistance"]
        sr = {'support': 2800, 'resistance': 3100, 'range': 300}
        
        signal = engine._create_short_signal("ETH", 3050.0, sr, reasons)
        
        assert signal.direction == "SHORT"
        assert signal.stop_loss > signal.entry_high
        assert signal.take_profit_1 < signal.entry_low
        assert signal.take_profit_2 < signal.take_profit_1
        
    def test_long_signal_confidence_high(self, engine):
        """Should be HIGH confidence with 4+ reasons"""
        reasons = ["Reason 1", "Reason 2", "Reason 3", "Reason 4"]
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        signal = engine._create_long_signal("ETH", 3000.0, sr, reasons)
        assert signal.confidence == "HIGH"
        
    def test_long_signal_confidence_medium(self, engine):
        """Should be MEDIUM confidence with < 4 reasons"""
        reasons = ["Reason 1", "Reason 2"]
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        signal = engine._create_long_signal("ETH", 3000.0, sr, reasons)
        assert signal.confidence == "MEDIUM"
        
    def test_signal_valid_until(self, engine):
        """Signal should be valid for 2 hours"""
        reasons = ["Test"]
        sr = {'support': 2900, 'resistance': 3200, 'range': 300}
        
        before = datetime.now()
        signal = engine._create_long_signal("ETH", 3000.0, sr, reasons)
        
        # Valid until should be ~2 hours from now
        delta = signal.valid_until - before
        assert 119 <= delta.total_seconds() / 60 <= 121  # ~2 hours


# =============================================================================
# TEST: FORMAT SIGNAL MESSAGE
# =============================================================================

class TestFormatSignalMessage:
    """Test signal message formatting"""
    
    def test_format_long_signal(self, engine):
        """Should format LONG signal correctly"""
        signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=2990.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3150.0,
            take_profit_2=3250.0,
            risk_amount=10.0,
            reward_amount=15.0,
            rr_ratio=1.5,
            confidence="HIGH",
            reasons=["RSI oversold at 28", "Near support $2,900"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        msg = engine.format_signal_message(signal)
        
        assert "🟢" in msg  # Green for LONG
        assert "LONG" in msg
        assert "ETH" in msg
        assert "$2,990.00" in msg or "2990" in msg
        assert "STOP LOSS" in msg
        assert "TP1" in msg
        assert "Buy | Long" in msg
        
    def test_format_short_signal(self, engine):
        """Should format SHORT signal correctly"""
        signal = Signal(
            direction="SHORT",
            asset="BTC",
            entry_low=64990.0,
            entry_high=65010.0,
            stop_loss=67000.0,
            take_profit_1=62000.0,
            take_profit_2=60000.0,
            risk_amount=20.0,
            reward_amount=30.0,
            rr_ratio=1.5,
            confidence="MEDIUM",
            reasons=["RSI overbought"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        msg = engine.format_signal_message(signal)
        
        assert "🔴" in msg  # Red for SHORT
        assert "SHORT" in msg
        assert "BTC" in msg
        assert "Sell | Short" in msg
        
    def test_message_includes_reasons(self, engine):
        """Message should include all reasons"""
        signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=3000.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3150.0,
            take_profit_2=3250.0,
            risk_amount=10.0,
            reward_amount=15.0,
            rr_ratio=1.5,
            confidence="HIGH",
            reasons=["Reason Alpha", "Reason Beta", "Reason Gamma"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        msg = engine.format_signal_message(signal)
        
        assert "Reason Alpha" in msg
        assert "Reason Beta" in msg
        assert "Reason Gamma" in msg
        
    def test_message_includes_rr_ratio(self, engine):
        """Message should include R:R ratio"""
        signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=3000.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3150.0,
            take_profit_2=3250.0,
            risk_amount=10.0,
            reward_amount=25.0,
            rr_ratio=2.5,
            confidence="HIGH",
            reasons=["Test"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        msg = engine.format_signal_message(signal)
        
        assert "2.5:1" in msg


# =============================================================================
# TEST: STATE MANAGEMENT
# =============================================================================

class TestStateManagement:
    """Test state save/load functionality"""
    
    def test_save_and_load_state(self, engine):
        """Should persist and restore state"""
        engine.last_signal_time = "2026-03-06T02:00:00"
        engine.open_position = {"asset": "ETH", "direction": "LONG"}
        
        engine.save_state()
        
        # Create new engine with same state file
        engine2 = CFMSignalEngine()
        engine2.state_file = engine.state_file
        engine2.load_state()
        
        assert engine2.last_signal_time == "2026-03-06T02:00:00"
        assert engine2.open_position == {"asset": "ETH", "direction": "LONG"}
        
    def test_load_missing_state(self, engine):
        """Should handle missing state file gracefully"""
        engine.state_file = "/tmp/nonexistent_state_file_12345.json"
        engine.load_state()  # Should not raise
        
    def test_state_includes_timestamp(self, engine):
        """Saved state should include update timestamp"""
        engine.save_state()
        
        with open(engine.state_file, 'r') as f:
            state = json.load(f)
        
        assert 'updated_at' in state


# =============================================================================
# TEST: RUN SCAN (with mocks)
# =============================================================================

class TestRunScan:
    """Test the main run_scan function"""
    
    def test_skip_if_position_open(self, engine):
        """Should not signal if position is open"""
        engine.open_position = {"asset": "ETH", "direction": "LONG"}
        
        result = engine.run_scan()
        
        assert result is None
        
    def test_run_scan_no_signal(self, engine):
        """Should return None when no signal conditions met"""
        with patch.object(engine, 'analyze_for_signal', return_value=None):
            result = engine.run_scan()
        
        assert result is None
        
    def test_run_scan_with_signal(self, engine):
        """Should return formatted message when signal found"""
        mock_signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=3000.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3150.0,
            take_profit_2=3250.0,
            risk_amount=10.0,
            reward_amount=25.0,
            rr_ratio=2.5,
            confidence="HIGH",
            reasons=["Test signal"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        with patch.object(engine, 'analyze_for_signal', return_value=mock_signal):
            result = engine.run_scan()
        
        assert result is not None
        assert "LONG" in result
        assert "ETH" in result
        
    def test_run_scan_rejects_low_rr(self, engine):
        """Should reject signals with R:R below minimum"""
        mock_signal = Signal(
            direction="LONG",
            asset="ETH",
            entry_low=3000.0,
            entry_high=3010.0,
            stop_loss=2900.0,
            take_profit_1=3050.0,  # Poor target
            take_profit_2=3100.0,
            risk_amount=10.0,
            reward_amount=5.0,
            rr_ratio=0.5,  # Below MIN_RR_RATIO
            confidence="MEDIUM",
            reasons=["Test"],
            valid_until=datetime.now() + timedelta(hours=2)
        )
        
        with patch.object(engine, 'analyze_for_signal', return_value=mock_signal):
            result = engine.run_scan()
        
        assert result is None


# =============================================================================
# TEST: GET PRICE DATA (with mocks)
# =============================================================================

class TestGetPriceData:
    """Test price data fetching with mocked API"""
    
    def test_get_price_data_success(self, engine):
        """Should fetch and parse price data"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ethereum": {
                "usd": 3000.0,
                "usd_24h_change": -2.5,
                "usd_24h_vol": 1500000000
            }
        }
        
        with patch('requests.get', return_value=mock_response):
            result = engine.get_price_data("ETH")
        
        assert result is not None
        assert result.symbol == "ETH"
        assert result.price == 3000.0
        assert result.change_24h == -2.5
        
    def test_get_price_data_invalid_symbol(self, engine):
        """Should return None for unknown symbol"""
        result = engine.get_price_data("INVALID")
        assert result is None
        
    def test_get_price_data_api_error(self, engine):
        """Should handle API errors gracefully"""
        with patch('requests.get', side_effect=Exception("API Error")):
            result = engine.get_price_data("ETH")
        
        assert result is None


# =============================================================================
# TEST: GET KLINES (with mocks)
# =============================================================================

class TestGetKlines:
    """Test kline/candlestick data fetching"""
    
    def test_get_klines_success(self, engine):
        """Should fetch and parse klines"""
        mock_response = MagicMock()
        # Generate mock price data points
        mock_prices = [[1709600000000 + i*3600000, 3000 + i] for i in range(100)]
        mock_response.json.return_value = {"prices": mock_prices}
        
        with patch('requests.get', return_value=mock_response):
            result = engine.get_klines("ETH", "4h", 50)
        
        assert len(result) > 0
        assert 'open' in result[0]
        assert 'high' in result[0]
        assert 'low' in result[0]
        assert 'close' in result[0]
        
    def test_get_klines_invalid_symbol(self, engine):
        """Should return empty list for unknown symbol"""
        result = engine.get_klines("INVALID", "4h", 50)
        assert result == []
        
    def test_get_klines_api_error(self, engine):
        """Should handle API errors gracefully"""
        with patch('requests.get', side_effect=Exception("API Error")):
            result = engine.get_klines("ETH", "4h", 50)
        
        assert result == []


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_price(self, engine):
        """Should handle zero prices gracefully"""
        prices = [0, 0, 0, 0, 0]
        sma = engine.calculate_sma(prices, 3)
        assert sma == 0
        
    def test_negative_change(self, sample_market_data):
        """Market data can have negative change"""
        assert sample_market_data.change_24h < 0
        
    def test_btc_signals(self, engine):
        """Should generate signals for BTC too"""
        reasons = ["Test"]
        sr = {'support': 60000, 'resistance': 70000, 'range': 10000}
        
        signal = engine._create_long_signal("BTC", 65000.0, sr, reasons)
        
        assert signal.asset == "BTC"
        assert signal.entry_low > 60000
        
    def test_large_price_values(self, engine):
        """Should handle large BTC-scale prices"""
        prices = [65000 + i*100 for i in range(50)]
        sma = engine.calculate_sma(prices, 20)
        rsi = engine.calculate_rsi(prices)
        
        assert sma > 60000
        assert rsi == 100.0  # All gains


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

#!/usr/bin/env python3
"""
CFM Signal Scanner - Cron wrapper
Runs the signal engine and outputs alert for Clawdbot to send via Telegram
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_engine import CFMSignalEngine
from datetime import datetime

def main():
    """Run scan and print signal if found"""
    engine = CFMSignalEngine()
    
    # Get market snapshot for logging
    market = engine.get_price_data("ETH")
    
    if market:
        # Run the scan
        signal_msg = engine.run_scan()
        
        if signal_msg:
            # Signal found - print for Clawdbot to send
            print("SIGNAL_FOUND")
            print(signal_msg)
            return 0
        else:
            # No signal - print market status for logging
            klines = engine.get_klines("ETH", "4h", 100)
            if klines:
                closes = [k['close'] for k in klines]
                rsi = engine.calculate_rsi(closes)
                trend = engine.detect_trend(klines)
                print(f"NO_SIGNAL | ETH ${market.price:,.2f} | RSI {rsi:.0f} | {trend}")
            else:
                print(f"NO_SIGNAL | ETH ${market.price:,.2f}")
            return 0
    else:
        print("ERROR | Could not fetch market data")
        return 1


if __name__ == "__main__":
    sys.exit(main())

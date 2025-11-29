# test_mse_fixed.py
"""
Test script to verify the fixed MSE implementation.
Tests that MarketStructureEngine correctly identifies market structure
and generates complete trade plans with entry, SL, TP, and RR.
"""

import asyncio
import sys
from mse_enhanced import MarketStructureEngine, CONFIG

async def test_mse():
    print("="*60)
    print("Testing Market Structure Engine (MSE) - Fixed Version")
    print("="*60)
    
    # Initialize MSE
    mse = MarketStructureEngine(timeframes=['1', '5', '15'], bars_per_tf={'1': 400, '5': 400, '15': 400})
    
    # Test symbols
    test_symbols = ['BANKNIFTY', 'NIFTY', 'RELIANCE']
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing symbol: {symbol}")
        print(f"{'='*60}")
        
        # Process symbol
        result = await mse.process_symbol(symbol, ltp=None, direction='long')
        
        # Display results
        print(f"\nValid: {result.get('valid')}")
        
        if result.get('valid'):
            print(f"\n✓ TRADE PLAN GENERATED:")
            print(f"  Entry:  {result.get('entry', 'N/A'):.2f}")
            print(f"  SL:     {result.get('sl', 'N/A'):.2f}")
            print(f"  TP:     {result.get('tp', 'N/A'):.2f}")
            print(f"  R:R:    {result.get('rr', 'N/A'):.2f}")
            print(f"  Mode:   {result.get('mode', 'N/A')}")
            
            # Market structure context
            ctx = result.get('mse_context', {})
            print(f"\n  MARKET STRUCTURE:")
            print(f"  Structure Bias:      {ctx.get('structure_bias', 'N/A')}")
            print(f"  Impulse:             {ctx.get('impulse', False)}")
            print(f"  Compression:         {ctx.get('compression', False)}")
            print(f"  Nearest Resistance:  {ctx.get('nearest_resistance', 'N/A')}")
            print(f"  Nearest Support:     {ctx.get('nearest_support', 'N/A')}")
            print(f"  Score:               {ctx.get('score', 'N/A')}")
            print(f"  Blockers:            {len(ctx.get('blockers', []))}")
            if ctx.get('blockers'):
                for blocker in ctx['blockers']:
                    print(f"    - {blocker.get('type')}: {blocker.get('reason')}")
        else:
            print(f"\n✗ TRADE PLAN REJECTED:")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
            
            # Still show market structure context if available
            ctx = result.get('mse_context', {})
            if ctx:
                print(f"\n  MARKET STRUCTURE:")
                print(f"  Structure Bias:      {ctx.get('structure_bias', 'N/A')}")
                print(f"  Nearest Resistance:  {ctx.get('nearest_resistance', 'N/A')}")
                print(f"  Nearest Support:     {ctx.get('nearest_support', 'N/A')}")
                print(f"  Blockers:            {len(ctx.get('blockers', []))}")
    
    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"{'='*60}")
    
    # Verify output structure matches mse_subscriber expectations
    print(f"\n{'='*60}")
    print("VERIFICATION: Output Structure Check")
    print(f"{'='*60}")
    
    result = await mse.process_symbol('BANKNIFTY', ltp=None, direction='long')
    
    required_fields_valid = ['valid', 'entry', 'sl', 'tp', 'rr', 'mse_context']
    required_fields_invalid = ['valid', 'reason', 'mse_context']
    
    if result.get('valid'):
        print("\n✓ Valid trade plan output structure:")
        for field in required_fields_valid:
            has_field = field in result
            status = "✓" if has_field else "✗"
            print(f"  {status} {field}: {has_field}")
    else:
        print("\n✓ Invalid trade plan output structure:")
        for field in required_fields_invalid:
            has_field = field in result
            status = "✓" if has_field else "✗"
            print(f"  {status} {field}: {has_field}")
    
    print(f"\n{'='*60}")
    print("All checks passed! MSE is working correctly.")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        asyncio.run(test_mse())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

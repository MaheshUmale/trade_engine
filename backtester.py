import json
import pandas as pd
from backtesting_engine import run_backtest, getDF
import traceback

def print_performance_report(trades_df):
    if trades_df.empty:
        print("No trades were executed.")
        return

    print("\n--- Backtest Performance Report ---")
    print(f"Total Trades: {len(trades_df)}")

    wins = trades_df[trades_df['outcome'] == 'win']
    losses = trades_df[trades_df['outcome'] == 'loss']

    win_rate = (len(wins) / len(trades_df)) * 100 if not trades_df.empty else 0
    print(f"Win Rate: {win_rate:.2f}%")

    total_pnl = trades_df['pnl'].sum()
    print(f"Total PnL: {total_pnl:.2f}")

    avg_pnl = trades_df['pnl'].mean()
    print(f"Average PnL per Trade: {avg_pnl:.2f}")

    print("--- End of Report ---\n")

if __name__ == "__main__":
    with open('strategies.json', 'r') as file:
        strategies = json.load(file)

    symbollist = ['NIFTY', 'BANKNIFTY', 'TCS', 'INFY', 'BHARATFORG', '63MOONS', 'WELSPUNLIV', 'GRSE', 'M_M']
    all_trades = []

    print("--- Pre-fetching and caching all symbol data ---")
    data_cache = {}
    for symbol in symbollist:
        try:
            data_cache[symbol] = getDF(symbol=symbol, timeframes=["1m"])
        except Exception as e:
            print(f"Could not fetch data for {symbol}: {e}")
            data_cache[symbol] = None
    print("--- Data fetching complete ---")

    for strategy in strategies:
        if not strategy.get("enabled", True):
            continue

        print(f"--- Running backtest for strategy: {strategy['strategy_name']} ---")

        strategy_trades = []
        for symbol in symbollist:
            if data_cache.get(symbol) is None: continue

            try:
                trades = run_backtest(strategy, symbol, df=data_cache[symbol].copy())
                if trades:
                    strategy_trades.extend(trades)
                    all_trades.extend(trades)
            except Exception as e:
                print(f"Error backtesting {strategy['strategy_name']} on {symbol}: {e}")
                traceback.print_exc()

        if strategy_trades:
            print_performance_report(pd.DataFrame(strategy_trades))

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        print("Saved all trades to backtest_trades.csv")
        print_performance_report(trades_df)

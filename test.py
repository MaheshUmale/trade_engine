from mse import MSEAnalyst, detect_trend_vectorized
import asyncio
import pandas as pd
from backtesting_engine import getDF, extract_order_blocks, extract_fvgs, resample_df

async def test():
    engine = MSEAnalyst()
    df = getDF("BANKNIFTY", ["1m"])

    df_5m = resample_df(df, '5T')
    df_15m = resample_df(df, '15T')
    trends = {
        'ltf': detect_trend_vectorized(df).reindex(df.index, method='ffill'),
        'mtf': detect_trend_vectorized(df_5m).reindex(df.index, method='ffill'),
        'htf': detect_trend_vectorized(df_15m).reindex(df.index, method='ffill')
    }

    zones = extract_order_blocks(df)
    fvgs = extract_fvgs(df)

    out = engine.build_context("BANKNIFTY", 50, 50000, trends, 18.0, zones, fvgs, "2024-01-01 12:00:00")
    print(out)

asyncio.run(test())

from mse import MarketStructureEngine
import asyncio

async def test():
    engine = MarketStructureEngine()
    out = await engine.process_symbol("BANKNIFTY", 50000)
    print(out)

asyncio.run(test())


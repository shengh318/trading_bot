import asyncio, json, websockets

async def test():
    async with websockets.connect("ws://127.0.0.1:8000/ws/backtest") as ws:
        await ws.send(json.dumps({
            "action": "run",
            "strategy_name": "SmaCrossover",
            "symbols": ["NVDA"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-05",
            "initial_cash": 10000,
            "parameters": {"short_window": 2, "long_window": 4},
            "timeframe": "15Min"
        }))
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=20)
            data = json.loads(msg)
            if data["type"] == "error":
                print(f"ERROR: {data['message']}")
                break
            elif data["type"] == "complete":
                print(f"COMPLETE: {json.dumps(data['metrics'], indent=2)}")
                break
            elif data["type"] == "bar":
                print(f"BAR: idx={data['bar_index']} equity={data['equity']:.2f} trades={len(data['trades'])}")

asyncio.run(test())

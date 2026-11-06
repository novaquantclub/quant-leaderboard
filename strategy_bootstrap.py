"""Guest entry point. Only this file and the submitted strategy enter the jail."""

import importlib.util
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

MAX_REPLY = 262_144
MAX_STATE = 16_000_000


def main():
    # This public strategy seed is independent of the private market seed.
    random.seed(0)
    np.random.seed(0)
    # Strategy prints cannot corrupt the protocol or fill a log file. This is
    # convenience, not a security boundary: the supervisor validates all replies.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - process-lifetime stdout, closed by the OS on exit

    def reply(value):
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode()) > MAX_REPLY:
            raise ValueError("Strategy response exceeds the size limit.")
        protocol.write(encoded + "\n")
        protocol.flush()

    try:
        spec = importlib.util.spec_from_file_location("user_strategy", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trader = module.Trader()
        reply(
            {
                "name": trader.name,
                "author": trader.author,
                "tickers": list(trader.tickers),
            }
        )
        while True:
            line = sys.stdin.buffer.readline(MAX_STATE + 1)
            if not line:
                return
            if len(line) > MAX_STATE:
                raise ValueError("State exceeds the size limit.")
            state = json.loads(line)
            state["history"] = {
                symbol: pd.DataFrame(
                    values,
                    columns=["Open", "High", "Low", "Close", "Volume"],
                    index=pd.RangeIndex(state["history_start"], state["step"] + 1),
                )
                for symbol, values in state["history"].items()
            }
            state.pop("history_start")
            result = trader.run(SimpleNamespace(**state))
            orders, trader_data = (
                result
                if isinstance(result, tuple) and len(result) == 2
                else (result, state["trader_data"])
            )
            if not isinstance(orders, dict) or len(orders) > 20:
                raise ValueError("Return an order dictionary with at most 20 symbols.")
            if not isinstance(trader_data, str) or len(trader_data.encode()) > 100_000:
                raise ValueError("trader_data must be a string of at most 100 KB.")
            clean = {}
            for symbol, quantity in orders.items():
                if not isinstance(symbol, str) or len(symbol) > 32:
                    raise ValueError("Invalid order symbol.")
                if isinstance(quantity, bool):
                    raise TypeError("Order quantities must be numeric, not boolean.")
                quantity = float(quantity)
                if not math.isfinite(quantity):
                    raise ValueError("Order quantities must be finite.")
                clean[symbol] = quantity
            reply({"orders": clean, "trader_data": trader_data})
    except BaseException as exc:  # noqa: BLE001 - user code can raise SystemExit or KeyboardInterrupt
        try:
            reply({"error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        except BaseException:  # noqa: BLE001, S110 - a broken guest pipe must not create a second error channel
            pass
        raise SystemExit(1)


if __name__ == "__main__":
    main()

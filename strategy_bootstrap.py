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


class HistoryBuffer:
    """Retain received observations and give each strategy call independent copies."""

    def __init__(self):
        self.start = None
        self.length = 0
        self.capacity = 0
        self.arrays = {"history": {}, "observations": {}}

    def restore(self, state):
        first = state["history_start"]
        update_start = state.pop("history_update_start", None)
        incremental = update_start is not None
        incoming = {
            "history": state["history"],
            "observations": state.get("observations", {}),
        }
        if incremental:
            if (
                first != self.start
                or update_start != self.start + self.length
                or any(
                    set(incoming[group]) != set(self.arrays[group])
                    for group in incoming
                )
            ):
                raise ValueError(
                    "History update must extend the previous observation prefix."
                )
        else:
            self.start, self.length = first, 0
            self.arrays = {"history": {}, "observations": {}}
            update_start = first
        count = state["step"] - update_start + 1
        if count <= 0:
            raise ValueError("History update needs at least one current observation.")
        total = self.length + count
        capacity = max(total, self.capacity, 128)
        if total > self.capacity:
            capacity = max(capacity, self.capacity * 2)
        for group, values in incoming.items():
            for name, rows in values.items():
                shape = (count, 5) if group == "history" else (count,)
                array = np.asarray(rows, dtype=float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(
                        "History update has invalid observation dimensions or values."
                    )
                previous = self.arrays[group].get(name)
                if previous is None or len(previous) < total:
                    buffer = np.empty((capacity, *shape[1:]), dtype=float)
                    if previous is not None:
                        buffer[: self.length] = previous[: self.length]
                    self.arrays[group][name] = buffer
                self.arrays[group][name][self.length : total] = array
        self.capacity, self.length = capacity, total
        index = pd.RangeIndex(first, state["step"] + 1)
        state["history"] = {
            name: pd.DataFrame(
                self.arrays["history"][name][:total].copy(),
                columns=["Open", "High", "Low", "Close", "Volume"],
                index=index,
            )
            for name in incoming["history"]
        }
        state["observations"] = {
            name: pd.Series(
                self.arrays["observations"][name][:total].copy(), index=index, name=name
            )
            for name in incoming["observations"]
        }
        state.pop("history_start")
        return state


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
        history = HistoryBuffer()
        while True:
            line = sys.stdin.buffer.readline(MAX_STATE + 1)
            if not line:
                return
            if len(line) > MAX_STATE:
                raise ValueError("State exceeds the size limit.")
            state = history.restore(json.loads(line))
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

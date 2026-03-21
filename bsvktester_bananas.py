from __future__ import annotations

import argparse
import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PRODUCT = "BANANAS"
POS_LIMIT = 20
DAYS = (-1, 0, 1)
TRACKED = ["Student", "MM Alpha", "MM Beta", "Momentum Hawk", "Reversion Owl"]
RULES = {
    "MM Alpha": {"limit": 12, "role": "Maker / defensive"},
    "MM Beta": {"limit": 20, "role": "Maker / pressure"},
    "Momentum Hawk": {"limit": 16, "role": "Taker / momentum"},
    "Reversion Owl": {"limit": 16, "role": "Taker / mean rev"},
}
PRIORITY = {"student": 2, "bot": 1, "market": 0}


@dataclass
class AgentState:
    cash: float = 0.0
    position: int = 0
    fills: int = 0
    traded_qty: int = 0


@dataclass
class Order:
    type: str
    price: int
    qty: int
    from_name: str


class Rng:
    def __init__(self) -> None:
        self.seed = 42

    def reset(self) -> None:
        self.seed = 42

    def random(self) -> float:
        self.seed = (self.seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.seed / 4294967295


class Backtester:
    def __init__(self, strategy: Callable, price_rows: list[dict]) -> None:
        self.strategy = strategy
        self.price_rows = price_rows
        self.agent_state = {name: AgentState() for name in TRACKED}
        self.mid_history: list[float] = []
        self.fair_history: list[float] = []
        self.last_mid: float | None = None
        self.trade_log: list[dict] = []
        self.rng = Rng()
        self.last_trade_price: float | None = None
        self.last_trade_qty: int | None = None
        self.last_trade_timestamp: int | None = None

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def round2(value: float) -> float:
        return round(value + 1e-12, 2)

    @staticmethod
    def avg(values: list[float], n: int | None = None) -> float:
        if not values:
            return 0.0
        subset = values[-n:] if n else values
        return sum(subset) / len(subset)

    def diff_std(self, values: list[float], n: int = 30) -> float:
        subset = values[-(n + 1):]
        if len(subset) < 3:
            return 0.0
        diffs = [subset[i] - subset[i - 1] for i in range(1, len(subset))]
        mean = self.avg(diffs)
        return (sum((value - mean) ** 2 for value in diffs) / len(diffs)) ** 0.5

    @staticmethod
    def derive_mid(row: dict) -> float:
        bid = row.get("bid_price_1")
        ask = row.get("ask_price_1")
        if row.get("mid_price") is not None:
            return float(row["mid_price"])
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return 0.0

    def pnl_of(self, name: str, fair: float) -> float:
        state = self.agent_state[name]
        return state.cash + state.position * fair

    def clip_order(self, name: str, side: str, price: float | int | None, qty: float | int | None) -> Order | None:
        if price is None or qty is None:
            return None
        try:
            price_val = float(price)
            qty_val = float(qty)
        except (TypeError, ValueError):
            return None
        if not (price_val == price_val and qty_val == qty_val):
            return None
        limit = POS_LIMIT if name == "Student" else RULES[name]["limit"]
        position = self.agent_state[name].position
        allowance = max(0, limit - position) if side == "bid" else max(0, limit + position)
        size = min(max(0, int(abs(qty_val))), allowance)
        return Order(side, int(round(price_val)), size, name) if size > 0 else None

    def signals(self, row: dict) -> dict:
        mid = self.derive_mid(row)
        b1 = row.get("bid_price_1") or mid - 3
        a1 = row.get("ask_price_1") or mid + 3
        b2 = row.get("bid_price_2") or b1 - 1
        a2 = row.get("ask_price_2") or a1 + 1
        b3 = row.get("bid_price_3") or b2 - 1
        a3 = row.get("ask_price_3") or a2 + 1
        bv1 = row.get("bid_volume_1") or 1
        av1 = row.get("ask_volume_1") or 1
        bv2 = row.get("bid_volume_2") or 0
        av2 = row.get("ask_volume_2") or 0
        bv3 = row.get("bid_volume_3") or 0
        av3 = row.get("ask_volume_3") or 0
        spread = max(1, a1 - b1)
        micro = (a1 * bv1 + b1 * av1) / max(1, bv1 + av1)
        bid_depth = bv1 + bv2 + bv3
        ask_depth = av1 + av2 + av3
        imb1 = (bv1 - av1) / max(1, bv1 + av1)
        imb3 = (bid_depth - ask_depth) / max(1, bid_depth + ask_depth)
        anchor = self.avg(self.mid_history, 40) if self.mid_history else mid
        mom = 0.0 if self.last_mid is None else mid - self.last_mid
        mean_rev = anchor - mid
        vol = self.diff_std(self.mid_history, 28)
        fair_ref = self.round2(mid + 0.85 * (micro - mid) + 1.35 * imb1 + 0.6 * imb3 + 0.18 * mean_rev - 0.15 * mom)
        prev_fair = self.fair_history[-1] if self.fair_history else fair_ref
        fair_trend = fair_ref - prev_fair
        fair_edge = fair_ref - mid
        return {
            "b1": b1, "a1": a1, "b2": b2, "a2": a2, "b3": b3, "a3": a3,
            "bv1": bv1, "av1": av1, "bv2": bv2, "av2": av2, "bv3": bv3, "av3": av3,
            "mid": mid, "spread": spread, "micro": micro, "imb1": imb1, "imb3": imb3,
            "anchor": anchor, "mom": mom, "mean_rev": mean_rev, "vol": vol,
            "fair_ref": fair_ref, "fair_trend": fair_trend, "fair_edge": fair_edge,
        }

    def remember(self, sig: dict) -> None:
        self.mid_history.append(sig["mid"])
        self.fair_history.append(sig["fair_ref"])
        self.last_mid = sig["mid"]

    def mm_alpha(self, sig: dict) -> list[Order]:
        st = self.agent_state["MM Alpha"]
        lean = 0.45 * (sig["micro"] - sig["mid"]) + 0.75 * sig["imb1"] + 0.2 * sig["fair_trend"]
        fair = sig["fair_ref"] + lean - st.position * 0.34
        half = self.clamp(1.3 + 0.18 * sig["spread"] + 0.55 * sig["vol"], 1.2, 3.6)
        size = int(self.clamp(5 - abs(st.position) // 3, 2, 6))
        return [
            order for order in [
                self.clip_order("MM Alpha", "bid", min(sig["a1"] - 1, int(fair - half)), size),
                self.clip_order("MM Alpha", "ask", max(sig["b1"] + 1, int(round(fair + half))), size),
            ] if order
        ]

    def mm_beta(self, sig: dict) -> list[Order]:
        st = self.agent_state["MM Beta"]
        pressure = 1.55 * sig["fair_edge"] + 1.1 * (sig["micro"] - sig["mid"]) + 1.8 * sig["imb1"] + 0.75 * sig["mom"] + 0.7 * sig["fair_trend"]
        fair = sig["fair_ref"] + pressure - st.position * 0.16
        half = self.clamp(1.8 + 0.22 * sig["spread"] + 0.9 * sig["vol"] + abs(sig["imb1"]) * 2.1, 1.8, 4.9)
        size = int(self.clamp(5 + int(abs(pressure) * 0.65), 4, 9))
        orders = [
            self.clip_order("MM Beta", "bid", min(sig["a1"] - 1, int(fair - half)), size),
            self.clip_order("MM Beta", "ask", max(sig["b1"] + 1, int(round(fair + half))), size),
        ]
        if sig["fair_ref"] >= sig["a1"] + 1 or pressure > 2:
            orders.append(self.clip_order("MM Beta", "bid", sig["a1"], 2 + int(self.rng.random() * 3)))
        elif sig["fair_ref"] <= sig["b1"] - 1 or pressure < -2:
            orders.append(self.clip_order("MM Beta", "ask", sig["b1"], 2 + int(self.rng.random() * 3)))
        return [order for order in orders if order]

    def momentum_hawk(self, sig: dict) -> list[Order]:
        impulse = sig["mom"] + sig["fair_trend"] + 1.6 * sig["imb1"]
        size = int(self.clamp(2 + int(abs(impulse) * 1.4 + abs(sig["fair_edge"]) * 0.6), 2, 8))
        if sig["fair_ref"] > sig["a1"] or impulse > 1.5:
            return [order for order in [self.clip_order("Momentum Hawk", "bid", sig["a1"] + (1 if sig["fair_ref"] > sig["a1"] + 2 else 0), size)] if order]
        if sig["fair_ref"] < sig["b1"] or impulse < -1.5:
            return [order for order in [self.clip_order("Momentum Hawk", "ask", sig["b1"] - (1 if sig["fair_ref"] < sig["b1"] - 2 else 0), size)] if order]
        return [
            order for order in [
                self.clip_order("Momentum Hawk", "bid", min(sig["a1"] - 1, sig["b1"] + (1 if sig["imb1"] > 0 else 0)), 1),
                self.clip_order("Momentum Hawk", "ask", max(sig["b1"] + 1, sig["a1"] - (1 if sig["imb1"] < 0 else 0)), 1),
            ] if order
        ]

    def reversion_owl(self, sig: dict) -> list[Order]:
        dist_fair = sig["mid"] - sig["fair_ref"]
        dist_anchor = sig["mid"] - sig["anchor"]
        size = int(self.clamp(2 + int(abs(dist_fair) * 0.45 + abs(dist_anchor) * 0.15), 2, 8))
        if dist_fair > 1.4 or dist_anchor > 2.2:
            return [order for order in [self.clip_order("Reversion Owl", "ask", sig["b1"] if dist_fair > 3.2 else max(sig["b1"] + 1, int(round(sig["fair_ref"] + 0.4))), size)] if order]
        if dist_fair < -1.4 or dist_anchor < -2.2:
            return [order for order in [self.clip_order("Reversion Owl", "bid", sig["a1"] if dist_fair < -3.2 else min(sig["a1"] - 1, int(sig["fair_ref"] - 0.4)), size)] if order]
        return [
            order for order in [
                self.clip_order("Reversion Owl", "bid", min(sig["a1"] - 1, int(sig["fair_ref"] - 1)), 1),
                self.clip_order("Reversion Owl", "ask", max(sig["b1"] + 1, int(round(sig["fair_ref"] + 1))), 1),
            ] if order
        ]

    def bot_orders(self, sig: dict) -> list[Order]:
        return self.mm_alpha(sig) + self.mm_beta(sig) + self.momentum_hawk(sig) + self.reversion_owl(sig)

    def build_book(self, row: dict, bots: list[Order], student: list[Order]) -> tuple[list[dict], list[dict]]:
        bids: list[dict] = []
        asks: list[dict] = []
        for level in (1, 2, 3):
            bid_price = row.get(f"bid_price_{level}")
            bid_volume = row.get(f"bid_volume_{level}")
            ask_price = row.get(f"ask_price_{level}")
            ask_volume = row.get(f"ask_volume_{level}")
            if bid_price is not None and bid_volume and bid_volume > 0:
                bids.append({"price": int(bid_price), "qty": int(bid_volume), "from_name": "Market", "priority": 0})
            if ask_price is not None and ask_volume and ask_volume > 0:
                asks.append({"price": int(ask_price), "qty": int(ask_volume), "from_name": "Market", "priority": 0})
        for order in bots + student:
            target = bids if order.type == "bid" else asks
            target.append({"price": order.price, "qty": order.qty, "from_name": order.from_name, "priority": PRIORITY["student"] if order.from_name == "Student" else PRIORITY["bot"]})
        bids.sort(key=lambda item: (-item["price"], -item["priority"]))
        asks.sort(key=lambda item: (item["price"], -item["priority"]))
        return bids, asks

    def match_book(self, bids: list[dict], asks: list[dict]) -> list[dict]:
        trades: list[dict] = []
        while bids and asks and bids[0]["price"] >= asks[0]["price"]:
            bid = bids[0]
            ask = asks[0]
            qty = min(bid["qty"], ask["qty"])
            trades.append({"bid_from": bid["from_name"], "ask_from": ask["from_name"], "price": ask["price"], "qty": qty})
            bid["qty"] -= qty
            ask["qty"] -= qty
            if bid["qty"] <= 0:
                bids.pop(0)
            if ask["qty"] <= 0:
                asks.pop(0)
        return trades

    def apply_fill(self, name: str, side: str, price: int, qty: int) -> None:
        state = self.agent_state.get(name)
        if state is None:
            return
        if side == "buy":
            state.cash -= qty * price
            state.position += qty
        else:
            state.cash += qty * price
            state.position -= qty
        state.fills += 1
        state.traded_qty += qty

    def run(self, day_label: str) -> dict:
        for step, row in enumerate(self.price_rows):
            enriched_row = dict(row)
            enriched_row["hidden_fair"] = None
            enriched_row["last_trade_price"] = self.last_trade_price
            enriched_row["last_trade_qty"] = self.last_trade_qty
            enriched_row["last_trade_timestamp"] = self.last_trade_timestamp
            sig = self.signals(enriched_row)
            fair = sig["fair_ref"]
            enriched_row["hidden_fair"] = fair
            order = self.strategy(enriched_row, self.agent_state["Student"].position, step) or {}
            student_orders = [
                order for order in [
                    self.clip_order("Student", "bid", order.get("bid"), order.get("bid_qty", 0)),
                    self.clip_order("Student", "ask", order.get("ask"), order.get("ask_qty", 0)),
                ] if order
            ]
            bids, asks = self.build_book(enriched_row, self.bot_orders(sig), student_orders)
            for trade in self.match_book(bids, asks):
                self.apply_fill(trade["bid_from"], "buy", trade["price"], trade["qty"])
                self.apply_fill(trade["ask_from"], "sell", trade["price"], trade["qty"])
                self.last_trade_price = float(trade["price"])
                self.last_trade_qty = int(trade["qty"])
                self.last_trade_timestamp = int(enriched_row.get("timestamp")) if enriched_row.get("timestamp") is not None else None
                student = self.agent_state["Student"]
                if trade["bid_from"] == "Student":
                    self.trade_log.append({"day": day_label, "timestamp": enriched_row.get("timestamp"), "side": "BUY", "price": trade["price"], "qty": trade["qty"], "counterparty": trade["ask_from"], "position": student.position, "pnl": self.pnl_of("Student", fair)})
                elif trade["ask_from"] == "Student":
                    self.trade_log.append({"day": day_label, "timestamp": enriched_row.get("timestamp"), "side": "SELL", "price": trade["price"], "qty": trade["qty"], "counterparty": trade["bid_from"], "position": student.position, "pnl": self.pnl_of("Student", fair)})
            self.remember(sig)
        final_fair = self.fair_history[-1] if self.fair_history else (self.derive_mid(self.price_rows[-1]) if self.price_rows else 0.0)
        student = self.agent_state["Student"]
        return {"pnl": self.pnl_of("Student", final_fair), "cash": student.cash, "position": student.position, "fair": final_fair, "fills": student.fills, "qty": student.traded_qty, "trades": len(self.trade_log)}


def coerce(value: str) -> float | str | None:
    if value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def load_csv(path: Path, product_field: str) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = []
        for row in reader:
            cooked = {key.strip().lower(): coerce(value.strip()) for key, value in row.items()}
            if str(cooked.get(product_field, "")).upper() != PRODUCT:
                continue
            rows.append(cooked)
        return rows


def load_strategy(path: Path) -> Callable:
    spec = importlib.util.spec_from_file_location("candidate_strategy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy = getattr(module, "strategy", None)
    if strategy is None or not callable(strategy):
        raise RuntimeError(f"{path} does not define callable strategy(row, position, step)")
    return strategy


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "MichalOkon imc_prosperity main datasets" / "island-data-bottle-round-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BANANAS BSVK tester for round-2 day -1, 0, 1")
    parser.add_argument("strategy", type=Path, help="Path to candidate strategy .py file")
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Directory containing round-2 price CSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = load_strategy(args.strategy)
    total_pnl = 0.0
    print("BANANAS BSVK backtest")
    print(f"strategy: {args.strategy}")
    print(f"data root: {args.data_root}")
    print("")
    print(f"{'Day':>6}  {'PnL':>10}  {'Cash':>10}  {'Pos':>5}  {'Fair':>8}  {'Fills':>6}  {'Qty':>6}")
    print("-" * 63)
    for day in DAYS:
        price_path = args.data_root / f"prices_round_2_day_{day}.csv"
        price_rows = load_csv(price_path, "product")
        tester = Backtester(strategy, price_rows)
        summary = tester.run(f"Day {day}")
        total_pnl += summary["pnl"]
        print(f"{day:>6}  {summary['pnl']:>10.2f}  {summary['cash']:>10.2f}  {summary['position']:>5}  {summary['fair']:>8.2f}  {summary['fills']:>6}  {summary['qty']:>6}")
    print("-" * 63)
    print(f"{'TOTAL':>6}  {total_pnl:>10.2f}")


if __name__ == "__main__":
    main()

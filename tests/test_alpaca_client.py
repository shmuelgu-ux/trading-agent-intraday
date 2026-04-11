from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from services.alpaca_client import AlpacaClient


def _make_order(filled_at, filled_avg_price, filled_qty):
    """Build a fake Alpaca Order object with the fields we actually read."""
    o = MagicMock()
    o.filled_at = filled_at
    o.filled_avg_price = filled_avg_price
    o.filled_qty = filled_qty
    return o


def _client_with_orders(orders):
    """Return an AlpacaClient whose underlying TradingClient.get_orders
    returns ``orders``. Bypasses __init__ so we don't need real API keys.
    """
    c = AlpacaClient.__new__(AlpacaClient)
    c._client = MagicMock()
    c._data_client = None
    c._client.get_orders.return_value = orders
    return c


def test_returns_fill_from_first_valid_order():
    now = datetime.now(timezone.utc)
    orders = [
        _make_order(filled_at=now - timedelta(minutes=1), filled_avg_price="101.5", filled_qty="50"),
        _make_order(filled_at=now - timedelta(minutes=5), filled_avg_price="100.0", filled_qty="50"),
    ]
    client = _client_with_orders(orders)

    fill = client.get_last_closing_fill("AAPL", "sell", after=now - timedelta(hours=1))

    assert fill is not None
    assert fill["exit_price"] == 101.5
    assert fill["filled_qty"] == 50.0
    assert fill["exit_timestamp"] == now - timedelta(minutes=1)


def test_returns_none_when_no_orders():
    client = _client_with_orders([])
    fill = client.get_last_closing_fill("AAPL", "sell", after=datetime.now(timezone.utc))
    assert fill is None


def test_skips_orders_without_filled_at():
    now = datetime.now(timezone.utc)
    orders = [
        _make_order(filled_at=None, filled_avg_price="100.0", filled_qty="10"),
        _make_order(filled_at=now, filled_avg_price="99.5", filled_qty="10"),
    ]
    client = _client_with_orders(orders)

    fill = client.get_last_closing_fill("AAPL", "sell", after=now - timedelta(hours=1))

    assert fill is not None
    assert fill["exit_price"] == 99.5


def test_skips_orders_with_zero_filled_qty():
    now = datetime.now(timezone.utc)
    orders = [
        _make_order(filled_at=now, filled_avg_price="100.0", filled_qty="0"),
        _make_order(filled_at=now - timedelta(seconds=10), filled_avg_price="99.0", filled_qty="5"),
    ]
    client = _client_with_orders(orders)

    fill = client.get_last_closing_fill("AAPL", "sell", after=now - timedelta(hours=1))

    assert fill is not None
    assert fill["exit_price"] == 99.0


def test_skips_orders_with_zero_price():
    now = datetime.now(timezone.utc)
    orders = [
        _make_order(filled_at=now, filled_avg_price="0", filled_qty="5"),
        _make_order(filled_at=now - timedelta(seconds=10), filled_avg_price="50.0", filled_qty="5"),
    ]
    client = _client_with_orders(orders)

    fill = client.get_last_closing_fill("AAPL", "sell", after=now - timedelta(hours=1))

    assert fill is not None
    assert fill["exit_price"] == 50.0


def test_skips_unparseable_numeric_fields():
    now = datetime.now(timezone.utc)
    orders = [
        _make_order(filled_at=now, filled_avg_price="NaN", filled_qty="abc"),
        _make_order(filled_at=now - timedelta(seconds=10), filled_avg_price="25.0", filled_qty="4"),
    ]
    client = _client_with_orders(orders)

    fill = client.get_last_closing_fill("AAPL", "sell", after=now - timedelta(hours=1))

    assert fill is not None
    assert fill["exit_price"] == 25.0


def test_returns_none_when_client_is_dry_run():
    c = AlpacaClient.__new__(AlpacaClient)
    c._client = None
    c._data_client = None
    fill = c.get_last_closing_fill("AAPL", "sell", after=datetime.now(timezone.utc))
    assert fill is None


def test_returns_none_when_get_orders_raises():
    c = AlpacaClient.__new__(AlpacaClient)
    c._client = MagicMock()
    c._data_client = None
    c._client.get_orders.side_effect = RuntimeError("Alpaca exploded")

    fill = c.get_last_closing_fill("AAPL", "sell", after=datetime.now(timezone.utc))
    assert fill is None


def test_passes_correct_side_enum_to_alpaca():
    """Verify that 'buy' (short cover) maps to OrderSide.BUY and vice versa."""
    from alpaca.trading.enums import OrderSide

    now = datetime.now(timezone.utc)
    client = _client_with_orders([])

    client.get_last_closing_fill("TSLA", "buy", after=now)
    req = client._client.get_orders.call_args.kwargs["filter"]
    assert req.side == OrderSide.BUY
    assert req.symbols == ["TSLA"]

    client._client.get_orders.reset_mock()
    client.get_last_closing_fill("AAPL", "sell", after=now)
    req = client._client.get_orders.call_args.kwargs["filter"]
    assert req.side == OrderSide.SELL

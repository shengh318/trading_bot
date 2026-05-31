"""Tests for Alpaca API routes (/api/alpaca/*)."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api.main import app

client = TestClient(app)


class TestAlpacaAccount:
    def test_account_returns_503_when_keys_missing(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", ""), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", ""):
            response = client.get("/api/alpaca/account")
            assert response.status_code == 503
            assert "Alpaca API keys not configured" in response.json()["detail"]

    def test_account_returns_summary(self):
        mock_acct = MagicMock()
        mock_acct.cash = "10000.00"
        mock_acct.equity = "15000.00"
        mock_acct.buying_power = "20000.00"
        mock_acct.last_equity = "14900.00"

        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_account.return_value = mock_acct
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/account")
            assert response.status_code == 200
            data = response.json()
            assert data["cash"] == 10000.0
            assert data["portfolio_value"] == 15000.0
            assert data["buying_power"] == 20000.0
            assert data["day_pnl"] == 100.0  # equity - last_equity

    def test_account_day_pnl_zero_when_no_last_equity(self):
        mock_acct = MagicMock()
        mock_acct.cash = "10000.00"
        mock_acct.equity = "15000.00"
        mock_acct.buying_power = "20000.00"
        mock_acct.last_equity = None

        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_account.return_value = mock_acct
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/account")
            data = response.json()
            assert data["day_pnl"] == 0.0


class TestAlpacaPositions:
    def test_positions_returns_503_when_keys_missing(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", ""), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", ""):
            response = client.get("/api/alpaca/positions")
            assert response.status_code == 503

    def test_positions_returns_empty_list(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_all_positions.return_value = []
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/positions")
            assert response.status_code == 200
            assert response.json() == []

    def test_positions_returns_formatted_positions(self):
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1550.00"
        mock_pos.avg_entry_price = "150.00"
        mock_pos.current_price = "155.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.unrealized_plpc = "0.0333"

        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_all_positions.return_value = [mock_pos]
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/positions")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["symbol"] == "AAPL"
            assert data[0]["qty"] == 10.0
            assert data[0]["market_value"] == 1550.0
            assert data[0]["unrealized_plpc"] == 3.33  # 0.0333 * 100


class TestAlpacaOrders:
    def test_orders_returns_503_when_keys_missing(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", ""), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", ""):
            response = client.get("/api/alpaca/orders")
            assert response.status_code == 503

    def test_orders_accepts_limit_param(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_orders.return_value = []
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/orders?limit=50")
            assert response.status_code == 200

    def test_orders_formats_order_data(self):
        mock_order = MagicMock()
        mock_order.id = "ord_123"
        mock_order.symbol = "AAPL"
        mock_order.side = MagicMock()
        mock_order.side.value = "buy"
        mock_order.type = MagicMock()
        mock_order.type.value = "market"
        mock_order.qty = "10"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.50"
        mock_order.status = MagicMock()
        mock_order.status.value = "filled"
        mock_order.created_at = MagicMock()
        mock_order.created_at.isoformat.return_value = "2024-01-15T10:00:00"
        mock_order.updated_at = MagicMock()
        mock_order.updated_at.isoformat.return_value = "2024-01-15T10:01:00"

        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_orders.return_value = [mock_order]
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/orders")
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "ord_123"
            assert data[0]["side"] == "buy"
            assert data[0]["filled_avg_price"] == 150.50

    def test_orders_handles_null_avg_price(self):
        mock_order = MagicMock()
        mock_order.id = "ord_456"
        mock_order.symbol = "GOOG"
        mock_order.side = MagicMock()
        mock_order.side.value = "sell"
        mock_order.type = MagicMock()
        mock_order.type.value = "limit"
        mock_order.qty = "5"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None
        mock_order.status = MagicMock()
        mock_order.status.value = "pending"
        mock_order.created_at = MagicMock()
        mock_order.created_at.isoformat.return_value = "2024-01-16T10:00:00"
        mock_order.updated_at = MagicMock()
        mock_order.updated_at.isoformat.return_value = "2024-01-16T10:00:00"

        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_orders.return_value = [mock_order]
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/orders")
            data = response.json()
            assert data[0]["filled_avg_price"] is None


class TestAlpacaPortfolioHistory:
    def test_portfolio_history_returns_empty_on_exception(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_portfolio_history.side_effect = Exception("API error")
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/portfolio-history")
            assert response.status_code == 200
            assert response.json() == []

    def test_portfolio_history_returns_formatted_data(self):
        with patch("backend.api.alpaca_routes.ALPACA_API_KEY", "test_key"), \
             patch("backend.api.alpaca_routes.ALPACA_SECRET_KEY", "test_secret"), \
             patch("backend.api.alpaca_routes.TradingClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_history = MagicMock()
            mock_history.timestamp = [1700000000, 1700003600]
            mock_history.equity = [10000.0, 10100.0]
            mock_client.get_portfolio_history.return_value = mock_history
            mock_client_cls.return_value = mock_client

            response = client.get("/api/alpaca/portfolio-history?period=1M&timeframe=1D")
            data = response.json()
            assert len(data) == 2
            assert data[0]["timestamp"] == 1700000000
            assert data[0]["equity"] == 10000.0
            assert data[1]["timestamp"] == 1700003600

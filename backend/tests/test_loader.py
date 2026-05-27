from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.data.loader import DataLoader


@pytest.fixture(autouse=True)
def mock_alpaca_client():
    mock_df = pd.DataFrame(
        {
            "open": [150.0, 151.0],
            "high": [152.0, 153.0],
            "low": [149.0, 150.0],
            "close": [151.0, 152.0],
            "volume": [10000, 12000],
            "trade_count": [200, 250],
            "vwap": [150.5, 151.5],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    mock_df.index.name = "timestamp"

    mock_bars = MagicMock()
    mock_bars.df = mock_df.reset_index()
    mock_bars.df.insert(0, "symbol", "AAPL")
    mock_bars.df = mock_bars.df.set_index(["symbol", "timestamp"])

    with patch("backend.data.loader.StockHistoricalDataClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_bars
        mock_client_cls.return_value = mock_client
        yield mock_client_cls


class TestDataLoader:
    def test_initializes_client(self, mock_alpaca_client):
        dl = DataLoader()
        mock_alpaca_client.assert_called_once()

    def test_load_bars_returns_dataframe(self, mock_alpaca_client):
        from datetime import datetime
        dl = DataLoader()
        df = dl.load_bars("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "close" in df.columns
        assert df.index.name == "timestamp"

    def test_load_bars_raises_on_empty_data(self, mock_alpaca_client):
        from datetime import datetime
        mock_alpaca_client.return_value.get_stock_bars.return_value.df = pd.DataFrame()
        dl = DataLoader()
        with pytest.raises(ValueError, match="No data returned"):
            dl.load_bars("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31), use_cache=False)

    def test_clear_cache(self, mock_alpaca_client, tmp_path):
        from datetime import datetime

        dl = DataLoader()
        df = dl.load_bars("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31))
        assert len(dl.list_cache()) == 1
        dl.clear_cache()
        assert dl.list_cache() == []

    def test_load_bars_uses_cache(self, mock_alpaca_client, tmp_path):
        from datetime import datetime

        dl = DataLoader()
        df1 = dl.load_bars("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31))
        dl.load_bars("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31))
        # Should only call the API once
        mock_alpaca_client.return_value.get_stock_bars.assert_called_once()

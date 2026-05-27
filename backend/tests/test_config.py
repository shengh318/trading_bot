from backend import config


def test_validate_config_returns_errors_when_keys_missing(monkeypatch):
    monkeypatch.setattr(config, "ALPACA_API_KEY", "")
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "")
    errors = config.validate_config()
    assert len(errors) == 2
    assert "ALPACA_API_KEY" in errors[0]
    assert "ALPACA_SECRET_KEY" in errors[1]


def test_validate_config_returns_empty_when_keys_present(monkeypatch):
    monkeypatch.setattr(config, "ALPACA_API_KEY", "test_key")
    monkeypatch.setattr(config, "ALPACA_SECRET_KEY", "test_secret")
    assert config.validate_config() == []


def test_db_path_defaults_to_backend_dir():
    assert config.DB_PATH.endswith("trader.db")


def test_alpaca_paper_defaults_to_true(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "true")
    config.load_dotenv()
    assert config.ALPACA_PAPER is True
    assert "paper-api" in config.ALPACA_BASE_URL

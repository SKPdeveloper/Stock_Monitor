# Stock Monitor - ML-Powered Stock Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Polygon.io](https://img.shields.io/badge/API-Polygon.io-green.svg)](https://polygon.io/)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)

**Professional stock monitoring system** with ensemble machine learning models for price prediction, real-time signal generation, and Telegram notifications.

---

## Features

### Machine Learning
- **Ensemble Models**: RandomForest, GradientBoosting, LightGBM, XGBoost, CatBoost
- **Multi-Horizon Predictions**: 1h, 3h, 6h, 12h, 24h (short-term) + 3d, 7d, 14d, 30d (long-term)
- **Automatic Retraining**: Every 6 hours per ticker
- **Historical Accuracy Tracking**: Model performance monitoring

### Technical Analysis
- **Indicators**: RSI, EMA (12/26), Bollinger Bands, MACD, ATR, ADX
- **Volume Analysis**: Spike detection (1.6x multiplier)
- **Pattern Recognition**: Breakouts, trend reversals, support/resistance

### News & Sentiment
- **Polygon News API**: Real-time market news integration
- **Sentiment Analysis**: Positive/negative scoring with intensity multipliers
- **Importance Scoring**: 0-5 scale based on keywords (earnings, mergers, FDA, etc.)
- **News Features for ML**: 1-day, 3-day, 7-day sentiment metrics

### Signal Generation
- **Priority Levels**: Critical, Important, Info
- **Signal Types**: Price changes, predictions, volume spikes, technical breakouts
- **Deduplication**: 30-minute window to prevent alert spam
- **Confidence Scoring**: Based on model consensus and confirming factors

### Trading Modes
- **Conservative**: 0.4% price change threshold
- **Balanced**: 0.2% price change threshold
- **Aggressive**: 0.1% price change (scalping)
- **Professional**: 0.15% price change (default)

---

## Architecture

```
+---------------------------------------------------------------+
|                    STOCK MONITOR SYSTEM                        |
+---------------------------------------------------------------+
|                                                                |
|  +------------+     +------------+     +------------------+    |
|  | Polygon.io |---->|   Cache    |---->|    Monitor.py    |    |
|  |    API     |     |  (JSON)    |     |   Main Engine    |    |
|  +------------+     +------------+     +--------+---------+    |
|                                                 |              |
|       +----------------+------------------------+              |
|       |                |                        |              |
|       v                v                        v              |
|  +----------+   +-------------+        +----------------+      |
|  |  ML      |   | Technical   |        | NewsAnalyzer   |      |
|  | Ensemble |   |  Indicators |        |  Sentiment     |      |
|  +----+-----+   +------+------+        +-------+--------+      |
|       |                |                       |               |
|       +----------------+-----------------------+               |
|                        |                                       |
|                        v                                       |
|              +-------------------+                              |
|              |  SignalEngine.py  |                              |
|              |  Signal Generator |                              |
|              +--------+----------+                              |
|                       |                                        |
|                       v                                        |
|              +-------------------+                              |
|              |  Telegram Bot     |                              |
|              |  Notifications    |                              |
|              +-------------------+                              |
|                                                                |
+---------------------------------------------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10+
- Polygon.io API key (free tier: 5 req/min, paid: 300 req/min)
- Telegram Bot Token

### Setup

```bash
# Clone repository
git clone https://github.com/SKPdeveloper/Stock_Monitor.git
cd Stock_Monitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.py`:

```python
# API Keys
POLYGON_API_KEY = "your_polygon_api_key"
TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

# Trading Mode
TRADING_MODE = "professional"  # conservative, balanced, aggressive, professional
```

---

## Usage

### Start Monitoring

```bash
# Activate environment
source venv/bin/activate

# Run main monitor
python Monitor.py
```

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Show main menu |
| `/add TICKER` | Add ticker to watchlist |
| `/remove TICKER` | Remove ticker |
| `/list` | Show all tickers |
| `/status` | System status |
| `/pause` | Pause monitoring |
| `/resume` | Resume monitoring |

---

## Project Structure

```
Stock_Monitor/
|-- Monitor.py           # Main engine (ML, data collection, predictions)
|-- SignalEngine.py      # Signal generation and prioritization
|-- NewsAnalyzer.py      # Sentiment analysis and news processing
|-- database.py          # SQLite database manager
|-- config.py            # Configuration settings
|-- requirements.txt     # Python dependencies
|
|-- cache/               # API response cache (JSON)
|-- data/                # Application state
|   |-- tickers.json     # Watchlist
|   +-- predictions_history.json
|-- models/              # Trained ML models (pickle)
|-- logs/                # Application logs
+-- reports/             # Generated reports
```

---

## ML System Details

### Feature Engineering
- Technical indicators (RSI, EMA, BB, MACD, ATR, ADX)
- Price action patterns
- Volume analysis
- News sentiment scores (1d, 3d, 7d)

### Ensemble Voting
```
Prediction = Weighted average of:
- RandomForest (base)
- HistGradientBoosting (fast)
- LightGBM (efficient)
- XGBoost (accurate)
- CatBoost (categorical)
```

### Confidence Calculation
- Model consensus (lower divergence = higher confidence)
- Number of confirming factors (+10-25%)
- Historical accuracy bonus (+20% for >60% accuracy)
- Volatility adjustment (+/-10%)

---

## Signal Thresholds

| Prediction Horizon | Threshold |
|--------------------|-----------|
| 1 hour | 0.15% |
| 3 hours | 0.25% |
| 6 hours | 0.40% |
| 1 day | 0.80% |

### Time-Based Multipliers

| Period | Multiplier |
|--------|------------|
| Market Open (9:30-10:30) | 0.5x |
| Market Close (15:00-16:00) | 0.6x |
| Regular Hours | 1.0x |
| After Hours | 1.2x |

---

## Database Schema

```sql
-- Predictions tracking
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR,
    prediction_time TIMESTAMP,
    horizon VARCHAR,
    predicted_change FLOAT,
    actual_change FLOAT,
    confidence FLOAT,
    is_verified BOOLEAN
);

-- Signals log
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    ticker VARCHAR,
    signal_type VARCHAR,
    priority VARCHAR,
    confidence FLOAT,
    created_at TIMESTAMP
);

-- Model performance
CREATE TABLE model_performance (
    ticker VARCHAR,
    model_type VARCHAR,
    accuracy_1d FLOAT,
    accuracy_7d FLOAT,
    accuracy_30d FLOAT,
    last_updated TIMESTAMP
);
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=1.7.0
catboost>=1.2.0
ta-lib>=0.4.25
python-telegram-bot>=20.0
requests>=2.31.0
textstat>=0.7.3
```

---

## License

MIT License

---

## Disclaimer

**This software is for educational and research purposes only.**

Stock trading involves substantial risk of loss. Past performance does not guarantee future results. Use this software at your own risk.

---

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

---

🌐 **[SKP-Degree](https://skp-degree.com.ua)** — Pair programming, курсові та дипломні роботи з програмування. Без передоплати!

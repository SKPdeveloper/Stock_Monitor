# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands

### Running the Application
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main monitoring application
python Monitor.py

# Run the Telegram bot
python telegram_bot.py
```

### Development Commands
```bash
# Check Python syntax errors
python -m py_compile Monitor.py SignalEngine.py telegram_bot.py

# Format code (if black is installed)
python -m black Monitor.py SignalEngine.py telegram_bot.py config.py

# Type checking (if mypy is installed)
python -m mypy Monitor.py SignalEngine.py telegram_bot.py
```

## Architecture Overview

This is a stock monitoring system that uses machine learning to predict price movements and generates trading signals via Telegram alerts.

### Core Components

1. **Monitor.py** - Main engine that:
   - Fetches data from Polygon.io API (with intelligent caching)
   - Calculates technical indicators (RSI, EMA, Bollinger Bands, MACD, etc.)
   - Trains ensemble ML models for short-term (hours) and long-term (days) predictions
   - Integrates multiple ML algorithms: RandomForest, GradientBoosting, LightGBM, XGBoost, CatBoost
   - Performs macroeconomic analysis and sentiment scoring

2. **SignalEngine.py** - Signal generation system that:
   - Analyzes price movements, volume spikes, and technical patterns
   - Prioritizes signals (Critical, Important, Info)
   - Implements deduplication to prevent alert spam
   - Generates contextualized trading signals

3. **telegram_bot.py** - User interface that:
   - Provides async Telegram bot for real-time notifications
   - Handles commands for ticker management and monitoring control
   - Delivers prioritized alerts based on signal importance

### Data Flow

1. **Data Collection**: Polygon.io API → Cache (JSON files) → DataFrame processing
2. **Feature Engineering**: Technical indicators + Market data → Feature vectors
3. **Prediction**: Ensemble models → Short/Long-term predictions with confidence scores
4. **Signal Generation**: Price/Volume/Technical analysis → Prioritized signals
5. **Delivery**: SignalEngine → Telegram Bot → User notifications

### Key Directories

- `cache/` - API response caching (historical and intraday data)
- `data/` - Application state (tickers.json, predictions_history.json)
- `models/` - Trained ML models (pickle files per ticker)
- `logs/` - Application logs

### Important Considerations

- **API Limits**: Free Polygon.io tier has 5 requests/minute limit
- **Trading Hours**: Configurable in config.py (default 9:00-20:00 EST)
- **Model Retraining**: Automatic every 6 hours per ticker for aggressive optimization
- **Caching Strategy**: Reduces API calls by storing historical/intraday data
- **Language**: Code comments primarily in Ukrainian/Russian

### **🚨 REAL TRADING CONFIGURATION (Реальні налаштування для трейдингу)**

Система налаштована для **професійного трейдингу** з максимально реалістичними порогами:

#### **Signal Thresholds (Пороги сигналів):**
- **1-hour predictions**: 0.15% (типовий внутрішньоденний рух)
- **3-hour predictions**: 0.25% (значуща зміна тренду)
- **6-hour predictions**: 0.4% (половина торгового дня)
- **Daily predictions**: 0.8% (денний рух стабільних акцій)
- **Price change alerts**: 0.15% (мінімум для скальпінгу)
- **Volume spikes**: 1.6x від середнього (реальний сплеск)
- **RSI oversold/overbought**: 25/75 (агресивні рівні для професіоналів)

#### **Trading Modes (Режими торгівлі в config.py):**
- **Conservative**: Безпечні сигнали (0.4% price change)
- **Balanced**: Збалансований підхід (0.2% price change)  
- **Aggressive**: Скальпінг (0.1% price change)
- **Professional** 🎯: Поточний режим (0.15% price change)

#### **Time-based Multipliers (Адаптація за часом торгівлі):**
- **Market Open** (9:30-10:30): 0.5x множник (найвища волатільність)
- **Market Close** (15:00-16:00): 0.6x множник (висока активність перед закриттям)
- **Regular Hours** (10:30-15:00): 1.0x множник (стандартна торгівля)
- **After Hours** (16:00-20:00): 1.2x множник (менша ліквідність)

#### **Signal Limits (Обмеження сигналів):**
- **Max signals per hour**: 15 (активний трейдинг)
- **Max signals per day**: 80 (професійний рівень)
- **Min confidence**: 50% (підвищено для якості)
- **Deduplication windows**: Скорочені для швидкої реакції

#### **🎯 ENHANCED CONFIDENCE SYSTEM (Покращена система довіри):**

**Ensemble Model Confidence** враховує:
- **Консенсус моделей**: Чим менша розбіжність прогнозів, тим вища довіра
- **Кількість моделей**: Більше моделей в ансамблі = вища довіра (до +20%)
- **Історична точність**: Бонус до +20% для моделей з точністю >60%
- **Якість прогнозів**: Враховується індивідуальна точність кожної моделі

**Price Signal Confidence** базується на:
- **Перевищення порогу**: Чим більше перевищення, тим вища довіра
- **Кількість факторів**: 2+ фактори дають +10%, 3+ дають +15%, 4+ дають +25%
- **Критичні фактори**: volume_spike, breakout, consensus дають +5% кожен
- **Волатільність**: Низька волатільність (+10%), висока (-10%)

**RSI Signal Confidence** включає:
- **Екстремальні значення**: RSI ≤20 або ≥80 (+15%), RSI ≤15 або ≥85 (+25%)
- **Дистанція від порогу**: Чим далі від 30/70, тим вища довіра
- **Додаткові фактори**: Дивергенція, тренд дають бонуси

**Volume Signal Confidence** враховує:
- **Величину сплеску**: 5x+ від середнього (90%), 3x+ (80%), 2x+ (70%)
- **Прорив рівнів**: Прорив 20-денних максимумів/мінімумів (+15%)
- **Підтверждуючі фактори**: Кожен додатковий фактор (+3%)

### Common Tasks

- **Add new ticker**: Add to `tickers_to_add.txt` or use Telegram bot `/add` command
- **Adjust sensitivity**: Modify `SIGNAL_CONFIG['default_sensitivity']` in `config.py`
- **Change prediction periods**: Update MODEL_CONFIG in `config.py`
- **Debug predictions**: Check `data/predictions_history.json` for historical accuracy
- **Clear cache**: Delete files in `cache/` directory to force fresh data fetch
- **Real-time monitoring**: Check logs for "SIGNAL DEBUG" and "PREDICTION FILTER" entries
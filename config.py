"""
Конфігурація для Stock Monitor
"""
import os
from datetime import datetime

# API ключі
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "Q4p8SrLUG0BgysTsxPBpqmwfaCJE0KlF")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8163805579:AAH34IbULdAwlm4zUSftyYTk6PTTPmiY4o8")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1002511990466")

# Обмеження безкоштовного тарифу Polygon
FREE_TIER_LIMITS = {
    'requests_per_minute': 5,
    'max_historical_years': 2,
    'data_delay_minutes': 15,
    'websocket_enabled': False
}

# Налаштування таймаутів для API запитів
TIMEOUT_CONFIG = {
    'http_timeout': 180,  # HTTP запити (збільшено до 3 хв)
    'connect_timeout': 60,  # Час для встановлення з'єднання
    'read_timeout': 180,  # Час для читання відповіді
    'large_response_timeout': 600,  # Великі відповіді (10 хв)
    'ticker_update_timeout': 900,  # Оновлення тікера (15 хв)
    'overall_processing_timeout': 1800,  # Загальне оброблення (30 хв)
    'model_training_timeout': 3600,  # Перенавчання моделі (60 хв)
    'model_training_fallback_timeout': 1200,  # Фолбек перенавчання (20 хв)
    'max_retries': 3,  # Кількість повторних спроб
    'retry_delay_base': 2,  # Базова затримка між спробами
    'retry_delay_multiplier': 2  # Множник затримки
}

# Налаштування моделей
MODEL_CONFIG = {
    'retrain_interval_hours': 6,  # Агресивне перенавчання кожні 6 годин
    'min_data_points': 200,
    'force_retrain_accuracy': 20,  # Примусове перенавчання при <20% точності
    'performance_check_interval': 3,  # Перевірка якості кожні 3 години
    
    # AutoML налаштування
    'automl_enabled': False,
    'automl_time_limit': 60,  # 1 хвилина на пошук кращої моделі (прискорено)
    'automl_memory_limit': 2000,  # 2GB RAM для AutoML
    'automl_ensemble_size': 5,  # Максимум 5 моделей в ансамблі (прискорено)
    'automl_optimization_metric': 'r2',  # Оптимізуємо по R² для регресії
    
    # Короткострокова модель (години)
    'short_term_model': {
        'periods': [1, 3, 6, 12, 24],  # години
        'features_window': 48,  # години історії для фіч
        'automl_preset': 'fast',  # Ускоренный режим
        'automl_algorithms': ['lgbm', 'xgboost', 'histgb']  # Только быстрые алгоритмы
    },
    
    # Довгострокова модель (дні)
    'long_term_model': {
        'periods': [3, 7, 14, 30],  # дні
        'features_window': 60,  # дні історії для фіч
        'automl_preset': 'fast',  # Ускоренный режим и для долгих
        'automl_algorithms': ['lgbm', 'xgboost', 'histgb'],  # Только быстрые алгоритмы
        'smoothing_enabled': False,  # Отключить искусственное сглаживание
    },
    
    'confidence_threshold': 0.6,
    'volatility_multiplier': 1.5,
    'news_impact_weight': 0.3  # Вага впливу новин
}

# Пороги сповіщень для професійного трейдингу
ALERT_THRESHOLDS = {
    'price_change_percent': 0.15,  # 0.15% - мінімальна значуща зміна для скальпінгу
    'volume_spike_multiplier': 1.6,  # 1.6x - реальний сплеск об'єму
    'rsi_oversold': 25,  # 25 - агресивна перепроданість
    'rsi_overbought': 75,  # 75 - агресивна перекупленість  
    'prediction_change_percent': 0.3,  # 0.3% - мінімальна зміна прогнозу
    'news_sentiment_threshold': 0.25  # 25% - чутливість до новин
}

# Технічні індикатори
TECHNICAL_INDICATORS = {
    'rsi_period': 14,
    'ema_short': 12,
    'ema_long': 26,
    'bb_period': 20,
    'bb_std': 2,
    'volume_ma': 20,
    'atr_period': 14,
    'adx_period': 14,
    'macd_signal': 9
}

# Новинний аналіз
NEWS_CONFIG = {
    'lookback_days': 30,  # Скільки днів новин аналізувати
    'max_news_per_ticker': 100,
    'sentiment_keywords': {
        'positive': ['growth', 'profit', 'gains', 'positive', 'strong', 'upgrade', 
                    'beat', 'exceed', 'outperform', 'bullish', 'surge', 'rally'],
        'negative': ['loss', 'decline', 'drop', 'negative', 'weak', 'downgrade',
                    'miss', 'underperform', 'bearish', 'concern', 'fall', 'plunge']
    }
}

SIGNAL_CONFIG = {
    'sensitivity_presets': {
        'conservative': {  # Консервативний трейдинг
            'price_change_percent': 0.4,
            'prediction_change_percent': 0.8,
            'volume_spike_multiplier': 2.2
        },
        'balanced': {  # Збалансований підхід
            'price_change_percent': 0.2,
            'prediction_change_percent': 0.4,
            'volume_spike_multiplier': 1.8
        },
        'aggressive': {  # Агресивний скальпінг
            'price_change_percent': 0.1,
            'prediction_change_percent': 0.2,
            'volume_spike_multiplier': 1.4
        },
        'professional': {  # Професійний трейдинг
            'price_change_percent': 0.15,
            'prediction_change_percent': 0.3,
            'volume_spike_multiplier': 1.6
        }
    },
    'default_sensitivity': 'professional'  # Встановлюємо професійний режим за замовчуванням
}

# Шляхи файлів
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
CACHE_DIR = "cache"

# База даних
DATABASE_CONFIG = {
    'path': 'data/stock_monitor.db',
    'backup_interval_hours': 12,
    'cleanup_days': 90,
    'vacuum_interval_days': 7
}

# Покращені налаштування прогнозів
PREDICTION_CONFIG = {
    'verification_tolerance': 0.03,  # 3% tolerance - більш строга точність
    'min_confidence_threshold': 0.4,  # 40% - підвищено для якісніших сигналів
    'dynamic_threshold_adjustment': True,
    'ensemble_weight_decay': 0.98,  # Повільніше зниження ваги старих моделей
    'feature_importance_tracking': True,
    'prediction_validation_timeout': 12,  # 12 годин - швидша валідація
    'auto_retrain_threshold': 35,  # 35% - менш агресивне перенавчання
    'critical_retrain_threshold': 20,  # 20% - критичний поріг
    'sliding_window_size': 15,  # Оцінка на останніх 15 прогнозах
    'min_predictions_for_eval': 8,  # Мінімум для оцінки якості
    'real_time_monitoring': True,  # Моніторинг у реальному часі
}

# Розширені налаштування фільтрації
FILTER_CONFIG = {
    'default_criteria': {
        'min_confidence': 0.5,  # 50% - підвищено для якісніших сигналів
        'min_price_change': 0.15,  # 0.15% - реалістичний мінімум для трейдингу
        'max_signals_per_hour': 15,  # 15 сигналів/годину - активний трейдинг
        'max_signals_per_day': 80,  # 80 сигналів/день - професійний рівень
        'noise_threshold': 0.25,  # 25% - менш строга фільтрація шуму
        'correlation_window_minutes': 10,  # 10 хв - швидша кореляція
        'min_model_accuracy': 35.0  # 35% - нижчий поріг точності моделі
    },
    'adaptive_filtering': True,
    'learning_rate': 0.15,  # Швидше навчання фільтрів
    'filter_optimization_interval': 12  # 12 годин - частіша оптимізація
}

# Database connection settings
DATABASE_URL = f"sqlite:///{DATABASE_CONFIG['path']}"

# Enhanced error handling
ERROR_HANDLING = {
    'max_retries': 3,
    'retry_delay': 2.0,
    'exponential_backoff': True,
    'safe_mode': True,  # Enable safe mode for production
    'log_all_errors': True
}

# Performance optimization
PERFORMANCE_CONFIG = {
    'connection_pool_size': 10,
    'query_timeout': 30,
    'batch_size': 100,
    'cache_size': 1000,
    'enable_wal_mode': True
}

# Створення директорій
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Торговые часы (EST/EDT)
TRADING_HOURS = {
    'start': '09:00',  # 9:00 AM EST
    'end': '20:00',    # 8:00 PM EST
    'timezone': 'US/Eastern'
}

# Поточний час для логів
CURRENT_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
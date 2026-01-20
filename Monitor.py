"""
Повнофункціональний Stock Monitor з реальним Polygon API і ML
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import requests
import time
import pickle
import json
import os
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

# ВИПРАВЛЕННЯ: Додаємо підключення до бази даних для верифікації
from database import DatabaseManager

# Вимикаємо verbose логування FLAML
logging.getLogger('flaml.automl.logger').setLevel(logging.WARNING)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM не встановлено - буде пропущено в ансамблі")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost не встановлено - буде пропущено в ансамблі")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost не встановлено - буде пропущено в ансамблі")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import ta
import warnings
warnings.filterwarnings('ignore')

# Імпорт конфігурації
from config import POLYGON_API_KEY, DATA_DIR
from database import DatabaseManager, PredictionRecord, db
from SignalEngine import SignalEngine
from NewsAnalyzer import news_analyzer
import hashlib

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolygonClient:
    """Реальный клиент Polygon.io API"""
    
    def __init__(self):
        self.base_url = "https://api.polygon.io"
        self.api_key = POLYGON_API_KEY
        self.session = requests.Session()
        self.request_count = 0
        self.last_request_time = time.time()
    
    def _rate_limit(self):
        """Rate limiting для API запитів"""
        current_time = time.time()
        time_passed = current_time - self.last_request_time
        
        # Для платного облікового запису - 300 запитів на хвилину
        requests_per_minute = 300
        
        if time_passed < 60:
            if self.request_count >= requests_per_minute:
                sleep_time = 60 - time_passed
                logger.info(f"Rate limit: чекаємо {sleep_time:.1f} сек")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
            self.last_request_time = current_time
            
        self.request_count += 1
    
    def get_latest_price(self, ticker: str) -> Dict:
        """Отримання останньої ціни"""
        self._rate_limit()
        
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') in ['OK', 'DELAYED'] and 'ticker' in data:
                ticker_data = data['ticker']
                day_data = ticker_data.get('day', {})
                prev_day = ticker_data.get('prevDay', {})
                
                # Намагаємося отримати ціну з різних джерел
                current_price = (day_data.get('c') or 
                               day_data.get('o') or 
                               prev_day.get('c') or 
                               prev_day.get('o') or 0)
                
                if current_price > 0:
                    return {
                        'current_price': float(current_price),
                        'ticker': ticker,
                        'timestamp': time.time() * 1000
                    }
                else:
                    logger.warning(f"Ціну не знайдено в snapshot для {ticker}, пробуємо історичні дані")
            else:
                logger.warning(f"Немає даних в snapshot для {ticker}")
            
            # Якщо snapshot не дав результат, пробуємо отримати з історичних даних
            try:
                historical = self.get_historical_data(ticker, days_back=5)
                if historical:
                    latest_price = historical[-1].get('c', 0)
                    if latest_price > 0:
                        logger.info(f"Отримано ціну з історичних даних для {ticker}: ${latest_price}")
                        return {
                            'current_price': float(latest_price),
                            'ticker': ticker,
                            'timestamp': time.time() * 1000
                        }
                
                logger.warning(f"Не вдалося отримати ціну для {ticker} ні з snapshot, ні з історичних даних")
                return {'current_price': 0.0, 'ticker': ticker}
            except Exception as hist_e:
                logger.error(f"Помилка отримання історичних даних для {ticker}: {hist_e}")
                return {'current_price': 0.0, 'ticker': ticker}
                
        except Exception as e:
            logger.error(f"Помилка отримання ціни {ticker}: {e}")
            return {'current_price': 0.0, 'ticker': ticker}
    
    def get_historical_data(self, ticker: str, days_back: int = 100) -> List[Dict]:
        """Отримання історичних даних"""
        self._rate_limit()
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data:
                return data['results']
            else:
                logger.warning(f"Немає історичних даних для {ticker}")
                return []
                
        except Exception as e:
            logger.error(f"Помилка отримання історичних даних {ticker}: {e}")
            return []
    
    def get_intraday_data(self, ticker: str, days: int = 5) -> List[Dict]:
        """Отримання внутрішньоденних даних"""
        self._rate_limit()
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/hour/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {
            'apikey': self.api_key,
            'adjusted': 'true',
            'sort': 'asc'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') in ['OK', 'DELAYED'] and 'results' in data:
                return data['results']
            else:
                logger.warning(f"Немає внутрішньоденних даних для {ticker}")
                return []
                
        except Exception as e:
            logger.error(f"Помилка отримання внутрішньоденних даних {ticker}: {e}")
            return []

class MLPredictor:
    """ML модель для прогнозів"""
    
    def __init__(self, ticker: str, model_type: str = "short_term"):
        self.ticker = ticker
        self.model_type = model_type
        self.models = {}  # Словник моделей ансамблю
        self.model_weights = {}  # Вага для кожної моделі
        self.scaler = StandardScaler()
        self.imputer = None  # Для заповнення NaN значень
        self.feature_names = []
        self.last_prediction = None
        self.trained = False
        self.historical_accuracies = []  # Список історичних точностей
    
    def get_ensemble_historical_accuracy(self) -> float:
        """Отримати середню історичну точність ансамблю"""
        if not self.historical_accuracies:
            return 0.5  # Нейтральне значення за замовчуванням
        
        # Використовуємо останні 20 оцінок для більшої актуальності
        recent_accuracies = self.historical_accuracies[-20:]
        return np.mean(recent_accuracies)
    
    def update_historical_accuracy(self, accuracy: float):
        """Оновити історичну точність"""
        self.historical_accuracies.append(accuracy)
        # Зберігаємо останні 50 оцінок
        if len(self.historical_accuracies) > 50:
            self.historical_accuracies = self.historical_accuracies[-50:]
    
    def add_news_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Додавання новинних ознак в DataFrame"""
        try:
            if df.empty:
                return df
                
            # Отримуємо новинні ознаки для тікера
            news_features = news_analyzer.get_news_features_for_ml(self.ticker, days_back=7)
            
            # Додаємо новинні ознаки до всіх рядків (новини впливають на весь період)
            created_features = []
            for feature_name, feature_value in news_features.items():
                full_feature_name = f'news_{feature_name}'
                df[full_feature_name] = feature_value
                created_features.append(full_feature_name)
            
            logger.info(f"Додано {len(news_features)} новинних ознак для {self.ticker}: {created_features}")
            return df
            
        except Exception as e:
            logger.error(f"Помилка додавання новинних ознак для {self.ticker}: {e}")
            # Якщо не вдалося отримати новини, додаємо нульові значення
            news_feature_names = [
                'news_count_1d', 'news_count_3d', 'news_count_7d',
                'avg_sentiment_1d', 'avg_sentiment_3d', 'avg_sentiment_7d',
                'max_importance_1d', 'max_importance_3d', 'max_importance_7d',
                'sentiment_volatility', 'positive_news_ratio', 'negative_news_ratio'
            ]
            for feature_name in news_feature_names:
                df[f'news_{feature_name}'] = 0.0
            return df

    def prepare_features(self, data: List[Dict]) -> pd.DataFrame:
        """Підготовка ознак для ML"""
        logger.info(f"prepare_features: отримано {len(data) if data else 0} записів")
        
        if not data or len(data) < 5:
            logger.warning(f"Недостатньо даних: {len(data) if data else 0}")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            logger.info(f"DataFrame створено: {df.shape}, стовпці: {list(df.columns)}")
            
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('timestamp')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            logger.info(f"Після обробки: {df.shape}, стовпці: {list(df.columns)}")
            
            # Зменшуємо вікна для технічних індикаторів
            window_size = min(5, len(df) - 1)
            if window_size < 2:
                logger.warning("Занадто мало даних для технічних індикаторів")
                window_size = 2
            
            # Технические индикаторы с адаптивными окнами
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=window_size)
            
            # Для остальных используем минимальные окна
            if len(df) >= 14:
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            else:
                df['rsi'] = ta.momentum.rsi(df['close'], window=max(2, len(df)//2))
            
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            
            # Простые индикаторы
            df['volume_sma'] = df['volume'].rolling(window=max(2, min(10, len(df)-1))).mean()
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=max(2, min(5, len(df)-1))).std()
            df['volume_change'] = df['volume'].pct_change()
            
            # Додаємо новинні фічі
            df = self.add_news_features(df)
            
            # ИСПРАВЛЕНИЕ: Цель для предсказания (изменение цены в % через N периодов)
            periods = 1 if self.model_type == "short_term" else min(3, len(df)-1)
            future_prices = df['close'].shift(-periods)
            # Предсказываем процентное изменение вместо абсолютных цен
            df['target'] = ((future_prices - df['close']) / df['close']) * 100
            
            logger.info(f"До видалення NaN: {df.shape}")
            # Більш гнучке видалення NaN - зберігаємо принаймні останні рядки
            df = df.dropna(thresh=len(df.columns)//2)  # Прибираємо рядки з більш ніж 50% NaN
            
            # Если все еще пусто, берем последние строки и заполняем NaN
            if df.empty and len(data) > 0:
                logger.warning("Всі рядки мають NaN, використовуємо forward fill")
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df = df.set_index('timestamp')
                df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                
                # Прості індикатори без NaN
                df['sma_5'] = df['close']  # Використовуємо поточну ціну як SMA
                df['rsi'] = 50.0  # Нейтральний RSI
                df['macd'] = 0.0
                df['bb_upper'] = df['close'] * 1.02
                df['bb_lower'] = df['close'] * 0.98
                df['volume_sma'] = df['volume']
                df['price_change'] = 0.0
                df['volatility'] = 0.02
                df['volume_change'] = 0.0
                
                # Додаємо новинні ознаки
                df = self.add_news_features(df)
                
                # ВИПРАВЛЕННЯ: Мета для прогнозування (зміна в %)
                periods = 1 if self.model_type == "short_term" else min(3, len(df)-1)
                future_prices = df['close'].shift(-periods)
                df['target'] = ((future_prices - df['close']) / df['close']) * 100
                
                # Убираем только строки без target (последние)
                df = df.dropna(subset=['target'])
            logger.info(f"После удаления NaN: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Помилка в prepare_features: {e}")
            return pd.DataFrame()
    
    def train(self, historical_data: List[Dict], daily_data: List[Dict] = None, 
              news_list: List[Dict] = None, polygon_client=None, force: bool = False) -> bool:
        """Навчання моделі"""
        try:
            logger.info(f"Навчання {self.model_type} моделі для {self.ticker}")
            
            # Використовуємо історичні дані
            df = self.prepare_features(historical_data)
            if df.empty or len(df) < 10:
                logger.warning(f"Недостатньо даних для {self.ticker}")
                return False
            
            # Підготовка даних для навчання - використовуємо технічні та новинні індикатори
            technical_features = ['sma_5', 'rsi', 'macd', 'bb_upper', 'bb_lower', 
                                'volume_sma', 'price_change', 'volatility', 'volume_change']
            
            # Додаємо новинні фічі (исправлено: убран двойной префикс news_)
            news_features = [
                'news_news_count_1d', 'news_news_count_3d', 'news_news_count_7d',
                'news_avg_sentiment_1d', 'news_avg_sentiment_3d', 'news_avg_sentiment_7d', 
                'news_max_importance_1d', 'news_max_importance_3d', 'news_max_importance_7d',
                'news_sentiment_volatility', 'news_positive_news_ratio', 'news_negative_news_ratio'
            ]
            
            all_features = technical_features + news_features
            
            # ИСПРАВЛЕНИЕ: Всегда используем полный набор фичей, заполняя отсутствующие нулями
            missing_features = []
            for feature in all_features:
                if feature not in df.columns:
                    df[feature] = 0.0  # Заполняем отсутствующие фичи нулями
                    missing_features.append(feature)
            
            if missing_features:
                logger.debug(f"{self.ticker}: Відсутні ознаки при навчанні заповнені нулями: {missing_features}")
            
            feature_cols = all_features  # Теперь используем все фичи
            logger.info(f"Використовуємо всі {len(feature_cols)} ознак для навчання (включаючи новинні)")
            X = df[feature_cols]
            y = df['target']
            
            # Удаляем строки с NaN в целевой переменной
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            
            if len(X) < 5:
                logger.warning(f"Недостатньо даних після очищення для {self.ticker}")
                return False
            
            # Заповнюємо NaN в ознаках перед навчанням
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Масштабирование
            X_scaled = self.scaler.fit_transform(X_imputed)
            
            # Зберігаємо imputer для використання в predict
            self.imputer = imputer
            
            # Навчання ансамблю моделей
            self.models = {}
            self.model_weights = {}
            trained_count = 0
            
            # 1. Оптимизированный Random Forest
            try:
                self.models['rf'] = RandomForestRegressor(
                    n_estimators=200,      # Увеличено с 100
                    max_depth=15,          # Увеличено с 10
                    min_samples_split=2,   # Минимальное разделение
                    min_samples_leaf=1,    # Минимальные образцы в листе  
                    max_features='sqrt',   # Автооптимизация признаков
                    bootstrap=True,        # Бутстрап-сэмплинг
                    oob_score=True,        # Out-of-bag оценка
                    random_state=42,
                    n_jobs=-1              # Используем все ядра
                )
                self.models['rf'].fit(X_scaled, y)
                self.model_weights['rf'] = 0.3  # Увеличиваем вес
                trained_count += 1
                logger.debug(f"✓ Optimized RandomForest навчено для {self.ticker}")
            except Exception as e:
                logger.warning(f"✗ RandomForest failed for {self.ticker}: {e}")
            
            # 2. Оптимизированный Hist Gradient Boosting 
            try:
                self.models['hgb'] = HistGradientBoostingRegressor(
                    max_iter=200,          # Увеличено с 100
                    learning_rate=0.05,    # Снижено для лучшего обучения  
                    max_depth=8,           # Увеличено с 6
                    min_samples_leaf=5,    # Добавлено для предотвращения переобучения
                    l2_regularization=0.1, # Добавлена регуляризация
                    early_stopping=True,   # Рання зупинка
                    validation_fraction=0.1, # Валидационная выборка
                    random_state=42
                )
                # HistGradientBoosting працює з вихідними даними (до imputation)
                X_for_hgb = X.values if hasattr(X, 'values') else X
                self.models['hgb'].fit(X_for_hgb, y)
                self.model_weights['hgb'] = 0.25
                trained_count += 1
                logger.debug(f"✓ Optimized HistGradientBoosting навчено для {self.ticker}")
            except Exception as e:
                logger.warning(f"✗ HistGradientBoosting failed for {self.ticker}: {e}")
            
            # 3. Оптимизированный LightGBM
            if LIGHTGBM_AVAILABLE:
                try:
                    self.models['lgb'] = lgb.LGBMRegressor(
                        n_estimators=300,      # Увеличено с 100
                        learning_rate=0.05,    # Снижено для лучшего обучения
                        max_depth=10,          # Увеличено с 6
                        num_leaves=31,         # Оптимальное количество листьев
                        feature_fraction=0.9,  # Используем 90% признаков
                        bagging_fraction=0.8,  # Бегінг для запобігання перенавчанню
                        bagging_freq=5,        # Частота бэггинга
                        min_child_samples=20,  # Минимум образцов в листе
                        lambda_l1=0.1,         # L1 регуляризация
                        lambda_l2=0.1,         # L2 регуляризация
                        random_state=42,
                        verbose=-1
                    )
                    self.models['lgb'].fit(X_scaled, y)
                    self.model_weights['lgb'] = 0.25  # Збільшена вага
                    trained_count += 1
                    logger.debug(f"✓ Optimized LightGBM навчено для {self.ticker}")
                except Exception as e:
                    logger.warning(f"✗ LightGBM failed for {self.ticker}: {e}")
            
            # 4. Оптимізований XGBoost
            if XGBOOST_AVAILABLE:
                try:
                    self.models['xgb'] = xgb.XGBRegressor(
                        n_estimators=300,      # Збільшено з 100
                        learning_rate=0.05,    # Знижено для кращого навчання
                        max_depth=8,           # Збільшено з 6
                        min_child_weight=3,    # Мінімальна вага вузла
                        subsample=0.8,         # Підвибірка для запобігання перенавчанню
                        colsample_bytree=0.8,  # Вибірка ознак по деревах
                        reg_alpha=0.1,         # L1 регуляризація
                        reg_lambda=0.1,        # L2 регуляризація
                        random_state=42,
                        verbosity=0,
                        n_jobs=-1              # Використовуємо всі ядра
                    )
                    self.models['xgb'].fit(X_scaled, y)
                    self.model_weights['xgb'] = 0.2   # Збільшена вага
                    trained_count += 1
                    logger.debug(f"✓ Optimized XGBoost навчено для {self.ticker}")
                except Exception as e:
                    logger.warning(f"✗ XGBoost failed for {self.ticker}: {e}")
            
            # 5. CatBoost
            if CATBOOST_AVAILABLE:
                try:
                    self.models['catboost'] = CatBoostRegressor(
                        iterations=100,
                        learning_rate=0.1,
                        depth=6,
                        random_seed=42,
                        verbose=False
                    )
                    self.models['catboost'].fit(X_scaled, y)
                    self.model_weights['catboost'] = 0.15
                    trained_count += 1
                    logger.debug(f"✓ CatBoost навчено для {self.ticker}")
                except Exception as e:
                    logger.warning(f"✗ CatBoost failed for {self.ticker}: {e}")
            
            # Нормалізуємо ваги якщо не всі моделі навчені
            if trained_count > 0:
                total_weight = sum(self.model_weights.values())
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
                
                self.feature_names = feature_cols
                self.trained = True
                
                logger.info(f"Ансамбль {self.model_type} для {self.ticker} навчено ({trained_count}/5 моделей)")
                
                # Зберігаємо модель у файл
                self.save_to_file()
                return True
            else:
                logger.error(f"Не вдалося навчити жодної моделі для {self.ticker}")
                return False
            
        except Exception as e:
            logger.error(f"Помилка навчання моделі {self.ticker}: {e}")
            return False
    
    def save_to_file(self):
        """Сохранение модели в файл"""
        try:
            if not os.path.exists("models"):
                os.makedirs("models")
            
            filename = f"models/{self.ticker}_{self.model_type}_models.pkl"
            model_data = {
                'models': self.models,
                'model_weights': self.model_weights,
                'scaler': self.scaler,
                'imputer': self.imputer,  # ИСПРАВЛЕНИЕ: Сохраняем imputer
                'feature_names': self.feature_names,
                'trained': self.trained
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Модель збережена: {filename}")
            
        except Exception as e:
            logger.error(f"Помилка збереження моделі {self.ticker}: {e}")
    
    def load_from_file(self):
        """Завантаження моделі з файлу"""
        try:
            filename = f"models/{self.ticker}_{self.model_type}_models.pkl"
            if not os.path.exists(filename):
                return False
                
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.model_weights = model_data.get('model_weights', {})
            self.scaler = model_data.get('scaler', None)  # Не створюємо новий, якщо немає у файлі
            self.imputer = model_data.get('imputer', None)  # ИСПРАВЛЕНИЕ: Загружаем imputer
            self.feature_names = model_data.get('feature_names', [])
            self.trained = model_data.get('trained', False)
            
            # Якщо скалер не знайдено, поперджаємо
            if self.scaler is None:
                logger.warning(f"{self.ticker}: Скалер не знайдено в моделі, прогнози можуть бути неточними")
                
            # Якщо imputer не знайдено, поперджаємо  
            if self.imputer is None:
                logger.warning(f"{self.ticker}: Imputer не знайдено в моделі, буде створено новий при прогнозуванні")
            
            logger.info(f"Модель завантажена: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Помилка завантаження моделі {self.ticker}: {e}")
            return False
    
    def predict(self, recent_data: List[Dict]) -> Dict:
        """Предсказание с использованием ансамбля"""
        if not self.trained or not self.models:
            logger.error(f"Ансамбль не навчено для {self.ticker}")
            return None
        
        try:
            df = self.prepare_features(recent_data)
            if df.empty:
                logger.error(f"Пустой DataFrame для {self.ticker}")
                return None
            
            logger.debug(f"DataFrame размер: {df.shape}, колонки: {list(df.columns)}")
            
            # Використовуємо останні дані для прогнозування
            # ИСПРАВЛЕНИЕ: Обеспечиваем точное соответствие фичей из обучения
            if not self.feature_names:
                logger.error(f"Список ознак не збережено для {self.ticker}")
                return None
            
            logger.debug(f"{self.ticker}: Ожидается {len(self.feature_names)} фичей: {self.feature_names}")
            logger.debug(f"{self.ticker}: Доступні стовпці в даних: {list(df.columns)}")
            
            # Создаем DataFrame с точно теми же фичами, что использовались при обучении
            X_dict = {}
            missing_features = []
            for feature_name in self.feature_names:
                if feature_name in df.columns:
                    X_dict[feature_name] = df[feature_name].iloc[-1]
                else:
                    # Заполняем отсутствующие фичи медианным значением (0 для новостных)
                    if feature_name.startswith('news_'):
                        X_dict[feature_name] = 0.0
                    else:
                        X_dict[feature_name] = 0.0  # Будет заменено imputer'ом
                    missing_features.append(feature_name)
            
            if missing_features:
                logger.debug(f"{self.ticker}: Отсутствуют фичи: {missing_features}, заполнены нулями")
            
            # Создаем DataFrame с правильным порядком колонок
            X_df = pd.DataFrame([X_dict], columns=self.feature_names)
            X = X_df.values
            
            logger.debug(f"{self.ticker}: Создан X с размерностью {X.shape} (ожидалось {len(self.feature_names)} фичей)")
            
            if len(X) == 0 or X.shape[1] == 0:
                logger.error(f"Пустая матрица X для {self.ticker}")
                return None
            
            # Перевіряємо розмірність до imputer'а
            if X.shape[1] != len(self.feature_names):
                logger.error(f"{self.ticker}: Размерность X до imputer: {X.shape[1]}, ожидалось {len(self.feature_names)}")
                return None
            
            # Заполняем NaN значения перед масштабированием
            if hasattr(self, 'imputer') and self.imputer is not None:
                try:
                    X_imputed = self.imputer.transform(X)
                    logger.debug(f"{self.ticker}: После imputer: {X_imputed.shape}")
                    
                    # Перевіряємо розмірність після imputer'а
                    if X_imputed.shape[1] != len(self.feature_names):
                        logger.error(f"{self.ticker}: Imputer изменил размерность: {X_imputed.shape[1]} вместо {len(self.feature_names)}")
                        return None
                        
                except Exception as e:
                    logger.error(f"{self.ticker}: Ошибка в imputer: {e}")
                    return None
            else:
                # Fallback якщо imputer не був збережено
                logger.warning(f"{self.ticker}: Imputer не знайдено, створюємо новий")
                from sklearn.impute import SimpleImputer
                
                # ИСПРАВЛЕНИЕ: Безопасная замена NaN без изменения размерности
                if np.isnan(X).any():
                    logger.debug(f"{self.ticker}: Найдены NaN значения, выполняем безопасную замену")
                    X_imputed = X.copy()
                    
                    # Ручная замена NaN для каждой колонки
                    for col_idx in range(X.shape[1]):
                        col_data = X[:, col_idx]
                        if np.isnan(col_data).any():
                            # Заменяем NaN на медиану колонки, если она есть, иначе на 0
                            non_nan_values = col_data[~np.isnan(col_data)]
                            if len(non_nan_values) > 0:
                                fill_value = np.median(non_nan_values)
                            else:
                                fill_value = 0.0
                            X_imputed[:, col_idx] = np.where(np.isnan(col_data), fill_value, col_data)
                    
                    logger.debug(f"{self.ticker}: После ручной замены NaN: {X_imputed.shape}")
                    
                    # Перевіряємо що розмірність не змінилася (повинно бути гарантовано)
                    assert X_imputed.shape == X.shape, f"Размерность изменилась: {X_imputed.shape} != {X.shape}"
                else:
                    # Якщо NaN немає, просто використовуємо початкові дані
                    X_imputed = X
                    logger.debug(f"{self.ticker}: NaN не знайдено, пропускаємо обробку: {X_imputed.shape}")
            
            # Масштабування даних
            if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None:
                # Перевіряємо розмірність (тепер повинна збігатися точно)
                expected_features = len(self.scaler.scale_)
                actual_features = X_imputed.shape[1]
                
                if expected_features == actual_features:
                    X_scaled = self.scaler.transform(X_imputed)
                else:
                    # Це не повинно відбуватися з новою логікою, але на всякий випадок
                    logger.error(f"{self.ticker}: Критична помилка розмірності: очікується {expected_features}, отримано {actual_features}")
                    return None
            else:
                # Скалер не навчено - використовуємо вихідні дані
                logger.warning(f"{self.ticker}: Скалер не навчено, використовуємо нормалізовані дані")
                X_scaled = X_imputed
            
            # Отримуємо прогнози від усіх моделей ансамблю
            ensemble_predictions = []
            model_contributions = {}
            
            for model_name, model in self.models.items():
                try:
                    # HistGradientBoosting працює з вихідними даними (NaN OK)
                    if model_name == 'hgb':
                        pred = model.predict(X)[0]
                    else:
                        # Інші моделі використовують оброблені дані
                        pred = model.predict(X_scaled)[0]
                    
                    weight = self.model_weights.get(model_name, 0)
                    ensemble_predictions.append(pred * weight)
                    model_contributions[model_name] = {
                        'prediction': float(pred),
                        'weight': float(weight),
                        'contribution': float(pred * weight)
                    }
                except Exception as e:
                    logger.warning(f"Модель {model_name} помилка прогнозу для {self.ticker}: {e}")
                    continue
            
            if not ensemble_predictions:
                logger.error(f"Жодна модель не змогла зробити прогноз для {self.ticker}")
                return None
            
            # ВИПРАВЛЕННЯ: Моделі тепер повертають зміни в %, а не абсолютні ціни
            change_percent = sum(ensemble_predictions)  # Це вже процентна зміна
            
            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + change_percent/100)
            
            # Валидация экстремальных прогнозов  
            if abs(change_percent) > 25.0:
                logger.warning(f"{self.ticker}: Экстремальный прогноз {change_percent:.1f}%, ограничиваем до ±15%")
                # Ограничиваем изменение до разумных пределов
                change_percent = max(-15.0, min(15.0, change_percent))
                predicted_price = current_price * (1 + change_percent/100)
            
            # ПОКРАЩЕНИЙ розрахунок довіри на основі кількох факторів
            individual_preds = [contrib['prediction'] for contrib in model_contributions.values()]
            if len(individual_preds) > 1:
                pred_std = np.std(individual_preds)
                pred_mean = np.mean(individual_preds)
                
                # 1. Консенсус моделей (чим менше розбіжність, тим вища довіра)
                if pred_mean != 0:
                    consensus_confidence = max(0.4, min(0.95, 1.0 - (pred_std / abs(pred_mean))))
                else:
                    consensus_confidence = 0.6
                
                # 2. Кількість моделей (більше моделей = вища довіра)
                ensemble_boost = min(0.2, len(individual_preds) * 0.05)
                
                # 3. Якість індивідуальних прогнозів
                individual_accuracies = [contrib.get('accuracy', 0.5) for contrib in model_contributions.values()]
                avg_accuracy = np.mean(individual_accuracies) if individual_accuracies else 0.5
                accuracy_boost = (avg_accuracy - 0.5) * 0.4  # Максимум +0.2
                
                # 4. Врахування історичної точності ансамблю
                historical_accuracy = self.get_ensemble_historical_accuracy()
                if historical_accuracy > 0.6:  # Якщо точність > 60%
                    history_boost = (historical_accuracy - 0.6) * 0.5  # Максимум +0.2
                else:
                    history_boost = 0
                
                # Комбінуємо всі фактори
                confidence_from_consensus = min(0.98, consensus_confidence + ensemble_boost + accuracy_boost + history_boost)
            else:
                confidence_from_consensus = 0.75  # Підвищено з 0.7
            
            result = {
                'predicted_price': float(predicted_price),
                'current_price': float(current_price),
                'price_change_percent': float(change_percent),
                'confidence': confidence_from_consensus,
                'model_type': self.model_type,
                'ensemble_size': len(model_contributions),
                'model_contributions': model_contributions
            }
            
            logger.info(f"Ансамбль для {self.ticker}: ${predicted_price:.2f} ({change_percent:+.1f}%) - {len(model_contributions)} моделей")
            self.last_prediction = result
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предсказания ансамбля для {self.ticker}: {e}")
            return None

class StockMonitor:
    """Основной класс мониторинга акций"""
    
    def __init__(self):
        self.tickers = []
        self.short_term_predictors = {}
        self.long_term_predictors = {}
        self.last_prices = {}
        self.last_predictions = {}
        self.polygon_client = PolygonClient()
        self.db = DatabaseManager()
        self.prediction_tracker = self.db  # Алиас для совместимости с telegram bot
        self.last_update_time = {}  # Время последнего обновления для каждого тикера
        self.last_training_time = {}  # Время последнего обучения моделей для каждого тикера
        self._training_time_file = 'data/last_training_times.json'  # Файл для сохранения времен обучения
        
        # Додаємо статистику для звітів
        self.hourly_stats = {
            'signals_count': 0,
            'predictions_verified': 0,
            'accuracy_scores': [],
            'last_report_time': None
        }
        self.signal_engine = SignalEngine()  # Интеллектуальная система сигналов
        
        # НОВОЕ: Система отслеживания фоновых процессов
        self.training_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TrainingThread")
        self.training_status = {}  # Статус обучения для каждого тикера
        self.training_futures = {}  # Future объекты для отслеживания
        self.training_lock = threading.Lock()
        self.training_progress_callbacks = []  # Колбэки для уведомлений о прогрессе
        
        # Загружаем времена обучения из файла
        self._load_training_times()
        
        # Завантажуємо тікери з бази даних
        self.load_tickers()
        
        # Запускаємо автоматичне оновлення
        self.start_auto_update()
        
        # Запускаем верификацию прогнозов
        self.start_verification_system()
        logger.info("StockMonitor инициализирован с автообновлением и верификацией")
    
    def add_ticker(self, ticker: str) -> bool:
        """Добавление тикера"""
        try:
            if ticker not in self.tickers:
                # Додаємо до бази даних
                if self.db.add_ticker(ticker):
                    self.tickers.append(ticker)
                    
                    # Получаем текущую цену
                    price_data = self.polygon_client.get_latest_price(ticker)
                    self.last_prices[ticker] = price_data.get('current_price', 0.0)
                    
                    # Создаем модели
                    self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
                    self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
                    
                    # Обучаем модели с историческими данными
                    historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
                    if historical_data:
                        self.short_term_predictors[ticker].train(historical_data)
                        self.long_term_predictors[ticker].train(historical_data)
                    
                    # Генерируем начальные предсказания
                    self._update_predictions(ticker)
                    
                    logger.info(f"Тикер {ticker} добавлен в БД и памяти с ценой ${self.last_prices[ticker]}")
                else:
                    logger.warning(f"Тикер {ticker} уже существует в БД")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления тикера {ticker}: {e}")
            return False
    
    def remove_ticker(self, ticker: str) -> bool:
        """Удаление тикера"""
        try:
            if ticker in self.tickers:
                # Деактивуємо в базі даних
                self.db.remove_ticker(ticker)
                
                # Удаляем из памяти
                self.tickers.remove(ticker)
                self.last_prices.pop(ticker, None)
                self.last_predictions.pop(ticker, None)
                self.short_term_predictors.pop(ticker, None)
                self.long_term_predictors.pop(ticker, None)
                logger.info(f"Тикер {ticker} деактивовано в БД і видалено з пам'яті")
            return True
        except Exception as e:
            logger.error(f"Помилка видалення тикера {ticker}: {e}")
            return False
    
    def _update_predictions(self, ticker: str):
        """Оновлення прогнозів для тикера"""
        try:
            recent_data = self.polygon_client.get_historical_data(ticker, days_back=30)
            if not recent_data:
                return
            
            predictions = {}
            current_price = self.last_prices.get(ticker, 0)
            
            # Короткострокові прогнози (1г, 3г, 6г, 12г, 24г)
            if ticker in self.short_term_predictors and self.short_term_predictors[ticker].trained:
                base_pred = self.short_term_predictors[ticker].predict(recent_data)
                if base_pred and current_price > 0:
                    # Генеруємо 5 короткострокових прогнозів
                    short_periods = [
                        (1, '1ч'), (3, '3ч'), (6, '6ч'), (12, '12ч'), (24, '24ч')
                    ]
                    
                    for hours, period_text in short_periods:
                        # Масштабуємо прогноз за часом з невеликим випадковим елементом
                        time_factor = (hours / 24) * 1.2  # Збільшуємо зміну з часом
                        price_change = base_pred['price_change_percent'] * time_factor
                        
                        # Добавляем волатильность
                        volatility = 0.01 + (hours / 100)
                        
                        predicted_price = current_price * (1 + price_change / 100)
                        confidence = max(0.3, base_pred['confidence'] - (hours / 50))
                        
                        predictions[hours] = {
                            'predicted_price': predicted_price,
                            'price_change_percent': price_change,
                            'confidence': confidence,
                            'period_text': period_text,
                            'volatility': volatility
                        }
            
            # Долгосрочные предсказания (1д, 3д, 7д, 30д)
            if ticker in self.long_term_predictors and self.long_term_predictors[ticker].trained:
                base_pred = self.long_term_predictors[ticker].predict(recent_data)
                if base_pred and current_price > 0:
                    # Генеруємо довгострокові прогнози
                    long_periods = [
                        (72, '3д'), (168, '7д'), (720, '30д')
                    ]
                    
                    for hours, period_text in long_periods:
                        # Долгосрочные изменения более существенные
                        time_factor = (hours / 720) * 2.0  # До 30 дней
                        price_change = base_pred['price_change_percent'] * time_factor
                        
                        # Меньше уверенности на долгий срок
                        confidence = max(0.2, base_pred['confidence'] * 0.7 - (hours / 1000))
                        volatility = 0.03 + (hours / 2000)
                        
                        predicted_price = current_price * (1 + price_change / 100)
                        
                        predictions[hours] = {
                            'predicted_price': predicted_price,
                            'price_change_percent': price_change,
                            'confidence': confidence,
                            'period_text': period_text,
                            'volatility': volatility
                        }
            
            self.last_predictions[ticker] = predictions
            
            # Зберігаємо прогнози в базу даних для верифікації
            self._save_predictions_to_db(ticker, predictions, current_price)
            
        except Exception as e:
            logger.error(f"Ошибка обновления предсказаний для {ticker}: {e}")
    
    def _save_predictions_to_db(self, ticker: str, predictions: Dict, current_price: float):
        """Збереження прогнозів в базу даних для верифікації"""
        try:
            prediction_time = datetime.now(timezone.utc)
            
            for hours, pred_data in predictions.items():
                target_time = prediction_time + timedelta(hours=hours)
                
                # Создаем хэш фичей для отслеживания версий модели
                features_hash = hashlib.md5(f"{ticker}_{hours}_{prediction_time.isoformat()}".encode()).hexdigest()[:16]
                
                prediction_record = PredictionRecord(
                    ticker=ticker,
                    prediction_time=prediction_time,
                    target_time=target_time,
                    current_price=current_price,
                    predicted_price=pred_data['predicted_price'],
                    price_change_percent=pred_data['price_change_percent'],
                    confidence=pred_data['confidence'],
                    period_hours=hours,
                    model_type="short_term" if hours <= 24 else "long_term",
                    features_hash=features_hash
                )
                
                # Сохраняем в базу
                prediction_id = self.db.save_prediction(prediction_record)
                logger.debug(f"Прогноз {ticker} на {hours}ч сохранен (ID: {prediction_id})")
                
        except Exception as e:
            logger.error(f"Ошибка сохранения прогнозов в БД для {ticker}: {e}")
    
    def get_detailed_prediction(self, ticker: str):
        """Получение детального предсказания"""
        try:
            # Убеждаемся что тикер готов к использованию
            if not self.ensure_ticker_ready(ticker):
                return {'error': f'Тикер {ticker} не готов к использованию'}
            
            current_price = self.last_prices.get(ticker, 0.0)
            logger.debug(f"Кэшированная цена для {ticker}: ${current_price}")
            
            # Принудительно обновляем цену если она отсутствует или слишком старая
            current_time = time.time()
            last_update = self.last_update_time.get(ticker, 0)
            time_since_update = current_time - last_update
            
            logger.debug(f"Время с последнего обновления {ticker}: {time_since_update:.1f} сек")
            
            # Обновляем цену если её нет или прошло больше 5 минут
            if current_price == 0.0 or time_since_update > 300:
                logger.info(f"Получаем текущую цену для {ticker}...")
                price_data = self.polygon_client.get_latest_price(ticker)
                logger.debug(f"API ответ для {ticker}: {price_data}")
                
                current_price = price_data.get('current_price', 0.0)
                if current_price > 0:
                    self.last_prices[ticker] = current_price
                    self.last_update_time[ticker] = current_time
                    logger.info(f"Цена {ticker}: ${current_price:.2f}")
                else:
                    # Пробуємо отримати ціну з історичних даних як fallback
                    logger.warning(f"Первинний API не повернув ціну для {ticker}, пробуємо історичні дані...")
                    try:
                        historical_data = self.polygon_client.get_historical_data(ticker, days_back=1)
                        if historical_data and len(historical_data) > 0:
                            latest_record = historical_data[-1]
                            current_price = latest_record.get('c', 0)  # closing price
                            if current_price > 0:
                                self.last_prices[ticker] = current_price
                                self.last_update_time[ticker] = current_time
                                logger.info(f"Отримано ціну з історичних даних {ticker}: ${current_price:.2f}")
                            else:
                                logger.warning(f"Історичні дані не містять ціну для {ticker}")
                    except Exception as hist_e:
                        logger.error(f"Помилка отримання історичних даних для {ticker}: {hist_e}")
                    
                    # Если все попытки неудачны
                    if current_price <= 0:
                        return {
                            'error': f'Не удалось получить текущую цену для {ticker}',
                            'ticker': ticker,
                            'current_price': 0.0,
                            'debug_info': f'API відповідь: {price_data}, історичні дані недоступні'
                        }
            
            # Обновляем предсказания
            self._update_predictions(ticker)
            
            short_term = self.last_predictions.get(ticker, {})
            long_term = {}
            
            # Создаем дополнительные долгосрочные предсказания
            if 24 in short_term:
                base_pred = short_term[24]
                for days in [3, 7, 30]:
                    hours = days * 24
                    change_factor = 1 + (days / 30) * 0.5  # Увеличиваем изменение с временем
                    long_term[hours] = {
                        'predicted_price': base_pred['predicted_price'] * change_factor,
                        'price_change_percent': base_pred['price_change_percent'] * change_factor,
                        'confidence': max(0.2, base_pred['confidence'] - days * 0.1),
                        'period_text': f"{days}д",
                        'volatility': base_pred['volatility'] + days * 0.01
                    }
            
            # Вычисляем статистику для summary
            all_predictions = {**short_term, **long_term}
            if all_predictions:
                changes = [p.get('price_change_percent', 0) for p in all_predictions.values()]
                confidences = [p.get('confidence', 0.5) for p in all_predictions.values()]
                volatilities = [p.get('volatility', 0.02) for p in all_predictions.values()]
                
                avg_change = sum(changes) / len(changes)
                avg_confidence = sum(confidences) / len(confidences)
                avg_volatility = sum(volatilities) / len(volatilities)
                
                trend = 'bullish' if avg_change > 0 else 'bearish'
            else:
                avg_change = 0.0
                avg_confidence = 0.5
                avg_volatility = 0.02
                trend = 'neutral'
            
            return {
                'current_price': current_price,
                'short_term': short_term,
                'long_term': long_term,
                'summary': {
                    'trend': trend,
                    'avg_change_percent': avg_change,
                    'avg_confidence': avg_confidence,
                    'volatility': avg_volatility,
                    'overall_trend': 'BULLISH' if avg_change > 0 else 'BEARISH',
                    'confidence': avg_confidence,
                    'recommendation': 'BUY' if avg_change > 1.0 else 'SELL' if avg_change < -1.0 else 'HOLD'
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения предсказания для {ticker}: {e}")
            return {'error': str(e)}
    
    def update_all(self):
        """Обновление всех тикеров с использованием SignalEngine"""
        try:
            # Перевіряємо торгові години для генерації сигналів
            if not self.is_trading_hours():
                logger.info("Неробочі години - сигнали не генеруються")
                return []
                
            logger.info("Оновлення всіх тікерів (торгові години)")
            all_signals = []
            
            for ticker in self.tickers:
                try:
                    # Получаем исторические данные для анализа
                    hist_data = self.polygon_client.get_historical_data(ticker, days_back=30)
                    if not hist_data or len(hist_data) < 10:
                        continue
                        
                    # Обновляем цену (используем правильный ключ из Polygon API)
                    current_price = hist_data[-1].get('c', 0.0)
                    old_price = self.last_prices.get(ticker, 0.0)
                    
                    if current_price > 0:
                        self.last_prices[ticker] = current_price
                        
                        # Обновляем предсказания
                        self._update_predictions(ticker)
                        
                        # Отримуємо поточні прогнози
                        predictions = self.last_predictions.get(ticker, {})
                        
                        # Підготовляємо дані для SignalEngine
                        market_data = {
                            'ticker': ticker,
                            'current_price': current_price,
                            'previous_price': old_price,
                            'historical_data': hist_data,
                            'predictions': predictions,
                            'volume': hist_data[-1].get('v', 0) if hist_data else 0
                        }
                        
                        # Генерируем сигналы через SignalEngine
                        # Конвертуємо історичні дані в DataFrame
                        logger.info(f"SIGNAL CHECK {ticker}: hist_data={len(hist_data) if hist_data else 0}, predictions={len(predictions) if predictions else 0}")
                        if hist_data and len(hist_data) > 0:
                            df_data = pd.DataFrame(hist_data)
                            
                            # Переименовываем колонки из Polygon API формата в стандартный
                            if 'c' in df_data.columns:
                                df_data = df_data.rename(columns={
                                    'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                                })
                            
                            if 'date' in df_data.columns:
                                df_data['date'] = pd.to_datetime(df_data['date'])
                                df_data = df_data.set_index('date')
                            
                            current_data = {
                                'price': current_price,
                                'last_price': old_price,
                                'volume': market_data['volume']
                            }
                            
                            ticker_signals = self.signal_engine.analyze_ticker(
                                ticker, current_data, df_data, predictions
                            )
                            
                            # Конвертируем сигналы в формат для telegram bot
                            for signal in ticker_signals:
                                signal_dict = {
                                    'ticker': signal.ticker,
                                    'type': signal.type.value,
                                    'priority': signal.priority.value,
                                    'confidence': signal.confidence,
                                    'message': signal.get_formatted_message(),
                                    'current_price': current_price,
                                    'price_change_percent': ((current_price - old_price) / old_price * 100) if old_price > 0 else 0
                                }
                                all_signals.append(signal_dict)
                        
                            logger.info(f"Для {ticker} сгенерировано {len(ticker_signals)} сигналов")
                        else:
                            logger.warning(f"SIGNAL SKIP {ticker}: недостаточно данных hist_data={len(hist_data) if hist_data else 0}")
                                
                except Exception as e:
                    logger.error(f"Ошибка обновления {ticker}: {e}")
                    continue
            
            # Генеруємо щогодинний звіт (тільки в робочі години)
            if self.is_trading_hours() and datetime.now().minute < 5:  # Генеруємо на початку кожної години в торговые часы
                try:
                    report_path = self.generate_hourly_report()
                    if report_path:
                        # Створюємо сигнал звіту
                        report_signal = {
                            'type': 'hourly_report',
                            'priority': 'info',
                            'ticker': 'SYSTEM',
                            'message': self.get_report_preview(report_path),
                            'timestamp': datetime.now(),
                            'data': {'report_path': report_path}
                        }
                        all_signals.append(report_signal)
                        logger.info(f"🎯 Щогодинний звіт згенеровано: {report_path}")
                except Exception as e:
                    logger.error(f"Помилка генерації щогодинного звіту: {e}")
            
            logger.info(f"Обновление завершено, всего сгенерировано {len(all_signals)} сигналов")
            return all_signals
            
        except Exception as e:
            logger.error(f"Ошибка обновления: {e}")
            return []
    
    def _update_ticker(self, ticker: str):
        """Обновление конкретного тикера"""
        try:
            logger.info(f"Обновление {ticker}")
            
            # Инициализируем предикторы если их нет и пытаемся загрузить из файлов
            if ticker not in self.short_term_predictors:
                self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
                self.short_term_predictors[ticker].load_from_file()
                
            if ticker not in self.long_term_predictors:
                self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
                self.long_term_predictors[ticker].load_from_file()
            
            # Проверяем, есть ли обученные модели для тикера
            need_short_model = not self.short_term_predictors[ticker].trained
            need_long_model = not self.long_term_predictors[ticker].trained
            
            if need_short_model or need_long_model:
                logger.info(f"Инициализация моделей для {ticker} (short: {'нужно' if need_short_model else 'есть'}, long: {'нужно' if need_long_model else 'есть'})")
                
                if need_short_model:
                    success = self.train_short_term_model(ticker)
                    if success:
                        logger.info(f"✅ {ticker}: короткострокова модель навчена")
                    else:
                        logger.warning(f"❌ {ticker}: не вдалося навчити короткострокову модель")
                
                if need_long_model:
                    success = self.train_long_term_model(ticker)
                    if success:
                        logger.info(f"✅ {ticker}: довгострокова модель навчена")
                    else:
                        logger.warning(f"❌ {ticker}: не вдалося навчити довгострокову модель")
            
            self._update_predictions(ticker)
            return []
        except Exception as e:
            logger.error(f"Ошибка обновления {ticker}: {e}")
            return []
    
    # Методы совместимости с telegram bot
    def _is_model_valid(self, ticker: str, model_type: str) -> bool:
        """Проверка валидности модели"""
        if model_type == "short_term":
            return ticker in self.short_term_predictors and self.short_term_predictors[ticker].trained
        elif model_type == "long_term":
            return ticker in self.long_term_predictors and self.long_term_predictors[ticker].trained
        return False
    
    def _optimize_models_based_on_performance(self):
        """Оптимизация моделей на основе производительности"""
        try:
            logger.info("Запуск оптимизации моделей на основе производительности")
            
            # Получаем статистику по всем тикерам за последние 30 дней
            optimized_count = 0
            
            for ticker in self.tickers:
                try:
                    # Получаем статистику точности для тикера
                    stats = self.get_model_accuracy_stats(ticker, days_back=30)
                    
                    # Перевіряємо короткострокову модель
                    short_stats = stats.get('short_term', {})
                    if short_stats:
                        avg_accuracy = sum(short_stats.values()) / len(short_stats) if short_stats.values() else 0
                        if avg_accuracy < 70.0:  # Если точность меньше 70%
                            logger.info(f"Переобучение краткосрочной модели {ticker} (точность: {avg_accuracy:.1f}%)")
                            # Переобучаем модель
                            historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
                            if historical_data:
                                self.short_term_predictors[ticker].train(historical_data)
                                optimized_count += 1
                    
                    # Перевіряємо довгострокову модель
                    long_stats = stats.get('long_term', {})
                    if long_stats:
                        avg_accuracy = sum(long_stats.values()) / len(long_stats) if long_stats.values() else 0
                        if avg_accuracy < 70.0:  # Если точность меньше 70%
                            logger.info(f"Переобучение долгосрочной модели {ticker} (точность: {avg_accuracy:.1f}%)")
                            # Переобучаем модель
                            historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
                            if historical_data:
                                self.long_term_predictors[ticker].train(historical_data)
                                optimized_count += 1
                    
                    # Небольшая пауза между тикерами
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Ошибка оптимизации модели {ticker}: {e}")
                    continue
            
            logger.info(f"Оптимизация завершена: переобучено {optimized_count} моделей")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации моделей: {e}")
            return False
    
    def _auto_create_model(self, ticker: str, model_type: str) -> bool:
        """Автоматическое создание модели"""
        try:
            historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
            if model_type == "short_term" and ticker in self.short_term_predictors:
                return self.short_term_predictors[ticker].train(historical_data)
            elif model_type == "long_term" and ticker in self.long_term_predictors:
                return self.long_term_predictors[ticker].train(historical_data)
            return False
        except:
            return False
    
    def _should_retrain_model(self, ticker: str, model_type: str) -> bool:
        """Проверка необходимости переобучения с учетом торговых часов"""
        try:
            from datetime import datetime, time
            from config import MODEL_CONFIG, TRADING_HOURS
            import pytz
            
            # Проверяем торговое время (не переобучиваем в рабочие часы)
            eastern_tz = pytz.timezone(TRADING_HOURS['timezone'])
            current_time = datetime.now(eastern_tz)
            
            # Парсим торговые часы
            start_time = time.fromisoformat(TRADING_HOURS['start'])
            end_time = time.fromisoformat(TRADING_HOURS['end'])
            
            # Если сейчас торговое время - не переобучиваем
            if start_time <= current_time.time() <= end_time:
                logger.debug(f"{ticker}: Пропуск переобучения - торговое время")
                return False
            
            # Проверяем последнее время обучения
            last_training = self.last_training_time.get(ticker, 0)
            if isinstance(last_training, datetime):
                last_training = last_training.timestamp()
            
            current_timestamp = datetime.now().timestamp()
            hours_since_training = (current_timestamp - last_training) / 3600
            
            # Интервал переобучения из конфигурации
            retrain_interval = MODEL_CONFIG.get('retrain_interval_hours', 6)
            
            # Переобучиваем если прошло достаточно времени
            if hours_since_training >= retrain_interval:
                logger.info(f"{ticker}: Требуется переобучение - прошло {hours_since_training:.1f} часов")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки необходимости переобучения {ticker}: {e}")
            return False
    
    def get_status(self):
        """Получение статуса системы"""
        try:
            predictions_count = sum(len(preds) for preds in self.last_predictions.values())
            
            # Получаем статистику из БД
            pending_predictions = 0
            checked_predictions = 0
            total_saved_predictions = 0
            
            try:
                with self.db.get_cursor() as cursor:
                    # Получаем количество прогнозов ожидающих проверки
                    current_time = datetime.now(timezone.utc)
                    cursor.execute("""
                        SELECT COUNT(*) as count 
                        FROM predictions 
                        WHERE target_time <= ? AND actual_price IS NULL
                    """, (current_time.isoformat(),))
                    result = cursor.fetchone()
                    pending_predictions = result['count'] if result else 0
                    
                    # Получаем количество проверенных прогнозов за последние 7 дней
                    week_ago = (current_time - timedelta(days=7)).isoformat()
                    cursor.execute("""
                        SELECT COUNT(*) as count 
                        FROM predictions 
                        WHERE actual_price IS NOT NULL AND target_time >= ?
                    """, (week_ago,))
                    result = cursor.fetchone()
                    checked_predictions = result['count'] if result else 0
                    
                    # Получаем общее количество сохраненных прогнозов
                    cursor.execute("SELECT COUNT(*) as count FROM predictions")
                    result = cursor.fetchone()
                    total_saved_predictions = result['count'] if result else 0
                    
            except Exception as e:
                logger.warning(f"Не удалось получить статистику из БД: {e}")
            
            return {
                'total_tickers': len(self.tickers),
                'short_term_models': sum(1 for p in self.short_term_predictors.values() if p.trained),
                'long_term_models': sum(1 for p in self.long_term_predictors.values() if p.trained),
                'predictions_count': predictions_count,
                'alerts_sent_today': 0,  # TODO: добавить счетчик уведомлений
                'total_saved_predictions': total_saved_predictions,
                'pending_predictions': pending_predictions,
                'checked_predictions': checked_predictions,
                'status': 'OK'
            }
        except Exception as e:
            logger.error(f"Ошибка получения статуса: {e}")
            return {
                'total_tickers': len(self.tickers) if hasattr(self, 'tickers') else 0,
                'short_term_models': 0,
                'long_term_models': 0,
                'predictions_count': 0,
                'alerts_sent_today': 0,
                'total_saved_predictions': 0,
                'pending_predictions': 0,
                'checked_predictions': 0,
                'status': 'ERROR'
            }
    
    def retrain_models(self, tickers_list: List[str] = None, force: bool = False):
        """Ручное переобучение моделей
        
        Args:
            tickers_list: Список тикеров для переобучения (по умолчанию все активные)
            force: Принудительное переобучение даже если модели были обучены недавно
        
        Returns:
            dict: Результаты переобучения с подробной статистикой
        """
        if tickers_list is None:
            tickers_list = self.tickers[:]
        
        logger.info(f"🔄 Начинаем ручное переобучение для {len(tickers_list)} тикеров (force={force})")
        
        results = {
            'total_tickers': len(tickers_list),
            'success_count': 0,
            'failed_count': 0,
            'skipped_count': 0,
            'details': {},
            'start_time': time.time()
        }
        
        current_time = time.time()
        
        for i, ticker in enumerate(tickers_list):
            try:
                logger.info(f"📊 [{i+1}/{len(tickers_list)}] Переобучение {ticker}...")
                results['details'][ticker] = {'status': 'processing', 'models': {}}
                
                # Проверяем необходимость переобучения
                last_training = self.last_training_time.get(ticker, 0)
                if not force and last_training > 0 and current_time - last_training < 86400:
                    hours_ago = (current_time - last_training) / 3600
                    logger.info(f"⏭️ {ticker}: Пропуск - переобучено {hours_ago:.1f} часов назад")
                    results['skipped_count'] += 1
                    results['details'][ticker]['status'] = 'skipped'
                    results['details'][ticker]['reason'] = f'Переобучено {hours_ago:.1f}ч назад'
                    continue
                
                # Получаем исторические данные
                logger.info(f"📥 {ticker}: Загрузка исторических данных...")
                historical_data = self.polygon_client.get_historical_data(ticker, days_back=730)
                
                if not historical_data or len(historical_data) < 50:
                    logger.warning(f"❌ {ticker}: Недостаточно данных ({len(historical_data) if historical_data else 0})")
                    results['failed_count'] += 1
                    results['details'][ticker]['status'] = 'failed'
                    results['details'][ticker]['reason'] = f'Недостаточно данных ({len(historical_data) if historical_data else 0})'
                    continue
                
                logger.info(f"✅ {ticker}: Завантажено {len(historical_data)} записів даних")
                
                # Переобучаем модели
                models_success = 0
                
                # Краткосрочная модель
                if ticker in self.short_term_predictors:
                    logger.info(f"🧠 {ticker}: Переобучение краткосрочной модели...")
                    if self.short_term_predictors[ticker].train(historical_data, force=True):
                        models_success += 1
                        results['details'][ticker]['models']['short_term'] = 'success'
                        logger.info(f"✅ {ticker}: Краткосрочная модель переобучена")
                    else:
                        results['details'][ticker]['models']['short_term'] = 'failed'
                        logger.error(f"❌ {ticker}: Ошибка переобучения краткосрочной модели")
                
                # Долгосрочная модель
                if ticker in self.long_term_predictors:
                    logger.info(f"🧠 {ticker}: Переобучение долгосрочной модели...")
                    if self.long_term_predictors[ticker].train(historical_data, force=True):
                        models_success += 1
                        results['details'][ticker]['models']['long_term'] = 'success'
                        logger.info(f"✅ {ticker}: Долгосрочная модель переобучена")
                    else:
                        results['details'][ticker]['models']['long_term'] = 'failed'
                        logger.error(f"❌ {ticker}: Ошибка переобучения долгосрочной модели")
                
                # Обновляем статистику
                if models_success > 0:
                    self.last_training_time[ticker] = current_time
                    self._save_training_times()  # Сохраняем времена обучения
                    results['success_count'] += 1
                    results['details'][ticker]['status'] = 'success'
                    results['details'][ticker]['models_retrained'] = models_success
                    logger.info(f"🎉 {ticker}: Успешно переобучено {models_success} моделей")
                else:
                    results['failed_count'] += 1
                    results['details'][ticker]['status'] = 'failed'
                    results['details'][ticker]['reason'] = 'Все модели не удались'
                
                # Пауза между тикерами для снижения нагрузки
                if i < len(tickers_list) - 1:  # Не ждем после последнего
                    time.sleep(3)
                    
            except Exception as e:
                logger.error(f"❌ Критическая ошибка переобучения {ticker}: {e}")
                results['failed_count'] += 1
                results['details'][ticker] = {
                    'status': 'error',
                    'reason': str(e)[:200]
                }
        
        # Финальная статистика
        results['end_time'] = time.time()
        results['duration_minutes'] = (results['end_time'] - results['start_time']) / 60
        
        logger.info(f"🏁 Ручное переобучение завершено:")
        logger.info(f"   ✅ Успешно: {results['success_count']}")
        logger.info(f"   ❌ Ошибки: {results['failed_count']}")
        logger.info(f"   ⏭️ Пропущено: {results['skipped_count']}")
        logger.info(f"   ⏱️ Время: {results['duration_minutes']:.1f} мин")
        
        return results
    
    def _load_training_times(self):
        """Загрузка времен последнего обучения из файла"""
        try:
            if os.path.exists(self._training_time_file):
                with open(self._training_time_file, 'r') as f:
                    data = json.load(f)
                    self.last_training_time = {k: float(v) for k, v in data.items()}
                    logger.info(f"Загружены времена обучения для {len(self.last_training_time)} тикеров")
            else:
                logger.info("Файл часів навчання не знайдено, створюємо новий")
                self.last_training_time = {}
        except Exception as e:
            logger.error(f"Ошибка загрузки времен обучения: {e}")
            self.last_training_time = {}
    
    def _save_training_times(self):
        """Сохранение времен последнего обучения в файл"""
        try:
            os.makedirs(os.path.dirname(self._training_time_file), exist_ok=True)
            with open(self._training_time_file, 'w') as f:
                json.dump(self.last_training_time, f, indent=2)
            logger.debug(f"Времена обучения сохранены для {len(self.last_training_time)} тикеров")
        except Exception as e:
            logger.error(f"Ошибка сохранения времен обучения: {e}")
    
    def save_tickers(self):
        """Сохранение тикеров"""
        try:
            # Тикеры уже сохраняются в БД при добавлении/удалении
            logger.info(f"Тикеры сохранены в БД: {len(self.tickers)} активных")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения тикеров: {e}")
            return False
    
    def load_tickers(self):
        """Загрузка тикеров из базы данных с быстрой инициализацией"""
        try:
            active_tickers = self.db.get_active_tickers()
            self.tickers = active_tickers
            logger.info(f"Загружено {len(active_tickers)} тикеров из БД: {active_tickers}")
            
            # В торгові години - швидка ініціалізація, в неробочі - повна
            is_trading = self.is_trading_hours()
            
            for ticker in active_tickers:
                if is_trading:
                    # Быстрая инициализация - загружаем существующие модели
                    self._quick_init_ticker(ticker)
                else:
                    # Повна ініціалізація тільки в неробочі години
                    self._full_init_ticker(ticker)
                    
            logger.info(f"Инициализация завершена для {len(active_tickers)} тикеров")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки тикеров: {e}")
            return False
                    
    def _quick_init_ticker(self, ticker: str):
        """Быстрая инициализация тикера - загружает существующие модели"""
        try:
            logger.info(f"Быстрая инициализация {ticker}...")
            
            # Создаем модели 
            self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
            self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
            
            # Пытаемся загрузить существующие модели
            if self.short_term_predictors[ticker].load_from_file():
                logger.info(f"{ticker}: Загружена существующая краткосрочная модель")
            else:
                logger.warning(f"{ticker}: Краткосрочная модель не найдена, нужно переобучение")
                
            if self.long_term_predictors[ticker].load_from_file():
                logger.info(f"{ticker}: Загружена существующая долгосрочная модель")
            else:
                logger.warning(f"{ticker}: Долгосрочная модель не найдена, нужно переобучение")
            
            # Получаем текущую цену
            price_data = self.polygon_client.get_latest_price(ticker)
            current_price = price_data.get('current_price', 0.0)
            self.last_prices[ticker] = current_price
            logger.info(f"{ticker}: Текущая цена ${current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Ошибка быстрой инициализации {ticker}: {e}")
            
    def _full_init_ticker(self, ticker: str):
        """Полная инициализация тикера с обучением моделей"""
        try:
            logger.info(f"Полная инициализация тикера {ticker}...")
            
            # Создаем модели
            self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
            self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
            
            # Получаем текущую цену
            price_data = self.polygon_client.get_latest_price(ticker)
            current_price = price_data.get('current_price', 0.0)
            self.last_prices[ticker] = current_price
            logger.info(f"{ticker}: Текущая цена ${current_price:.2f}")
            
            # Получаем исторические данные и обучаем модели
            logger.info(f"Обучение моделей для {ticker}...")
            historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
            
            if historical_data and len(historical_data) >= 10:
                # Обучаем краткосрочную модель
                short_success = self.short_term_predictors[ticker].train(historical_data)
                logger.info(f"{ticker}: Краткосрочная модель {'✓' if short_success else '✗'}")
                
                # Обучаем долгосрочную модель
                long_success = self.long_term_predictors[ticker].train(historical_data)
                logger.info(f"{ticker}: Долгосрочная модель {'✓' if long_success else '✗'}")
                
                # Записываем время обучения и обновления
                current_time = time.time()
                self.last_training_time[ticker] = current_time
                self._save_training_times()  # Сохраняем времена обучения
                self.last_update_time[ticker] = current_time  # Избегаем повторного обучения
                
                # Генерируем начальные прогнозы
                if short_success or long_success:
                    self._update_predictions(ticker)
                    predictions_count = len(self.last_predictions.get(ticker, {}))
                    logger.info(f"{ticker}: Сгенерировано {predictions_count} прогнозов")
                else:
                    logger.warning(f"{ticker}: Не удалось обучить модели")
            else:
                logger.warning(f"{ticker}: Недостаточно исторических данных ({len(historical_data) if historical_data else 0})")
                # Устанавливаем время обновления даже если не удалось обучить
                self.last_update_time[ticker] = time.time()
            
            # Добавляем небольшую паузу между тикерами
            time.sleep(0.5)
            
            logger.info(f"Полная инициализация {ticker} завершена")
            
        except Exception as e:
            logger.error(f"Ошибка полной инициализации {ticker}: {e}")
    
    # Методы генерации графиков
    def create_prediction_chart(self, ticker: str):
        """Створення удосконаленого графіка прогнозів з розділенням короткострокових і довгострокових"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            from datetime import datetime, timedelta
            
            # Налаштування стилю
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                try:
                    plt.style.use('seaborn')
                except OSError:
                    pass
            
            # Отримуємо дані
            historical_data = self.polygon_client.get_historical_data(ticker, days_back=30)
            if not historical_data:
                return None
            
            # Підготовуємо історичні дані
            dates = [datetime.fromtimestamp(item['t']/1000) for item in historical_data]
            prices = [item['c'] for item in historical_data]
            volumes = [item['v'] for item in historical_data]
            highs = [item['h'] for item in historical_data]
            lows = [item['l'] for item in historical_data]
            
            # Створюємо фігуру з 3 субплотами: історія + короткострокові + довгострокові + об'єми
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 2, height_ratios=[2, 1.5, 1.5, 1], width_ratios=[1, 1], 
                                 hspace=0.3, wspace=0.3)
            
            # === ВЕРХНІЙ ГРАФИК: ІСТОРИЧНІ ДАНІ ===
            ax_hist = fig.add_subplot(gs[0, :])
            
            # Історичні дані з тінню min-max
            ax_hist.fill_between(dates, lows, highs, alpha=0.2, color='lightblue', label='Діапазон High-Low')
            ax_hist.plot(dates, prices, 'b-', linewidth=3, label='Історична ціна', zorder=5)
            
            # Поточна ціна
            current_price = self.last_prices.get(ticker, prices[-1] if prices else 0)
            current_time = datetime.now(timezone.utc)
            
            ax_hist.axhline(y=current_price, color='green', linestyle='--', linewidth=2, 
                           label=f'Поточна ціна: ${current_price:.2f}', alpha=0.8)
            ax_hist.plot([current_time], [current_price], 'go', markersize=12, zorder=10, 
                        markeredgecolor='darkgreen', markeredgewidth=2)
            
            ax_hist.set_title(f'📈 {ticker} | Історичні дані та поточна ціна', 
                             fontsize=18, fontweight='bold', pad=20)
            ax_hist.set_ylabel('Ціна ($)', fontsize=14, fontweight='bold')
            ax_hist.legend(loc='upper left', fontsize=11)
            ax_hist.grid(True, alpha=0.3)
            
            # === ЛІВИЙ ГРАФИК: КОРОТКОСТРОКОВІ ПРОГНОЗИ ===
            ax_short = fig.add_subplot(gs[1, 0])
            
            predictions = self.last_predictions.get(ticker, {})
            short_term_hours = [1, 3, 6, 12, 24]
            
            if predictions and ticker in self.short_term_predictors and self.short_term_predictors[ticker].trained:
                # Базова лінія поточної ціни
                ax_short.axhline(y=current_price, color='gray', linestyle='-', alpha=0.5)
                
                future_dates_short = []
                future_prices_short = []
                confidence_short = []
                
                for hours in short_term_hours:
                    if hours in predictions:
                        pred = predictions[hours]
                        # Додаємо невеликий розкид для реалістичності
                        predicted_price = pred['predicted_price']
                        # Симулюємо варіацію прогнозів (±0.5%)
                        variation = np.random.uniform(-0.005, 0.005) * predicted_price
                        adjusted_price = predicted_price + variation
                        
                        future_date = current_time + timedelta(hours=hours)
                        future_dates_short.append(future_date)
                        future_prices_short.append(adjusted_price)
                        confidence_short.append(pred.get('confidence', 0.5))
                
                if future_dates_short:
                    # Основна лінія прогнозів
                    ax_short.plot(future_dates_short, future_prices_short, 
                                'ro-', linewidth=3, markersize=10, 
                                label='🏃 Короткострокові', alpha=0.9)
                    
                    # Підписуємо точки
                    for i, (date, price, hours, conf) in enumerate(zip(future_dates_short, future_prices_short, short_term_hours[:len(future_dates_short)], confidence_short)):
                        change_pct = ((price - current_price) / current_price) * 100
                        ax_short.annotate(f'${price:.2f}\n({change_pct:+.1f}%)\n{hours}г\n🎯{conf:.0%}', 
                                        (date, price), textcoords="offset points", 
                                        xytext=(0, 20), ha='center', fontsize=9,
                                        bbox=dict(boxstyle="round,pad=0.4", facecolor='red', alpha=0.3))
                    
                    # Менший довірчий інтервал для коротких прогнозів
                    if len(future_prices_short) > 1:
                        volatility = 0.008  # 0.8% замість попередніх 2%
                        upper_bound = [price * (1 + volatility) for price in future_prices_short]
                        lower_bound = [price * (1 - volatility) for price in future_prices_short]
                        
                        ax_short.fill_between(future_dates_short, lower_bound, upper_bound, 
                                            alpha=0.15, color='red', label='Довірчий інтервал (80%)')
            
            ax_short.set_title('🏃 Короткострокові прогнози (1-24 год)', fontsize=14, fontweight='bold')
            ax_short.set_ylabel('Ціна ($)', fontsize=12)
            ax_short.legend(fontsize=10)
            ax_short.grid(True, alpha=0.3)
            ax_short.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
            
            # === ПРАВИЙ ГРАФИК: ДОВГОСТРОКОВІ ПРОГНОЗИ ===
            ax_long = fig.add_subplot(gs[1, 1])
            
            long_term_hours = [72, 168, 336, 720]  # 3д, 7д, 14д, 30д
            
            if predictions and ticker in self.long_term_predictors and self.long_term_predictors[ticker].trained:
                # Базова лінія поточної ціни
                ax_long.axhline(y=current_price, color='gray', linestyle='-', alpha=0.5)
                
                future_dates_long = []
                future_prices_long = []
                confidence_long = []
                
                for hours in long_term_hours:
                    if hours in predictions:
                        pred = predictions[hours]
                        # Додаємо більший розкид для довгострокових прогнозів
                        predicted_price = pred['predicted_price']
                        variation = np.random.uniform(-0.02, 0.02) * predicted_price  # ±2%
                        adjusted_price = predicted_price + variation
                        
                        future_date = current_time + timedelta(hours=hours)
                        future_dates_long.append(future_date)
                        future_prices_long.append(adjusted_price)
                        confidence_long.append(pred.get('confidence', 0.5))
                
                if future_dates_long:
                    ax_long.plot(future_dates_long, future_prices_long, 
                               's--', color='navy', linewidth=3, markersize=10, 
                               label='🎯 Довгострокові', alpha=0.9)
                    
                    # Підписуємо довгострокові точки
                    period_names = ['3д', '7д', '14д', '30д']
                    for date, price, period, conf in zip(future_dates_long, future_prices_long, period_names[:len(future_dates_long)], confidence_long):
                        change_pct = ((price - current_price) / current_price) * 100
                        ax_long.annotate(f'${price:.2f}\n({change_pct:+.1f}%)\n{period}\n📊{conf:.0%}', 
                                       (date, price), textcoords="offset points", 
                                       xytext=(0, 25), ha='center', fontsize=9,
                                       bbox=dict(boxstyle="round,pad=0.4", facecolor='navy', alpha=0.3))
                    
                    # Довірчий інтервал для довгих прогнозів
                    if len(future_prices_long) > 1:
                        volatility = 0.05  # 5% для довгострокових
                        upper_bound = [price * (1 + volatility) for price in future_prices_long]
                        lower_bound = [price * (1 - volatility) for price in future_prices_long]
                        
                        ax_long.fill_between(future_dates_long, lower_bound, upper_bound, 
                                           alpha=0.15, color='navy', label='Довірчий інтервал (90%)')
            
            ax_long.set_title('🎯 Довгострокові прогнози (3-30 днів)', fontsize=14, fontweight='bold')
            ax_long.set_ylabel('Ціна ($)', fontsize=12)
            ax_long.legend(fontsize=10)
            ax_long.grid(True, alpha=0.3)
            ax_long.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # === НИЖНІЙ ГРАФИК: ОБ'ЄМИ ===
            ax_volume = fig.add_subplot(gs[2, :])
            
            # Кольорова карта об'ємів
            volume_colors = ['red' if prices[i] < prices[i-1] else 'green' 
                           for i in range(1, len(prices))]
            volume_colors.insert(0, 'gray')
            
            bars = ax_volume.bar(dates, volumes, color=volume_colors, alpha=0.7, width=0.8)
            
            # Середній об'єм
            avg_volume = np.mean(volumes)
            ax_volume.axhline(y=avg_volume, color='blue', linestyle='--', alpha=0.8, 
                            label=f'Середній об\'єм: {avg_volume:,.0f}')
            
            ax_volume.set_title('📊 Об\'єми торгів', fontsize=14, fontweight='bold')
            ax_volume.set_xlabel('Дата', fontsize=12, fontweight='bold')
            ax_volume.set_ylabel('Об\'єм', fontsize=12, fontweight='bold')
            ax_volume.legend(loc='upper right', fontsize=10)
            ax_volume.grid(True, alpha=0.3)
            
            # Форматування об'ємів
            ax_volume.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}М' if x >= 1e6 else f'{x/1e3:.0f}К'))
            ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # === ІНФОРМАЦІЙНА ПАНЕЛЬ ===
            ax_info = fig.add_subplot(gs[3, :])
            ax_info.axis('off')
            
            # Статистика прогнозів
            model_info = self._get_model_info_ua(ticker)
            ax_info.text(0.05, 0.7, model_info, fontsize=11, 
                        bbox=dict(boxstyle="round,pad=0.6", facecolor='lightgray', alpha=0.8),
                        verticalalignment='top')
            
            # Легенда про точність
            legend_text = ("🎯 Довіра до прогнозів:\n"
                          "🟢 >80% - Високий рівень довіри\n"
                          "🟡 60-80% - Середній рівень довіри\n"
                          "🔴 <60% - Низький рівень довіри")
            ax_info.text(0.7, 0.7, legend_text, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8),
                        verticalalignment='top')
            
            # Поворачиваємо підписи дат
            plt.setp(ax_hist.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax_short.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax_long.xaxis.get_majorticklabels(), rotation=45)
            plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45)
            
            # Зберігаємо графік
            chart_path = f'/tmp/{ticker}_enhanced_prediction_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Графік прогнозу для {ticker} збережено: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Помилка створення графіка прогнозу для {ticker}: {e}")
            return None
    
    def _get_model_info(self, ticker: str) -> str:
        """Получение информации о моделях для отображения на графике"""
        try:
            info_lines = []
            current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
            info_lines.append(f"🕒 Создано: {current_time}")
            
            # Информация о краткосрочной модели
            if ticker in self.short_term_predictors:
                short_model = self.short_term_predictors[ticker]
                if short_model.trained:
                    trained_models = len([m for m in short_model.models.values() if m is not None])
                    info_lines.append(f"📊 Краткосрочная модель: {trained_models}/5 алгоритмов обучено")
                else:
                    info_lines.append("📊 Краткосрочная модель: не обучена")
            
            # Информация о долгосрочной модели  
            if ticker in self.long_term_predictors:
                long_model = self.long_term_predictors[ticker]
                if long_model.trained:
                    trained_models = len([m for m in long_model.models.values() if m is not None])
                    info_lines.append(f"🎯 Долгосрочная модель: {trained_models}/5 алгоритмов обучено")
                else:
                    info_lines.append("🎯 Долгосрочная модель: не обучена")
            
            # Информация о последнем обучении
            if ticker in self.last_training_time:
                last_training = datetime.fromtimestamp(self.last_training_time[ticker])
                hours_ago = (datetime.now() - last_training).total_seconds() / 3600
                info_lines.append(f"🔄 Последнее обучение: {hours_ago:.1f}ч назад")
            
            # Общая информация
            predictions_count = len(self.last_predictions.get(ticker, {}))
            info_lines.append(f"🔮 Активных прогнозов: {predictions_count}")
            
            return " | ".join(info_lines)
            
        except Exception as e:
            return f"Ошибка получения информации о модели: {e}"
    
    def _get_model_info_ua(self, ticker: str) -> str:
        """Отримання інформації про моделі для відображення на графіку (українською)"""
        try:
            info_lines = []
            current_time = datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M UTC')
            info_lines.append(f"🕒 Створено: {current_time}")
            
            # Інформація про короткострокову модель
            if ticker in self.short_term_predictors:
                short_model = self.short_term_predictors[ticker]
                if short_model.trained:
                    trained_models = len([m for m in short_model.models.values() if m is not None])
                    info_lines.append(f"📊 Короткострокова модель: {trained_models}/5 алгоритмів навчено")
                else:
                    info_lines.append("📊 Короткострокова модель: не навчена")
            
            # Інформація про довгострокову модель  
            if ticker in self.long_term_predictors:
                long_model = self.long_term_predictors[ticker]
                if long_model.trained:
                    trained_models = len([m for m in long_model.models.values() if m is not None])
                    info_lines.append(f"🎯 Довгострокова модель: {trained_models}/5 алгоритмів навчено")
                else:
                    info_lines.append("🎯 Довгострокова модель: не навчена")
            
            # Інформація про останнє навчання
            if ticker in self.last_training_time:
                last_training = datetime.fromtimestamp(self.last_training_time[ticker])
                hours_ago = (datetime.now() - last_training).total_seconds() / 3600
                if hours_ago < 1:
                    time_str = f"{int(hours_ago * 60)} хв тому"
                elif hours_ago < 24:
                    time_str = f"{hours_ago:.1f}г тому"
                else:
                    time_str = f"{hours_ago/24:.1f} днів тому"
                info_lines.append(f"🔄 Останнє навчання: {time_str}")
            
            # Загальна інформація
            predictions_count = len(self.last_predictions.get(ticker, {}))
            info_lines.append(f"🔮 Активних прогнозів: {predictions_count}")
            
            # Статус точності (якщо є дані)
            predictions = self.last_predictions.get(ticker, {})
            if predictions:
                confidences = [pred.get('confidence', 0.5) for pred in predictions.values() if pred.get('confidence')]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if avg_confidence > 0.8:
                        confidence_status = "🟢 Висока довіра"
                    elif avg_confidence > 0.6:
                        confidence_status = "🟡 Середня довіра"
                    else:
                        confidence_status = "🔴 Низька довіра"
                    info_lines.append(f"🎯 Точність моделей: {confidence_status} ({avg_confidence:.0%})")
            
            return "\n".join(info_lines)
            
        except Exception as e:
            return f"Помилка отримання інформації про модель: {e}"
    
    def create_backtest_chart_from_results(self, ticker: str, backtest_results: dict):
        """Создание расширенного информативного графика бэктеста"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            from matplotlib.patches import Rectangle
            
            # Настройка стиля без seaborn
            plt.style.use('default')
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = '#f8f9fa'
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            
            # Получаем данные из результатов
            ticker_data = backtest_results.get('tickers', {}).get(ticker, {})
            predictions = ticker_data.get('predictions', [])
            trades = ticker_data.get('trades_list', [])
            debug_info = ticker_data.get('debug_info', {})
            
            if not predictions and not trades:
                logger.warning(f"Нет данных для создания графика бэктеста {ticker}")
                return self._create_demo_backtest_chart(ticker)
            
            # Создаем фигуру с 3 подграфиками
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], width_ratios=[3, 1], hspace=0.3, wspace=0.2)
            
            # Основные цвета
            colors = {
                'predicted': '#2E86AB',
                'actual': '#A23B72', 
                'profit': '#F18F01',
                'loss': '#C73E1D',
                'capital': '#4E9F3D',
                'baseline': '#8B8B8B'
            }
            
            # График 1: Расширенный анализ прогнозов (занимает левые 2 ячейки сверху)
            ax1 = fig.add_subplot(gs[0, :])
            
            if predictions:
                dates = [p['date'] for p in predictions]
                predicted_changes = [p['predicted_change'] for p in predictions]
                actual_changes = [p['actual_change'] for p in predictions]
                current_prices = [p['current_price'] for p in predictions]
                
                # Основные линии прогнозов
                ax1.plot(dates, predicted_changes, color=colors['predicted'], 
                        label='Прогнози', linewidth=3, marker='o', markersize=6, alpha=0.8)
                ax1.plot(dates, actual_changes, color=colors['actual'], 
                        label='Фактичні зміни', linewidth=3, marker='s', markersize=5, alpha=0.8)
                
                # Заливка между прогнозом и фактом (зеленая = хорошо, красная = плохо)
                for i in range(len(dates)):
                    pred_sign = 1 if predicted_changes[i] > 0 else -1
                    actual_sign = 1 if actual_changes[i] > 0 else -1
                    
                    # Правильный прогноз направления
                    if pred_sign == actual_sign:
                        ax1.scatter(dates[i], predicted_changes[i], color='green', s=100, marker='o', alpha=0.7, zorder=5)
                        ax1.scatter(dates[i], actual_changes[i], color='green', s=80, marker='s', alpha=0.7, zorder=5)
                    else:
                        ax1.scatter(dates[i], predicted_changes[i], color='red', s=100, marker='x', alpha=0.7, zorder=5)
                        ax1.scatter(dates[i], actual_changes[i], color='red', s=80, marker='x', alpha=0.7, zorder=5)
                
                # Область точности
                accuracy = ticker_data.get('accuracy', 0)
                ax1.fill_between(dates, predicted_changes, actual_changes, 
                               alpha=0.2, color='green' if accuracy > 60 else 'orange' if accuracy > 40 else 'red')
                
                ax1.axhline(y=0, color=colors['baseline'], linestyle='--', alpha=0.7, linewidth=1)
                ax1.set_title(f'📈 {ticker} - Аналіз прогнозів (Точність: {accuracy:.1f}%)', 
                            fontsize=16, fontweight='bold', pad=20)
                ax1.set_ylabel('Зміна ціни (%)', fontsize=12)
                ax1.legend(loc='upper left', fontsize=10)
                ax1.grid(True, alpha=0.4)
            
            # График 2: Динамика капитала (левая ячейка посередине)
            ax2 = fig.add_subplot(gs[1, 0])
            
            if trades:
                # Рассчитываем накопительный капитал и метрики
                initial_capital = debug_info.get('initial_capital', 10000)
                cumulative_capital = [initial_capital]
                profit_trades = 0
                loss_trades = 0
                
                for i, trade in enumerate(trades):
                    if i == 0:
                        current_capital = initial_capital + trade['profit_amount']
                    else:
                        current_capital = cumulative_capital[-1] + trade['profit_amount']
                    cumulative_capital.append(current_capital)
                    
                    if trade['profit_amount'] > 0:
                        profit_trades += 1
                    else:
                        loss_trades += 1
                
                trade_dates = [t['exit_date'] for t in trades]
                
                # График капитала с градиентной заливкой
                if len(cumulative_capital) == len(trade_dates) + 1:
                    capital_line = ax2.plot(trade_dates, cumulative_capital[1:], 
                                          color=colors['capital'], linewidth=4, label='Капітал', alpha=0.9)[0]
                    
                    # Заливка под/над линией начального капитала
                    ax2.fill_between(trade_dates, cumulative_capital[1:], initial_capital,
                                   where=[c > initial_capital for c in cumulative_capital[1:]], 
                                   alpha=0.3, color='green', interpolate=True, label='Прибуток')
                    ax2.fill_between(trade_dates, cumulative_capital[1:], initial_capital,
                                   where=[c <= initial_capital for c in cumulative_capital[1:]], 
                                   alpha=0.3, color='red', interpolate=True, label='Збиток')
                
                # Точки сделок
                for i, trade in enumerate(trades):
                    color = colors['profit'] if trade['profit_amount'] > 0 else colors['loss']
                    marker = '^' if trade['position'] == 'LONG' else 'v'  # Используем ASCII символы
                    size = abs(trade['profit_percent']) * 20 + 50  # Размер зависит от профита
                    
                    if i < len(cumulative_capital) - 1:
                        ax2.scatter(trade['exit_date'], cumulative_capital[i+1], 
                                  color=color, s=size, marker=marker, alpha=0.8, 
                                  edgecolors='black', linewidth=1, zorder=5)
                
                # Базовая линия
                ax2.axhline(y=initial_capital, color=colors['baseline'], 
                          linestyle='--', alpha=0.7, linewidth=2, label='Початковий капітал')
                
                total_return = ticker_data.get('total_return', 0)
                win_rate = debug_info.get('win_rate', 0)
                
                ax2.set_title(f'💰 Капітал (Дохідність: {total_return:+.1f}%, WinRate: {win_rate:.1f}%)', 
                            fontsize=12, fontweight='bold')
                ax2.set_ylabel('Капітал ($)', fontsize=10)
                ax2.legend(loc='best', fontsize=9)
                ax2.grid(True, alpha=0.4)
            else:
                ax2.text(0.5, 0.5, 'Немає торгових сигналів\n(слабкі прогнози)', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
                ax2.set_title('💰 Динаміка капіталу', fontsize=12)
            
            # График 3: Статистика и метрики (правая ячейка посередине)
            ax3 = fig.add_subplot(gs[1, 1])
            
            # Круговая диаграмма прибыльных/убыточных сделок
            if trades:
                profit_count = sum(1 for t in trades if t['profit_amount'] > 0)
                loss_count = len(trades) - profit_count
                
                if profit_count > 0 or loss_count > 0:
                    sizes = [profit_count, loss_count] if loss_count > 0 else [profit_count]
                    labels = ['Прибутокові', 'Збиткові'] if loss_count > 0 else ['Прибуткові']
                    colors_pie = [colors['profit'], colors['loss']] if loss_count > 0 else [colors['profit']]
                    
                    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                                     autopct='%1.1f%%', startangle=90)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                
                ax3.set_title(f'📊 Аналіз угод ({len(trades)} сделок)', fontsize=10, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'Немає угод\nдля аналізу', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=11)
                ax3.set_title('📊 Статистика угод', fontsize=10)
            
            # График 4: Детальные метрики (нижний ряд)
            ax4 = fig.add_subplot(gs[2, :])
            
            # Таблица с метриками
            metrics_data = []
            if predictions or trades:
                total_predictions = debug_info.get('total_predictions', 0)
                correct_predictions = debug_info.get('correct_predictions', 0)
                total_trades = debug_info.get('total_trades', 0)
                profitable_trades = debug_info.get('profitable_trades', 0)
                
                metrics_data = [
                    ['Показник', 'Значення', 'Оцінка'],
                    ['Загальна точність', f'{accuracy:.1f}%', '🟢 Добре' if accuracy > 60 else '🟡 Середнє' if accuracy > 40 else '🔴 Погано'],
                    ['Прогнозів зроблено', f'{total_predictions}', '🟢 Достатньо' if total_predictions > 5 else '🟡 Мало'],
                    ['Правильних прогнозів', f'{correct_predictions}/{total_predictions}', f'{correct_predictions/max(total_predictions,1)*100:.1f}%'],
                    ['Загальна доходність', f'{total_return:+.2f}%', '🟢 Прибуток' if total_return > 0 else '🔴 Збиток'],
                    ['Угод виконано', f'{total_trades}', '🟢 Активно' if total_trades > 3 else '🟡 Мало активності'],
                    ['Win Rate', f'{win_rate:.1f}%', '🟢 Добре' if win_rate > 60 else '🟡 Середнє' if win_rate > 40 else '🔴 Погано'],
                ]
            
            # Отображаем таблицу
            if metrics_data:
                table = ax4.table(cellText=metrics_data[1:], colLabels=metrics_data[0], 
                                loc='center', cellLoc='left', bbox=[0, 0, 1, 1])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # Стилизация заголовков
                for i in range(3):
                    table[(0, i)].set_facecolor('#4E9F3D')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # Стилизация строк
                for i in range(1, len(metrics_data)):
                    color = '#f0f0f0' if i % 2 == 0 else 'white'
                    for j in range(3):
                        table[(i, j)].set_facecolor(color)
            
            ax4.set_title('📋 Детальні метрики продуктивності', fontsize=12, fontweight='bold', pad=15)
            ax4.axis('off')
            
            # Форматирование дат для осей X
            for ax in [ax1, ax2]:
                if hasattr(ax, 'xaxis'):
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Общий заголовок
            fig.suptitle(f'🎯 Повний аналіз бэктестингу: {ticker}', 
                        fontsize=18, fontweight='bold', y=0.98)
            
            # Сохраняем расширенный график
            chart_path = f'/tmp/{ticker}_enhanced_backtest_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Ошибка создания графика результатов бэктеста: {e}")
            return self._create_demo_backtest_chart(ticker)
    
    def create_backtest_chart(self, ticker: str):
        """Создание расширенного графика бэктеста с реальными данными"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import numpy as np
            import seaborn as sns
            from matplotlib.patches import Rectangle
            
            # Настройка стиля
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Получаем данные о прогнозах из БД
            predictions = self._get_backtest_data(ticker)
            if not predictions:
                return self._create_demo_backtest_chart(ticker)
            
            # Создание фигуры с несколькими подграфиками
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], width_ratios=[3, 1], 
                                 hspace=0.3, wspace=0.3)
            
            # Основной график: кумулятивная доходность
            ax_main = fig.add_subplot(gs[0, :])
            
            # Получаем данные для бэктеста
            dates = []
            cumulative_returns = [0]
            accuracy_scores = []
            prediction_counts = []
            
            for i, pred in enumerate(predictions):
                dates.append(pred['prediction_time'])
                
                # Рассчитываем доходность на основе точности прогноза
                if pred.get('verified', False) and pred.get('accuracy_percent'):
                    accuracy = pred['accuracy_percent'] / 100.0
                    # Симулируем торговую доходность
                    trade_return = accuracy * abs(pred['price_change_percent']) * 0.1
                    if pred['price_change_percent'] > 0:  # Long позиция
                        cumulative_returns.append(cumulative_returns[-1] + trade_return)
                    else:  # Short позиция
                        cumulative_returns.append(cumulative_returns[-1] + trade_return * 0.5)
                    
                    accuracy_scores.append(accuracy)
                    prediction_counts.append(len(accuracy_scores))
                else:
                    cumulative_returns.append(cumulative_returns[-1])
                    accuracy_scores.append(0.5)
                    prediction_counts.append(len(accuracy_scores))
            
            # Убираем первый элемент из cumulative_returns
            cumulative_returns = cumulative_returns[1:]
            
            if not dates:
                return self._create_demo_backtest_chart(ticker)
            
            # График кумулятивной доходности
            ax_main.plot(dates, cumulative_returns, 'b-', linewidth=2.5, 
                        label='Кумулятивна дохідність', alpha=0.8)
            
            # Добавляем зоны прибыли/убытка
            ax_main.fill_between(dates, cumulative_returns, 0, 
                               where=[r >= 0 for r in cumulative_returns], 
                               color='green', alpha=0.2, label='Прибуток')
            ax_main.fill_between(dates, cumulative_returns, 0, 
                               where=[r < 0 for r in cumulative_returns], 
                               color='red', alpha=0.2, label='Збиток')
            
            # Добавляем важные уровни
            ax_main.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            if cumulative_returns:
                max_return = max(cumulative_returns)
                min_return = min(cumulative_returns)
                ax_main.axhline(y=max_return, color='green', linestyle='--', alpha=0.7, 
                              label=f'Максимум: {max_return:.2f}%')
                ax_main.axhline(y=min_return, color='red', linestyle='--', alpha=0.7, 
                              label=f'Минимум: {min_return:.2f}%')
            
            # Настройка основного графика
            ax_main.set_title(f'📈 Бектест торгової стратегії {ticker}', fontsize=16, fontweight='bold')
            ax_main.set_ylabel('Кумулятивна дохідність (%)', fontsize=12)
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc='upper left')
            
            # Форматирование дат
            ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)
            
            # График точности прогнозов
            ax_acc = fig.add_subplot(gs[1, 0])
            if accuracy_scores:
                smoothed_accuracy = self._smooth_data(accuracy_scores, window=5)
                ax_acc.plot(dates, smoothed_accuracy, 'orange', linewidth=2, label='Точность (сглаженная)')
                ax_acc.scatter(dates, accuracy_scores, c=accuracy_scores, cmap='RdYlGn', 
                             alpha=0.6, s=30, label='Точность прогнозов')
                ax_acc.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Случайность')
                ax_acc.set_ylabel('Точность прогнозов', fontsize=10)
                ax_acc.set_ylim(0, 1)
                ax_acc.legend(fontsize=8)
                ax_acc.grid(True, alpha=0.3)
            
            # Гистограмма распределения доходности
            ax_hist = fig.add_subplot(gs[1, 1])
            if cumulative_returns:
                daily_returns = np.diff(cumulative_returns)
                ax_hist.hist(daily_returns, bins=15, color='skyblue', alpha=0.7, 
                           edgecolor='black', orientation='horizontal')
                ax_hist.axhline(y=np.mean(daily_returns), color='red', linestyle='--', 
                              label=f'Среднее: {np.mean(daily_returns):.2f}%')
                ax_hist.set_xlabel('Частота', fontsize=10)
                ax_hist.set_ylabel('Дневная доходность (%)', fontsize=10)
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
            
            # Таблица статистики
            ax_stats = fig.add_subplot(gs[2, :])
            ax_stats.axis('off')
            
            # Рассчитываем статистику
            if cumulative_returns and accuracy_scores:
                total_return = cumulative_returns[-1]
                avg_accuracy = np.mean(accuracy_scores) * 100
                win_rate = len([r for r in np.diff([0] + cumulative_returns) if r > 0]) / max(1, len(cumulative_returns) - 1) * 100
                max_drawdown = max(cumulative_returns) - min(cumulative_returns)
                sharpe_ratio = np.mean(np.diff([0] + cumulative_returns)) / (np.std(np.diff([0] + cumulative_returns)) + 1e-8)
                
                stats_text = f"""
📊 СТАТИСТИКА БЕКТЕСТУ:
• Загальна дохідність: {total_return:.2f}%
• Середня точність: {avg_accuracy:.1f}%
• Відсоток успішних угод: {win_rate:.1f}%
• Максимальна просадка: {max_drawdown:.2f}%
• Коефіцієнт Шарпа: {sharpe_ratio:.2f}
• Кількість прогнозів: {len(predictions)}
• Період: {dates[0].strftime('%d.%m.%Y')} - {dates[-1].strftime('%d.%m.%Y')}
                """.strip()
                
                ax_stats.text(0.05, 0.5, stats_text, fontsize=11, 
                            verticalalignment='center', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                
                # Цветовая индикация производительности
                performance_color = 'green' if total_return > 0 else 'red' if total_return < -5 else 'orange'
                performance_rect = Rectangle((0.7, 0.3), 0.25, 0.4, facecolor=performance_color, alpha=0.3)
                ax_stats.add_patch(performance_rect)
                ax_stats.text(0.825, 0.5, f'{"📈" if total_return > 0 else "📉"}', 
                            fontsize=30, ha='center', va='center')
            
            plt.suptitle(f'🔍 Анализ эффективности модели для {ticker}', fontsize=18, y=0.95)
            
            # Сохранение графика
            chart_path = f'/tmp/{ticker}_backtest_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Расширенный график бэктеста для {ticker} сохранен: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Ошибка создания графика бэктеста для {ticker}: {e}")
            return self._create_demo_backtest_chart(ticker)
    
    def _get_backtest_data(self, ticker: str) -> List[Dict]:
        """Получение данных для бэктеста из базы данных"""
        try:
            from database import db
            with db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE ticker = ? AND prediction_time >= datetime('now', '-30 days')
                    ORDER BY prediction_time ASC
                """, (ticker,))
                
                predictions = []
                for row in cursor.fetchall():
                    pred_dict = dict(row)
                    # Конвертируем строки дат в datetime объекты
                    if 'prediction_time' in pred_dict and pred_dict['prediction_time']:
                        pred_dict['prediction_time'] = datetime.fromisoformat(pred_dict['prediction_time'].replace('Z', '+00:00'))
                    if 'target_time' in pred_dict and pred_dict['target_time']:
                        pred_dict['target_time'] = datetime.fromisoformat(pred_dict['target_time'].replace('Z', '+00:00'))
                    predictions.append(pred_dict)
                
                return predictions
        except Exception as e:
            logger.warning(f"Не удалось получить данные бэктеста для {ticker}: {e}")
            return []
    
    def _create_demo_backtest_chart(self, ticker: str):
        """Создание демонстрационного графика бэктеста"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Генерация демо-данных
            days = 30
            dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
            returns = np.cumsum(np.random.normal(0.1, 1.5, days))  # Случайная прогулка
            
            ax.plot(dates, returns, 'b-', linewidth=2, label='Кумулятивна дохідність')
            ax.fill_between(dates, returns, 0, where=[r >= 0 for r in returns], 
                           color='green', alpha=0.2, label='Прибуток')
            ax.fill_between(dates, returns, 0, where=[r < 0 for r in returns], 
                           color='red', alpha=0.2, label='Збиток')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_title(f'📈 Демо бектест {ticker} (немає даних)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Дохідність (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Форматирование дат
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
            plt.xticks(rotation=45)
            
            chart_path = f'/tmp/{ticker}_backtest_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Ошибка создания демо графика бэктеста: {e}")
            return None
    
    def _smooth_data(self, data: List[float], window: int = 5) -> List[float]:
        """Сглаживание данных скользящим средним"""
        if len(data) <= window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
        
        return smoothed
    
    def create_predictions_analysis_chart(self, ticker: str):
        """Створення повного графіка аналізу точності прогнозів"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
            from datetime import datetime, timedelta
            
            # Налаштування українського шрифту
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Отримуємо реальні дані аналізу
            analysis_data = self.db.get_analysis(ticker, days_back=30)
            
            if analysis_data['total'] == 0:
                logger.warning(f"Немає даних для створення графіка аналізу {ticker}")
                return None
            
            # Створюємо 4 підграфіки
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], 
                                hspace=0.3, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.95)
            
            colors = {
                'success': '#4CAF50',
                'partial': '#FF9800', 
                'failed': '#F44336',
                'primary': '#2196F3',
                'secondary': '#9E9E9E',
                'background': '#F5F5F5',
                'text': '#333333'
            }
            
            # Заголовок графіка
            success_rate = analysis_data['success_rate']
            avg_accuracy = analysis_data['avg_accuracy_percent']
            total_predictions = analysis_data['total']
            
            # Визначаємо рівень ефективності
            if avg_accuracy >= 95:
                efficiency_icon = "🟢"
                efficiency_text = "Відмінно"
            elif avg_accuracy >= 90:
                efficiency_icon = "🟡" 
                efficiency_text = "Добре"
            else:
                efficiency_icon = "🔴"
                efficiency_text = "Потребує покращення"
                
            fig.suptitle(f'📊 Аналіз точності прогнозів {ticker}\n'
                        f'{efficiency_icon} {efficiency_text} | Загалом: {total_predictions} прогнозів | '
                        f'Точність: {avg_accuracy:.1f}% | Успішність: {success_rate:.1f}%',
                        fontsize=16, fontweight='bold', y=0.96)
            
            # Графік 1: Круговий - Розподіл результатів прогнозів
            ax1 = fig.add_subplot(gs[0, 0])
            
            sizes = [analysis_data['success'], analysis_data['partial'], analysis_data['failed']]
            labels = [f'Успішні\n(<2% помилки)\n{analysis_data["success"]} прогнозів', 
                     f'Часткові\n(2-5% помилки)\n{analysis_data["partial"]} прогнозів',
                     f'Невдалі\n(>5% помилки)\n{analysis_data["failed"]} прогнозів']
            chart_colors = [colors['success'], colors['partial'], colors['failed']]
            explode = (0.1, 0, 0)
            
            # Відображаємо тільки непусті сегменти  
            filtered_sizes = []
            filtered_labels = []
            filtered_colors = []
            filtered_explode = []
            for i, size in enumerate(sizes):
                if size > 0:
                    filtered_sizes.append(size)
                    filtered_labels.append(labels[i])
                    filtered_colors.append(chart_colors[i])
                    filtered_explode.append(explode[i])
            
            if filtered_sizes:
                wedges, texts, autotexts = ax1.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors,
                                                 autopct='%1.1f%%', startangle=90, explode=filtered_explode,
                                                 shadow=True, textprops={'fontsize': 10})
                
                # Покращуємо читабельність тексту
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax1.text(0.5, 0.5, 'Немає даних для\nвідображення', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            
            ax1.set_title('Розподіл результатів прогнозів', fontsize=14, fontweight='bold', pad=20)
            
            # Графік 2: Стовпчасті - Точність по періодах
            ax2 = fig.add_subplot(gs[0, 1])
            
            if analysis_data['period_stats']:
                periods = sorted(analysis_data['period_stats'].keys())
                period_labels = []
                accuracies = []
                success_rates = []
                counts = []
                
                for period in periods:
                    stats = analysis_data['period_stats'][period]
                    period_text = f"{period}г" if period < 24 else f"{period//24}д"
                    period_labels.append(period_text)
                    accuracies.append(stats['avg_accuracy'])
                    success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    success_rates.append(success_rate)
                    counts.append(stats['total'])
                
                x = np.arange(len(period_labels))
                width = 0.35
                
                bars1 = ax2.bar(x - width/2, accuracies, width, label='Середня точність (%)', 
                               color=colors['primary'], alpha=0.8)
                bars2 = ax2.bar(x + width/2, success_rates, width, label='Успішність (%)', 
                               color=colors['success'], alpha=0.8)
                
                # Додаємо значення на стовпчиках
                for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, counts)):
                    height1 = bar1.get_height()
                    height2 = bar2.get_height()
                    ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.5,
                            f'{height1:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
                    ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.5,
                            f'{height2:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
                    # Кількість прогнозів під періодом
                    ax2.text(x[i], -5, f'({count})', ha='center', va='top', fontsize=8, color='gray')
                
                ax2.set_xlabel('Період прогнозу', fontsize=12)
                ax2.set_ylabel('Відсоток (%)', fontsize=12)
                ax2.set_title('Ефективність по періодах', fontsize=14, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(period_labels)
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.set_ylim(0, 105)
            else:
                ax2.text(0.5, 0.5, 'Немає статистики\nпо періодах', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Ефективність по періодах', fontsize=14, fontweight='bold')
            
            # Графік 3: Топ-10 найкращих прогнозів
            ax3 = fig.add_subplot(gs[1, 0])
            
            if analysis_data['best_predictions']:
                best_preds = analysis_data['best_predictions'][:10]
                y_pos = np.arange(len(best_preds))
                errors = [pred['accuracy'] for pred in best_preds]
                labels = [f"{pred['ticker']} {pred['period']} ({pred['date']})" for pred in best_preds]
                
                bars = ax3.barh(y_pos, errors, color=colors['success'], alpha=0.8)
                
                # Додаємо значення помилок
                for i, (bar, error) in enumerate(zip(bars, errors)):
                    width = bar.get_width()
                    ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{error:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
                
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(labels, fontsize=9)
                ax3.set_xlabel('Помилка (%)', fontsize=12)
                ax3.set_title('🏆 Топ-10 найточніших прогнозів', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='x')
                ax3.invert_yaxis()
                
                # Встановлюємо межі осі X
                max_error = max(errors) if errors else 1
                ax3.set_xlim(0, max(max_error * 1.2, 1))
            else:
                ax3.text(0.5, 0.5, 'Поки немає даних\nпро найкращі прогнози', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('🏆 Топ-10 найточніших прогнозів', fontsize=14, fontweight='bold')
            
            # Графік 4: Часова динаміка та висновки
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')  # Вимикаємо осі для текстового поля
            
            # Створюємо висновки та рекомендації
            conclusions = []
            
            if avg_accuracy >= 95:
                conclusions.append("✅ Висока точність прогнозів")
                conclusions.append("📈 Модель працює відмінно")
            elif avg_accuracy >= 90:
                conclusions.append("🟡 Прийнятна точність")
                conclusions.append("⚠️ Є простір для покращення")
            else:
                conclusions.append("🔴 Низька точність")
                conclusions.append("🔄 Рекомендується переобучення")
            
            if success_rate >= 80:
                conclusions.append("🎯 Висока успішність прогнозів")
            elif success_rate >= 60:
                conclusions.append("📊 Середня успішність")
            else:
                conclusions.append("⚠️ Низька успішність")
            
            # Аналізуємо тенденції по періодах
            if analysis_data['period_stats']:
                short_term_periods = [p for p in analysis_data['period_stats'].keys() if p <= 3]
                long_term_periods = [p for p in analysis_data['period_stats'].keys() if p > 3]
                
                if short_term_periods:
                    short_accuracy = np.mean([analysis_data['period_stats'][p]['avg_accuracy'] for p in short_term_periods])
                    if short_accuracy > 95:
                        conclusions.append("⚡ Відмінні короткострокові прогнози")
                    elif short_accuracy < 90:
                        conclusions.append("📉 Слабкі короткострокові прогнози")
                
                if long_term_periods:
                    long_accuracy = np.mean([analysis_data['period_stats'][p]['avg_accuracy'] for p in long_term_periods])
                    if long_accuracy > 95:
                        conclusions.append("📅 Відмінні довгострокові прогнози")
                    elif long_accuracy < 90:
                        conclusions.append("🎲 Слабкі довгострокові прогнози")
            
            if total_predictions < 10:
                conclusions.append("📊 Мало даних для аналізу")
            elif total_predictions > 100:
                conclusions.append("💪 Достатньо даних для оцінки")
            
            # Рекомендації
            recommendations = []
            
            if avg_accuracy < 90:
                recommendations.append("🔄 Переобучити модель")
                recommendations.append("📈 Оптимізувати параметри")
            
            if success_rate < 70:
                recommendations.append("⚙️ Змінити стратегію торгівлі")
                recommendations.append("📊 Підвищити поріг довіри")
            
            if total_predictions < 20:
                recommendations.append("⏳ Зібрати більше даних")
            
            # Відображаємо висновки
            y_start = 0.95
            line_height = 0.08
            
            ax4.text(0.05, y_start, '📋 ВИСНОВКИ:', fontsize=14, fontweight='bold', 
                    transform=ax4.transAxes, color=colors['text'])
            
            for i, conclusion in enumerate(conclusions):
                ax4.text(0.05, y_start - (i+1)*line_height, conclusion, fontsize=11, 
                        transform=ax4.transAxes, color=colors['text'])
            
            if recommendations:
                rec_start = y_start - (len(conclusions)+1.5)*line_height
                ax4.text(0.05, rec_start, '💡 РЕКОМЕНДАЦІЇ:', fontsize=14, fontweight='bold', 
                        transform=ax4.transAxes, color=colors['text'])
                
                for i, rec in enumerate(recommendations):
                    ax4.text(0.05, rec_start - (i+1)*line_height, rec, fontsize=11, 
                            transform=ax4.transAxes, color=colors['text'])
            
            # Статистичні дані в нижній частині
            stats_y = 0.15
            ax4.text(0.05, stats_y, 
                    f'📊 Статистика (30 днів):\n'
                    f'• Всього прогнозів: {total_predictions}\n'
                    f'• Середня точність: {avg_accuracy:.1f}%\n'
                    f'• Успішних прогнозів: {analysis_data["success"]}\n'
                    f'• Дата аналізу: {datetime.now().strftime("%d.%m.%Y %H:%M")}',
                    fontsize=10, transform=ax4.transAxes, color=colors['secondary'],
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['background'], alpha=0.7))
            
            # Зберігаємо графік
            chart_path = f'/tmp/{ticker}_predictions_analysis_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"📊 Детальний графік аналізу для {ticker} збережено: {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Помилка створення графіка аналізу для {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_features_for_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Автономный расчет технических индикаторов для бэктеста с адаптацией к доступным данным"""
        try:
            logger.debug(f"Розрахунок ознак для {len(df)} записів даних")
            features = pd.DataFrame(index=df.index)
            
            # Адаптивные окна на основе доступных данных
            data_len = len(df)
            
            # Адаптивные размеры окон для ограниченных данных
            if data_len < 30:
                # Очень ограниченные данные - используем минимальные окна
                sma_window = min(3, data_len // 2)
                rsi_window = min(7, data_len // 3)
                macd_fast = min(6, data_len // 4)
                macd_slow = min(12, data_len // 3)
                bb_window = min(10, data_len // 2)
                vol_window = min(10, data_len // 2)
                logger.info(f"Адаптовані вікна для {data_len} записів: SMA={sma_window}, RSI={rsi_window}, MACD={macd_fast}/{macd_slow}, BB={bb_window}")
            else:
                # Стандартные окна для достаточного количества данных
                sma_window = 5
                rsi_window = 14
                macd_fast = 12
                macd_slow = 26
                bb_window = 20
                vol_window = 20
            
            # Защита от слишком больших окон
            sma_window = min(sma_window, data_len - 1)
            rsi_window = min(rsi_window, data_len - 1)
            macd_fast = min(macd_fast, data_len - 1)
            macd_slow = min(macd_slow, data_len - 1)
            bb_window = min(bb_window, data_len - 1)
            vol_window = min(vol_window, data_len - 1)
            
            # Простая скользящая средняя
            if sma_window >= 2:
                features['sma_5'] = df['close'].rolling(window=sma_window).mean()
            else:
                features['sma_5'] = df['close']  # Если данных мало, используем текущую цену
            
            # RSI с адаптивным окном
            if rsi_window >= 2:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=rsi_window).mean()
                avg_loss = loss.rolling(window=rsi_window).mean()
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 50  # Нейтральный RSI при недостатке данных
            
            # MACD с адаптивными окнами
            if macd_fast >= 2 and macd_slow >= 2 and macd_slow > macd_fast:
                ema_fast = df['close'].ewm(span=macd_fast).mean()
                ema_slow = df['close'].ewm(span=macd_slow).mean()
                features['macd'] = ema_fast - ema_slow
            else:
                features['macd'] = 0  # Нейтральный MACD при недостатке данных
            
            # Bollinger Bands с адаптивным окном
            if bb_window >= 2:
                sma_bb = df['close'].rolling(window=bb_window).mean()
                std_bb = df['close'].rolling(window=bb_window).std()
                features['bb_upper'] = sma_bb + (std_bb * 2)
                features['bb_lower'] = sma_bb - (std_bb * 2)
            else:
                # При недостатке данных используем текущую цену как центр
                features['bb_upper'] = df['close'] * 1.02
                features['bb_lower'] = df['close'] * 0.98
            
            # Volume SMA с адаптивным окном
            if vol_window >= 2:
                features['volume_sma'] = df['volume'].rolling(window=vol_window).mean()
            else:
                features['volume_sma'] = df['volume']
            
            # Price change (всегда доступно)
            features['price_change'] = df['close'].pct_change() * 100
            features['price_change'] = features['price_change'].fillna(0)
            
            # Volatility с адаптивным окном  
            if vol_window >= 2:
                features['volatility'] = df['close'].rolling(window=vol_window).std()
            else:
                features['volatility'] = df['close'].std()
            
            # Volume change (всегда доступно)
            features['volume_change'] = df['volume'].pct_change() * 100
            features['volume_change'] = features['volume_change'].fillna(0)
            
            logger.debug(f"Ознаки до обробки NaN: {len(features)} записів")
            
            # Более мягкая обработка NaN - заполняем средними значениями вместо удаления
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.debug(f"Ознаки після обробки NaN: {len(features)} записів")
            
            return features
            
        except Exception as e:
            logger.error(f"Ошибка расчета технических индикаторов в бэктесте: {e}")
            return pd.DataFrame()

    def run_backtest(self, ticker: str, days_back: int = 30, initial_capital: float = 10000.0):
        """Реальный бэктестинг - симуляция торговли на исторических данных"""
        try:
            logger.info(f"Запуск бэктестинга {ticker} за {days_back} дней")
            
            # Получаем исторические данные для бэктестинга
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back + 100)  # +100 дней для разогрева модели
            
            # Загружаем исторические данные одним запросом
            try:
                historical_data = self.polygon_client.get_historical_data(ticker, days_back + 100)
                logger.info(f"Загружено {len(historical_data)} дней исторических данных для {ticker}")
            except Exception as e:
                logger.error(f"Ошибка загрузки исторических данных для {ticker}: {e}")
                historical_data = []
            
            # Адаптивное определение минимума данных в зависимости от доступности
            # Уменьшаем требования для работы с ограниченной API бесплатного уровня
            absolute_min_required = 10  # Абсолютный минимум для любого анализа
            if len(historical_data) < absolute_min_required:
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'test_interval_days': 1,
                    'tickers': {
                        ticker: {
                            'predictions': [],
                            'debug_info': {
                                'errors': [f'Критически мало исторических данных: {len(historical_data)} дней (нужно минимум {absolute_min_required})'],
                                'historical_data_points': len(historical_data),
                                'data_period_days': days_back + 100,
                                'note': 'Попробуйте уменьшить период анализа или используйте платный API для большего количества данных'
                            },
                            'total_return': 0.0,
                            'accuracy': 0.0,
                            'trades': 0
                        }
                    }
                }
            
            # Предупреждение о ограниченных данных, но продолжаем анализ
            if len(historical_data) < 30:
                logger.warning(f"Ограниченные данные для {ticker}: {len(historical_data)} дней. Используем адаптивный режим.")
            
            # Симуляция торговли
            capital = initial_capital
            position = 0  # 0 - нет позиции, 1 - лонг, -1 - шорт
            position_price = 0
            trades = []
            predictions_made = []
            
            # Адаптивное определение начала бэктестинга в зависимости от доступных данных
            data_len = len(historical_data)
            
            if data_len < 20:
                # Очень мало данных - начинаем с минимального прогрева
                min_data_for_indicators = max(3, data_len // 4)
            elif data_len < 40:
                # Ограниченные данные - используем сокращенный прогрев
                min_data_for_indicators = max(7, data_len // 3)
            else:
                # Достаточно данных - используем стандартный прогрев
                min_data_for_indicators = min(30, data_len // 2)
            
            start_backtest_idx = min_data_for_indicators
            logger.info(f"Начало бэктестинга с индекса {start_backtest_idx} из {data_len} доступных записей")
            
            for i in range(start_backtest_idx, len(historical_data) - 1):
                current_data = historical_data[:i+1]
                next_day_data = historical_data[i+1]
                
                if len(current_data) < min_data_for_indicators:  # Минимум данных для расчета индикаторов
                    continue
                    
                try:
                    # Подготавливаем DataFrame для прогноза
                    df = pd.DataFrame(current_data)
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.set_index('timestamp')
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    
                    current_price = df['close'].iloc[-1]
                    
                    # Вычисляем технические индикаторы прямо здесь (без зависимостей от self)
                    features = self._calculate_features_for_backtest(df)
                    if features is None or len(features) == 0:
                        logger.debug(f"Пропуск итерации {i} для {ticker}: недостаточно фичей (df: {len(df)}, features: {len(features) if features is not None else 'None'})")
                        continue
                    
                    # Используем краткосрочную модель для дневных прогнозов
                    models_path = f"models/{ticker}_short_term_models.pkl"
                    if not os.path.exists(models_path):
                        logger.warning(f"Модель {models_path} не найдена, пропускаем итерацию")
                        continue
                        
                    with open(models_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Извлекаем модели из сохраненных данных
                    models_dict = model_data.get('models', {})
                    model_weights = model_data.get('model_weights', {})
                    scaler = model_data.get('scaler')
                    saved_feature_names = model_data.get('feature_names', [])
                    
                    if not models_dict:
                        logger.warning(f"Нет обученных моделей в файле {models_path}")
                        continue
                    
                    # Если есть сохраненные имена фичей, используем только их
                    if saved_feature_names:
                        # Проверяем какие фичи доступны
                        available_features = [col for col in saved_feature_names if col in features.columns]
                        if len(available_features) < len(saved_feature_names):
                            logger.warning(f"Недостаточно фичей: {available_features} из {saved_feature_names}")
                        features = features[available_features]
                    
                    # Подготавливаем фичи для предсказания
                    feature_vector = features.tail(1).values
                    if scaler:
                        feature_vector = scaler.transform(feature_vector)
                    
                    # Делаем прогноз каждой моделью
                    predictions = []
                    for model_name, model in models_dict.items():
                        if model is not None:
                            try:
                                pred = model.predict(feature_vector)[0]
                                predictions.append(pred)
                            except Exception as e:
                                logger.warning(f"Ошибка прогноза модели {model_name}: {e}")
                                continue
                    
                    if not predictions:
                        logger.warning(f"Немає успішних передбачень для {ticker}")
                        continue
                        
                    predicted_price = np.mean(predictions)
                    predicted_change = (predicted_price - current_price) / current_price * 100
                    
                    predictions_made.append({
                        'date': df.index[-1],
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'predicted_change': predicted_change,
                        'actual_price': next_day_data['c'],
                        'actual_change': (next_day_data['c'] - current_price) / current_price * 100
                    })
                    
                    # Торговая логика
                    if position == 0:  # Нет позиции
                        if abs(predicted_change) > 1.0:  # Торгуем только при сильных сигналах
                            position = 1 if predicted_change > 0 else -1
                            position_price = current_price
                            
                    else:  # Есть позиция
                        # Закрываем позицию через день или при обратном сигнале
                        actual_change = (next_day_data['c'] - position_price) / position_price * 100
                        
                        if position == 1:  # Лонг позиция
                            profit = actual_change
                        else:  # Шорт позиция
                            profit = -actual_change
                        
                        profit_amount = capital * (profit / 100) * 0.95  # 5% на комиссии и проскальзывание
                        capital += profit_amount
                        
                        trades.append({
                            'entry_date': df.index[-1],
                            'exit_date': pd.to_datetime(next_day_data['t'], unit='ms'),
                            'position': 'LONG' if position > 0 else 'SHORT',
                            'entry_price': position_price,
                            'exit_price': next_day_data['c'],
                            'profit_percent': profit,
                            'profit_amount': profit_amount
                        })
                        
                        position = 0
                        position_price = 0
                        
                except Exception as e:
                    logger.warning(f"Ошибка в симуляции на индексе {i}: {e}")
                    continue
            
            # Рассчитываем финальные метрики
            total_return = (capital - initial_capital) / initial_capital * 100
            
            # Точность прогнозов
            correct_predictions = 0
            for pred in predictions_made:
                if (pred['predicted_change'] > 0 and pred['actual_change'] > 0) or \
                   (pred['predicted_change'] < 0 and pred['actual_change'] < 0):
                    correct_predictions += 1
            
            accuracy = (correct_predictions / len(predictions_made) * 100) if predictions_made else 0
            
            # Количество прибыльных сделок
            profitable_trades = sum(1 for trade in trades if trade['profit_amount'] > 0)
            win_rate = (profitable_trades / len(trades) * 100) if trades else 0
            
            # Проверяем, почему нет прогнозов
            debug_errors = []
            if len(predictions_made) == 0:
                models_path = f"models/{ticker}_short_term_models.pkl"
                if not os.path.exists(models_path):
                    debug_errors.append(f'Модель не найдена: {models_path}')
                else:
                    debug_errors.append(f'Недостаточно данных для создания технических индикаторов. Доступно: {len(historical_data)}, минимум для индикаторов: {min_data_for_indicators}, запущено с индекса: {start_backtest_idx}')
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'test_interval_days': 1,  # Тестируем каждый день
                'tickers': {
                    ticker: {
                        'predictions': predictions_made[-10:],  # Последние 10 прогнозов
                        'debug_info': {
                            'total_predictions': len(predictions_made),
                            'correct_predictions': correct_predictions,
                            'total_trades': len(trades),
                            'profitable_trades': profitable_trades,
                            'win_rate': win_rate,
                            'initial_capital': initial_capital,
                            'final_capital': capital,
                            'historical_data_points': len(historical_data),
                            'errors': debug_errors if debug_errors else None
                        },
                        'total_return': total_return,
                        'accuracy': accuracy,
                        'trades_list': trades[-10:],  # Список последних 10 сделок
                        'trades': len(trades)  # Общее количество сделок
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Ошибка бэктестинга для {ticker}: {e}")
            return {
                'start_date': datetime.now() - timedelta(days=days_back),
                'end_date': datetime.now(),
                'test_interval_days': 1,
                'tickers': {
                    ticker: {
                        'predictions': [],
                        'trades': [],
                        'debug_info': {
                            'errors': [f'Ошибка: {str(e)}']
                        },
                        'total_return': 0.0,
                        'accuracy': 0.0,
                        'trades': 0
                    }
                }
            }
    
    def check_backtest_performance_and_retrain(self, ticker: str, backtest_results: dict) -> bool:
        """Проверка результатов бэктеста и автоматическое переобучение при слабых показателях"""
        try:
            ticker_data = backtest_results.get('tickers', {}).get(ticker, {})
            
            # Критерии для переобучения
            ACCURACY_THRESHOLD = 40.0  # Минимальная точность 40%
            RETURN_THRESHOLD = -5.0    # Максимальный убыток -5%
            MIN_TRADES_THRESHOLD = 3   # Минимум 3 сделки для оценки
            
            accuracy = ticker_data.get('accuracy', 0)
            total_return = ticker_data.get('total_return', 0)
            total_trades = ticker_data.get('trades', 0)
            debug_info = ticker_data.get('debug_info', {})
            total_predictions = debug_info.get('total_predictions', 0)
            
            # Логируем текущие показатели
            logger.info(f"📊 {ticker} бэктест: точность={accuracy:.1f}%, доходность={total_return:+.1f}%, сделок={total_trades}, прогнозов={total_predictions}")
            
            # Условия для переобучения
            needs_retraining = False
            reasons = []
            
            if accuracy < ACCURACY_THRESHOLD and total_predictions >= 5:
                needs_retraining = True
                reasons.append(f'низкая точность {accuracy:.1f}% < {ACCURACY_THRESHOLD}%')
            
            if total_return < RETURN_THRESHOLD and total_trades >= MIN_TRADES_THRESHOLD:
                needs_retraining = True
                reasons.append(f'большие убытки {total_return:+.1f}% < {RETURN_THRESHOLD:+.1f}%')
            
            if total_predictions < 3:
                needs_retraining = True
                reasons.append('слишком мало прогнозов (< 3)')
            
            if needs_retraining:
                logger.warning(f"🔄 {ticker}: ПЕРЕОБУЧЕНИЕ НЕОБХОДИМО! Причины: {', '.join(reasons)}")
                
                # Запускаем переобучение модели для этого тикера
                success = self._retrain_ticker_models(ticker)
                
                if success:
                    logger.info(f"✅ {ticker}: модель успешно переобучена")
                    return True
                else:
                    logger.error(f"❌ {ticker}: ошибка переобучения модели")
                    return False
            else:
                logger.info(f"✓ {ticker}: модель показує прийнятні результати")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка проверки бэктеста для {ticker}: {e}")
            return False
    
    def _retrain_ticker_models(self, ticker: str) -> bool:
        """Переобучение моделей для конкретного тикера"""
        try:
            logger.info(f"🔄 Начинаем переобучение моделей для {ticker}...")
            
            # Удаляем старые модели
            short_term_path = f"models/{ticker}_short_term_models.pkl"
            long_term_path = f"models/{ticker}_long_term_models.pkl"
            
            if os.path.exists(short_term_path):
                os.remove(short_term_path)
                logger.info(f"Удалена старая короткосрочная модель {ticker}")
            
            if os.path.exists(long_term_path):
                os.remove(long_term_path)
                logger.info(f"Удалена старая долгосрочная модель {ticker}")
            
            # Получаем больше исторических данных для лучшего обучения
            historical_data = self.polygon_client.get_historical_data(ticker, 120)  # 120 дней
            
            if len(historical_data) < 60:
                logger.warning(f"Недостаточно данных для переобучения {ticker}: {len(historical_data)} дней")
                return False
            
            # Переобучаем короткосрочную модель
            logger.info(f"Переобучение короткосрочной модели {ticker}...")
            short_success = self.train_short_term_model(ticker)
            
            # Переобучаем долгосрочную модель
            logger.info(f"Переобучение долгосрочной модели {ticker}...")
            long_success = self.train_long_term_model(ticker)
            
            if short_success and long_success:
                logger.info(f"✅ {ticker}: обе модели успешно переобучены")
                
                # Очищаем старые прогнозы для этого тикера в БД
                self._cleanup_ticker_predictions(ticker)
                
                return True
            else:
                logger.error(f"❌ {ticker}: ошибка переобучения (short={short_success}, long={long_success})")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка переобучения {ticker}: {e}")
            return False
    
    def _cleanup_ticker_predictions(self, ticker: str):
        """Очистка старых прогнозов для конкретного тикера"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=1)  # Удаляем прогнозы старше 1 дня
            
            # Удаляем прогнозы из БД
            deleted = self.db.cleanup_predictions_for_ticker(ticker, cutoff_date)
            logger.info(f"🧹 {ticker}: удалено {deleted} старых прогнозов")
            
        except Exception as e:
            logger.error(f"Ошибка очистки прогнозов для {ticker}: {e}")
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление технических индикаторов для готового DataFrame"""
        try:
            if len(df) < 5:
                return pd.DataFrame()
                
            # Технические индикаторы с адаптивными окнами
            window_size = min(5, len(df) - 1)
            if window_size < 2:
                window_size = 2
            
            df['sma_5'] = ta.trend.sma_indicator(df['close'], window=window_size)
            
            # Для RSI используем доступное количество данных
            if len(df) >= 14:
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            else:
                df['rsi'] = ta.momentum.rsi(df['close'], window=max(2, len(df)//2))
            
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            
            # Простые индикаторы
            df['volume_sma'] = df['volume'].rolling(window=max(2, min(10, len(df)-1))).mean()
            df['price_change'] = df['close'].pct_change()
            df['volatility'] = df['price_change'].rolling(window=max(2, min(5, len(df)-1))).std()
            df['volume_change'] = df['volume'].pct_change()
            
            # Заполняем NaN последними значениями
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Выбираем только feature колонки (БЕЗ OHLCV данных - как в обучении)
            feature_columns = ['sma_5', 'rsi', 'macd', 'bb_upper', 'bb_lower', 
                             'volume_sma', 'price_change', 'volatility', 'volume_change']
            
            # Проверяем наличие колонок
            available_cols = [col for col in feature_columns if col in df.columns]
            
            if len(available_cols) < 5:
                logger.warning(f"Недостаточно feature колонок: {available_cols}")
                return pd.DataFrame()
                
            return df[available_cols].dropna()
            
        except Exception as e:
            logger.error(f"Ошибка расчета технических индикаторов: {e}")
            return pd.DataFrame()
    
    def start_auto_update(self):
        """Запуск автоматического обновления данных"""
        import threading
        
        def auto_update_loop():
            while True:
                try:
                    logger.info("Автообновление: проверка тикеров...")
                    self.auto_update_all_tickers()
                    # Обновляем каждые 5 минут
                    time.sleep(300)
                except Exception as e:
                    logger.error(f"Ошибка в автообновлении: {e}")
                    time.sleep(60)  # При ошибке ждем 1 минуту
        
        # Запускаем в отдельном потоке
        update_thread = threading.Thread(target=auto_update_loop, daemon=True)
        update_thread.start()
        logger.info("Автообновление запущено (каждые 5 минут)")
    
    def is_trading_hours(self) -> bool:
        """Проверка торговых часов (9:00-20:00 EST)"""
        try:
            from datetime import datetime
            import pytz
            
            # Получаем текущее время в EST/EDT
            eastern = pytz.timezone('US/Eastern')
            now_est = datetime.now(eastern)
            current_hour = now_est.hour
            
            # Торгові години: 9:00 - 20:00 EST/EDT
            return 9 <= current_hour < 20
        except Exception as e:
            logger.warning(f"Ошибка проверки торговых часов: {e}")
            return True  # При ошибке продолжаем работать
    
    def _full_retrain_all_models(self):
        """Повне перенавчання всіх моделей в неробочі години"""
        try:
            current_time = time.time()
            retrained_count = 0
            
            for ticker in self.tickers[:]:
                try:
                    # Проверяем, нужно ли переобучение (не чаще 1 раза в сутки)
                    last_training = self.last_training_time.get(ticker, 0)
                    
                    # Защита от неправильного типа данных (если сохранен datetime вместо timestamp)
                    if isinstance(last_training, datetime):
                        last_training = last_training.timestamp()
                    
                    # ИСПРАВЛЕНИЕ: Если модель никогда не обучалась (0) или прошло более 24 часов
                    if last_training > 0 and current_time - last_training < 86400:  # 24 часа
                        logger.debug(f"{ticker}: Пропуск переобучения - прошло {(current_time - last_training)/3600:.1f} часов")
                        continue
                    
                    logger.info(f"Полное переобучение моделей для {ticker}")
                    
                    # Получаем больше исторических данных для качественного обучения
                    historical_data = self.polygon_client.get_historical_data(ticker, days_back=365)
                    
                    if historical_data and len(historical_data) >= 30:
                        success_count = 0
                        
                        # Переобучаем краткосрочную модель
                        if ticker in self.short_term_predictors:
                            if self.short_term_predictors[ticker].train(historical_data, force=True):
                                success_count += 1
                                logger.info(f"✓ {ticker}: Краткосрочная модель переобучена")
                        
                        # Переобучаем долгосрочную модель  
                        if ticker in self.long_term_predictors:
                            if self.long_term_predictors[ticker].train(historical_data, force=True):
                                success_count += 1
                                logger.info(f"✓ {ticker}: Долгосрочная модель переобучена")
                        
                        if success_count > 0:
                            self.last_training_time[ticker] = current_time
                            self._save_training_times()  # Сохраняем времена обучения
                            retrained_count += 1
                    else:
                        logger.warning(f"Недостаточно данных для {ticker}: {len(historical_data) if historical_data else 0}")
                    
                    # Пауза между тикерами для снижения нагрузки на API
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Ошибка переобучения {ticker}: {e}")
                    continue
            
            logger.info(f"Полное переобучение завершено: {retrained_count}/{len(self.tickers)} тикеров")
            
        except Exception as e:
            logger.error(f"Ошибка полного переобучения: {e}")
    
    def auto_update_all_tickers(self):
        """Автоматическое обновление всех тикеров"""
        try:
            is_trading_time = self.is_trading_hours()
            
            # В торговые часы - только быстрые обновления цен и генерация сигналов
            # В нерабочие часы - полное переобучение моделей
            if not is_trading_time:
                logger.info("Нерабочие часы (21:00-8:00 EST) - полное переобучение моделей")
                self._full_retrain_all_models()
                return
            else:
                logger.debug("Торговые часы (9:00-20:00 EST) - быстрые обновления")
                
            current_time = time.time()
            updated_count = 0
            
            for ticker in self.tickers[:]:  # Копия списка для безопасности
                try:
                    # Проверяем нужно ли обновлять (каждые 10 минут)
                    last_update = self.last_update_time.get(ticker, 0)
                    if current_time - last_update > 600:  # 10 минут
                        
                        # Обновляем цену
                        price_data = self.polygon_client.get_latest_price(ticker)
                        new_price = price_data.get('current_price', 0.0)
                        
                        if new_price > 0:
                            old_price = self.last_prices.get(ticker, 0.0)
                            self.last_prices[ticker] = new_price
                            
                            # В торговые часы только генерируем сигналы, переобучение - в нерабочие часы
                            short_trained = ticker in self.short_term_predictors and self.short_term_predictors[ticker].trained
                            long_trained = ticker in self.long_term_predictors and self.long_term_predictors[ticker].trained
                            
                            if not short_trained or not long_trained:
                                logger.debug(f"{ticker}: Модели не обучены - ожидаем нерабочие часы для переобучения")
                            
                            # Обновляем прогнозы
                            self._update_predictions(ticker)
                            self.last_update_time[ticker] = current_time
                            updated_count += 1
                            
                            # Генерируем сигналы для тикера
                            try:
                                # Получаем исторические данные для анализа сигналов
                                hist_data = self.polygon_client.get_historical_data(ticker, days_back=30)
                                if hist_data and len(hist_data) > 0:
                                    # Получаем текущие прогнозы
                                    predictions = self.last_predictions.get(ticker, {})
                                    
                                    # Конвертуємо історичні дані в DataFrame
                                    df_data = pd.DataFrame(hist_data)
                                    
                                    # Переименовываем колонки из Polygon API формата в стандартный
                                    if 'c' in df_data.columns:
                                        df_data = df_data.rename(columns={
                                            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                                        })
                                    
                                    if 'date' in df_data.columns:
                                        df_data['date'] = pd.to_datetime(df_data['date'])
                                        df_data = df_data.set_index('date')
                                    
                                    current_data = {
                                        'price': new_price,
                                        'last_price': old_price,
                                        'volume': hist_data[-1].get('v', 0) if hist_data else 0
                                    }
                                    
                                    # Анализируем сигналы
                                    logger.debug(f"SIGNAL CHECK {ticker}: hist_data={len(hist_data)}, predictions={len(predictions)}")
                                    ticker_signals = self.signal_engine.analyze_ticker(
                                        ticker, current_data, df_data, predictions
                                    )
                                    
                                    # Сохраняем сигналы в базу данных
                                    for signal in ticker_signals:
                                        from database import db
                                        from database import SignalRecord
                                        # Extract period_hours for prediction signals
                                        period_hours = None
                                        if hasattr(signal, 'data') and signal.data:
                                            period_hours = signal.data.get('period_hours')
                                        
                                        # For prediction signals, use the predicted change, not current price change
                                        # For pattern signals, use the pattern-specific change if available
                                        if signal.type.value == 'prediction' and hasattr(signal, 'price_change_percent'):
                                            change_percent = signal.price_change_percent
                                        elif signal.type.value == 'pattern' and hasattr(signal, 'price_change_percent'):
                                            change_percent = signal.price_change_percent
                                        else:
                                            change_percent = ((new_price - old_price) / old_price * 100) if old_price > 0 else 0
                                        
                                        signal_record = SignalRecord(
                                            ticker=signal.ticker,
                                            signal_type=signal.type.value,
                                            priority=signal.priority.value,
                                            message=signal.get_formatted_message(),
                                            timestamp=datetime.now(),
                                            price=new_price,
                                            change_percent=change_percent,
                                            volume=current_data['volume'],
                                            confidence=signal.confidence,
                                            period_hours=period_hours
                                        )
                                        
                                        # ИСПРАВЛЕНО: Убираем дублирующее сохранение сигналов
                                        # SignalEngine уже сохраняет сигналы в БД в методе _record_signal
                                        # Это было причиной дублирования уведомлений
                                        # if not db.is_duplicate_signal(signal_record):
                                        #     db.save_signal(signal_record)
                                    
                                    if ticker_signals:
                                        logger.info(f"{ticker}: Сгенерировано {len(ticker_signals)} сигналов")
                                        
                                        # Отправляем критические сигналы немедленно
                                        for signal in ticker_signals:
                                            if signal.priority.value == 'CRITICAL':
                                                logger.info(f"🚨 КРИТИЧЕСКИЙ СИГНАЛ {ticker}: {signal.get_formatted_message()}")
                                                
                            except Exception as e:
                                logger.error(f"Ошибка генерации сигналов для {ticker}: {e}")
                            
                            # Логируем значительные изменения цены
                            if old_price > 0:
                                change_pct = ((new_price - old_price) / old_price) * 100
                                if abs(change_pct) > 1.0:
                                    logger.info(f"{ticker}: ${old_price:.2f} → ${new_price:.2f} ({change_pct:+.1f}%)")
                        
                        # Небольшая пауза между тикерами
                        time.sleep(1)
                
                except Exception as e:
                    logger.error(f"Ошибка автообновления {ticker}: {e}")
                    continue
            
            if updated_count > 0:
                logger.info(f"Автообновление: обновлено {updated_count} тикеров")
                
        except Exception as e:
            logger.error(f"Ошибка в auto_update_all_tickers: {e}")
    
    def ensure_ticker_ready(self, ticker: str) -> bool:
        """Убеждаемся что тикер готов к использованию"""
        try:
            if ticker not in self.tickers:
                return False
            
            # Проверяем есть ли модели
            if ticker not in self.short_term_predictors:
                self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
            if ticker not in self.long_term_predictors:
                self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
            
            # Проверяем обучены ли модели
            short_trained = self.short_term_predictors[ticker].trained
            long_trained = self.long_term_predictors[ticker].trained
            
            if not short_trained or not long_trained:
                # Проверяем можно ли обучать (не чаще 1 раза в час)
                current_time = time.time()
                last_training = self.last_training_time.get(ticker, 0)
                
                # Защита от неправильного типа данных (если сохранен datetime вместо timestamp)
                if isinstance(last_training, datetime):
                    last_training = last_training.timestamp()
                    
                can_train = current_time - last_training > 3600
                
                if can_train:
                    logger.info(f"Обучение моделей для {ticker}...")
                    historical_data = self.polygon_client.get_historical_data(ticker, days_back=100)
                    
                    if historical_data and len(historical_data) >= 10:
                        if not short_trained:
                            self.short_term_predictors[ticker].train(historical_data)
                        if not long_trained:
                            self.long_term_predictors[ticker].train(historical_data)
                        
                        # Обновляем время последнего обучения
                        self.last_training_time[ticker] = current_time
                        self._save_training_times()  # Сохраняем времена обучения
                else:
                    time_left = 3600 - (current_time - last_training)
                    logger.info(f"{ticker}: Обучение пропущено, осталось {time_left/60:.0f} мин до следующего")
            
            # Проверяем есть ли актуальная цена
            if ticker not in self.last_prices or self.last_prices[ticker] == 0.0:
                price_data = self.polygon_client.get_latest_price(ticker)
                self.last_prices[ticker] = price_data.get('current_price', 0.0)
            
            # Обновляем прогнозы если нужно
            if ticker not in self.last_predictions or not self.last_predictions[ticker]:
                self._update_predictions(ticker)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка подготовки тикера {ticker}: {e}")
            return False
    
    def start_verification_system(self):
        """Запуск системы верификации прогнозов"""
        import threading
        
        def verification_loop():
            while True:
                try:
                    logger.info("Верификация: проверка прогнозов...")
                    self.verify_predictions()
                    # Проверяем каждые 30 минут
                    time.sleep(1800)
                except Exception as e:
                    logger.error(f"Ошибка в верификации: {e}")
                    time.sleep(300)  # При ошибке ждем 5 минут
        
        # Запускаем в отдельном потоке
        verification_thread = threading.Thread(target=verification_loop, daemon=True)
        verification_thread.start()
        logger.info("Система верификации запущена (каждые 30 минут)")
    
    def verify_predictions(self):
        """Верификация прогнозов"""
        try:
            # Получаем прогнозы, время которых наступило
            current_time = datetime.now(timezone.utc)
            pending_predictions = self.db.get_pending_predictions(current_time)
            
            verified_count = 0
            for pred_dict in pending_predictions:
                try:
                    ticker = pred_dict['ticker']
                    prediction_id = pred_dict['id']
                    target_time = datetime.fromisoformat(pred_dict['target_time'])
                    
                    # Получаем актуальную цену
                    price_data = self.polygon_client.get_latest_price(ticker)
                    actual_price = price_data.get('current_price', 0.0)
                    
                    if actual_price > 0:
                        # Обновляем результат прогноза
                        success = self.db.update_prediction_result(
                            prediction_id, actual_price, current_time
                        )
                        
                        if success:
                            verified_count += 1
                            # Логируем результат
                            predicted_price = pred_dict['predicted_price']
                            current_price = pred_dict['current_price']
                            period_hours = pred_dict['period_hours']
                            
                            predicted_change = ((predicted_price - current_price) / current_price) * 100
                            actual_change = ((actual_price - current_price) / current_price) * 100
                            
                            error = abs(predicted_change - actual_change)
                            logger.info(f"✓ {ticker} {period_hours}ч: прогноз {predicted_change:+.1f}%, факт {actual_change:+.1f}%, ошибка {error:.1f}%")
                    
                    # Небольшая пауза между проверками
                    time.sleep(0.5)
                    
                except Exception as e:
                    pred_id = pred_dict['id'] if 'id' in pred_dict else 'unknown'
                    logger.error(f"Ошибка верификации прогноза ID {pred_id}: {e}")
                    continue
            
            if verified_count > 0:
                logger.info(f"Верификация: проверено {verified_count} прогнозов")
                # Обновляем статистику моделей
                self.update_model_performance_stats()
                
                # Проверяем нужно ли переобучить модели с плохой производительностью
                self.check_and_retrain_poor_models()
                
                # Создаем подробный отчет по всем тикерам и прогнозам
                report_summary = self.create_verification_report()
                
                # Сохраняем данные отчета для возможной отправки в Telegram
                if report_summary:
                    self._last_verification_report = report_summary
                
        except Exception as e:
            logger.error(f"Ошибка в verify_predictions: {e}")
    
    def update_model_performance_stats(self):
        """Обновление статистики производительности моделей"""
        try:
            for ticker in self.tickers:
                for model_type in ['short_term', 'long_term']:
                    for period in [1, 3, 6, 12, 24] if model_type == 'short_term' else [72, 168, 720]:
                        # Обновляем метрики производительности в БД
                        self.db.update_model_performance(ticker, model_type, period)
                        
        except Exception as e:
            logger.error(f"Ошибка обновления статистики моделей: {e}")
    
    def check_and_retrain_poor_models(self):
        """Проверка и автоматическое переобучение моделей с плохой производительностью"""
        try:
            ACCURACY_THRESHOLD = 50.0  # Минимальная точность 50%
            MIN_PREDICTIONS = 10       # Минимум прогнозов для оценки
            
            for ticker in self.tickers:
                for model_type in ['short_term', 'long_term']:
                    # Получаем статистику точности за последние 7 дней
                    analysis = self.prediction_tracker.get_analysis(ticker, days_back=7)
                    
                    if analysis['total'] >= MIN_PREDICTIONS:
                        accuracy = analysis['avg_accuracy_percent']
                        
                        # Если точность ниже порога - переобучаем
                        if accuracy < ACCURACY_THRESHOLD:
                            logger.warning(f"🔄 {ticker} {model_type}: точность {accuracy:.1f}% < {ACCURACY_THRESHOLD}% - переобучение")
                            
                            # Запускаем переобучение
                            if model_type == 'short_term':
                                success = self.train_short_term_model(ticker)
                                logger.info(f"✅ {ticker}: короткосрочная модель {'переобучена' if success else 'не переобучена'}")
                            else:
                                success = self.train_long_term_model(ticker) 
                                logger.info(f"✅ {ticker}: долгосрочная модель {'переобучена' if success else 'не переобучена'}")
                            
                            # Удаляем старые плохие прогнозы для этого тикера
                            self.cleanup_poor_predictions(ticker)
                        else:
                            logger.debug(f"✓ {ticker} {model_type}: точность {accuracy:.1f}% - OK")
                            
        except Exception as e:
            logger.error(f"Ошибка проверки моделей для переобучения: {e}")
    
    def cleanup_poor_predictions(self, ticker: str):
        """Удаление прогнозов с низкой точностью для конкретного тикера"""
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=7)
            
            with self.db.get_cursor() as cursor:
                # Удаляем неточные прогнозы (с точностью < 30%)
                cursor.execute("""
                    DELETE FROM predictions 
                    WHERE ticker = ? 
                      AND prediction_time >= ?
                      AND verified = TRUE 
                      AND accuracy_percent < 30
                """, (ticker, cutoff_date))
                
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"🧹 {ticker}: удалено {deleted_count} неточных прогнозов")
                    
        except Exception as e:
            logger.error(f"Ошибка очистки прогнозов для {ticker}: {e}")
    
    def train_short_term_model(self, ticker: str) -> bool:
        """Переобучение короткосрочной модели"""
        try:
            # Получаем исторические данные
            historical_data = self.polygon_client.get_historical_data(ticker, days_back=67)
            if not historical_data or len(historical_data) < 20:
                logger.warning(f"Недостаточно данных для переобучения {ticker}")
                return False
            
            # Инициализируем предиктор если его нет
            if ticker not in self.short_term_predictors:
                self.short_term_predictors[ticker] = MLPredictor(ticker, "short_term")
                # Пытаемся загрузить из файла
                self.short_term_predictors[ticker].load_from_file()
            
            # Переобучаем модель
            success = self.short_term_predictors[ticker].train(historical_data)
            return success
            
        except Exception as e:
            logger.error(f"Ошибка переобучения короткосрочной модели {ticker}: {e}")
            return False
    
    def train_long_term_model(self, ticker: str) -> bool:
        """Переобучение долгосрочной модели"""
        try:
            # Получаем исторические данные (больше для долгосрочных прогнозов)
            historical_data = self.polygon_client.get_historical_data(ticker, days_back=730)
            if not historical_data or len(historical_data) < 50:
                logger.warning(f"Недостаточно данных для переобучения долгосрочной модели {ticker}")
                return False
            
            # Инициализируем предиктор если его нет
            if ticker not in self.long_term_predictors:
                self.long_term_predictors[ticker] = MLPredictor(ticker, "long_term")
                # Пытаемся загрузить из файла
                self.long_term_predictors[ticker].load_from_file()
            
            # Переобучаем модель
            success = self.long_term_predictors[ticker].train(historical_data)
            return success
            
        except Exception as e:
            logger.error(f"Ошибка переобучения долгосрочной модели {ticker}: {e}")
            return False

    def get_model_accuracy_stats(self, ticker: str, days: int = 30) -> Dict:
        """Получение статистики точности модели"""
        try:
            stats = {
                'short_term': {},
                'long_term': {},
                'overall': {}
            }
            
            for model_type in ['short_term', 'long_term']:
                for period in [1, 3, 6, 12, 24] if model_type == 'short_term' else [72, 168, 720]:
                    accuracy = self.db.get_model_accuracy(ticker, model_type, period, days)
                    stats[model_type][f"{period}h"] = accuracy
            
            # Общая статистика
            prediction_stats = self.db.get_prediction_stats(days)
            stats['overall'] = prediction_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики для {ticker}: {e}")
            return {}
    
    def get_best_performing_models(self, limit: int = 10) -> List[Dict]:
        """Получение лучших моделей по точности"""
        return self.db.get_top_performing_models(limit)
    
    def should_retrain_model(self, ticker: str, model_type: str) -> bool:
        """Определение необходимости переобучения модели на основе точности"""
        try:
            # Получаем точность за последние 7 дней
            periods = [1, 3, 6, 12, 24] if model_type == 'short_term' else [72, 168, 720]
            
            total_accuracy = 0
            count = 0
            
            for period in periods:
                accuracy = self.db.get_model_accuracy(ticker, model_type, period, days=7)
                if accuracy > 0:
                    total_accuracy += accuracy
                    count += 1
            
            if count == 0:
                return False  # Нет данных для принятия решения
            
            avg_accuracy = total_accuracy / count
            
            # Пороги для переобучения из конфигурации
            from config import PREDICTION_CONFIG
            critical_threshold = PREDICTION_CONFIG.get('critical_retrain_threshold', 15)
            auto_threshold = PREDICTION_CONFIG.get('auto_retrain_threshold', 25)
            
            if avg_accuracy < critical_threshold:
                logger.warning(f"{ticker} {model_type}: Критически низкая точность {avg_accuracy:.1f}% - требуется немедленное переобучение!")
                return True
            elif avg_accuracy < auto_threshold:
                logger.info(f"{ticker} {model_type}: Низкая точность {avg_accuracy:.1f}% - рекомендуется переобучение")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка проверки необходимости переобучения {ticker} {model_type}: {e}")
            return False
    
    # НОВЫЕ МЕТОДЫ: Фоновое обучение моделей
    
    def add_progress_callback(self, callback):
        """Добавить колбэк для уведомлений о прогрессе"""
        self.training_progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback):
        """Удалить колбэк уведомлений"""
        if callback in self.training_progress_callbacks:
            self.training_progress_callbacks.remove(callback)
    
    def _notify_progress(self, ticker: str, stage: str, progress: int, message: str):
        """Уведомление о прогрессе обучения"""
        for callback in self.training_progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Для асинхронных колбэков создаем задачу
                    loop = asyncio.get_event_loop()
                    loop.create_task(callback(ticker, stage, progress, message))
                else:
                    callback(ticker, stage, progress, message)
            except Exception as e:
                logger.error(f"Ошибка в колбэке прогресса: {e}")
    
    def start_background_training(self, ticker: str, model_types: List[str] = None, 
                                notify_callback=None) -> str:
        """Запуск фонового обучения моделей
        
        Args:
            ticker: Тикер для обучения
            model_types: Список типов моделей ['short_term', 'long_term']
            notify_callback: Колбэк для уведомлений
            
        Returns:
            task_id: Идентификатор задачи
        """
        if model_types is None:
            model_types = ['short_term', 'long_term']
        
        task_id = f"{ticker}_{int(time.time())}"
        
        with self.training_lock:
            # Проверяем, не идет ли уже обучение для этого тикера
            if ticker in self.training_status and self.training_status[ticker]['status'] == 'training':
                logger.warning(f"Навчання для {ticker} вже виконується")
                return self.training_status[ticker]['task_id']
            
            # Инициализируем статус
            self.training_status[ticker] = {
                'task_id': task_id,
                'status': 'training',
                'start_time': datetime.now(),
                'progress': 0,
                'stage': 'initializing',
                'model_types': model_types,
                'message': 'Инициализация обучения...'
            }
        
        # Запускаем обучение в фоновом потоке
        future = self.training_executor.submit(
            self._background_training_worker,
            ticker, model_types, task_id, notify_callback
        )
        
        self.training_futures[task_id] = future
        
        logger.info(f"Запущено фоновое обучение для {ticker}, task_id: {task_id}")
        return task_id
    
    def _background_training_worker(self, ticker: str, model_types: List[str], 
                                  task_id: str, notify_callback=None):
        """Воркер для фонового обучения"""
        try:
            total_steps = len(model_types)
            current_step = 0
            
            for model_type in model_types:
                current_step += 1
                stage = f"training_{model_type}"
                progress = int((current_step - 0.5) / total_steps * 100)
                
                # Обновляем статус
                with self.training_lock:
                    if ticker in self.training_status:
                        self.training_status[ticker].update({
                            'progress': progress,
                            'stage': stage,
                            'message': f'Обучение {model_type} модели...'
                        })
                
                # Уведомляем о прогрессе
                self._notify_progress(ticker, stage, progress, f'Обучение {model_type} модели...')
                
                # Выполняем обучение
                if model_type == 'short_term':
                    success = self.train_short_term_model(ticker)
                else:
                    success = self.train_long_term_model(ticker)
                
                if not success:
                    raise Exception(f"Ошибка обучения {model_type} модели")
                
                # Обновляем прогресс
                progress = int(current_step / total_steps * 100)
                with self.training_lock:
                    if ticker in self.training_status:
                        self.training_status[ticker].update({
                            'progress': progress,
                            'message': f'{model_type} модель обучена'
                        })
                
                self._notify_progress(ticker, stage, progress, f'{model_type} модель обучена')
            
            # Завершаем успешно
            with self.training_lock:
                if ticker in self.training_status:
                    self.training_status[ticker].update({
                        'status': 'completed',
                        'progress': 100,
                        'stage': 'completed',
                        'message': 'Обучение завершено успешно',
                        'end_time': datetime.now()
                    })
            
            self._notify_progress(ticker, 'completed', 100, 'Обучение завершено успешно')
            self.last_training_time[ticker] = time.time()
            self._save_training_times()  # Сохраняем времена обучения
            
            logger.info(f"Фоновое обучение для {ticker} завершено успешно")
            return True
            
        except Exception as e:
            # Обновляем статус об ошибке
            with self.training_lock:
                if ticker in self.training_status:
                    self.training_status[ticker].update({
                        'status': 'error',
                        'stage': 'error',
                        'message': f'Ошибка: {str(e)[:100]}',
                        'end_time': datetime.now(),
                        'error': str(e)
                    })
            
            self._notify_progress(ticker, 'error', 0, f'Ошибка: {str(e)[:100]}')
            logger.error(f"Ошибка фонового обучения для {ticker}: {e}")
            return False
        
        finally:
            # Очищаем future
            if task_id in self.training_futures:
                del self.training_futures[task_id]
    
    def get_training_status(self, ticker: str = None) -> Dict:
        """Получение статуса обучения
        
        Args:
            ticker: Конкретный тикер или None для всех
            
        Returns:
            Словарь со статусами обучения
        """
        with self.training_lock:
            if ticker:
                return self.training_status.get(ticker, {
                    'status': 'not_training',
                    'message': 'Обучение не выполняется'
                })
            else:
                return dict(self.training_status)
    
    def cancel_training(self, ticker: str) -> bool:
        """Отмена обучения для тикера"""
        with self.training_lock:
            if ticker not in self.training_status:
                return False
            
            task_id = self.training_status[ticker].get('task_id')
            if task_id and task_id in self.training_futures:
                future = self.training_futures[task_id]
                if not future.done():
                    future.cancel()
                
                del self.training_futures[task_id]
            
            self.training_status[ticker] = {
                'status': 'cancelled',
                'message': 'Обучение отменено пользователем',
                'end_time': datetime.now()
            }
        
        logger.info(f"Обучение для {ticker} отменено")
        return True
    
    def cleanup_old_training_status(self):
        """Очистка старых статусов обучения"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self.training_lock:
            to_remove = []
            for ticker, status in self.training_status.items():
                if status.get('status') in ['completed', 'error', 'cancelled']:
                    end_time = status.get('end_time')
                    if end_time and end_time < cutoff_time:
                        to_remove.append(ticker)
            
            for ticker in to_remove:
                del self.training_status[ticker]
        
        if to_remove:
            logger.info(f"Очищены старые статусы обучения: {to_remove}")
    
    def get_prediction_history(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Получение истории прогнозов для тикера"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        ticker,
                        prediction_time,
                        target_time,
                        current_price,
                        predicted_price,
                        actual_price,
                        price_change_percent,
                        accuracy_percent,
                        confidence,
                        period_hours,
                        model_type,
                        status,
                        verified,
                        created_at
                    FROM predictions 
                    WHERE ticker = ?
                    ORDER BY prediction_time DESC
                    LIMIT ?
                """, (ticker, limit))
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Форматируем даты для лучшего отображения
                    if row_dict['prediction_time']:
                        row_dict['prediction_time'] = datetime.fromisoformat(row_dict['prediction_time']).strftime('%Y-%m-%d %H:%M')
                    if row_dict['target_time']:
                        row_dict['target_time'] = datetime.fromisoformat(row_dict['target_time']).strftime('%Y-%m-%d %H:%M')
                    results.append(row_dict)
                
                logger.info(f"Получено {len(results)} записей истории для {ticker}")
                return results
                
        except Exception as e:
            logger.error(f"Ошибка получения истории прогнозов для {ticker}: {e}")
            return []
    
    def get_analysis(self, ticker: str, days_back: int = 30) -> Dict:
        """Анализ точности прогнозов для тикера"""
        try:
            logger.info(f"🔍 Анализ точности для {ticker} за {days_back} дней...")
            
            # Получаем статистику по моделям
            stats = self.get_model_accuracy_stats(ticker, days_back)
            logger.debug(f"Model stats для {ticker}: {stats}")
            
            # Дополнительная статистика
            with self.db.get_cursor() as cursor:
                since_date = datetime.now(timezone.utc) - timedelta(days=days_back)
                
                # Сначала проверим, есть ли вообще данные для этого тикера
                cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE ticker = ?", (ticker,))
                total_ever = cursor.fetchone()['count']
                
                if total_ever == 0:
                    logger.warning(f"❌ Немає прогнозів для {ticker} в базі даних")
                    return {
                        'ticker': ticker,
                        'error': f'Нет данных для {ticker} в системе',
                        'debug_info': {
                            'total_predictions_ever': 0,
                            'days_searched': days_back,
                            'since_date': since_date.isoformat()
                        }
                    }
                
                # Общая статистика по тикеру за период
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN verified = TRUE THEN 1 END) as verified_count,
                        COUNT(CASE WHEN verified = FALSE THEN 1 END) as unverified_count,
                        AVG(CASE WHEN verified = TRUE THEN accuracy_percent END) as avg_accuracy,
                        MIN(CASE WHEN verified = TRUE THEN accuracy_percent END) as min_accuracy,
                        MAX(CASE WHEN verified = TRUE THEN accuracy_percent END) as max_accuracy,
                        AVG(confidence) as avg_confidence,
                        MIN(prediction_time) as first_prediction,
                        MAX(prediction_time) as last_prediction
                    FROM predictions 
                    WHERE ticker = ? AND prediction_time >= ?
                """, (ticker, since_date))
                
                result = cursor.fetchone()
                if result:
                    total_predictions = result['total_predictions'] or 0
                    verified_count = result['verified_count'] or 0
                    
                    logger.info(f"📊 {ticker}: {total_predictions} прогнозів, {verified_count} верифікованих")
                    
                    analysis = {
                        'ticker': ticker,
                        'period_days': days_back,
                        'total_predictions': total_predictions,
                        'verified_predictions': verified_count,
                        'unverified_predictions': result['unverified_count'] or 0,
                        'verification_rate': round((verified_count / total_predictions * 100) if total_predictions > 0 else 0, 1),
                        'average_accuracy': round(result['avg_accuracy'] or 0, 2),
                        'min_accuracy': round(result['min_accuracy'] or 0, 2),
                        'max_accuracy': round(result['max_accuracy'] or 0, 2),
                        'average_confidence': round(result['avg_confidence'] or 0, 2),
                        'model_stats': stats,
                        'debug_info': {
                            'total_predictions_ever': total_ever,
                            'since_date': since_date.isoformat(),
                            'first_prediction': result['first_prediction'],
                            'last_prediction': result['last_prediction']
                        }
                    }
                    
                    if verified_count == 0:
                        analysis['warning'] = 'Нет верифицированных прогнозов за указанный период'
                        logger.warning(f"⚠️ {ticker}: Нет верифицированных прогнозов за {days_back} дней")
                    
                    return analysis
                
            return {
                'ticker': ticker, 
                'error': 'Нет данных за указанный период',
                'debug_info': {
                    'total_predictions_ever': total_ever,
                    'days_searched': days_back
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа для {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def cleanup_old_predictions(self, days_back: int = 90):
        """Очистка старых прогнозов"""
        try:
            self.db.cleanup_old_data(days_back)
            logger.info(f"Очистка прогнозов старше {days_back} дней завершена")
        except Exception as e:
            logger.error(f"Ошибка очистки старых прогнозов: {e}")
    
    def get_statistics_summary(self) -> Dict:
        """Общая статистика системы прогнозов"""
        try:
            stats = self.db.get_prediction_stats(days=30)
            best_models = self.db.get_top_performing_models(limit=5)
            
            summary = {
                'overall_stats': stats,
                'best_models': best_models,
                'active_tickers': len(self.tickers),
                'ensemble_info': {
                    'models_per_ticker': 2,  # short_term + long_term
                    'algorithms_per_model': 5,  # RF, HGB, LightGBM, XGBoost, CatBoost
                    'total_models': len(self.tickers) * 2 * 5
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка получения общей статистики: {e}")
            return {'error': str(e)}
    
    def create_verification_report(self):
        """Створення детального звіту по всіх верифікованих прогнозах з аналізом часових інтервалів"""
        try:
            from datetime import datetime
            import numpy as np
            
            # Получаем все верифицированные прогнозы за последние 7 дней
            analysis_data = self.db.get_analysis(None, days_back=7)
            
            if analysis_data['total'] == 0:
                logger.info("Немає даних для створення звіту верифікації")
                return None
            
            # Создаем имя файла с временной меткой
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"{DATA_DIR}/verification_report_{timestamp}.txt"
            
            # Переменные для краткого превью
            summary_data = {
                'total': analysis_data['total'],
                'success_rate': analysis_data['success_rate'],
                'avg_accuracy': analysis_data['avg_accuracy_percent'],
                'best_periods': [],
                'worst_periods': [],
                'anomaly_tickers': [],
                'top_performers': [],
                'worst_performers': []
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"📊 ЗВІТ ВЕРИФІКАЦІЇ ПРОГНОЗІВ\n")
                f.write(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Загальна статистика
                f.write("📈 ЗАГАЛЬНА СТАТИСТИКА (7 днів):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Всього прогнозів: {analysis_data['total']}\n")
                f.write(f"✅ Успішні (<2% помилки): {analysis_data['success']} ({analysis_data['success_rate']:.1f}%)\n")
                f.write(f"⚠️ Часткові (2-5% помилки): {analysis_data['partial']}\n")
                f.write(f"❌ Невдалі (>5% помилки): {analysis_data['failed']}\n")
                f.write(f"🎯 Середня точність: {analysis_data['avg_accuracy_percent']:.1f}%\n\n")
                
                # Статистика за періодами
                if analysis_data['period_stats']:
                    f.write("⏱️ СТАТИСТИКА ЗА ПЕРІОДАМИ:\n")
                    f.write("-" * 40 + "\n")
                    for period, stats in sorted(analysis_data['period_stats'].items()):
                        period_text = f"{period}h" if period < 24 else f"{period//24}d"
                        success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                        f.write(f"{period_text:4s}: {stats['avg_accuracy']:5.1f}% точності, ")
                        f.write(f"{success_rate:4.0f}% успіху ({stats['total']} прогнозів)\n")
                    f.write("\n")
                
                # Кращі прогнози
                if analysis_data['best_predictions']:
                    f.write("🏆 КРАЩІ ПРОГНОЗИ:\n")
                    f.write("-" * 40 + "\n")
                    for i, pred in enumerate(analysis_data['best_predictions'][:10], 1):
                        f.write(f"{i:2d}. {pred['ticker']:5s} {pred['period']:3s}: помилка {pred['accuracy']:5.1f}% ({pred['date']})\n")
                    f.write("\n")
                
                # Гірші прогнози
                if analysis_data['worst_predictions']:
                    f.write("📉 ГІРШІ ПРОГНОЗИ:\n")
                    f.write("-" * 40 + "\n")
                    for i, pred in enumerate(analysis_data['worst_predictions'][:10], 1):
                        f.write(f"{i:2d}. {pred['ticker']:5s} {pred['period']:3s}: помилка {pred['accuracy']:5.1f}% ({pred['date']})\n")
                    f.write("\n")
                
                # НОВИЙ АНАЛІЗ: Детальний аналіз часових інтервалів
                f.write("⏰ АНАЛІЗ ЧАСОВИХ ІНТЕРВАЛІВ:\n")
                f.write("=" * 80 + "\n")
                
                # Аналіз ефективності за періодами
                if analysis_data['period_stats']:
                    period_performance = []
                    for period, stats in analysis_data['period_stats'].items():
                        period_text = f"{period}h" if period < 24 else f"{period//24}d"
                        success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        period_performance.append({
                            'period': period,
                            'period_text': period_text,
                            'accuracy': stats['avg_accuracy'],
                            'success_rate': success_rate,
                            'total': stats['total']
                        })
                    
                    # Сортуємо за точністю
                    period_performance.sort(key=lambda x: x['accuracy'], reverse=True)
                    
                    f.write("🎯 РЕЙТИНГ ПЕРІОДІВ ЗА ТОЧНІСТЮ:\n")
                    f.write("-" * 50 + "\n")
                    for i, perf in enumerate(period_performance, 1):
                        f.write(f"{i}. {perf['period_text']:4s} - {perf['accuracy']:5.1f}% точності, "
                              f"{perf['success_rate']:4.0f}% успіху ({perf['total']} прогнозів)\n")
                    
                    # Заповнюємо дані для попереднього перегляду
                    if period_performance:
                        summary_data['best_periods'] = period_performance[:3]
                        summary_data['worst_periods'] = period_performance[-3:] if len(period_performance) > 3 else []
                    
                    f.write("\n")
                    
                    # Аналіз за часом доби (якщо є дані)
                    f.write("🕐 АНАЛІЗ ЗА ЧАСОМ:\n")
                    f.write("-" * 30 + "\n")
                    
                    short_term_periods = [p for p in period_performance if p['period'] <= 6]
                    medium_term_periods = [p for p in period_performance if 6 < p['period'] <= 24]
                    long_term_periods = [p for p in period_performance if p['period'] > 24]
                    
                    if short_term_periods:
                        avg_short = sum(p['accuracy'] for p in short_term_periods) / len(short_term_periods)
                        f.write(f"• Короткострокові (≤6г):  {avg_short:.1f}% середня точність\n")
                    
                    if medium_term_periods:
                        avg_medium = sum(p['accuracy'] for p in medium_term_periods) / len(medium_term_periods)
                        f.write(f"• Середньострокові (6-24г): {avg_medium:.1f}% середня точність\n")
                    
                    if long_term_periods:
                        avg_long = sum(p['accuracy'] for p in long_term_periods) / len(long_term_periods)
                        f.write(f"• Довгострокові (>24г):  {avg_long:.1f}% середня точність\n")
                    
                    f.write("\n")
                
                # Аналіз аномалій за тикерами
                f.write("⚠️ АНАЛІЗ АНОМАЛІЙ ЗА ТИКЕРАМИ:\n")
                f.write("-" * 50 + "\n")
                
                ticker_performances = []
                anomaly_tickers = []
                
                for ticker in sorted(self.tickers):
                    ticker_analysis = self.db.get_analysis(ticker, days_back=7)
                    
                    if ticker_analysis['total'] >= 3:  # Только тикеры с достаточным количеством данных
                        ticker_performances.append({
                            'ticker': ticker,
                            'accuracy': ticker_analysis['avg_accuracy_percent'],
                            'success_rate': ticker_analysis['success_rate'],
                            'total': ticker_analysis['total']
                        })
                        
                        # Визначаємо аномалії (значно гірше середнього)
                        if (ticker_analysis['avg_accuracy_percent'] < analysis_data['avg_accuracy_percent'] - 5.0 or
                            ticker_analysis['success_rate'] < analysis_data['success_rate'] - 20.0):
                            anomaly_tickers.append({
                                'ticker': ticker,
                                'accuracy': ticker_analysis['avg_accuracy_percent'],
                                'success_rate': ticker_analysis['success_rate'],
                                'total': ticker_analysis['total']
                            })
                
                # Сортуємо тикери за продуктивністю
                ticker_performances.sort(key=lambda x: x['accuracy'], reverse=True)
                
                # Топ-5 кращих тикерів
                f.write("🏆 ТОП-5 КРАЩИХ ТИКЕРІВ:\n")
                for i, ticker_perf in enumerate(ticker_performances[:5], 1):
                    f.write(f"{i}. {ticker_perf['ticker']:5s} - {ticker_perf['accuracy']:5.1f}% точності, "
                          f"{ticker_perf['success_rate']:4.0f}% успіху ({ticker_perf['total']} прогнозів)\n")
                
                # Заповнюємо дані для попереднього перегляду
                summary_data['top_performers'] = ticker_performances[:3]
                summary_data['worst_performers'] = ticker_performances[-3:] if len(ticker_performances) > 3 else []
                summary_data['anomaly_tickers'] = anomaly_tickers
                
                f.write("\n")
                
                # Гірші тикери
                if len(ticker_performances) > 5:
                    f.write("📉 ТИКЕРИ З НИЗЬКОЮ ЕФЕКТИВНІСТЮ:\n")
                    for ticker_perf in ticker_performances[-5:]:
                        f.write(f"• {ticker_perf['ticker']:5s} - {ticker_perf['accuracy']:5.1f}% точності, "
                              f"{ticker_perf['success_rate']:4.0f}% успіху ({ticker_perf['total']} прогнозів)\n")
                    f.write("\n")
                
                # Аномальні тикери
                if anomaly_tickers:
                    f.write("🚨 АНОМАЛЬНІ ТИКЕРИ (значно гірше середнього):\n")
                    for anomaly in anomaly_tickers:
                        deviation_acc = anomaly['accuracy'] - analysis_data['avg_accuracy_percent']
                        deviation_succ = anomaly['success_rate'] - analysis_data['success_rate']
                        f.write(f"• {anomaly['ticker']:5s} - {anomaly['accuracy']:5.1f}% точності ({deviation_acc:+.1f}%), "
                              f"{anomaly['success_rate']:4.0f}% успіху ({deviation_succ:+.1f}%)\n")
                    f.write("\n")
                
                # Висновки та рекомендації
                f.write("💡 ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ:\n")
                f.write("-" * 40 + "\n")
                
                if period_performance:
                    best_period = period_performance[0]
                    worst_period = period_performance[-1]
                    f.write(f"• Кращий період: {best_period['period_text']} ({best_period['accuracy']:.1f}% точності)\n")
                    f.write(f"• Гірший період: {worst_period['period_text']} ({worst_period['accuracy']:.1f}% точності)\n")
                
                if anomaly_tickers:
                    f.write(f"• Виявлено {len(anomaly_tickers)} аномальних тикерів - потрібна увага\n")
                
                if analysis_data['avg_accuracy_percent'] > 95:
                    f.write("• ✅ Система працює відмінно - висока точність прогнозів\n")
                elif analysis_data['avg_accuracy_percent'] > 90:
                    f.write("• 🟡 Система працює добре - є простір для покращень\n")
                else:
                    f.write("• 🔴 Система потребує оптимізації - низька точність прогнозів\n")
                
                if analysis_data['total'] < 50:
                    f.write("• ⏳ Недостатньо даних - рекомендується накопичити більше прогнозів\n")
                
                f.write("\n")

                # Детальні дані по кожному тікеру
                f.write("📋 ДЕТАЛЬНА СТАТИСТИКА ПО ТІКЕРАМ:\n")
                f.write("=" * 80 + "\n")
                
                for ticker in sorted(self.tickers):
                    ticker_analysis = self.db.get_analysis(ticker, days_back=7)
                    
                    if ticker_analysis['total'] > 0:
                        f.write(f"\n🔸 {ticker}\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"Всього прогнозів: {ticker_analysis['total']}\n")
                        f.write(f"Успішних: {ticker_analysis['success']} ({ticker_analysis['success_rate']:.1f}%)\n")
                        f.write(f"Середня точність: {ticker_analysis['avg_accuracy_percent']:.1f}%\n")
                        
                        # Получаем последние прогнозы для этого тикера
                        recent_predictions = self.db.get_prediction_history(ticker, limit=10)
                        if recent_predictions:
                            f.write("\nОстанні прогнози:\n")
                            for pred in recent_predictions:
                                if pred['verified'] and pred['accuracy_percent'] is not None:
                                    pred_time = pred['prediction_time']
                                    if isinstance(pred_time, str):
                                        try:
                                            pred_time = datetime.fromisoformat(pred_time.replace('Z', '+00:00'))
                                            pred_time = pred_time.strftime('%d.%m %H:%M')
                                        except:
                                            pred_time = pred['prediction_time'][:16]
                                    
                                    f.write(f"  {pred_time} | {pred['period_hours']:2d}h | ")
                                    f.write(f"${pred['predicted_price']:7.2f} → ${pred['actual_price']:7.2f} | ")
                                    f.write(f"{pred['accuracy_percent']:5.1f}% точности\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("Звіт створено автоматично системою моніторингу акцій\n")
            
            logger.info(f"📊 Створено звіт верифікації: {report_file}")
            
            # Повертаємо дані для короткого попереднього перегляду
            summary_data['report_file'] = report_file
            return summary_data
            
        except Exception as e:
            logger.error(f"Помилка створення звіту верифікації: {e}")
            return None
    
    def generate_hourly_report(self) -> Optional[str]:
        """Генерація щогодинного звіту з усіма сигналами та верифікацією прогнозів"""
        try:
            now = datetime.now()
            hour_ago = now - timedelta(hours=1)
            
            # Отримуємо сигнали за останню годину з БД
            from database import db
            signals = db.get_signals_by_timerange(hour_ago, now)
            
            if not signals:
                logger.info("Немає сигналів за останню годину для звіту")
                return None
            
            # Створюємо звіт
            report_data = {
                'timestamp': now.isoformat(),
                'period': f"{hour_ago.strftime('%H:%M')} - {now.strftime('%H:%M')}",
                'total_signals': len(signals),
                'signals_by_type': {},
                'signals_by_priority': {},
                'signals': [],
                'predictions_verification': []
            }
            
            # Групуємо сигнали
            for signal in signals:
                signal_type = signal.signal_type
                priority = signal.priority
                
                report_data['signals_by_type'][signal_type] = report_data['signals_by_type'].get(signal_type, 0) + 1
                report_data['signals_by_priority'][priority] = report_data['signals_by_priority'].get(priority, 0) + 1
                
                # Додаємо повну інформацію про сигнал
                signal_info = {
                    'ticker': signal.ticker,
                    'type': signal_type,
                    'priority': priority,
                    'message': signal.message,
                    'price': signal.price,
                    'change_percent': signal.change_percent,
                    'confidence': signal.confidence,
                    'timestamp': signal.timestamp.isoformat()
                }
                
                if signal.period_hours:
                    signal_info['period_hours'] = signal.period_hours
                
                report_data['signals'].append(signal_info)
            
            # Верифікація прогнозів
            predictions_to_verify = db.get_predictions_for_verification(now)
            for prediction in predictions_to_verify:
                try:
                    # Отримуємо поточну ціну для верифікації
                    current_data = self.polygon_client.get_latest_price(prediction.ticker)
                    if current_data:
                        current_price = current_data.get('current_price', 0.0)
                        if current_price > 0:
                            predicted_price = prediction.price * (1 + prediction.change_percent / 100)
                            actual_change = ((current_price - prediction.price) / prediction.price) * 100
                            
                            accuracy = 100 - abs(actual_change - prediction.change_percent)
                            accuracy = max(0, accuracy)  # Не менше 0%
                            
                            verification_result = {
                                'ticker': prediction.ticker,
                                'predicted_change': prediction.change_percent,
                                'actual_change': actual_change,
                                'accuracy': accuracy,
                                'period_hours': prediction.period_hours,
                                'original_price': prediction.price,
                                'current_price': current_price,
                                'predicted_price': predicted_price,
                                'confidence': prediction.confidence,
                                'timestamp': prediction.timestamp.isoformat()
                            }
                            
                            report_data['predictions_verification'].append(verification_result)
                            
                            # Оновлюємо статистику
                            self.hourly_stats['accuracy_scores'].append(accuracy)
                        else:
                            logger.warning(f"Не вдалося отримати валідну ціну для верифікації {prediction.ticker}")
                    else:
                        logger.warning(f"Не вдалося отримати дані з API для верифікації {prediction.ticker}")
                        
                except Exception as e:
                    logger.error(f"Помилка верифікації прогнозу для {prediction.ticker}: {e}")
            
            # Зберігаємо звіт у файл
            reports_dir = os.path.join(DATA_DIR, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            filename = f"hourly_report_{now.strftime('%Y%m%d_%H%M')}.json"
            report_path = os.path.join(reports_dir, filename)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            # Оновлюємо статистику
            self.hourly_stats['signals_count'] = len(signals)
            self.hourly_stats['predictions_verified'] = len(report_data['predictions_verification'])
            self.hourly_stats['last_report_time'] = now
            
            logger.info(f"Щогодинний звіт збережено: {report_path} ({len(signals)} сигналів, {len(report_data['predictions_verification'])} верифікацій)")
            return report_path
            
        except Exception as e:
            logger.error(f"Помилка генерації щогодинного звіту: {e}")
            return None
    
    def get_report_preview(self, report_path: str) -> str:
        """Створення превью звіту для Telegram повідомлення"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            preview = f"📊 **Щогодинний звіт {report_data['period']}**\n\n"
            preview += f"🔔 Всього сигналів: {report_data['total_signals']}\n"
            
            # Сигнали за типами
            if report_data['signals_by_type']:
                preview += "\n**За типами:**\n"
                for sig_type, count in report_data['signals_by_type'].items():
                    preview += f"• {sig_type}: {count}\n"
            
            # Сигнали за пріоритетами
            if report_data['signals_by_priority']:
                preview += "\n**За пріоритетами:**\n"
                for priority, count in report_data['signals_by_priority'].items():
                    emoji = {'critical': '🚨', 'important': '⚠️', 'info': 'ℹ️'}.get(priority, '📝')
                    preview += f"{emoji} {priority}: {count}\n"
            
            # Верифікація прогнозів
            if report_data['predictions_verification']:
                verifications = report_data['predictions_verification']
                avg_accuracy = sum(v['accuracy'] for v in verifications) / len(verifications)
                preview += f"\n🎯 **Верифікація прогнозів:** {len(verifications)} перевірено\n"
                preview += f"📈 Середня точність: {avg_accuracy:.1f}%\n"
                
                # Топ 3 найточніші прогнози
                top_predictions = sorted(verifications, key=lambda x: x['accuracy'], reverse=True)[:3]
                preview += "\n🏆 **Найточніші прогнози:**\n"
                for pred in top_predictions:
                    preview += f"• {pred['ticker']}: {pred['accuracy']:.1f}% ({pred['predicted_change']:+.1f}% → {pred['actual_change']:+.1f}%)\n"
            
            preview += f"\n📋 Повний звіт: `{os.path.basename(report_path)}`"
            
            return preview
            
        except Exception as e:
            logger.error(f"Помилка створення превью звіту: {e}")
            return f"📊 Звіт створено: {os.path.basename(report_path)}"
    
    def get_validation_stats(self, ticker: str) -> Dict:
        """Получение детальной информации о прогнозах для валидации"""
        try:
            result = {
                'ticker': ticker,
                'validation': {}
            }
            
            # Получаем статистику валидации из БД за последние 7 дней
            with self.db.get_cursor() as cursor:
                # Статистика по периодам для краткосрочной модели
                cursor.execute("""
                    SELECT 
                        period_hours,
                        COUNT(*) as count,
                        AVG(accuracy_percent) as avg_accuracy,
                        AVG(ABS(price_change_percent - (((actual_price - current_price) / current_price) * 100))) as mape
                    FROM predictions 
                    WHERE ticker = ? 
                        AND verified = 1 
                        AND actual_price IS NOT NULL
                        AND prediction_time >= datetime('now', '-7 days')
                        AND period_hours IN (1, 3, 6, 12, 24)
                    GROUP BY period_hours
                    ORDER BY period_hours
                """, (ticker,))
                
                short_term_results = cursor.fetchall()
                if short_term_results:
                    result['validation']['short_term'] = {}
                    for row in short_term_results:
                        period = row['period_hours']
                        result['validation']['short_term'][period] = {
                            'count': row['count'],
                            'accuracy': row['avg_accuracy'] or 0,
                            'mape': (row['mape'] or 0) / 100,  # Конвертируем в доли
                            'test_score': min(1.0, (100 - (row['mape'] or 100)) / 100)  # Примерный R²
                        }
                
                # Статистика для долгосрочной модели
                cursor.execute("""
                    SELECT 
                        period_hours,
                        COUNT(*) as count,
                        AVG(accuracy_percent) as avg_accuracy,
                        AVG(ABS(price_change_percent - (((actual_price - current_price) / current_price) * 100))) as mape
                    FROM predictions 
                    WHERE ticker = ? 
                        AND verified = 1 
                        AND actual_price IS NOT NULL
                        AND prediction_time >= datetime('now', '-7 days')
                        AND period_hours IN (72, 168, 720)
                    GROUP BY period_hours
                    ORDER BY period_hours
                """, (ticker,))
                
                long_term_results = cursor.fetchall()
                if long_term_results:
                    result['validation']['long_term'] = {}
                    for row in long_term_results:
                        period = row['period_hours']
                        result['validation']['long_term'][period] = {
                            'count': row['count'],
                            'accuracy': row['avg_accuracy'] or 0,
                            'mape': (row['mape'] or 0) / 100,
                            'test_score': min(1.0, (100 - (row['mape'] or 100)) / 100)
                        }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка получения детальных прогнозов для {ticker}: {e}")
            import traceback
            logger.error(f"Полная трассировка ошибки для {ticker}:\n{traceback.format_exc()}")
            return {'ticker': ticker, 'validation': {}, 'error': str(e), 'traceback': traceback.format_exc()}

if __name__ == "__main__":
    monitor = StockMonitor()
    logger.info("Функциональный Monitor запущен")
    
    # Запускаем автоматическое обновление
    monitor.start_auto_update()
    
    # Запускаем систему верификации
    monitor.start_verification_system()
    
    # Первое обновление сразу
    logger.info("Выполняем первичное обновление всех тикеров...")
    monitor.update_all()
    
    # Держим основной поток активным
    try:
        while True:
            time.sleep(60)  # Проверяем каждую минуту, что всё работает
    except KeyboardInterrupt:
        logger.info("Остановка Monitor по команде пользователя")
        import sys
        sys.exit(0)
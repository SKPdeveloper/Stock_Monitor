"""
Database module for Stock Monitor with SQLite
"""
import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    ticker: str
    prediction_time: datetime
    target_time: datetime
    current_price: float
    predicted_price: float
    price_change_percent: float
    confidence: float
    period_hours: int
    model_type: str
    features_hash: str
    actual_price: Optional[float] = None
    actual_time: Optional[datetime] = None
    status: str = 'pending'
    accuracy_percent: Optional[float] = None
    verified: bool = False

@dataclass
class SignalRecord:
    ticker: str
    signal_type: str
    priority: str
    message: str
    timestamp: datetime
    price: float
    change_percent: float
    volume: Optional[int] = None
    confidence: Optional[float] = None
    period_hours: Optional[int] = None
    sent: bool = False
    user_filtered: bool = False

@dataclass
class ModelPerformance:
    ticker: str
    model_type: str
    period_hours: int
    accuracy_1d: float
    accuracy_7d: float
    accuracy_30d: float
    prediction_count: int
    last_updated: datetime
    confidence_threshold: float

class DatabaseManager:
    """Enhanced database manager with connection pooling and thread safety"""
    
    def __init__(self, db_path: str = "data/stock_monitor.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.create_tables()
        
    def get_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                isolation_level=None  # autocommit mode
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA foreign_keys=ON")
        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """Context manager for database operations"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            try:
                cursor.close()
            except Exception as e:
                logger.warning(f"Error closing cursor: {e}")

    def create_tables(self):
        """Create all necessary tables"""
        with self.get_cursor() as cursor:
            # Predictions table with enhanced tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    prediction_time TIMESTAMP NOT NULL,
                    target_time TIMESTAMP NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    price_change_percent REAL NOT NULL,
                    confidence REAL NOT NULL,
                    period_hours INTEGER NOT NULL,
                    model_type TEXT NOT NULL,
                    features_hash TEXT NOT NULL,
                    actual_price REAL,
                    actual_time TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    accuracy_percent REAL,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes separately
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_time ON predictions(ticker, prediction_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_target_time ON predictions(target_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON predictions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verified ON predictions(verified)")
            
            # Signals table with advanced filtering
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    change_percent REAL NOT NULL,
                    volume INTEGER,
                    confidence REAL,
                    sent BOOLEAN DEFAULT FALSE,
                    user_filtered BOOLEAN DEFAULT FALSE,
                    dedup_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes separately
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker_time ON signals(ticker, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_priority ON signals(priority)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_sent ON signals(sent)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_dedup ON signals(dedup_hash)")
            
            # Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    period_hours INTEGER NOT NULL,
                    accuracy_1d REAL DEFAULT 0,
                    accuracy_7d REAL DEFAULT 0,
                    accuracy_30d REAL DEFAULT 0,
                    prediction_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_threshold REAL DEFAULT 0.5,
                    UNIQUE(ticker, model_type, period_hours)
                )
            """)
            
            # Cache table for API responses
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_cache (
                    cache_key TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ttl_hours INTEGER DEFAULT 1
                )
            """)
            
            # Create index separately
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_timestamp ON api_cache(timestamp)")
            
            # User preferences
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    signal_filter TEXT DEFAULT 'all',
                    min_price_change REAL DEFAULT 0.5,
                    time_period_filter INTEGER DEFAULT 0,
                    min_confidence REAL DEFAULT 0.6,
                    enabled_signal_types TEXT DEFAULT '[]',
                    notification_hours TEXT DEFAULT '[]',
                    max_signals_per_hour INTEGER DEFAULT 10,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Migration: додаємо поле time_period_filter якщо його ще немає
            try:
                cursor.execute("ALTER TABLE user_preferences ADD COLUMN time_period_filter INTEGER DEFAULT 0")
            except sqlite3.OperationalError:
                pass  # Поле вже існує
            
            # Migration: додаємо поле period_hours до signals якщо його ще немає
            try:
                cursor.execute("ALTER TABLE signals ADD COLUMN period_hours INTEGER")
            except sqlite3.OperationalError:
                pass  # Поле вже існує
            
            # Migration: додаємо поле user_filtered до signals якщо його ще немає
            try:
                cursor.execute("ALTER TABLE signals ADD COLUMN user_filtered BOOLEAN DEFAULT FALSE")
            except sqlite3.OperationalError:
                pass  # Поле вже існує
            
            # Tickers management
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_update TIMESTAMP,
                    prediction_enabled BOOLEAN DEFAULT TRUE,
                    signals_enabled BOOLEAN DEFAULT TRUE
                )
            """)

    # ====== PREDICTIONS METHODS ======
    
    def save_prediction(self, prediction: PredictionRecord) -> int:
        """Save prediction to database"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO predictions (
                    ticker, prediction_time, target_time, current_price,
                    predicted_price, price_change_percent, confidence,
                    period_hours, model_type, features_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.ticker,
                prediction.prediction_time,
                prediction.target_time,
                prediction.current_price,
                prediction.predicted_price,
                prediction.price_change_percent,
                prediction.confidence,
                prediction.period_hours,
                prediction.model_type,
                prediction.features_hash
            ))
            return cursor.lastrowid

    def update_prediction_result(self, prediction_id: int, actual_price: float, 
                               actual_time: datetime) -> bool:
        """Update prediction with actual result"""
        with self.get_cursor() as cursor:
            # Calculate accuracy
            cursor.execute("""
                SELECT predicted_price, current_price FROM predictions WHERE id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                return False
                
            predicted_price = result['predicted_price']
            current_price = result['current_price']
            
            # ИСПРАВЛЕНО: Улучшенный расчет точности
            # Используем простую процентную ошибку относительно предсказанной цены
            if predicted_price == 0:
                accuracy = 0.0
            else:
                # Процентная ошибка = abs(фактическая - предсказанная) / предсказанная * 100
                percent_error = abs(actual_price - predicted_price) / predicted_price * 100
                # Точность = 100% - процентная ошибка (ограничиваем от 0 до 100%)
                accuracy = max(0.0, min(100.0, 100.0 - percent_error))
            
            cursor.execute("""
                UPDATE predictions 
                SET actual_price = ?, actual_time = ?, accuracy_percent = ?,
                    status = 'completed', verified = TRUE
                WHERE id = ?
            """, (actual_price, actual_time, accuracy, prediction_id))
            
            return cursor.rowcount > 0

    def get_pending_predictions(self, before_time: datetime) -> List[Dict]:
        """Get predictions that need verification"""
        with self.get_cursor() as cursor:
            # Убираем ограничение на сегодняшний день - проверяем все прогнозы, чье время цели прошло
            cursor.execute("""
                SELECT * FROM predictions 
                WHERE status = 'pending' 
                  AND target_time <= ?
                ORDER BY target_time ASC
                LIMIT 1000
            """, (before_time,))
            
            return [dict(row) for row in cursor.fetchall()]

    def get_model_accuracy(self, ticker: str, model_type: str, 
                          period_hours: int, days: int = 30) -> float:
        """Get model accuracy for specific parameters"""
        with self.get_cursor() as cursor:
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT AVG(accuracy_percent) as avg_accuracy
                FROM predictions 
                WHERE ticker = ? AND model_type = ? AND period_hours = ?
                  AND verified = TRUE AND prediction_time >= ?
            """, (ticker, model_type, period_hours, since_date))
            
            result = cursor.fetchone()
            return result['avg_accuracy'] if result['avg_accuracy'] else 0.0

    # ====== SIGNALS METHODS ======
    
    def save_signal(self, signal: SignalRecord) -> int:
        """Save signal to database"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO signals (
                    ticker, signal_type, priority, message, timestamp,
                    price, change_percent, volume, confidence, period_hours, dedup_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.ticker, signal.signal_type, signal.priority,
                signal.message, signal.timestamp, signal.price,
                signal.change_percent, signal.volume, signal.confidence,
                signal.period_hours, self._generate_dedup_hash(signal)
            ))
            return cursor.lastrowid

    def _generate_dedup_hash(self, signal: SignalRecord) -> str:
        """Generate deduplication hash for signal"""
        import hashlib
        content = f"{signal.ticker}_{signal.signal_type}_{signal.priority}_{round(signal.change_percent, 1)}"
        return hashlib.md5(content.encode()).hexdigest()

    def is_duplicate_signal(self, signal: SignalRecord, window_minutes: int = 30) -> bool:
        """Check if signal is duplicate within time window"""
        with self.get_cursor() as cursor:
            since_time = signal.timestamp - timedelta(minutes=window_minutes)
            dedup_hash = self._generate_dedup_hash(signal)
            
            cursor.execute("""
                SELECT COUNT(*) as count FROM signals 
                WHERE dedup_hash = ? AND timestamp >= ?
            """, (dedup_hash, since_time))
            
            return cursor.fetchone()['count'] > 0

    def get_filtered_signals(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get filtered signals based on user preferences"""
        with self.get_cursor() as cursor:
            # Get user preferences
            cursor.execute("""
                SELECT * FROM user_preferences WHERE user_id = ?
            """, (user_id,))
            
            prefs = cursor.fetchone()
            if not prefs:
                # Default preferences
                min_change = 0.5
                min_confidence = 0.6
                signal_types = []
            else:
                min_change = prefs['min_price_change']
                min_confidence = prefs['min_confidence']
                signal_types = json.loads(prefs['enabled_signal_types'])
            
            # Build query with filters
            query = """
                SELECT * FROM signals 
                WHERE sent = FALSE 
                  AND abs(change_percent) >= ?
                  AND (confidence IS NULL OR confidence >= ?)
            """
            params = [min_change, min_confidence]
            
            if signal_types:
                placeholders = ','.join(['?' for _ in signal_types])
                query += f" AND signal_type IN ({placeholders})"
                params.extend(signal_types)
            
            query += " ORDER BY timestamp DESC, priority DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def mark_signal_sent(self, signal_id: int):
        """Mark signal as sent"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE signals SET sent = TRUE WHERE id = ?
            """, (signal_id,))

    # ====== CACHE METHODS ======
    
    def get_cached_data(self, cache_key: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Get cached data if not expired"""
        with self.get_cursor() as cursor:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            cursor.execute("""
                SELECT data FROM api_cache 
                WHERE cache_key = ? AND timestamp >= ?
            """, (cache_key, cutoff_time))
            
            result = cursor.fetchone()
            if result:
                return json.loads(result['data'])
            return None

    def set_cached_data(self, cache_key: str, data: Dict, ttl_hours: int = 1):
        """Cache data with TTL"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO api_cache (cache_key, data, timestamp, ttl_hours)
                VALUES (?, ?, ?, ?)
            """, (cache_key, json.dumps(data), datetime.now(), ttl_hours))

    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM api_cache 
                WHERE timestamp < datetime('now', '-' || ttl_hours || ' hours')
            """)

    # ====== TICKERS METHODS ======
    
    def get_active_tickers(self) -> List[str]:
        """Get list of active tickers"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT symbol FROM tickers WHERE active = TRUE
                ORDER BY symbol
            """)
            return [row['symbol'] for row in cursor.fetchall()]

    def add_ticker(self, symbol: str, name: str = None, sector: str = None) -> bool:
        """Add new ticker"""
        with self.get_cursor() as cursor:
            try:
                cursor.execute("""
                    INSERT INTO tickers (symbol, name, sector)
                    VALUES (?, ?, ?)
                """, (symbol.upper(), name, sector))
                return True
            except sqlite3.IntegrityError:
                return False

    def remove_ticker(self, symbol: str) -> bool:
        """Deactivate ticker"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE tickers SET active = FALSE WHERE symbol = ?
            """, (symbol.upper(),))
            return cursor.rowcount > 0

    # ====== ANALYTICS METHODS ======
    
    def get_prediction_stats(self, days: int = 30) -> Dict:
        """Get prediction statistics"""
        with self.get_cursor() as cursor:
            since_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN verified = TRUE THEN 1 END) as verified_predictions,
                    AVG(CASE WHEN verified = TRUE THEN accuracy_percent END) as avg_accuracy,
                    AVG(confidence) as avg_confidence
                FROM predictions 
                WHERE prediction_time >= ?
            """, (since_date,))
            
            result = cursor.fetchone()
            return dict(result) if result else {
                'total_predictions': 0,
                'verified_predictions': 0,
                'avg_accuracy': 0.0,
                'avg_confidence': 0.5
            }

    def get_top_performing_models(self, limit: int = 10) -> List[Dict]:
        """Get best performing models"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT ticker, model_type, period_hours, accuracy_30d, prediction_count
                FROM model_performance 
                WHERE prediction_count >= 10
                ORDER BY accuracy_30d DESC, prediction_count DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]

    def update_model_performance(self, ticker: str, model_type: str, period_hours: int):
        """Update model performance metrics"""
        with self.get_cursor() as cursor:
            # Calculate accuracies for different periods
            for days, column in [(1, 'accuracy_1d'), (7, 'accuracy_7d'), (30, 'accuracy_30d')]:
                since_date = datetime.now() - timedelta(days=days)
                
                cursor.execute("""
                    SELECT AVG(accuracy_percent) as avg_accuracy, COUNT(*) as count
                    FROM predictions 
                    WHERE ticker = ? AND model_type = ? AND period_hours = ?
                      AND verified = TRUE AND prediction_time >= ?
                """, (ticker, model_type, period_hours, since_date))
                
                result = cursor.fetchone()
                accuracy = result['avg_accuracy'] if result['avg_accuracy'] else 0.0
                count = result['count']
                
                cursor.execute(f"""
                    INSERT OR REPLACE INTO model_performance 
                    (ticker, model_type, period_hours, {column}, prediction_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, model_type, period_hours, accuracy, count, datetime.now()))

    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to maintain performance"""
        with self.get_cursor() as cursor:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean old predictions
            cursor.execute("""
                DELETE FROM predictions WHERE prediction_time < ?
            """, (cutoff_date,))
            
            # Clean old signals
            cursor.execute("""
                DELETE FROM signals WHERE timestamp < ?
            """, (cutoff_date,))
            
            logger.info(f"Cleaned up data older than {days} days")

    def vacuum_database(self):
        """Optimize database"""
        with self.get_cursor() as cursor:
            cursor.execute("VACUUM")
            logger.info("Database optimized")
    
    # ====== PREDICTION TRACKER COMPATIBILITY METHODS ======
    
    def get_analysis(self, ticker: Optional[str] = None, days_back: int = 30) -> Dict:
        """Get prediction analysis compatible with telegram bot"""
        try:
            with self.get_cursor() as cursor:
                # Если days_back == 1, то берем только сегодняшние данные
                if days_back == 1:
                    since_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    since_date = datetime.now() - timedelta(days=days_back)
                
                # Base query
                base_query = """
                    SELECT 
                        ticker,
                        accuracy_percent,
                        period_hours,
                        price_change_percent,
                        confidence,
                        verified,
                        prediction_time
                    FROM predictions 
                    WHERE prediction_time >= ? AND verified = TRUE
                """
                params = [since_date]
                
                # Add ticker filter if specified
                if ticker:
                    base_query += " AND ticker = ?"
                    params.append(ticker)
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                if not results:
                    return {
                        'total': 0, 'success': 0, 'partial': 0, 'failed': 0,
                        'success_rate': 0.0, 'avg_accuracy_percent': 0.0,
                        'period_stats': {},
                        'best_predictions': [],
                        'worst_predictions': []
                    }
                
                # Calculate statistics
                total = len(results)
                success = 0  # <2% error
                partial = 0  # 2-5% error
                failed = 0   # >5% error
                total_accuracy = 0
                period_stats = {}
                best_predictions = []
                worst_predictions = []
                
                for row in results:
                    accuracy = row['accuracy_percent'] or 0
                    period = row['period_hours']
                    # ИСПРАВЛЕНО: accuracy_percent уже содержит процент точности (0-100%)
                    # Не нужно вычитать из 100, используем accuracy напрямую
                    error_rate = 100 - accuracy  # Это правильно - конвертируем в процент ошибки
                    
                    total_accuracy += accuracy
                    
                    # Categorize predictions - используем accuracy, а не error_rate!
                    if accuracy >= 98:  # >98% точности = <2% ошибки
                        success += 1
                        period_success_key = 'success'
                    elif accuracy >= 95:  # >95% точности = <5% ошибки
                        partial += 1
                        period_success_key = 'partial'
                    else:  # <95% точности = >5% ошибки
                        failed += 1
                        period_success_key = 'failed'
                    
                    # Period statistics
                    if period not in period_stats:
                        period_stats[period] = {
                            'total': 0, 'success': 0, 'partial': 0, 'failed': 0, 
                            'accuracy': 0, 'avg_accuracy': 0
                        }
                    period_stats[period]['total'] += 1
                    period_stats[period][period_success_key] += 1
                    period_stats[period]['accuracy'] += accuracy
                    
                    # Track best and worst predictions for display
                    pred_date = row['prediction_time']
                    if isinstance(pred_date, str):
                        try:
                            pred_date = datetime.fromisoformat(pred_date.replace('Z', '+00:00'))
                        except:
                            pred_date = 'Recent'
                    
                    if isinstance(pred_date, datetime):
                        pred_date = pred_date.strftime('%d.%m')
                    
                    pred_info = {
                        'accuracy': error_rate,  # Для отображения используем error_rate
                        'ticker': row['ticker'],
                        'period': f"{period}h",
                        'date': pred_date
                    }
                    
                    if len(best_predictions) < 10:
                        best_predictions.append(pred_info)
                    else:
                        # Keep only top predictions (lowest error rates)
                        worst_in_best = max(best_predictions, key=lambda x: x['accuracy'])
                        if error_rate < worst_in_best['accuracy']:
                            best_predictions.remove(worst_in_best)
                            best_predictions.append(pred_info)
                    
                    if len(worst_predictions) < 10:
                        worst_predictions.append(pred_info)
                    else:
                        # Keep only worst predictions (highest error rates)
                        best_in_worst = min(worst_predictions, key=lambda x: x['accuracy'])
                        if error_rate > best_in_worst['accuracy']:
                            worst_predictions.remove(best_in_worst)
                            worst_predictions.append(pred_info)
                
                # Calculate averages for periods
                for period in period_stats:
                    if period_stats[period]['total'] > 0:
                        period_stats[period]['avg_accuracy'] = period_stats[period]['accuracy'] / period_stats[period]['total']
                
                # Sort predictions
                best_predictions.sort(key=lambda x: x['accuracy'])  # Lowest error first
                worst_predictions.sort(key=lambda x: x['accuracy'], reverse=True)  # Highest error first
                
                return {
                    'total': total,
                    'success': success,
                    'partial': partial,
                    'failed': failed,
                    'success_rate': (success / total * 100) if total > 0 else 0.0,
                    'avg_accuracy_percent': (total_accuracy / total) if total > 0 else 0.0,
                    'period_stats': period_stats,
                    'best_predictions': best_predictions[:3],  # Top 3 best
                    'worst_predictions': worst_predictions[:3]  # Top 3 worst
                }
                
        except Exception as e:
            logger.error(f"Error getting analysis: {e}")
            return {
                'total': 0, 'success': 0, 'partial': 0, 'failed': 0,
                'success_rate': 0.0, 'avg_accuracy_percent': 0.0,
                'period_stats': {},
                'best_predictions': [],
                'worst_predictions': []
            }
    
    def get_prediction_history(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Get prediction history for ticker"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE ticker = ?
                    ORDER BY prediction_time DESC 
                    LIMIT ?
                """, (ticker, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []
    
    def get_statistics_summary(self) -> Dict:
        """Get overall statistics summary"""
        try:
            with self.get_cursor() as cursor:
                # Get general statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        COUNT(CASE WHEN verified = TRUE THEN 1 END) as verified_predictions,
                        AVG(CASE WHEN verified = TRUE THEN accuracy_percent END) as avg_accuracy,
                        COUNT(DISTINCT ticker) as total_tickers
                    FROM predictions 
                    WHERE prediction_time >= datetime('now', '-7 days')
                """)
                
                result = cursor.fetchone()
                
                return {
                    'total_predictions': result['total_predictions'] or 0,
                    'verified_predictions': result['verified_predictions'] or 0,
                    'avg_accuracy': result['avg_accuracy'] or 0.0,
                    'total_tickers': result['total_tickers'] or 0
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics summary: {e}")
            return {
                'total_predictions': 0,
                'verified_predictions': 0, 
                'avg_accuracy': 0.0,
                'total_tickers': 0
            }
    
    def cleanup_old_predictions(self, days: int = 90):
        """Cleanup old predictions - alias for cleanup_old_data"""
        self.cleanup_old_data(days)
    
    def get_ticker_info(self, ticker: str) -> Dict:
        """Get detailed ticker information for enhanced UI"""
        try:
            with self.get_cursor() as cursor:
                # Get recent accuracy
                cursor.execute("""
                    SELECT AVG(accuracy_percent) as avg_accuracy
                    FROM predictions 
                    WHERE ticker = ? AND verified = TRUE 
                      AND prediction_time >= datetime('now', '-1 days')
                """, (ticker,))
                result = cursor.fetchone()
                accuracy = result['avg_accuracy'] if result['avg_accuracy'] else 0.0
                
                # Get next predictions (short and long term)
                cursor.execute("""
                    SELECT 
                        period_hours,
                        predicted_price,
                        target_time,
                        confidence,
                        price_change_percent
                    FROM predictions 
                    WHERE ticker = ? AND status = 'pending'
                      AND target_time > datetime('now')
                    ORDER BY target_time ASC
                    LIMIT 10
                """, (ticker,))
                predictions = cursor.fetchall()
                
                short_pred = None
                long_pred = None
                
                for pred in predictions:
                    if pred['period_hours'] <= 24 and not short_pred:
                        short_pred = dict(pred)
                    elif pred['period_hours'] > 24 and not long_pred:
                        long_pred = dict(pred)
                
                # Calculate volatility from recent price changes
                cursor.execute("""
                    SELECT price_change_percent
                    FROM predictions 
                    WHERE ticker = ? AND verified = TRUE
                      AND prediction_time >= datetime('now', '-7 days')
                """, (ticker,))
                changes = [row['price_change_percent'] for row in cursor.fetchall()]
                volatility = np.std(changes) if changes else 0.0
                
                return {
                    'accuracy': accuracy,
                    'volatility': volatility,
                    'short_prediction': short_pred,
                    'long_prediction': long_pred,
                    'prediction_count': len(predictions)
                }
                
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {e}")
            return {
                'accuracy': 0.0,
                'volatility': 0.0,
                'short_prediction': None,
                'long_prediction': None,
                'prediction_count': 0
            }
    
    def get_enhanced_ticker_list(self, tickers_list: List[str] = None, sort_by: str = 'symbol', filter_by: str = 'all') -> List[Dict]:
        """Get enhanced ticker list with additional info for UI"""
        try:
            # Если список тикеров не передан, получаем из базы данных
            if tickers_list is None:
                tickers = self.get_active_tickers()
            else:
                tickers = tickers_list
                
            enhanced_list = []
            
            for ticker in tickers:
                info = self.get_ticker_info(ticker)
                enhanced_list.append({
                    'symbol': ticker,
                    'accuracy': info['accuracy'],
                    'volatility': info['volatility'],
                    'short_prediction': info['short_prediction'],
                    'long_prediction': info['long_prediction'],
                    'prediction_count': info['prediction_count']
                })
            
            # Apply filtering
            if filter_by == 'high_accuracy':
                enhanced_list = [t for t in enhanced_list if t['accuracy'] > 80]
            elif filter_by == 'high_volatility':
                enhanced_list = [t for t in enhanced_list if t['volatility'] > 2.0]
            elif filter_by == 'has_predictions':
                enhanced_list = [t for t in enhanced_list if t['prediction_count'] > 0]
            
            # Apply sorting
            if sort_by == 'accuracy':
                enhanced_list.sort(key=lambda x: x['accuracy'], reverse=True)
            elif sort_by == 'volatility':
                enhanced_list.sort(key=lambda x: x['volatility'], reverse=True)
            elif sort_by == 'predictions':
                enhanced_list.sort(key=lambda x: x['prediction_count'], reverse=True)
            else:  # symbol
                enhanced_list.sort(key=lambda x: x['symbol'])
            
            return enhanced_list
            
        except Exception as e:
            logger.error(f"Error getting enhanced ticker list: {e}")
            return []
    
    def get_filtered_signals_enhanced(self, signal_type: str = 'all', 
                                    min_change: float = 0.0, 
                                    priority: str = 'all',
                                    limit: int = 50,
                                    include_sent: bool = False,
                                    hours_back: int = 24) -> List[Dict]:
        """Get enhanced filtered signals for UI"""
        try:
            with self.get_cursor() as cursor:
                query = """
                    SELECT * FROM signals 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours_back)
                
                if not include_sent:
                    query += " AND sent = FALSE"
                params = []
                
                # Apply filters
                if signal_type != 'all':
                    query += " AND signal_type = ?"
                    params.append(signal_type)
                
                if min_change > 0:
                    query += " AND abs(change_percent) >= ?"
                    params.append(min_change)
                
                if priority != 'all':
                    query += " AND priority = ?"
                    params.append(priority)
                
                query += " ORDER BY timestamp DESC, priority DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting filtered signals: {e}")
            return []
    
    def get_signal_types(self) -> List[str]:
        """Get available signal types for filtering"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT signal_type FROM signals 
                    WHERE timestamp >= datetime('now', '-7 days')
                    ORDER BY signal_type
                """)
                return [row['signal_type'] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting signal types: {e}")
            return []
    
    def get_signals_by_timerange(self, start_time: datetime, end_time: datetime) -> List[SignalRecord]:
        """Отримання сигналів за часовим діапазоном"""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT * FROM signals 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp DESC
                """, (start_time, end_time))
                
                signals = []
                for row in cursor.fetchall():
                    signal = SignalRecord(
                        ticker=row['ticker'],
                        signal_type=row['signal_type'],
                        priority=row['priority'],
                        message=row['message'],
                        timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else datetime.fromisoformat(row['timestamp']),
                        price=row['price'],
                        change_percent=row['change_percent'],
                        volume=row['volume'],
                        confidence=row['confidence'],
                        period_hours=row['period_hours'],
                        sent=row['sent'],
                        user_filtered=row['user_filtered']
                    )
                    signals.append(signal)
                
                return signals
                
        except Exception as e:
            logger.error(f"Помилка отримання сигналів за часовим діапазоном: {e}")
            return []
    
    def get_predictions_for_verification(self, current_time: datetime) -> List[PredictionRecord]:
        """Отримання прогнозів, які потрібно верифікувати"""
        try:
            with self.get_cursor() as cursor:
                # Отримуємо прогнози, чий час виконання настав і які ще не верифіковані
                cursor.execute("""
                    SELECT * FROM predictions 
                    WHERE target_time <= ? 
                      AND verified = FALSE 
                      AND status = 'pending'
                    ORDER BY target_time ASC
                    LIMIT 50
                """, (current_time,))
                
                predictions = []
                for row in cursor.fetchall():
                    prediction = PredictionRecord(
                        ticker=row['ticker'],
                        prediction_time=row['prediction_time'] if isinstance(row['prediction_time'], datetime) else datetime.fromisoformat(row['prediction_time']),
                        target_time=row['target_time'] if isinstance(row['target_time'], datetime) else datetime.fromisoformat(row['target_time']),
                        current_price=row['current_price'],
                        predicted_price=row['predicted_price'],
                        price_change_percent=row['price_change_percent'],
                        confidence=row['confidence'],
                        period_hours=row['period_hours'],
                        model_type=row['model_type'],
                        features_hash=row['features_hash'],
                        actual_price=row['actual_price'],
                        actual_time=row['actual_time'] if row['actual_time'] and isinstance(row['actual_time'], datetime) else (datetime.fromisoformat(row['actual_time']) if row['actual_time'] else None),
                        status=row['status'],
                        accuracy_percent=row['accuracy_percent'],
                        verified=row['verified']
                    )
                    predictions.append(prediction)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Помилка отримання прогнозів для верифікації: {e}")
            return []

# Global database instance
db = DatabaseManager()
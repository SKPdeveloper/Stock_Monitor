"""
SignalEngine - Інтелектуальна система генерації та аналізу сигналів
Оптимізована версія з покращеною дедуплікацією
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
from collections import defaultdict, deque
import hashlib
from database import db, SignalRecord

logger = logging.getLogger(__name__)


class SignalPriority(Enum):
   """Пріоритети сигналів"""
   CRITICAL = "critical"     # 🚨 Критичні
   IMPORTANT = "important"    # ⚠️ Важливі
   INFO = "info"             # ℹ️ Інформаційні
   
   def get_emoji(self) -> str:
       return {
           self.CRITICAL: "🚨",
           self.IMPORTANT: "⚠️", 
           self.INFO: "ℹ️"
       }[self]
   
   def get_weight(self) -> int:
       return {
           self.CRITICAL: 100,
           self.IMPORTANT: 50,
           self.INFO: 10
       }[self]


class SignalType(Enum):
   """Типи сигналів"""
   PRICE_CHANGE = "price_change"
   PREDICTION = "prediction"
   VOLUME_SPIKE = "volume_spike"
   TECHNICAL_BREAKOUT = "technical_breakout"
   TREND_REVERSAL = "trend_reversal"
   PATTERN = "pattern"
   DIVERGENCE = "divergence"
   SUPPORT_RESISTANCE = "support_resistance"
   MODEL_CONSENSUS = "model_consensus"
   VOLATILITY_CHANGE = "volatility_change"
   NEWS_IMPACT = "news_impact"
   PREDICTION_RESULT = "prediction_result"  # Для результатів перевірки прогнозів
   AGGREGATED = "aggregated"  # Для агрегованих сигналів
   # ПОКРАЩЕННЯ: Додаємо типи звітів, які не повинні фільтруватися
   HOURLY_REPORT = "hourly_report"       # Щогодинний звіт
   HALF_HOURLY_REPORT = "half_hourly_report"  # Півгодинний звіт
   DAILY_SUMMARY = "daily_summary"       # Денна зведена
   MARKET_STATUS = "market_status"       # Статус ринку


@dataclass
class Signal:
   """Клас для представлення сигналу"""
   type: SignalType
   priority: SignalPriority
   ticker: str
   timestamp: datetime
   
   # Основні дані
   message: str
   title: str
   
   # Деталі сигналу
   current_price: float
   price_change: float = 0
   price_change_percent: float = 0
   volume: int = 0
   
   # Метадані
   confidence: float = 0.5
   strength: int = 1  # 1-5 зірок
   factors: List[str] = field(default_factory=list)
   
   # Додаткові дані
   data: Dict[str, Any] = field(default_factory=dict)
   
   # Рекомендації
   action_hint: Optional[str] = None
   target_price: Optional[float] = None
   stop_loss: Optional[float] = None
   
   # Для дедуплікації
   signature: Optional[str] = None
   hash: Optional[str] = None
   
   # Для групування
   related_signals: List['Signal'] = field(default_factory=list)
   is_grouped: bool = False
   
   def to_dict(self) -> Dict:
       """Конвертація в словник для серіалізації"""
       return {
           'type': self.type.value,
           'priority': self.priority.value,
           'ticker': self.ticker,
           'timestamp': self.timestamp.isoformat(),
           'message': self.message,
           'title': self.title,
           'current_price': self.current_price,
           'price_change': self.price_change,
           'price_change_percent': self.price_change_percent,
           'volume': self.volume,
           'confidence': self.confidence,
           'strength': self.strength,
           'factors': self.factors,
           'data': self.data,
           'action_hint': self.action_hint,
           'target_price': self.target_price,
           'stop_loss': self.stop_loss
       }
   
   def get_formatted_message(self) -> str:
       """Форматоване повідомлення для відправки"""
       emoji = self.priority.get_emoji()
       stars = "⭐" * self.strength
       
       # Якщо це агрегований сигнал
       if self.type == SignalType.AGGREGATED and self.related_signals:
           return self._format_aggregated_message()
       
       msg = f"{emoji} **{self.title}**\n"
       msg += f"{self.message}\n\n"
       
       if self.factors:
           msg += "**Фактори:**\n"
           for factor in self.factors[:3]:  # Максимум 3 фактори
               msg += f"• {factor}\n"
           if len(self.factors) > 3:
               msg += f"• ... та ще {len(self.factors) - 3}\n"
           msg += "\n"
       
       msg += f"💪 Сила: {stars}\n"
       msg += f"🎯 Впевненість: {self.confidence:.0%}\n"
       
       if self.action_hint:
           msg += f"\n💡 {self.action_hint}\n"
       
       return msg
   
   def _format_aggregated_message(self) -> str:
       """Форматування агрегованого повідомлення"""
       emoji = self.priority.get_emoji()
       
       msg = f"{emoji} **{self.title}**\n\n"
       msg += f"📊 Згруповано {len(self.related_signals) + 1} сигналів:\n\n"
       
       # Групуємо за типами
       by_type = defaultdict(list)
       by_type[self.type].append(self)
       for sig in self.related_signals:
           by_type[sig.type].append(sig)
       
       # Виводимо згруповану інформацію
       for sig_type, signals in by_type.items():
           if sig_type == SignalType.PRICE_CHANGE:
               changes = [s.price_change_percent for s in signals]
               avg_change = sum(changes) / len(changes)
               msg += f"📈 Зміна ціни: {avg_change:+.1f}% (сер. з {len(signals)} сигналів)\n"
               
           elif sig_type == SignalType.PREDICTION:
               # Групуємо прогнози за періодами
               by_period = defaultdict(list)
               for s in signals:
                   period = s.data.get('period_hours', 0)
                   by_period[period].append(s.price_change_percent)
               
               if by_period:
                   msg += "🔮 Прогнози:\n"
                   for period, changes in sorted(by_period.items()):
                       avg = sum(changes) / len(changes)
                       period_text = f"{period}h" if period <= 24 else f"{period//24}d"
                       msg += f"  • {period_text}: {avg:+.1f}%\n"
                       
           elif sig_type == SignalType.VOLUME_SPIKE:
               volumes = [s.data.get('volume_ratio', 1) for s in signals]
               max_vol = max(volumes)
               msg += f"📊 Об'єм: до {max_vol:.1f}x від норми\n"
       
       # Загальні фактори
       all_factors = set()
       for sig in [self] + self.related_signals:
           all_factors.update(sig.factors)
       
       if all_factors:
           msg += "\n**Загальні фактори:**\n"
           for factor in list(all_factors)[:5]:
               msg += f"• {factor}\n"
           if len(all_factors) > 5:
               msg += f"• ... та ще {len(all_factors) - 5}\n"
       
       # Середня впевненість
       all_confidences = [self.confidence] + [s.confidence for s in self.related_signals]
       avg_confidence = sum(all_confidences) / len(all_confidences)
       msg += f"\n🎯 Середня впевненість: {avg_confidence:.0%}\n"
       
       return msg
   
   def calculate_hash(self) -> str:
        """Обчислення хешу сигналу для дедуплікації"""
        # Округлюємо числові значення для групування схожих
        rounded_change = round(self.price_change_percent, 1)
        rounded_price = round(self.current_price, 2)
    
        # Створюємо рядок для хешування
        hash_string = f"{self.ticker}_{self.type.value}_{self.priority.value}"
        hash_string += f"_{rounded_change}_{rounded_price}"
    
        # Додаємо період для прогнозів
        if self.type == SignalType.PREDICTION:
            period = self.data.get('period_hours', 0)
            hash_string += f"_{period}"
    
        # ВИПРАВЛЕННЯ: Переконуємось що timestamp це datetime об'єкт
        if isinstance(self.timestamp, datetime):
            # Часове вікно (групуємо сигнали в межах 5 хвилин)
            time_window = self.timestamp.replace(second=0, microsecond=0)
            time_window = time_window.replace(minute=time_window.minute // 5 * 5)
            hash_string += f"_{time_window.isoformat()}"
        else:
            # Якщо timestamp некоректний, використовуємо поточний час
            logger.warning(f"Некоректний timestamp для сигналу: {self.timestamp}")
            current_time = datetime.now()
            time_window = current_time.replace(second=0, microsecond=0)
            time_window = time_window.replace(minute=time_window.minute // 5 * 5)
            hash_string += f"_{time_window.isoformat()}"
    
        # Обчислюємо MD5 хеш
        self.hash = hashlib.md5(hash_string.encode()).hexdigest()
        return self.hash


class SignalConfig:
   """Конфігурація системи сигналів"""
   def __init__(self):
       # ПРОФЕСІЙНІ пороги для реального трейдингу
       self.base_thresholds = {
           'price_change_percent': 0.2,  # 0.2% - мінімальний значущий рух для скальпінгу
           'volume_spike_multiplier': 1.8,  # 1.8x від середнього - реальний сплеск об'єму
           'prediction_change_percent': 0.3,  # 0.3% - мінімальна зміна для дії
           'rsi_oversold': 25,  # 25 - більш агресивний поріг перепроданості
           'rsi_overbought': 75,  # 75 - більш агресивний поріг перекупленості
           'volatility_spike': 1.5  # 1.5x - реальний сплеск волатильності
       }
       
       # РЕАЛІСТИЧНІ адаптивні множники (базуються на реальній ринковій активності)
       self.time_multipliers = {
           'market_open': 0.5,      # 09:30-10:30 - найвища волатильність, нижчі пороги
           'market_close': 0.6,     # 15:00-16:00 - висока активність перед закриттям
           'regular_hours': 1.0,    # 10:30-15:00 - стандартна торгівля
           'after_hours': 1.2,      # 16:00-20:00 - менша ліквідність, більші пороги
           'night': 1.8            # 20:00-09:30 - мінімальна активність
       }
       
       # РЕАЛІСТИЧНІ пороги для професійного трейдингу
       self.prediction_period_thresholds = {
           # Пороги базуються на реальній волатильності S&P 500 та типових рухах акцій
           1: 0.15,   # 1 година - 0.15% (середній внутрішньоденний рух)
           3: 0.25,   # 3 години - 0.25% (значуща зміна тренду)
           6: 0.4,    # 6 годин - 0.4% (половина торгового дня)
           12: 0.6,   # 12 годин - 0.6% (денна сесія + pre/after market)
           24: 0.8,   # 1 день - 0.8% (типовий денний рух для стабільних акцій)
           72: 1.5,   # 3 дні - 1.5% (короткострокова зміна тренду)
           168: 2.5,  # 7 днів - 2.5% (тижневий тренд)
           360: 4.0,  # 15 днів - 4.0% (середньостроковий рух)
           720: 6.0   # 30 днів - 6.0% (місячна тенденція)
       }
       
       # ОПТИМІЗОВАНІ вікна дедуплікації для активного трейдингу
       self.prediction_dedup_windows = {
           # Коротші вікна для більш частих але релевантних сигналів
           1: 1800,     # 1 година прогноз -> 30 хв дедуплікації (було 1 год)
           3: 3600,     # 3 години прогноз -> 1 година дедуплікації (було 1.5 год)
           6: 5400,     # 6 годин прогноз -> 1.5 години дедуплікації (було 2 год)
           12: 7200,    # 12 годин прогноз -> 2 години дедуплікації (було 3 год)
           24: 10800,   # 1 день прогноз -> 3 години дедуплікації (було 4 год)
           72: 14400,   # 3 дні прогноз -> 4 години дедуплікації (було 6 год)
           168: 21600,  # 7 днів прогноз -> 6 годин дедуплікації (було 8 год)
           360: 28800,  # 15 днів прогноз -> 8 годин дедуплікації (було 12 год)
           720: 43200   # 30 днів прогноз -> 12 годин дедуплікації (було 24 год)
       }
       
       # Старі адаптивні вікна для інших типів сигналів
       self.adaptive_dedup_windows = {
           SignalPriority.CRITICAL: {
               'small_change': 1800,   # 30 мин для изменений < 2%
               'medium_change': 4320,  # 1.2 часа для изменений 2-5% 
               'large_change': 10800   # 3 часа для изменений > 5%
           },
           SignalPriority.IMPORTANT: {
               'small_change': 3600,   # 1 час для изменений < 1.5%
               'medium_change': 7200,  # 2 часа для изменений 1.5-3%
               'large_change': 14400   # 4 часа для изменений > 3%
           },
           SignalPriority.INFO: {
               'small_change': 5400,   # 1.5 часа для изменений < 1%
               'medium_change': 10800, # 3 часа для изменений 1-2%
               'large_change': 21600   # 6 часов для изменений > 2%
           }
       }
       
       # Старые окна для обратной совместимости
       self.dedup_windows = {
           SignalPriority.CRITICAL: 900,    # 15 хвилин
           SignalPriority.IMPORTANT: 3600,  # 1 година
           SignalPriority.INFO: 7200        # 2 години
       }
       
       # Налаштування групування
       self.grouping_config = {
           'time_window': 300,  # 5 хвилин для групування
           'min_signals_to_group': 2,  # Мінімум сигналів для групування
           'similarity_threshold': 0.8  # Поріг схожості для групування
       }
       
       # Максимальна кількість сигналів
       self.max_signals_per_ticker = {
           'per_hour': 5,
           'per_day': 20
       }
       
       # Історія для адаптації
       self.volatility_history = defaultdict(lambda: deque(maxlen=100))
       self.signal_effectiveness = defaultdict(float)


class BaseAnalyzer:
   """Базовий клас для аналізаторів"""
   def __init__(self, config: SignalConfig):
       self.config = config
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       """Аналіз даних та генерація сигналів"""
       raise NotImplementedError


class PriceChangeAnalyzer(BaseAnalyzer):
   """Аналізатор зміни ціни"""
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       signals = []
       
       current_price = data.get('current_price', 0)
       last_price = data.get('last_price', 0)
       volume = data.get('volume', 0)
       
       if not current_price or not last_price:
           return signals
       
       # Розрахунок зміни
       price_change = current_price - last_price
       price_change_percent = (price_change / last_price) * 100
       
       # Адаптивний поріг на основі волатильності
       volatility = self._calculate_volatility(ticker, history)
       adaptive_threshold = self.config.base_thresholds['price_change_percent'] * (1 + volatility)
       
       # Часовий множник
       time_multiplier = self._get_time_multiplier()
       effective_threshold = adaptive_threshold * time_multiplier
       
       # ОТЛАДКА: добавляем лог для понимания почему сигналы не проходят
       logger.info(f"SIGNAL DEBUG {ticker}: price_change={price_change_percent:.2f}%, threshold={effective_threshold:.2f}%, base={self.config.base_thresholds['price_change_percent']}, volatility={volatility:.2f}, time_mult={time_multiplier:.2f}")
       
       if abs(price_change_percent) >= effective_threshold:
           # Визначення пріоритету
           priority = self._determine_priority(price_change_percent, volatility, volume)
           
           # Аналіз контексту
           factors = self._analyze_context(ticker, price_change_percent, history, volume)
           
           # Сила сигналу
           strength = self._calculate_strength(price_change_percent, effective_threshold, factors)
           
           # Генерація сигналу
           direction = "зростання" if price_change_percent > 0 else "падіння"
           emoji = "📈" if price_change_percent > 0 else "📉"
           
           signal = Signal(
               type=SignalType.PRICE_CHANGE,
               priority=priority,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Значна зміна ціни",
               message=f"{emoji} Ціна {direction} на {abs(price_change_percent):.2f}%",
               current_price=current_price,
               price_change=price_change,
               price_change_percent=price_change_percent,
               volume=volume,
               confidence=self._calculate_enhanced_confidence(
                   price_change_percent, effective_threshold, factors, volatility
               ),
               strength=strength,
               factors=factors,
               data={
                   'last_price': last_price,
                   'volatility': volatility,
                   'threshold_used': effective_threshold
               }
           )
           
           # Додавання рекомендацій
           self._add_recommendations(signal, history)
           
           signals.append(signal)
           logger.info(f"SIGNAL CREATED: {ticker} {priority.value} {price_change_percent:+.2f}% ({direction})")
       
       return signals
   
   def _calculate_volatility(self, ticker: str, history: pd.DataFrame) -> float:
       """Розрахунок поточної волатильності"""
       if len(history) < 20:
           return 1.0
       
       returns = history['close'].pct_change().dropna()
       current_vol = returns.tail(20).std()
       historical_vol = returns.std()
       
       # Відносна волатильність
       rel_volatility = current_vol / (historical_vol + 1e-10)
       
       # Зберігаємо для історії
       self.config.volatility_history[ticker].append(rel_volatility)
       
       return rel_volatility
   
   def _get_time_multiplier(self) -> float:
       """Отримання часового множника"""
       now = datetime.now()
       hour = now.hour
       
       # Перевірка вихідних
       if now.weekday() >= 5:
           return 2.0
       
       # Час торгів (для NYSE)
       if 9 <= hour < 10:
           return self.config.time_multipliers['market_open']
       elif 10 <= hour < 15:
           return self.config.time_multipliers['regular_hours']
       elif 15 <= hour < 16:
           return self.config.time_multipliers['market_close']
       elif 16 <= hour < 20:
           return self.config.time_multipliers['after_hours']
       else:
           return self.config.time_multipliers['night']
   
   def _determine_priority(self, change_percent: float, volatility: float, volume: int) -> SignalPriority:
       """Визначення пріоритету сигналу"""
       abs_change = abs(change_percent)
       
       # Критичний: велика зміна з великим об'ємом
       if abs_change > 5.0 or (abs_change > 3.0 and volatility < 0.5):
           return SignalPriority.CRITICAL
       
       # Важливий: значна зміна
       if abs_change > 2.0 or (abs_change > 1.0 and volume > 1000000):
           return SignalPriority.IMPORTANT
       
       return SignalPriority.INFO
   
   def _analyze_context(self, ticker: str, change_percent: float, history: pd.DataFrame, volume: int) -> List[str]:
       """Аналіз контексту зміни"""
       factors = []
       
       if len(history) < 5:
           return factors
       
       # Напрямок тренду
       sma_20 = history['close'].tail(20).mean()
       current = history['close'].iloc[-1]
       
       if change_percent > 0 and current > sma_20:
           factors.append("Рух в напрямку тренду")
       elif change_percent < 0 and current < sma_20:
           factors.append("Рух в напрямку тренду")
       else:
           factors.append("Рух проти тренду")
       
       # Об'єм
       avg_volume = history['volume'].tail(20).mean()
       if volume > avg_volume * 2:
           factors.append("Високий об'єм торгів")
       elif volume > avg_volume * 1.5:
           factors.append("Підвищений об'єм")
       
       # Рівні
       high_52w = history['high'].tail(252).max() if len(history) > 252 else history['high'].max()
       low_52w = history['low'].tail(252).min() if len(history) > 252 else history['low'].min()
       
       if abs(current - high_52w) / high_52w < 0.02:
           factors.append("Біля 52-тижневого максимуму")
       elif abs(current - low_52w) / low_52w < 0.02:
           factors.append("Біля 52-тижневого мінімуму")
       
       # Послідовні рухи
       last_changes = history['close'].pct_change().tail(3)
       if all(last_changes > 0) and change_percent > 0:
           factors.append("Третій день зростання")
       elif all(last_changes < 0) and change_percent < 0:
           factors.append("Третій день падіння")
       
       return factors[:5]  # Максимум 5 факторів
   
   def _calculate_strength(self, change_percent: float, threshold: float, factors: List[str]) -> int:
       """Розрахунок сили сигналу (1-5)"""
       base_strength = min(5, int(abs(change_percent) / threshold))
       factor_bonus = min(2, len(factors) // 2)
       
       return min(5, base_strength + factor_bonus)
   
   def _calculate_enhanced_confidence(self, change_percent: float, threshold: float, 
                                    factors: List[str], volatility: float) -> float:
       """ПОКРАЩЕНИЙ розрахунок довіри до сигналу на основі кількох факторів"""
       
       # 1. Базова довіра - наскільки сильно перевищено поріг
       threshold_ratio = abs(change_percent) / threshold if threshold > 0 else 1.0
       base_confidence = min(0.9, 0.5 + (threshold_ratio - 1) * 0.2)
       
       # 2. Бонус за кількість підтверджуючих факторів
       factor_count = len(factors)
       if factor_count >= 4:
           factor_bonus = 0.25
       elif factor_count >= 3:
           factor_bonus = 0.15
       elif factor_count >= 2:
           factor_bonus = 0.1
       else:
           factor_bonus = 0.0
       
       # 3. Корекція за волатильність (менша волатільність = більша довіра)
       if volatility < 0.5:
           volatility_bonus = 0.1  # Низька волатільність - стабільний сигнал
       elif volatility > 1.5:
           volatility_bonus = -0.1  # Висока волатільність - менша довіра
       else:
           volatility_bonus = 0.0
       
       # 4. Бонус за критичні фактори
       critical_factors = ['volume_spike', 'breakout', 'consensus', 'technical_confluence']
       critical_bonus = sum(0.05 for factor in factors if any(cf in factor for cf in critical_factors))
       
       # Підсумковий розрахунок
       total_confidence = base_confidence + factor_bonus + volatility_bonus + critical_bonus
       
       # Обмежуємо в розумних межах
       return max(0.3, min(0.98, total_confidence))
   
   def _calculate_rsi_confidence(self, rsi: float, factors: List[str], oversold: bool = True) -> float:
       """Розрахунок довіри до RSI сигналу"""
       
       if oversold:
           # Для перепроданості: чим нижче RSI, тим вища довіра
           threshold = self.config.base_thresholds['rsi_oversold']
           base_confidence = 0.6 + (threshold - rsi) / threshold * 0.3
       else:
           # Для перекупленості: чим вище RSI, тим вища довіра  
           threshold = self.config.base_thresholds['rsi_overbought']
           base_confidence = 0.6 + (rsi - threshold) / (100 - threshold) * 0.3
       
       # Бонус за додаткові фактори (дивергенція, тренд)
       factor_bonus = len(factors) * 0.05
       
       # Екстремальні значення RSI мають вищу довіру
       if rsi <= 20 or rsi >= 80:
           extreme_bonus = 0.15
       elif rsi <= 15 or rsi >= 85:
           extreme_bonus = 0.25
       else:
           extreme_bonus = 0.0
       
       total_confidence = base_confidence + factor_bonus + extreme_bonus
       return max(0.5, min(0.95, total_confidence))
   
   def _add_recommendations(self, signal: Signal, history: pd.DataFrame):
       """Додавання рекомендацій"""
       if signal.price_change_percent > 0:
           # Зростання
           resistance = history['high'].tail(20).max()
           signal.target_price = resistance * 0.98
           signal.stop_loss = signal.current_price * 0.98
           signal.action_hint = f"Можливе досягнення ${signal.target_price:.2f}"
       else:
           # Падіння
           support = history['low'].tail(20).min()
           signal.target_price = support * 1.02
           signal.stop_loss = signal.current_price * 1.02
           signal.action_hint = f"Можлива підтримка на ${signal.target_price:.2f}"


class PredictionAnalyzer(BaseAnalyzer):
   """Аналізатор сигналів на основі прогнозів"""
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       signals = []
       predictions = data.get('predictions', {})
       current_price = data.get('current_price', 0)
       
       if not predictions or not current_price:
           return signals
       
       # Аналіз узгодженості моделей
       consensus_signal = self._check_model_consensus(ticker, predictions, current_price)
       if consensus_signal:
           signals.append(consensus_signal)
       
       # Аналіз окремих прогнозів
       for period_hours, prediction in predictions.items():
           signal = self._analyze_prediction(ticker, period_hours, prediction, current_price, history)
           if signal:
               signals.append(signal)
       
       return signals
   
   def _check_model_consensus(self, ticker: str, predictions: Dict, current_price: float) -> Optional[Signal]:
       """Перевірка узгодженості моделей"""
       if len(predictions) < 2:
           return None
       
       # Розділяємо на короткі та довгі
       short_term = {k: v for k, v in predictions.items() if k <= 24}
       long_term = {k: v for k, v in predictions.items() if k > 24}
       
       if not short_term or not long_term:
           return None
       
       # Середні зміни
       short_changes = [p['price_change_percent'] for p in short_term.values()]
       long_changes = [p['price_change_percent'] for p in long_term.values()]
       
       avg_short = np.mean(short_changes)
       avg_long = np.mean(long_changes)
       
       # Перевірка узгодженості
       if np.sign(avg_short) == np.sign(avg_long) and abs(avg_short) > 1 and abs(avg_long) > 1:
           direction = "зростання" if avg_short > 0 else "падіння"
           emoji = "📈" if avg_short > 0 else "📉"
           
           factors = [
               f"Короткі моделі: {avg_short:+.1f}%",
               f"Довгі моделі: {avg_long:+.1f}%",
               "Всі моделі показують один напрямок"
           ]
           
           # Середня впевненість
           all_confidences = [p['confidence'] for p in predictions.values()]
           avg_confidence = np.mean(all_confidences)
           
           return Signal(
               type=SignalType.MODEL_CONSENSUS,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Консенсус моделей",
               message=f"{emoji} Всі моделі прогнозують {direction}",
               current_price=current_price,
               price_change_percent=(avg_short + avg_long) / 2,
               confidence=avg_confidence,
               strength=4,
               factors=factors,
               data={
                   'short_term_avg': avg_short,
                   'long_term_avg': avg_long,
                   'model_count': len(predictions)
               }
           )
       
       return None
   
   def _analyze_prediction(self, ticker: str, period_hours: int, prediction: Dict, 
                          current_price: float, history: pd.DataFrame) -> Optional[Signal]:
       """Аналіз окремого прогнозу"""
       change_percent = prediction['price_change_percent']
       confidence = prediction['confidence']
       
       # УЛУЧШЕНИЕ: Адаптивный порог на основе периода прогнозирования
       # Находим наиболее подходящий период из конфигурации
       available_periods = sorted(self.config.prediction_period_thresholds.keys())
       closest_period = min(available_periods, key=lambda x: abs(x - period_hours))
       
       base_threshold = self.config.prediction_period_thresholds[closest_period]
       
       # Корекція на основі впевненості моделі
       confidence_multiplier = 2.0 - confidence  # Чим вища впевненість, тим нижчий поріг
       effective_threshold = base_threshold * confidence_multiplier
       
       # Додаткове логування для налагодження
       logger.info(f"PREDICTION FILTER {ticker}: period={period_hours}h, closest={closest_period}h, "
                  f"change={change_percent:.2f}%, threshold={effective_threshold:.2f}%, "
                  f"base={base_threshold:.2f}%, confidence={confidence:.2f}")
       
       if abs(change_percent) < effective_threshold:
           return None
       
       # Перевірка підтвердження тренду
       factors = []
       
       # Поточний рух в бік прогнозу?
       if len(history) >= 2:
           recent_change = (current_price - history['close'].iloc[-2]) / history['close'].iloc[-2] * 100
           if np.sign(recent_change) == np.sign(change_percent):
               factors.append("Ціна вже рухається в прогнозованому напрямку")
       
       # Висока впевненість
       if confidence > 0.8:
           factors.append(f"Висока впевненість моделі: {confidence:.0%}")
       
       # Близький прогноз важливіший
       if period_hours <= 6:
           factors.append("Короткостроковий прогноз")
           priority = SignalPriority.IMPORTANT if abs(change_percent) > 2 else SignalPriority.INFO
       else:
           priority = SignalPriority.INFO
       
       # Екстремальні прогнози
       if abs(change_percent) > 5:
           factors.append("Екстремальний прогноз")
           priority = SignalPriority.CRITICAL
       
       direction = "зростання" if change_percent > 0 else "падіння"
       emoji = "🔮" if change_percent > 0 else "🔻"
       
       return Signal(
           type=SignalType.PREDICTION,
           priority=priority,
           ticker=ticker,
           timestamp=datetime.now(),
           title=f"{ticker}: Прогноз {prediction['period_text']}",
           message=f"{emoji} Очікується {direction} на {abs(change_percent):.1f}%",
           current_price=current_price,
           price_change_percent=change_percent,
           confidence=confidence,
           strength=min(5, int(abs(change_percent) / 2) + 1),
           factors=factors,
           target_price=prediction['predicted_price'],
           data={
               'period_hours': period_hours,
               'period_text': prediction['period_text'],
               'predicted_price': prediction['predicted_price']
           }
       )


class TechnicalAnalyzer(BaseAnalyzer):
   """Аналізатор технічних індикаторів"""
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       signals = []
       
       if 'indicators' not in data or len(history) < 20:
           return signals
       
       indicators = data['indicators']
       current_price = data['current_price']
       
       # RSI сигнали
       rsi_signal = self._check_rsi(ticker, indicators, current_price)
       if rsi_signal:
           signals.append(rsi_signal)
       
       # Bollinger Bands
       bb_signal = self._check_bollinger_bands(ticker, indicators, current_price, history)
       if bb_signal:
           signals.append(bb_signal)
       
       # MACD
       macd_signal = self._check_macd(ticker, indicators, current_price)
       if macd_signal:
           signals.append(macd_signal)
       
       # Support/Resistance
       sr_signal = self._check_support_resistance(ticker, current_price, history)
       if sr_signal:
           signals.append(sr_signal)
       
       return signals
   
   def _check_rsi(self, ticker: str, indicators: Dict, current_price: float) -> Optional[Signal]:
       """Перевірка RSI"""
       rsi = indicators.get('RSI')
       if not rsi:
           return None
       
       factors = []
       
       if rsi < self.config.base_thresholds['rsi_oversold']:
           factors.append(f"RSI = {rsi:.1f} (перепроданість)")
           
           # Перевірка дивергенції
           if 'rsi_history' in indicators:
               rsi_hist = indicators['rsi_history']
               if len(rsi_hist) >= 5 and rsi > min(rsi_hist[-5:]):
                   factors.append("Позитивна дивергенція RSI")
           
           return Signal(
               type=SignalType.TECHNICAL_BREAKOUT,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: RSI перепроданість",
               message=f"📊 RSI досяг {rsi:.1f} - можливий розворот",
               current_price=current_price,
               confidence=self._calculate_rsi_confidence(rsi, factors, oversold=True),
               strength=3,
               factors=factors,
               action_hint="Можливість для покупки",
               data={'rsi': rsi}
           )
       
       elif rsi > self.config.base_thresholds['rsi_overbought']:
           factors.append(f"RSI = {rsi:.1f} (перекупленість)")
           
           return Signal(
               type=SignalType.TECHNICAL_BREAKOUT,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: RSI перекупленість",
               message=f"📊 RSI досяг {rsi:.1f} - можлива корекція",
               current_price=current_price,
               confidence=self._calculate_rsi_confidence(rsi, factors, oversold=False),
               strength=3,
               factors=factors,
               action_hint="Розгляньте фіксацію прибутку",
               data={'rsi': rsi}
           )
       
       return None
   
   def _check_bollinger_bands(self, ticker: str, indicators: Dict, current_price: float, 
                              history: pd.DataFrame) -> Optional[Signal]:
       """Перевірка Bollinger Bands"""
       bb_upper = indicators.get('BB_upper')
       bb_lower = indicators.get('BB_lower')
       bb_position = indicators.get('BB_position')
       
       if not all([bb_upper, bb_lower, bb_position]):
           return None
       
       factors = []
       
       # Пробій верхньої межі
       if current_price > bb_upper:
           factors.append("Пробій верхньої смуги Боллінджера")
           
           # Перевірка об'єму
           if len(history) >= 2:
               current_vol = history['volume'].iloc[-1]
               avg_vol = history['volume'].tail(20).mean()
               if current_vol > avg_vol * 1.5:
                   factors.append("Підтверджено об'ємом")
           
           return Signal(
               type=SignalType.TECHNICAL_BREAKOUT,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Пробій BB вгору",
               message=f"📈 Ціна пробила верхню смугу Боллінджера",
               current_price=current_price,
               confidence=0.6,
               strength=3,
               factors=factors,
               target_price=current_price * 1.02,
               data={
                   'bb_upper': bb_upper,
                   'bb_lower': bb_lower,
                   'bb_position': bb_position
               }
           )
       
       # Пробій нижньої межі
       elif current_price < bb_lower:
           factors.append("Пробій нижньої смуги Боллінджера")
           
           return Signal(
               type=SignalType.TECHNICAL_BREAKOUT,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Пробій BB вниз",
               message=f"📉 Ціна пробила нижню смугу Боллінджера",
               current_price=current_price,
               confidence=0.6,
               strength=3,
               factors=factors,
               target_price=current_price * 0.98,
               data={
                   'bb_upper': bb_upper,
                   'bb_lower': bb_lower,
                   'bb_position': bb_position
               }
           )
       
       return None
   
   def _check_macd(self, ticker: str, indicators: Dict, current_price: float) -> Optional[Signal]:
       """Перевірка MACD"""
       macd = indicators.get('MACD')
       macd_signal = indicators.get('MACD_signal')
       macd_diff = indicators.get('MACD_diff')
       
       if not all([macd is not None, macd_signal is not None, macd_diff is not None]):
           return None
       
       # Перевірка перетину
       if 'MACD_diff_prev' in indicators:
           prev_diff = indicators['MACD_diff_prev']
           
           # Бичачий перетин
           if prev_diff < 0 and macd_diff > 0:
               return Signal(
                   type=SignalType.TECHNICAL_BREAKOUT,
                   priority=SignalPriority.INFO,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: MACD бичачий сигнал",
                   message="📈 MACD перетнув сигнальну лінію знизу вгору",
                   current_price=current_price,
                   confidence=0.6,
                   strength=2,
                   factors=["Бичаче перетинання MACD"],
                   action_hint="Можливий початок висхідного тренду",
                   data={'macd': macd, 'macd_signal': macd_signal}
               )
           
           # Ведмежий перетин
           elif prev_diff > 0 and macd_diff < 0:
               return Signal(
                   type=SignalType.TECHNICAL_BREAKOUT,
                   priority=SignalPriority.INFO,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: MACD ведмежий сигнал",
                   message="📉 MACD перетнув сигнальну лінію зверху вниз",
                   current_price=current_price,
                   confidence=0.6,
                   strength=2,
                   factors=["Ведмеже перетинання MACD"],
                   action_hint="Можливий початок низхідного тренду",
                   data={'macd': macd, 'macd_signal': macd_signal}
               )
       
       return None
   
   def _check_support_resistance(self, ticker: str, current_price: float, 
                                 history: pd.DataFrame) -> Optional[Signal]:
       """Перевірка рівнів підтримки/опору"""
       if len(history) < 50:
           return None
       
       # Пошук ключових рівнів
       highs = history['high'].tail(50)
       lows = history['low'].tail(50)
       
       # Локальні максимуми/мінімуми
       resistance_levels = []
       support_levels = []
       
       for i in range(2, len(highs) - 2):
           if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
               resistance_levels.append(highs.iloc[i])
           
           if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
               support_levels.append(lows.iloc[i])
       
       if not resistance_levels and not support_levels:
           return None
       
       # Перевірка наближення до рівнів
       for resistance in sorted(resistance_levels, reverse=True)[:3]:
           if abs(current_price - resistance) / resistance < 0.005:  # В межах 0.5%
               return Signal(
                   type=SignalType.SUPPORT_RESISTANCE,
                   priority=SignalPriority.INFO,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: Наближення до опору",
                   message=f"📊 Ціна наближається до рівня опору ${resistance:.2f}",
                   current_price=current_price,
                   confidence=0.7,
                   strength=2,
                   factors=[f"Історичний опір на ${resistance:.2f}"],
                   target_price=resistance,
                   data={'resistance': resistance}
               )
       
       for support in sorted(support_levels)[:3]:
           if abs(current_price - support) / support < 0.005:  # В межах 0.5%
               return Signal(
                   type=SignalType.SUPPORT_RESISTANCE,
                   priority=SignalPriority.INFO,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: Наближення до підтримки",
                   message=f"📊 Ціна наближається до рівня підтримки ${support:.2f}",
                   current_price=current_price,
                   confidence=0.7,
                   strength=2,
                   factors=[f"Історична підтримка на ${support:.2f}"],
                   target_price=support,
                   data={'support': support}
               )
       
       return None


class PatternAnalyzer(BaseAnalyzer):
   """Аналізатор патернів"""
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       signals = []
       
       if len(history) < 10:
           return signals
       
       # Розворотні патерни
       reversal = self._check_reversal_pattern(ticker, data, history)
       if reversal:
           signals.append(reversal)
       
       # Прискорення руху
       acceleration = self._check_acceleration(ticker, data, history)
       if acceleration:
           signals.append(acceleration)
       
       # Дивергенція
       divergence = self._check_divergence(ticker, data, history)
       if divergence:
           signals.append(divergence)
       
       return signals
   
   def _check_reversal_pattern(self, ticker: str, data: Dict, history: pd.DataFrame) -> Optional[Signal]:
       """Перевірка розворотних патернів"""
       if len(history) < 5:
           return None
       
       current_price = data['current_price']
       recent_prices = history['close'].tail(5)
       recent_changes = recent_prices.pct_change().dropna()
       
       # V-подібний розворот
       if len(recent_changes) >= 4:
           # Падіння і різке зростання
           if recent_changes.iloc[-3] < -0.01 and recent_changes.iloc[-2] < -0.01 and recent_changes.iloc[-1] > 0.02:
               return Signal(
                   type=SignalType.TREND_REVERSAL,
                   priority=SignalPriority.IMPORTANT,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: V-подібний розворот",
                   message="📈 Можливий розворот після падіння",
                   current_price=current_price,
                   confidence=0.7,
                   strength=3,
                   factors=["Різкий розворот після падіння", "V-подібний патерн"],
                   action_hint="Можливий початок відновлення"
               )
           
           # Зростання і різке падіння
           elif recent_changes.iloc[-3] > 0.01 and recent_changes.iloc[-2] > 0.01 and recent_changes.iloc[-1] < -0.02:
               return Signal(
                   type=SignalType.TREND_REVERSAL,
                   priority=SignalPriority.IMPORTANT,
                   ticker=ticker,
                   timestamp=datetime.now(),
                   title=f"{ticker}: Перевернутий V-патерн",
                   message="📉 Можливий розворот після зростання",
                   current_price=current_price,
                   confidence=0.7,
                   strength=3,
                   factors=["Різкий розворот після зростання", "Перевернутий V-патерн"],
                   action_hint="Можлива корекція"
               )
       
       return None
   
   def _check_acceleration(self, ticker: str, data: Dict, history: pd.DataFrame) -> Optional[Signal]:
       """Перевірка прискорення руху"""
       if len(history) < 10:
           return None
       
       current_price = data['current_price']
       recent_changes = history['close'].pct_change().tail(10).dropna()
       
       if len(recent_changes) < 5:
           return None
       
       # Порівняння швидкості зміни
       early_changes = recent_changes.iloc[:5].mean()
       late_changes = recent_changes.iloc[-3:].mean()
       
       # Прискорення зростання
       if early_changes > 0 and late_changes > early_changes * 2:
           return Signal(
               type=SignalType.PATTERN,
               priority=SignalPriority.INFO,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Прискорення зростання",
               message=f"🚀 Темп зростання прискорюється (з {early_changes*100:.1f}% до {late_changes*100:.1f}%)",
               current_price=current_price,
               price_change_percent=late_changes*100,  # Устанавливаем процентное изменение
               confidence=0.6,
               strength=2,
               factors=[
                   f"Раннє зростання: {early_changes*100:.1f}%",
                   f"Поточне зростання: {late_changes*100:.1f}%"
               ],
               action_hint="Тренд набирає силу"
           )
       
       # Прискорення падіння
       elif early_changes < 0 and late_changes < early_changes * 2:
           return Signal(
               type=SignalType.PATTERN,
               priority=SignalPriority.INFO,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Прискорення падіння",
               message=f"⬇️ Темп падіння прискорюється (з {early_changes*100:.1f}% до {late_changes*100:.1f}%)",
               current_price=current_price,
               price_change_percent=late_changes*100,  # Устанавливаем процентное изменение
               confidence=0.6,
               strength=2,
               factors=[
                   f"Раннє падіння: {early_changes*100:.1f}%",
                   f"Поточне падіння: {late_changes*100:.1f}%"
               ],
               action_hint="Негативний тренд посилюється"
           )
       
       return None
   
   def _check_divergence(self, ticker: str, data: Dict, history: pd.DataFrame) -> Optional[Signal]:
       """Перевірка дивергенції ціни та індикаторів"""
       indicators = data.get('indicators', {})
       rsi = indicators.get('RSI')
       
       if not rsi or 'rsi_history' not in indicators or len(history) < 10:
           return None
       
       current_price = data['current_price']
       price_history = history['close'].tail(10)
       rsi_history = indicators['rsi_history'][-10:]
       
       if len(rsi_history) < 10:
           return None
       
       # Пошук екстремумів
       price_trend = np.polyfit(range(len(price_history)), price_history.values, 1)[0]
       rsi_trend = np.polyfit(range(len(rsi_history)), rsi_history, 1)[0]
       
       # Бичача дивергенція: ціна падає, RSI зростає
       if price_trend < -0.1 and rsi_trend > 0.5:
           return Signal(
               type=SignalType.DIVERGENCE,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Бичача дивергенція",
               message="📊 RSI зростає при падінні ціни",
               current_price=current_price,
               confidence=0.7,
               strength=3,
               factors=[
                   "Ціна падає, RSI зростає",
                   "Можливий розворот вгору"
               ],
               action_hint="Сигнал до покупки",
               data={'price_trend': price_trend, 'rsi_trend': rsi_trend}
           )
       
       # Ведмежа дивергенція: ціна зростає, RSI падає
       elif price_trend > 0.1 and rsi_trend < -0.5:
           return Signal(
               type=SignalType.DIVERGENCE,
               priority=SignalPriority.IMPORTANT,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Ведмежа дивергенція",
               message="📊 RSI падає при зростанні ціни",
               current_price=current_price,
               confidence=0.7,
               strength=3,
               factors=[
                   "Ціна зростає, RSI падає",
                   "Можливий розворот вниз"
               ],
               action_hint="Розгляньте продаж",
               data={'price_trend': price_trend, 'rsi_trend': rsi_trend}
           )
       
       return None


class VolumeAnalyzer(BaseAnalyzer):
   """Аналізатор об'ємів"""
   
   def analyze(self, ticker: str, data: Dict, history: pd.DataFrame) -> List[Signal]:
       signals = []
       
       volume = data.get('volume', 0)
       if not volume or len(history) < 20:
           return signals
       
       avg_volume = history['volume'].tail(20).mean()
       volume_ratio = volume / (avg_volume + 1e-10)
       
       # Сплеск об'єму
       if volume_ratio >= self.config.base_thresholds['volume_spike_multiplier']:
           current_price = data['current_price']
           price_change = data.get('price_change_percent', 0)
           
           # ВИПРАВЛЕННЯ: Не генеруємо сигнали з нульовою зміною ціни
           if abs(price_change) < 0.05:  # Менше 0.05% зміни - ігноруємо
               return signals
           
           factors = [f"Об'єм в {volume_ratio:.1f}x вище середнього"]
           
           # Аналіз контексту
           if abs(price_change) > 1:
               if price_change > 0:
                   factors.append("Зростання на високому об'ємі")
                   message = "📈 Сильне зростання підтверджене об'ємом"
               else:
                   factors.append("Падіння на високому об'ємі")
                   message = "📉 Сильне падіння підтверджене об'ємом"
               
               priority = SignalPriority.IMPORTANT
               strength = 4
           else:
               # Значний об'єм з помірною зміною ціни (0.05% - 1%)
               factors.append("Високий об'єм з помірною зміною ціни")
               if price_change > 0:
                   message = "📊 Помірне зростання на високому об'ємі"
               else:
                   message = "📊 Помірне падіння на високому об'ємі"
               priority = SignalPriority.INFO
               strength = 2
           
           # Перевірка рівнів
           if len(history) >= 50:
               high_50d = history['high'].tail(50).max()
               low_50d = history['low'].tail(50).min()
               
               if abs(current_price - high_50d) / high_50d < 0.02:
                   factors.append("Біля 50-денного максимуму")
               elif abs(current_price - low_50d) / low_50d < 0.02:
                   factors.append("Біля 50-денного мінімуму")
           
           signal = Signal(
               type=SignalType.VOLUME_SPIKE,
               priority=priority,
               ticker=ticker,
               timestamp=datetime.now(),
               title=f"{ticker}: Сплеск об'єму",
               message=message,
               current_price=current_price,
               volume=volume,
               price_change_percent=price_change,  # ВИПРАВЛЕННЯ: Додаємо зміну ціни
               confidence=self._calculate_volume_confidence(volume_ratio, factors, current_price, history),
               strength=strength,
               factors=factors,
               data={
                   'volume_ratio': volume_ratio,
                   'avg_volume': avg_volume
               }
           )
           
           signals.append(signal)
       
       return signals

   def _calculate_volume_confidence(self, volume_ratio: float, factors: List[str], 
                                  current_price: float, history: pd.DataFrame) -> float:
       """Розрахунок довіри до сигналу об'єму"""
       
       # 1. Базова довіра залежить від величини сплеску об'єму
       if volume_ratio >= 5.0:
           base_confidence = 0.9  # Екстремальний сплеск
       elif volume_ratio >= 3.0:
           base_confidence = 0.8  # Сильний сплеск
       elif volume_ratio >= 2.0:
           base_confidence = 0.7  # Помірний сплеск
       else:
           base_confidence = 0.6  # Слабкий сплеск
       
       # 2. Бонус за підтверджуючі фактори
       factor_bonus = len(factors) * 0.03
       
       # 3. Бонус за прорив ключових рівнів
       price_level_bonus = 0.0
       if len(history) >= 20:
           high_20 = history['high'].tail(20).max()
           low_20 = history['low'].tail(20).min()
           
           if current_price > high_20 * 1.01:  # Прорив максимуму
               price_level_bonus = 0.15
           elif current_price < low_20 * 0.99:  # Прорив мінімуму
               price_level_bonus = 0.15
       
       # 4. Зниження за дуже високий об'єм (можливий викид)
       if volume_ratio > 10.0:
           outlier_penalty = -0.1
       else:
           outlier_penalty = 0.0
       
       total_confidence = base_confidence + factor_bonus + price_level_bonus + outlier_penalty
       return max(0.6, min(0.95, total_confidence))


class SignalAggregator:
   """Агрегатор для групування схожих сигналів"""
   
   def __init__(self, config: SignalConfig):
       self.config = config
       self.signal_groups = defaultdict(list)
       
   def aggregate_signals(self, signals: List[Signal]) -> List[Signal]:
       """Агрегування схожих сигналів"""
       if len(signals) < self.config.grouping_config['min_signals_to_group']:
           return signals
       
       # Групуємо за тікерами
       by_ticker = defaultdict(list)
       for signal in signals:
           by_ticker[signal.ticker].append(signal)
       
       aggregated = []
       
       for ticker, ticker_signals in by_ticker.items():
           # Групуємо за хешами
           groups = defaultdict(list)
           for signal in ticker_signals:
               signal.calculate_hash()
               groups[signal.hash].append(signal)
           
           # Обробляємо кожну групу
           for hash_key, group in groups.items():
               if len(group) >= self.config.grouping_config['min_signals_to_group']:
                   # Створюємо агрегований сигнал
                   aggregated_signal = self._create_aggregated_signal(group)
                   aggregated.append(aggregated_signal)
               else:
                   # Додаємо як є
                   aggregated.extend(group)
       
       return aggregated
   
   def _create_aggregated_signal(self, signals: List[Signal]) -> Signal:
       """Створення агрегованого сигналу з групи"""
       # Беремо сигнал з найвищим пріоритетом як основу
       main_signal = max(signals, key=lambda s: (s.priority.get_weight(), s.confidence))
       
       # Збираємо всі унікальні фактори
       all_factors = set()
       for sig in signals:
           all_factors.update(sig.factors)
       
       # Обчислюємо середні значення
       avg_change = sum(s.price_change_percent for s in signals) / len(signals)
       avg_confidence = sum(s.confidence for s in signals) / len(signals)
       max_strength = max(s.strength for s in signals)
       
       # Створюємо агрегований сигнал
       aggregated = Signal(
           type=SignalType.AGGREGATED,
           priority=main_signal.priority,
           ticker=main_signal.ticker,
           timestamp=main_signal.timestamp,
           title=f"{main_signal.ticker}: Комплексний сигнал",
           message=f"Виявлено {len(signals)} пов'язаних сигналів",
           current_price=main_signal.current_price,
           price_change=main_signal.price_change,
           price_change_percent=avg_change,
           volume=main_signal.volume,
           confidence=avg_confidence,
           strength=max_strength,
           factors=list(all_factors)[:10],  # Максимум 10 факторів
           data={
               'aggregated_count': len(signals),
               'signal_types': [s.type.value for s in signals],
               'original_data': main_signal.data
           },
           action_hint=main_signal.action_hint,
           target_price=main_signal.target_price,
           stop_loss=main_signal.stop_loss,
           related_signals=signals[1:] if len(signals) > 1 else []
       )
       
       # Позначаємо вихідні сигнали як згруповані
       for sig in signals:
           sig.is_grouped = True
       
       return aggregated


class EnhancedSignalFilter:
   """Покращена фільтрація та дедуплікація сигналів"""
   
   def __init__(self, config: SignalConfig):
       self.config = config
       self.signal_history = defaultdict(lambda: deque(maxlen=1000))
       self.last_signals = {}
       self.sent_hashes = defaultdict(set)  # Зберігання відправлених хешів за тікерами
       self.daily_counts = defaultdict(int)  # Лічильники за день
       self.hourly_counts = defaultdict(int)  # Лічильники за годину
       self.last_cleanup = datetime.now()
       
   def filter_signals(self, signals: List[Signal]) -> List[Signal]:
       """Фільтрація списку сигналів"""
       # Періодичне очищення старих даних
       self._cleanup_old_data()
       
       filtered = []
       
       for signal in signals:
           if self._should_send_signal(signal):
               filtered.append(signal)
               self._record_signal(signal)
       
       # Застосовуємо ліміти
       filtered = self._apply_rate_limits(filtered)
       
       # Сортування за пріоритетом та силою
       filtered.sort(key=lambda s: (s.priority.get_weight(), s.strength), reverse=True)
       
       return filtered
   
   def _should_send_signal(self, signal: Signal) -> bool:
       """Перевірка, чи потрібно відправляти сигнал"""
       
       # УЛУЧШЕНИЕ: Отчеты всегда пропускаем без фильтрации
       report_types = {
           SignalType.HOURLY_REPORT,
           SignalType.HALF_HOURLY_REPORT,
           SignalType.DAILY_SUMMARY,
           SignalType.MARKET_STATUS,
           SignalType.PREDICTION_RESULT
       }
       
       if signal.type in report_types:
           logger.debug(f"Пропускаем отчет без фильтрации: {signal.type.value} для {signal.ticker}")
           return True
       
       # ВИПРАВЛЕННЯ: Фільтруємо сигнали з нульовою або мізерною зміною ціни
       if abs(signal.price_change_percent) < 0.05:
           logger.debug(f"Відфільтровано сигнал з мізерною зміною ціни: {signal.price_change_percent:.3f}% для {signal.ticker}")
           return False
       
       # Обчислюємо хеш
       signal.calculate_hash()
       
       # Перевіряємо, чи вже був відправлений такий сигнал
       if signal.hash in self.sent_hashes[signal.ticker]:
           logger.debug(f"Сигнал {signal.hash} вже був відправлений для {signal.ticker}")
           return False
       
       # Перевірка часового вікна для схожих сигналів
       if self._has_similar_recent_signal(signal):
           return False
       
       # Перевірка лімітів
       if not self._check_rate_limits(signal):
           return False
       
       return True
   
   def _get_adaptive_dedup_window(self, signal: Signal) -> int:
        """Получение адаптивного окна дедупликации на основе типа и периода сигнала"""
        
        # УЛУЧШЕНИЕ: Для прогнозов используем окна на основе периода прогнозирования
        if signal.type == SignalType.PREDICTION:
            period_hours = signal.data.get('period_hours', 24)
            
            # Находим наиболее подходящий период из конфигурации
            available_periods = sorted(self.config.prediction_dedup_windows.keys())
            closest_period = min(available_periods, key=lambda x: abs(x - period_hours))
            
            dedup_window = self.config.prediction_dedup_windows[closest_period]
            
            logger.debug(f"DEDUP WINDOW {signal.ticker}: prediction period={period_hours}h, "
                        f"closest={closest_period}h, window={dedup_window}s")
            
            return dedup_window
        
        # Для ценовых сигналов - адаптивные окна на основе величины изменения
        if signal.type == SignalType.PRICE_CHANGE:
            abs_change = abs(signal.price_change_percent)
            windows = self.config.adaptive_dedup_windows[signal.priority]
            
            if signal.priority == SignalPriority.CRITICAL:
                if abs_change < 2.0:
                    return windows['small_change']
                elif abs_change < 5.0:
                    return windows['medium_change']
                else:
                    return windows['large_change']
            elif signal.priority == SignalPriority.IMPORTANT:
                if abs_change < 1.5:
                    return windows['small_change']
                elif abs_change < 3.0:
                    return windows['medium_change']
                else:
                    return windows['large_change']
            else:  # INFO
                if abs_change < 1.0:
                    return windows['small_change']
                elif abs_change < 2.0:
                    return windows['medium_change']
                else:
                    return windows['large_change']
        
        # Для всех остальных типов сигналов используем базовые окна
        return self.config.dedup_windows[signal.priority]

   def _has_similar_recent_signal(self, signal: Signal) -> bool:
        """Перевірка на наявність схожих недавніх сигналів"""
        ticker_history = self.signal_history[signal.ticker]
    
        # УЛУЧШЕНИЕ: Используем адаптивное окно
        dedup_window = self._get_adaptive_dedup_window(signal)
        cutoff_time = signal.timestamp - timedelta(seconds=dedup_window)
    
        for hist_record in reversed(ticker_history):
            # ВИПРАВЛЕННЯ: Конвертуємо timestamp з рядка в datetime якщо потрібно
            hist_timestamp = hist_record['timestamp']
            if isinstance(hist_timestamp, str):
                try:
                    hist_timestamp = datetime.fromisoformat(hist_timestamp)
                except:
                    # Якщо не вдалось розпарсити, пропускаємо
                    continue
        
            if hist_timestamp < cutoff_time:
                break
            
            # Перевіряємо схожість
            if self._are_signals_similar(signal, hist_record):
                return True
    
        return False
   
   def _are_signals_similar(self, signal: Signal, hist_record: Dict) -> bool:
       """Перевірка схожості сигналів"""
       # Однаковий тип та пріоритет
       if (hist_record['type'] != signal.type or 
           hist_record['priority'] != signal.priority):
           return False
       
       # Для зміни ціни - перевіряємо близькість значень
       if signal.type == SignalType.PRICE_CHANGE:
           hist_change = hist_record.get('price_change_percent', 0)
           if abs(signal.price_change_percent - hist_change) < 0.5:
               return True
       
       # Для прогнозів - перевіряємо період
       elif signal.type == SignalType.PREDICTION:
           hist_period = hist_record.get('data', {}).get('period_hours', -1)
           signal_period = signal.data.get('period_hours', -2)
           if hist_period == signal_period:
               # Перевіряємо близькість прогнозів
               hist_change = hist_record.get('price_change_percent', 0)
               if abs(signal.price_change_percent - hist_change) < 1.0:
                   return True
       
       # Для об'єму - завжди вважаємо схожими в межах вікна
       elif signal.type == SignalType.VOLUME_SPIKE:
           return True
       
       return False
   
   def _check_rate_limits(self, signal: Signal) -> bool:
       """Перевірка лімітів відправки"""
       current_hour = signal.timestamp.replace(minute=0, second=0, microsecond=0)
       current_day = signal.timestamp.date()
       
       # Ключі для лічильників
       hour_key = f"{signal.ticker}_{current_hour}"
       day_key = f"{signal.ticker}_{current_day}"
       
       # Перевірка годинного ліміту
       if self.hourly_counts[hour_key] >= self.config.max_signals_per_ticker['per_hour']:
           logger.debug(f"Досягнуто годинний ліміт для {signal.ticker}")
           return False
       
       # Перевірка денного ліміту
       if self.daily_counts[day_key] >= self.config.max_signals_per_ticker['per_day']:
           logger.debug(f"Досягнуто денний ліміт для {signal.ticker}")
           return False
       
       return True
   
   def _record_signal(self, signal: Signal):
        """Запис відправленого сигналу в пам'ять та БД"""
        current_time = signal.timestamp
    
        # Зберігаємо в БД
        try:
            # Extract period_hours for prediction signals
            period_hours = None
            if hasattr(signal, 'data') and signal.data:
                period_hours = signal.data.get('period_hours')
            
            signal_record = SignalRecord(
                ticker=signal.ticker,
                signal_type=signal.type.value if hasattr(signal.type, 'value') else str(signal.type),
                priority=str(signal.priority.value if hasattr(signal.priority, 'value') else signal.priority),
                message=signal.message,
                timestamp=signal.timestamp,
                price=signal.current_price,
                change_percent=signal.price_change_percent,
                volume=getattr(signal, 'volume', None),
                confidence=signal.confidence,
                period_hours=period_hours
            )
            
            # Перевіряємо на дублікати через БД
            if not db.is_duplicate_signal(signal_record, window_minutes=30):
                db.save_signal(signal_record)
                logger.debug(f"Сигнал збережено в БД: {signal.ticker} - {signal.message}")
        except Exception as e:
            logger.error(f"Помилка збереження сигналу в БД: {e}")
    
        # Зберігаємо в пам'яті
        if signal.ticker not in self.sent_hashes:
            self.sent_hashes[signal.ticker] = {}
    
        # Перевіряємо тип і мігруємо якщо потрібно
        if isinstance(self.sent_hashes[signal.ticker], set):
            old_hashes = self.sent_hashes[signal.ticker]
            self.sent_hashes[signal.ticker] = {h: current_time for h in old_hashes}
    
        self.sent_hashes[signal.ticker][signal.hash] = current_time
    
        # Оновлюємо лічильники
        current_hour = signal.timestamp.replace(minute=0, second=0, microsecond=0)
        current_day = signal.timestamp.date()
    
        hour_key = f"{signal.ticker}_{current_hour.isoformat()}"
        day_key = f"{signal.ticker}_{current_day.isoformat()}"
    
        self.hourly_counts[hour_key] += 1
        self.daily_counts[day_key] += 1
    
        # Зберігаємо в історії
        self.signal_history[signal.ticker].append({
            'timestamp': signal.timestamp,
            'type': signal.type,
            'priority': signal.priority,
            'price_change_percent': round(signal.price_change_percent, 2),
            'hash': signal.hash[:8] if signal.hash else '',
            'data': signal.data
        })
   
   def _apply_rate_limits(self, signals: List[Signal]) -> List[Signal]:
       """Застосування лімітів до списку сигналів"""
       # Групуємо за тікерами
       by_ticker = defaultdict(list)
       for signal in signals:
           by_ticker[signal.ticker].append(signal)
       
       limited = []
       
       for ticker, ticker_signals in by_ticker.items():
           # Сортуємо за пріоритетом
           ticker_signals.sort(key=lambda s: s.priority.get_weight(), reverse=True)
           
           # Беремо тільки топ сигнали якщо забагато
           if len(ticker_signals) > 3:
               # Беремо всі критичні + топ важливі/інфо
               critical = [s for s in ticker_signals if s.priority == SignalPriority.CRITICAL]
               others = [s for s in ticker_signals if s.priority != SignalPriority.CRITICAL][:3]
               ticker_signals = critical + others
           
           limited.extend(ticker_signals)
       
       return limited
   
   def _cleanup_old_data(self):
        """Очищення старих даних"""
        now = datetime.now()
    
        # Очищаємо раз на годину
        if (now - self.last_cleanup).total_seconds() < 3600:
            return
    
        self.last_cleanup = now
    
        # ВИПРАВЛЕННЯ: Додаємо метрики для моніторингу використання пам'яті
        total_signals = sum(len(hist) for hist in self.signal_history.values())
        total_hashes = sum(len(hashes) for hashes in self.sent_hashes.values())
    
        logger.info(f"Очищення даних: {total_signals} сигналів, {total_hashes} хешів в пам'яті")
    
        # ВИПРАВЛЕННЯ: Видаляємо дані для неактивних тікерів
        # Визначаємо активні тікери (ті що мали сигнали за останні 24 години)
        active_cutoff = now - timedelta(hours=24)
        active_tickers = set()
    
        for ticker, history in list(self.signal_history.items()):
            if history:
                # Перевіряємо останній сигнал
                last_signal = history[-1]
                last_timestamp = last_signal['timestamp']
            
                # ВИПРАВЛЕННЯ: Конвертуємо timestamp якщо це рядок
                if isinstance(last_timestamp, str):
                    try:
                        last_timestamp = datetime.fromisoformat(last_timestamp)
                    except:
                        # Якщо не вдалось конвертувати, видаляємо
                        logger.warning(f"Некоректний timestamp для {ticker}, видаляємо історію")
                        del self.signal_history[ticker]
                        if ticker in self.sent_hashes:
                            del self.sent_hashes[ticker]
                        continue
            
                if last_timestamp > active_cutoff:
                    active_tickers.add(ticker)
                else:
                    # Видаляємо історію для неактивного тікера
                    logger.info(f"Видалення історії для неактивного тікера: {ticker}")
                    del self.signal_history[ticker]
                
                    # Також видаляємо хеші
                    if ticker in self.sent_hashes:
                        del self.sent_hashes[ticker]
                
                    # Також видаляємо хеші
                    if ticker in self.sent_hashes:
                        del self.sent_hashes[ticker]
    
        # ВИПРАВЛЕННЯ: Більш агресивне очищення хешів
        MAX_HASHES_PER_TICKER = 200  # Зменшено з 500
        MAX_HASH_AGE_HOURS = 12  # Максимальний вік хешу
    
        for ticker in list(self.sent_hashes.keys()):
            if ticker not in active_tickers:
                # Видаляємо хеші для неактивних тікерів
                del self.sent_hashes[ticker]
                continue
            
            current_hashes = self.sent_hashes[ticker]
            if len(current_hashes) > MAX_HASHES_PER_TICKER:
                # ВИПРАВЛЕННЯ: Зберігаємо хеші з часовими мітками для кращого контролю
                # Конвертуємо в список з часовими мітками
                if isinstance(current_hashes, set):
                    # Мігруємо старий формат до нового
                    new_hash_dict = {}
                    for hash_val in current_hashes:
                        new_hash_dict[hash_val] = now
                    self.sent_hashes[ticker] = new_hash_dict
                    current_hashes = new_hash_dict
            
                # Видаляємо старі хеші
                if isinstance(current_hashes, dict):
                    cutoff_time = now - timedelta(hours=MAX_HASH_AGE_HOURS)
                    old_hashes = [h for h, t in current_hashes.items() if t < cutoff_time]
                
                    for old_hash in old_hashes:
                        del current_hashes[old_hash]
                
                    # Якщо все ще забагато, залишаємо тільки найновіші
                    if len(current_hashes) > MAX_HASHES_PER_TICKER:
                        sorted_hashes = sorted(current_hashes.items(), key=lambda x: x[1], reverse=True)
                        self.sent_hashes[ticker] = dict(sorted_hashes[:MAX_HASHES_PER_TICKER//2])
    
        # ВИПРАВЛЕННЯ: Очищення старих лічильників з кращою логікою
        # Денні лічильники - зберігаємо тільки за останні 2 дні
        cutoff_date = (now - timedelta(days=2)).date()
    
        old_day_keys = []
        for key in list(self.daily_counts.keys()):
            if isinstance(key, str) and '_' in key:
                try:
                    # Більш надійний парсинг дати
                    parts = key.split('_')
                    if len(parts) >= 2:
                        date_part = parts[-1]
                        # Спробуємо різні формати дати
                        for date_format in ['%Y-%m-%d', '%Y%m%d']:
                            try:
                                key_date = datetime.strptime(date_part, date_format).date()
                                if key_date < cutoff_date:
                                    old_day_keys.append(key)
                                break
                            except ValueError:
                                continue
                except Exception as e:
                    # Видаляємо некоректні ключі
                    logger.debug(f"Видалення некоректного ключа: {key}")
                    old_day_keys.append(key)
    
        for key in old_day_keys:
            del self.daily_counts[key]
    
        # Годинні лічильники - зберігаємо тільки за останні 6 годин
        cutoff_hour = now - timedelta(hours=6)
        old_hour_keys = []
    
        for key in list(self.hourly_counts.keys()):
            if isinstance(key, str) and '_' in key:
                try:
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        time_str = parts[1]
                        # Пробуємо парсити як ISO формат
                        key_time = datetime.fromisoformat(time_str.replace('_', ' '))
                        if key_time < cutoff_hour:
                            old_hour_keys.append(key)
                except Exception:
                    # Видаляємо некоректні ключі
                    old_hour_keys.append(key)
    
        for key in old_hour_keys:
            del self.hourly_counts[key]
    
        # ВИПРАВЛЕННЯ: Обмеження розміру історії сигналів
        MAX_HISTORY_SIZE = 200  # Зменшено з 500
    
        for ticker in list(self.signal_history.keys()):
            history = self.signal_history[ticker]
            if len(history) > MAX_HISTORY_SIZE:
                # Залишаємо тільки останні записи
                # Конвертуємо в список, обрізаємо, конвертуємо назад в deque
                recent_items = list(history)[-MAX_HISTORY_SIZE:]
                self.signal_history[ticker] = deque(recent_items, maxlen=1000)
    
        # ВИПРАВЛЕННЯ: Додаємо глобальне обмеження пам'яті
        # Якщо використовується забагато пам'яті, видаляємо найстаріші дані
        memory_usage_estimate = (
            total_signals * 200 +  # ~200 байт на сигнал
            total_hashes * 50      # ~50 байт на хеш
        ) / (1024 * 1024)  # MB
    
        if memory_usage_estimate > 100:  # 100MB ліміт
            logger.warning(f"Високе використання пам'яті: ~{memory_usage_estimate:.1f}MB")
        
            # Сортуємо тікери по активності (останній сигнал)
            ticker_activity = []
            for ticker, history in self.signal_history.items():
                if history:
                    last_signal_time = history[-1]['timestamp']
                    ticker_activity.append((ticker, last_signal_time))
        
            ticker_activity.sort(key=lambda x: x[1])
        
            # Видаляємо 25% найменш активних тікерів
            to_remove = len(ticker_activity) // 4
            for ticker, _ in ticker_activity[:to_remove]:
                logger.info(f"Видалення даних для неактивного тікера: {ticker}")
                if ticker in self.signal_history:
                    del self.signal_history[ticker]
                if ticker in self.sent_hashes:
                    del self.sent_hashes[ticker]
    
        # Фінальна статистика
        total_signals_after = sum(len(hist) for hist in self.signal_history.values())
        total_hashes_after = sum(
            len(hashes) if isinstance(hashes, (set, list)) else len(hashes)
            for hashes in self.sent_hashes.values()
        )
    
        logger.info(f"Очищення завершено: {len(old_day_keys)} денних, {len(old_hour_keys)} годинних лічильників видалено")
        logger.info(f"Після очищення: {total_signals_after} сигналів, {total_hashes_after} хешів")
        logger.info(f"Активні тікери: {len(active_tickers)}")



class SignalEngine:
   """Головний клас системи сигналів"""
   
   def __init__(self, config_override: Optional[Dict] = None):
       self.config = SignalConfig()
       
       # Застосовуємо користувацькі налаштування
       if config_override:
           for key, value in config_override.items():
               if hasattr(self.config, key):
                   setattr(self.config, key, value)
       
       # Ініціалізація аналізаторів
       self.analyzers = [
           PriceChangeAnalyzer(self.config),
           PredictionAnalyzer(self.config),
           TechnicalAnalyzer(self.config),
           PatternAnalyzer(self.config),
           VolumeAnalyzer(self.config)
       ]
       
       # Фільтр та агрегатор сигналів
       self.filter = EnhancedSignalFilter(self.config)
       self.aggregator = SignalAggregator(self.config)
       
       # Історія для аналізу ефективності
       self.effectiveness_tracker = defaultdict(list)
       
       logger.info("SignalEngine ініціалізовано з покращеною дедуплікацією")
   
   def analyze_ticker(self, ticker: str, current_data: Dict, 
                     history: pd.DataFrame, predictions: Dict) -> List[Signal]:
       """Комплексний аналіз тікера"""
       all_signals = []
       
       # Підготовка даних для аналізаторів
       analysis_data = {
           'current_price': current_data.get('price', 0),
           'last_price': current_data.get('last_price', 0),
           'volume': current_data.get('volume', 0),
           'predictions': predictions,
           'indicators': self._extract_indicators(history)
       }
       
       # Запуск всіх аналізаторів
       for analyzer in self.analyzers:
           try:
               signals = analyzer.analyze(ticker, analysis_data, history)
               all_signals.extend(signals)
           except Exception as e:
               logger.error(f"Помилка в {analyzer.__class__.__name__}: {e}")
       
       # Агрегування схожих сигналів
       aggregated_signals = self.aggregator.aggregate_signals(all_signals)
       
       # Фільтрація та пріоритизація
       filtered_signals = self.filter.filter_signals(aggregated_signals)
       
       # Збагачення сигналів додатковою інформацією
       for signal in filtered_signals:
           self._enrich_signal(signal, history)
       
       # Відстеження ефективності
       self._track_effectiveness(ticker, filtered_signals)
       
       logger.info(f"Проаналізовано {len(all_signals)} сигналів, "
                  f"агреговано до {len(aggregated_signals)}, "
                  f"відфільтровано до {len(filtered_signals)} для {ticker}")
       
       return filtered_signals
   
   def _extract_indicators(self, history: pd.DataFrame) -> Dict:
       """Вилучення індикаторів з історії"""
       indicators = {}
       
       if len(history) < 20:
           return indicators
       
       # Беремо останні значення якщо вони є
       last_row = history.iloc[-1]
       
       for col in ['RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_upper', 
                   'BB_lower', 'BB_position', 'ATR']:
           if col in history.columns:
               indicators[col] = last_row[col]
       
       # Історія для деяких індикаторів
       if 'RSI' in history.columns:
           indicators['rsi_history'] = history['RSI'].tail(10).tolist()
       
       # Зберігаємо попередні значення для порівняння
       if len(history) >= 2:
           prev_row = history.iloc[-2]
           if 'MACD_diff' in history.columns:
               indicators['MACD_diff_prev'] = prev_row['MACD_diff']
       
       return indicators
   
   def _enrich_signal(self, signal: Signal, history: pd.DataFrame):
       """Збагачення сигналу додатковою інформацією"""
       # Додаємо історичний контекст
       if len(history) >= 252:  # Рік даних
           year_high = history['high'].tail(252).max()
           year_low = history['low'].tail(252).min()
           
           signal.data['year_high'] = year_high
           signal.data['year_low'] = year_low
           signal.data['year_range_position'] = (signal.current_price - year_low) / (year_high - year_low + 1e-10)
       
       # Додаємо часову мітку для групування
       signal.data['hour'] = signal.timestamp.hour
       signal.data['weekday'] = signal.timestamp.weekday()
   
   def _track_effectiveness(self, ticker: str, signals: List[Signal]):
       """Відстеження ефективності сигналів"""
       for signal in signals:
           self.effectiveness_tracker[ticker].append({
               'timestamp': signal.timestamp,
               'type': signal.type,
               'priority': signal.priority,
               'predicted_change': signal.price_change_percent,
               'confidence': signal.confidence
           })
   
   def get_statistics(self) -> Dict:
       """Отримання статистики роботи"""
       stats = {
           'total_signals': sum(len(self.filter.signal_history[t]) for t in self.filter.signal_history),
           'tickers_monitored': len(self.filter.signal_history),
           'signals_by_type': defaultdict(int),
           'signals_by_priority': defaultdict(int),
           'avg_confidence': 0,
           'active_config': {
               'base_thresholds': self.config.base_thresholds,
               'time_multipliers': self.config.time_multipliers
           }
       }
       
       # Підрахунок за типами та пріоритетами
       all_confidences = []
       
       for ticker_history in self.filter.signal_history.values():
           for signal_record in ticker_history:
               stats['signals_by_type'][signal_record['type'].value] += 1
               stats['signals_by_priority'][signal_record['priority'].value] += 1
       
       # Середня впевненість з історії ефективності
       for ticker_records in self.effectiveness_tracker.values():
           all_confidences.extend([r['confidence'] for r in ticker_records])
       
       if all_confidences:
           stats['avg_confidence'] = np.mean(all_confidences)
       
       return stats
   
   def update_config(self, new_config: Dict):
       """Оновлення конфігурації"""
       for key, value in new_config.items():
           if hasattr(self.config, key):
               setattr(self.config, key, value)
               logger.info(f"Оновлено параметр {key}: {value}")
   
   def reset_history(self, ticker: Optional[str] = None):
       """Скидання історії сигналів"""
       if ticker:
           if ticker in self.filter.signal_history:
               del self.filter.signal_history[ticker]
           if ticker in self.effectiveness_tracker:
               del self.effectiveness_tracker[ticker]
           
           # Очищення last_signals для цього тікера
           keys_to_remove = [k for k in self.filter.last_signals.keys() if ticker in k]
           for key in keys_to_remove:
               del self.filter.last_signals[key]
       else:
           # Повне скидання
           self.filter.signal_history.clear()
           self.filter.last_signals.clear()
           self.effectiveness_tracker.clear()
       
       logger.info(f"Історія сигналів скинута {'для ' + ticker if ticker else 'повністю'}")
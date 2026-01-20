"""
News Analyzer для Stock Monitor з використанням Polygon News API
"""
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from textstat import flesch_reading_ease
from collections import Counter

from config import POLYGON_API_KEY, CACHE_DIR
import os

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """Структура елементу новин"""
    title: str
    description: str
    url: str
    published_at: datetime
    tickers: List[str]
    publisher: str
    sentiment_score: float = 0.0
    importance_score: float = 0.0
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class NewsAnalyzer:
    """Аналізатор новин з Polygon API"""
    
    def __init__(self, api_key: str = POLYGON_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/reference/news"
        self.cache_dir = os.path.join(CACHE_DIR, "news")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Ключові слова для визначення важливості новин
        self.importance_keywords = {
            'earnings': 5.0,
            'revenue': 4.0,
            'profit': 4.0,
            'acquisition': 5.0,
            'merger': 5.0,
            'fda approval': 5.0,
            'partnership': 3.0,
            'contract': 3.0,
            'lawsuit': 4.0,
            'bankruptcy': 5.0,
            'dividend': 3.0,
            'split': 4.0,
            'ceo': 3.5,
            'management': 2.5,
            'guidance': 4.0,
            'forecast': 3.5,
            'upgrade': 4.0,
            'downgrade': 4.0,
            'target price': 3.0,
            'analyst': 2.5,
            'investigation': 4.0,
            'recall': 4.5,
            'expansion': 3.0,
            'closure': 4.0
        }
        
        # Позитивні та негативні слова для sentiment аналізу
        self.positive_words = [
            'growth', 'increase', 'profit', 'success', 'strong', 'positive', 
            'gain', 'rise', 'boost', 'improvement', 'expansion', 'achievement',
            'excellent', 'outstanding', 'record', 'milestone', 'breakthrough',
            'upgrade', 'bullish', 'optimistic', 'confident', 'progress'
        ]
        
        self.negative_words = [
            'decline', 'loss', 'decrease', 'weak', 'negative', 'fall', 
            'drop', 'concern', 'issue', 'problem', 'challenge', 'risk',
            'warning', 'alert', 'lawsuit', 'investigation', 'scandal',
            'downgrade', 'bearish', 'pessimistic', 'crisis', 'failure'
        ]
    
    def get_news(self, ticker: str = None, days_back: int = 7, limit: int = 50) -> List[NewsItem]:
        """
        Получить новости для тикера или общие новости
        
        Args:
            ticker: Тикер акции (например, AAPL)
            days_back: Количество дней назад для поиска новостей
            limit: Максимальное количество новостей
            
        Returns:
            Список новостных элементов
        """
        try:
            # Проверяем кеш
            cache_key = f"news_{ticker or 'general'}_{days_back}_{limit}"
            cached_news = self._get_cached_news(cache_key)
            if cached_news:
                logger.info(f"Используем кешированные новости для {ticker or 'general'}")
                return cached_news
            
            # Формируем параметры запроса
            params = {
                'apikey': self.api_key,
                'limit': limit,
                'order': 'desc',
                'sort': 'published_utc'
            }
            
            if ticker:
                params['ticker'] = ticker
            
            # Добавляем фильтр по времени
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            params['published_utc.gte'] = start_date.strftime('%Y-%m-%d')
            params['published_utc.lte'] = end_date.strftime('%Y-%m-%d')
            
            # Делаем запрос к API
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Ошибка API Polygon News: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            
            if 'results' not in data:
                logger.warning(f"Нет результатов в ответе API для {ticker}")
                return []
            
            # Обрабатываем новости
            news_items = []
            for item in data['results']:
                try:
                    news_item = self._parse_news_item(item)
                    if news_item:
                        news_items.append(news_item)
                except Exception as e:
                    logger.error(f"Помилка парсингу новини: {e}")
                    continue
            
            # Кешируем результат
            self._cache_news(cache_key, news_items)
            
            logger.info(f"Получено {len(news_items)} новостей для {ticker or 'general'}")
            return news_items
            
        except Exception as e:
            logger.error(f"Ошибка получения новостей: {e}")
            return []
    
    def _parse_news_item(self, item: Dict) -> Optional[NewsItem]:
        """Парсим элемент новости из API ответа"""
        try:
            # Извлекаем основные поля
            title = item.get('title', '')
            description = item.get('description', '')
            url = item.get('article_url', '')
            publisher = item.get('publisher', {}).get('name', 'Unknown')
            
            # Парсим дату публикации
            published_str = item.get('published_utc', '')
            try:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except:
                from datetime import timezone
                published_at = datetime.now(timezone.utc)
            
            # Извлекаем тикеры
            tickers = []
            if 'tickers' in item:
                tickers = [t.upper() for t in item['tickers']]
            
            # Создаем объект новости
            news_item = NewsItem(
                title=title,
                description=description,
                url=url,
                published_at=published_at,
                tickers=tickers,
                publisher=publisher
            )
            
            # Анализируем sentiment и важность
            news_item.sentiment_score = self._analyze_sentiment(title + ' ' + description)
            news_item.importance_score = self._calculate_importance(title + ' ' + description)
            news_item.keywords = self._extract_keywords(title + ' ' + description)
            
            return news_item
            
        except Exception as e:
            logger.error(f"Помилка парсингу новини: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Анализ тональности текста
        Возвращает значение от -1 (негативное) до 1 (позитивное)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Подсчитываем позитивные и негативные слова
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Учитываем интенсификаторы
        intensifiers = ['very', 'extremely', 'significantly', 'dramatically', 'substantially']
        intensity_multiplier = 1.0
        for intensifier in intensifiers:
            if intensifier in text_lower:
                intensity_multiplier += 0.2
        
        # Рассчитываем итоговый sentiment
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        sentiment *= intensity_multiplier
        
        # Ограничиваем от -1 до 1
        return max(-1.0, min(1.0, sentiment))
    
    def _calculate_importance(self, text: str) -> float:
        """
        Расчет важности новости на основе ключевых слов
        Возвращает значение от 0 до 5
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        importance = 0.0
        
        # Проверяем наличие важных ключевых слов
        for keyword, weight in self.importance_keywords.items():
            if keyword in text_lower:
                importance += weight
        
        # Учитываем длину и читаемость текста
        if len(text) > 200:
            importance += 0.5
        
        # Проверяем наличие числовых данных (цифры, проценты, доллары)
        if re.search(r'[\d\$%]', text):
            importance += 0.5
        
        # Ограничиваем максимальное значение
        return min(5.0, importance)
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Извлечение ключевых слов из текста"""
        if not text:
            return []
        
        # Удаляем знаки пунктуации и приводим к нижнему регистру
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Убираем стоп-слова
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'a', 'an', 'as', 'if', 'than', 'this', 'that', 'these', 'those'
        }
        
        # Фильтруем слова
        meaningful_words = [
            word for word in words 
            if len(word) > 3 and word not in stop_words and word.isalpha()
        ]
        
        # Подсчитываем частоту и возвращаем самые частые
        word_freq = Counter(meaningful_words)
        return [word for word, _ in word_freq.most_common(max_keywords)]
    
    def get_ticker_news_summary(self, ticker: str, days_back: int = 7) -> Dict:
        """
        Получить сводку новостей по тикеру
        
        Returns:
            Dict с ключевыми метриками новостей
        """
        try:
            news_items = self.get_news(ticker, days_back)
            
            if not news_items:
                return {
                    'total_news': 0,
                    'avg_sentiment': 0.0,
                    'avg_importance': 0.0,
                    'top_keywords': [],
                    'recent_news': [],
                    'sentiment_trend': 'neutral'
                }
            
            # Расчитываем метрики
            total_news = len(news_items)
            avg_sentiment = sum(item.sentiment_score for item in news_items) / total_news
            avg_importance = sum(item.importance_score for item in news_items) / total_news
            
            # Собираем все ключевые слова
            all_keywords = []
            for item in news_items:
                all_keywords.extend(item.keywords)
            
            top_keywords = [word for word, _ in Counter(all_keywords).most_common(10)]
            
            # Последние важные новости
            important_news = sorted(
                news_items, 
                key=lambda x: x.importance_score, 
                reverse=True
            )[:5]
            
            recent_news = [
                {
                    'title': item.title[:100] + '...' if len(item.title) > 100 else item.title,
                    'sentiment': item.sentiment_score,
                    'importance': item.importance_score,
                    'published': item.published_at.strftime('%Y-%m-%d %H:%M'),
                    'url': item.url
                }
                for item in important_news
            ]
            
            # Определяем тренд sentiment
            if avg_sentiment > 0.2:
                sentiment_trend = 'positive'
            elif avg_sentiment < -0.2:
                sentiment_trend = 'negative'
            else:
                sentiment_trend = 'neutral'
            
            return {
                'total_news': total_news,
                'avg_sentiment': round(avg_sentiment, 3),
                'avg_importance': round(avg_importance, 2),
                'top_keywords': top_keywords,
                'recent_news': recent_news,
                'sentiment_trend': sentiment_trend
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения сводки новостей для {ticker}: {e}")
            return {
                'total_news': 0,
                'avg_sentiment': 0.0,
                'avg_importance': 0.0,
                'top_keywords': [],
                'recent_news': [],
                'sentiment_trend': 'neutral'
            }
    
    def get_news_features_for_ml(self, ticker: str, days_back: int = 7) -> Dict:
        """
        Получить новостные фичи для ML модели
        
        Returns:
            Dict с новостными фичами
        """
        try:
            news_items = self.get_news(ticker, days_back)
            
            if not news_items:
                return {
                    'news_count_1d': 0,
                    'news_count_3d': 0,
                    'news_count_7d': 0,
                    'avg_sentiment_1d': 0.0,
                    'avg_sentiment_3d': 0.0,
                    'avg_sentiment_7d': 0.0,
                    'max_importance_1d': 0.0,
                    'max_importance_3d': 0.0,
                    'max_importance_7d': 0.0,
                    'sentiment_volatility': 0.0,
                    'positive_news_ratio': 0.0,
                    'negative_news_ratio': 0.0
                }
            
            from datetime import timezone
            now = datetime.now(timezone.utc)
            
            # Разделяем новости по временным периодам
            news_1d = [item for item in news_items if (now - item.published_at).days <= 1]
            news_3d = [item for item in news_items if (now - item.published_at).days <= 3]
            news_7d = news_items
            
            # Функция для расчета метрик по периоду
            def calculate_period_metrics(period_news):
                if not period_news:
                    return {
                        'count': 0,
                        'avg_sentiment': 0.0,
                        'max_importance': 0.0
                    }
                
                return {
                    'count': len(period_news),
                    'avg_sentiment': sum(item.sentiment_score for item in period_news) / len(period_news),
                    'max_importance': max(item.importance_score for item in period_news)
                }
            
            metrics_1d = calculate_period_metrics(news_1d)
            metrics_3d = calculate_period_metrics(news_3d)
            metrics_7d = calculate_period_metrics(news_7d)
            
            # Расчет дополнительных метрик
            sentiment_values = [item.sentiment_score for item in news_7d]
            sentiment_volatility = 0.0
            if len(sentiment_values) > 1:
                import numpy as np
                sentiment_volatility = float(np.std(sentiment_values))
            
            positive_news = sum(1 for item in news_7d if item.sentiment_score > 0.1)
            negative_news = sum(1 for item in news_7d if item.sentiment_score < -0.1)
            total_news = len(news_7d)
            
            positive_ratio = positive_news / total_news if total_news > 0 else 0.0
            negative_ratio = negative_news / total_news if total_news > 0 else 0.0
            
            return {
                'news_count_1d': metrics_1d['count'],
                'news_count_3d': metrics_3d['count'],
                'news_count_7d': metrics_7d['count'],
                'avg_sentiment_1d': round(metrics_1d['avg_sentiment'], 3),
                'avg_sentiment_3d': round(metrics_3d['avg_sentiment'], 3),
                'avg_sentiment_7d': round(metrics_7d['avg_sentiment'], 3),
                'max_importance_1d': round(metrics_1d['max_importance'], 2),
                'max_importance_3d': round(metrics_3d['max_importance'], 2),
                'max_importance_7d': round(metrics_7d['max_importance'], 2),
                'sentiment_volatility': round(sentiment_volatility, 3),
                'positive_news_ratio': round(positive_ratio, 3),
                'negative_news_ratio': round(negative_ratio, 3)
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения новостных фич для {ticker}: {e}")
            return {
                'news_count_1d': 0,
                'news_count_3d': 0,
                'news_count_7d': 0,
                'avg_sentiment_1d': 0.0,
                'avg_sentiment_3d': 0.0,
                'avg_sentiment_7d': 0.0,
                'max_importance_1d': 0.0,
                'max_importance_3d': 0.0,
                'max_importance_7d': 0.0,
                'sentiment_volatility': 0.0,
                'positive_news_ratio': 0.0,
                'negative_news_ratio': 0.0
            }
    
    def _get_cached_news(self, cache_key: str) -> Optional[List[NewsItem]]:
        """Получить новости из кеша"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            if not os.path.exists(cache_file):
                return None
            
            # Проверяем актуальность кеша (1 час)
            file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
            if file_age > 3600:  # 1 час
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Восстанавливаем объекты NewsItem
            news_items = []
            for item_data in cached_data:
                news_item = NewsItem(**item_data)
                # Восстанавливаем datetime с правильной timezone
                published_str = item_data['published_at']
                try:
                    if published_str.endswith('+00:00') or published_str.endswith('Z'):
                        news_item.published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                    else:
                        # Если timezone отсутствует, добавляем UTC
                        from datetime import timezone
                        news_item.published_at = datetime.fromisoformat(published_str).replace(tzinfo=timezone.utc)
                except:
                    from datetime import timezone
                    news_item.published_at = datetime.now(timezone.utc)
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Ошибка чтения кеша новостей: {e}")
            return None
    
    def _cache_news(self, cache_key: str, news_items: List[NewsItem]) -> None:
        """Кешировать новости"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            
            # Преобразуем в сериализуемый формат
            cached_data = []
            for item in news_items:
                item_dict = {
                    'title': item.title,
                    'description': item.description,
                    'url': item.url,
                    'published_at': item.published_at.isoformat(),
                    'tickers': item.tickers,
                    'publisher': item.publisher,
                    'sentiment_score': item.sentiment_score,
                    'importance_score': item.importance_score,
                    'keywords': item.keywords
                }
                cached_data.append(item_dict)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка кеширования новостей: {e}")
    
    def cleanup_cache(self, max_age_hours: int = 24) -> None:
        """Очистка старого кеша"""
        try:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logger.info(f"Удален старый кеш новостей: {filename}")
                        
        except Exception as e:
            logger.error(f"Ошибка очистки кеша новостей: {e}")


# Глобальный экземпляр анализатора новостей
news_analyzer = NewsAnalyzer()
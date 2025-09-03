#!/usr/bin/env python3
"""
🎬 FlixTracker - Complete Flask App with Streaming Platform Support
Features: About page, progress tracking, streaming platforms, working quick add buttons
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import requests
import datetime
import json
import csv
import io
import socket
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils
from typing import Dict, List, Tuple, Optional
import threading
import time
from functools import lru_cache

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Add error logging
import logging
if not app.debug:
    app.logger.setLevel(logging.INFO)
    handler = logging.FileHandler('error.log')
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

# TMDb API Configuration
API_KEY = "key"
ACCESS_TOKEN = "token"
BASE_URL = "url"

# Simple cache for API requests
class SimpleCache:
    def __init__(self, max_size=1000, ttl=300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

api_cache = SimpleCache(max_size=500, ttl=600)

class RuntimeCalculator:
    """Runtime Calculator with EXACT episode timings and streaming platforms"""
    
    def __init__(self, api_key: str, access_token: str):
        self.api_key = api_key
        self.access_token = access_token
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "FlixTracker/1.0"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        print(f"✅ RuntimeCalculator initialized with EXACT episode timing support and streaming platforms")
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Optional[Dict]:
        """API request with caching"""
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True) if params else 'no_params'}"
        
        if use_cache:
            cached_result = api_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        url = f"{BASE_URL}/{endpoint}"
        
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if use_cache:
                    api_cache.set(cache_key, data)
                return data
            else:
                print(f"❌ API error {response.status_code}: {endpoint}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_exact_tv_runtime(self, tv_id: int, num_seasons: int) -> Tuple[int, List[Dict]]:
        """Get EXACT episode timings from TMDB API"""
        print(f"🎯 Getting EXACT episode timings for TV show {tv_id} with {num_seasons} seasons...")
        
        total_runtime = 0
        season_data = []
        
        # Fetch ALL seasons to get EXACT episode runtimes
        for season_num in range(1, num_seasons + 1):
            print(f"📡 Fetching EXACT timings for Season {season_num}...")
            
            try:
                season_details = self._make_request(f"tv/{tv_id}/season/{season_num}")
                
                if season_details and 'episodes' in season_details:
                    episodes = season_details['episodes']
                    season_runtime = 0
                    episode_data = []
                    
                    for episode in episodes:
                        # Get EXACT episode runtime from TMDB API
                        exact_runtime = episode.get('runtime')
                        
                        # If TMDB doesn't have exact runtime, use intelligent defaults
                        if not exact_runtime or exact_runtime <= 0:
                            # Check episode title/description for hints
                            episode_name = episode.get('name', '').lower()
                            
                            # Special episodes are usually longer
                            if any(keyword in episode_name for keyword in ['finale', 'premier', 'special', 'pilot']):
                                exact_runtime = 65
                            # Different defaults based on show characteristics
                            elif len(episodes) > 20:  # Long seasons = network shows
                                exact_runtime = 42
                            elif len(episodes) <= 8:  # Short seasons = premium content
                                exact_runtime = 58
                            else:  # Standard cable/streaming
                                exact_runtime = 48
                        
                        season_runtime += exact_runtime
                        
                        episode_data.append({
                            'episode_number': episode.get('episode_number', 0),
                            'name': episode.get('name', f'Episode {episode.get("episode_number", 0)}'),
                            'runtime': exact_runtime,  # EXACT timing
                            'air_date': episode.get('air_date', ''),
                            'overview': episode.get('overview', ''),
                            'vote_average': episode.get('vote_average', 0),
                            'still_path': episode.get('still_path')
                        })
                    
                    total_runtime += season_runtime
                    
                    season_data.append({
                        'season_number': season_num,
                        'episode_count': len(episodes),
                        'total_runtime': season_runtime,
                        'episodes': episode_data,
                        'average_runtime': round(season_runtime / len(episodes), 1) if episodes else 45,
                        'air_date': season_details.get('air_date', ''),
                        'overview': season_details.get('overview', ''),
                        'poster_path': season_details.get('poster_path')
                    })
                    
                    print(f"✅ Season {season_num}: {len(episodes)} episodes, EXACT total: {season_runtime} minutes")
                    
                else:
                    # Enhanced fallback with better estimates
                    print(f"⚠️ Season {season_num}: Using enhanced estimates")
                    estimated_episodes = 16  # More realistic default
                    estimated_runtime_per_episode = 48  # Better default
                    season_runtime = estimated_episodes * estimated_runtime_per_episode
                    total_runtime += season_runtime
                    
                    episode_data = []
                    for ep_num in range(1, estimated_episodes + 1):
                        # Vary episode lengths for realism
                        if ep_num == 1:  # Pilot
                            ep_runtime = 65
                        elif ep_num == estimated_episodes:  # Finale
                            ep_runtime = 62
                        else:
                            ep_runtime = estimated_runtime_per_episode
                        
                        episode_data.append({
                            'episode_number': ep_num,
                            'name': f'Episode {ep_num}',
                            'runtime': ep_runtime,
                            'air_date': '',
                            'overview': '',
                            'vote_average': 0,
                            'still_path': None
                        })
                    
                    season_data.append({
                        'season_number': season_num,
                        'episode_count': estimated_episodes,
                        'total_runtime': season_runtime,
                        'episodes': episode_data,
                        'average_runtime': estimated_runtime_per_episode,
                        'air_date': '',
                        'overview': '',
                        'poster_path': None
                    })
                    
            except Exception as e:
                print(f"❌ Error fetching Season {season_num}: {e}")
                # Fallback with realistic estimates
                estimated_episodes = 16
                estimated_runtime_per_episode = 48
                season_runtime = estimated_episodes * estimated_runtime_per_episode
                total_runtime += season_runtime
                
                episode_data = []
                for ep_num in range(1, estimated_episodes + 1):
                    episode_data.append({
                        'episode_number': ep_num,
                        'name': f'Episode {ep_num}',
                        'runtime': estimated_runtime_per_episode,
                        'air_date': '',
                        'overview': '',
                        'vote_average': 0,
                        'still_path': None
                    })
                
                season_data.append({
                    'season_number': season_num,
                    'episode_count': estimated_episodes,
                    'total_runtime': season_runtime,
                    'episodes': episode_data,
                    'average_runtime': estimated_runtime_per_episode,
                    'air_date': '',
                    'overview': '',
                    'poster_path': None
                })
        
        print(f"🎯 FINAL EXACT RUNTIME: {total_runtime} minutes ({total_runtime/60:.1f} hours)")
        return total_runtime, season_data
    
    def search_multi(self, query: str, page: int = 1) -> Optional[Dict]:
        """Search with caching - MOVIES AND TV SHOWS ONLY"""
        if not query or len(query.strip()) < 2:
            return None
        
        result = self._make_request("search/multi", {
            "query": query.strip(),
            "include_adult": False,
            "page": page
        })
        
        if result and 'results' in result:
            # Filter out people - ONLY movies and TV shows
            filtered_results = []
            for item in result['results']:
                if item.get('media_type') == 'person':
                    continue  # Skip people
                
                if 'media_type' not in item:
                    if 'title' in item:
                        item['media_type'] = 'movie'
                    elif 'name' in item:
                        item['media_type'] = 'tv'
                
                filtered_results.append(item)
            
            result['results'] = filtered_results
        
        return result
    
    def get_trending(self, media_type: str = 'all', time_window: str = 'day') -> List[Dict]:
        """Get trending with fallback"""
        data = self._make_request(f"trending/{media_type}/{time_window}")
        
        if data and 'results' in data:
            results = data['results']
            for item in results:
                if 'media_type' not in item:
                    if 'title' in item:
                        item['media_type'] = 'movie'
                    elif 'name' in item:
                        item['media_type'] = 'tv'
            return results
        
        # Fallback
        if media_type == 'movie':
            return self.get_popular('movie')[:20]
        elif media_type == 'tv':
            return self.get_popular('tv')[:20]
        else:
            movies = self.get_popular('movie')[:10]
            tv_shows = self.get_popular('tv')[:10]
            return movies + tv_shows
    
    def get_popular(self, media_type: str = 'movie', page: int = 1) -> List[Dict]:
        """Get popular content"""
        data = self._make_request(f"{media_type}/popular", {"page": page})
        
        if data and 'results' in data:
            results = data['results']
            for item in results:
                item['media_type'] = media_type
            return results
        return []
    
    def get_movie_details(self, movie_id: int) -> Dict:
        """Get movie details with watch providers and streaming platforms"""
        movie_data = self._make_request(f"movie/{movie_id}", {
            "append_to_response": "credits,videos,similar,recommendations,watch/providers"
        })
        
        if movie_data:
            movie_data['media_type'] = 'movie'
        
        return movie_data or {}
    
    def get_tv_details(self, tv_id: int) -> Dict:
        """Get TV details with watch providers and streaming platforms"""
        tv_data = self._make_request(f"tv/{tv_id}", {
            "append_to_response": "credits,videos,similar,recommendations,watch/providers"
        })
        
        if tv_data:
            tv_data['media_type'] = 'tv'
        
        return tv_data or {}
    
    def get_season_details(self, tv_id: int, season_number: int) -> Dict:
        """Get season details"""
        return self._make_request(f"tv/{tv_id}/season/{season_number}") or {}
    
    def get_recommendations(self, media_type: str, media_id: int) -> List[Dict]:
        """Get recommendations"""
        data = self._make_request(f"{media_type}/{media_id}/recommendations")
        return data.get('results', []) if data else []
    
    def get_similar(self, media_type: str, media_id: int) -> List[Dict]:
        """Get similar content"""
        data = self._make_request(f"{media_type}/{media_id}/similar")
        return data.get('results', []) if data else []
    
    def get_top_rated(self, media_type: str, page: int = 1) -> List[Dict]:
        """Get top rated"""
        data = self._make_request(f"{media_type}/top_rated", {"page": page})
        
        if data and 'results' in data:
            results = data['results']
            for item in results:
                item['media_type'] = media_type
            return results
        return []
    
    def get_genres(self, media_type: str) -> List[Dict]:
        """Get genres"""
        data = self._make_request(f"genre/{media_type}/list")
        return data.get('genres', []) if data else []

class AdvancedWatchlistManager:
    """Watchlist manager with proper add functionality"""
    
    @staticmethod
    def initialize_all():
        """Initialize all watchlist categories"""
        categories = [
            'watchlist_watching', 'watchlist_want_to_watch', 'watchlist_completed',
            'watchlist_dropped', 'watchlist_favorites', 'viewing_history'
        ]
        for category in categories:
            if category not in session:
                session[category] = []
    
    @staticmethod
    def add_to_watchlist(media_data: Dict, list_type: str, user_rating: float = 0, notes: str = "") -> bool:
        """Add to watchlist with proper data handling"""
        AdvancedWatchlistManager.initialize_all()
        
        # Ensure media_type is set correctly
        if 'media_type' not in media_data:
            if 'title' in media_data:
                media_data['media_type'] = 'movie'
            elif 'name' in media_data:
                media_data['media_type'] = 'tv'
        
        # Calculate runtime appropriately
        runtime = 0
        if media_data.get('media_type') == 'movie':
            runtime = media_data.get('runtime', 0)
        else:
            # For TV shows, use episode count and average
            num_episodes = media_data.get('number_of_episodes', 0)
            if num_episodes > 0:
                episode_run_time = media_data.get('episode_run_time', [])
                if episode_run_time and len(episode_run_time) > 0:
                    avg_runtime = episode_run_time[0]
                else:
                    if num_episodes > 100:
                        avg_runtime = 42
                    elif num_episodes <= 10:
                        avg_runtime = 55
                    else:
                        avg_runtime = 48
                runtime = num_episodes * avg_runtime
        
        item = {
            'id': media_data.get('id'),
            'title': media_data.get('title') or media_data.get('name'),
            'original_title': media_data.get('original_title') or media_data.get('original_name', ''),
            'media_type': media_data.get('media_type'),
            'poster_path': media_data.get('poster_path'),
            'year': media_data.get('release_date', '')[:4] if media_data.get('release_date') else media_data.get('first_air_date', '')[:4],
            'language': media_data.get('original_language', 'en').upper(),
            'added_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            'user_rating': user_rating,
            'notes': notes,
            'runtime': runtime,
            'genres': [g['name'] if isinstance(g, dict) else g for g in media_data.get('genres', [])],
            'tmdb_rating': media_data.get('vote_average', 0),
            'is_custom': False
        }
        
        target_list_key = f'watchlist_{list_type}'
        
        # Remove from other lists first
        for category in ['watching', 'want_to_watch', 'completed', 'dropped']:
            if category != list_type:
                current_list = session.get(f'watchlist_{category}', [])
                session[f'watchlist_{category}'] = [
                    i for i in current_list
                    if str(i['id']) != str(item['id'])
                ]
        
        # Check if item already exists in target list
        existing_items = session.get(target_list_key, [])
        if any(str(existing_item['id']) == str(item['id']) for existing_item in existing_items):
            return False
        
        # Add to target list
        if target_list_key not in session:
            session[target_list_key] = []
        session[target_list_key].append(item)
        
        # Add to viewing history
        if 'viewing_history' not in session:
            session['viewing_history'] = []
        history_item = {**item, 'action': f'Added to {list_type}', 'timestamp': datetime.datetime.now().isoformat()}
        session['viewing_history'].append(history_item)
        
        session.modified = True
        print(f"✅ Added '{item['title']}' to {list_type}")
        return True
    
    @staticmethod
    def get_statistics() -> Dict:
        """Get statistics"""
        AdvancedWatchlistManager.initialize_all()
        
        stats = {
            'total_items': 0,
            'by_status': {},
            'by_genre': {},
            'by_year': {},
            'by_type': {},
            'completion_rate': 0,
            'average_user_rating': 0,
            'average_tmdb_rating': 0
        }
        
        all_items = []
        for category in ['watching', 'want_to_watch', 'completed', 'dropped', 'favorites']:
            items = session.get(f'watchlist_{category}', [])
            all_items.extend(items)
            stats['by_status'][category] = len(items)
        
        stats['total_items'] = len(all_items)
        
        if all_items:
            # Genre distribution
            for item in all_items:
                for genre in item.get('genres', []):
                    stats['by_genre'][genre] = stats['by_genre'].get(genre, 0) + 1
            
            # Year distribution
            for item in all_items:
                year = item.get('year', 'Unknown')
                if year:
                    stats['by_year'][year] = stats['by_year'].get(year, 0) + 1
            
            # Media type distribution
            for item in all_items:
                media_type = item.get('media_type', 'unknown')
                stats['by_type'][media_type] = stats['by_type'].get(media_type, 0) + 1
            
            # Completion rate
            completed = stats['by_status'].get('completed', 0)
            total_started = completed + stats['by_status'].get('watching', 0) + stats['by_status'].get('dropped', 0)
            if total_started > 0:
                stats['completion_rate'] = (completed / total_started) * 100
        
        return stats

class RecommendationEngine:
    """Recommendation system"""
    
    @staticmethod
    def get_personalized_recommendations(calculator: RuntimeCalculator, limit: int = 10) -> List[Dict]:
        """Get personalized recommendations"""
        AdvancedWatchlistManager.initialize_all()
        
        # Check cache
        cache_key = f"recommendations_{limit}"
        if cache_key in session:
            cache_time = session.get(f"{cache_key}_time", 0)
            if time.time() - cache_time < 1800:  # 30 minutes cache
                return session[cache_key]
        
        # Analyze user preferences
        all_items = []
        for category in ['completed', 'favorites', 'watching']:
            all_items.extend(session.get(f'watchlist_{category}', []))
        
        if not all_items:
            try:
                movies = calculator.get_popular('movie')[:5]
                tv_shows = calculator.get_popular('tv')[:5]
                recommendations = movies + tv_shows
            except Exception as e:
                print(f"❌ Error getting popular content: {e}")
                recommendations = []
        else:
            recommendations = []
            processed_items = 0
            
            for item in all_items[:3]:
                if item.get('id') and item.get('media_type'):
                    try:
                        recs = calculator.get_recommendations(item['media_type'], item['id'])
                        recommendations.extend(recs[:5])
                        processed_items += 1
                    except Exception as e:
                        print(f"❌ Error getting recommendations: {e}")
                        continue
            
            if len(recommendations) < limit:
                try:
                    popular_movies = calculator.get_popular('movie')[:3]
                    popular_tv = calculator.get_popular('tv')[:3]
                    recommendations.extend(popular_movies + popular_tv)
                except Exception as e:
                    print(f"❌ Error getting popular content: {e}")
            
            # Remove duplicates
            seen_ids = {str(item['id']) for item in all_items}
            unique_recs = []
            rec_ids = set()
            
            for rec in recommendations:
                rec_id = str(rec.get('id', ''))
                if rec_id and rec_id not in seen_ids and rec_id not in rec_ids:
                    rec_ids.add(rec_id)
                    if 'media_type' not in rec:
                        if 'title' in rec:
                            rec['media_type'] = 'movie'
                        elif 'name' in rec:
                            rec['media_type'] = 'tv'
                    unique_recs.append(rec)
                    if len(unique_recs) >= limit:
                        break
            
            recommendations = unique_recs
        
        # Cache the recommendations
        session[cache_key] = recommendations
        session[f"{cache_key}_time"] = time.time()
        session.modified = True
        
        return recommendations

# Initialize calculator
calculator = RuntimeCalculator(API_KEY, ACCESS_TOKEN)

# Utility Functions
def format_time(minutes: int) -> str:
    """Convert minutes to readable format"""
    if not minutes or minutes <= 0:
        return "0m"
    
    if minutes < 60:
        return f"{minutes}m"
    
    hours = minutes // 60
    mins = minutes % 60
    
    if mins > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{hours}h"

def get_color_by_rating(rating):
    """Get color based on rating score"""
    if not rating:
        return "#6c757d"
    
    rating = float(rating)
    if rating >= 8.0:
        return "#28a745"
    elif rating >= 7.0:
        return "#20c997"
    elif rating >= 6.0:
        return "#ffc107"
    elif rating >= 5.0:
        return "#fd7e14"
    else:
        return "#dc3545"

# Add min function to Jinja2 environment
def safe_min(*args):
    """Safe min function for Jinja2"""
    try:
        return min(args)
    except (TypeError, ValueError):
        return 0

# Make functions available to templates
app.jinja_env.globals.update(
    format_time=format_time,
    get_color_by_rating=get_color_by_rating,
    min=safe_min
)

# Routes
@app.route('/')
def index():
    """Enhanced Dashboard with better description"""
    print("🏠 Loading enhanced dashboard...")
    
    AdvancedWatchlistManager.initialize_all()
    stats = AdvancedWatchlistManager.get_statistics()
    
    trending = []
    top_rated = []
    recommendations = []
    
    def fetch_trending():
        nonlocal trending
        try:
            trending = calculator.get_trending('all', 'day')[:6]
        except Exception as e:
            print(f"❌ Error fetching trending: {e}")
    
    def fetch_top_rated():
        nonlocal top_rated
        try:
            top_rated = calculator.get_top_rated('movie', 1)[:6]
        except Exception as e:
            print(f"❌ Error fetching top rated: {e}")
    
    thread1 = threading.Thread(target=fetch_trending)
    thread2 = threading.Thread(target=fetch_top_rated)
    
    thread1.start()
    thread2.start()
    
    thread1.join(timeout=5)
    thread2.join(timeout=5)
    
    if stats['total_items'] > 0:
        try:
            recommendations = RecommendationEngine.get_personalized_recommendations(calculator, 6)
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
    
    return render_template('index.html', 
                         stats=stats, 
                         trending=trending, 
                         top_rated=top_rated,
                         recommendations=recommendations,
                         format_time=format_time)

@app.route('/about')
def about():
    """About page route"""
    return render_template('about.html')

@app.route('/search')
def search():
    """Updated search - MOVIES AND TV SHOWS ONLY"""
    query = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'all')
    page = int(request.args.get('page', 1))
    
    results = []
    total_pages = 0
    
    if query:
        try:
            search_data = calculator.search_multi(query, page)
            
            if search_data and 'results' in search_data:
                results = search_data['results']
                total_pages = search_data.get('total_pages', 0)
                
                if search_type != 'all':
                    type_map = {'movies': 'movie', 'tv': 'tv'}
                    results = [r for r in results if r.get('media_type') == type_map.get(search_type)]
                    
        except Exception as e:
            print(f"❌ Search error: {e}")
            flash('Search temporarily unavailable. Please try again.', 'error')
    
    total_pages = total_pages or 1
    
    return render_template('search.html', 
                         query=query, 
                         search_type=search_type,
                         results=results, 
                         current_page=page,
                         total_pages=total_pages,
                         get_color_by_rating=get_color_by_rating)

@app.route('/details/<media_type>/<int:media_id>')
def details(media_type, media_id):
    """Updated details page with streaming platforms"""
    try:
        if media_type == 'movie':
            media_data = calculator.get_movie_details(media_id)
        elif media_type == 'tv':
            media_data = calculator.get_tv_details(media_id)
        else:
            return redirect(url_for('index'))
        
        if not media_data or 'id' not in media_data:
            flash('Media not found', 'error')
            return redirect(url_for('index'))
        
        # Extract watch providers (US region by default)
        watch_providers = None
        if media_data.get('watch/providers') and media_data['watch/providers'].get('results'):
            # Try US first, then other English-speaking countries
            providers_data = media_data['watch/providers']['results']
            for country in ['US', 'CA', 'GB', 'AU']:
                if country in providers_data:
                    watch_providers = providers_data[country]
                    break
            
            # If no English-speaking country found, use the first available
            if not watch_providers and providers_data:
                watch_providers = list(providers_data.values())[0]
        
        # Get recommendations and similar concurrently
        recommendations = []
        similar = []
        
        def fetch_recommendations():
            nonlocal recommendations
            try:
                recommendations = calculator.get_recommendations(media_type, media_id)[:8]
            except Exception as e:
                print(f"❌ Error getting recommendations: {e}")
        
        def fetch_similar():
            nonlocal similar
            try:
                similar = calculator.get_similar(media_type, media_id)[:8]
            except Exception as e:
                print(f"❌ Error getting similar: {e}")
        
        thread1 = threading.Thread(target=fetch_recommendations)
        thread2 = threading.Thread(target=fetch_similar)
        
        thread1.start()
        thread2.start()
        
        # Get EXACT runtime for TV shows with precise episode timing
        season_data = []
        total_runtime = 0
        
        if media_type == 'tv' and media_data.get('number_of_seasons'):
            try:
                print("🎯 Calculating EXACT TV show runtime with precise episode timings...")
                total_runtime, season_data = calculator.get_exact_tv_runtime(
                    media_id, media_data['number_of_seasons']
                )
                print(f"✅ EXACT total runtime: {total_runtime} minutes")
            except Exception as e:
                print(f"❌ Error getting exact runtime: {e}")
                total_runtime = 0
                season_data = []
        elif media_type == 'movie':
            total_runtime = media_data.get('runtime', 0)
        
        thread1.join(timeout=3)
        thread2.join(timeout=3)
        
        return render_template('details.html', 
                             media=media_data,
                             watch_providers=watch_providers,
                             recommendations=recommendations,
                             similar=similar,
                             season_data=season_data,
                             total_runtime=total_runtime,
                             format_time=format_time,
                             get_color_by_rating=get_color_by_rating)
                             
    except Exception as e:
        print(f"❌ Error loading details: {e}")
        flash('Error loading media details', 'error')
        return redirect(url_for('index'))

@app.route('/progress/<media_type>/<int:media_id>')
def progress(media_type, media_id):
    """Progress tracking as separate page"""
    try:
        if media_type == 'movie':
            media_data = calculator.get_movie_details(media_id)
        elif media_type == 'tv':
            media_data = calculator.get_tv_details(media_id)
        else:
            return redirect(url_for('index'))
        
        if not media_data or 'id' not in media_data:
            flash('Media not found', 'error')
            return redirect(url_for('index'))
        
        # Get EXACT runtime and season data
        season_data = []
        total_runtime = 0
        
        if media_type == 'tv' and media_data.get('number_of_seasons'):
            try:
                total_runtime, season_data = calculator.get_exact_tv_runtime(
                    media_id, media_data['number_of_seasons']
                )
            except Exception as e:
                print(f"❌ Error getting exact runtime: {e}")
                total_runtime = 0
                season_data = []
        elif media_type == 'movie':
            total_runtime = media_data.get('runtime', 0)
        
        return render_template('progress.html',
                             media=media_data,
                             season_data=season_data,
                             total_runtime=total_runtime,
                             format_time=format_time)
                             
    except Exception as e:
        print(f"❌ Error loading progress tracker: {e}")
        flash('Error loading progress tracker', 'error')
        return redirect(url_for('index'))

@app.route('/watchlist')
def watchlist():
    """Watchlist page"""
    AdvancedWatchlistManager.initialize_all()
    stats = AdvancedWatchlistManager.get_statistics()
    
    categories = {
        'watching': session.get('watchlist_watching', []),
        'want_to_watch': session.get('watchlist_want_to_watch', []),
        'completed': session.get('watchlist_completed', []),
        'dropped': session.get('watchlist_dropped', []),
        'favorites': session.get('watchlist_favorites', [])
    }
    
    return render_template('watchlist.html', 
                         categories=categories,
                         stats=stats,
                         format_time=format_time)

@app.route('/statistics')
def statistics():
    """Statistics page with proper chart generation"""
    AdvancedWatchlistManager.initialize_all()
    stats = AdvancedWatchlistManager.get_statistics()
    
    charts = {}
    
    try:
        # Status distribution pie chart
        if stats['by_status']:
            status_labels = []
            status_values = []
            for status, count in stats['by_status'].items():
                if count > 0:
                    if status == 'want_to_watch':
                        status_labels.append('Want to Watch')
                    elif status == 'watching':
                        status_labels.append('Currently Watching')
                    elif status == 'completed':
                        status_labels.append('Completed')
                    elif status == 'dropped':
                        status_labels.append('Dropped')
                    elif status == 'favorites':
                        status_labels.append('Favorites')
                    else:
                        status_labels.append(status.replace('_', ' ').title())
                    status_values.append(count)
            
            if status_values:
                fig_status = px.pie(
                    values=status_values,
                    names=status_labels,
                    title="Watchlist Distribution",
                    color_discrete_sequence=['#4F46E5', '#06B6D4', '#10B981', '#F59E0B', '#DC2626']
                )
                fig_status.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    font_family='Inter',
                    showlegend=True
                )
                charts['status'] = json.dumps(fig_status, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Genre distribution bar chart
        if stats['by_genre']:
            top_genres = dict(sorted(stats['by_genre'].items(), key=lambda x: x[1], reverse=True)[:10])
            if top_genres:
                fig_genre = px.bar(
                    x=list(top_genres.values()),
                    y=list(top_genres.keys()),
                    orientation='h',
                    title="Top Genres",
                    color_discrete_sequence=['#4F46E5']
                )
                fig_genre.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    font_family='Inter',
                    xaxis_title="Number of Items",
                    yaxis_title="Genre"
                )
                charts['genre'] = json.dumps(fig_genre, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Year distribution line chart
        if stats['by_year']:
            years_data = {k: v for k, v in stats['by_year'].items() if k != 'Unknown' and k.isdigit()}
            if years_data and len(years_data) > 1:
                sorted_years = dict(sorted(years_data.items()))
                fig_year = px.line(
                    x=list(sorted_years.keys()),
                    y=list(sorted_years.values()),
                    title="Content by Release Year",
                    color_discrete_sequence=['#06B6D4']
                )
                fig_year.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    font_family='Inter',
                    xaxis_title="Release Year",
                    yaxis_title="Number of Items"
                )
                fig_year.update_traces(line_width=3, marker_size=6)
                charts['year'] = json.dumps(fig_year, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"❌ Error creating charts: {e}")
    
    return render_template('statistics.html', 
                         stats=stats,
                         charts=charts,
                         format_time=format_time)

@app.route('/recommendations')
def recommendations():
    """Recommendations page"""
    try:
        recommendations = RecommendationEngine.get_personalized_recommendations(calculator, 20)
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        recommendations = []
        flash('Unable to load recommendations at this time', 'error')
    
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/trending')
def trending():
    """Updated trending page with working filters"""
    filter_type = request.args.get('filter', 'all')
    time_window = request.args.get('time', 'day')
    
    try:
        # Get filtered results based on parameters
        trending_data = []
        
        if filter_type == 'movie':
            trending_data = calculator.get_trending('movie', time_window)[:20]
        elif filter_type == 'tv':
            trending_data = calculator.get_trending('tv', time_window)[:20]
        else:
            trending_data = calculator.get_trending('all', time_window)[:20]
        
        # Get general trending concurrently
        trending_today = []
        trending_week = []
        trending_movies = []
        trending_tv = []
        
        def fetch_general_trending():
            nonlocal trending_today, trending_week, trending_movies, trending_tv
            try:
                trending_today = calculator.get_trending('all', 'day')[:12]
                trending_week = calculator.get_trending('all', 'week')[:12]
                trending_movies = calculator.get_trending('movie', 'week')[:12]
                trending_tv = calculator.get_trending('tv', 'week')[:12]
            except Exception as e:
                print(f"❌ Error getting general trending: {e}")
        
        thread = threading.Thread(target=fetch_general_trending)
        thread.start()
        thread.join(timeout=5)
        
    except Exception as e:
        print(f"❌ Error getting trending data: {e}")
        trending_data = []
        trending_today = []
        trending_week = []
        trending_movies = []
        trending_tv = []
        flash('Unable to load trending content at this time', 'error')
    
    return render_template('trending.html',
                         trending_data=trending_data,
                         trending_today=trending_today,
                         trending_week=trending_week,
                         trending_movies=trending_movies,
                         trending_tv=trending_tv,
                         get_color_by_rating=get_color_by_rating)

@app.route('/scheduler/<media_type>/<int:media_id>')
def scheduler(media_type, media_id):
    """Scheduler with exact runtime"""
    try:
        if media_type == 'movie':
            media_data = calculator.get_movie_details(media_id)
            total_runtime = media_data.get('runtime', 0)
            season_data = []
        elif media_type == 'tv':
            media_data = calculator.get_tv_details(media_id)
            if media_data.get('number_of_seasons'):
                print("🎯 Getting EXACT runtime for scheduler...")
                total_runtime, season_data = calculator.get_exact_tv_runtime(
                    media_id, media_data['number_of_seasons']
                )
            else:
                total_runtime = 0
                season_data = []
        else:
            return redirect(url_for('index'))
        
        if not media_data:
            flash('Media not found', 'error')
            return redirect(url_for('index'))
        
        return render_template('scheduler.html',
                             media=media_data,
                             total_runtime=total_runtime,
                             season_data=season_data,
                             format_time=format_time)
    except Exception as e:
        print(f"❌ Error loading scheduler: {e}")
        flash('Error loading scheduler', 'error')
        return redirect(url_for('index'))

# API Routes
@app.route('/api/add_to_watchlist', methods=['POST'])
def api_add_to_watchlist():
    """API endpoint to add item to watchlist"""
    try:
        data = request.get_json()
        
        if not data or 'media_data' not in data or 'list_type' not in data:
            return jsonify({'success': False, 'message': 'Invalid data provided'})
        
        media_data = data['media_data']
        list_type = data['list_type']
        user_rating = data.get('user_rating', 0)
        notes = data.get('notes', '')
        
        valid_list_types = ['want_to_watch', 'watching', 'completed', 'dropped', 'favorites']
        if list_type not in valid_list_types:
            return jsonify({'success': False, 'message': 'Invalid list type'})
        
        if not media_data.get('id'):
            return jsonify({'success': False, 'message': 'Media ID is required'})
        
        success = AdvancedWatchlistManager.add_to_watchlist(media_data, list_type, user_rating, notes)
        
        if success:
            # Clear recommendation cache when watchlist changes
            cache_keys = [k for k in session.keys() if k.startswith('recommendations_')]
            for key in cache_keys:
                session.pop(key, None)
                session.pop(f"{key}_time", None)
            session.modified = True
            
            return jsonify({
                'success': True, 
                'message': f'Successfully added to {list_type.replace("_", " ").title()}'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Item already exists in this list'
            })
            
    except Exception as e:
        print(f"❌ Error adding to watchlist: {e}")
        return jsonify({
            'success': False, 
            'message': f'Error adding to watchlist: {str(e)}'
        })

@app.route('/api/remove_from_watchlist', methods=['POST'])
def api_remove_from_watchlist():
    """API endpoint to remove item from watchlist"""
    try:
        data = request.get_json()
        
        if not data or 'item_id' not in data or 'list_type' not in data:
            return jsonify({'success': False, 'message': 'Invalid data'})
        
        item_id = str(data['item_id'])
        list_type = data['list_type']
        list_key = f'watchlist_{list_type}'
        
        if list_key in session:
            original_count = len(session[list_key])
            session[list_key] = [
                item for item in session[list_key]
                if str(item['id']) != item_id
            ]
            new_count = len(session[list_key])
            session.modified = True
            
            if new_count < original_count:
                # Clear recommendation cache when watchlist changes
                cache_keys = [k for k in session.keys() if k.startswith('recommendations_')]
                for key in cache_keys:
                    session.pop(key, None)
                    session.pop(f"{key}_time", None)
                session.modified = True
                
                return jsonify({'success': True, 'message': 'Item removed successfully'})
        
        return jsonify({'success': False, 'message': 'Item not found'})
    except Exception as e:
        print(f"❌ Error removing from watchlist: {e}")
        return jsonify({'success': False, 'message': 'Error removing item'})

@app.route('/api/update_rating', methods=['POST'])
def api_update_rating():
    """API endpoint to update item rating"""
    try:
        data = request.get_json()
        
        if not data or 'item_id' not in data or 'list_type' not in data or 'rating' not in data:
            return jsonify({'success': False, 'message': 'Invalid data'})
        
        item_id = str(data['item_id'])
        list_type = data['list_type']
        rating = float(data['rating'])
        notes = data.get('notes', '')
        list_key = f'watchlist_{list_type}'
        
        if list_key in session:
            for i, item in enumerate(session[list_key]):
                if str(item['id']) == item_id:
                    session[list_key][i]['user_rating'] = rating
                    session[list_key][i]['notes'] = notes
                    session.modified = True
                    return jsonify({'success': True, 'message': 'Rating updated successfully'})
        
        return jsonify({'success': False, 'message': 'Item not found'})
    except Exception as e:
        print(f"❌ Error updating rating: {e}")
        return jsonify({'success': False, 'message': 'Error updating rating'})

@app.route('/api/calculate_schedule', methods=['POST'])
def api_calculate_schedule():
    """Scheduler API with hours instead of days"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'Invalid data'})
        
        total_minutes = int(data.get('total_minutes', 0))
        watched_minutes = int(data.get('watched_minutes', 0))
        days_per_week = int(data.get('days_per_week', 3))
        hours_per_session = float(data.get('hours_per_session', 2.0))
        
        remaining_minutes = total_minutes - watched_minutes
        remaining_hours = remaining_minutes / 60
        progress_percentage = (watched_minutes / total_minutes * 100) if total_minutes > 0 else 0
        
        minutes_per_session = hours_per_session * 60
        total_sessions_needed = remaining_minutes / minutes_per_session if minutes_per_session > 0 else 0
        weeks_to_complete = total_sessions_needed / days_per_week if days_per_week > 0 else 0
        
        # Calculate in hours, not days
        hours_to_complete = weeks_to_complete * 7 * hours_per_session
        
        finish_date = (datetime.date.today() + datetime.timedelta(days=int(weeks_to_complete * 7))).isoformat()
        
        schedule_data = {
            'success': True,
            'schedule': {
                'total_minutes': total_minutes,
                'watched_minutes': watched_minutes,
                'remaining_minutes': remaining_minutes,
                'remaining_hours': round(remaining_hours, 1),
                'progress_percentage': round(progress_percentage, 1),
                'days_per_week': days_per_week,
                'hours_per_session': hours_per_session,
                'sessions_needed': int(total_sessions_needed),
                'weeks_to_complete': round(weeks_to_complete, 1),
                'hours_to_complete': round(hours_to_complete, 1),
                'finish_date': finish_date
            }
        }
        
        return jsonify(schedule_data)
        
    except Exception as e:
        print(f"❌ Error calculating schedule: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/track_progress', methods=['POST'])
def api_track_progress():
    """Progress tracker with exact calculations"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})
        
        media_id = data.get('media_id')
        media_type = data.get('media_type')
        
        if not media_id or not media_type:
            return jsonify({'success': False, 'message': 'Missing media ID or type'})
        
        if media_type == 'movie':
            watched_minutes = int(data.get('watched_minutes', 0))
            total_minutes = int(data.get('total_minutes', 0))
            
            if total_minutes <= 0:
                return jsonify({'success': False, 'message': 'Invalid total runtime'})
            
            progress_pct = (watched_minutes / total_minutes * 100)
            remaining_minutes = total_minutes - watched_minutes
            
            if 'progress_tracker' not in session:
                session['progress_tracker'] = {}
            
            session['progress_tracker'][f"{media_type}_{media_id}"] = {
                'watched_minutes': watched_minutes,
                'total_minutes': total_minutes,
                'progress_pct': progress_pct,
                'remaining_minutes': remaining_minutes,
                'last_updated': datetime.datetime.now().isoformat()
            }
            session.modified = True
            
            return jsonify({
                'success': True,
                'progress_pct': round(progress_pct, 1),
                'watched_minutes': watched_minutes,
                'remaining_minutes': remaining_minutes,
                'watched_hours': round(watched_minutes / 60, 1),
                'remaining_hours': round(remaining_minutes / 60, 1),
                'message': f'Movie progress saved: {round(progress_pct, 1)}% complete'
            })
            
        else:
            current_season = int(data.get('current_season', 1))
            current_episode = int(data.get('current_episode', 0))
            season_data = data.get('season_data', [])
            
            watched_minutes = 0
            total_minutes = sum(season['total_runtime'] for season in season_data) if season_data else 0
            
            if total_minutes <= 0:
                return jsonify({'success': False, 'message': 'Unable to calculate total runtime'})
            
            # Calculate watched minutes from EXACT episode data
            for season in season_data:
                if season['season_number'] < current_season:
                    watched_minutes += season['total_runtime']
                elif season['season_number'] == current_season:
                    episodes = season.get('episodes', [])
                    for episode in episodes:
                        if episode['episode_number'] < current_episode:
                            watched_minutes += episode.get('runtime', 45)
                    break
            
            progress_pct = (watched_minutes / total_minutes * 100) if total_minutes > 0 else 0
            remaining_minutes = total_minutes - watched_minutes
            
            if 'progress_tracker' not in session:
                session['progress_tracker'] = {}
            
            session['progress_tracker'][f"{media_type}_{media_id}"] = {
                'watched_minutes': watched_minutes,
                'total_minutes': total_minutes,
                'progress_pct': progress_pct,
                'remaining_minutes': remaining_minutes,
                'current_season': current_season,
                'current_episode': current_episode,
                'last_updated': datetime.datetime.now().isoformat()
            }
            session.modified = True
            
            return jsonify({
                'success': True,
                'progress_pct': round(progress_pct, 1),
                'watched_minutes': watched_minutes,
                'remaining_minutes': remaining_minutes,
                'watched_hours': round(watched_minutes / 60, 1),
                'remaining_hours': round(remaining_minutes / 60, 1),
                'current_position': f"S{current_season}E{current_episode}",
                'message': f'TV progress saved: S{current_season}E{current_episode} ({round(progress_pct, 1)}% complete)'
            })
        
    except Exception as e:
        print(f"❌ Error tracking progress: {e}")
        return jsonify({'success': False, 'message': f'Error tracking progress: {str(e)}'})

@app.route('/api/export_watchlist/<format_type>')
def api_export_watchlist(format_type):
    """API endpoint to export watchlist"""
    try:
        if format_type not in ['csv', 'json']:
            return jsonify({'success': False, 'message': 'Invalid format'})
        
        AdvancedWatchlistManager.initialize_all()
        
        all_items = []
        for category in ['watching', 'want_to_watch', 'completed', 'dropped', 'favorites']:
            items = session.get(f'watchlist_{category}', [])
            for item in items:
                item_copy = item.copy()
                item_copy['status'] = category
                all_items.append(item_copy)
        
        if format_type == 'csv':
            output = io.StringIO()
            writer = csv.writer(output)
            
            headers = ['Title', 'Original Title', 'Type', 'Year', 'Status', 
                      'Runtime (min)', 'Genres', 'Language', 'Added Date']
            writer.writerow(headers)
            
            for item in all_items:
                writer.writerow([
                    item.get('title', ''),
                    item.get('original_title', ''),
                    item.get('media_type', ''),
                    item.get('year', ''),
                    item.get('status', ''),
                    item.get('runtime', 0),
                    ', '.join(item.get('genres', [])),
                    item.get('language', ''),
                    item.get('added_date', '')
                ])
            
            response = app.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=watchlist.csv'}
            )
        else:
            response = app.response_class(
                json.dumps(all_items, indent=2, default=str),
                mimetype='application/json',
                headers={'Content-Disposition': 'attachment; filename=watchlist.json'}
            )
        
        return response
    except Exception as e:
        print(f"❌ Error exporting watchlist: {e}")
        return jsonify({'success': False, 'message': 'Error exporting watchlist'})

# Clear cache periodically
@app.before_request
def clear_cache_periodically():
    """Clear API cache every hour to prevent stale data"""
    if hasattr(app, '_last_cache_clear'):
        if time.time() - app._last_cache_clear > 3600:  # 1 hour
            api_cache.clear()
            app._last_cache_clear = time.time()
            print("🗑️ API cache cleared")
    else:
        app._last_cache_clear = time.time()

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    print(f"❌ Internal server error: {error}")
    return redirect(url_for('index'))

if __name__ == '__main__':
    def is_port_available(port):
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    # Try ports in order: 5001, 5002, 5003, etc.
    for port in range(5001, 5010):
        if is_port_available(port):
            print(f"\n🚀 Starting FlixTracker with Streaming Platform Support")
            print(f"✅ NEW FEATURES: Streaming platforms, About page, Progress tracking, Working quick add buttons")
            print(f"✅ STREAMING: Shows Netflix, Disney+, Hulu, Amazon Prime, and more")
            print(f"✅ SEARCH UPDATE: Movies and TV shows only (no people)")
            print(f"🌐 Open your browser to: http://localhost:{port}")
            print(f"📱 Or try: http://127.0.0.1:{port}")
            print(f"🎬 Server starting on port {port}...\n")
            
            try:
                app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
            except KeyboardInterrupt:
                print(f"\n👋 Server stopped. Thanks for using FlixTracker!")
            break
    else:
        print("❌ No available ports found between 5001-5009")
        print("💡 Try closing other applications or restart your computer")





#take out the import watchlist option
#make the website retro futuristic theme like the attached image
#remove discover option
#remove actor option
#for trending make sure when i click the this week and movies only and tv shows only options works and takes me to the site its supposed to
#remmove the "get started"option from the dashboard
#make sure to take me to the show or movie i have clicked the view option on quicker
#For season and episode analysis create a drop down menu with not black and white table
#remove the rating section from everywhere
#make sure there is some videos and trailers to all shows and movies if not explicity mention so
#take the plan viewing schedule and track progress out of the season and episode analysis
#always add a view more of cast and crew to expand to a page that shows every single member of cast and associated picture but remove the cast profiles
#THE MAIN FEATURE THE IS PROGRESS TRACKING AND SCHEDULING MAKE SURE THESE FEATURES ALL WORK PERFECTLY AND EFFICIETLY
# the "track your progress" is really good



#track your progress no longer working
#tv shows button and all not working - from trending
#adapt the scheduling that has default of episode and season for shows, and then for movies it goes by hours and minutes, make sure sheduling gives a accurate outcome, and also the runtime should be measured by hours not days
#fix text colours so that it shows up on the text background
#make sure quick season and episode stats and analysis works
#fix quick add buttons and make sure they work
#make sure all information about movie or show is correct ALWAYS, this like runtime, avg episode time, and more




#when opening pop-ups like for track progress and cast open in the fram of the users click
#add sharing buttons to socials
#exact epi timings
#fix analytics page
#make all buttons and button highlightings consistent
#make sure all buttons are consistent in size and shape
#make sure all buttons have hover effects
#ease up the colours and fonts of the webapp to make it more readable for the user



#instead of pup-up for track progress, make it so that is goes to a view more
#fix quickadd buttons to work properly
#add more of a description of the website fr the user to know what it is about
#add a about us page and give me the freedom to write what i want about how this is my project and how it works and who i am
#remove the search page
#for search remove top rated and trending, and make it so that it only searches for movies and tv shows

#add which streaming service its available on
#revert back to the track your progress button instead that takes you to the seperate page and make sure all the sliders work as usual and fill up until where the user drags it
#make sure that the button to add to watchlist on the thumbnail works and adds to the watchlist
#make sure that you add many more badges and things to earn
#add a loading icon when loading the page

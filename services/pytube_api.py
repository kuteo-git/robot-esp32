import os
import re
import json
import glob
from pathlib import Path
from flask import Flask, jsonify, send_file, request
from pytubefix import YouTube, Playlist
from pytubefix.cli import on_progress
import yt_dlp
import logging
import time
import random
import yaml
from urllib.parse import parse_qs, urlparse

# Flask app setup
app = Flask(__name__)

# Configuration
# cache_dir is read from data/.config.yaml (plugins.pytube.cache_dir) -> a single place to
# configure for the whole project (instead of scattering env vars across scripts).
# PYTUBE_CACHE_DIR / the local default are now just a fallback for when config.yaml doesn't
# have this key yet or couldn't be read.
_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "xiaozhi-esp32-server/main/xiaozhi-server/data/.config.yaml"
)


def _load_cache_dir():
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cache_dir = (cfg.get("plugins") or {}).get("pytube", {}).get("cache_dir")
        if cache_dir:
            return cache_dir
    except Exception as e:
        logging.getLogger(__name__).warning(f"Không đọc được {_CONFIG_PATH}: {e}")
    return os.environ.get(
        "PYTUBE_CACHE_DIR", str(Path(__file__).resolve().parent / "pytube_cache")
    )


folder_path = _load_cache_dir()
HOST = '0.0.0.0'  # Allow external access
PORT = 114
MAX_RETRIES = 1

# Set up logging — xiaozhi-style format: '2026-06-19 19:34:07 - pytube - INFO - <msg>'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - pytube - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Lightweight cache (TTL) for search/related/playlist_url
_API_CACHE = {}
_API_CACHE_TTL = 600  # seconds


def _cache_get(key):
    item = _API_CACHE.get(key)
    if item and (time.time() - item[0]) < _API_CACHE_TTL:
        return item[1]
    return None


def _cache_set(key, value):
    _API_CACHE[key] = (time.time(), value)


def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            # logger.info(f"Created directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {str(e)}")
        return False


def sanitize_filename(filename):
    """Remove invalid characters from filename"""
    # Remove or replace invalid characters for filenames
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace multiple spaces with single space and strip
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized


def get_device_token_path(device):
    """Get the token file path for a specific device"""
    tokens_dir = "./tokens"
    ensure_directory_exists(tokens_dir)
    return os.path.join(tokens_dir, f"{device}.json")


def extract_playlist_id(url):
    """Extract playlist ID from YouTube playlist URL"""
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if 'list' in query_params:
            return query_params['list'][0]
        return None
    except Exception:
        return None


def get_playlist_cache_path(playlist_id):
    """Get cache file path for playlist"""
    return os.path.join(folder_path, f"playlist_{playlist_id}.json")


def save_playlist_cache(playlist_id, data):
    """Save playlist data to cache"""
    try:
        cache_path = get_playlist_cache_path(playlist_id)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved playlist cache: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save playlist cache: {str(e)}")
        return False


def load_playlist_cache(playlist_id):
    """Load playlist data from cache"""
    try:
        cache_path = get_playlist_cache_path(playlist_id)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded playlist from cache: {cache_path}")
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to load playlist cache: {str(e)}")
        return None


def get_video_metadata_cache_path(video_id, title):
    """Get cache file path for video metadata"""
    sanitized_title = sanitize_filename(title)
    return os.path.join(folder_path, f"{sanitized_title}_{video_id}.json")


def save_video_metadata_cache(video_id, title, metadata):
    """Save video metadata to cache"""
    try:
        cache_path = get_video_metadata_cache_path(video_id, title)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved video metadata cache: {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save video metadata cache: {str(e)}")
        return False


def load_video_metadata_cache(cache_path):
    """Load video metadata from cache"""
    if cache_path is None:
        return None

    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded video metadata from cache: {cache_path}")
            return data
        return None
    except Exception as e:
        logger.error(f"Failed to load video metadata cache: {str(e)}")
        return None

def find_cached_metadata_file(video_id):
    """Find cached MP3 file by video_id using glob pattern"""
    try:
        pattern = os.path.join(folder_path, f"*_{video_id}.json")
        matches = glob.glob(pattern)
        if matches:
            # Return the first match (should only be one)
            return matches[0]
        return None
    except Exception as e:
        logger.error(f"Failed to find cached MP3 file: {str(e)}")
        return None

def find_cached_mp3_file(video_id):
    """Find cached MP3 file by video_id using glob pattern"""
    try:
        pattern = os.path.join(folder_path, f"*_{video_id}.mp3")
        matches = glob.glob(pattern)
        if matches:
            # Return the first match (should only be one)
            return matches[0]
        return None
    except Exception as e:
        logger.error(f"Failed to find cached MP3 file: {str(e)}")
        return None


def create_youtube_object_with_retry(video_url, max_retries=MAX_RETRIES, device=None):
    """
    Create YouTube object with retry logic and exponential backoff
    Returns YouTube object or None if all attempts fail
    """
    for attempt in range(max_retries):
        try:
            # Add some randomization to avoid thundering herd
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 1)
                # logger.info(f"Retrying YouTube object creation after "
                #             f"{delay:.2f}s (attempt {attempt + 1}/"
                #             f"{max_retries})")
                time.sleep(delay)
            
            if device:
                # V2 API with device-specific token
                # token_file = get_device_token_path(device)  # Currently unused
                yt = YouTube(
                    video_url,
                    # use_oauth=False,
                    # allow_oauth_cache=True,
                    # token_file=token_file,
                    on_progress_callback=on_progress
                )
            else:
                # V1 API (original)
                yt = YouTube(
                    video_url,
                    use_oauth=True,
                    allow_oauth_cache=True,
                    on_progress_callback=on_progress
                )
            
            # Test that the object is actually accessible
            # _ = yt.title  # This will trigger the API call
            return yt
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for "
                           f"{video_url}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} attempts failed for "
                             f"{video_url}")
                return None
    return YouTube(video_url, on_progress_callback=on_progress)

class VideoUnavailableError(Exception):
    """Video private/deleted/restricted - can't be downloaded (as opposed to a system error)."""
    pass


def download_audio_with_ytdlp(video_id):
    """Download audio using yt-dlp and get info in single call"""
    youtube_url = f"https://youtube.com/watch?v={video_id}"

    # Prefer AUDIO. FFmpegExtractAudio will extract the audio -> a real mp3 (machine has ffmpeg).
    format_strategies = [
        ('bestaudio/best', 'Best audio'),
        ('bestaudio*', 'Any audio stream'),
        (None, 'Default format selection'),
    ]
    
    # Check if cookies file exists, if not, don't use it
    cookie_file = 'cookies.txt'
    use_cookies = os.path.exists(cookie_file)

    last_error = None  # keep the last error to determine the actual reason (private/deleted...)

    for strategy, description in format_strategies:
        ydl_opts = {
            'outtmpl': os.path.join(folder_path, '%(title)s_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
            # tv is now DRM-gated, ios needs a PO token (2026-07) -> went back to android/web
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],
                }
            },
            # Extract audio -> a real mp3 via ffmpeg (machine has ffmpeg in PATH)
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            }
        }
        
        # Only add format if specified
        if strategy:
            ydl_opts['format'] = strategy
            
        # Only add cookies if file exists
        if use_cookies:
            ydl_opts['cookiefile'] = cookie_file
        
        try:
            logger.info(f"Trying strategy: {description} (format: {strategy})")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download directly (download=True already returns the info) — skip the redundant extra extract_info call
                info = ydl.extract_info(youtube_url, download=True)
                
                # Find the downloaded file (might not be mp3)
                pattern = os.path.join(folder_path, f"*_{video_id}.*")
                matches = glob.glob(pattern)
                if matches:
                    downloaded_file = matches[0]
                    logger.info(f"Downloaded file: {downloaded_file}")
                    
                    # Rename to .mp3 for consistency (even if it's not audio)
                    mp3_file = downloaded_file.rsplit('.', 1)[0] + '.mp3'
                    if downloaded_file != mp3_file:
                        os.rename(downloaded_file, mp3_file)
                        logger.info(f"Renamed to: {mp3_file}")
                    
                    return mp3_file, info
                else:
                    logger.warning("No files found matching pattern")
                    
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Strategy '{description}' failed: {last_error}")
            continue

    # If the reason is that the video can't be accessed (private/deleted/removed...) -> report it
    # clearly so the endpoint returns 404 instead of 500, letting HA know to skip this song.
    err_lower = (last_error or '').lower()
    unavailable_markers = [
        'private video', 'video unavailable', 'has been removed',
        'no longer available', 'this video is not available',
        'account associated with this video has been terminated',
        'sign in to confirm your age', 'members-only', 'video has been removed',
    ]
    if any(m in err_lower for m in unavailable_markers):
        raise VideoUnavailableError(
            f"Video {video_id} is unavailable (private/deleted/restricted). "
            f"yt-dlp: {last_error}"
        )

    # If all strategies fail for other reasons, provide detailed error
    error_msg = f"All {len(format_strategies)} download strategies failed for video {video_id}. "
    error_msg += f"Video URL: {youtube_url}. "
    error_msg += f"Cookies available: {use_cookies}. "
    error_msg += f"Last error: {last_error}. "
    error_msg += "This might be due to: 1) Video is private/deleted, 2) Network restrictions in HA, "
    error_msg += "3) yt-dlp version compatibility, 4) Missing dependencies in HA container"

    raise Exception(error_msg)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'YouTube Downloader API',
        'cache_directory': folder_path,
        'directory_exists': os.path.exists(folder_path)
    })

@app.route('/check-ip-status', methods=['GET'])
def check_ip_status():
    """Check if IP is blocked by YouTube"""
    import requests
    
    try:
        # Try a simple request to YouTube
        response = requests.get(
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            if 'Sign in to confirm' in response.text or 'bot' in response.text.lower():
                return jsonify({
                    'status': 'blocked',
                    'message': 'Your IP is likely blocked by YouTube. Wait 24-48 hours or use VPN.',
                    'http_status': response.status_code
                })
            else:
                return jsonify({
                    'status': 'ok',
                    'message': 'Your IP appears to be working normally with YouTube',
                    'http_status': response.status_code
                })
        else:
            return jsonify({
                'status': 'unknown',
                'message': f'Unexpected response from YouTube',
                'http_status': response.status_code
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'service': 'YouTube Downloader API',
        'version': '2.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /v2/playlist?url=<playlist_url>&device=<device_id>': (
                'Get simplified playlist info with smart caching (using pytubefix)'
            ),
            'GET /v2/video/<video_id>?device=<device_id>': (
                'Get video information with mp3_url if cached (using pytubefix)'
            ),
            'GET /v2/mp3/<video_id>?device=<device_id>': (
                'Serve cached MP3 file directly'
            ),
            'GET /v3/playlist?url=<playlist_url>&device=<device_id>': (
                'Get simplified playlist info with smart caching (using yt-dlp)'
            ),
            'GET /v3/video/<video_id>?device=<device_id>': (
                'Get video information with mp3_url if cached (using yt-dlp)'
            ),
            'GET /v3/mp3/<video_id>?device=<device_id>': (
                'Serve cached MP3 file directly'
            )
        },
        'cache_directory': folder_path,
        'tokens_directory': './tokens'
    })


# ================= V2 API ENDPOINTS =================

@app.route('/v2/playlist', methods=['GET'])
def get_playlist_videos_v2():
    """
    V2: Get videos from a YouTube playlist with device-specific tokens
    Expected query parameters: url (YouTube playlist URL), device (device identifier)
    Returns: JSON array with simplified video information and caching
    """
    try:
        playlist_url = request.args.get('url')
        device = request.args.get('device')
        
        if not playlist_url:
            return jsonify({
                'error': 'Missing required parameter: url',
                'message': 'Please provide a YouTube playlist URL'
            }), 400
            
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        # Extract playlist ID for caching
        playlist_id = extract_playlist_id(playlist_url)
        if not playlist_id:
            return jsonify({
                'error': 'Invalid playlist URL',
                'message': 'Could not extract playlist ID from URL'
            }), 400

        # Ensure cache directory exists
        if not ensure_directory_exists(folder_path):
            logger.warning(f"Could not create cache directory: {folder_path}")

        logger.info(
            f"Processing playlist (v2): {playlist_url} for device: {device}"
        )

        try:
            # Always try to get playlist from YouTube API first
            playlist = Playlist(playlist_url)
            video_urls = playlist.video_urls

            if not video_urls:
                # If no videos found, try to return cached data
                cached_data = load_playlist_cache(playlist_id)
                if cached_data:
                    logger.info(
                        f"No videos found in API, returning cached data for {playlist_id}"
                    )
                    return jsonify(cached_data)
                
                return jsonify({
                    'error': 'No videos found',
                    'message': (
                        'The playlist appears to be empty or inaccessible'
                    )
                }), 404

            # Build response data
            videos_info = []
            for video_url in video_urls:
                video_id = video_url.split('watch?v=')[-1].split('&')[0]
                video_info = {
                    "video_url": video_url,
                    "video_id": video_id
                }
                videos_info.append(video_info)

            # Save successful results to cache
            save_playlist_cache(playlist_id, videos_info)
            logger.info(
                f"Successfully processed {len(videos_info)} videos (v2)"
            )

            return jsonify(videos_info)

        except Exception as e:
            logger.error(f"Error processing playlist: {str(e)}")
            # Only if API fails, try to return cached data
            cached_data = load_playlist_cache(playlist_id)
            if cached_data:
                logger.info(
                    f"API failed, returning cached data for {playlist_id}: {str(e)}"
                )
                return jsonify(cached_data)
            
            return jsonify({
                'error': 'Failed to process playlist',
                'message': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error in v2 playlist endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to process playlist',
            'message': str(e)
        }), 500


@app.route('/v3/playlist', methods=['GET'])
def get_playlist_videos_v3():
    """
    V3: Get videos from a YouTube playlist using yt-dlp instead of pytubefix
    Expected query parameters: url (YouTube playlist URL), device (device identifier)
    Returns: JSON array with simplified video information and caching
    """
    try:
        playlist_url = request.args.get('url')
        device = request.args.get('device')
        
        if not playlist_url:
            return jsonify({
                'error': 'Missing required parameter: url',
                'message': 'Please provide a YouTube playlist URL'
            }), 400
            
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        # Extract playlist ID for caching
        playlist_id = extract_playlist_id(playlist_url)
        if not playlist_id:
            return jsonify({
                'error': 'Invalid playlist URL',
                'message': 'Could not extract playlist ID from URL'
            }), 400

        # Ensure cache directory exists
        if not ensure_directory_exists(folder_path):
            logger.warning(f"Could not create cache directory: {folder_path}")

        logger.info(
            f"Processing playlist (v3): {playlist_url} for device: {device}"
        )

        try:
            # Lay danh sach video: uu tien ytmusicapi (lay HET — yt-dlp bi cap 100 bai/
            # playlist), union them yt-dlp de bo sung. Giu thu tu playlist, khu trung.
            ordered_ids = []
            seen = set()

            # 1) ytmusicapi (day du, dung thu tu playlist)
            try:
                ytm = _get_ytmusic()
                pl = ytm.get_playlist(playlist_id, limit=None)
                for t in (pl.get('tracks') or []):
                    vid = t.get('videoId')
                    if vid and vid not in seen:
                        seen.add(vid)
                        ordered_ids.append(vid)
                logger.info(f"ytmusicapi: got {len(ordered_ids)} videos for {playlist_id}")
            except Exception as e:
                logger.warning(f"ytmusicapi failed for {playlist_id}: {e}")

            # 2) yt-dlp bo sung (phong khi ytmusicapi thieu vai bai)
            try:
                ydl_opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': True,
                    'ignoreerrors': True,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['android', 'web'],
                        }
                    },
                    'http_headers': {
                        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    }
                }
                cookie_file = 'cookies.txt'
                if os.path.exists(cookie_file):
                    ydl_opts['cookiefile'] = cookie_file

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    logger.info("Supplementing playlist info with yt-dlp...")
                    info = ydl.extract_info(playlist_url, download=False)
                for entry in ((info.get('entries') or []) if info else []):
                    if entry and entry.get('id'):
                        vid = entry['id']
                        if vid not in seen:
                            seen.add(vid)
                            ordered_ids.append(vid)
            except Exception as e:
                logger.warning(f"yt-dlp failed for {playlist_id}: {e}")

            if not ordered_ids:
                # Khong lay duoc gi -> tra cache cu neu co
                cached_data = load_playlist_cache(playlist_id)
                if cached_data:
                    logger.info(
                        f"No videos from API, returning cached data for {playlist_id}"
                    )
                    return jsonify(cached_data)

                return jsonify({
                    'error': 'No videos found',
                    'message': 'The playlist appears to be empty or inaccessible'
                }), 404

            # Build response data (giu thu tu)
            videos_info = [
                {
                    "video_url": f"https://youtube.com/watch?v={vid}",
                    "video_id": vid
                }
                for vid in ordered_ids
            ]

            # Save successful results to cache
            save_playlist_cache(playlist_id, videos_info)
            logger.info(
                f"Successfully processed {len(videos_info)} videos (v3 ytmusicapi+yt-dlp)"
            )

            return jsonify(videos_info)

        except Exception as e:
            logger.error(f"Error processing playlist: {str(e)}")
            # Only if API fails, try to return cached data
            cached_data = load_playlist_cache(playlist_id)
            if cached_data:
                logger.info(
                    f"API failed, returning cached data for {playlist_id}: {str(e)}"
                )
                return jsonify(cached_data)

            return jsonify({
                'error': 'Failed to process playlist',
                'message': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Error in v3 playlist endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to process playlist',
            'message': str(e)
        }), 500


@app.route('/v2/video/<video_id>', methods=['GET'])
def get_video_info_v2(video_id):
    """
    V2: Get video information by video ID with device-specific tokens
    Expected query parameter: device (device identifier)
    Returns: JSON with video information
    """
    try:
        device = request.args.get('device')
        
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        logger.info(f"Processing video info (v2): {video_id} for device: {device}")

        # Create YouTube object with device-specific token
        youtube_url = f"https://youtube.com/watch?v={video_id}"

        # Check if MP3 file already exists in cache
        cached_mp3_file = find_cached_mp3_file(video_id)
        
        if cached_mp3_file and os.path.exists(cached_mp3_file):
            mp3_url = f"/v2/mp3/{video_id}?device={device}"
            # Load metadata
            cached_meta_data_path = find_cached_metadata_file(video_id)
            cached_meta_data = load_video_metadata_cache(cached_meta_data_path)
            if not cached_meta_data:
                cached_meta_data = {
                        "video_title": "",
                        "video_thumbnail_url": "",
                        "video_id": video_id,
                        "video_url": youtube_url,
                        "video_duration": "0",
                        "mp3_url": mp3_url
                    }
                
            # logger.info(f"Returning cached MP3 info (v2): {cached_meta_data.get("video_title", "")}. Metadata: {cached_meta_data}")
                        
            # File exists, return info with mp3_url
            video_info = {
                "video_title": cached_meta_data.get("video_title", ""),
                "video_thumbnail_url": cached_meta_data["video_thumbnail_url"],
                "video_id": video_id,
                "video_url": youtube_url,
                "video_duration": cached_meta_data["video_duration"],
                "mp3_url": mp3_url,
                "is_loaded_from_cache": True
            }
            return jsonify(video_info)
        
        # File doesn't exist, download it
        logger.info(f"MP3 not cached, downloading (v2): {video_id}")
        
        # Create YouTube object with device-specific token
        yt = create_youtube_object_with_retry(youtube_url, max_retries=MAX_RETRIES, device=device)
        if not yt:
            return jsonify({
                'error': 'Video unavailable',
                'message': f'Could not access video {video_id}. The video may be private, deleted, or temporarily unavailable.',
                'video_id': video_id
            }), 404
        
        # Ensure download directory exists
        if not ensure_directory_exists(folder_path):
            return jsonify({
                'error': 'Directory creation failed',
                'message': f'Could not create or access directory: {folder_path}'
            }), 500

        # Download audio stream with retry logic
        audio_stream = None
        max_retries = MAX_RETRIES
        try:
            audio_stream = yt.streams.get_audio_only()
        except Exception as e:
            logger.warning(
                f"failed: {str(e)}"
            )
            return jsonify({
                'error': 'Download failed',
                'message': (
                    f'Failed to get audio stream for video {video_id} '
                    f'after {max_retries} attempts: {str(e)}'
                ),
                'video_id': video_id
            }), 500

        # Download the file with retry logic
        sanitized_title = sanitize_filename(yt.title)
        temp_filename = f"{sanitized_title}_{video_id}.mp4"
        
        downloaded_file = None
        try:
            downloaded_file = audio_stream.download(
                output_path=folder_path,
                filename=temp_filename
            )
        except Exception as e:
            logger.warning(
                f"failed: {str(e)}"
            )
            return jsonify({
                'error': 'Download failed',
                'message': (
                    f'Failed to download video {video_id} '
                    f'after {max_retries} attempts: {str(e)}'
                ),
                'video_id': video_id
            }), 500

        # Rename to .mp3 extension
        mp3_filepath = downloaded_file.replace('.mp4', '.mp3')
        if downloaded_file != mp3_filepath:
            os.rename(downloaded_file, mp3_filepath)

        # Prepare metadata for caching
        metadata = {
            "video_title": yt.title,
            "video_thumbnail_url": yt.thumbnail_url,
            "video_id": video_id,
            "video_url": youtube_url,
            "video_duration": yt.length,
            "mp3_url": mp3_filepath
        }

        # Save metadata to cache
        save_video_metadata_cache(video_id, yt.title, metadata)

        # Return video info with mp3_url
        mp3_url = f"/v2/mp3/{video_id}?device={device}"
        video_info = {
            "video_title": yt.title,
            "video_thumbnail_url": yt.thumbnail_url,
            "video_id": video_id,
            "video_url": youtube_url,
            "video_duration": yt.length,
            "mp3_url": mp3_url,
            "is_loaded_from_cache": False
        }

        logger.info(f"Successfully downloaded and cached (v2): {yt.title}")
        return jsonify(video_info)

    except Exception as e:
        logger.error(f"Error getting video info (v2) for {video_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to get video information',
            'message': str(e),
            'video_id': video_id
        }), 500

  
@app.route('/v3/video/<video_id>', methods=['GET'])
def get_video_info_v3(video_id):
    """
    V3: Get video information by video ID with device-specific tokens using yt-dlp
    Expected query parameter: device (device identifier)
    Returns: JSON with video information
    """
    try:
        device = request.args.get('device')
        
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        logger.info(f"Processing video info (v3): {video_id} for device: {device}")

        # Create YouTube URL
        youtube_url = f"https://youtube.com/watch?v={video_id}"

        # Ensure download directory exists
        if not ensure_directory_exists(folder_path):
            return jsonify({
                'error': 'Directory creation failed',
                'message': f'Could not create or access directory: {folder_path}'
            }), 500

        # Check if MP3 file already exists in cache
        cached_mp3_file = find_cached_mp3_file(video_id)
        
        if cached_mp3_file and os.path.exists(cached_mp3_file):
            mp3_url = f"/v3/mp3/{video_id}?device={device}"
            
            # Load metadata
            cached_meta_data_path = find_cached_metadata_file(video_id)
            cached_meta_data = load_video_metadata_cache(cached_meta_data_path)
            
            if not cached_meta_data:
                # If metadata doesn't exist, create minimal metadata
                cached_meta_data = {
                    "video_title": "",
                    "video_thumbnail_url": "",
                    "video_id": video_id,
                    "video_url": youtube_url,
                    "video_duration": "0",
                    "mp3_url": mp3_url
                }
                
            logger.info(f"Returning cached MP3 info (v3): {cached_meta_data.get('video_title', 'Unknown')}. Metadata: {cached_meta_data}")
                        
            # File exists, return info with mp3_url
            video_info = {
                "video_title": cached_meta_data.get("video_title", ""),
                "video_thumbnail_url": cached_meta_data.get("video_thumbnail_url", ""),
                "video_id": video_id,
                "video_url": youtube_url,
                "video_duration": str(cached_meta_data.get("video_duration", "0")),
                "mp3_url": mp3_url,
                "is_loaded_from_cache": True
            }
            return jsonify(video_info)
        
        # File doesn't exist, download it
        logger.info(f"MP3 not cached, downloading (v3): {video_id}")
        
        # Download the audio file
        try:
            downloaded_file, video_info_data = download_audio_with_ytdlp(video_id)

        except VideoUnavailableError as e:
            # Video private/deleted/restricted -> 404 de HA biet bo qua bai nay
            logger.warning(f"Video unavailable {video_id}: {str(e)}")
            return jsonify({
                'error': 'Video unavailable',
                'message': str(e),
                'video_id': video_id,
                'unavailable': True
            }), 404
        except Exception as e:
            logger.error(f"Download failed for {video_id}: {str(e)}")
            return jsonify({
                'error': 'Download failed',
                'message': f'Failed to download video {video_id}: {str(e)}',
                'video_id': video_id
            }), 500

        # Extract video information
        video_title = video_info_data.get('title', 'Unknown')
        video_duration = video_info_data.get('duration', 0)
        video_thumbnail_url = video_info_data.get('thumbnail', '')

        # Prepare metadata for caching
        metadata = {
            "video_title": video_title,
            "video_thumbnail_url": video_thumbnail_url,
            "video_id": video_id,
            "video_url": youtube_url,
            "video_duration": video_duration,
            "mp3_url": downloaded_file
        }

        # Save metadata to cache
        save_video_metadata_cache(video_id, video_title, metadata)

        # Return video info with mp3_url
        mp3_url = f"/v3/mp3/{video_id}?device={device}"
        video_info = {
            "video_title": video_title,
            "video_thumbnail_url": video_thumbnail_url,
            "video_id": video_id,
            "video_url": youtube_url,
            "video_duration": str(video_duration),
            "mp3_url": mp3_url,
            "is_loaded_from_cache": False
        }

        logger.info(f"Successfully downloaded and cached (v3): {video_title}")
        return jsonify(video_info)

    except Exception as e:
        logger.error(f"Error getting video info (v3) for {video_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to get video information',
            'message': str(e),
            'video_id': video_id
        }), 500


@app.route('/v2/mp3/<video_id>', methods=['GET'])
def serve_mp3_v2(video_id):
    """
    V2: Serve cached MP3 file by video_id
    Expected query parameter: device (device identifier)
    Returns: MP3 file or 404 if not cached
    """
    try:
        device = request.args.get('device')
        
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        # Find cached MP3 file
        cached_mp3_file = find_cached_mp3_file(video_id)
        
        if not cached_mp3_file or not os.path.exists(cached_mp3_file):
            return jsonify({
                'error': 'MP3 not found',
                'message': f'No cached MP3 file found for video {video_id}',
                'video_id': video_id
            }), 404

        # Extract filename for download
        filename = os.path.basename(cached_mp3_file)
        
        logger.info(f"Serving cached MP3 (v2): {filename}")
        
        return send_file(
            cached_mp3_file,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/mpeg'
        )

    except Exception as e:
        logger.error(f"Error serving MP3 (v2) for {video_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to serve MP3',
            'message': str(e),
            'video_id': video_id
        }), 500
        
@app.route('/v3/mp3/<video_id>', methods=['GET'])
def serve_mp3_v3(video_id):
    """
    V3: Serve cached MP3 file by video_id
    Expected query parameter: device (device identifier)
    Returns: MP3 file or 404 if not cached
    """
    try:
        device = request.args.get('device')
        
        if not device:
            return jsonify({
                'error': 'Missing required parameter: device',
                'message': 'Please provide a device identifier'
            }), 400

        # Find cached MP3 file
        cached_mp3_file = find_cached_mp3_file(video_id)
        
        if not cached_mp3_file or not os.path.exists(cached_mp3_file):
            return jsonify({
                'error': 'MP3 not found',
                'message': f'No cached MP3 file found for video {video_id}',
                'video_id': video_id
            }), 404

        # Extract filename for download
        filename = os.path.basename(cached_mp3_file)
        
        logger.info(f"Serving cached MP3 (v2): {filename}")
        
        return send_file(
            cached_mp3_file,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/mpeg'
        )

    except Exception as e:
        logger.error(f"Error serving MP3 (v3) for {video_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to serve MP3',
            'message': str(e),
            'video_id': video_id
        }), 500


# ===== Search + Related (ytmusicapi) - added for the voice assistant =====
_YTMUSIC = None


def _get_ytmusic():
    global _YTMUSIC
    if _YTMUSIC is None:
        from ytmusicapi import YTMusic
        _YTMUSIC = YTMusic()
    return _YTMUSIC


@app.route('/v3/search', methods=['GET'])
def search_v3():
    """Tim bai hat tren YouTube Music theo ten. ?q=<query>&limit=N"""
    query = request.args.get('q', '').strip()
    try:
        limit = int(request.args.get('limit', 5))
    except Exception:
        limit = 5
    if not query:
        return jsonify({'error': 'missing q'}), 400
    ckey = f"search:{query}:{limit}"
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info(f"[search] q='{query}' (cache) -> {cached.get('count', 0)} bai")
        for i, it in enumerate(cached.get('results', [])):
            logger.info(f"    [{i}] {it.get('title','')} - {it.get('artist','')} ({it.get('duration','')})")
        return jsonify(cached)
    try:
        yt = _get_ytmusic()
        results = yt.search(query, filter='songs', limit=limit) or []
        items = []
        for r in results:
            vid = r.get('videoId')
            if not vid:
                continue
            items.append({
                'video_id': vid,
                'title': r.get('title', ''),
                'artist': ', '.join(a['name'] for a in (r.get('artists') or []) if a.get('name')),
                'duration': r.get('duration', ''),
                'thumbnail': (r.get('thumbnails') or [{}])[-1].get('url', ''),
            })
            if len(items) >= limit:
                break
        out = {'query': query, 'count': len(items), 'results': items}
        logger.info(f"[search] q='{query}' -> {len(items)} bai:")
        for i, it in enumerate(items):
            logger.info(f"    [{i}] {it['title']} - {it['artist']} ({it['duration']})")
        _cache_set(ckey, out)
        return jsonify(out)
    except Exception as e:
        logger.error(f"[search_v3] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/v3/related/<video_id>', methods=['GET'])
def related_v3(video_id):
    """Bai hat lien quan / autoplay (nhu YouTube up-next) cho 1 video_id. ?limit=N"""
    try:
        limit = int(request.args.get('limit', 10))
    except Exception:
        limit = 10
    ckey = f"related:{video_id}:{limit}"
    cached = _cache_get(ckey)
    if cached is not None:
        return jsonify(cached)
    try:
        yt = _get_ytmusic()
        wp = yt.get_watch_playlist(videoId=video_id, limit=limit + 5)
        items = []
        for t in (wp.get('tracks') or []):
            vid = t.get('videoId')
            if not vid or vid == video_id:
                continue
            items.append({
                'video_id': vid,
                'title': t.get('title', ''),
                'artist': ', '.join(a['name'] for a in (t.get('artists') or []) if a.get('name')),
            })
            if len(items) >= limit:
                break
        out = {'video_id': video_id, 'count': len(items), 'results': items}
        _cache_set(ckey, out)
        return jsonify(out)
    except Exception as e:
        logger.error(f"[related_v3] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/v3/playlist_url', methods=['GET'])
def playlist_url_v3():
    """Tu mot tu khoa (ten ca si/the loai/bai hat) -> tra ve URL playlist YouTube de phat."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'missing q'}), 400
    ckey = f"plurl:{query}"
    cached = _cache_get(ckey)
    if cached is not None:
        return jsonify(cached)
    try:
        yt = _get_ytmusic()
        # 1) Uu tien playlist that su khop tu khoa
        for r in (yt.search(query, filter='playlists', limit=5) or []):
            pid = r.get('playlistId') or (r.get('browseId') or '')
            if pid.startswith('VL'):
                pid = pid[2:]
            if pid:
                out = {
                    'query': query, 'playlist_id': pid,
                    'url': f'https://music.youtube.com/playlist?list={pid}',
                    'title': r.get('title', ''), 'type': 'playlist',
                }
                _cache_set(ckey, out)
                return jsonify(out)
        # 2) Khong co playlist -> dung radio mix tu bai hat dau (auto-play lien quan)
        songs = yt.search(query, filter='songs', limit=1) or []
        if songs and songs[0].get('videoId'):
            vid = songs[0]['videoId']
            out = {
                'query': query, 'video_id': vid,
                'url': f'https://www.youtube.com/watch?v={vid}&list=RD{vid}',
                'title': songs[0].get('title', ''), 'type': 'radio',
            }
            _cache_set(ckey, out)
            return jsonify(out)
        return jsonify({'error': 'not found'}), 404
    except Exception as e:
        logger.error(f"[playlist_url_v3] {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    # Ensure the cache directory exists on startup
    if ensure_directory_exists(folder_path):
        logger.info(f"Cache directory ready: {folder_path}")
    else:
        logger.warning(f"Could not create cache directory: {folder_path}")

    logger.info(f"Starting YouTube Downloader API on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
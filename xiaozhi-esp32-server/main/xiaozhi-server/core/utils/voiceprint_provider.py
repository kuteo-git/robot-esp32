import asyncio
import time
import aiohttp
import requests
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict
from config.logger import setup_logging
from core.utils.cache.manager import cache_manager
from core.utils.cache.config import CacheType

TAG = __name__
logger = setup_logging()


class VoiceprintProvider:
    """声纹识别服务提供者"""
    
    def __init__(self, config: dict):
        self.original_url = config.get("url", "")
        self.speakers = config.get("speakers", [])
        self.speaker_map = self._parse_speakers()
        # 声纹识别相似度阈值，默认0.4
        self.similarity_threshold = float(config.get("similarity_threshold", 0.4))
        
        # 解析API地址和密钥
        self.api_url = None
        self.api_key = None
        self.speaker_ids = []
        
        if not self.original_url:
            logger.bind(tag=TAG).warning("Voiceprint recognition URL Not configured, voiceprint recognition will be disabled")
            self.enabled = False
        else:
            # 解析URL和key
            parsed_url = urlparse(self.original_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # 从查询参数中提取key
            query_params = parse_qs(parsed_url.query)
            self.api_key = query_params.get('key', [''])[0]
            
            if not self.api_key:
                logger.bind(tag=TAG).error("URL not found in key params, voiceprint recognition will be disabled")
                self.enabled = False
            else:
                # 构造identify接口地址
                self.api_url = f"{base_url}/voiceprint/identify"
                
                # 提取speaker_ids
                for speaker_str in self.speakers:
                    try:
                        parts = speaker_str.split(",", 2)
                        if len(parts) >= 1:
                            speaker_id = parts[0].strip()
                            self.speaker_ids.append(speaker_id)
                    except Exception:
                        continue
                
                # 检查是否有有效的说话人配置
                if not self.speaker_ids:
                    logger.bind(tag=TAG).warning("No valid speaker configured, voiceprint recognition will be disabled")
                    self.enabled = False
                else:
                    # 进行健康检查，验证服务器是否可用
                    if self._check_server_health():
                        self.enabled = True
                        logger.bind(tag=TAG).info(f"Voiceprint recognition enabled: API={self.api_url}, speaker={len(self.speaker_ids)} , similarity threshold={self.similarity_threshold}")
                    else:
                        self.enabled = False
                        logger.bind(tag=TAG).warning(f"Voiceprint server unavailable, voiceprint recognition disabled: {self.api_url}")
    
    def _parse_speakers(self) -> Dict[str, Dict[str, str]]:
        """解析说话人配置"""
        speaker_map = {}
        for speaker_str in self.speakers:
            try:
                parts = speaker_str.split(",", 2)
                if len(parts) >= 3:
                    speaker_id, name, description = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    speaker_map[speaker_id] = {
                        "name": name,
                        "description": description
                    }
            except Exception as e:
                logger.bind(tag=TAG).warning(f"Failed to parse speaker config: {speaker_str}, error: {e}")
        return speaker_map
    
    def _check_server_health(self) -> bool:
        """检查声纹识别服务器健康状态"""
        if not self.api_url or not self.api_key:
            return False
    
        cache_key = f"{self.api_url}:{self.api_key}"
        
        # 检查缓存
        cached_result = cache_manager.get(CacheType.VOICEPRINT_HEALTH, cache_key)
        if cached_result is not None:
            logger.bind(tag=TAG).debug(f"Using cached health status: {cached_result}")
            return cached_result
        
        # 缓存过期或不存在
        logger.bind(tag=TAG).info("Run voiceprint server health check")
        
        try:
            # 健康检查URL
            parsed_url = urlparse(self.api_url)
            health_url = f"{parsed_url.scheme}://{parsed_url.netloc}/voiceprint/health?key={self.api_key}"
            
            # 发送健康检查请求
            response = requests.get(health_url, timeout=3)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "healthy":
                    logger.bind(tag=TAG).info("Voiceprint server health check passed")
                    is_healthy = True
                else:
                    logger.bind(tag=TAG).warning(f"Voiceprint server status abnormal: {result}")
                    is_healthy = False
            else:
                logger.bind(tag=TAG).warning(f"Voiceprint server health check failed: HTTP {response.status_code}")
                is_healthy = False
                
        except requests.exceptions.ConnectTimeout:
            logger.bind(tag=TAG).warning("Voiceprint server connection timeout")
            is_healthy = False
        except requests.exceptions.ConnectionError:
            logger.bind(tag=TAG).warning("Voiceprint server connection refused")
            is_healthy = False
        except Exception as e:
            logger.bind(tag=TAG).warning(f"Voiceprint server health check exception: {e}")
            is_healthy = False
        
        # 使用全局缓存管理器缓存结果
        cache_manager.set(CacheType.VOICEPRINT_HEALTH, cache_key, is_healthy)
        logger.bind(tag=TAG).info(f"Health check result cached: {is_healthy}")
        
        return is_healthy
    
    async def identify_speaker(self, audio_data: bytes, session_id: str) -> Optional[str]:
        """识别说话人"""
        if not self.enabled or not self.api_url or not self.api_key:
            logger.bind(tag=TAG).debug("Voiceprint recognition disabled or not configured, skipping")
            return None
            
        try:
            api_start_time = time.monotonic()
            
            # 准备请求头
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Accept': 'application/json'
            }
            
            # 准备multipart/form-data数据
            data = aiohttp.FormData()
            data.add_field('speaker_ids', ','.join(self.speaker_ids))
            data.add_field('file', audio_data, filename='audio.wav', content_type='audio/wav')
            
            timeout = aiohttp.ClientTimeout(total=10)
            
            # 网络请求
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.api_url, headers=headers, data=data) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        speaker_id = result.get("speaker_id")
                        score = result.get("score", 0)
                        total_elapsed_time = time.monotonic() - api_start_time
                        
                        logger.bind(tag=TAG).info(f"Voiceprint recognition time: {total_elapsed_time:.3f}s")
                        
                        # 相似度阈值检查
                        if score < self.similarity_threshold:
                            logger.bind(tag=TAG).warning(f"Voiceprint similarity {score:.3f} below threshold {self.similarity_threshold}")
                            return "未知说话人"
                        
                        if speaker_id and speaker_id in self.speaker_map:
                            result_name = self.speaker_map[speaker_id]["name"]
                            logger.bind(tag=TAG).info(f"Voiceprint recognition succeeded: {result_name} (similarity: {score:.3f})")
                            return result_name
                        else:
                            logger.bind(tag=TAG).warning(f"Unrecognized speaker ID: {speaker_id}")
                            return "未知说话人"
                    else:
                        logger.bind(tag=TAG).error(f"Voiceprint recognition API error: HTTP {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - api_start_time
            logger.bind(tag=TAG).error(f"Voiceprint recognition timeout: {elapsed:.3f}s")
            return None
        except Exception as e:
            elapsed = time.monotonic() - api_start_time
            logger.bind(tag=TAG).error(f"Voiceprint recognition failed: {e}")
            return None


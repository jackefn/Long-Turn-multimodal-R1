from PIL import Image
import os
import requests
import pickle
from typing import List, Tuple, Dict, Optional
from io import BytesIO

# =======================
# Configuration Constants
# =======================
# TODO: 请将以下 API Key 替换为您的实际密钥，或从环境变量中读取
# 建议从环境变量读取: os.getenv("SERPAPI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # 替换为您的 SerpAPI API key

# SerpAPI 端点
SERPAPI_BASE_URL = "https://serpapi.com/search.json"

# 默认返回的搜索结果数量
DEFAULT_TOP_K = 3

# 缓存文件路径
CACHE_DIR = "datasets/fvqa_image_caches"
TRAIN_CACHE_FILE = os.path.join(CACHE_DIR, "fvqa_train_image_search_results_cache.pkl")
TEST_CACHE_FILE = os.path.join(CACHE_DIR, "fvqa_test_image_search_results_cache.pkl")

# 全局缓存变量，避免重复加载
_image_search_cache = {"train": None, "test": None}


def download_image_from_url(image_url: str, timeout: int = 300) -> Optional[Image.Image]:
    """
    从 URL 下载图像并转换为 PIL Image 对象。

    Args:
        image_url: 图像 URL
        timeout: 请求超时时间（秒）

    Returns:
        PIL Image 对象，如果下载失败则返回 None
    """
    try:
        response = requests.get(image_url, stream=True, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        # 转换为 RGB 模式（如果图像是 RGBA 或其他格式）
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Warning: Failed to download image from {image_url}: {e}")
        return None


def _load_image_search_cache(split: str = "test") -> Dict:
    """加载图像搜索缓存文件"""
    global _image_search_cache
    if _image_search_cache[split] is None:
        cache_file = TRAIN_CACHE_FILE if split == "train" else TEST_CACHE_FILE
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    _image_search_cache[split] = pickle.load(f)
                print(f"[Info] Loaded image search cache from {cache_file}")
            except Exception as e:
                print(f"[Warning] Failed to load cache file {cache_file}: {e}")
                _image_search_cache[split] = {}
        else:
            print(f"[Warning] Cache file not found: {cache_file}")
            _image_search_cache[split] = {}
    return _image_search_cache[split]


def call_image_search(image_url: str, top_k: int = DEFAULT_TOP_K, search_type: str = "visual_matches", 
                     data_id: Optional[str] = None, data_source: Optional[str] = None) -> Tuple[str, List[Image.Image], Dict]:
    """
    使用 SerpAPI Google Lens API 进行反向图像搜索。
    
    根据 Google Lens API 文档实现，仅支持通过图像 URL 进行搜索。
    如果 image_url 为空且提供了 data_id，则从缓存文件读取搜索结果。
    
    Args:
        image_url (str): 查询图像的 URL（必须是可公开访问的 HTTP/HTTPS URL），如果为空则从缓存读取
        top_k (int): 返回的搜索结果数量（默认 3）
        search_type (str): 搜索类型，可选值: "all", "products", "exact_matches", "visual_matches"（默认）
        data_id (str, optional): 数据ID，用于从缓存文件读取结果
        data_source (str, optional): 数据源，用于确定使用训练集还是测试集缓存（"train" 或 "test"）

    Returns:
        tool_returned_str (str): 格式化的搜索结果字符串，包含每个结果的图像和标题信息
        tool_returned_images (List[PIL.Image.Image]): 从搜索结果下载的图像对象列表
        tool_stat (dict): 工具执行状态字典，包含成功标志、结果数量等元数据
    """
    
    # 初始化返回变量
    tool_returned_images: List[Image.Image] = []
    tool_returned_str = "[Image Search Results] The result of the image search consists of web page information related to the image from the user's original question. Each result includes the main image from the web page and its title, ranked in descending order of search relevance, as demonstrated below:\n"

    tool_success = False
    tool_stat = {
        "success": False,
        "num_images": 0,
        "num_results": 0,
        "error": None,
    }
    
    # 如果 image_url 为空或无效，尝试从缓存文件读取
    if (not image_url or image_url == "" or 
        (not image_url.startswith("http://") and not image_url.startswith("https://"))):
        if data_id:
            print(f"[Info] Image URL is empty, trying to load from cache using data_id: {data_id}")
            # 确定使用哪个缓存文件
            split = "train" if data_source and "train" in str(data_source).lower() else "test"
            cache = _load_image_search_cache(split)
            
            if data_id in cache:
                cache_data = cache[data_id]
                tool_returned_web_title_list = cache_data.get('tool_returned_web_title_list', [])
                tool_returned_images_urls = cache_data.get('tool_returned_images_urls', [])
                
                print(f"[Info] Found cached results for {data_id}, titles: {len(tool_returned_web_title_list)}, images: {len(tool_returned_images_urls)}")
                
                # 处理缓存的图像URL（限制为 top_k 个）
                image_urls_to_process = tool_returned_images_urls[:top_k]
                titles_to_process = tool_returned_web_title_list[:top_k] if len(tool_returned_web_title_list) >= len(image_urls_to_process) else tool_returned_web_title_list + [f"Result {i+1}" for i in range(len(image_urls_to_process) - len(tool_returned_web_title_list))]
                
                for idx, (img_url, title) in enumerate(zip(image_urls_to_process, titles_to_process), 1):
                    try:
                        # 如果 img_url 是 PIL Image 对象，直接使用
                        if isinstance(img_url, Image.Image):
                            tool_returned_images.append(img_url)
                            tool_returned_str += f"{idx}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
                        # 如果是 URL 字符串，下载图像
                        elif isinstance(img_url, str) and (img_url.startswith("http://") or img_url.startswith("https://")):
                            image = download_image_from_url(img_url)
                            if image:
                                tool_returned_images.append(image)
                                tool_returned_str += f"{idx}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
                            else:
                                tool_returned_str += f"{idx}. title: {title}\n"
                        else:
                            # 无法处理的图像，只添加标题
                            tool_returned_str += f"{idx}. title: {title}\n"
                    except Exception as e:
                        print(f"[Warning] Error processing cached image {idx}: {e}")
                        tool_returned_str += f"{idx}. title: {title}\n"
                        continue
                
                tool_success = len(tool_returned_images) > 0
                tool_stat = {
                    "success": tool_success,
                    "num_images": len(tool_returned_images),
                    "num_results": len(image_urls_to_process),
                    "source": "cache",
                }
                
                if not tool_success:
                    tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
                
                return tool_returned_str, tool_returned_images, tool_stat
            else:
                print(f"[Warning] data_id {data_id} not found in cache")
        else:
            error_msg = "Image URL is not set and no data_id provided. Cannot perform search."
            print(f"[Error] {error_msg}")
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
            tool_stat["error"] = error_msg
            return tool_returned_str, tool_returned_images, tool_stat
    
    # 检查 API Key（只有在实际调用 API 时才需要）
    if SERPAPI_API_KEY == "REDACTED" or not SERPAPI_API_KEY:
        error_msg = "SERPAPI_API_KEY is not set. Please set it as an environment variable or update the code."
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
        return tool_returned_str, tool_returned_images, tool_stat
    
    try:
        # 根据 Google Lens API 文档构建请求参数
        # 必需参数: engine=google_lens, url, api_key
        # 可选参数: type (all/products/exact_matches/visual_matches)
        params = {
            "engine": "google_lens",
            "url": image_url,
            "api_key": SERPAPI_API_KEY,
            "type": search_type,  # 搜索类型
        }
        
        print(f"[Info] Calling SerpAPI Google Lens for reverse image search...")
        print(f"[Debug] Image URL: {image_url}")
        print(f"[Debug] Search type: {search_type}")
        
        response = requests.get(SERPAPI_BASE_URL, params=params, timeout=300)
        
        # 输出错误响应信息
        if response.status_code != 200:
            error_msg = f"SerpAPI returned status code {response.status_code}: {response.text}"
            print(f"[Error] {error_msg}")
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
            tool_stat["error"] = error_msg
            return tool_returned_str, tool_returned_images, tool_stat
        
        results = response.json()
        
        # 检查搜索状态
        search_metadata = results.get("search_metadata", {})
        if search_metadata.get("status") != "Success":
            error_msg = f"Search failed with status: {search_metadata.get('status')}"
            if "error" in search_metadata:
                error_msg += f" - {search_metadata['error']}"
            print(f"[Error] {error_msg}")
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
            tool_stat["error"] = error_msg
            return tool_returned_str, tool_returned_images, tool_stat
        
        # 解析 Google Lens API 响应
        # 根据文档，结果在 visual_matches 字段中
        visual_matches = results.get("visual_matches", [])
        
        if not visual_matches:
            print("[Warning] No visual matches found from SerpAPI Google Lens")
            tool_returned_str = "[Image Search Results] No relevant results found for the provided image."
            tool_stat["num_results"] = 0
            return tool_returned_str, tool_returned_images, tool_stat
        
        # 处理每个搜索结果（限制为 top_k 个）
        search_results = visual_matches[:top_k]
        
        for idx, result in enumerate(search_results, 1):
            try:
                # 根据 Google Lens API 文档，visual_matches 包含以下字段：
                # position, title, link, source, thumbnail, image, thumbnail_width, thumbnail_height, etc.
                thumbnail_url = result.get("thumbnail", "") or result.get("image", "")
                title = result.get("title", "") or f"Result {idx}"
                link = result.get("link", "")
                source = result.get("source", "")
                
                # 下载图像（优先使用 thumbnail，如果没有则使用 image）
                if thumbnail_url:
                    image = download_image_from_url(thumbnail_url)
                    if image:
                        tool_returned_images.append(image)
                        tool_returned_str += f"{idx}. image: <|vision_start|><|image_pad|><|vision_end|>\ntitle: {title}\n"
                        if source:
                            tool_returned_str += f"source: {source}\n"
                        if link:
                            tool_returned_str += f"link: {link}\n"
                    else:
                        # 即使图像下载失败，也添加文本结果
                        tool_returned_str += f"{idx}. title: {title}\n"
                        if source:
                            tool_returned_str += f"source: {source}\n"
                        if link:
                            tool_returned_str += f"link: {link}\n"
                else:
                    # 没有图像 URL，只添加文本信息
                    tool_returned_str += f"{idx}. title: {title}\n"
                    if source:
                        tool_returned_str += f"source: {source}\n"
                    if link:
                        tool_returned_str += f"link: {link}\n"
                        
            except Exception as e:
                print(f"[Warning] Error processing search result {idx}: {e}")
                continue
        
        tool_success = len(tool_returned_images) > 0
        tool_stat = {
            "success": tool_success,
            "num_images": len(tool_returned_images),
            "num_results": len(search_results),
        }
        
        if not tool_success:
            tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error when calling SerpAPI: {e}"
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
    except KeyError as e:
        error_msg = f"Unexpected response structure from SerpAPI: {e}"
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
    except Exception as e:
        error_msg = f"Unexpected error during image search: {e}"
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Image Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg

    return tool_returned_str, tool_returned_images, tool_stat

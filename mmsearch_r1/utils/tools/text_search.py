import os
import asyncio
import aiohttp
import json
from typing import List, Tuple, Dict, Optional

# 尝试导入 nest_asyncio 以支持嵌套事件循环
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio 不可用，将在运行时处理

# =======================
# Configuration Constants
# =======================
# TODO: 请将以下 API Key 替换为您的实际密钥，或从环境变量中读取
# 建议从环境变量读取: os.getenv("API_KEY_NAME")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # 替换为您的 OpenRouter API key
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")  # 替换为您的 SERPAPI API key
JINA_API_KEY = os.getenv("JINA_API_KEY", "")  # 替换为您的 JINA API key

# Endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SERPAPI_URL = "https://serpapi.com/search.json"
JINA_BASE_URL = "https://r.jina.ai/"

# Default LLM model (Qwen3-32B)
DEFAULT_MODEL = "qwen/qwen3-32b"

# 默认返回的搜索结果数量
DEFAULT_TOP_K = 3


# ============================
# Asynchronous Helper Functions
# ============================

async def call_openrouter_async(session: aiohttp.ClientSession, messages: List[Dict], model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    异步调用 OpenRouter chat completion API。
    返回助手回复的内容。
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError) as e:
                    print(f"Unexpected OpenRouter response structure: {result}")
                    return None
            else:
                text = await resp.text()
                print(f"OpenRouter API error: {resp.status} - {text}")
                return None
    except Exception as e:
        print(f"Error calling OpenRouter: {e}")
        return None


async def perform_search_async(session: aiohttp.ClientSession, query: str) -> List[str]:
    """
    使用 SERPAPI 异步执行 Google 搜索。
    返回结果 URL 列表。
    """
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"
    }
    try:
        async with session.get(SERPAPI_URL, params=params) as resp:
            if resp.status == 200:
                results = await resp.json()
                if "organic_results" in results:
                    links = [item.get("link") for item in results["organic_results"] if "link" in item]
                    return links
                else:
                    print("No organic results in SERPAPI response.")
                    return []
            else:
                text = await resp.text()
                print(f"SERPAPI error: {resp.status} - {text}")
                return []
    except Exception as e:
        print(f"Error performing SERPAPI search: {e}")
        return []


async def fetch_webpage_text_async(session: aiohttp.ClientSession, url: str) -> str:
    """
    使用 Jina Reader 异步获取网页的文本内容。
    """
    full_url = f"{JINA_BASE_URL}{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    try:
        async with session.get(full_url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                text = await resp.text()
                print(f"Jina fetch error for {url}: {resp.status} - {text}")
                return ""
    except Exception as e:
        print(f"Error fetching webpage text with Jina: {e}")
        return ""


async def generate_summary_async(session: aiohttp.ClientSession, user_query: str, page_text: str) -> str:
    """
    使用 Qwen3-32B 根据原始查询生成网页内容的摘要。
    """
    # 限制页面文本长度以避免超出 token 限制
    max_chars = 20000
    truncated_text = page_text[:max_chars] if len(page_text) > max_chars else page_text
    
    prompt = (
        "You are an expert information summarizer. Given the user's query and the webpage content, "
        "generate a concise and relevant summary that addresses the query. "
        "Return only the summary text without any additional commentary."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes web content."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content:\n{truncated_text}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        return response.strip()
    return ""


async def process_url_async(session: aiohttp.ClientSession, url: str, user_query: str) -> Optional[Tuple[str, str]]:
    """
    处理单个 URL：获取内容并生成摘要。
    返回 (summary, url) 元组，如果失败则返回 None。
    """
    try:
        print(f"Fetching content from: {url}")
        page_text = await fetch_webpage_text_async(session, url)
        if not page_text:
            print(f"No content retrieved for {url}")
            return None
        
        print(f"Generating summary for: {url}")
        summary = await generate_summary_async(session, user_query, page_text)
        if not summary:
            print(f"Failed to generate summary for {url}")
            return None
        
        return (summary, url)
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None


# =========================
# Main Asynchronous Routine
# =========================

async def async_text_search(text_query: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, Dict]:
    """
    异步执行文本搜索的主函数。

    Args:
        text_query (str): 输入查询字符串
        top_k (int): 返回前 k 个最相关的搜索结果

    Returns:
        tool_returned_str (str): 格式化的搜索结果字符串
        tool_stat (dict): 工具执行状态字典
    """
    tool_returned_str = "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:\n"
    tool_stat = {
        "success": False,
        "num_results": 0,
        "error": None,
    }
    
    # 检查 API Keys
    if SERPAPI_API_KEY == "REDACTED" or not SERPAPI_API_KEY:
        error_msg = "SERPAPI_API_KEY is not set. Please set it as an environment variable."
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
        return tool_returned_str, tool_stat
    
    if JINA_API_KEY == "REDACTED" or not JINA_API_KEY:
        error_msg = "JINA_API_KEY is not set. Please set it as an environment variable."
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
        return tool_returned_str, tool_stat
    
    if OPENROUTER_API_KEY == "REDACTED" or not OPENROUTER_API_KEY:
        error_msg = "OPENROUTER_API_KEY is not set. Please set it as an environment variable."
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg
        return tool_returned_str, tool_stat
    
    try:
        async with aiohttp.ClientSession() as session:
            # 1. 使用 SerpAPI 搜索相关网页 URL
            print(f"[Info] Searching with SerpAPI for query: {text_query}")
            search_urls = await perform_search_async(session, text_query)
            
            if not search_urls:
                print("[Warning] No search results found from SerpAPI")
                tool_returned_str = "[Text Search Results] No relevant results found for the provided query."
                tool_stat["num_results"] = 0
                return tool_returned_str, tool_stat
            
            # 限制为 top_k 个 URL
            search_urls = search_urls[:top_k]
            print(f"[Info] Found {len(search_urls)} URLs, processing...")
            
            # 2. 并发处理每个 URL：使用 JINA Reader 获取内容，使用 Qwen3-32B 生成摘要
            url_tasks = [process_url_async(session, url, text_query) for url in search_urls]
            url_results = await asyncio.gather(*url_tasks)
            
            # 3. 收集成功的结果
            valid_results = [result for result in url_results if result is not None]
            
            if not valid_results:
                print("[Warning] No valid summaries generated")
                tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
                tool_stat["num_results"] = 0
                return tool_returned_str, tool_stat
            
            # 4. 格式化输出结果
            for idx, (summary, url) in enumerate(valid_results, 1):
                tool_returned_str += f"{idx}. ({url}) {summary}\n"
            
            tool_stat["success"] = True
            tool_stat["num_results"] = len(valid_results)
            
            print(f"[Info] Successfully generated {len(valid_results)} summaries")
            
    except Exception as e:
        error_msg = f"Unexpected error during text search: {e}"
        print(f"[Error] {error_msg}")
        tool_returned_str = "[Text Search Results] There is an error encountered in performing search. Please reason with your own capaibilities."
        tool_stat["error"] = error_msg

    return tool_returned_str, tool_stat


def call_text_search(text_query: str):
    """
    文本搜索工具的主要函数。
    
    该工具结合了 SerpAPI、JINA Reader 和 Qwen3-32B 进行文本摘要：
    1. 使用 SerpAPI 检索前 k 个最相关的网页 URL
    2. 使用 JINA Reader 解析并清理这些网页的内容
    3. 使用 Qwen3-32B 根据原始查询生成摘要
    4. 返回包含前 k 个最相关网页摘要段落及其对应链接的格式化结果

    Args:
        text_query (str): 输入查询字符串，用于基于文本的搜索。

    Returns:
        tool_returned_str (str): 格式化的搜索结果字符串，包含排名搜索结果。
        tool_stat (dict): 指示工具执行状态和附加元数据的字典。
    """
    
    # 使用 asyncio.run() 运行异步函数
    # nest_asyncio 已在文件开头应用（如果可用），允许嵌套事件循环
    try:
        return asyncio.run(async_text_search(text_query))
    except RuntimeError as e:
        # 如果已经在事件循环中运行，尝试获取现有循环并创建任务
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环正在运行且 nest_asyncio 不可用，提示用户
                print("[Error] Cannot run async function in a running event loop. Please install nest_asyncio: pip install nest_asyncio")
                raise
            else:
                return loop.run_until_complete(async_text_search(text_query))
        except RuntimeError:
            # 最后尝试：直接运行（nest_asyncio 应该已经处理了嵌套情况）
            return asyncio.run(async_text_search(text_query))

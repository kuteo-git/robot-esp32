import requests
import sys
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# Define the base function description template
SEARCH_FROM_RAGFLOW_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_from_ragflow",
        "description": "从知识库中查询信息",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "查询的问题"}},
            "required": ["question"],
        },
    },
}


@register_function(
    "search_from_ragflow", SEARCH_FROM_RAGFLOW_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def search_from_ragflow(conn: "ConnectionHandler", question=None):
    # Make sure the string parameter's encoding is handled correctly
    if question and isinstance(question, str):
        # Make sure the question parameter is a UTF-8 encoded string
        pass
    else:
        question = str(question) if question is not None else ""

    ragflow_config = conn.config.get("plugins", {}).get("search_from_ragflow", {})
    base_url = ragflow_config.get("base_url", "")
    api_key = ragflow_config.get("api_key", "")
    dataset_ids = ragflow_config.get("dataset_ids", [])

    url = base_url + "/api/v1/retrieval"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Make sure all strings in the payload are UTF-8 encoded
    payload = {"question": question, "dataset_ids": dataset_ids}

    try:
        # Use ensure_ascii=False to handle Chinese correctly during JSON serialization
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=5,
            verify=False,
        )

        # Explicitly set the response encoding to utf-8
        response.encoding = "utf-8"

        response.raise_for_status()

        # Get the text content first, then decode the JSON manually
        response_text = response.text
        import json

        result = json.loads(response_text)

        if result.get("code") != 0:
            error_detail = result.get("error", {}).get("detail", "未知错误")
            error_message = result.get("error", {}).get("message", "")
            error_code = result.get("code", "")

            # Log the error info safely
            logger.bind(tag=TAG).error(
                f"RAGFlow API调用失败，响应码：{error_code}，错误详情：{error_detail}，完整响应：{result}"
            )

            # Build a detailed error response
            error_response = f"RAG Giao diện trả về lỗi (mã lỗi: {error_code}) "

            if error_message:
                error_response += f"：{error_message}"
            if error_detail:
                error_response += f"\n详情：{error_detail}"

            return ActionResponse(Action.RESPONSE, None, error_response)

        chunks = result.get("data", {}).get("chunks", [])
        contents = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if content:
                # Handle the content string safely
                if isinstance(content, str):
                    contents.append(content)
                elif isinstance(content, bytes):
                    contents.append(content.decode("utf-8", errors="replace"))
                else:
                    contents.append(str(content))

        if contents:
            # Format the knowledge-base content as a quoted block
            context_text = f"# 关于问题【{question}】查到知识库如下\n"
            context_text += "```\n\n\n".join(contents[:5])
            context_text += "\n```"
        else:
            context_text = "Tra trong kho dữ liệu không có thông tin gì liên quan hết."
        return ActionResponse(Action.REQLLM, context_text, None)

    except requests.exceptions.RequestException as e:
        # Network request exception
        error_type = type(e).__name__
        logger.bind(tag=TAG).error(
            f"RAGflow网络请求失败，异常类型：{error_type}，详情：{str(e)}"
        )

        # Provide more detailed error info and a suggested fix based on the exception type
        if isinstance(e, requests.exceptions.ConnectTimeout):
            error_response = "RAG Kết nối giao diện quá thời gian ( 5 giây)"
            error_response += "\n可能原因：RAGflow服务未启动或网络连接问题"
            error_response += "\n解决方案：请检查RAGflow服务状态和网络连接"

        elif isinstance(e, requests.exceptions.ConnectionError):
            error_response = "Không kết nối được tới RAG giao diện"
            error_response += "\n可能原因：RAGflow服务地址错误或服务未运行"
            error_response += "\n解决方案：请检查RAGflow服务地址配置和服务状态"

        elif isinstance(e, requests.exceptions.Timeout):
            error_response = "RAG Yêu cầu giao diện quá thời gian"
            error_response += "\n可能原因：RAGflow服务响应缓慢或网络延迟"
            error_response += "\n解决方案：请稍后重试或检查RAGflow服务性能"

        elif isinstance(e, requests.exceptions.HTTPError):
            # Handle the HTTP error status code
            if hasattr(e.response, "status_code"):
                status_code = e.response.status_code
                error_response = f"RAG giao diện HTTP Lỗi (mã trạng thái: {status_code}) "

                # Try to get the error message from the response body
                try:
                    error_detail = e.response.json().get("error", {}).get("message", "")
                    if error_detail:
                        error_response += f"\n错误详情：{error_detail}"
                except:
                    pass
            else:
                error_response = f"RAG giao diện HTTP lỗi: {str(e)}"

        else:
            error_response = f"RAG Giao diện lỗi mạng ( {error_type}) : {str(e)}"

        return ActionResponse(Action.RESPONSE, None, error_response)

    except Exception as e:
        # Other exceptions
        error_type = type(e).__name__
        logger.bind(tag=TAG).error(
            f"RAGflow处理异常，异常类型：{error_type}，详情：{str(e)}"
        )

        # Provide detailed error info
        error_response = f"RAG Giao diện xử lý lỗi ( {error_type}) : {str(e)}"
        return ActionResponse(Action.RESPONSE, None, error_response)

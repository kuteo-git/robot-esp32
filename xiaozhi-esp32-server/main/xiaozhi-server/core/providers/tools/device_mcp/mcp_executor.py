"""设备端MCP工具执行器"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from ..base import ToolType, ToolDefinition, ToolExecutor
from plugins_func.register import Action, ActionResponse
from .mcp_handler import call_mcp_tool


class DeviceMCPExecutor(ToolExecutor):
    """设备端MCP工具执行器"""

    def __init__(self, conn):
        self.conn = conn

    async def execute(
        self, conn: "ConnectionHandler", tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        """执行设备端MCP工具"""
        if not hasattr(conn, "mcp_client") or not conn.mcp_client:
            return ActionResponse(
                action=Action.ERROR,
                response="phía thiết bị MCP Thiết bị chưa khởi tạo",
            )

        if not await conn.mcp_client.is_ready():
            return ActionResponse(
                action=Action.ERROR,
                response="phía thiết bị MCP Thiết bị chưa sẵn sàng",
            )

        try:
            # 转换参数为JSON字符串
            import json

            args_str = json.dumps(arguments) if arguments else "{}"

            # 调用设备端MCP工具
            result = await call_mcp_tool(conn, conn.mcp_client, tool_name, args_str)

            resultJson = None
            if isinstance(result, str):
                try:
                    resultJson = json.loads(result)
                except Exception as e:
                    pass

            # 视觉大模型不经过二次LLM处理
            if (
                resultJson is not None
                and isinstance(resultJson, dict)
                and "action" in resultJson
            ):
                return ActionResponse(
                    action=Action[resultJson["action"]],
                    response=resultJson.get("response", ""),
                )

            return ActionResponse(action=Action.REQLLM, result=str(result))

        except ValueError as e:
            return ActionResponse(action=Action.NOTFOUND, response=str(e))
        except Exception as e:
            return ActionResponse(action=Action.ERROR, response=str(e))

    def _excluded_tools(self) -> set:
        # Tool MCP chạy TRÊN thiết bị (vd PHICOMM R1) bị loại khỏi danh sách gửi LLM.
        # Lý do: R1 RAM ít (Android 5.1.1), tool nặng (weather/news/music...) tự fetch+parse
        # ngay trên loa -> OOM -> app crash. Để server lo (đã có get_weather/get_news_vietnam/
        # play_music_room/hass_*). Khai báo ở config: mcp_device_exclude_tools.
        cfg = getattr(self.conn, "config", None) or {}
        return set(cfg.get("mcp_device_exclude_tools", []) or [])

    def get_tools(self) -> Dict[str, ToolDefinition]:
        """获取所有设备端MCP工具"""
        if not hasattr(self.conn, "mcp_client") or not self.conn.mcp_client:
            return {}

        excluded = self._excluded_tools()
        tools = {}
        mcp_tools = self.conn.mcp_client.get_available_tools()

        for tool in mcp_tools:
            func_def = tool.get("function", {})
            tool_name = func_def.get("name", "")

            if tool_name and tool_name not in excluded:
                tools[tool_name] = ToolDefinition(
                    name=tool_name, description=tool, tool_type=ToolType.DEVICE_MCP
                )

        return tools

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定的设备端MCP工具"""
        if not hasattr(self.conn, "mcp_client") or not self.conn.mcp_client:
            return False

        if tool_name in self._excluded_tools():
            return False

        return self.conn.mcp_client.has_tool(tool_name)

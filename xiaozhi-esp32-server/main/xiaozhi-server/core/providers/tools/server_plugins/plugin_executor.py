"""服务端插件工具执行器"""

import asyncio
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler
from ..base import ToolType, ToolDefinition, ToolExecutor
from plugins_func.register import all_function_registry, Action, ActionResponse

class ServerPluginExecutor(ToolExecutor):
    """服务端插件工具执行器"""

    def __init__(self, conn: "ConnectionHandler"):
        self.conn = conn
        self.config = conn.config

    async def execute(
        self, conn: "ConnectionHandler", tool_name: str, arguments: Dict[str, Any]
    ) -> ActionResponse:
        """执行服务端插件工具"""
        func_item = all_function_registry.get(tool_name)
        if not func_item:
            return ActionResponse(
                action=Action.NOTFOUND, response=f"hàm plugin {tool_name} không tồn tại"
            )

        try:
            # 根据工具类型决定如何调用
            func_type = getattr(func_item, "type", None)
            if func_type is not None and func_type.code in [3, 4, 5]:  # CHANGE_SYS_PROMPT, SYSTEM_CTL, IOT_CTL (需要conn参数)
                args = (conn,)
            else:
                args = ()

            # The event loop is shared by every device. Plugins are synchronous and many make HTTP
            # calls of 20-60s; run on the loop, one froze audio and messages for every connection.
            return await asyncio.to_thread(func_item.func, *args, **arguments)

        except Exception as e:
            return ActionResponse(
                action=Action.ERROR,
                response=str(e),
            )

    def get_tools(self) -> Dict[str, ToolDefinition]:
        """获取所有注册的服务端插件工具"""
        tools = {}

        # 获取必要的函数
        necessary_functions = ["handle_exit_intent", "get_lunar"]

        # 获取配置中的函数
        config_functions = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ].get("functions", [])

        # 转换为列表
        if not isinstance(config_functions, list):
            try:
                config_functions = list(config_functions)
            except TypeError:
                config_functions = []

        # 合并所有需要的函数
        all_required_functions = list(set(necessary_functions + config_functions))

        for func_name in all_required_functions:
            func_item = all_function_registry.get(func_name)
            if func_item:
                # 从函数注册中获取描述
                fun_description = (
                    self.config.get("plugins", {})
                    .get(func_name, {})
                    .get("description", "")
                )
                if fun_description is not None and len(fun_description) > 0:
                    if "function" in func_item.description and isinstance(
                        func_item.description["function"], dict
                    ):
                        func_item.description["function"][
                            "description"
                        ] = fun_description

                # 新闻插件：根据配置更新新闻源参数描述
                if func_name == "get_news_from_newsnow":
                    self._init_news_source_description(func_item, func_name)

                tools[func_name] = ToolDefinition(
                    name=func_name,
                    description=func_item.description,
                    tool_type=ToolType.SERVER_PLUGIN,
                )

        return tools

    def has_tool(self, tool_name: str) -> bool:
        """检查是否有指定的服务端插件工具"""
        return tool_name in all_function_registry

    def _init_news_source_description(self, func_item, func_name):
        """根据连接配置初始化新闻工具的参数描述"""
        news_sources = (
            self.config.get("plugins", {})
            .get(func_name, {})
            .get("news_sources", "")
        )
        if not news_sources:
            news_sources = "澎湃新闻;百度热搜;财联社"
        sources_str = news_sources.replace(";", "、")
        try:
            func_item.description["function"]["parameters"]["properties"]["source"][
                "description"
            ] = f"新闻源的标准中文名称，例如{sources_str}等。可选参数，如果不提供则使用默认新闻源"
        except (KeyError, TypeError):
            pass

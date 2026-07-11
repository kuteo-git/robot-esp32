from plugins_func.register import register_function, ToolType, ActionResponse, Action
from plugins_func.functions.hass_init import initialize_hass_handler
from config.logger import setup_logging
import asyncio
import requests
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

hass_get_state_function_desc = {
    "type": "function",
    "function": {
        "name": "hass_get_state",
        "description": "获取homeassistant里设备的状态,包括查询灯光亮度、颜色、色温,媒体播放器的音量,设备的暂停、继续操作",
        "parameters": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "需要操作的设备id,homeassistant里的entity_id",
                }
            },
            "required": ["entity_id"],
        },
    },
}


@register_function("hass_get_state", hass_get_state_function_desc, ToolType.SYSTEM_CTL)
def hass_get_state(conn: "ConnectionHandler", entity_id=""):
    try:
        ha_response = handle_hass_get_state(conn, entity_id)
        return ActionResponse(Action.REQLLM, ha_response, None)
    except asyncio.TimeoutError:
        logger.bind(tag=TAG).error("Đọc trạng thái Home Assistant quá thời gian")
        return ActionResponse(Action.ERROR, "Hết thời gian chờ", None)
    except Exception as e:
        error_msg = "Đọc trạng thái Home Assistant thất bại"
        logger.bind(tag=TAG).error(error_msg)
        return ActionResponse(Action.ERROR, error_msg, None)


def handle_hass_get_state(conn: "ConnectionHandler", entity_id):
    ha_config = initialize_hass_handler(conn)
    api_key = ha_config.get("api_key")
    base_url = ha_config.get("base_url")
    url = f"{base_url}/api/states/{entity_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code == 200:
        responsetext = "Trạng thái thiết bị: " + response.json()["state"] + " "
        logger.bind(tag=TAG).info(f"api Returned content: {response.json()}")

        if "media_title" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Đang phát: "
                + str(response.json()["attributes"]["media_title"])
                + " "
            )
        if "volume_level" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Âm lượng: "
                + str(response.json()["attributes"]["volume_level"])
                + " "
            )
        if "color_temp_kelvin" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Nhiệt màu: "
                + str(response.json()["attributes"]["color_temp_kelvin"])
                + " "
            )
        if "rgb_color" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Màu RGB: "
                + str(response.json()["attributes"]["rgb_color"])
                + " "
            )
        if "brightness" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Độ sáng: "
                + str(response.json()["attributes"]["brightness"])
                + " "
            )
        # Also return the 'result' attribute (e.g. the gemini weather sensor has its full description in result)
        if "result" in response.json()["attributes"]:
            responsetext = (
                responsetext
                + "Nội dung chi tiết: "
                + str(response.json()["attributes"]["result"])
                + " "
            )
        logger.bind(tag=TAG).info(f"Query returned content: {responsetext}")
        return responsetext
        # return response.json()['attributes']
        # response.attributes

    else:
        return f"Không đọc được trạng thái, mã lỗi: {response.status_code}"

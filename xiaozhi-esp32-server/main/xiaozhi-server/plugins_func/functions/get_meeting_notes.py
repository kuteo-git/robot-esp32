"""
get_meeting_notes — calls the local meeting-notes Q&A service (git/meeting note, server.js :8080).
That service does RAG over logged meeting notes + summaries, then answers via an LLM (OmniRoute).
2026-07-24: lets the robot answer questions about past meetings/work updates on demand.

Response format of POST /api/ai-ask (see server.js's aiAskStream): first line is a JSON
{"citations": {...}} blob, everything after that is the actual answer text -- we drop the
citations line and return just the answer for the robot's own LLM to summarize aloud.
"""
import os
import requests
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

MEETING_NOTES_URL = os.environ.get("MEETING_NOTES_SERVICE_URL", "http://127.0.0.1:8080/api/ai-ask")

GET_MEETING_NOTES_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "get_meeting_notes",
        "description": (
            "Hỏi về các cuộc họp/công việc đã ghi lại (vd: tuần này team X có blocker gì, "
            "họp hôm qua bàn gì, ai phụ trách việc Y). Dùng khi người dùng hỏi về nội dung "
            "họp, cập nhật công việc, hoặc thông tin từ ghi chú cuộc họp trước đây."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Câu hỏi của người dùng, giữ nguyên ý gốc (có thể dịch/viết lại cho rõ nghĩa nếu cần).",
                },
            },
            "required": ["question"],
        },
    },
}


@register_function("get_meeting_notes", GET_MEETING_NOTES_FUNCTION_DESC, ToolType.SYSTEM_CTL)
def get_meeting_notes(conn, question: str):
    try:
        r = requests.post(MEETING_NOTES_URL, json={"q": question}, timeout=60)
        r.raise_for_status()
        # First line = {"citations": {...}} metadata, drop it -- the rest is the answer text.
        _, _, answer = r.text.partition("\n")
        answer = answer.strip()
        if not answer:
            answer = "Không tìm thấy thông tin liên quan trong ghi chú cuộc họp."
        return ActionResponse(Action.REQLLM, answer, None)
    except Exception as e:
        logger.bind(tag=TAG).error(f"gọi meeting-notes service lỗi: {e}")
        return ActionResponse(
            Action.REQLLM, "Tạm thời chưa hỏi được ghi chú cuộc họp, thử lại sau nha", None
        )

"""Every assistant tool_calls message is followed at once by a tool message for each call.

OpenAI-style APIs reject a history where a tool_call id has no tool message right after its
assistant message (400). Two ways the history broke that:
- a REQLLM tool returning empty text got no tool message at all;
- the repair in Dialogue._ensure_tool_calls_complete added the missing tool message at the END of
  the history, after later assistant and user messages, which is still out of order.

Run: services/.venv-xiaozhi/bin/python tests/test_dialogue_tool_order.py
"""

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.connection import ConnectionHandler
from core.utils.dialogue import Dialogue, Message
from plugins_func.register import Action, ActionResponse


def _tc(call_id, name="get_weather"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _assert_tool_calls_answered_in_place(history):
    for i, m in enumerate(history):
        if m.get("tool_calls"):
            ids = [tc["id"] for tc in m["tool_calls"]]
            following = history[i + 1 : i + 1 + len(ids)]
            got = [f.get("tool_call_id") for f in following if f["role"] == "tool"]
            assert sorted(got) == sorted(ids), (
                f"tool_calls {ids} at {i} not followed by their tool messages: "
                f"{[(f['role'], f.get('tool_call_id')) for f in history[i + 1:]]}"
            )


def test_missing_tool_message_is_repaired_in_place():
    d = Dialogue()
    d.put(Message(role="system", content="sys"))
    d.put(Message(role="user", content="thời tiết?"))
    d.put(Message(role="assistant", tool_calls=[_tc("c1"), _tc("c2")]))
    d.put(Message(role="tool", tool_call_id="c1", content="nắng"))
    d.put(Message(role="assistant", content="Trời nắng."))
    d.put(Message(role="user", content="cảm ơn"))
    _assert_tool_calls_answered_in_place(d.get_llm_dialogue())
    print("  dangling tool_call repaired right after its assistant message")


def test_empty_reqllm_result_still_writes_a_tool_message():
    class _Conn:
        def __init__(self):
            self.dialogue = Dialogue()
            self.recursed = False

        def chat(self, query, depth=0, sentence_id=None):
            self.recursed = True

    conn = _Conn()
    conn.dialogue.put(Message(role="user", content="tìm giùm"))
    tool_results = [(ActionResponse(action=Action.REQLLM, result="", response=None), {"id": "c1", "name": "web_search", "arguments": "{}"})]
    ConnectionHandler._handle_function_result(conn, tool_results, depth=0, sentence_id="A")
    assert conn.recursed
    raw = [(m.role, m.tool_call_id) for m in conn.dialogue.dialogue]
    assert ("tool", "c1") in raw, f"empty tool result got no tool message: {raw}"
    _assert_tool_calls_answered_in_place(conn.dialogue.get_llm_dialogue())
    print("  empty REQLLM result answered with a tool message")


def test_a_complete_tool_exchange_is_left_untouched():
    d = Dialogue()
    d.put(Message(role="system", content="sys"))
    d.put(Message(role="user", content="thời tiết?"))
    d.put(Message(role="assistant", tool_calls=[_tc("c1"), _tc("c2")]))
    d.put(Message(role="tool", tool_call_id="c2", content="mưa"))
    d.put(Message(role="tool", tool_call_id="c1", content="nắng"))
    d.put(Message(role="assistant", content="Nắng và mưa."))
    out = d.get_llm_dialogue()
    assert len(out) == 6 and not any("interrupted" in str(m.get("content")) for m in out), out
    _assert_tool_calls_answered_in_place(out)
    print("  complete tool exchange (answers in any order) not altered")


def test_a_tool_call_dangling_at_the_very_end_is_answered():
    d = Dialogue()
    d.put(Message(role="system", content="sys"))
    d.put(Message(role="user", content="mở nhạc"))
    d.put(Message(role="assistant", tool_calls=[_tc("c1", "play_music")]))
    out = d.get_llm_dialogue()
    assert out[-1]["role"] == "tool" and out[-1]["tool_call_id"] == "c1", out
    print("  trailing dangling tool_call gets its tool message")


if __name__ == "__main__":
    test_missing_tool_message_is_repaired_in_place()
    test_empty_reqllm_result_still_writes_a_tool_message()
    test_a_complete_tool_exchange_is_left_untouched()
    test_a_tool_call_dangling_at_the_very_end_is_answered()
    print("ALL dialogue_tool_order TESTS PASSED")

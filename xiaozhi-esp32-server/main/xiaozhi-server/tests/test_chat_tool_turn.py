"""chat() turns that go through a tool call.

1. Truncated tool arguments. The LLM can stop mid-JSON (token limit, dropped stream). chat()
   parsed the arguments for the report outside any try, so the JSONDecodeError left chat(), the
   executor swallowed it, no LAST was queued and the R1 stayed on "đang trả lời". The turn must
   close, with the tool handler's own error reply spoken.

2. A superseded turn. While a tool runs, the user can start a new turn (conn.sentence_id moves
   to B, client_abort back to False). When the tool returns, the old turn must not write anything
   under B: its reply (or its follow-up LLM pass) would be read out inside the new answer.

Run: services/.venv-xiaozhi/bin/python tests/test_chat_tool_turn.py
"""

import os, sys, json, queue, asyncio, threading
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.connection import ConnectionHandler
from core.utils.dialogue import Dialogue
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.register import Action, ActionResponse


class _FakeLogger:
    def bind(self, **kw):
        return self

    def debug(self, *a, **kw):
        pass

    info = warning = error = debug


class _FakeTTS:
    def __init__(self):
        self.tts_text_queue = queue.Queue()

    tts_one_sentence = TTSProviderBase.tts_one_sentence

    def store_tts_text(self, sentence_id, text):
        pass

    def start_thinking_loop(self):
        pass

    def stop_thinking_loop(self):
        pass

    def drain(self):
        items = []
        while not self.tts_text_queue.empty():
            items.append(self.tts_text_queue.get_nowait())
        return items


def _tool_call(name, arguments):
    return SimpleNamespace(index=0, id="call-1", function=SimpleNamespace(name=name, arguments=arguments))


class _FakeLLM:
    """First pass asks for one tool; any later pass (after a REQLLM tool) answers in text."""

    def __init__(self, tool_name, arguments):
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    def response_with_functions(self, session_id, dialogue, functions=None):
        self.calls += 1
        if self.calls == 1:
            yield None, [_tool_call(self.tool_name, self.arguments)]
        else:
            yield "old turn follow-up.", None


class _BlockingHandler:
    """Holds the tool call until released, as a slow plugin (weather, web search) would."""

    def __init__(self, result):
        self.result = result
        self.entered = threading.Event()
        self.release = threading.Event()

    def get_functions(self):
        return []

    async def handle_llm_function_call(self, conn, data):
        self.entered.set()
        await asyncio.to_thread(self.release.wait)
        return self.result


class _RealArgParsingHandler:
    """The real UnifiedToolHandler.handle_llm_function_call, up to its argument parsing."""

    def __init__(self):
        self.logger = _FakeLogger()
        self.conn = None
        self.called = 0

    def get_functions(self):
        return []

    async def handle_llm_function_call(self, conn, data):
        self.called += 1
        return await UnifiedToolHandler.handle_llm_function_call(self, conn, data)


def _make_conn(llm, handler, loop):
    conn = object.__new__(ConnectionHandler)
    conn.logger = _FakeLogger()
    conn.dialogue = Dialogue()
    conn.tts = _FakeTTS()
    conn.llm = llm
    conn.func_handler = handler
    conn.intent_type = "function_call"
    conn.memory = None
    conn.session_id = "sid"
    conn.config = {}
    conn.loop = loop
    conn.client_abort = False
    conn.features = {"emoji": False}
    conn.read_config_from_api = False
    conn.sentence_id = None
    return conn


def _start_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def test_truncated_arguments_close_the_turn(loop):
    handler = _RealArgParsingHandler()
    conn = _make_conn(_FakeLLM("get_weather", '{"city": "Hà N'), handler, loop)
    raised = None
    try:
        conn.chat_turn("thời tiết hà nội")
    except Exception as e:
        raised = e
    items = conn.tts.drain()
    kinds = [i.sentence_type for i in items]
    texts = "".join(i.content_detail or "" for i in items if i.sentence_type == SentenceType.MIDDLE)
    assert raised is None, f"chat() raised {type(raised).__name__}: {raised}; the turn never closed"
    assert kinds[-1] == SentenceType.LAST, f"turn not closed: {kinds}"
    assert handler.called == 1, "the tool handler should answer the bad arguments with its error"
    assert texts.strip(), "the user should hear the tool error reply"
    print("  truncated arguments: turn closed, error reply spoken")


def test_superseded_turn_writes_nothing_under_new_id(loop, action):
    result = (
        ActionResponse(action=Action.RESPONSE, response="old turn answer")
        if action == Action.RESPONSE
        else ActionResponse(action=Action.REQLLM, result="sunny", response=None)
    )
    handler = _BlockingHandler(result)
    llm = _FakeLLM("get_weather", '{"city": "Huế"}')
    conn = _make_conn(llm, handler, loop)
    t = threading.Thread(target=conn.chat_turn, args=("thời tiết huế",), daemon=True)
    t.start()
    assert handler.entered.wait(5), "tool handler never called"
    conn.tts.drain()
    conn.sentence_id = "B"
    conn.client_abort = False
    handler.release.set()
    t.join(5)
    assert not t.is_alive(), "old turn did not finish"
    under_b = [(i.sentence_type, i.content_detail) for i in conn.tts.drain() if i.sentence_id == "B"]
    assert not under_b, f"{action.name}: old turn wrote under the new turn's id: {under_b}"
    assert llm.calls == 1, f"{action.name}: old turn ran another LLM pass after being superseded"
    print(f"  superseded turn ({action.name}): nothing under the new id")


def test_live_turn_still_speaks_its_result(loop, action):
    result = (
        ActionResponse(action=Action.RESPONSE, response="live answer")
        if action == Action.RESPONSE
        else ActionResponse(action=Action.REQLLM, result="sunny", response=None)
    )
    handler = _BlockingHandler(result)
    handler.release.set()
    conn = _make_conn(_FakeLLM("get_weather", '{"city": "Huế"}'), handler, loop)
    conn.chat_turn("thời tiết huế")
    items = conn.tts.drain()
    sid = conn.sentence_id
    texts = "".join(i.content_detail or "" for i in items if i.sentence_id == sid)
    expected = "live answer" if action == Action.RESPONSE else "old turn follow-up."
    assert expected in texts, f"{action.name}: live turn lost its reply: {texts!r}"
    assert items[-1].sentence_type == SentenceType.LAST and items[-1].sentence_id == sid
    print(f"  live turn ({action.name}): reply spoken, turn closed")


if __name__ == "__main__":
    loop = _start_loop()
    test_truncated_arguments_close_the_turn(loop)
    test_superseded_turn_writes_nothing_under_new_id(loop, Action.RESPONSE)
    test_superseded_turn_writes_nothing_under_new_id(loop, Action.REQLLM)
    test_live_turn_still_speaks_its_result(loop, Action.RESPONSE)
    test_live_turn_still_speaks_its_result(loop, Action.REQLLM)
    print("ALL chat_tool_turn TESTS PASSED")

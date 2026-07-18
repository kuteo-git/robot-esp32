"""
Unit tests for the adapter's parsing / prompt / helper logic.

Run:  python -m unittest test_server -v      (uses the built-in unittest,
no extra dependency). Importing server.py needs fastapi/uvicorn installed
(they are in the adapter venv); it has no import-time side effects beyond
creating the FastAPI app and opening the log file, so we point the log at a
throwaway path before importing.
"""
import os
import unittest

os.environ.setdefault("LOG_FILE", "/tmp/claude-adapter-test.log")

import server  # noqa: E402


class IterBalancedJsonObjects(unittest.TestCase):
    def _objs(self, text):
        return [raw for _s, _e, raw in server._iter_balanced_json_objects(text)]

    def test_single_object(self):
        self.assertEqual(self._objs('hi {"a": 1} bye'), ['{"a": 1}'])

    def test_two_objects(self):
        self.assertEqual(self._objs('{"a":1} x {"b":2}'), ['{"a":1}', '{"b":2}'])

    def test_nested_braces(self):
        self.assertEqual(self._objs('{"a": {"b": 1}}'), ['{"a": {"b": 1}}'])

    def test_braces_inside_string_ignored(self):
        # A "}" inside a string value must not close the object early.
        raw = '{"code": "print({\'x\': 1})"}'
        self.assertEqual(self._objs(raw), [raw])

    def test_escaped_quote_inside_string(self):
        raw = '{"s": "a \\" b"}'
        self.assertEqual(self._objs(raw), [raw])

    def test_no_object(self):
        self.assertEqual(self._objs("no json here"), [])


class ExtractToolCall(unittest.TestCase):
    def _extract(self, obj_text):
        import json
        data = json.loads(obj_text, object_pairs_hook=server._dict_keep_pairs)
        return server._extract_tool_call(data)

    def test_canonical(self):
        self.assertEqual(
            self._extract('{"name": "terminal", "input": {"command": "ls"}}'),
            {"name": "terminal", "input": {"command": "ls"}},
        )

    def test_key_aliases(self):
        self.assertEqual(
            self._extract('{"tool_name": "terminal", "tool_input": {"command": "ls"}}'),
            {"name": "terminal", "input": {"command": "ls"}},
        )

    def test_arguments_alias(self):
        self.assertEqual(
            self._extract('{"name": "get_weather", "arguments": {"city": "Oslo"}}'),
            {"name": "get_weather", "input": {"city": "Oslo"}},
        )

    def test_duplicate_name_key(self):
        # Haiku bug: reuses "name" for both the tool and an argument value.
        self.assertEqual(
            self._extract('{"name": "skill_view", "name": "google-workspace"}'),
            {"name": "skill_view", "input": {"name": "google-workspace"}},
        )

    def test_single_item_list_unwrapped(self):
        self.assertEqual(
            self._extract('[{"name": "terminal", "input": {"command": "ls"}}]'),
            {"name": "terminal", "input": {"command": "ls"}},
        )

    def test_top_level_args_folded_into_input(self):
        # No input-shaped field: other top-level pairs become the input.
        self.assertEqual(
            self._extract('{"name": "get_weather", "city": "Oslo", "units": "metric"}'),
            {"name": "get_weather", "input": {"city": "Oslo", "units": "metric"}},
        )

    def test_not_a_tool_call(self):
        self.assertIsNone(self._extract('{"foo": "bar"}'))


class ParseToolCalls(unittest.TestCase):
    TOOLS = {"terminal", "skill_view", "get_weather"}

    def test_canonical_tag(self):
        raw = 'Let me check.\n<tool_call>\n{"name": "terminal", "input": {"command": "ls"}}\n</tool_call>'
        blocks, text = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["name"], "terminal")
        self.assertEqual(blocks[0]["input"], {"command": "ls"})
        self.assertEqual(text, "Let me check.")

    def test_mismatched_tags(self):
        raw = '<tool_call>\n{"name": "terminal", "input": {"command": "ls"}}\n</function_calls>\n\nfake result'
        blocks, text = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["name"], "terminal")
        # Everything from the first tool call onward (incl. hallucinated
        # trailing text) is dropped.
        self.assertEqual(text, "")

    def test_duplicate_key_call(self):
        raw = 'Tao kiểm tra nha.\n<call>\n{"name": "skill_view", "name": "google-workspace"}\n</tool_call>'
        blocks, text = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["name"], "skill_view")
        self.assertEqual(blocks[0]["input"], {"name": "google-workspace"})
        self.assertEqual(text, "Tao kiểm tra nha.")

    def test_array_wrapped_tool_name(self):
        raw = '<function_calls>\n[{"tool_name": "skill_view", "tool_input": {"name": "x"}}]\n</function_calls>'
        blocks, text = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["name"], "skill_view")
        self.assertEqual(blocks[0]["input"], {"name": "x"})
        self.assertEqual(text, "")

    def test_code_with_braces_in_command(self):
        raw = '<tool_call>\n{"name": "terminal", "input": {"command": "python -c \\"print({1:2})\\""}}\n</tool_call>'
        blocks, _ = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["input"]["command"], 'python -c "print({1:2})"')

    def test_unknown_tool_name_ignored(self):
        raw = '{"name": "not_a_real_tool", "input": {}}'
        blocks, text = server._parse_tool_calls(raw, self.TOOLS)
        self.assertEqual(blocks, [])
        self.assertEqual(text, raw)

    def test_plain_text_no_call(self):
        blocks, text = server._parse_tool_calls("just a normal answer", self.TOOLS)
        self.assertEqual(blocks, [])
        self.assertEqual(text, "just a normal answer")


class BuildContentBlocks(unittest.TestCase):
    def test_no_tools_is_plain_text(self):
        blocks, stop = server._build_content_blocks("hello", set())
        self.assertEqual(blocks, [{"type": "text", "text": "hello"}])
        self.assertEqual(stop, "end_turn")

    def test_tool_call_sets_stop_reason(self):
        raw = 'ok\n<tool_call>\n{"name": "terminal", "input": {"command": "ls"}}\n</tool_call>'
        blocks, stop = server._build_content_blocks(raw, {"terminal"})
        self.assertEqual(stop, "tool_use")
        self.assertEqual(blocks[0], {"type": "text", "text": "ok"})
        self.assertEqual(blocks[1]["type"], "tool_use")

    def test_tools_present_but_no_call(self):
        blocks, stop = server._build_content_blocks("no call here", {"terminal"})
        self.assertEqual(stop, "end_turn")
        self.assertEqual(blocks, [{"type": "text", "text": "no call here"}])


class RateLimitNote(unittest.TestCase):
    def test_below_threshold_none(self):
        self.assertIsNone(server._rate_limit_note(
            {"utilization": 0.5, "surpassedThreshold": False, "isUsingOverage": False}
        ))

    def test_surpassed_threshold_warns(self):
        note = server._rate_limit_note(
            {"utilization": 0.97, "surpassedThreshold": True,
             "isUsingOverage": False, "rateLimitType": "five_hour"}
        )
        self.assertIsNotNone(note)
        self.assertIn("97%", note)
        self.assertIn("five_hour", note)

    def test_overage_warns(self):
        note = server._rate_limit_note(
            {"utilization": 1.0, "surpassedThreshold": True,
             "isUsingOverage": True, "rateLimitType": "five_hour"}
        )
        self.assertIn("OVERAGE", note)

    def test_bad_input(self):
        self.assertIsNone(server._rate_limit_note(None))
        self.assertIsNone(server._rate_limit_note({}))


class SystemPromptAndMessages(unittest.TestCase):
    def test_extract_system_text_string(self):
        self.assertEqual(server._extract_system_text("hi"), "hi")

    def test_extract_system_text_blocks(self):
        self.assertEqual(
            server._extract_system_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
            "a\nb",
        )

    def test_build_system_prompt_no_tools_passthrough(self):
        self.assertEqual(server._build_system_prompt("base", []), "base")

    def test_build_system_prompt_with_tools_includes_instructions(self):
        out = server._build_system_prompt("base", [{"name": "terminal", "input_schema": {}}])
        self.assertIn("terminal", out)
        self.assertIn("<tool_call>", out)
        self.assertIn("base", out)

    def test_messages_to_prompt_roundtrip(self):
        out = server._messages_to_prompt([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        self.assertIn("Human: hi", out)
        self.assertIn("Assistant: hello", out)
        self.assertTrue(out.rstrip().endswith("Assistant:"))

    def test_messages_to_prompt_tool_result_block(self):
        out = server._messages_to_prompt([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "output here"}
            ]},
        ])
        self.assertIn("<tool_result id=t1>output here</tool_result>", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)

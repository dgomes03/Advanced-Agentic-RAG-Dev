"""Tool call parser utilities shared across all generators."""

import json
import re


def safe_json_loads(s):
    """Parse JSON, retrying with sanitized control characters if the first attempt fails.

    LLMs sometimes emit literal control characters (\\n, \\t, etc.) inside
    JSON string values, which json.loads rejects with 'Invalid control character'.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        _escape_map = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
        sanitized = re.sub(r'[\x00-\x1f]', lambda m: _escape_map.get(m.group(), ''), s)
        return json.loads(sanitized)


def parse_tool_calls(response_text):
    """Parse tool calls from response text. Handles three formats:
    1. Array format: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
    2. Named format: [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
    3. Simple format: [TOOL_CALLS]name\n{json} or name{json}

    For LRM responses, pass only the text after [TOOL_CALLS] marker
    (with any stray [/THINK] already stripped).

    Returns a list of {"name": ..., "arguments": ...} dicts, or empty list on failure.
    """
    # For standard generator, the full response_text contains [TOOL_CALLS]
    # For LRM, the caller strips the prefix before calling this.
    if "[TOOL_CALLS]" in response_text:
        tc_idx = response_text.find("[TOOL_CALLS]")
        after_tc = response_text[tc_idx + len("[TOOL_CALLS]"):].strip()
    else:
        after_tc = response_text.strip()

    tool_calls = []

    if after_tc.startswith('['):
        # Array format: [{"name": "...", "arguments": {...}}]
        bracket_count = 1
        end_idx = 1
        while end_idx < len(after_tc) and bracket_count > 0:
            if after_tc[end_idx] == '[':
                bracket_count += 1
            elif after_tc[end_idx] == ']':
                bracket_count -= 1
            end_idx += 1
        tool_json = after_tc[:end_idx]
        tool_calls = safe_json_loads(tool_json)

    elif '[ARGS]' in after_tc:
        # Named format: name[ARGS]{json}
        # For standard generator, multiple [TOOL_CALLS]...[ARGS] pairs may exist
        # For LRM, after_tc already has the marker stripped
        tool_pattern = re.compile(r'(?:\[TOOL_CALLS\])?([^\[]+)\[ARGS\]')
        matches = list(tool_pattern.finditer(after_tc))

        for match in matches:
            tool_name = match.group(1).strip()
            args_start = match.end()
            args_part = after_tc[args_start:].strip()

            brace_count = 0
            end_idx = 0
            for i, char in enumerate(args_part):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            args_json = args_part[:end_idx] if end_idx > 0 else "{}"
            tool_args = safe_json_loads(args_json) if args_json else {}
            tool_calls.append({"name": tool_name, "arguments": tool_args})

    elif '{' in after_tc:
        # Simple format: name\n{json} or name{json}
        brace_idx = after_tc.find('{')
        tool_name = after_tc[:brace_idx].strip()
        args_str = after_tc[brace_idx:]

        depth = 0
        end_idx = 0
        for i, c in enumerate(args_str):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        args_json = args_str[:end_idx] if end_idx > 0 else "{}"
        tool_args = safe_json_loads(args_json)

        if tool_name:
            tool_calls.append({"name": tool_name, "arguments": tool_args})
        elif isinstance(tool_args, dict) and "name" in tool_args:
            tool_calls.append(tool_args)

    return tool_calls

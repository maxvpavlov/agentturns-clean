# Token Abstraction in llama-cpp-python

Understanding how llama-cpp-python handles Llama 3.1's special tokens, particularly `<|eom_id|>` and `<|eot_id|>`.

---

## Key Finding

**llama-cpp-python abstracts special tokens away from the API surface** - they are generated internally by the model but are not directly exposed in the response content or logprobs.

---

## What Happens Under the Hood

### 1. Token Generation (Model Level)

When Llama 3.1 generates a response:
```
{"name": "get_time", "parameters": {}}<|eot_id|>
```

The model literally generates:
1. Regular text tokens: `{`, `"`, `name`, `"`, `:`, etc.
2. Special token: `<|eot_id|>` (token ID 128009)

### 2. Token Filtering (llama-cpp-python Level)

llama-cpp-python's `create_chat_completion()`:
- ✅ Returns the text content: `{"name": "get_time", "parameters": {}}`
- ❌ **Filters out** the `<|eot_id|>` token from the response
- ✅ Sets `finish_reason: "stop"` to indicate generation stopped

### 3. What You See in the API Response

```python
response = llm.create_chat_completion(messages=..., tools=...)

response["choices"][0]["message"]["content"]
# Returns: '{"name": "get_time", "parameters": {}}'
# Note: No <|eot_id|> visible

response["choices"][0]["finish_reason"]
# Returns: "stop"
# This IMPLIES <|eot_id|> was generated
```

---

## Evidence from Testing

### Test 1: Logprobs Content
```python
response = llm.create_chat_completion(..., logprobs=True)
logprobs = response["choices"][0]["logprobs"]["content"]

# Last 10 tokens shown:
# '":', ' "', 'get', '_time', '",', ' "', 'parameters', '":', ' {', '}}'
# Notice: No special tokens in the list!
```

The `logprobs["content"]` field only contains visible text tokens, not special tokens.

### Test 2: Raw Completion with Stop Sequences
```python
llm(prompt, stop=['<|eot_id|>', '<|eom_id|>'])
# Generation stops when special token is encountered
# The stop token itself is NOT included in output
```

When generation stops at a stop sequence, it confirms a special token was generated.

### Test 3: finish_reason Field
```python
finish_reason = response["choices"][0]["finish_reason"]

# Possible values:
# - "stop": Model generated <|eot_id|> (End of Turn - done)
# - "length": Max tokens reached
# - Other implementation-specific reasons
```

---

## Special Token Behavior Summary

| Token | Token ID | Generated? | Visible in Content? | How to Detect |
|-------|----------|------------|---------------------|---------------|
| `<|eot_id|>` | 128009 | ✅ Yes | ❌ No | `finish_reason: "stop"` |
| `<|eom_id|>` | 128008 | Depends* | ❌ No | Would need different finish_reason |
| `<|python_tag|>` | 128010 | ❌ No** | ❌ No | N/A for JSON format |
| `<|begin_of_text|>` | 128000 | ✅ Yes | ❌ No | Added automatically |
| `<|start_header_id|>` | 128006 | ✅ Yes | ❌ No | Part of chat template |
| `<|end_header_id|>` | 128007 | ✅ Yes | ❌ No | Part of chat template |

\* `<|eom_id|>` requires `Environment: ipython` in system prompt
\*\* `<|python_tag|>` used for built-in tools, not custom JSON tool calling

---

## Why This Matters for Tool Calling

### Llama 3.1's Tool Calling Design

Llama 3.1 was trained with two modes:

1. **Built-in Tools Mode** (with `Environment: ipython`):
   ```
   <|python_tag|>tool_name.call(arg="value")<|eom_id|>
   ```
   - Uses `<|python_tag|>` marker
   - Ends with `<|eom_id|>` (continue, need tool execution)

2. **Custom Tools Mode** (JSON format):
   ```
   {"name": "tool_name", "parameters": {"arg": "value"}}<|eot_id|>
   ```
   - No `<|python_tag|>` marker
   - Ends with `<|eot_id|>` (done with this response)

### What llama-cpp-python Does

For custom tools (what we use), llama-cpp-python:
1. Adds tool definitions to the prompt via chat template
2. Model generates JSON tool call
3. Model generates `<|eot_id|>` to end the turn
4. Returns JSON in `content`, sets `finish_reason: "stop"`

The special tokens ARE being generated correctly, but they're abstracted away by the library for cleaner API usage.

---

## Practical Implications

### 1. You Can't Directly See Special Tokens

```python
# This won't show special tokens:
response["choices"][0]["message"]["content"]

# This also won't show them:
response["choices"][0]["logprobs"]["content"]
```

### 2. Use finish_reason to Infer Token Type

```python
finish_reason = response["choices"][0]["finish_reason"]

if finish_reason == "stop":
    # Model generated <|eot_id|> - done with turn
    print("Turn completed")
elif finish_reason == "length":
    # Hit max_tokens limit
    print("Generation truncated")
```

### 3. Tool Calls Are Parsed from JSON Content

```python
content = response["choices"][0]["message"]["content"]

try:
    tool_call = json.loads(content)
    if "name" in tool_call:
        # This is a tool call!
        execute_tool(tool_call["name"], tool_call["parameters"])
except json.JSONDecodeError:
    # Regular text response
    pass
```

---

## Debug Mode in Our Implementation

In `8.agent-llama_cpp-tool-calling.py`, we added `--debug-tokens` flag:

```bash
python 8.agent-llama_cpp-tool-calling.py "What time is it?" --debug-tokens
```

Output shows:
```
[DEBUG] Raw Output with Special Tokens Highlighted
{"name": "run_shell_command", "parameters": {"command": "date"}}

Finish reason: stop (implies <|eot_id|> token generated - End of Turn)
```

This confirms:
- The text content is returned without special tokens
- `finish_reason: "stop"` indicates `<|eot_id|>` was generated
- The model is correctly using Llama 3.1's tool calling format

---

## Comparison: Raw vs Abstracted

### Raw Tokenization (What Model Sees)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are helpful<|eot_id|><|start_header_id|>user<|end_header_id|>
Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Hi there!<|eot_id|>
```

### API Response (What Developer Sees)
```python
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hi there!"
    },
    "finish_reason": "stop"
  }]
}
```

All structural tokens are hidden - you only see the content!

---

## Conclusion

**The special tokens ARE working correctly**, but llama-cpp-python provides a clean abstraction:

✅ **Pros**:
- Cleaner API - no special token noise in responses
- Easier to work with - just parse JSON or read text
- Consistent with OpenAI API conventions

❌ **Cons**:
- Less transparency about what tokens are generated
- Can't easily distinguish `<|eom_id|>` vs `<|eot_id|>` without checking finish_reason
- Debugging token-level behavior requires understanding the abstraction

For tool calling, this abstraction works well - you get JSON tool calls without worrying about special tokens, and `finish_reason` tells you when generation is complete.

---

## References

- [LLAMA_3.1_SPECIAL_TOKENS_REFERENCE.md](./LLAMA_3.1_SPECIAL_TOKENS_REFERENCE.md) - Complete special tokens reference
- [llama-cpp-python API](https://github.com/abetlen/llama-cpp-python) - Official library
- [Llama 3.1 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/) - Official Meta documentation

---

## Source Code Verification

We verified this behavior by examining the actual source code:

### llama-cpp-python (Python Bindings)

From `llama_cpp/llama.py`, the finish_reason is set to "stop" when:

```python
if llama_cpp.llama_token_is_eog(self._model.vocab, token):
    text = self.detokenize(completion_tokens, prev_tokens=prompt_tokens)
    finish_reason = "stop"
    break
```

**Source**: [llama_cpp/llama.py](https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py)

### llama.cpp (Core Library)

From `include/llama.h`, the EOG check function:

```c
// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);
```

The implementation in `src/llama-vocab.cpp` uses a set to track EOG tokens:

```cpp
std::set<llama_token> special_eog_ids;
```

This set includes tokens like:
- `<|eot_id|>` (token ID 128009) - End of Turn
- `<|eom_id|>` (token ID 128008) - End of Message (in certain contexts)
- `<|end_of_text|>` (token ID 128001) - End of Text
- Other model-specific EOG tokens

**Sources**:
- [include/llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
- [src/llama-vocab.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-vocab.cpp)

### Confirmed Flow

1. **Model generates token** → Llama 3.1 outputs `<|eot_id|>` (token 128009)
2. **llama.cpp checks EOG** → `llama_vocab_is_eog()` returns `true` for token 128009
3. **Python binding sets finish_reason** → Sets `finish_reason = "stop"`
4. **API response** → Token filtered out, only text content returned

This confirms that `finish_reason: "stop"` **definitively indicates** an EOG token (like `<|eot_id|>`) was generated.

---

**Key Takeaway**: When you see `finish_reason: "stop"` after a tool call, you can be **factually certain** that Llama 3.1 generated an EOG token (typically `<|eot_id|>` token ID 128009) under the hood, even though you don't see it directly in the API response. This has been verified through source code inspection of both llama-cpp-python and llama.cpp.

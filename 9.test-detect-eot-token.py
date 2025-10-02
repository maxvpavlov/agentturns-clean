"""
Test script to detect <|eot_id|> token (ID 128009) after tool calls.
This experiments with various llama-cpp-python APIs to see if we can
detect the End of Turn token that should follow tool call JSON.
"""

from llama_cpp import Llama
import json

print("Loading model with verbose=True...")
llm = Llama(
    model_path="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=True,  # Enable verbose logging
    logits_all=True  # Enable logits for all tokens
)
print("\nModel loaded!\n")

# Tool schema
tools = [{
    'type': 'function',
    'function': {
        'name': 'run_shell_command',
        'description': 'Execute a shell command',
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {'type': 'string', 'description': 'The command to execute'}
            },
            'required': ['command']
        }
    }
}]

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant with access to tools.'},
    {'role': 'user', 'content': 'Run the date command'}
]

print("=" * 80)
print("TEST 1: Standard create_chat_completion with logprobs")
print("=" * 80)

try:
    response = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice='auto',
        max_tokens=256,
        logprobs=True  # Request logprobs
    )

    msg = response['choices'][0]['message']
    print(f"\n=== MESSAGE STRUCTURE ===")
    print(f"Message keys: {list(msg.keys())}")
    print(f"Content: {msg.get('content')}")
    print(f"Content type: {type(msg.get('content'))}")

    # Explicitly check for tool_calls array
    tool_calls = msg.get('tool_calls')
    print(f"\n=== TOOL_CALLS FIELD ===")
    print(f"tool_calls field exists: {tool_calls is not None}")
    print(f"tool_calls value: {tool_calls}")
    print(f"tool_calls type: {type(tool_calls)}")
    if tool_calls:
        print(f"tool_calls length: {len(tool_calls)}")
        print(f"tool_calls content: {json.dumps(tool_calls, indent=2)}")
    else:
        print("tool_calls is None or empty")

    print(f"\n=== RESPONSE STRUCTURE ===")
    print(f"Finish reason: {response['choices'][0]['finish_reason']}")
    print(f"Response choice keys: {list(response['choices'][0].keys())}")

    # Check if logprobs are available
    logprobs = response['choices'][0].get('logprobs')
    if logprobs:
        print(f"\n=== LOGPROBS ===")
        print(f"Logprobs structure keys: {logprobs.keys()}")
        content_logprobs = logprobs.get('content', [])
        print(f"Number of tokens in logprobs: {len(content_logprobs)}")

        if content_logprobs:
            print("\nLast 5 tokens:")
            for i, token_info in enumerate(content_logprobs[-5:]):
                token = token_info.get('token', 'N/A')
                # Try to get token ID if available
                print(f"  Token {i}: '{token}'")
    else:
        print("\n=== NO LOGPROBS ===")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("TEST 2: Low-level __call__ method to get raw tokens")
print("=" * 80)

# Create prompt manually using chat template
try:
    # Get the formatted prompt
    prompt = llm.create_chat_completion_prompt(
        messages=messages,
        tools=tools
    )
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Prompt preview (last 200 chars): ...{prompt[-200:]}")

    # Generate with raw __call__
    print("\nGenerating with raw __call__ to capture tokens...")
    output = llm(
        prompt,
        max_tokens=256,
        stop=["<|eot_id|>", "<|eom_id|>"],  # Stop at special tokens
        echo=False
    )

    print(f"\nRaw output keys: {output.keys()}")
    print(f"Choices: {len(output['choices'])}")

    text = output['choices'][0]['text']
    finish_reason = output['choices'][0]['finish_reason']

    print(f"\nGenerated text: {text}")
    print(f"Finish reason: {finish_reason}")
    print(f"Text ends with: '{text[-20:]}'")

    # Check if it stopped at a special token
    if finish_reason == "stop":
        print("\n✓ Generation stopped (likely hit <|eot_id|> or <|eom_id|>)")

except AttributeError:
    print("create_chat_completion_prompt not available, trying alternative...")

    # Alternative: use raw completion
    output = llm(
        "Test prompt",
        max_tokens=50,
        stop=["<|eot_id|>"]
    )
    print(f"Text: {output['choices'][0]['text']}")
    print(f"Finish reason: {output['choices'][0]['finish_reason']}")

print("\n" + "=" * 80)
print("TEST 3: Check token vocabulary for <|eot_id|>")
print("=" * 80)

# Try to verify token ID
EOT_TOKEN_ID = 128009
EOM_TOKEN_ID = 128008

print(f"\n<|eot_id|> token ID should be: {EOT_TOKEN_ID}")
print(f"<|eom_id|> token ID should be: {EOM_TOKEN_ID}")

# Try to detokenize these IDs to confirm
try:
    eot_text = llm.detokenize([EOT_TOKEN_ID])
    print(f"Token {EOT_TOKEN_ID} detokenizes to: {repr(eot_text)}")

    eom_text = llm.detokenize([EOM_TOKEN_ID])
    print(f"Token {EOM_TOKEN_ID} detokenizes to: {repr(eom_text)}")
except Exception as e:
    print(f"Error detokenizing: {e}")

print("\n" + "=" * 80)
print("TEST 4: Generate completion and tokenize result")
print("=" * 80)

try:
    # Generate again
    response = llm.create_chat_completion(
        messages=messages,
        tools=tools,
        tool_choice='auto',
        max_tokens=256
    )

    content = response['choices'][0]['message']['content']
    print(f"\nGenerated content: {content}")

    # Try to tokenize the response
    tokens = llm.tokenize(content.encode('utf-8'))
    print(f"\nTokenized into {len(tokens)} tokens")
    print(f"Token IDs: {tokens}")

    # Check if EOT token is present
    if EOT_TOKEN_ID in tokens:
        print(f"\n✓ Found <|eot_id|> (token {EOT_TOKEN_ID}) in tokenized output!")
    else:
        print(f"\n✗ <|eot_id|> (token {EOT_TOKEN_ID}) NOT found in tokenized output")

    # Try adding the token manually and see what happens
    tokens_with_eot = tokens + [EOT_TOKEN_ID]
    reconstructed = llm.detokenize(tokens_with_eot)
    print(f"\nReconstructed with <|eot_id|>: {repr(reconstructed)}")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If <|eot_id|> is being filtered:
- It won't appear in content field
- It won't appear in logprobs
- finish_reason="stop" implies it was generated
- We cannot explicitly detect it via llama-cpp-python API

This confirms the token abstraction documented in
TOKEN_ABSTRACTION_IN_LLAMA_CPP_PYTHON.md
""")

# Agent Turns - ReAct Pattern Demos

A collection of Python agents demonstrating the ReAct (Reasoning and Acting) pattern with various approaches to tool calling.

## Scripts Overview

1. **1.agent.py** - Basic ReAct agent with Ollama
2. **2.agent-streaming.py** - Streaming response version
3. **3.agent-streaming-styled.py** - Styled output with rich library
4. **4.agent-streaming-styled-context.py** - Context-aware agent
5. **5.agent-streaming-styled-final-check.py** - With final answer verification
6. **6.agent-streaming-styled-final-check-safeguards.py** - With command safety checks
7. **7.agent-uses-model-tool-calling.py** - Ollama with native tool calling
8. **8.agent-llama_cpp-tool-calling.py** - llama-cpp-python with native tool calling
9. **9.test-detect-eot-token.py** - Token detection experiments

## Setup for 8.agent-llama_cpp-tool-calling.py

This script uses llama-cpp-python with a local GGUF model file.

### Download the Model

The model file is not included in this repository due to its large size (~4.6GB).

**Download from HuggingFace:**
```bash
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

Or download manually from: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

Place the downloaded file in the project root directory.

### Install Dependencies

```bash
pip install llama-cpp-python rich
```

### Run the Agent

```bash
python 8.agent-llama_cpp-tool-calling.py "What is the current time?"
```

## Documentation

- **LLAMA_3.1_SPECIAL_TOKENS_REFERENCE.md** - Comprehensive guide to Llama 3.1 special tokens
- **TOKEN_ABSTRACTION_IN_LLAMA_CPP_PYTHON.md** - How llama-cpp-python abstracts special tokens

## Features

- **Three-phase ReAct pattern**: Planning → Execution → Synthesis
- **Native tool calling**: Uses Llama 3.1's JSON tool calling format
- **Step-by-step execution**: Executes plan items one at a time
- **Rich console output**: Styled terminal output with syntax highlighting

## How It Works

1. **Planning Phase**: Model creates a numbered plan
2. **Execution Phase**: Model executes tools one by one, guided by the plan
3. **Synthesis Phase**: Model generates final answer from gathered information

The agent uses Llama 3.1's native JSON tool calling format:
```json
{"name": "run_shell_command", "parameters": {"command": "date"}}
```

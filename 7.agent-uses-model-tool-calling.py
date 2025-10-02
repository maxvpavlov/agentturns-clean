import ollama
import subprocess
import sys
import json
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule

console = Console()

# Define available tools
def run_shell_command(command: str):
    """
    Execute a shell command and return its output.

    Args:
        command: The shell command to execute.
    
    Returns:
        str: The output of the command, or an error message.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "action execution failed with timeout"
    except Exception as e:
        return f"Exception: {str(e)}"

# Map tool names to their actual functions for execution
TOOLS = {
    "run_shell_command": run_shell_command
}

SYSTEM_PROMPT_PLANNING = """You are a helpful AI assistant.
Create a concise, step-by-step plan to answer the user's question.
Output ONLY the plan as a numbered list. Do not execute any actions yet."""

SYSTEM_PROMPT_ACTING = """You are a helpful AI assistant that uses tools to answer questions.

CRITICAL: You MUST use the provided tools by making actual function calls. Do NOT write JSON descriptions of tool calls - the system will automatically format them for you.

Follow this pattern:
1. Make a tool call to gather information
2. Wait for the result
3. Reflect on the result
4. Continue with more tool calls if needed
5. Provide final answer when you have enough information"""

def run_agent(query, max_steps=10):
    """Main agent loop using ReAct pattern with explicit planning phase."""

    # Define tools once for reuse
    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'run_shell_command',
                'description': 'Execute a shell command and return its output.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'command': {
                            'type': 'string',
                            'description': 'The shell command to execute.',
                        },
                    },
                    'required': ['command'],
                },
            },
        },
    ]

    # PHASE 1: PLANNING (no tools available)
    console.print(Rule("[bold blue]Step 1: Planning"))

    planning_history = [
        {"role": "system", "content": SYSTEM_PROMPT_PLANNING},
        {"role": "user", "content": query}
    ]

    with console.status("[bold green]Creating plan..."):
        planning_response = ollama.chat(
            model='llama3.1:8b',
            messages=planning_history
        )

    plan = planning_response['message']['content']
    console.print(Panel(plan, title="Plan", border_style="cyan"))

    # PHASE 2: EXECUTION (with tools)
    # Start fresh history for execution with the acting system prompt
    # Add few-shot examples to guide proper tool calling
    execution_history = [
        {"role": "system", "content": SYSTEM_PROMPT_ACTING},
        # Few-shot example 1: Show correct tool usage
        {"role": "user", "content": "What is the current directory?"},
        {
            "role": "assistant",
            "content": "I'll use the run_shell_command tool to check the current directory.",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_shell_command",
                        "arguments": {"command": "pwd"}
                    }
                }
            ]
        },
        {"role": "tool", "content": "/home/user/projects"},
        {"role": "assistant", "content": "The current directory is /home/user/projects"},
        # Now the actual query
        {"role": "user", "content": f"Original question: {query}\n\nPlan to follow:\n{plan}\n\nNow execute this plan using available tools."}
    ]

    for step in range(2, max_steps + 1):
        console.print(Rule(f"[bold blue]Step {step}"))

        # Call model with tools available
        with console.status("[bold green]Thinking..."):
            response = ollama.chat(
                model='llama3.1:8b',
                messages=execution_history,
                tools=tools,
            )

        assistant_message = response.get('message', {})
        execution_history.append(assistant_message)

        # Display reasoning if present
        if assistant_message.get('content'):
            console.print(Panel(assistant_message['content'], title="Thought", border_style="yellow"))

        # Check for tool calls
        tool_calls = assistant_message.get('tool_calls')
        if tool_calls:
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                function_args = tool_call['function']['arguments']

                if function_name in TOOLS:
                    command_to_run = function_args.get('command')
                    syntax = Syntax(command_to_run, "bash", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, title=f"Action: {function_name}", border_style="dark_orange"))

                    observation = TOOLS[function_name](command_to_run)
                    console.print(Panel(observation, title="Observation", border_style="green"))

                    execution_history.append({'role': 'tool', 'content': observation})
                else:
                    error_msg = f"Error: Unknown tool: {function_name}"
                    console.print(Panel(error_msg, title="Error", border_style="bold red"))
                    execution_history.append({'role': 'tool', 'content': error_msg})
        else:
            # No tool calls - model has reached final answer
            final_answer = assistant_message.get('content', "Task completed.")
            console.print(Panel(final_answer, title="Final Answer", border_style="sky_blue1"))
            return final_answer

    return "Max steps reached without final answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = run_agent(query)
        console.print(Rule("[bold magenta]Result"))
        console.print(result)
    else:
        print("Usage: python 7.agent-uses-model-tool-calling.py 'your query'")

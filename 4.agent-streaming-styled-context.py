import ollama
import subprocess
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule
from rich.progress import Progress

console = Console()

# Define available tools
def run_shell_command(command):
    """Execute a shell command and return its output."""
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

TOOLS = {
    "run_shell_command": run_shell_command
}

# System prompt for ReAct pattern
SYSTEM_PROMPT = """
You are an AI agent that solves tasks iteratively using Reasoning and Acting (ReAct).
Your response MUST be in the following format for each step:
|Thought:| [Your reasoning process here]
ONE OF THE FOLLOWING ELEMENTS:
|Action:| [tool_name: argument] 
|Final Answer:| [your final answer]

Explanaition about these possible output sections:
- |Thought:| Reason step-by-step about what to do next.
- |Action:| If needed, call a tool in the format 'tool_name: argument'. Available tools: run_shell_command (e.g., 'run_shell_command: ls -l /home').
- If you have the final answer, output '|Final Answer:| [your answer]'.

If you need to run several commands, don't suggest more then one command in |Action:| section, you will get another turn to suggest subsequent tools to call. DO NOT RESPOND WITH MULTIPLE |Action:| Sections in one response
You can't have multiple instances of |Thought:| in one response, make all the toughts appear in on |Thought:| element.
Make sure to never provide both |Action:| and |Final Answer:| elements in one response.
Even if action to perform is none, do not add |Action:| section to the respose.
Do not repeat actions unnecessarily. Stop when the query is solved.
Do not try to install additional software on the computer where you are being executed.
"""

def get_llm_response(history):
    """Query Ollama with conversation history and stream the response."""
    stream = ollama.chat(model='llama3.1:8b', messages=history, stream=True)
    for chunk in stream:
        yield chunk['message']['content']

def parse_output(output):
    """Parse LLM output for Thought, Action, or Final Answer."""
    thought_match = re.search(r"\|Thought:\|(.*?)(?:\|Action:\||\|Final Answer:\||$)", output, re.DOTALL)
    action_match = re.search(r"\|Action:\|(.*?)(?:\|Thought:\||\|Final Answer:\||$)", output, re.DOTALL)
    final_answer_match = re.search(r"\|Final Answer:\|(.*?)(?:\|Thought:\||\|Action:\||$)", output, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    action_str = action_match.group(1).strip() if action_match else ""
    final_answer = final_answer_match.group(1).strip() if final_answer_match else ""

    action = None
    if action_str and ':' in action_str:
        tool, arg = action_str.split(':', 1)
        action = (tool.strip(), arg.strip())

    return thought, action, final_answer

def get_context_window_size(model_name='llama3.1:8b'):
    """Get the context window size of the model."""
    try:
        result = subprocess.run(f"ollama show {model_name}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'context length' in line:
                    return int(line.split()[-1])
    except Exception as e:
        console.print(Panel(f"Could not get context window size: {e}", title="Error", border_style="bold red"))
        return None

def run_agent(query, max_steps=10):
    """Main agent loop implementing ReAct."""
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    context_window_size = get_context_window_size()

    for step in range(max_steps):
        console.print(Rule(f"[bold blue]Step {step + 1}"))
        
        # Calculate context utilization
        if context_window_size:
            total_chars = sum(len(message['content']) for message in history)
            estimated_tokens = int(total_chars / 4) # Rough estimation
            percentage = (estimated_tokens / context_window_size) * 100
            console.print(f"[cyan]Context Utilization: {estimated_tokens} / {context_window_size} tokens ({percentage:.2f}%)")
            with Progress() as progress:
                task = progress.add_task("", total=context_window_size)
                progress.update(task, advance=estimated_tokens)

        # Reason: Get LLM response
        full_response = ""
        with console.status("[bold green]Thinking..."):
            for chunk in get_llm_response(history):
                full_response += chunk
        
        console.print(Panel(full_response, title="LLM Output", border_style="green", expand=False))

        # Parse
        thought, action, final_answer = parse_output(full_response)
        
        if thought:
            console.print(Panel(thought, title="Thought", border_style="yellow", expand=False))
        
        if final_answer:
            console.print(Panel(final_answer, title="Final Answer", border_style="sky_blue1", expand=False))
            return final_answer, history
        
        if action:
            tool_name, arg = action
            if tool_name in TOOLS:
                syntax = Syntax(arg, "bash", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"Action: {tool_name}", border_style="dark_orange", expand=False))
                observation = TOOLS[tool_name](arg)
                console.print(Panel(observation, title="Observation", border_style="green", expand=False))
                # Add to history
                history.append({"role": "assistant", "content": full_response})
                history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                console.print(Panel(f"Unknown tool: {tool_name}", title="Error", border_style="bold red", expand=False))
        else:
            # No action or final, continue
            history.append({"role": "assistant", "content": full_response})
    
    return "Max steps reached without final answer.", history

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result, history = run_agent(query)
        console.print(Rule("[bold magenta]Result"))
        console.print(result)

        context_window_size = get_context_window_size()
        if context_window_size:
            total_chars = sum(len(message['content']) for message in history)
            estimated_tokens = int(total_chars / 4) # Rough estimation
            percentage = (estimated_tokens / context_window_size) * 100
            console.print(Rule("[bold cyan]Final Context Utilization"))
            console.print(f"[cyan]{estimated_tokens} / {context_window_size} tokens ({percentage:.2f}%)")
            with Progress() as progress:
                task = progress.add_task("", total=context_window_size)
                progress.update(task, advance=estimated_tokens)

    else:
        print("Usage: python agent-streaming-styled-context.py 'your query'")
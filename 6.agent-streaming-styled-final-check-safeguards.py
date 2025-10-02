import ollama
import subprocess
import sys
import re
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule

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
SYSTEM_PROMPT = '''
You are an AI agent that solves tasks iteratively using Reasoning and Acting (ReAct).
Your response MUST be in the following format for each step:
|Thought:| [Your reasoning process here]
ONE OF THE FOLLOWING ELEMENTS:
|Action:| [tool_name: argument]
OR 
|Final Answer:| [your final answer]

Explanaition about these possible output sections:
- |Thought:| Reason step-by-step about what to do next.
- |Action:| If needed, call a tool in the format 'tool_name: argument'. Available tools: run_shell_command (e.g., 'run_shell_command: ls -l /home').
- If you have the final answer, output '|Final Answer:| [your answer]'.

You can't have multiple instances of |Thought:| in one response, make all the toughts appear in |Thought:| element.
Make sure to never provide both |Action:| and |Final Answer:| elements in one response.
Even if action to perform is none, do not add |Action:| section to the respose.
Do not repeat actions unnecessarily. Stop when the query is solved.
Do not try to install additional software on the computer where you are being executed.
'''

def get_llm_response(history):
    """Query Ollama with conversation history and stream the response."""
    stream = ollama.chat(model='gemma3:12b', messages=history, stream=True) 
    for chunk in stream: 
        yield chunk['message']['content']

import re

def parse_output(output):
    """Parse LLM output for Thought, Action, or Final Answer."""
    thought_match = re.search(r"\|Thought:\|(.*?)\|?\s*(?:\|Action:|\|Final Answer:|$)", output, re.DOTALL)
    action_match = re.search(r"\|Action:\|(.*?)\|?\s*(?:\|Thought:|\|Final Answer:|$)", output, re.DOTALL)
    final_answer_match = re.search(r"\|Final Answer:\|(.*?)\|?\s*(?:\|Thought:|\|Action:|$)", output, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    action_str = action_match.group(1).strip() if action_match else ""
    final_answer = final_answer_match.group(1).strip() if final_answer_match else ""

    action = None
    if action_str and ':' in action_str:
        tool, arg = action_str.split(':', 1)
        action = (tool.strip(), arg.strip())

    return thought, action, final_answer

def is_command_safe(command: str) -> bool:
    """Check with the LLM if a command is safe to execute."""
    prompt = f"You have suggested to execute the following command as part of resolving user query: {command}. Is it possible that the command alters user system in an irreversible manner resulting in data loss or system instability. Please answer POSSIBLE or NOT POSSIBLE."
    
    safeguard_history = [{"role": "user", "content": prompt}]
    
    response = ""
    with console.status("[bold yellow]Verifying command safety..."):
        for chunk in get_llm_response(safeguard_history):
            response += chunk
            
    console.print(Panel(response, title="Safety Check Response", border_style="yellow", expand=False))

    clean_response = response.strip().upper()
    if clean_response.startswith("NOT POSSIBLE"):
        return True
    else:
        return False

from rich.rule import Rule

def run_agent(query, max_steps=10):
    """Main agent loop implementing ReAct."""
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    for step in range(max_steps):
        console.print(Rule(f"[bold blue]Step {step + 1}"))
        
        # Reason: Get LLM response
        full_response = ""
        with console.status("[bold green]Thinking..."):
            for chunk in get_llm_response(history):
                full_response += chunk

        # Check for malformed response containing both Action and Final Answer
        if "|Action:|" in full_response and "|Final Answer:|" in full_response:
            console.print(Panel(full_response, title="[bold red]Malformed LLM Output (Retrying)", border_style="red"))
            continue
        
        console.print(Panel(full_response, title="LLM Output", border_style="green", expand=False))

        # Parse
        thought, action, final_answer = parse_output(full_response)
        
        if thought:
            console.print(Panel(thought, title="Thought", border_style="yellow", expand=False))
        
        if final_answer:
            console.print(Panel(final_answer, title="Final Answer", border_style="sky_blue1", expand=False))
            
            # Final verification step
            console.print(Rule("[bold yellow]Final Check"))
            
            steps_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            
            verification_prompt = f"""Original ask was: {query}
Final answer is: {final_answer}
Steps are:
{steps_string}

Is this a good answer given the request? If you have better answer please respond with |Better Answer:| element."""
            
            verification_history = [{"role": "user", "content": verification_prompt}]
            verification_response = ""
            with console.status("[bold green]Verifying final answer..."):
                for chunk in get_llm_response(verification_history):
                    verification_response += chunk
            
            console.print(Panel(verification_response, title="Verification Result", border_style="purple", expand=False))
            
            # Check for a better answer
            better_answer_match = re.search(r"\|Better Answer:\|(.*?)$", verification_response, re.DOTALL)
            if better_answer_match:
                better_answer = better_answer_match.group(1).strip()
                console.print(Panel(better_answer, title="Better Answer", border_style="bold green", expand=False))
                return better_answer
            
            return final_answer
        
        if action:
            tool_name, arg = action
            if tool_name in TOOLS:
                # Safeguard check for shell commands
                if tool_name == "run_shell_command":
                    if not is_command_safe(arg):
                        observation = f"Error: Command '{arg}' was blocked by the safety guard as potentially harmful. The command was not executed."
                        console.print(Panel(observation, title="Safety Alert", border_style="bold red", expand=False))
                        history.append({"role": "assistant", "content": full_response})
                        history.append({"role": "user", "content": f"Observation: {observation}"})
                        continue # Skip to the next agent step
                
                # If we are here, the command is safe to execute
                syntax = Syntax(arg, "bash", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title=f"Action: {tool_name}", border_style="dark_orange", expand=False))
                observation = TOOLS[tool_name](arg)
                console.print(Panel(observation, title="Observation", border_style="green", expand=False))
                # Add to history
                history.append({"role": "assistant", "content": full_response})
                history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # Handle unknown tool
                observation = f"Error: Unknown tool: {tool_name}"
                console.print(Panel(observation, title="Error", border_style="bold red", expand=False))
                history.append({"role": "assistant", "content": full_response})
                history.append({"role": "user", "content": f"Observation: {observation}"})
        else:
            # No action or final, continue
            history.append({"role": "assistant", "content": full_response})
    
    return "Max steps reached without final answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = run_agent(query)
        console.print(Rule("[bold magenta]Result"))
        console.print(result)
    else:
        print("Usage: python agent-streaming-styled.py 'your query'")

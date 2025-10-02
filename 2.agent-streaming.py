import ollama
import subprocess
import sys
import re

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
OR 
|Final Answer:| [your final answer]

Explanaition about these possible output sections:
- |Thought:| Reason step-by-step about what to do next.
- |Action:| If needed, call a tool in the format 'tool_name: argument'. Available tools: run_shell_command (e.g., 'run_shell_command: ls -l /home').
- If you have the final answer, output '|Final Answer:| [your answer]'.

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

import re

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

def run_agent(query, max_steps=10):
    """Main agent loop implementing ReAct."""
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]
    
    for step in range(max_steps):
        print(f"\nStep {step + 1}:")
        
        # Reason: Get LLM response
        full_response = ""
        print("LLM Output:", end="", flush=True)
        for chunk in get_llm_response(history):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()

        # Parse
        thought, action, final_answer = parse_output(full_response)
        
        if thought:
            print("Thought:", thought)
        
        if final_answer:
            print("Final Answer:", final_answer)
            return final_answer
        
        if action:
            tool_name, arg = action
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](arg)
                print("Action:", action)
                print("Observation:", observation)
                # Add to history
                history.append({"role": "assistant", "content": full_response})
                history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print("Unknown tool:", tool_name)
        else:
            # No action or final, continue
            history.append({"role": "assistant", "content": full_response})
    
    return "Max steps reached without final answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = run_agent(query)
        print("\nResult:", result)
    else:
        print("Usage: python agent-streaming.py 'your query'")

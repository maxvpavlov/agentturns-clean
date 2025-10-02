import ollama
import subprocess
import sys

# Define available tools
def run_shell_command(command):
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

TOOLS = {
    "run_shell_command": run_shell_command
}

# System prompt for ReAct pattern
SYSTEM_PROMPT = """
You are an AI agent that solves tasks iteratively using Reasoning and Acting (ReAct).
For each step:
- Thought: Reason step-by-step about what to do next.
- Action: If needed, call a tool in the format 'tool_name: argument'. Available tools: run_shell_command (e.g., 'run_shell_command: ls -l /home').
- If you have the final answer, output 'Final Answer: [your answer]'.

Do not repeat actions unnecessarily. Stop when the query is solved.
"""

def get_llm_response(history):
    """Query Ollama with conversation history."""
    response = ollama.chat(model='llama3.1:8b', messages=history)  # Change model if needed
    return response['message']['content']

def parse_output(output):
    """Parse LLM output for Thought, Action, or Final Answer."""
    lines = output.strip().split('\n')
    thought = ""
    action = None
    final_answer = None

    for i, line in enumerate(lines):
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action_str = line[len("Action:"):].strip()
            if ':' in action_str:
                tool, arg = action_str.split(':', 1)
                action = (tool.strip(), arg.strip())
        elif line.startswith("Final Answer:"):
            # Capture everything from "Final Answer:" onwards (including subsequent lines)
            final_answer = '\n'.join([line[len("Final Answer:"):].strip()] + lines[i+1:])
            break  # Stop processing once we find Final Answer

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
        output = get_llm_response(history)
        print("LLM Output:", output)
        
        # Parse
        thought, action, final_answer = parse_output(output)
        
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
                history.append({"role": "assistant", "content": output})
                history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print("Unknown tool:", tool_name)
        else:
            # No action or final, continue
            history.append({"role": "assistant", "content": output})
    
    return "Max steps reached without final answer."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        result = run_agent(query)
        print("\nResult:", result)
    else:
        print("Usage: python agent.py 'your query'")

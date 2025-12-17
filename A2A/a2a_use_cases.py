from python_a2a import A2AServer, agent, skill, run_server, TaskStatus, TaskState

# --- 1. DEFINE THE AGENT ---
@agent(
    name="EchoBot",
    description="I repeat back what you say.",
    version="0.1.0"
)
class EchoAgent(A2AServer):

    # --- 2. DEFINE THE SKILL ---
    # Note: No 'input_schema' needed; it learns from 'text: str'
    @skill(
        name="Echo",
        description="I repeat back what you say."
    )
    def echo(self, text: str):
        print(f"🤖 Agent received: {text}")
        return f"Echo: {text}"

    # --- 3. HANDLE THE TASK ---
    def handle_task(self, task):
        # Safely extract text from the incoming message
        input_content = task.message.content
        if isinstance(input_content, dict):
            input_text = input_content.get("text", "")
        else:
            input_text = str(input_content)

        # Execute the skill logic
        result = self.echo(input_text)

        # format the output
        task.artifacts = [{
            "parts": [{"type": "text", "text": result}]
        }]

        # Mark as done
        task.status = TaskStatus(state=TaskState.COMPLETED)
        return task

# --- 4. RUN THE SERVER ---
if __name__ == "__main__":
    print("🚀 Starting EchoBot on port 8000...")

    # FIX IS HERE: We MUST tell the agent its own URL address
    my_agent = EchoAgent(url="http://localhost:8000")

    run_server(my_agent, port=8000)
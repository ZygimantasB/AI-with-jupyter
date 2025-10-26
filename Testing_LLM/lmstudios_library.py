from lmstudio.client import Client
import os

try:
    client = Client()
except Exception:
    print("Error: Failed to connect. Is the LM Studio *application* running?")
    exit()

print("Connecting to LM Studio app...")

try:
    model = client.models.get_loaded()

    if not model:
        print("Error: No model is loaded in the LM Studio app.")
        print("Please go to the 'Local Server' tab, load a model, and try again.")
        exit()

    print(f"Using model: {model['id']}")

    response = model.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what a 'virtual environment' is in Python."}
        ],
        temperature=0.7,
    )

    print("\nModel Response:")
    print(response['choices'][0]['message']['content'])

except Exception as e:
    print(f"An error occurred: {e}")
    print("Tip: Make sure you have clicked 'Start Server' in the LM Studio app.")

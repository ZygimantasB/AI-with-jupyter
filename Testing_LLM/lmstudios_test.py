import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:1234/v1",
)

print("Connecting to local model...")

try:
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what a 'virtual environment' is in Python."}
        ],
        temperature=0.7,
    )

    print("\nModel Response:")
    print(completion.choices[0].message.content)

except openai.RateLimitError:
    print("Error: Make sure a model is loaded in LM Studio and the server is running.")
except openai.APIConnectionError:
    print("Error: Failed to connect. Is the LM Studio server running on http://127.0.0.1:1234?")
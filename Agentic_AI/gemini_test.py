import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")

genai.configure(api_key=API_KEY)

# Create model and generate content
model = genai.GenerativeModel("gemini-2.0-flash-lite")

response = model.generate_content("Explain how AI works in a few words")

print(response.text)
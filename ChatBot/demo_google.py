
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
results=model.invoke("Whats the capital of India?")

print(results.content[0]["text"])


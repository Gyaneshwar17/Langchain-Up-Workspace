from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
llm=OpenAI(model_name='gpt-3.5-turbo')
results=llm.invoke("Whats the capital of India?")

print(results)

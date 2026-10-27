from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

messages=[
    SystemMessage(content='You are a expert in F1'),
    HumanMessage(content='Tell me when did Max Verstappen won his 1st title and who was this compitator')
]

results=chat.invoke(messages)

messages.append(AIMessage(content=results.content[0]['text']))
print(messages)


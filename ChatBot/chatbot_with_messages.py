from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=500)

history=[
    SystemMessage(content="You are a expert in F1.")
]

while True:
    user=input('You: ')
    history.append(HumanMessage(content=user))
    if user=='exit':
        break
    result=chat.invoke(history)
    history.append(AIMessage(content=result.content[0]['text'])) 
    print("Gemini: ",result.content[0]['text'])


print(f"Whole Chat History{history}")
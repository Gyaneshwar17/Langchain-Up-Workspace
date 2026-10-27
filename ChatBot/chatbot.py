from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

history=[]

while True:
    user=input('You: ')
    history.append(user)
    if user=='exit':
        break
    result=chat.invoke(history)
    history.append(result.content[0]['text']) 
    print("Gemini: ",result.content[0]['text'])


print(f"Whole Chat History{history}")
'''
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from typing import TypedDict

load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=300)
#schema 
class review(TypedDict):
    summary:str
    sentiment:str

strut_model=chat.with_structured_output(review)

result=strut_model.invoke("""
I recently bought the Sony WH-1000XM5 wireless headphones, and overall they are incredible. 
The active noise cancellation is top-tier, easily drowning out office chatter and airplane engines. 
Battery life is solid at around 30 hours, and they are very lightweight for long listening sessions. 
However, the design is no longer foldable like the previous XM4 version, making them bulkier to pack in a backpack. 
Also, the touch controls can occasionally be finicky in cold weather. But for sound quality and comfort, they are hard to beat.
""")

print(type(result),result)
print(result['summary']) 
print(result['sentiment'])
'''

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from typing import TypedDict,Annotated

load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=300)
#schema 
class review(TypedDict):
    summary:Annotated[str,"A brief summary of the review"]
    sentiment:str

strut_model=chat.with_structured_output(review)

result=strut_model.invoke("""
I recently bought the Sony WH-1000XM5 wireless headphones, and overall they are incredible. 
The active noise cancellation is top-tier, easily drowning out office chatter and airplane engines. 
Battery life is solid at around 30 hours, and they are very lightweight for long listening sessions. 
However, the design is no longer foldable like the previous XM4 version, making them bulkier to pack in a backpack. 
Also, the touch controls can occasionally be finicky in cold weather. But for sound quality and comfort, they are hard to beat.
""")

print(type(result),result)
print(result['summary']) 
print(result['sentiment'])


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def countwords(str):
    c=0
    for i in str:
        if i==' ':
            c+=1
    return c

prompt=PromptTemplate(
    template="Write a interesting joke about {topic}",
    input_variables=['topic']
)


model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=50)

parser= StrOutputParser()

joke_chain = RunnableSequence(prompt,model,parser)



para_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'count': RunnableLambda(countwords)
})

chain = RunnableSequence(joke_chain,para_chain)

print(chain.invoke({'topic':'F1'}))

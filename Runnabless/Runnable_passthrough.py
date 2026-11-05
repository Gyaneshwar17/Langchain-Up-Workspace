
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

prompt1=PromptTemplate(
    template="Write a interesting fact about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="summersie {text}",
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=200)

parser= StrOutputParser()

fact_chain = RunnableSequence(prompt1,model,parser)

para_chain = RunnableParallel({
    'fact': RunnablePassthrough(),
    'summary': RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(fact_chain,para_chain)

print(chain.invoke({'topic':'F1'}))

from google.auth import default
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnableBranch, RunnablePassthrough, RunnableLambda

load_dotenv()

prompt1=PromptTemplate(
    template="Write a report about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Summerise {text} and give in 5 points",
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=500)


parser= StrOutputParser()

report_chain = prompt1 | model | parser

branch_chian = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = report_chain | branch_chian

print(chain.invoke({'topic':'F1'}))
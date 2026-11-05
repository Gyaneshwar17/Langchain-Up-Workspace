from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

prompt1=PromptTemplate(
    template="Write a interesting fact about {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Write a disgraceful fact about {topic}",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=200)

parser= StrOutputParser()

chain = RunnableParallel( 
    {
        'Good Fact': RunnableSequence(prompt1,model,parser),
        'Bad Fact': RunnableSequence(prompt2,model,parser)  
    }
)

result=chain.invoke({'topic':'F1'})

print(result)

# print(result['Good Fact'])

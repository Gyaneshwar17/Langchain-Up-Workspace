from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=200)

parser = StrOutputParser()

chain = RunnableSequence( prompt, model, parser)

print(chain.invoke({'topic':'AI'}))

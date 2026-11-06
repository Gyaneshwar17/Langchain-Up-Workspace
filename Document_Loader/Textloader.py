from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

load_dotenv()

prompt=PromptTemplate(
    template="Write a summary on  {text}",
    input_variables=['text']
)

loader = TextLoader(file_path='Document_Loader\F1.txt')

docs = loader.load()

print(type(docs))


model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=50)

parser= StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'text': docs[0].page_content}))

from langchain_classic import vectorstores
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_classic.vectorstores import FAISS
 


loader = TextLoader("docs.txt")
documents=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs=text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings)

retriver=vectorstore.as_retriever()

query = "What "
retrieved_docs=retriver._get_relevant_documents(query)

retrieved_text="\n".join([doc.page_content for doc in retrieved_docs])

llm=GoogleGenerativeAI(model_name="gemini-1.5-flash", temperature=0.7)

prompt=f"Based on the following text, answer the questions: {query}\n\n{retrieved_text}"
answer=llm.predict(prompt)

print("Answer:",answer) 
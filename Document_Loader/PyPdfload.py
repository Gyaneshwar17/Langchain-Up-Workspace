from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Document_Loader/OOPS.pdf")

docs=loader.load()

print(type(docs))

print(docs[2].metadata)

print(docs[2].page_content)
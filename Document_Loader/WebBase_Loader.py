from langchain_community.document_loaders import WebBaseLoader

URL="https://en.wikipedia.org/wiki/CNN"

loader = WebBaseLoader(URL)

docs = loader.load()

print(len(docs))

print(docs[0].page_content )
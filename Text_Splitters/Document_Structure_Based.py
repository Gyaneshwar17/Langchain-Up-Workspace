from joblib.externals.loky.backend.context import physical_cores_cache
from h11._abnf import chunk_size
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

python_code = """
class Calculator:
    def __init__(self, initial=0):
        self.total = initial

    def add(self, n):
        self.total += n
        return self.total

    def subtract(self, n):
        self.total -= n
        return self.total

    def multiply(self, n):
        self.total *= n
        return self.total

    def divide(self, n):
        if n != 0:
            self.total /= n
        return self.total

    def reset(self):
        self.total = 0
        return self.total
"""
splitter =RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)


# print(len(splitter.split_text(python_code)))
# print(splitter.split_text(python_code))

chunks = splitter.split_text(python_code)

print(len(chunks))

print(chunks[1])
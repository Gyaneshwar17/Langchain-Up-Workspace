from h11._abnf import chunk_size
from langchain_classic.text_splitter import CharacterTextSplitter

text ="A transistor is a semiconductor device used to amplify or switch electrical signals and power. It is one of the fundamental building blocks of modern electronics.[1] A transistor is composed of semiconductor material, often with at least three terminals for connection to an electronic circuit. A voltage or current applied to one pair of the transistor's terminals controls the current through another pair of terminals. Because the controlled (output) power can be higher than the controlling (input) power, a transistor can be used to amplify a signal. Some transistors are packaged individually, but many more in miniature form are found embedded in integrated circuits. Because transistors are the key active components in practically all modern electronics, they are considered one of the 20th century's greatest inventions."


splitter =CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
print(splitter.split_text(text))
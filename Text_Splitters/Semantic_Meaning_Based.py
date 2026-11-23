from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)


sample = """
Formula 1 cars rely heavily on aerodynamic downforce to take high-speed corners without losing grip. Drivers like Max Verstappen push their machines to the absolute limit during qualifying laps to secure pole position. Pit stop crews can change all four tires in under two seconds, making strategy and precision just as critical as raw engine power.South Indian breakfast is famous for its light, fermented dishes packed with flavor. Crispy masala dosa served with piping hot sambar and fresh coconut chutney is a staple across households and local darshinis. Steamed idlis and crunchy medu vadas offer a comforting balance of textures alongside a cup of strong filter coffee.India is the seventh-largest country in the world by land area and the most populous democracy. Its geography ranges from the snow-capped Himalayan peaks in the north to the tropical coastlines of the Indian Ocean in the south. With hundreds of spoken languages and diverse cultural traditions, every state offers a distinct heritage and history.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))
for i in docs:
    print(i)
    print("/n")


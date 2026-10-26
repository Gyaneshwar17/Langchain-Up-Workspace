
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
embbedings=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001',output_dimensionality=300)

documents=[
    "Lewis Hamilton is a British racing driver holding a record-tying seven Formula One World Drivers' Championship titles, known for his unmatched speed and fierce competitive drive.",
    "Max Verstappen is a Dutch racing driver competing for Red Bull Racing, renowned for his aggressive overtaking style and dominance in claiming multiple consecutive World Championships.",
    "Fernando Alonso is a Spanish racing driver celebrated for his exceptional racecraft, tactical brilliance, and longevity in Formula 1 spanning over two decades.",
    "Charles Leclerc is a Monégasque racing driver driving for Ferrari, celebrated for his breathtaking single-lap pace and mastery in securing pole positions.",
    "Lando Norris is a British racing driver known for his precise technical feedback, natural speed, and charismatic presence on and off the track.",
]

query='tell me about Max'

doc_embeddings=embbedings.embed_documents(documents)

query_embedding=embbedings.embed_query(query)

# print(cosine_similarity([query_embedding],doc_embeddings))
#always should be a 2D list

scores=cosine_similarity([query_embedding],doc_embeddings)[0]#Get Simple list

index, score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity score is:",score)

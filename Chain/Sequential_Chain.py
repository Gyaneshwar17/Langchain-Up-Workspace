from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=500)

prompt=PromptTemplate(
    template="what is {topic} ",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="summerise {text} in 5 lines",
    input_variables=["text"]
)

parser=StrOutputParser()

chain= prompt | model | parser |prompt2 | model | parser

result=chain.invoke({'topic':'schrodinger\'s cat'})

print(result)

chain.get_graph().print_ascii()


'''

Here is a 5-line summary of Schrödinger’s Cat:

1. Created by Erwin Schrödinger in 1935, this thought experiment places a cat in a sealed box with a lethal, quantum-triggered mechanism.
2. According to quantum mechanics, the unseen trigger is in "superposition" (both decayed and not decayed), meaning the cat is simultaneously alive and dead.
3. Opening the box causes this superposition to "collapse," forcing the cat into one definite state: either alive or dead.
4. Schrödinger did not believe this literally; instead, he used the absurdity to criticize flaws in mainstream quantum theory regarding everyday objects.
5. Today, the concept serves as a famous scientific thought experiment and a cultural metaphor for uncertainty and unresolved outcomes.
      +-------------+      
      | PromptInput |
      +-------------+
             *
             *
             *
    +----------------+
    | PromptTemplate |
    +----------------+
             *
             *
             *
+------------------------+
| ChatGoogleGenerativeAI |
+------------------------+
             *
             *
             *
    +-----------------+
    | StrOutputParser |
    +-----------------+
             *
             *
             *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+
             *
             *
             *
    +----------------+
    | PromptTemplate |
    +----------------+
             *
             *
             *
+------------------------+
| ChatGoogleGenerativeAI |
+------------------------+
             *
             *
             *
    +-----------------+
    | StrOutputParser |
    +-----------------+
             *
             *
             *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+

'''
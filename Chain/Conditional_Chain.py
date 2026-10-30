
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


model1=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=50)
model2=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',max_tokens=200)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive','Negative'] = Field(description='Give the sentiment of the feedback')

parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1= PromptTemplate(
    template='Classify the feedback->{feedback} as positve or negative \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)


classifier_chain = prompt1 | model1 | parser2


prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback} Output ONLY the direct response text. Do not provide options',
    input_variables=['feedback'
    ]
)

prompt3=PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback} Output ONLY the direct response text. Do not provide options',
    input_variables=['feedback'
    ]
)

branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='Positive', prompt2 | model2 | parser),
    (lambda x:x.sentiment=='Negative', prompt3 | model2 | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result=chain.invoke({'feedback':'It\'s phone.'})

print(result)


chain.get_graph().print_ascii()

'''

Thank you so much for your kind words! We’re thrilled to hear that you had such a positive experience, and we truly appreciate your support.
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
 +----------------------+
 | PydanticOutputParser |
 +----------------------+
             *
             *
             *
        +--------+
        | Branch |
        +--------+
             *
             *
             *
     +--------------+
     | BranchOutput |
     +--------------+
'''
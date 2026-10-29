from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=500)

template1= PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2= PromptTemplate(
    template='write a 5 line summary on the following text./n {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'Nothern Lights'})

print(result)

#output Parser
'''
Here is a 5-line summary of the text:

1. The Northern Lights, or Aurora Borealis, are a stunning natural light spectacle visible mainly in high Arctic latitudes.
2. They are scientifically caused by interactions between the Sun's charged solar wind and Earth's magnetic field.
3. The magnetosphere channels these trapped high-energy particles toward Earth's polar regions.
4. Upon entering the upper atmosphere, these particles collide with oxygen and nitrogen gases.
5. These collisions excite atmospheric gases, releasing photons that create the dancing green, red, blue, and violet lights.
'''

'''
prompt1=template1.invoke({'topic':'Northern Lights'})
result1=model.invoke(prompt1)

prompt2=template2.invoke({'text':result1.content[0]['text']})
result2=model.invoke(prompt2)

print(result2.content[0]['text'])
'''
#output content
'''
The Northern Lights, or Aurora Borealis, are a stunning atmospheric phenomenon visible primarily in high-latitude Arctic regions. 
They are scientifically created when charged particles from solar winds interact with Earth's magnetic field and upper atmosphere. 
These high-energy particles are funneled toward the poles, where they collide with atmospheric gases like oxygen and nitrogen. 
The resulting energy release, or photon emission, generates the dynamic and vibrant colors that light up the night sky. 
While green is the most common color produced by oxygen, the specific hues vary based on altitude and the type of gas involved.
'''
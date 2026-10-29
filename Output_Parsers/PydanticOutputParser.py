
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=-1,
    max_new_tokens=100,
    do_sample=True,
    temperature=1,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

class Person(BaseModel):
    name: str=Field(description='Name of the person')
    age: int= Field(description='Age of the person')
    city: str= Field(description='Name of the city the person belongs to')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person\n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain= template|llm|parser

result=chain.invoke({'place':'India'})

print(result)
'''
Prompt=template.invoke({'place':'indian'})

result=llm.invoke(Prompt)

print(parser.parse(result))
name='Rajesh Kumar' age=32 city='Delhi'
'''
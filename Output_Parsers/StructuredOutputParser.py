from networkx.generators import classic
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

schema=[
    ResponseSchema(name='fact_1',description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='Fact 3 about the topic'),

]

parser=StructuredOutputParser.from_response_schemas(schema)
template=PromptTemplate(
    template='Give 3 facts about {topic}\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


chain=template | llm | parser

result=chain.invoke({'topic':'Captain America'})


print(result)












'''
prompt=template.invoke({'topic':'Captain America'})

result=llm.invoke(prompt)

final_result=parser.parse(result)
print(final_result)

{'fact_1': 'Captain America is one of the most iconic superheroes in comic book history.', 'fact_2': 'He was first introduced by Jack Kirby as Captain John Rodgers of the Avengers', 'fact_3': 'Originally called Steve Rogers after his super strength gene experiment was reversed, he became known simply as Captain America after World War II'}
'''
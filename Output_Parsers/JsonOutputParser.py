from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device=-1,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.5,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

parser = JsonOutputParser()

template = PromptTemplate(
    template=(
        "Give me the name, age and city of a Marvel fictional person.\n"
        "{format_instruction}"
    ),
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    },
)

'''
prompt=template.format()
print(prompt)
Give me the name, age and city of a Marvel fictional person
 Return a JSON object.
'''

chain=template | llm | parser

result=chain.invoke({})

print(result)



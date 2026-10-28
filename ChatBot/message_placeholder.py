from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate(
    [
        ('system','You are a helpful F1 expert agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{query}')
    ]
)

chat_history=[]

with open('Chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

chat_template.invoke({'chat_history':chat_history, 'query':'Who is my Fav F1 driver.'}) 
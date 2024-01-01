from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)
prompt = ChatPromptTemplate(
    input_variables = ['content'],
    messages=[
       SystemMessage(content='You are chatbot having a conversation with a human.'), 
       HumanMessagePromptTemplate.from_template('{content}')
    
    ]
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

while True:
    content = input('Your prompt: ')
    if content in ['quit', 'exit', 'bye']:
        print('Goodbye!')
        break
    response = chain.run({'content':content})
    print(response)
    print('-'* 50)

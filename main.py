import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# print(os.environ.get('PINECONE_API_KEY'))

##### LLM Models GPT-3 #####
from langchain.llms import OpenAI
llm = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=512)
# print(llm)
output = llm('explain quantum mechanics in one sentence')
# print(output)

print(llm.get_num_tokens('explain quantum mechanics in one sentence'))
output = llm.generate(['... is the capital city of France.', 'What is the formula for the area of a circle?'])
print(output.generations)
print(output.generations[1][0].text)


#### GPT 3.5 #####
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI

chat  = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
message = [
    SystemMessage(content='You are physicist and respond only in German.'),
    HumanMessage(content='explain quantum mechanics in one sentence'),
]

output = chat(message)
print(output.content)
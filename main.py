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
# print(output.generations)
# print(output.generations[1][0].text)


#### GPT 3.5 #####
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.chat_models import ChatOpenAI

chat  = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
message = [
    SystemMessage(content='You are physicist and respond only in German.'),
    HumanMessage(content='explain quantum mechanics in one sentence'),
]

output = chat(message)
# print(output.content)

from langchain import PromptTemplate

template = '''You are an experienced virologist. Write a few sentences about the following {virus} in {language}.'''

prompt = PromptTemplate(input_variables=['virus','language'], template=template)

# print(prompt)

output = llm(prompt.format(virus='virus', language='Romanian'))
# print(output)


### Simple Chains ###
from langchain.chains import LLMChain, SimpleSequentialChain
chain = LLMChain(llm=chat, prompt=prompt)
output = chain.run({'virus':'HSV', 'language':'french'})
print(output)

### Sequential chains ### 
llm1 = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=1024)
template = '''You are an experienced scientist and Python programmer. Write a few sentences about the following {concept}.'''

prompt1 = PromptTemplate(input_variables=['concept'], template=template)
chain1 = LLMChain(llm=llm1, prompt=prompt1)

llm2 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1.2)
prompt2 = PromptTemplate(input_variables=['function'], template='Given the Python function {function}, described it as detailed as possible.')

chain2 = LLMChain(llm=llm2, prompt=prompt2)
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
output = overall_chain.run('linear')
print(output)

### Langchain Agents ###
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
agent_executor = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
output = agent_executor.run('Calculate the square root of the factorial of 20 and display it with 4 decimal points.')
print(output)

### Vector DB ###
import pinecone
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
print(pinecone.info.version())

pinecone.list_indexes()
index_name = 'langchain-pinecone'
if index_name not in pinecone.list_indexes():
    print(f'Creating index {index_name}')
    pinecone.create_index(index_name, dimension=1536, metric='cosine',pods=1, pod_type='p1.x2' )
    print('Done')
else:
    print(f'Index {index_name} already exists')

print(pinecone.describe_index(index_name))
# if index_name in pinecone.list_indexes():
#     print(f'Deleting index {index_name}')
#     pinecone.delete_index(index_name)
#     print('Done')
# else:
#     print(f'Index {index_name} does not exist')
index= pinecone.Index(index_name=index_name)
index.describe_index_stats()
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_document(file):
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {file}...')
    loader = PyPDFLoader(file)
    document = loader.load()
    return document

data = load_document('us_constitution.pdf')
print(data[1].page_content)
print(data[10].metadata)




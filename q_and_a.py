import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}...')
        loader = Docx2txtLoader(file)   
    else:
        print('Document format is not supported.')
        return None
    document = loader.load()
    return document

def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query, lang=lang, load_max_docs=load_max_docs)
    document = loader.load()
    return document

def chunk_data(data, chunk_size=256, chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

### Embedding Cost ### 
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Totak tokens: {total_tokens}')
    print(f'Embedding cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

def insert_of_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings...')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Index {index_name} does not exist. Creating index and loading embeddings...')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_new_index(index_name, embeddings, index_name)
        print('Ok')

    return vector_store

data = load_document('us_constitution.pdf')
# print(data[1].page_content)
# print(data[10].metadata)

print(f'Number of pages: {len(data)}')
print(f'Number of characters: {data[20].page_content}')

# data = load_document('the_great_gatsby.docx')
# print(data[0].page_content)

# data = load_from_wikipedia('GPT-4')
# print(data[0].page_content)


chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)
print_embedding_cost(chunks)


# %%
from dotenv import load_dotenv
load_dotenv('.env')


# %%
# ---- Chroma DB ---- #
# 1. PDF 데이터 로드
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

pdf_path = './data/replicating anomalies.pdf'

loader = PyPDFLoader(pdf_path)
documents = loader.load()


# 2. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, 
    chunk_overlap=100,
    encoding_name='cl100k_base'
)

splited_documents = text_splitter.split_documents(documents)


# 3. 임베딩 + 저장
embeddings_model = OpenAIEmbeddings()
splited_texts = [doc.page_content for doc in splited_documents]

db = Chroma.from_texts(
    splited_texts,
    embeddings_model,
    collection_name='anomalies',
    persist_directory='./db/chroma_db',
    collection_metadata={'hnsw:space': 'cosine'}    # l2 is default
)


# 4. DB 검색
query = "What is the main idea of the paper?"
docs = db.similarity_search(query, k=3)
print(docs[0].page_content)

# %%
# max_marginal_relevance_search

'''
fetch_k:    검색 결과에서 가져올 문서의 수
      k:    검색 결과에서 반환할 문서의 수
'''
query = "What is the q-anomalies?"
mmr_docs = db.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(mmr_docs[0].page_content)

# %%
# ---- 벡터 스토어를 로컬에 저장하기 ---- #

db = Chroma.from_texts(
    splited_texts,
    embeddings_model,
    collection_name='anomalies',
    persist_directory='./db/chroma_db',
    collection_metadata={'hnsw:space': 'cosine'}    # l2 is default
)

# Chroma 벡터 저장소를 로컬 디스크에 저장하는 기능
db.persist()

# %%
db2 = Chroma(
    persist_directory='./db/chroma_db',
    collection_name='anomalies',
    embedding_function=embeddings_model
)

query = "What is the main idea of the paper?"
mmr_docs = db2.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(mmr_docs[0].page_content)

# %%
# ---- 검색 도구(Retrivers)들을 사용하기 ---- #

# 1. 제일 유사한 n개의 문서를 가져오기

query = "What is q-anomalies?"
retriever = db2.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

documents = retriever.get_relevant_documents(query)
print(documents[0].page_content)


# %%
# ---- 답변 생성 ---- #

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

query = "What is q-anomalies?"

retriever = db2.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.get_relevant_documents(query)

# LLM에 전달할 프롬프트 템플릿 정의
template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 모델 선택
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-0125',  # 제일 싸다
    temperature=0,
    max_tokens=500,
)

# 출력 파서
output_parser = StrOutputParser()

# 체인 생성
chain = prompt | llm | output_parser

# Run
response = chain.invoke({'context': format_docs(docs), 'question': query})
print(response)
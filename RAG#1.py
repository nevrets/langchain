# %%
from dotenv import load_dotenv
load_dotenv('.env')


# %%
# ---- RAG 1단계 : 데이터 로드 ---- #
from langchain_community.document_loaders import WebBaseLoader

# 나무위키 : 언어모델
url = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
documents = loader.load()
print(documents[0].page_content)


# %%
# ---- RAG 2단계 : 텍스트 분할 ---- #

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 나무위키 : 언어모델
url = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
documents = loader.load()

# 텍스트 분할기 정의
# 1000글자 단위로 나눌 때 200개 정도는 겹치도록 진행(문맥이 잘려나가지 않게 하기 위함)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = text_splitter.split_documents(documents)

print(len(splitted_documents))  # 분할된 문장 개수
print(splitted_documents[10].page_content)


# %%
# ---- RAG 3단계 : 인덱싱 ---- #

# Data Loader - 웹페이지 데이터 가져오기
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


url = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"
loader = WebBaseLoader(url)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = text_splitter.split_documents(documents)

# 벡터 저장소 정의
vector_store = Chroma.from_documents(
    documents=splitted_documents,
    embedding=OpenAIEmbeddings(),
)

documents = vector_store.similarity_search(query="관련 유니콘 기업들을 소개해달라")
print(len(documents))
print(documents[0].page_content)


# %%
# ---- RAG 4단계 : 검색 ---- #

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


url = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"
loader = WebBaseLoader(url)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = text_splitter.split_documents(documents)

# 벡터 저장소 정의
vector_store = Chroma.from_documents(
    documents=splitted_documents,
    embedding=OpenAIEmbeddings(),
)

prompt_template = '''
Answer the question base on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(prompt_template)

model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

retriever = vector_store.as_retriever()

# Combine Documents
def format_documents(documents):
    return "\n".join([document.page_content for document in documents])

# Chain 연결
chain = (
    {'context': retriever | format_documents, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("대표 유니콘 기업들을 알려줘")
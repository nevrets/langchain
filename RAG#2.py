# %%
from dotenv import load_dotenv
load_dotenv('.env')


# %%

# ---- RAG 과정 ---- #
# 1. 문서를 가져옴 (Document Loader)
# 2. 문서를 잘 나눔 (Text Splitter)
# 3. AI가 이해할 수 있게 숫자로 바꿈. 임베딩 (Embedding)
# 4. 임베딩 벡터들을 검색할 수 있게 저장함. 벡터 저장소 (Vector Store)
# 5. 벡터 저장소에서 문서를 검색 (Retriever)
# 6. 답변 생성 (Prompt + LLM)


# %%

# ---- 1. 웹 문서 불러오기 ---- #

# 1. 문서를 가져온다. (Document Loader)
# 웹문서
# 텍스트
# 폴더에서 한꺼번에
# CSV
# PDF
# 데이터베이스

import bs4
from langchain_community.document_loaders import WebBaseLoader

url1 = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"  # 나무위키: 언어모델
url2 = "https://namu.wiki/w/ChatGPT"

# 웹페이지를 통째로 읽어오면 나중에 사용하기 불편
# loader = WebBaseLoader(web_path = (url1, url2))  # 파이썬 tuple로

loader = WebBaseLoader(web_path = (url1, url2),
                       bs_kwargs = dict(
                           parse_only = bs4.SoupStrainer(
                               class_ = ["wiki-heading-content", "wiki-paragraph"])
                            )
                       )

# 웹페이지 텍스트 -> Documents
documents = loader.load()

print(len(documents))
print(documents[0])


# %%
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document


url1 = "https://namu.wiki/w/%EC%96%B8%EC%96%B4%20%EB%AA%A8%EB%8D%B8"
url2 = "https://namu.wiki/w/ChatGPT"

def load_namuwiki(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 본문 내용 추출
    content = soup.select('div.wiki-heading-content, div.wiki-paragraph')
    text = ' '.join([p.get_text().strip() for p in content])
    
    return Document(page_content=text, metadata={'source': url})

# 각 URL에서 문서 로드
documents = [load_namuwiki(url) for url in [url1, url2]]

print(len(documents))
print(documents[0])


# %%
# ---- 2. 텍스트 불러오기 ---- #

from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path = "example.txt")
documents = loader.load()

print(len(documents))
print(documents[0])


# %%
# ---- 3. 특정 폴더의 모든 문서 가져오기 ---- #

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path = "./data", glob = "*.txt")
documents = loader.load()

print(len(documents))
print(documents[0])


# %%
from langchain_community.document_loaders import TextLoader
import os
import glob

def load_text_files(directory_path):
    documents = []
    
    # 디렉토리 내의 모든 .txt 파일 찾기
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"로드됨: {file_path}")
        except Exception as e:
            print(f"파일 로드 실패 {file_path}: {str(e)}")
    
    return documents

# 텍스트 파일 로드
directory_path = "./data" 
documents = load_text_files(directory_path)

print(f"\n총 로드된 문서 수: {len(documents)}")
if documents:
    print("\n첫 번째 문서 내용:")
    print(documents[0])
    
    
# %%
# ---- 4. CSV 불러오기 ---- #

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path = "./data/sample.csv", encoding = "utf-8")
data = loader.load()

print(len(data))
print(data[0])
print(data[1])
print(data[2])


# %%
# ---- ***** 5. PDF 불러오기 ***** ---- #

from langchain_community.document_loaders import PyPDFLoader

pdf_path = "./data/BERTopic_API.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(len(documents))
print(documents[0])


# %%
from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_path = "./data/BERTopic_API.pdf"

# 전체 텍스트를 단일 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_path)
documents = loader.load()

print(len(documents))


# %%

from langchain_community.document_loaders import PyPDFDirectoryLoader

pdf_directory = "./data"

loader = PyPDFDirectoryLoader(pdf_directory)
documents = loader.load()

print(len(documents))
print(documents[-1])
# %%

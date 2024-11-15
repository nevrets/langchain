# %%
from dotenv import load_dotenv
load_dotenv('.env')


# %%
# ---- 임베딩 (Embedding) ---- #
# 1. 불러온 긴 문서를 작은 단위인 chunk로 나누는 도구
# 2. chunk로 나눠진 텍스트 데이터를 숫자로 변환 (벡터 변환)


# %%
# ---- 1. 텍스트 분리하기 (Text Splitter) ---- #
# 1. 단순하게 나누기 (CharacterTextSplitter)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("./data/example.txt", encoding = "utf-8")
documents = loader.load()

print(len(documents))
print(documents[0].page_content)


# %%

# 2. 문자 단위로 나누기

from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("./data/example.txt", encoding = "utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator = "",
    chunk_size = 100,
    chunk_overlap = 10,
    length_function = len,
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))
print(len(chunks[0].page_content))
print(chunks[0])


# %%

# 줄바꿈 문자를 기준으로 분할
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 100,
    chunk_overlap = 10,
    length_function = len,
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))
print(len(chunks[0].page_content))
print(chunks[0])


# %%

# 3. 조금 더 의미적으로 나누기 (RecursiveCharacterTextSplitter) **

from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("./data/example.txt", encoding = "utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # sperator 지정 필요없음
    chunk_size = 100,
    chunk_overlap = 10,
    length_function = len,
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))
print(len(chunks[0].page_content))
print(chunks[0])


# %%

# ---- 2. 임베딩 (Embedding) ---- #
# 1. OpenAI Embeddings

from langchain_openai import OpenAIEmbeddings

text_embedding_model = OpenAIEmbeddings()

texts = [
    '봄바람이 부는 날, 꽃가루가 나뭇잎 사이로 흘날립니다.',
    '수확은 묵제 해결 능력을 키우는 데에 아주 중요한 과목입니다.',
    '커피 한 잔의 여유를 즐기며 일상의 스트레스를 잠시 잊어보세요.',
    '지구의 날씨 변화는 기후 변화의 직접적인 증거 중 하나입니다.',
    '새로운 기술은 종종 기존의 작업을 변화시키고 새로운 기회를 만들었습니다.',
    '연희 감성은 문화적 이해를 넓히는 좋은 방법이 될 수 있습니다.',
    '바다 깊은 곳에 서식하는 생물들은 매우 독특한 생태계를 이루고 있습니다.',
    '책을 읽는 것은 지식을 넓히고 사고력을 향상시키는 활동입니다.'
]

embeddings = text_embedding_model.embed_documents(texts)

print(len(embeddings))
print(len(embeddings[0]))    # 임베딩 벡터의 차원
print(embeddings[0][:10])


# %%
# 하나의 문장 임베딩
import numpy as np

embedded_query = text_embedding_model.embed_query("커피 한 잔의 여유를 즐기며 일상의 스트레스를 잠시 잊어보세요.")
embedded_query[:10]


# %%

from sklearn.metrics.pairwise import cosine_similarity

embedded_query = text_embedding_model.embed_query("스트레스가 심할 때 대처할 수 있는 방법은 무엇인가요?")
embedded_query_np = np.array(embedded_query)
embedded_query_np_reshaped = embedded_query_np.reshape(1, -1)

cosine_similarities = cosine_similarity(embedded_query_np_reshaped, embeddings)
cosine_similarities

# "커피 한 잔의 여유를 즐기며 일상의 스트레스를 잠시 잊어보세요."가 가장 높게 나옴
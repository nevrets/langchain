# %%
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

from dotenv import load_dotenv
load_dotenv('.env')


# %%
import os
from langchain_openai import ChatOpenAI

# model
model = ChatOpenAI(model="gpt-3.5-turbo")

# chain 실행
model.invoke("대전에서 가장 아파트값이 비싼 지역은? 특히 대표하는 아파트는?")


# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("넌 프로게이머다. 질문에 대답해라. {input}이라는 게임의 특징은? ")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()


# chain 연결
chain = prompt | llm | output_parser

# chain 실행
chain.invoke({"input": "리그오브레전드"})


# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_kr = ChatPromptTemplate.from_template("넌 프로게이머다. 질문에 대답해라. {input}이라는 게임의 특징은? ")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = StrOutputParser()

# chain 연결
chain1 = prompt_kr | llm | output_parser

prompt_translate = ChatPromptTemplate.from_template("넌 번역가이다. 입력이 영어가 아니면 영어로 변환해서 대답해라. {word}")
chain2 = (
    {"word": chain1} | prompt_translate | llm | output_parser
)

# chain 실행
chain2.invoke({"input": "리그오브레전드"})


# %%
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = '안녕하세요, 저는 {name}이고, 나이는 {age}살입니다.'

# PromptTemplate 인스턴스 생성
prompt_template = PromptTemplate(template=template_text, input_variables=['name', 'age'])

# 템플릿에 값을 채워서 프롬프트 완성
prompt = prompt_template.format(name='홍길동', age=32)

prompt


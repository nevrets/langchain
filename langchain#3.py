# %%
import os
import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv('.env')


# %%
'''
AttributeError: 'bool' object has no attribute 'lookup'
'''
from langchain_openai import OpenAI

llm = OpenAI()

print(llm.invoke("대한민국의 각 주에서 가장 큰 대표도시들은 무엇이 있습니까?"))


# %%
chat = ChatOpenAI()

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 지리 전문가입니다."),
    ("user", "{user_input}"),
])

chain = chat_prompt | chat
chain.invoke({"user_input": "대한민국의 각 주에서 가장 큰 대표도시들은 무엇이 있습니까?"})


# %%

params = {
    "temperature": 0.5,
    "max_tokens": 100,
}

model = ChatOpenAI(model="gpt-3.5-turbo", **params)

question = "What is the capital of France?"
response = model.invoke(question)

print(response.content)


# %%
params = {
    "temperature": 0.5,
    "max_tokens": 100,
}

mode_kwargs = {
    "frequency_penalty": 0.5,    # 이미 등장한 단어의 재등장 확률
    "presence_penalty": 0.5,     # 새로운 단어의 도입을 장려
    "stop": ["\n"],              # 정지 시퀀스 설정
}

model = ChatOpenAI(model="gpt-3.5-turbo", **params, model_kwargs=mode_kwargs)

question = "태양계와 가장 가까운 항성계는 어디인가요?"
response = model.invoke(question)

print(response.content)


# %%
prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학자입니다."),
    ("user", "{user_input}"),
])

model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=100)
messages = prompt.format_messages(user_input="태양계와 가장 가까운 항성계는 어디인가요?")

before_binding_answer = model.invoke(messages)

# 모델 파라미터 추가로 바인딩
chain = prompt | model.bind(max_tokens=10)
after_binding_answer = chain.invoke({"user_input": "태양계와 가장 가까운 항성계는 어디인가요?"})

print("before binding: " + before_binding_answer.content)
print("after binding: " + after_binding_answer.content)


# %%
## 출력되는 형식 제어하기

from langchain_core.output_parsers import JsonOutputParser

output_parser = JsonOutputParser()
format_instructions = output_parser.get_format_instructions()

# prompt 구성
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

chain = prompt | model | output_parser

chain.invoke({"query": "토마토 파스타랑 비빔밥 만드는 레시피 알려주세요."})


# %%

## pydantic: python에서 데이터 유효성 검사와 설정 관리를 위해 사용되는 라이브러리.
import json
from langchain_core.pydantic_v1 import BaseModel, Field

# pydantic 모델 정의
class Recipe(BaseModel):
    name: str = Field(description="The name of the recipe")
    ingredients: list[str] = Field(description="The ingredients needed for the recipe")
    instructions: str = Field(description="The instructions to prepare the recipe")

# 출력 파서 정의
output_parser = JsonOutputParser(pydantic_object=Recipe)
format_instructions = output_parser.get_format_instructions()

# prompt 구성
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | model | output_parser

output_json = chain.invoke({"query": "토마토 파스타랑 비빔밥 만드는 레시피 알려주세요."})

pretty_json = json.dumps(output_json, indent=4)
print(pretty_json)
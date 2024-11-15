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

# model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# %%

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = '안녕하세요, 저는 {name}이고, 나이는 {age}살입니다.'

# PromptTemplate 인스턴스 생성
prompt_template = PromptTemplate.from_template(template_text)

# 템플릿에 값을 채워서 프롬프트 완성
prompt = prompt_template.format(name='홍길동', age=32)

prompt
# %%

# PromptTemplate 인스턴스를 생성
template_text1 = "안녕하세요, 저는 {name}이고, 나이는 {age}살입니다."
template_text2 = "\n\n{city}에서 살고 직업은 {job}입니다."
text = "\n\n{language}로 번역해라."

prompt_template1 = PromptTemplate.from_template(template_text1)
prompt_template2 = PromptTemplate.from_template(template_text2)

# 문자열 연결
# text는 이 프롬프트 템플릿들이 근본적으로 text와 별다르지 않다는걸 보여주기 위해 만들었음
merged_prompt = (prompt_template1 + prompt_template2 + text)

# 모델 인스턴스 생성
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# 체인 인스턴스 생성
chain = merged_prompt | llm | StrOutputParser()

# 체인 실행
chain.invoke({"name": "홍길동", "age": 32, "city": "대전", "job": "머신러닝 엔지니어", "language": "영어"})

# %%

'''
PromptTemplate : 사실상 일반 텍스트와 크게 다르지 않음
ChatPromptTemplate : 프롬프트 템플릿을 챗모델에 맞게 조정 (상호간에 채팅을 한다고 가정하고 대답을 유도)
                     즉, 챗봇에게 역할을 부여하고 어떻게 대답할지를 지정하는 템플릿

'''

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 챗봇은 LLM 과학자입니다. LLM과 관련된 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

messages = chat_prompt.format_messages(user_input="현재 파라미터가 가장 많은 llm 모델은 무엇인가요?")
messages

# %%

chain = chat_prompt | llm | StrOutputParser()
output = chain.invoke({"user_input": "현재 파라미터가 가장 많은 llm 모델은 무엇인가요?"})
print(output)
# %%

'''
chat_prompt1과 같은 동작을 하는 chat_prompt2를 만들었다.
같은 동작을 하지만 chat_prompt2처럼 작성하는 것은 시스템과 사용자 간의 대화 흐름을 좀 더 명확하게 표현하기 때문에,
언어 모델이 더 적절한 응답을 할 수 있게 된다.

SystemMessagePromptTemplate : 시스템 메시지 프롬프트 템플릿
HumanMessagePromptTemplate : 사용자 메시지 프롬프트 템플릿
'''

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt1 = ChatPromptTemplate.from_messages([
    ("system", "이 챗봇은 LLM 과학자입니다. LLM과 관련된 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

chat_prompt2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("이 챗봇은 LLM 과학자입니다. LLM과 관련된 질문에 답변할 수 있습니다."),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])

# %%

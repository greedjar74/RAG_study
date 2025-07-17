'''
RAG 구현 순서

1. pdf 파일에서 텍스트 추출
2. 텍스트를 splitter를 사용하여 chunk로 분할
3. 각 chunk를 임베딩
4. FAISS, ChromaDB 등을 사용하여 vectorstore 생성
5. vectorstore를 사용하여 retriever 생성
6. 사용자 query 입력
7. retriever.invoke()를 통해 유사한 문서 검색 -> 사용자 입력은 자동으로 임베딩 수행
8. 문서, query를 사용하여 prompt 완성
9. llm.invoke()를 통해 답변 생성
'''

import os
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from pypdf import PdfReader

# api key 설정
os.environ["OPENAI_API_KEY"] = ""
# pdf 파일에서 텍스트 데이터 추출
def pdf_text_extract(file_path):
    text = ""
    reader = PdfReader(file_path)
    
    for page in reader.pages:
        text += f"{page.extract_text()} \n" or "\n"
    
    return text

text = pdf_text_extract("커리큘럼 제작 참고자료.pdf")

# 문서를 chunk 단위로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# embadding 모델 설정
embadding = OpenAIEmbeddings(model="text-embedding-3-small")

# 문서를 임베딩 및 retriever 생성
vectorstore = FAISS.from_documents(docs, embadding)
retriever = vectorstore.as_retriever()

# LLM 모델 설정
llm = ChatOpenAI(model='gpt-4.1-mini')

# prompt 형식 설정
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
너는 문서를 참고하여 답변한다.

[문서내용]
{context}

[질문]
{question}

[답변]
"""
)

user_input = input("질문을 입력하세요 : ")

# 사용자 입력에 대한 답변 생성
docs = retriever.invoke(user_input) # 유사한 문서 검색
context = "\n\n".join(doc.page_content for doc in docs) # 찾은 문서를 하나의 문자열으로 만든다.
prompt_text = prompt.format(context=context, question=f'user: {user_input}') # 프롬프트 생성
response = llm.invoke(prompt_text) # llm 답변 생성

print(response.content)
'''
# RAG V1
- RAG 시스템 구축

문제점
- 매번 임시 DB 생성 -> api 사용량 증가
- 한 가지 질문에만 답변
'''

from langchain import hub
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import os

# 환경 설정
os.environ["OPENAI_API_KEY"] = ""  # 실제 키로 교체하세요
sys.stdout.reconfigure(encoding='utf-8')

# PDF 불러오기 및 텍스트 추출
reader = PdfReader("커리큘럼 제작 참고자료.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 텍스트를 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

#ChromaDB에 청크들을 벡터 임베딩으로 저장(OpenAI 임베딩 모델 활용)
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(model = 'text-embedding-3-small'))
retriever = vectorstore.as_retriever()

# 모델 및 prompt 형태 설정
llm = ChatOpenAI(model='gpt-4o-mini')
prompt = hub.pull('rlm/rag-prompt')

#Retriever로 검색한 유사 문서의 내용을 하나의 string으로 결합
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("타일러 모형이 뭐야?")
# print(answer)

# 한 문장씩 줄 바꿈 적용
for sentence in answer.split('. '):
    print(sentence.strip())
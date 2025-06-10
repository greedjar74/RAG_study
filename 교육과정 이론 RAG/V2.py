'''
# V2
- 대화형

문제점
- 기존 대화 저장 X
- 매번 임시 DB 생성
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
reader = PdfReader("교육과정 이론 RAG/커리큘럼 제작 참고자료.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 텍스트를 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# 벡터 DB 생성 및 리트리버 설정
vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(model='text-embedding-3-small'))
retriever = vectorstore.as_retriever()

# LLM 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")

# RAG 체인 정의
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 대화 루프
print("RAG 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)\n")
while True:
    user_input = input("👤 질문: ").strip()
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("🛑 대화를 종료합니다.")
        break

    try:
        response = rag_chain.invoke(user_input)
        print("\n🤖 답변:")
        for sentence in response.split('. '):
            print(" -", sentence.strip())
        print()  # 줄바꿈
    except Exception as e:
        print("⚠️ 오류 발생:", e)
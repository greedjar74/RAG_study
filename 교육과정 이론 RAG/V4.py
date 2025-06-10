'''
# V4
- 대화형
- 기존 대화 기억
- DB 영구 저장
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

# 영구 벡터 DB 생성 및 저장
db_path = "./교육과정_이론_DB"

if os.path.exists(db_path):
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )
else :
    vectorstore = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=db_path
    )
    vectorstore.persist()

# retriever 설정
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

chat_history = []

# 대화 루프
print("RAG 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)\n")
while True:
    user_input = input("👤 질문: ").strip()
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("🛑 대화를 종료합니다.")
        break

    full_prompt = '\n'.join(chat_history + [f"user: {user_input}"])

    try:
        response = rag_chain.invoke(full_prompt)
        print("\n🤖 답변:")
        for sentence in response.split('. '):
            print(" -", sentence.strip())
        print()  # 줄바꿈
    except Exception as e:
        print("⚠️ 오류 발생:", e)
    
    chat_history.append(f"user: {user_input}")
    chat_history.append(f"assistant: {response}")

    
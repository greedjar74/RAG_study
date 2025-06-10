# V7 - Chroma DB에 대화 내용 저장
import os
import sys
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# 환경 설정
os.environ["OPENAI_API_KEY"] = ""  # 🔐 OpenAI API 키 입력
sys.stdout.reconfigure(encoding='utf-8')

# PDF 로딩 및 텍스트 추출
reader = PdfReader("summary of curriculum theory.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# DB 경로
db_path = "./curriculum_theory_DB"

# DB가 있는 경우 -> 로드
if os.path.exists(db_path):
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

# DB가 없는 경우 -> 생성
else:
    vectorstore = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=db_path
    )
    vectorstore.persist()

# DB retriever 설정
retriever = vectorstore.as_retriever()

# LLM 설정
llm = ChatOpenAI(model="gpt-4.1-mini")

# 프롬프트 템플릿
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 전문적인 도움을 주는 어시스턴트입니다.
다음 문서 내용을 참고하여 사용자 질문에 구체적으로 답해주세요.

[문서 내용]
{context}

[대화 내용 및 질문]
{question}

[답변]
"""
)

# 문서 포맷
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🔄 대화 내용을 Chroma DB에 저장
def store_chat_to_chroma(user_input, assistant_reply):
    conversation_text = f"user: {user_input}\nassistant: {assistant_reply}"
    doc = Document(page_content=conversation_text)
    vectorstore.add_documents([doc])
    vectorstore.persist()

# RAG 실행 함수
def run_RAG(user_input, chat_history):
    docs = retriever.invoke(user_input)
    context = format_docs(docs)
    conversation = '\n'.join(chat_history + [f'user: {user_input}'])
    prompt_text = prompt.format(context=context, question=conversation)
    response = llm.invoke(prompt_text)
    return response.content, docs

# 🔁 이전 대화 불러오기 (메모리 기준, DB에 저장 X)
chat_history = []

# 대화 루프
print("RAG 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)\n")
while True:
    user_input = input("👤 질문: ").strip()
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("🛑 대화를 종료합니다.")
        break

    try:
        response_text, source_docs = run_RAG(user_input, chat_history)

        print("\n🤖 답변:")
        for sentence in response_text.split('. '):
            print(sentence.strip())

        print('\n📄 참고한 문서 내용:')
        for i, doc in enumerate(source_docs, 1):
            print(f'\n[{i}] {doc.page_content.strip()[:300]}...')

        # ✅ Chroma DB에 대화 저장
        store_chat_to_chroma(user_input, response_text)

        # 메모리에도 저장 (맥락 유지)
        chat_history.append(f"user: {user_input}")
        chat_history.append(f"assistant: {response_text}")

    except Exception as e:
        print("⚠️ 오류 발생:", e)

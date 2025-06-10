'''
# V6
- 대화형
- 기존 대화 기억
- DB 영구 저장
- prompt 형태 사용자화
- 문서에서 참고한 내용을 출력
- rag_chain 대신 직접 파이썬 코드로 rag 구현 -> 대화 히스토리 등 세부적인 설정 가능
'''

from langchain import hub
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import sys
import os

# 환경 설정
os.environ["OPENAI_API_KEY"] = ""  # 실제 키로 교체하세요
sys.stdout.reconfigure(encoding='utf-8')

# PDF 불러오기 및 텍스트 추출
reader = PdfReader("summary of curriculum theory.pdf")
text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# 텍스트를 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

# 영구 벡터 DB 생성 및 저장
db_path = "./curriculum_theory_DB"

# DB가 있는 경우
if os.path.exists(db_path):
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    ) # 기존 DB 사용

# DB가 없는 경우 -> DB 생성
else :
    vectorstore = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=db_path
    )
    vectorstore.persist() # DB 생성

# retriever 설정
retriever = vectorstore.as_retriever()

# LLM 및 프롬프트 설정
llm = ChatOpenAI(model="gpt-4.1-mini")
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
다음 문서를 참고하여 사용자 질문에 성실하고 구체적으로 답변하세요.

[문서 내용]
{context}

[질문]
{question}

[답변]
"""
)

# 문서 스타일 변환
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_RAG(user_input, chat_history):
    docs = retriever.invoke(user_input) # 문서에서 관련 내용 탐색
    context = format_docs(docs) # 참고 자료를 문장 형식으로 변환

    conversation = '\n'.join(chat_history + [f'user: {user_input}']) # 기존 대화 내용, 사용자 입력을 병합

    prompt_text = prompt.format(context=context, question=conversation) # 프롬프트 생성

    response = llm.invoke(prompt_text) # 질문에 대한 결과 생성

    return response.content, docs # 결과 반환

chat_history = []

# 대화 루프
print("RAG 챗봇에 오신 것을 환영합니다! (종료하려면 'exit' 입력)\n")
while True:
    user_input = input("👤 질문: ").strip()
    if user_input.lower() in ["exit", "quit", "종료"]:
        print("🛑 대화를 종료합니다.")
        break

    try:
        response_text, source_docs = run_RAG(user_input, chat_history) # 결과 및 참고 자료 반환
        
        print("\n🤖 답변:")
        for sentence in response_text.split('. '):
            print(sentence.strip())

        print('\n참고한 문서 내용 : ')
        for i, doc in enumerate(source_docs, 1):
            print(f'\n[{i}] {doc.page_content.strip()[:300]}...')
        print()  # 줄바꿈

    except Exception as e:
        print("⚠️ 오류 발생:", e)
    
    chat_history.append(f"user: {user_input}")
    chat_history.append(f"assistant: {response_text}")

    
# RAG V1 with FAISS
# - 임시 DB 생성 → FAISS로 개선
# - 여러 질문 대응 가능 구조

from langchain import hub
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # ✅ Chroma → FAISS로 변경
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

# ✅ FAISS를 사용한 벡터스토어 생성
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = FAISS.from_documents(docs, embedding_model)

# 필요 시 저장: vectorstore.save_local("faiss_index")
# 필요 시 불러오기: FAISS.load_local("faiss_index", embedding_model)

retriever = vectorstore.as_retriever()

# LLM 및 Prompt 설정
llm = ChatOpenAI(model='gpt-4o-mini')
prompt = hub.pull('rlm/rag-prompt')

# 검색 결과 포맷팅
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 정의
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = '타일러 모형이 뭐야?'

# 🔍 관련 문서 검색 및 출력
related_docs = retriever.invoke(query)
formatted_context = format_docs(related_docs)

print("🔍 관련 문서:\n")
print(formatted_context)
print("\n🧠 답변:\n")

# 🧠 답변 생성
answer = rag_chain.invoke(query)

# 출력 포맷
for sentence in answer.split('. '):
    print(sentence.strip())
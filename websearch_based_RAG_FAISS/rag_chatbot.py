import streamlit as st
import os
import sys
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from pypdf import PdfReader

# 🔍 웹 검색 및 크롤링
def search_web(query, num_results=3, api_key=None):
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"} # API 호출 시 필요한 정보를 담고 있는 HTTP 헤더
    payload = {"q": query, "num": num_results} # 검색 요청 데이터 -> 검색어, 요청 결과 개수
    response = requests.post("https://google.serper.dev/search", json=payload, headers=headers) # post 요청
    results = response.json().get("organic", []) # 응답을 json 형식으로 파싱, organic: 일반 검색 결과 목록을 의미
    return [item["link"] for item in results] # 검색 결과에서 link만 추출하여 반환 -> 제목, 요약 설명, 웹사이트 아이콘 등은 필요 없기에 사용하지 않는다.

# url에서 텍스트 추출
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5) # url으로 이동 요청
        soup = BeautifulSoup(response.text, "html.parser") # html 텍스트를 BeautifulSoup 객체로 파싱
        for tag in soup(["script", "style"]): tag.decompose() # <script>, <style> 태그 제거 -> 버튼 동작 정의, 글꼴 정의 등 의미가 없는 정보
        return soup.get_text(separator="\n").strip() # html에서 텍스트만 추출하여 반환
    except Exception:
        return ""

# 🧠 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 📘 PDF 텍스트 추출 (pypdf 사용)
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file) # pdf파일 로드
        for page in reader.pages: 
            text += page.extract_text() or "" # 각 페이지에서 텍스트 추출 후 text 변수에 추가
        return text
    except Exception:
        return ""

# 📚 문서 벡터화 - 웹 기반
def get_search_docs(query, embedding_model, serper_key, k=5):
    urls = search_web(query, num_results=3, api_key=serper_key) # 관련 url 크롤링
    docs = []
    for url in urls:
        text = extract_text_from_url(url) # 웹 페이지에서 텍스트 추출
        if text:
            chunks = text_splitter.split_text(text) # 문서를 chunk 단위로 분할
            docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]) # docs에 각 chunk를 추가
    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_model) # chunk단위로 나눠진 문서 데이터를 FAISS에 저장
        retriever = vectorstore.as_retriever(search_kwargs={"k": k}) # retriever 구성
        return retriever, docs
    return None, []

# 📚 문서 벡터화 - PDF 기반
def get_pdf_docs(file, embedding_model, k=5):
    text = extract_text_from_pdf(file) # pdf에서 텍스트 추출
    if not text:
        return None, []

    chunks = text_splitter.split_text(text) # 텍스트를 chunk 단위로 분할
    docs = [Document(page_content=chunk, metadata={"source": "업로드된 PDF"}) for chunk in chunks] # docs에 각 chunk를 추가
    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_model) # chunk 단위로 나눠진 문서 데이터를 FAISS에 저장
        retriever = vectorstore.as_retriever(search_kwargs={"k": k}) # retriever 구성
        return retriever, docs
    return None, []

# 🧠 RAG 처리
def run_RAG(user_input, chat_history, mode, embedding_model, serper_key=None, pdf_file=None):
    # query에 대한 retriever 생성
    if mode == "웹 검색":
        retriever, all_docs = get_search_docs(user_input, embedding_model, serper_key)
    elif mode == "PDF 파일":
        retriever, all_docs = get_pdf_docs(pdf_file, embedding_model)
    else:
        return "❌ 지원하지 않는 모드입니다.", [], []

    if not retriever:
        return "검색된 정보가 부족하거나 파일이 비어있습니다.", [], []

    docs = retriever.invoke(user_input) # 사용자 입력과 가장 유사한 문서 탐색
    context = "\n\n".join(doc.page_content for doc in docs) # 찾음 문서를 하나의 문자열 형태로 만든다.
    conversation = "\n".join(chat_history + [f"user: {user_input}"]) # 대화 내용을 추가

    # prompt template 생성
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
너는 문서를 참고하여 질문에 답변하는 AI 비서야.

[문서 내용]
{context}

[질문과 대화 내용]
{question}

[답변]
"""
    )

    prompt_text = prompt_template.format(context=context, question=conversation) # prompt 형식에 맞춰 prompt 완성
    llm = ChatOpenAI(model="gpt-4.1-mini") # llm 모델 설정
    response = llm.invoke(prompt_text) # llm 답변 생성
    return response.content, docs, [doc.metadata.get("source", "출처 없음") for doc in docs] # 결과 반환

# 🖥️ Streamlit UI
def rag_chatbot():
    st.title("🔍 실시간 검색 & PDF 기반 RAG Chatbot")

    st.sidebar.header("🔧 모드 선택")
    mode = st.sidebar.radio("질문에 참고할 소스 선택", ["웹 검색", "PDF 파일"])

    st.sidebar.header("🔐 API 설정")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serper_key = None
    pdf_file = None

    if mode == "웹 검색":
        serper_key = st.sidebar.text_input("Serper.dev API Key", type="password")
    elif mode == "PDF 파일":
        pdf_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])

    if not openai_key or (mode == "웹 검색" and not serper_key) or (mode == "PDF 파일" and not pdf_file):
        st.warning("필요한 정보를 모두 입력하거나 업로드해주세요.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = openai_key
    sys.stdout.reconfigure(encoding="utf-8")

    # 대화 내역이 있는지 확인 -> 없는 경우 chat_hisory 생성
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("질문을 입력하세요")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("답변 생성 중..."):
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 임베딩 모델 설정
                # RAG 실행 : 웹 or 문서에서 텍스트 추출 -> 임베딩 -> 벡터 DB에 저장 -> 사용자 쿼리 임베딩 -> 가장 유사한 문서 탐색 -> prompt 완성 -> llm 답변 생성
                response_text, docs_used, sources = run_RAG(
                    user_input, # 사용자 입력
                    [f'{m["role"]}: {m["content"]}' for m in st.session_state.chat_history], # 대화내역 전달 {llm or 사람: 내용} 형식
                    mode, # web search or pdf 모드 설정
                    embeddings, # 임베딩 모델
                    serper_key=serper_key,
                    pdf_file=pdf_file,
                )

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    with st.expander("📄 참고 문서"):
                        for i, (doc, src) in enumerate(zip(docs_used, sources), 1):
                            preview = doc.page_content[:300].replace("\n", " ")
                            st.markdown(f"**[{i}]** [{src}]({src})\n\n{preview}...")

                # chat history에 사용자 입력, LLM 답변 추가
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

    if st.button("🔁 대화 리셋"):
        st.session_state.chat_history = []
        st.rerun()
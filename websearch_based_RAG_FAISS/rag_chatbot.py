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
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    response = requests.post("https://google.serper.dev/search", json=payload, headers=headers)
    results = response.json().get("organic", [])
    return [item["link"] for item in results]

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]): tag.decompose()
        return soup.get_text(separator="\n").strip()
    except Exception:
        return ""

# 🧠 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 📘 PDF 텍스트 추출 (pypdf 사용)
def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception:
        return ""

# 📚 문서 벡터화 - 웹 기반
def get_search_docs(query, embedding_model, serper_key, k=5):
    urls = search_web(query, num_results=3, api_key=serper_key)
    docs = []
    for url in urls:
        text = extract_text_from_url(url)
        if text:
            chunks = text_splitter.split_text(text)
            docs.extend([Document(page_content=chunk, metadata={"source": url}) for chunk in chunks])
    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever, docs
    return None, []

# 📚 문서 벡터화 - PDF 기반
def get_pdf_docs(file, embedding_model, k=5):
    text = extract_text_from_pdf(file)
    if not text:
        return None, []

    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=chunk, metadata={"source": "업로드된 PDF"}) for chunk in chunks]
    if docs:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever, docs
    return None, []

# 🧠 RAG 처리
def run_RAG(user_input, chat_history, mode, embedding_model, serper_key=None, pdf_file=None):
    if mode == "웹 검색":
        retriever, all_docs = get_search_docs(user_input, embedding_model, serper_key)
    elif mode == "PDF 파일":
        retriever, all_docs = get_pdf_docs(pdf_file, embedding_model)
    else:
        return "❌ 지원하지 않는 모드입니다.", [], []

    if not retriever:
        return "검색된 정보가 부족하거나 파일이 비어있습니다.", [], []

    docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in docs)
    conversation = "\n".join(chat_history + [f"user: {user_input}"])

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

    prompt_text = prompt_template.format(context=context, question=conversation)
    llm = ChatOpenAI(model="gpt-4.1-mini")
    response = llm.invoke(prompt_text)
    return response.content, docs, [doc.metadata.get("source", "출처 없음") for doc in docs]

# 🖥️ Streamlit UI
def rag_chatbot():
    st.title("🔍 실시간 검색 & PDF 기반 RAG Chatbot (FAISS)")

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
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                response_text, docs_used, sources = run_RAG(
                    user_input,
                    [f'{m["role"]}: {m["content"]}' for m in st.session_state.chat_history],
                    mode,
                    embeddings,
                    serper_key=serper_key,
                    pdf_file=pdf_file,
                )

                with st.chat_message("assistant"):
                    st.markdown(response_text)
                    with st.expander("📄 참고 문서"):
                        for i, (doc, src) in enumerate(zip(docs_used, sources), 1):
                            preview = doc.page_content[:300].replace("\n", " ")
                            st.markdown(f"**[{i}]** [{src}]({src})\n\n{preview}...")

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")

    if st.button("🔁 대화 리셋"):
        st.session_state.chat_history = []
        st.rerun()
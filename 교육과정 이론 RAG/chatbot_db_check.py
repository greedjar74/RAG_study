from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st


def chatbot_db_check():
    # Chroma DB 경로
    db_path = "./curriculum_theory_DB"

    # Vectorstore 로드
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 저장된 모든 문서 보기
    all_data = vectorstore.get()
    all_docs = all_data['documents']
    all_ids = all_data['ids']

    # 출력
    st.text(f"저장된 문서 수: {len(all_docs)}\n")

    for i, doc in enumerate(all_docs, 1):
        st.text(f"--- 문서 {i} [id: {all_ids[i-1]}] ---")
        st.text(doc[:1500])  # 500자까지 출력

    # DB에서 문서 삭제
    delete_id = st.sidebar.text_input("삭제할 문서의 id를 입력하세요") # 삭제할 데이터 id
    if st.sidebar.button("문서 삭제"):
        if delete_id in all_ids:
            vectorstore.delete(delete_id)
            st.success(f"문서를 삭제했습니다.")
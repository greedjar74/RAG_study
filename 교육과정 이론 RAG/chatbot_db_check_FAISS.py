from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os

def chatbot_db_check():
    st.title("📚 FAISS DB 전체 문서 출력")

    # FAISS DB 경로
    db_path = "./faiss_curriculum_index"

    if not os.path.exists(db_path):
        st.error("❌ FAISS DB 폴더가 존재하지 않습니다.")
        return

    # Embedding 및 DB 로딩
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    try:
        vectorstore = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
        st.success("✅ FAISS DB 로드 완료")
    except Exception as e:
        st.error(f"❌ DB 로드 실패: {e}")
        return

    # 전체 문서를 모두 출력하기 위해 FAISS 내부 저장소에서 직접 접근
    # -> _index_to_docstore_id, _docstore 등의 내부 속성 사용
    doc_ids = list(vectorstore.index_to_docstore_id.values())
    docs = [vectorstore.docstore._dict[doc_id] for doc_id in doc_ids]

    st.text(f"📄 저장된 문서 수: {len(docs)}")

    for i, doc in enumerate(docs, 1):
        st.markdown(f"### 📄 문서 {i}")
        st.text(doc.page_content[:1500])  # 최대 1500자 출력

    # 문서 삭제 (선택적 기능)
    delete_id = st.sidebar.text_input("삭제할 문서의 내부 ID 입력")
    if st.sidebar.button("문서 삭제"):
        try:
            vectorstore.delete([delete_id])
            vectorstore.save_local(db_path)
            st.success(f"✅ 문서 {delete_id} 삭제 완료")
        except Exception as e:
            st.error(f"❌ 삭제 중 오류 발생: {e}")

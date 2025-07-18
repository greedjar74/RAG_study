# 면접 예상 질문 생성

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS

from funcs.get_combined_docs import get_combined_docs

# 면접 질문 생성
def generate_interview_questions(company_name, pdf_file, embedding_model, serper_key):
    retriever, all_docs = get_combined_docs(company_name, pdf_file, embedding_model, serper_key)

    if not retriever:
        return "❌ 정보가 충분하지 않습니다.", [], []

    question = f"{company_name} 기업 면접을 준비 중이야. 면접 예상 질문을 만들어줘." # 입력한 기업 이름을 기반으로 질문 완성
    docs = retriever.invoke(question) # 관련 문서 탐색
    context = "\n\n".join(f"[출처: {doc.metadata.get('source', '출처없음')}] \n {doc.page_content}" for doc in docs) # 찾은 문서를 하나의 문자열으로 만들어준다.

    # prompt template 설정
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
너는 인사담당자처럼 면접 질문을 만들어주는 AI야.

[문서 내용]
{context}

[요청]
{question}

[면접 예상 질문]
"""
    )

    prompt_text = prompt_template.format(context=context, question=question) # prompt 완성
    llm = ChatOpenAI(model="gpt-4.1-mini") # llm 모델 설정
    response = llm.invoke(prompt_text) # 답변 생성
    return response.content, docs, [doc.metadata.get("source", "출처 없음") for doc in docs] # 결과 반환
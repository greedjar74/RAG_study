import os
from pypdf import PdfReader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

os.environ['OPENAI_API_KEY'] = ""

def extract_text_pdf(path):
    text = ""
    reader = PdfReader(path)

    for page in reader.pages:
        text += f'{page.extract_text()} \n' or '\n'

    return text

text = extract_text_pdf("")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model='gpt-4.1-mini')

prompt = PromptTemplate(
    input_variables=['context', 'query'],
    template="""
너는 문서를 참고하여 정확한 답변을 생성한다.

[문서 내용]
{context}

[사용자 질문]
{query}

[답변]
"""
)

user_input = input("질문을 입력하세요 : ")

docs = retriever.invoke(user_input)
context = '\n'.join(doc.page_content for doc in docs)
prompt_text = prompt.format(context=context, query=f'user: {user_input}')
response = llm.invoke(prompt_text)

print(response.content)
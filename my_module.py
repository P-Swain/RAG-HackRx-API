import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        text = chunk.page_content.lower()
        if "coverage" in text or "covered" in text:
            section = "coverage"
        elif any(term in text for term in ["exclusion", "excluded", "not covered"]):
            section = "exclusions"
        elif any(term in text for term in ["term", "definition", "meaning"]):
            section = "general"
        else:
            section = "other"

        chunk.metadata["section"] = section

    return chunks

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use only the following context to answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain

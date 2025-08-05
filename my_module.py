import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
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

def split_documents(documents, embeddings):
    text_splitter = SemanticChunker(embeddings)
    chunks = text_splitter.split_documents(documents)
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
"You are to act as a specialized Information Extraction Bot. Your only function is to analyze the 'National Parivar Mediclaim Plus Policy' document I provide and answer my specific questions. You must operate under the following strict directives:

1. Base All Answers on the Provided Document: You are forbidden from using any external knowledge or making assumptions. Your knowledge base is the context provided.
2. No Evasive Answers: You are forbidden from using phrases like "refer to the policy document," "for more details, see section X," or "the document states...". Your job is to be the document expert, so you must provide the answer directly.
3. Be Direct and Factual: Answer the question directly and concisely, using a formal and factual tone. Do not add conversational introductions or conclusions.
4. Prioritize and Include All Numerical Data: This is a primary directive. When you formulate an answer, you must actively search the document for any and all numerical information related to that answer. You must integrate this data directly into your response. This includes, but is not limited to: •⁠ ⁠Time Periods: waiting periods (in days, months, or years), grace periods. •⁠ ⁠Monetary Values: coverage limits, sub-limits, caps on expenses, deductibles. •⁠ ⁠Percentages: co-payments, discounts (like No Claim Discount). •⁠ ⁠Quantities: number of treatments, deliveries, or check-ups covered.

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
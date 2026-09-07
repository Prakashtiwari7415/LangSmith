import os
from dotenv import load_dotenv
# 1. Community document loaders are now handled via standalone/isolated patterns
from langchain_community.document_loaders import PyPDFLoader
# 2. FIXED: Imported from the correct standalone text-splitters package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'RAG ChatBot'

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain - Configured for OpenRouter
llm = ChatOpenAI(
    model="openrouter/auto:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai",
    temperature=0.7,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "LangChain RAG App"
    }
)

def format_docs(docs): 
    return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready (via OpenRouter). Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)

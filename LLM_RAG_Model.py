import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

class LLMRAGModel:
    def __init__(self, llm_name="google/flan-t5-base", retriever_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Loading a smaller model for CPU
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        self.llm_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # Use CPU
            max_new_tokens=250,
            #temperature=0.8,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)

    def get_new_chain(self):
        prompt = self.get_prompt_from_template()
        memory = ConversationBufferMemory(input_key="question", memory_key="history", max_length=5)
        retriever = self.build_retrieval()  # Ensure retriever is correctly initialized
        llm_chain = LLMChain(prompt=prompt, llm=self.llm, verbose=True, memory=memory)
        return llm_chain, retriever

    def get_prompt_from_template(self):
        system_prompt = """You are a helpful assistant. Use the provided context and history to answer questions.
                            If you cannot find the answer in the context, inform the user. Provide detailed and accurate answers. 
                            Try to explain concepts in a comprehensive and elaborated manner."""
        
        instruction = """History: {history}\nContext: {context}\nUser: {question}"""

        prompt_template = system_prompt + "\n" + instruction
        return PromptTemplate(input_variables=["history", "question", "context"], template=prompt_template)

    def build_retrieval(self, model_name="sentence-transformers/all-MiniLM-L6-v2", text_dir="C:/Users/Awais/Desktop/Assignment 3/Data"):
        all_docs = []
        for filename in os.listdir(text_dir):
            file_path = os.path.join(text_dir, filename)
            loader = TextLoader(file_path)
            all_docs.extend(loader.load())
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=". ")
        texts = text_splitter.split_documents(all_docs)
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()  
        return retriever
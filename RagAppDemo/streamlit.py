import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler

from css_template import css, bot_template, user_template
from setting import Setting
from typing import Optional, Type
import os
import hmac

import qdrant_client

# Initiate parameters
# ! Please set your parameters in 'env/local.yaml' before running the code. 

os.environ['WHICH_CONFIG'] = 'local.yaml'
config = Setting()

BASE_URL = config["BASE_URL"]
API_KEY = config["API_KEY"]
API_VERSION = config["API_VERSION"]
DEPLOYMENT_NAME = config["DEPLOYMENT_NAME"]

os.environ['AZURE_OPENAI_API_KEY'] = API_KEY
os.environ['AZURE_OPENAI_ENDPOINT'] = BASE_URL

admin_name = config["credentials"]['username']
admin_password = config["credentials"]['password']

os.environ["TAVILY_API_KEY"] = config['TAVILY_API_KEY']

# Logging in Langsmith for usage tracking (optional)
# os.environ["LANGCHAIN_TRACING_V2"] = config['LANGCHAIN_TRACING_V2']
# os.environ["LANGCHAIN_PROJECT"] = config['LANGCHAIN_PROJECT']
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"]

# qdrant credential
QDRANT_URL = config["QDRANT_URL"]
QDRANT_API_KEY = config["QDRANT_API_KEY"]

#initiate LLM model
llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=DEPLOYMENT_NAME,
        azure_endpoint=BASE_URL,
        openai_api_version=API_VERSION,
        openai_api_key=API_KEY,
        streaming=True
    )

# model_path = "llama.cpp/models/llama-2-7b-chat/llama-2_q4.gguf"
# llm = LlamaCpp(
#     model_path=model_path,
#     n_gpu_layers=100,
#     n_batch=512,
#     n_ctx=2048,
#     f16_kv=True,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     verbose=True,
# )



def get_similar_content(question, retriever):

    """
    Retrieve the top n document similar to the question
    """
    top_similar_document = retriever.get_relevant_documents(str(question))

    return format_doc(top_similar_document)

def format_doc(documents):
    most_similar_content = ""
    for result in documents:
        doc_content = result.dict()['page_content']
        doc_source = result.dict()['metadata']['source']

        most_similar_content += '>>> Document Source: '
        most_similar_content += doc_source
        most_similar_content += '\n'
        most_similar_content += 'Document Content: '
        most_similar_content += doc_content
        most_similar_content += '\n\n'

    return most_similar_content

def format_tavily_res(tavily_res):
    formatted_result = ""
    try:
        for result in tavily_res:
            doc_content = result['content']
            doc_source = result['url']

            formatted_result += '>> Document Content: '
            formatted_result += doc_content
            formatted_result += ' | Document Source: '
            formatted_result += doc_source
            formatted_result += '\n\n'
    except:
        formatted_result = str(tavily_res)
    return formatted_result

def read_PDF_doc(uploaded_pdf_files):
    """
    Extract and read all the text in PDF
    """
    text = ""
    for file in uploaded_pdf_files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)


    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] == admin_name and hmac.compare_digest(
            st.session_state["password"],
            admin_password,
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True
    
    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False
    
def display_chat(user_query):
    res = st.session_state.conversation({'question':user_query})
    st.session_state.chat_history = res['chat_history']
    for idx, msg in enumerate(st.session_state.chat_history):
        if idx % 2 == 0:
            st.write(user_template.replace("{{message}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{message}}", msg.content), unsafe_allow_html=True)


class basicRAG():
    def __init__(self, context_text, query) -> None:
        self.template = """
        Answer the question the best you can based on the provided context, if you cannot find related information from the context that is helpful to answer the question, say you do not have enough information to answer. Otherwise, you should provide your answer with the citation source or link.

        ----
        Answer the question based only on the following context:
        {context}
        
        ----
        Question: {question}.
        Your answer: 
        """
        prompt_template = PromptTemplate.from_template(self.template)
        self.prompt = prompt_template.format(context=context_text, question=query)

class ChainOfThoughtAgent():
    def __init__(self, tools, llm) -> None:

        self.template = '''
        You are the expert in gathering and analayzing news/ article or any text information.
        Answer the following questions as best you can. In your final answer, You should provide the citation of the source and the link if you have utilized any of them obtained from the tools.
        If you cannot find useful or relevant information from the tools, you can use your own knowledge to answer, however you must state that your answer doesn't use the information from the tools. If you are not able to answer the question, your Final Answer should be 'Sorry, I am not able to answer using the current set of tools.'

        You have access to the following tools:
        {tools}

        Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input question to the action, you can ask different question using the tool in next iteration if it is helpful for you to answer the main question from the user input
        Observation: the response or text obtained from the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        You must avoid repeating the same Action using the same Action Input.

        Begin!
        Question: {input}
        Thought:{agent_scratchpad}'''
        self.tools = tools
        self.llm = llm
        self.prompt = PromptTemplate.from_template(self.template)
        self.search_agent = create_react_agent(self.llm,self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.search_agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_iterations=4
        ) 


def main():
    st.write(css, unsafe_allow_html=True)

    # # load LLM model of your choice
    llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=DEPLOYMENT_NAME,
            azure_endpoint=BASE_URL,
            openai_api_version=API_VERSION,
            openai_api_key=API_KEY,
            streaming=True
        )
    

    # method to split the text into chunk
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " "],
        chunk_size=3000,
        chunk_overlap=200,
        length_function =len
    )

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'loaded_vector_store' not in st.session_state:
        st.session_state.loaded_vector_store = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 5

    st.header("News Portal Assistant - Retrieval-Augmented Generation (RAG)")
    newspaper_image = "newspaper.jpeg"
    st.image(newspaper_image, use_column_width=True)
    user_query = st.text_input("Ask a question:")
    enable_online_search = None

    with st.sidebar:
        st.subheader("Set the parameters")
        st.session_state.top_p = st.sidebar.slider("top_p docs for retrieval", 1, 10, 5)
        
        selected_method = st.selectbox('Select the prompt method', ['Without RAG','Basic RAG','ReAct (Reason-Action) with Agent'])
        if selected_method == 'ReAct (Reason-Action) with Agent':
            enable_online_search = st.checkbox("Enable Online Search (Tavily)", value=False)
            st.write("Remark: Due to the limited free quota of TavilySearch, please disable this if the Q&A has issue.")

        st.subheader("Store Information to database")
        # here allows uploading new doc for indexing, disable this for this project for cost management
        # uploaded_pdf_files = st.file_uploader("Upload your PDF document Here.", accept_multiple_files=False, type='pdf')
        url = st.text_input("Input an URL")
        
        if st.button("Add to vector store"):
            with st.spinner("Processing..."):
                # get the content in pdf
                # if uploaded_pdf_files:
                #     data = read_PDF_doc(uploaded_pdf_files)
                #     text_chunks = text_splitter.split_documents(data)

                if url:
                    loaders = UnstructuredURLLoader(urls=[url])
                    data = loaders.load()
                    new_documents = text_splitter.split_documents(data)
                    st.write('Below are the chucked documents to be stored in vector store.')
                    st.write(new_documents)

                    embedding_model = AzureOpenAIEmbeddings(model="text-embedding-3-small",
                                    azure_endpoint=BASE_URL, 
                                    deployment="text-embedding-3-small", 
                                    openai_api_key=API_KEY,
                                    openai_api_version=API_VERSION)
                    
                    # update the vectore stores with new information
                    vector_store = Qdrant.from_documents(
                    new_documents, embedding_model, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name="news"
                    )
                    st.session_state.loaded_vector_store = vector_store
                    st.session_state.retriever = st.session_state.loaded_vector_store.as_retriever(search_kwargs={"k":st.session_state.top_p})
                else:
                    st.write('Invalid data source.')

                

    if st.button("Submit question"):

        # load vectore db 
        client = qdrant_client.QdrantClient(
        QDRANT_URL,
        api_key=QDRANT_API_KEY,
        )

        loaded_vector_store = Qdrant(
            client=client, collection_name="news", 
            embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-small",
                                                    azure_endpoint=BASE_URL, 
                                                    deployment="text-embedding-3-small", 
                                                    openai_api_key=API_KEY,
                                                    openai_api_version=API_VERSION),
        )

        st.session_state.loaded_vector_store = loaded_vector_store
        st.session_state.retriever = st.session_state.loaded_vector_store.as_retriever(search_kwargs={"k":st.session_state.top_p})

        # create custom retriever tool
        class SearchInput(BaseModel):
            query: str = Field(description="should be a search query")

        class CustomRetrieverTool(BaseTool):
            name = "search_private_db"
            description = "you must use this when the user input contains @search_private_db. useful when you need to find information from the private database"
            args_schema: Type[BaseModel] = SearchInput

            def _run(
                self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> str:
                """Use the tool to retrieve information from the private database."""
                try:
                    #find similar text
                    print('running private search...')
                    similar_content = get_similar_content(query, st.session_state.retriever)
                    print('obtained similar content.')
                    response = f"Here are the top relevant content found in the private database:\n {str(similar_content)}"
                    
                    return response

                    return self.api_wrapper.results(
                        query,
                    )
                except Exception as e:
                    return repr(e)
                
            async def _arun(
                self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
            ) -> str:
                """Use the tool asynchronously."""
                raise NotImplementedError("custom_search does not support async")

        def ask_tavily(query):

            tavily = TavilySearchResults(max_results=2, name='tavily_search')
            tavily_response = tavily(query)
            return format_tavily_res(tavily_response)

        tavily_tool = Tool(
        name="tavily_search",
        func=ask_tavily,
        description="A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
        )
        
        retriver_tool = CustomRetrieverTool()
        pubmed_tool = PubmedQueryRun(description="A wrapper around PubMed. Useful for when you need to answer questions about medicine, health, and biomedical topics from biomedical literature, MEDLINE, life science journals, and online books. Only use this tool when the query is related to medical. Input should be a search query.")

        if selected_method == 'Without RAG':
            response = llm.invoke(input=user_query)
            try:
                response = response.content
                st.write(bot_template.replace("{{message}}", response), unsafe_allow_html=True)
            except:
                response = 'Invalid Output. Please try again.'

        elif selected_method == 'Basic RAG':
            
            similar_content = get_similar_content(user_query, st.session_state.retriever)
            rag = basicRAG(context_text=similar_content, query=user_query)
            response = llm.invoke(input=rag.prompt)
            try:
                response = response.content
                st.write(bot_template.replace("{{message}}", response), unsafe_allow_html=True)
                st.write('Below Documents are retrieved: ')
                st.write(similar_content)
            except:
                response = 'Invalid Output. Please try again.'
                st.write(response)
            
        elif selected_method == 'ReAct (Reason-Action) with Agent':

            if enable_online_search == True:
                tools = [tavily_tool, retriver_tool, pubmed_tool]
            else:
                tools = [retriver_tool, pubmed_tool]
        
            cot_agent_executor = ChainOfThoughtAgent(tools, llm).agent_executor

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = cot_agent_executor({'input':user_query}, callbacks=[st_callback])
                st.write(bot_template.replace("{{message}}", response["output"]), unsafe_allow_html=True)


if __name__ == '__main__':
    if not check_password():
        st.stop()
    main()
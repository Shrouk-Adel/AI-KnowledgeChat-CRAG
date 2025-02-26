import os 
import getpass
from fastapi import FastAPI
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    TAVILY_API_KEY = getpass.getpass("Tavily API key:\n")

os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ['LangChain_API_Key']=os.getenv('LangChain_API_Key')
os.environ['LangChain_Project']=os.getenv('LangChain_Project')
os.environ['Langchain_Tracing'] ='true'
groq_api_key = os.getenv('Groq_API_Key')

app =FastAPI()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


urls =[
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs =[WebBaseLoader(web_path=url).load() for url in urls]
docs_items =[item for sublist in docs for item in sublist]

text_spliter =RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250,chunk_overlap=0)
doc_split =text_spliter.split_documents(docs_items)

# Add vectordb
vectorestore =Chroma.from_documents(
    documents=doc_split,
    collection_name='rag_chroma',
    embedding=OllamaEmbeddings(model='nomic-embed-text')
)

retriver =vectorestore.as_retriever(k=10)


# retrival grader
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_groq import ChatGroq

class GradDocument(BaseModel):
    binary_score:str =Field(
        description='Documents are relevant to the question are ,yes or no'
    )

llm =ChatGroq(
    groq_api_key=groq_api_key ,
    model ='qwen-2.5-32b',
    temperature=0
    )

structured_llm_grader =llm.with_structured_output(GradDocument)

# prompt 
system ='''you are grader assesing relevance of of retrived document to user question\n
if the document contains keywords or semantic meaning related to the question grad it is relevant\n
Give a binary scroe 'yes' or 'no' score to indicate whether the document is relevant to the question.'''

grad_prompt =ChatPromptTemplate.from_messages([
    ('system',system),
    ('human',"Retrieved document: \n\n {document} \n\n User question: {question}")
])

retrival_grader_chain=(
    grad_prompt
    |structured_llm_grader
)

# invoke chain 
question ='agent memory'
docs =retriver.invoke(question)
doc_txt =docs[1].page_content

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
prompt =hub.pull('rlm/rag-prompt')

# post processing 
def formate_documnets(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# chain 
rag_chain =(
    prompt|llm|StrOutputParser()
)


# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()



from langchain_community.tools import TavilySearchResults
web_search_tool = TavilySearchResults(max_results=5)

from typing import List 
from typing_extensions import TypedDict

class GraphState(TypedDict):
    documents:List[str]
    generation:str
    web_search:str
    question:str

from langchain.schema import Document

def retrive(state):
    print('------Retriver-----')
    question=state['question']
    docs =retriver.get_relevant_documents(question)
    return {'documents':docs,'question':question}


def generate(state):
    print('----------generate------------')
    question= state['question']
    documents =state['documents']
    generation =rag_chain.invoke({'context':documents,'question':question})

    return {'documents':documents,'question':question,'generation':generation}

def document_grader(state):
    print('--check document relevant to the question--')
    question = state['question']
    documents = state['documents']

    filtered_docs = []
    for d in documents:
        score = retrival_grader_chain.invoke({'question': question, 'document': d.page_content})
        if score.binary_score.lower() == 'yes':
            print('---document is relevant')
            filtered_docs.append(d)
        else:
            print('--document is not relevant--')

    # If none are relevant, set web_search
    web_search = 'Yes' if len(filtered_docs) < len(documents) else 'No'

    return {
        'documents': filtered_docs,
        'question': question,
        'web_search': web_search
    }



def transform_query(state):

    print('----Transform Query---')
    question =state['question']
    documents =state['documents']

    better_question =question_rewriter.invoke({'question':question})
    return {'question':better_question,'documents':documents}


 

# First, modify the web_search function to properly handle the search results
def web_search(state):
    """
    Perform web search using Tavily API and add results to documents.
    """
    print('---PERFORMING WEB SEARCH---')
    
    # Validate 'question'
    question = state.get('question', '').strip()

    if not question:
        print("--- ERROR: Question is empty or missing ---")
        return {'documents': state.get('documents', []), 'question': question}

    documents = state.get('documents', [])


    # Get search results from Tavily
    try:
        search_results = web_search_tool.invoke(question)
        # print("Search Results:", search_results)

        # Convert search results to Document objects
        new_documents = []
        for result in search_results:
            if isinstance(result, dict) and 'content' in result:
                web_result = Document(
                    page_content=result['content'],
                    metadata={'source': 'web_search'}
                )
                new_documents.append(web_result)

        # Combine existing documents with new web search results
        all_documents = documents + new_documents
        
        print(f'---FOUND {len(new_documents)} WEB SEARCH RESULTS---')
        return {'documents': all_documents, 'question': question}

    except Exception as e:
        print(f'---WEB SEARCH ERROR: {str(e)}---')
        return {'documents': documents, 'question': question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


from langgraph.graph import END,StateGraph,START

workflow =StateGraph(GraphState)

# Define Nodes
workflow.add_node('retrive',retrive)
workflow.add_node('generate',generate)
workflow.add_node('WebSearch_node',web_search)
workflow.add_node('transform_query',transform_query)
workflow.add_node('document_grader',document_grader)

# build Graph
workflow.add_edge(START,'retrive')
workflow.add_edge('retrive','document_grader')
workflow.add_conditional_edges('document_grader',
decide_to_generate,
{
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "WebSearch_node")
workflow.add_edge("WebSearch_node", "generate")
workflow.add_edge("generate", END)


workflow_app=workflow.compile()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post('/query', response_model=QueryResponse)
async def run_workflow(request: QueryRequest):
    inputs = {'question': request.question}
    response = None
    
    # Use the compiled workflow app
    for output in workflow_app.stream(inputs):
        for key, value in output.items():
            if isinstance(value, dict) and 'generation' in value:
                response = value['generation']
    
    if response is None:
        return {"answer": "No response was generated. Please check your query."}
    
    return {"answer": response}

# Make sure your FastAPI app is properly configured for deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
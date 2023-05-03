import time  # Add this import at the beginning of your file
from flask import Flask, render_template, request, jsonify
from threading import Thread

app = Flask(__name__)

# Import the provided script
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pinecone
import openai

# Use your provided API keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "dedd1ea1-8aea-4b2e-a091-afaaabc6693e"
PINECONE_API_ENV = "us-east4-gcp"

# Initialize the document loader and text splitter
loader = UnstructuredPDFLoader("data.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

# Initialize the embeddings and document search
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "pinex"
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

# Initialize the chat models and templates
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY, max_tokens=1900)
template = "You are a helpful Moroccan work code and penal code expert, Answer the questions in detail in the language the question was asked, {documents}"
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=chat_prompt)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["question"]
        print(query)
        docs = docsearch.similarity_search(query, include_metadata=True)

        start_time = time.time()  # Add this line
        response = chain.run(documents=docs, question=query)
        print(f"Processing time: {time.time() - start_time} seconds")  # Add this line

        return jsonify(response)
    return render_template("index.html")

def run():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    server = Thread(target=run)
    server.start()

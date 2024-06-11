import os
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

persist_directory = os.getcwd() + "/data/hpo_vector_db"
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding)


llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
# Create a compression retriever using GooglePalm (Gemini Pro)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=vector_db.as_retriever(search_type="mmr")
)

# Example Usage
clinical_symptom_description = "Encephalopathy. Refractory epilepsy. Signs of demyelination. Hypotonia. Psychomotor retardation. Eczema. Anemia. Atelectasis in the right lung field. Normal levels of folic acid and vitamin B12 in plasma. Normal organic acids."
retrieved_hpo_docs = compression_retriever.get_relevant_documents(
    clinical_symptom_description
)


# Create a prompt template
template = """
You are a medical expert specializing in rare diseases.
Your task is to analyze the patient's symptoms and provide a list of possible HPO (Human Phenotype Ontology) terms. With them, you should attach the url from HPO. Always attach them in this template: Common name. HPO term. URL.

Patient Symptoms: {symptom}

Relevant HPO Information:
{hpo_context}

Most Relevant HPO Terms:
"""
prompt = PromptTemplate(
    input_variables=["symptom", "hpo_context"],
    template=template,
)

# Create an LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Combine the retrieved HPO information
hpo_context = "\n".join([doc.page_content for doc in retrieved_hpo_docs])

# --- Run the LLM Chain ---

result = llm_chain.run(symptom=clinical_symptom_description, hpo_context=hpo_context)

print(result)

import os
import getpass
import warnings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ignore specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


class HPOAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            self.api_key = getpass.getpass("Provide your Google API Key")
        os.environ["GOOGLE_API_KEY"] = self.api_key

        self.persist_directory = os.getcwd() + "/data/hpo_vector_db"
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        self.vector_db = Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embedding
        )

        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vector_db.as_retriever(search_type="mmr"),
        )

        self.prompt_template = """
        You are a medical expert specializing in rare diseases.
        Your task is to analyze the patient's symptoms and provide a list of possible HPO (Human Phenotype Ontology) terms. With them, you should attach the url from HPO. Always attach them in this template: Common name. HPO term. URL.

        Patient Symptoms: {symptom}

        Relevant HPO Information:
        {hpo_context}

        Most Relevant HPO Terms:
        """
        self.prompt = PromptTemplate(
            input_variables=["symptom", "hpo_context"],
            template=self.prompt_template,
        )
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def analyze(self, clinical_symptom_description):
        retrieved_hpo_docs = self.compression_retriever.invoke(
            clinical_symptom_description
        )
        hpo_context = "\n".join([doc.page_content for doc in retrieved_hpo_docs])
        result = self.llm_chain.invoke(
            {"symptom": clinical_symptom_description, "hpo_context": hpo_context}
        )
        return result


# Example usage
if __name__ == "__main__":
    clinical_symptom_description = input(
        "Please enter the clinical symptom description: "
    )
    analyzer = HPOAnalyzer()
    result = analyzer.analyze(clinical_symptom_description)
    print(result)

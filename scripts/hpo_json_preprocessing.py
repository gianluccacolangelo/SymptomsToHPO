##{{{
import json
import os
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Provide your Google API key here")


current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory of the current script directory
repo_root = os.path.dirname(current_script_dir)

# Construct the path to the desired data directory
data_directory = os.path.join(repo_root, "data")

with open(data_directory + "/hp.json", "r") as f:
    hpo_data = json.load(f)


def extract_hp_id(id: str) -> str:
    return id.split("/")[-1].replace("_",":")


def extract_json_hpo_data(hpo_json: [{str: str}]) -> [{str: str}]:
    hpo_terms = []
    for node in hpo_data["graphs"][0]["nodes"]:
        if "lbl" in node and "id" in node:
            term_info = {
                "id": extract_hp_id(node["id"]),
                "name": node["lbl"],
                "url": f"https://hpo.jax.org/browse/term/{extract_hp_id(node["id"])}"
            }

            if (
                "meta" in node
                and "definition" in node["meta"]
                and "val" in node["meta"]["definition"]
            ):
                term_info["description"] = node["meta"]["definition"]["val"]

            hpo_terms.append(term_info)
    return hpo_terms


hpo_terms = extract_json_hpo_data(hpo_data)

hpo_documents = []

for term in hpo_terms:
    content = f"{term['name']}\n{term.get('description', '')}"  # Combine name and description
    metadata = {'hpo_id': term['id'],'url': term['url']}
    doc = Document(page_content=content, metadata=metadata)
    hpo_documents.append(doc)


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = Chroma.from_documents(hpo_documents, embedding_model)

persist_directory = data_directory + "/hpo_vector_db"  # Choose a directory to store your database
vector_db = Chroma.from_documents(hpo_documents, embedding_model, persist_directory=persist_directory)
vector_db.persist()

##}}

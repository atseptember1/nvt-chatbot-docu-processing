import logging
import os
import requests
import time

from uuid import uuid4
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.functions import InputStream
from .utils import *

logging.basicConfig(format='%(asctime)s - [%(levelname)s]: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

batch_size = 75
embedder = AzureOpenAIEmbeddings(deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"], 
                                 chunk_size=batch_size, 
                                 max_retries=2, 
                                 retry_min_seconds= 60,
                                 retry_max_seconds= 70)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

cosmoshelper = CosmosDBHelper(
    CONNECTION_STRING=os.environ.get("COSMOS_CONNECTION_STRING"),
    COSMOS_DATABASE_NAME=os.environ.get("COSMOS_DATABASE_NAME"),
    COSMOS_CONTAINER_NAME=os.environ.get("COSMOS_CONTAINER_NAME")
)

cosmoshelper.initialize_db()

def main(myblob: InputStream):
    
    filename = myblob.name.split("/")[1]
    fileurl = myblob.uri
    fileurl_for_formrecognizer = fileurl + "?" + os.environ.get("BLOB_CONTAINER_SAS") if os.environ.get("BLOB_CONTAINER_SAS")[0] != "?" else fileurl + os.environ.get("BLOB_CONTAINER_SAS")

    logging.warning("Document Processing Begin")
    logging.warning(f"Processing: {fileurl}")

    document_id = str(uuid4())
    document_processing_status = cosmoshelper.create_item(
        id=document_id,
        document_name=filename,
        document_url=myblob.uri,
        status="Processing"
    )

    page_map = parse_pdf(file=fileurl_for_formrecognizer, from_url=True)
    # contents = [page[2] for page in page_map]
    # chunk_vectors = embedder.embed_documents(contents)

    upload_payload = {"value": []}

    try:

        for i, page in enumerate(page_map):
            try:
                page_num = page[0] + 1
                page_content = page[2]
                # page_id = f"{document_id}-page-{str(page_num).zfill(5)}"
                page_chunks = text_splitter.create_documents([page_content])

                for chunk_id, page_chunk_content in enumerate(page_chunks):
                    page_chunk_id = f"{document_id}-page-{str(page_num).zfill(5)}-chunk-{str(chunk_id).zfill(5)}"

                    payload = {
                        "@search.action": "upload",
                        "id": page_chunk_id,
                        "title": f"{filename}_page_{str(page_num)}_chunk_{str(chunk_id)}",
                        "chunk": page_content,
                        "chunkVector": embedder.embed_query(page_chunk_content.page_content),
                        "name": filename,
                        "location": fileurl,
                        "page_num": page_num,
                    }

                    upload_payload["value"].append(payload)

                logging.warning(f"File: {filename} - Processing: {page_num}/{len(page_map)} - {round(page_num/len(page_map)*100)}%")
            
            except Exception as e:
                logging.error(f"------------------- Exception happened at page: {page_num}-------------------")
                logging.error("Exception:", e)
                continue

        r = requests.post(url=os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + os.environ['AZURE_SEARCH_INDEX'] + "/docs/index",
                          data=json.dumps(upload_payload), 
                          headers=headers, 
                          params=params)

        cosmoshelper.update_item(
            status_info=document_processing_status,
            pages=len(page_map),
            status="Completed"
        )

        logging.warning("Document Processing Ends - Successfully")
    
    except Exception as e:
        logging.error(e)

        cosmoshelper.update_item(
            status_info=document_processing_status,
            status="Failed",
            error=e
        )

        logging.info("Document Processing Ends - Failed")
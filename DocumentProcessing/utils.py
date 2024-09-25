import re
import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests
import logging

from collections import OrderedDict
from copy import deepcopy
import base64

# import docx2txt
import html
import time
# import openai
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
from datetime import datetime

# import docx
# from docx.table import Table
# from docx.text.paragraph import Paragraph

class CosmosDBHelper:

    def __init__(self, CONNECTION_STRING, COSMOS_DATABASE_NAME, COSMOS_CONTAINER_NAME):
        self.CONNECTION_STRING = CONNECTION_STRING
        self.COSMOS_DATABASE_NAME = COSMOS_DATABASE_NAME
        self.COSMOS_CONTAINER_NAME = COSMOS_CONTAINER_NAME
        self.client, self.database, self.container = None, None, None

    def initialize_db(self):
        self.client = CosmosClient.from_connection_string(conn_str=self.CONNECTION_STRING)
        self.database = self.client.create_database_if_not_exists(self.COSMOS_DATABASE_NAME)
        self.container = self.database.create_container_if_not_exists(
            self.COSMOS_CONTAINER_NAME,
            partition_key=PartitionKey("/id"),
        )
    
    def _get_template(self):
        template = {
            "id": "",
            "document_name": "",    
            "document_url": "",
            "pages": "",
            "status": "", #Processing, Completed, Failed
            "error": "",
            "updated_at": ""
        }
        return deepcopy(template)
    
    def create_item(self, id, document_name, document_url, status):
        
        status_info = self._get_template()
        status_info["id"] = id
        status_info["document_name"] = document_name
        status_info["document_url"] = document_url
        status_info["status"] = status
        status_info["updated_at"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        self.container.create_item(body=status_info)

        return status_info
    
    def update_item(self, status_info, pages=None, document_name=None, document_url=None, status=None, error=None):
        if document_name:
            status_info["document_name"] = document_name
        if document_url:
            status_info["document_url"] = document_url
        if pages:
            status_info["pages"] = pages
        if status:
            status_info["status"] = status
        if error:
            status_info["error"] = error

        status_info["updated_at"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
        self.container.replace_item(item=status_info["id"], body=status_info)
        
        return status_info

    def delete_item(self):
        # TODO
        pass

    def get_item(self):
        # TODO
        pass


def text_to_base64(text):
    # Convert text to bytes using UTF-8 encoding
    bytes_data = text.encode('utf-8')

    # Perform Base64 encoding
    base64_encoded = base64.b64encode(bytes_data)

    # Convert the result back to a UTF-8 string representation
    base64_text = base64_encoded.decode('utf-8')

    return base64_text


def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def parse_pdf(file, model="prebuilt-document", from_url=False):
    """Parses PDFs using PyPDF or Azure Document Intelligence SDK (former Azure Form Recognizer)"""
    offset = 0
    page_map = []
    
    credential = AzureKeyCredential(os.environ.get("FORM_RECOGNIZER_KEY"))
    form_recognizer_client = DocumentAnalysisClient(endpoint=os.environ.get("FORM_RECOGNIZER_ENDPOINT"), credential=credential)
    
    if not from_url:
        with open(file, "rb") as filename:
            poller = form_recognizer_client.begin_analyze_document(model, document = filename)
    else:
        poller = form_recognizer_client.begin_analyze_document_from_url(model, document_url = file)
        
    form_recognizer_results = poller.result()

    for page_num, page in enumerate(form_recognizer_results.pages):
        logging.warning(f"File: {file} - Processing: {page_num+1}/{len(form_recognizer_results.pages)}")

        tables_on_page = [table for table in form_recognizer_results.tables if table.bounding_regions[0].page_number == page_num + 1]

        # mark all positions of the table spans in the page
        page_offset = page.spans[0].offset
        page_length = page.spans[0].length
        table_chars = [-1]*page_length
        for table_id, table in enumerate(tables_on_page):
            for span in table.spans:
                # replace all table spans with "table_id" in table_chars array
                for i in range(span.length):
                    idx = span.offset - page_offset + i
                    if idx >=0 and idx < page_length:
                        table_chars[idx] = table_id

        # build page text by replacing charcters in table spans with table html
        page_text = ""
        added_tables = set()
        for idx, table_id in enumerate(table_chars):
            if table_id == -1:
                page_text += form_recognizer_results.content[page_offset + idx]
            elif not table_id in added_tables:
                page_text += table_to_html(tables_on_page[table_id])
                added_tables.add(table_id)

        page_text += " "
        page_map.append((page_num, offset, page_text))
        offset += len(page_text)

    return page_map

def docx_table_to_html(table):
    table_html = "<table>"
    for i, row in enumerate(table.rows):
        table_html += "<tr>"
        text = [cell.text for cell in row.cells]
        for c in text:
            table_html += "<th>" if i == 0 else "<td>"
            table_html += c.replace("\xa0", "")
            table_html += "</th>" if i == 0 else "</td>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

# def parse_docx(file=None, url=None, step_size=2500):
#     if url:
#         response = requests.get(url)
#         file = BytesIO(response.content)
#         doc = docx.Document(file)
#     elif file:
#         doc = docx.Document(file)
#     else:
#         raise ValueError("either file or url must have a valid value")

#     document_content = ""
#     for part in doc.iter_inner_content():
#         if isinstance(part, Paragraph):
#             document_content += part.text
#         elif isinstance(part, Table):
#             document_content += docx_table_to_html(part)
#         document_content += "\n"

#     num_steps = len(document_content) // step_size

#     page_map = []
#     for i in range(num_steps):
#         content_chunk = document_content[i*step_size:(i+1)*step_size]
#         page_map.append((None, i*step_size, content_chunk))
        
#     return page_map

def read_pdf_files(files, form_recognizer=False, verbose=False, formrecognizer_endpoint=None, formrecognizerkey=None):
    """This function will go through pdf and extract and return list of page texts (chunks)."""
    text_list = []
    sources_list = []
    for file in files:
        page_map = parse_pdf(file, form_recognizer=form_recognizer, verbose=verbose, formrecognizer_endpoint=formrecognizer_endpoint, formrecognizerkey=formrecognizerkey)
        for page in enumerate(page_map):
            text_list.append(page[1][2])
            sources_list.append(file.name + "_page_"+str(page[1][0]+1))
    return [text_list,sources_list]
    
    
# def parse_docx(file: BytesIO) -> str:
#     text = docx2txt.process(file)
#     # Remove multiple newlines
#     text = re.sub(r"\n\s*\n", "\n\n", text)
#     return text


def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])



# def summarize_doc(document_map):
#     document_content = "\n".join([p[2] for p in document_map])

#     system_prompt = """
#     # Instruction
#     - You are a professional document summarizer.
#     - Your job is to understand the document content and summarize the content based user requirements
#     - Using the sample language as the provided document to write the summary
#     """

#     docu_summarization_command = f"""
#     - Write a concise summary in less than 10 sentences of the following document content delimited by triple backsticks, use the same language as the content inside the triple backsticks
#     ```{document_content}```
#     """

#     message_for_summarization = [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": docu_summarization_command}
#     ]

#     completion = openai.ChatCompletion.create(
#         engine="gpt-4",
#         messages=message_for_summarization,
#         max_tokens=1000,
#         temperature=0.2,
#         n=1
#     )

#     return completion
import mysql.connector
import json
import requests
import re

import wp_config

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean
import bs4
from bs4 import BeautifulSoup


def pdebug(msg=None):
    if wp_config.DEBUG and msg:
        print("DEBUG: {}".format(msg), flush=True)
        if wp_config.DEBUG_PAUSE:
            input("Press Enter to continue...")


myconfig = {
    "user": wp_config.DB_USER,
    "password": wp_config.DB_PASSWORD,
    "host": wp_config.DB_HOST,
    "port": wp_config.DB_PORT,
    "database": wp_config.DB_SCHEMA,
}


def connectMySQL(myconfig):
    cnx = mysql.connector.connect(**myconfig)
    return cnx


# Used to format response and return references
# Used to format response and return references
class Document:

    doc_id: int
    doc_text: str

    def __init__(self, id, text) -> None:

        self.doc_id = id
        self.doc_text = text

    def __str__(self):
        return f"doc_id:{self.doc_id},doc_text:{self.doc_text}"


def generate_embeddings_for_question(question_list):

    payload = {
                   "input": question
              }

    print("Performing Embeddings of the prompt...")
    response = requests.post(wp_config.LLAMA_EMBEDDINGS_URL, json=payload)
    if response.status_code == 200:
        embeddings = response.json().get("data", [{}])[0].get("embedding", [])
        return embeddings
    else:
        print("Error:", response.status_code, response.text)
        return False


def query_llm_with_prompt(documents, prompt):

    print("Generating the Answer...")
    my_documents = "" 
    i = 1
    print("   I will use the following blog posts to generate my answer:")
    for docs in documents:
        cursor = cnx.cursor()
        cursor.execute("SELECT post_title from wp_posts where id = {}".format(docs.doc_id))
        result = cursor.fetchone()
        print("     - [{}]: {}".format(docs.doc_id, result[0]))
        my_documents = my_documents + f"Doc {i}: {docs.doc_text}\n"
        i+=1

    full_prompt = f"""
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.

    Context: {my_documents}
    
    Question: {question}
    
    Helpful Answer:"""

    payload = {
            "prompt": full_prompt,
            "max_tokens": 500,
            "temperature": 0.6,
            "top_p": 0.75,
            "stop": "\n"
    }

    llm_response_result = requests.post(wp_config.LLAMA_COMPLETIONS_URL, json=payload)
    response_text = llm_response_result.json().get("choices", [{}])[0].get("text", "").strip()
    #pdebug(response_text)

    # Extract only the answer part
    if "Helpful Answer:" in response_text:
        answer = response_text.split("Helpful Answer:", 1)[1].strip()
    else:
        answer = response_text.strip()
    answer = answer.split("Conclustion: ",1)[0].strip()
    return answer


# Find relevant records from HeatWave using Dot Product similarity.
def search_data(cursor, query_vec, list_dict_docs):

    print("Performing Vector Search Similarity...")
    myvectorStr = ",".join(str(item) for item in query_vec)
    myvectorStr = "[" + myvectorStr + "]"

    relevant_docs = []
    mydata = myvectorStr
    cursor.execute(
        """
        select distinct wp_post_id from (
          select id, wp_post_id, distance from
            (select id, wp_post_id,
                    DISTANCE_COSINE(string_to_vector(%s), vec) distance
             from {}.wp_embeddings
             order by distance limit 100) a
             where distance < 1 order by distance) b limit 10 

    """.format(
            wp_config.DB_SCHEMA
        ),
        [myvectorStr],
    )

    for row in cursor:
        id = row[0]
        #pdebug(f"id: {id}")
        result_post = []
        with connectMySQL(myconfig) as db2:
            cursor2 = db2.cursor()
            cursor2.execute(
                "SELECT post_content from wp_posts where id = {} ".format(id)
            )
            result_post = cursor2.fetchone()

        soup = BeautifulSoup(result_post[0], "html.parser")
        for element in soup(
            text=lambda text: isinstance(text, bs4.element.ProcessingInstruction)
        ):
            element.extract()
        content_text = soup.get_text()

        if len(content_text) > 0:
            content = clean(content_text, extra_whitespace=True)

        temp_dict = {id: content}
        list_dict_docs.append(temp_dict)
        doc = Document(id, content)
        # print(doc)
        relevant_docs.append(doc)


    return relevant_docs


# Perform RAG
def answer_user_question(query):

    question_list = []
    question_list.append(query)

    question_vector = generate_embeddings_for_question(question_list)

    #question_vector = embed_text_response.data.embeddings[0]


    with connectMySQL(myconfig) as db:
        cursor = db.cursor()
        list_dict_docs = []
        # query vector db to search relevant records
        similar_docs = search_data(cursor, question_vector, list_dict_docs)

        # prepare documents for the prompt
        context_documents = []
        relevant_doc_ids = []
        similar_docs_subset = []

        rerank_docs = []
        for docs in similar_docs:
            content = docs.doc_text
            rerank_docs.append(content)

        response = None
        if len(rerank_docs) > 1:
            print("Performing Reranking...")
            payload = {
                "query": query,
                "documents": rerank_docs,
                "top_n": 5,
            }
            #pdebug(payload)
            response = requests.post(wp_config.LLAMA_RERANK_URL, json=payload)

            if response.status_code == 200:
                reranked_results = response.json()
                #pdebug(f"Reranked Results: {reranked_results}")
                sorted_results = sorted(reranked_results['results'], key=lambda x: x['relevance_score'], reverse=True)
                sorted_indices = [result['index'] for result in sorted_results]
                pdebug(f"Sorted Indices: {sorted_indices}")
            else:
                print("Error:", response.status_code, response.text)

        else:
            print("No corresponding document found, using GenAI...")

        myresult = reranked_results["results"]


        rerank_docs_subset = []
        for indice in sorted_indices[:5]:
            rerank_docs_subset.append(similar_docs[int(indice)])
            #pdebug(f"indice: {indice} ====>>> {similar_docs[int(indice)]}")

        prompt_template = """
        {question} \n
        Answer: Please provide the answer based on the documents provided. If the text doesn't contain the answer, reply that the answer is not available.
        """

        prompt = prompt_template.format(question=query)

        response = query_llm_with_prompt(rerank_docs_subset, prompt)

        return response


# Main Function

cnx = connectMySQL(myconfig)
if cnx.is_connected():
    cursor = cnx.cursor()
    cursor.execute("SELECT @@version, @@version_comment")
    results = cursor.fetchone()

    print("You are now connected to {} {}".format(results[1], results[0]))

    question = input("What is your question? ")
    myanswer = answer_user_question(question)

    # print(myanswer['text']['data'])

    print(myanswer)

    cnx.close()

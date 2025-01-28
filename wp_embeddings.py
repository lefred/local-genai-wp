import mysql.connector
import time
import wp_config
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean
from bs4 import BeautifulSoup
import bs4
import requests

myconfig = {
    "user": wp_config.DB_USER,
    "password": wp_config.DB_PASSWORD,
    "host": wp_config.DB_HOST,
    "port": wp_config.DB_PORT,
    "database": wp_config.DB_SCHEMA,
}


def pdebug(msg=None):
    if wp_config.DEBUG and msg:
        print("DEBUG: {}".format(msg), flush=True)
        if wp_config.DEBUG_PAUSE:
            input("Press Enter to continue...")


def connectMySQL(myconfig):
    cnx = mysql.connector.connect(**myconfig)
    return cnx


# Main Function

cnx = connectMySQL(myconfig)
if cnx.is_connected():
    cursor = cnx.cursor()
    cursor.execute("SELECT @@version, @@version_comment")
    results = cursor.fetchone()

    print("You are now connected to {} {}".format(results[1], results[0]))
    print("Starting Embeddings....")

    # generate embedding for column post_content
    cursor.execute("SELECT ID, post_content from wp_posts WHERE post_status='publish'")
    results = cursor.fetchall()

    for row in results:
        pdebug(f"We are now at wp_post_id: {row[0]}")
        # time.sleep(0.5)
        print(".", flush=True, end="")
        inputs = []
        content = row[1]
        pdebug(content)
        content_text = BeautifulSoup(content, "html.parser")
        for pi in content_text.find_all(
            string=lambda text: isinstance(text, bs4.element.ProcessingInstruction)
        ):
            pi.extract()

        clean_html = str(content_text)
        # content_text = partition_html(text=content_text.get_text())
        try:
            content_text = partition_html(text=clean_html)
            for content_text_el in content_text:
                content = clean(content_text_el.text, extra_whitespace=True)
                inputs.append(content)

        except Exception as e:
            print(f"Error with wp_post_id: {row[0]} !")

        if len(inputs) > 1:
            for input_block in range(0, len(inputs), 96):
                block = inputs[input_block : input_block + 96]
                pdebug(block)

                payload = {
                   "input": block
                }

                response = requests.post(wp_config.LLAMA_EMBEDDINGS_URL, json=payload)
                if response.status_code == 200:
                    embeddings_data = response.json().get("data", [])
                    embeddings = [entry.get("embedding", []) for entry in embeddings_data]
                    pdebug(f"Total number of embeddings: {len(embeddings)}")
                    pdebug(f"Each embedding size: {[len(emb) for emb in embeddings]}")

                    insert_stmt = (
                        "INSERT INTO wp_embeddings(vec, wp_post_id) "
                        "VALUES (string_to_vector(%s), %s)"
                    )
                    for emb in embeddings:
                      myvectorStr = ",".join(
                            str(item) for item in list(emb)
                      )
                      myvectorStr = "[" + myvectorStr + "]"
                      data = (myvectorStr, row[0])
                      cursor.execute(insert_stmt, data)

        cnx.commit()


cnx.close()
print("\nDone with Embeddings.")

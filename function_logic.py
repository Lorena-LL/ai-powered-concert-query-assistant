from my_utils import create_embedding, ingest_text, answer_query


def process_file(path):
    with open(path, 'r') as f:
        text = f.read()

    text_embedding = create_embedding(text)
    if text_embedding is None:
        print("Sorry, I cannot ingest documents with other themes.")
        return

    ingest_text(text, path, text_embedding)


def rag_process_question(sent_querry):
    answer = answer_query(sent_querry, search_online=False)
    print(answer)


def rag_or_online_process_question(sent_querry):
    answer = answer_query(sent_querry, search_online=True)
    print(answer)
from sentence_transformers import SentenceTransformer, util
import hashlib
import numpy as np
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import os
from dotenv import load_dotenv


# Embeddings Model
sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

# NER Model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", framework="pt")

# Summarizer Model
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# answer query Model
qa_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

# ChromaDB instantiation
chroma_client = chromadb.PersistentClient(path="./concert_db")
collection = chroma_client.get_or_create_collection("concerts")

def create_embedding(text):
    concert_keywords = [
        "concert announcement",
        "music tour",
        "live tour dates",
        "artist performing live",
        "tour locations",
        "music event",
        "live show",
        "VIP concert package",
        "arena concert",
        "festival lineup",
        "tickets available",
        "on stage performance",
        "tour name",
        "concert in Europe",
        "stadium performance"
    ]
    text_embedding = sentence_transformer.encode(text, convert_to_tensor=True)
    keyword_embeddings = sentence_transformer.encode(concert_keywords, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, keyword_embeddings)
    max_score = float(similarities.max())

    threshold_val = 0.4
    if max_score > threshold_val:
        embedding_as_list = text_embedding.tolist() # I need to return the embedding, but not in the tensor form
        return embedding_as_list
    return None

# Named Entity Recognition to get all artists and all locations from text
def extract_artists_locations(text):
    # Get all artists and location names
    entities = ner_pipeline(text)
    artists = []
    locations = []

    # entity_group for artists = "PER" and locations = "LOC"
    for entity in entities:
        if entity["entity_group"] == "PER":
            artists.append(entity['word'])
        elif entity["entity_group"] == "LOC":
            locations.append(entity['word'])
    return artists, locations


def summarize_text(text, max_length=160, min_length=50):
    summary = summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def ingest_text(text, file_path, text_embedding):
    # unique id for ChromaDB
    doc_id = hashlib.md5(text.encode('utf-8')).hexdigest()
    artists, locations = extract_artists_locations(text)
    # matadata in ChromaDB doesn't allow lists
    artists = ", ".join(artists)
    locations = ", ".join(locations)
    summary = summarize_text(text)

    collection.add(
        documents=[summary],
        embeddings=[text_embedding],
        ids=[doc_id],
        metadatas=[{
            "source": file_path,
            "artists": artists,
            "locations": locations
        }]
    )
    print("Thank you for sharing! Your document has been successfully added to the database. Here is a brief summary of the data from the document:")
    print(summary)


def query_database(user_question, top_k=3, similarity_threshold=0.35):
    query_embedding = sentence_transformer.encode(user_question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )
    print(f"\nQuestion: {user_question}")

    found = False
    documents = []
    similarities = []
    for doc, score in zip(results["documents"][0], results["distances"][0]):
        similarity = 1 - score  # convert distance to similarity
        if similarity >= similarity_threshold:
            found = True
            documents.append(doc)
            similarities.append(similarity)

    if not found:
        return None
    return documents, similarities



load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def answer_query(user_question, search_online=False, top_k=3, similarity_threshold=0.35):
    result = query_database(user_question, top_k=top_k, similarity_threshold=similarity_threshold)

    if result is not None:
        documents, _ = result
        context = "\n\n".join(documents)
    elif search_online:
        web_results = search_artist_online(user_question)

        if not web_results:
            return f"Sorry, I couldn’t find online the concert information you were looking for."

        context = "\n\n".join(web_results)
    else:
        return "I am sorry, but I do not have enough information to generate an answer."

    prompt = f"""You are a helpful assistant. Based on the context, give a natural and informative answer to the user's question.

Context: {context}

Question: {user_question}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = qa_model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "The answer to your question is: " + answer



def get_all_documents_with_metadata():
    query_embedding = np.zeros(384)  # an empty embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1000,
        include=["documents", "metadatas"]
    )

    print("\n ALL DOCUMENTS FORM CHROMA DB: \n")
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        print(f"Document: {doc}")
        print(f"Metadate: {metadata}")
        print("---")


def search_artist_online(user_question, max_results=5):
    if not SERPAPI_KEY:
        raise ValueError("SerpAPI key is missing. Please set it in your .env file.")
    query = f"{user_question}"
    url = f"https://serpapi.com/search.json?q={query}&engine=google&api_key={SERPAPI_KEY}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # classic google search results (organic)
        results = data.get("organic_results", [])
        extracted = []

        for result in results[:max_results]:
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")
            extracted.append(f"{title}\n{snippet}\n{link}")

        return extracted

    except Exception as e:
        print(f"Search error: {e}")
        return None
from function_logic import process_file, rag_process_question, rag_or_online_process_question
from my_utils import get_all_documents_with_metadata


# ingest a file
process_file("./InputFiles/format1nr2.txt")

# prints the data contained in chromaDB
get_all_documents_with_metadata()

sent_querry1 = "When is Robbie Williams going to perform in Bucharest?"
sent_querry2 = "What is Dua Lipa new world tour from 2026 called?"
sent_querry3 = "Where will The Weeknd's tour take place?"
sent_querry4 = "What is the date Noah Kahan will perfom in Hyde Park, London in 2025?"  # information not available using RAG

# uses only ingested data
rag_process_question(sent_querry4)

# if the function doesn't find an answer in the ingested data, it searches online
rag_or_online_process_question(sent_querry4)
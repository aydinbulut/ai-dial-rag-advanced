from task._constants import API_KEY, EMBEDDING_DIMENSIONS
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


#TODO:
# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant designed to help users with questions related to microwave ovens.
You will be provided with relevant context from documents and user questions."""

#TODO:
# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """RAG Context:
{context}

User Question:{question}
"""


#TODO:
# - create embeddings client with 'text-embedding-3-small-1' model
embeddings_client = DialEmbeddingsClient(
    deployment_name='text-embedding-3-small-1',
    api_key=API_KEY,
)
# - create chat completion client
chat_client = DialChatCompletionClient(
    deployment_name='gpt-4o',
    api_key=API_KEY,
)
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
db_config = {
    'host': 'localhost',
    'port': 5433,
    'database': 'vectordb',
    'user': 'postgres',
    'password': 'postgres'
}
text_processor = TextProcessor(embeddings_client, db_config)
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)
def run_console_chat():
    print("🎯 Microwave RAG Assistant")
    print("="*100)

    load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
    if load_context.lower().strip() in ['y', 'yes']:
        # TODO:
        #  With `text_processor` process text file:
        #  - file_name: 'embeddings/microwave_manual.txt' 
        #  - chunk_size: 150 (or you can experiment, usually we set it as 300)
        #  - overlap: 40 (chars overlap from previous chunk)
        text_processor.process_text_file(
            file_name="./task/embeddings/microwave_manual.txt",
            chunk_size=300,
            overlap=20,
            dimensions=EMBEDDING_DIMENSIONS,
            truncate_table=True
        )
        print("Context loaded into VectorDB successfully!")

        print("="*100)


    conversation = Conversation()
    conversation.add_message(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))

    print("Welcome to the RAG-powered Microwave Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break

        # Retrieve context
        context = text_processor.search(
            search_mode=SearchMode.EUCLIDIAN_DISTANCE,
            user_request=user_input,
            top_k=10,
            min_score=0.99,
            dimensions=EMBEDDING_DIMENSIONS
        )
        print("-"*100)
        print(f"\n--- Retrieved {len(context)} Context Chunks ---")
        print("-"*100)

        # Prepare augmented user prompt
        augmented_user_prompt = USER_PROMPT.format(context=context, question=user_input)
        print("-"*100)
        print(f"\n--- Augmented User Prompt ---\n{augmented_user_prompt}\n")
        print("-"*100)
        conversation.add_message(Message(role=Role.USER, content=augmented_user_prompt))

        # Generate response
        response = chat_client.get_completion(conversation.get_messages(), print_request=False)
        conversation.add_message(Message(role=Role.AI, content=response))

        print(f"Assistant: {response.content}\n")



# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
if __name__ == "__main__":
    run_console_chat()
#  RUN docker-compose.yml
import chainlit as cl
from LLM_RAG_Model import LLMRAGModel

# Initialize the model and chain
llm_model = LLMRAGModel()
llm_chain, retriever = llm_model.get_new_chain()

@cl.on_chat_start
async def on_chat_start():
    # Set the chain and retriever for the user session
    cl.user_session.set("llm_chain", llm_chain)
    cl.user_session.set("retriever", retriever)

@cl.on_message
async def process_message(message: cl.Message):
    # Get the chain and retriever from the user session
    llm_chain = cl.user_session.get("llm_chain")
    retriever = cl.user_session.get("retriever")

    # Check if the objects are initialized
    if not llm_chain or not retriever:
        await cl.Message(content="Model or retriever not initialized.").send()
        return
    context = retriever.get_relevant_documents(message.content)
    if not context:
        await cl.Message(content="No relevant documents found for your query.").send()
        return

    # Run the LLM chain to get the response
    response = llm_chain.invoke({"question": message.content, "context": context})

    # Send the response back to the user
    await cl.Message(content=response["text"]).send()

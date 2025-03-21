from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
st.header("RAGRO 🚜")

prompt = st.text_input("Pergunta", placeholder="Digite sua pergunta aqui...")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


def create_sources_string(sources: set[str]) -> str:
    if not sources:
        return ""
    
    source_list = list(sources)
    source_list.sort()
    sources_string = "Fontes:\n"
    for i, source in enumerate(source_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Aguarde gerando resposta..."):
        generate_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set(
            [
                doc.metadata["source"]
                for doc in generate_response["source_documents"]
            ]
        )
        sources_string = create_sources_string(sources)

        formatted_response = (
            f"{generate_response['result']}\n\n{create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generate_response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"], 
        st.session_state["user_prompt_history"]
    ):
        message(user_query, is_user=True)
        message(generated_response)

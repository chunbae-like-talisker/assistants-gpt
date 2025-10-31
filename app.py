import streamlit as st
import json
from openai import OpenAI
import my_functions
import pathlib

st.set_page_config(
    page_title="Chunbae's AssistantsGPT",
    page_icon="🍔",
)

# Session
file_path = pathlib.Path("./result.txt")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "conversation_id" not in st.session_state:
    st.session_state["conversation_id"] = ""


# Common Functions
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# Sidebar
with st.sidebar:
    st.session_state["openai_api_key"] = st.text_input("OpenAI API Key")

    st.markdown(
        "[🍔 Chunbae's Repo](https://github.com/chunbae-like-talisker/assistants-gpt)"
    )

# Intro
if st.session_state["openai_api_key"] == "":
    st.markdown(
        """
    # Hello!

    Welcome to Chunbae's AssistantsGPT!

    Ask anything you’re curious about! AssistantsGPT will summarize the results and provide them in a file.

    Please start by entering your OpenAI API Key from sidebar.
    """
    )

    st.stop()

# Body
query = st.text_input("Ask anything you’re curious about!")

if not query:
    st.stop()


# Responses API
@st.cache_data(show_spinner="Thinking...")
def queryToLLM(query):
    if file_path.exists():
        file_path.unlink()

    openai = OpenAI(api_key=st.session_state["openai_api_key"])
    conversation = openai.conversations.create()
    st.session_state["conversation_id"] = conversation.id

    # LLM Functions
    def send(input):
        response = openai.responses.create(
            model="gpt-4.1-mini-2025-04-14",
            input=input,
            conversation=st.session_state["conversation_id"],
            tools=my_functions.functions,
            tool_choice="required",
        )
        print(f"\n\nResponse output: {response.output}")  # TODO: Remove before deploy
        return response

    def get_tool_output(output):
        function_name = output.name
        print(
            f"\n\nTool execute: {function_name}, Args: {output.arguments}"
        )  # TODO: Remove before deploy
        return my_functions.functions_map[function_name](json.loads(output.arguments))

    response = send(
        [
            {
                "role": "system",
                "content": """
                    You are an expert skilled at searching for and organizing information.
                    Use Wikipedia and DuckDuckGo to compile comprehensive answers to the user’s questions.

                    First, check Wikipedia for search results related to the keyword.
                    Then, use DuckDuckGo to gather a maximum of five relevant news article URLs (excluding Wikipedia) and extract their content.
                    Finally, combine all the gathered information and export it as a file using 'save_as_file' functions just once.
                """,
            },
            {"role": "user", "content": query},
        ]
    )

    while True:
        if response.output_text:
            send_message(response.output_text, "ai")

        final_labs = False
        outputs = []
        for output in response.output:
            if output.type == "function_call":
                tool_output = get_tool_output(output)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": json.dumps(tool_output),
                    }
                )

                if output.name == "save_as_file":
                    final_labs = True

        if outputs:
            response = send(outputs)

        if final_labs:
            break


send_message(query, "user")
queryToLLM(query)

if file_path.exists():
    send_message("Your file is ready! Would you like to download it now?", "ai")
    with file_path.open("rb") as f:
        st.download_button(
            label="📝 Go ahead and take this report home.",
            data=f.read(),
            file_name=file_path.name,
            mime="text",
            use_container_width=True,
        )
else:
    send_message("Something is wrong! I can't create the file...", "ai")

import os
import json
import time
import streamlit as st
from google import genai
from google.genai import types

API_KEY = st.secrets.get("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)

st.title("BartBot")

BOT_NAME = "Bartholemew"
USER_NAME = "You"

if "messages" not in st.session_state:
	st.session_state.messages = []

@st.cache_resource
def get_chat_session():
    return client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=f"Your name is {BOT_NAME}. You are a witty AI assistant.",
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )

	
chat_session = get_chat_session()

for message in st.session_state.messages:
	name = USER_NAME if message["role"] == "user" else BOT_NAME
	with st.chat_message(message["role"]):
		st.markdown(f"**{name}**: {message["content"]}")
		
if prompt := st.chat_input("What can I help you with?"):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(f"**{USER_NAME}**: {prompt}")
		
	with st.chat_message("assistant"):
		response = chat_session.send_message(prompt)
		st.markdown(f"**{BOT_NAME}**: {response.text}")
		st.session_state.messages.append({"role": "assistant", "content": response.text})
	
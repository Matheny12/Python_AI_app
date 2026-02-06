import os
import json
import time
import uuid
import streamlit as st
from google import genai
from google.genai import types

DB_FILE = "bartbot_history.json"

def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE,"r") as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_data(data):
	with open (DB_FILE, "w") as f:
		json.dump(data, f, indent=4)

API_KEY = st.secrets.get("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)

BOT_NAME = "Bartholemew"
USER_NAME = "You"

if "all_chats" not in st.session_state:
	st.session_state.all_chats = load_data()
if "active_chat_id" not in st.session_state:
	st.session_state.active_chat_id = list(st.session_state.all_chats.keys())[0] if st.session_state.all_chats else None

with st.sidebar:
	st.title("Chat History")
	if st.button("Start New Chat", use_container_width=True):
		new_id = str(uuid.uuid4())
		st.session_state.all_chats[new_id] = []
		st.session_state.active_chat_id = new_id
		save_data(st.session_state.all_chats)
		st.rerun()
	
	st.divider()

	for chat_id in reversed(list(st.session_state.all_chats.keys())):
		history = st.session_state.all_chats[chat_id]
		label = history[0]["content"][:25] + "..." if history else "New Conversation"

		if st.button(label, key=chat_id, use_container_width=True):
			st.session_state.active_chat_id = chat_id
			st.rerun()

st.title("BartBot")

if st.session_state.active_chat_id:
	current_id = st.session_state.active_chat_id
	messages = st.session_state.all_chats[current_id]
	
	def get_chat_session(history_to_send):
		return client.chats.create(
        	model="gemini-2.5-flash-lite",
        	config=types.GenerateContentConfig(
        	    system_instruction=f"Your name is {BOT_NAME}. You are a witty AI assistant.",
        	    tools=[types.Tool(google_search=types.GoogleSearch())]
        	),
			history=history_to_send
	    )
	for message in messages:
		name = USER_NAME if message["role"] == "user" else BOT_NAME
		with st.chat_message(message["role"]):
			st.markdown(f"**{name}**: {message["content"]}")
		
	if prompt := st.chat_input("What can I help you with? For image generation, start prompt with '/image'"):
		messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(f"**{USER_NAME}**: {prompt}")
	
		with st.chat_message("assistant"):
			if prompt.lower().startswith("/image"):
				try:
					with st.spinner("Bartholemew is painting..."):
						image_prompt = prompt[7:]
						response = client.models.generate_images(
							model='imagen-3.0-generate-001',
							prompt=image_prompt,
							config=types.GenerateImagesConfig(number_of_images=1)
						)

						generated_image = response.generated_images[0]
						st.image(generated_image.image.image_bytes, caption=image_prompt)

						messages.append({"role": "assistant", "content": f"Generated Image: {image_prompt}"})
						save_data(st.session_state.all_chats)
				except Exception as e:
					st.error(f"Image Error: {e}")
			else:
				formatted_history = []
				for m in messages[:-1]:
					gemini_role = "model" if m["role"] == "assistant" else "user"
					formatted_history.append({"role": gemini_role, "parts": [{"text": m["content"]}]})
				try:
					chat_session = get_chat_session(formatted_history)
					response = chat_session.send_message(prompt)
					st.markdown(f"**{BOT_NAME}**: {response.text}")
					messages.append({"role": "assistant", "content": response.text})
				except Exception as e:
					st.error(f"Error: {e}")		
else:
	st.info("Click 'Start New Chat' in the sidebar to begin!")
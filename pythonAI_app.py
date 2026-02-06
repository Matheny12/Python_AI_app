import os
import json
import time
import uuid
import base64
import streamlit as st
import extra_streamlit_components as stx
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

cookie_manager = stx.CookieManager()

def get_logged_in_user():
	if "username" in st.session_state:
		return st.session_state.username
	
	saved_user = cookie_manager.get(cookie="bartbot_user")
	if saved_user and saved_user in all_data:
		st.session_state.username = saved_user
		return saved_user
	return None

current_user = get_logged_in_user()

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

all_data = load_data()

if "username" not in st.session_state:
	st.title("Welcome to BartBot")
	tab1, tab2 = st.tabs(["Login", "Create Account"])

	with tab1:
		u_login = st.text_input("Username", key="l_user")
		p_login = st.text_input("Password", type="password", key="l_pass")
		remember_me = st.checkbox("Keep me logged in")

		if st.button("Enter BartBot"):
			if u_login in all_data and all_data[u_login].get("password") == p_login:
				st.session_state.username = u_login
				if remember_me:
					cookie_manager.set("bartbot_user", u_login, expires_at=datetime.now() + timedelta(days=30))
				st.rerun()
			else:
				st.error("Invalid username or password")

	with tab2:
		u_new = st.text_input("Choose Username", key="n_user")
		p_new = st.text_input("Choose Password", type="password", key="n_pass")
		if st.button("Register"):
			if u_new in all_data:
				st.error("Username already exists!")
			elif u_new and p_new:
				all_data[u_new] = {"password": p_new, "chats": {}}
				save_data(all_data)
				st.success("Account Created! Please login now.")
			else:
				st.warning("Please fill in both fields.")
	st.stop()

username = st.session_state.username

if username not in all_data:
	del st.session_state.username
	st.rerun()

user_chats = all_data[username]["chats"]

if st.session_state.get("active_chat_id") not in user_chats:
	st.session_state.active_chat_id = None

API_KEY = st.secrets.get("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)

BOT_NAME = "Bartholemew"
USER_NAME = "You"

if "all_chats" not in st.session_state:
	st.session_state.all_chats = load_data()
if "active_chat_id" not in st.session_state:
	st.session_state.active_chat_id = None

with st.sidebar:
	st.write(f"Logged in as: **{username.capitalize()}**")
	if st.button("Logout"):
		cookie_manager.delete("bartbot_user")
		for key in list(st.session_state.keys()):
			del st.session_state.username
		st.rerun()

	st.divider()

	if st.button("Start New Chat", use_container_width=True):
		new_id = str(uuid.uuid4())
		user_chats[new_id] = []
		st.session_state.active_chat_id = new_id
		save_data(all_data)
		st.rerun()
	
	chat_ids = list(user_chats.keys())

	for chat_id in reversed(list(user_chats.keys())):
		history = user_chats[chat_id]
		label = history[0]["content"][:25] + "..." if history else "New Conversation"

		col1, col2 = st.sidebar.columns([0.8, 0.2])

		with col1:
			if st.button(label, key=f"select_{chat_id}", use_container_width=True):
				st.session_state.active_chat_id = chat_id
				st.rerun()
		with col2:
			if st.button("X", key=f"del_{chat_id}", help="Delete this chat"):
				del st.session_state.all_chats[chat_id]

				if st.session_state.active_chat_id == chat_id:
					st.session_state.active_chat_id = None
				save_data(st.session_state.all_chats)
				st.rerun()



st.title("BartBot")

if st.session_state.active_chat_id:
	current_id = st.session_state.active_chat_id
	messages = user_chats[current_id]
	
	def get_chat_session(history_to_send):
		return client.chats.create(
        	model="gemini-2.5-flash-lite",
        	config=types.GenerateContentConfig(
        	    system_instruction=f"Your name is {BOT_NAME}. You are a witty AI assistant.",
        	    tools=[types.Tool(google_search=types.GoogleSearch())]
        	),
			history=history_to_send
	    )
	
	for i, message in enumerate(messages):
		name = USER_NAME if message["role"] == "user" else BOT_NAME
		with st.chat_message(message["role"]):
			content = message["content"]
			if isinstance(content, str) and content.startswith("IMAGE_DATA:"):
				base64_str = content.replace("IMAGE_DATA:", "")
				img_bytes = base64.b64decode(base64_str)
				st.image(img_bytes, caption=message.get("caption", ""))
				st.download_button(
					label="Download",
					data=img_bytes,
					file_name=f"bartbot_{i}.png",
					mime="image/png",
					key=f"download_btn_{current_id}_{i}"
				)
			else:
				st.markdown(f"**{name}**: {content}")

	if prompt := st.chat_input("What can I help you with? For image generation, start prompt with '/image'"):
		messages.append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(f"**{USER_NAME}**: {prompt}")
	
		with st.chat_message("assistant"):
			if prompt.lower().startswith("/image"):
				model_options = [
					'imagen-4.0-generate-001',
					'imagen-3.0-generate-002',
					'gemini-3-pro-image-preview'
				]
				image_prompt = prompt[7:].strip()
				success = False
				for model_id in model_options:
					try:
						with st.spinner("Bartholemew is painting..."):
							response = client.models.generate_images(
								model=model_id,
								prompt=image_prompt,
								config=types.GenerateImagesConfig(
									number_of_images=1,
									aspect_ratio="1:1"
								)
							)
							img_data = response.generated_images[0].image.image_bytes
							encoded_img = base64.b64encode(img_data).decode('utf-8')
							st.image(img_data, caption=image_prompt)
							st.download_button(
											label="Download",
											data=img_bytes,
											file_name=f"bartbot_{i}.png",
											mime="image/png",
											key=f"download_btn_{current_id}_{i}"
										)
							messages.append({
								"role": "assistant",
								"content": f"IMAGE_DATA:{encoded_img}",
								"caption": image_prompt
							})
							save_data(st.session_state.all_chats)
							success = True
							break
					except:
						continue
					
				if not success:
					st.error("Image generation failed with all models.")
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
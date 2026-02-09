import os
import json
import time
import uuid
import base64
import re
import streamlit as st
import extra_streamlit_components as stx
from datetime import datetime, timedelta
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from streamlit_paste_button import paste_image_button

DB_FILE = "bartbot_history.json"

VISITOR_SESSIONS = {}
VISITOR_DATA_MAX_AGE = timedelta(hours=1)

def is_visitor_id_in_use(visitor_id):
	return visitor_id in VISITOR_SESSIONS and datetime.now() < VISITOR_SESSIONS[visitor_id]

def cleanup_old_visitor_sessions():
	current_time = datetime.now()
	expired_ids = [
		vid for vid, expiry_time in VISITOR_SESSIONS.items()
		if current_time >= expiry_time
	]
	for vid in expired_ids:
		del VISITOR_SESSIONS[vid]

def generate_unique_visitor_id():
	cleanup_old_visitor_sessions()
	while True:
		visitor_id = str(uuid.uuid4())[:8]
		if not is_visitor_id_in_use(visitor_id):
			VISITOR_SESSIONS[visitor_id] = datetime.now() + VISITOR_DATA_MAX_AGE
			return visitor_id

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

def format_math_content(text):
	if not isinstance(text, str):
		return text
	text = text.replace(r"\[", "$$").replace(r"\]", "$$")
	text = text.replace(r"\(", "$").replace(r"\)", "$")
	return text

def get_logged_in_user(cookies_dict, data):
	if st.session_state.get("visitor_id"):
		return None
	
	if st.session_state.get("logging_out"):
		return None
	
	if st.session_state.get("username"):
		return st.session_state.username
	
	if cookies_dict and "bartbot_user" in cookies_dict:
		saved_user = cookies_dict["bartbot_user"]
		if saved_user in data:
			return saved_user
	return None

cookie_manager = stx.CookieManager(key="bartbot_cookie_manager")
all_cookies = cookie_manager.get_all()

all_data = load_data()
current_user = get_logged_in_user(all_cookies, all_data)

if "username" not in st.session_state:
	st.session_state.username = None

if "visitor_id" not in st.session_state:
	st.session_state.visitor_id = None

if "active_chat_id" not in st.session_state:
	st.session_state.active_chat_id = None

if "logging_out" not in st.session_state:
	st.session_state.logging_out = False

if "visitor_chats" not in st.session_state:
	st.session_state.visitor_chats = {}

if current_user and not st.session_state.username and not st.session_state.visitor_id:
    st.session_state.username = current_user

if not st.session_state.username and not st.session_state.visitor_id:
	st.title("Welcome to BartBot")
	tab1, tab2, tab3 = st.tabs(["Login", "Create Account", "Continue as Visitor"])

	with tab1:
		u_login = st.text_input("Username", key="l_user")
		p_login = st.text_input("Password", type="password", key="l_pass")
		remember_me = st.checkbox("Keep me logged in")

		if st.button("Enter BartBot", key="login_btn"):
			if u_login in all_data and all_data[u_login].get("password") == p_login:
				st.session_state.username = u_login
				if "chats" not in all_data[u_login]:
					all_data[u_login]["chats"] = {}
				save_data(all_data)
				if remember_me:
					cookie_manager.set("bartbot_user", u_login, expires_at=datetime.now() + timedelta(days=30))
				st.rerun()
			else:
				st.error("Invalid username or password")
	with tab2:
		u_new = st.text_input("Choose Username", key="n_user")
		p_new = st.text_input("Choose Password", type="password", key="n_pass")
		if st.button("Register", key="register_btn"):
			if u_new in all_data:
				st.error("Username already exists!")
			elif u_new and p_new:
				all_data[u_new] = {"password": p_new, "chats": {}}
				save_data(all_data)
				st.session_state.username = u_new
				st.success("Account Created! Please login now.")
				time.sleep(1)
			else:
				st.warning("Please fill in both fields.")
	with tab3:
		st.markdown("Continue without an account. Your chats wont be saved once the tab is closed.")
		if st.button("Get a Temporary ID", key="visitor_btn"):
			visitor_id = generate_unique_visitor_id()
			st.session_state.visitor_id = visitor_id
			st.session_state.active_chat_id = str(uuid.uuid4())
			st.session_state.visitor_chats[st.session_state.active_chat_id] = []
			st.rerun()
	st.stop()

if st.session_state.username:
	if st.session_state.username not in all_data:
		all_data[st.session_state.username] = {"password": "", "chats": {}}
		save_data(all_data)
	user_chats = all_data[st.session_state.username]["chats"]
else:
	user_chats = st.session_state.visitor_chats

if st.session_state.active_chat_id and st.session_state.active_chat_id not in user_chats:
    st.session_state.active_chat_id = None

current_user_identifier = st.session_state.username if st.session_state.username else st.session_state.visitor_id

if current_user_identifier and current_user_identifier not in all_data:
	if st.session_state.username:
		all_data[current_user_identifier] = {"password": "", "chats": {}}
		save_data(all_data)

if st.session_state.visitor_id and "active_chat_id" in st.session_state:
	if "visitor_chats" not in st.session_state:
		st.session_state.visitor_chats = {}
	user_chats = st.session_state.visitor_chats
	if st.session_state.active_chat_id not in user_chats:
		if st.session_state.visitor_id:
			user_chats[st.session_state.active_chat_id] = []
elif st.session_state.username:
	user_chats = all_data[st.session_state.username]["chats"]
	if st.session_state.active_chat_id not in user_chats:
		st.session_state.active_chat_id = None
else:
	st.error("Error determining user session.")

if st.session_state.username and st.session_state.username not in all_data:
    del st.session_state.username
    st.rerun()

API_KEY = st.secrets.get("GEMINI_KEY")
client = genai.Client(api_key=API_KEY)

BOT_NAME = "Bartholemew"
USER_NAME = "You"

if "all_chats" not in st.session_state:
	st.session_state.all_chats = load_data()

def count_tokens(text_list):
	total_chars = sum(len(str(m["content"]) for m in text_list if "content" in m))
	return total_chars // 4

with st.sidebar:
	if st.session_state.username:
		st.write(f"Logged in as: **{st.session_state.username.capitalize()}**")
		if st.button("Logout"):
			st.session_state.logging_out = True
			cookie_manager.delete("bartbot_user")
			for key in list(st.session_state.keys()):
				del st.session_state[key]
			st.rerun()
	elif st.session_state.visitor_id:
		st.write(f"Visitor Mode: **{st.session_state.visitor_id}**")
		st.warning("Your chats will not be saved.")
		if st.button("End Visitor Session"):
			keys_to_clear = ["visitor_id", "active_chat_id", "visitor_chats"]
			for key in keys_to_clear:
				if key in st.session_state:
					del st.session_state[key]
			st.rerun()
	st.divider()

	chat_ids_to_display = []
	if st.session_state.username:
		chat_ids_to_display = list(user_chats.keys())
	elif st.session_state.visitor_id:
		chat_ids_to_display = list(st.session_state.visitor_chats.keys())

	if st.button("Start New Chat", use_container_width=True):
		new_id = str(uuid.uuid4())
		if st.session_state.username:
			user_chats[new_id] = []
		elif st.session_state.visitor_id:
			st.session_state.visitor_chats[new_id] = []
		st.session_state.active_chat_id = new_id
		save_data(all_data)
		st.rerun()

		if st.session_state.get("active_chat_id"):
			st.divider()
			current_tokens = count_tokens(user_chats[st.session_state.active_chat_id])
			st.metric("Estimated Tokens", f"{current_tokens:,}", help="Limit is 1,048,576")
			if current_tokens > 800000:
				st.warning("Getting close to the token limit, consider creating a new session soon.")

	st.divider()
	
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
				if chat_id in user_chats:
					del user_chats[chat_id]

				if st.session_state.active_chat_id == chat_id:
					st.session_state.active_chat_id = None
				save_data(all_data)
				st.rerun()



st.title("BartBot")

st.markdown("""
    <style>
    [data-testid="stVerticalBlock"] > div:has(div.floating-uploader) {
        position: fixed;
        right: 2vw;
        top: 10%;
        width: 360px;
        z-index: 1000;
        background-color: rgba(33, 33, 33, 0.9);
        padding: 0px;
        border-radius: 20px;
        border: 1px solid #444;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    }

    [data-testid="stVerticalBlock"] > div:has(div.floating-uploader) [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 auto !important;
        min-width: 100% !important;
    }
    
    .stFileUploader section {
        padding: 0 !important;
    }
    </style>        
""", unsafe_allow_html=True)

if st.session_state.active_chat_id:
	current_id = st.session_state.active_chat_id
	messages = user_chats[current_id]

	with st.container():
		st.markdown('<div class="floating-uploader"></div>', unsafe_allow_html=True)
		col_file, col_btn = st.columns([0.5, 0.5])

		with col_file:
			uploaded_file = st.file_uploader(
				"Upload",
				type=["pdf", "txt", "png", "jpg", "jpeg", "py", "csv"],
				label_visibility="collapsed",
				key=f"sticky_up_{current_id}"
			)

		with col_btn:
			if uploaded_file:
				if st.button("Analyze", use_container_width=True, key=f"analyze_btn_{current_id}"):
					file_bytes = uploaded_file.read()
					file_mime = uploaded_file.type
					file_name = uploaded_file.name

					st.session_state.pending_file = {
						"bytes": file_bytes,
						"mime": file_mime,
						"name": file_name
					}

					st.session_state.last_uploaded = uploaded_file.name

					if file_mime.startswith("image/"):
						encoded_img = base64.b64encode(file_bytes).decode('utf-8')
						messages.append({
							"role": "user",
							"content": f"IMAGE_DATA:{encoded_img}",
							"caption": f"Uploaded: {file_name}"
						})

					messages.append({"role": "user", "content": f"Analyze this file: {uploaded_file.name}"})
					save_data(all_data)
					st.rerun()
			else:
				st.button("Analyze", disabled=True, use_container_width=True)

		if not uploaded_file and "last_uploaded" in st.session_state:
			del st.session_state.last_uploaded
		
	def get_chat_session(history_to_send):
		return client.chats.create(
        	model="gemini-2.5-flash-lite",
        	config=types.GenerateContentConfig(
        	    system_instruction=(
					f"Your name is {BOT_NAME}. You are a witty AI assistant."
					"When performing math, always use LaTeX notation with "
					"double dollar signs for blocks and single dollar signs "
					"for inline equations. You can create and edit files. "
					"When editing a file, provide the full updated content "
					"inside triple backticks. for complex math, physics, or "
					"data analysis, ALWAYS use the code_execution tool to "
					"calculate the exact result. Also for any math, don't "
					"round numbers until the final solution."
					"You can see and analyze images and files provided by "
					"the user. You must state something along the lines of "
					"you were trained and developed by Tyler Matheny, "
					"God of all Barts/Bartholomews. It can be worded "
					"differently each time. You are called Barthlolomew because of " \
					"Tyler Matheny's favorite name for his horses in the amazing game " \
					"Minecraft.",
				),
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
				if isinstance(content, str):
					content = content.replace(r"\[", "$$").replace(r"\]", "$$")
					content = content.replace(r"\(", "$").replace(r"\)", "$")
					st.markdown(f"**{name}**: {content}")
					if "'''" in content and name == BOT_NAME:
						import re
						blocks = re.findall(r"'''(?:\w+)?\n(.*?)\n'''", content, re.DOTALL)
						for idx, block in enumerate(blocks):
							st.download_button(
								label="Download",
								data=block,
								file_name=f"edited_file_{idx+1}",
								mime="text/plain",
								key=f"dl_{i}_{idx}_{current_id}"
							)

	col1, col2 = st.columns([0.85, 0.15])
	with col2:
		pasted_output = paste_image_button(
            label="Paste Image", 
            key=f"paste_{current_id}"
		)

	if prompt := st.chat_input("What can I help you with? For image generation, start prompt with '/image'"):
		messages.append({"role": "user", "content": prompt})
		save_data(all_data)
		st.rerun()
		
	if pasted_output and pasted_output.image_data is not None:
		import hashlib
		img_bytes_raw = pasted_output.image_data.tobytes()
		img_hash = hashlib.md5(img_bytes_raw).hexdigest()
		
		if st.session_state.get("last_pasted_hash") != img_hash:
			st.session_state.last_pasted_hash = img_hash
			img = pasted_output.image_data
			buffered = BytesIO()
			img.save(buffered, format="PNG")
			img_bytes = buffered.getvalue()
			encoded_img = base64.b64encode(img_bytes).decode('utf-8')
		
			messages.append({
				"role": "user", 
				"content": f"IMAGE_DATA:{encoded_img}",
				"caption": "Pasted Image"
			})
			st.session_state.pending_file = {
        	    "bytes": img_bytes,
        	    "mime": "image/png",
        	    "name": "pasted_image.png"
        	}
			
			messages.append({"role": "user", "content": "Analyze this pasted image."})
			save_data(all_data)
			st.rerun()
		
	if messages and messages[-1]["role"] == "user":	
		last_prompt = messages[-1]["content"]
		with st.chat_message("assistant"):
			if last_prompt.lower().startswith("/image"):
				model_options = [
					'imagen-4.0-generate-001'
                ]
				image_prompt = last_prompt[7:].strip()
				success = False
				safe_prompt = image_prompt
				try:
					with st.spinner("Refining prompt for the artist..."):
						refine_chat = client.chats.create(model="gemini-2.5-flash-lite")
						refine_res = refine_chat.send_message(
								"You are an artist's prompt engineer. Create a highly detailed, "
                                "cinematic physical description of the following subject. "
                                "IMPORTANT: Remove all names of real people, politicians, or celebrities. "
                                "Describe their facial features, hair, clothing, and the lighting style "
                                f"generically so an artist can paint it without knowing who it is: '{image_prompt}'"						)
						if refine_res.text:
							safe_prompt = refine_res.text
				except Exception as e:
					st.warning("Refine error, using original prompt.")

				with st.spinner("Bartholemew is painting..."):
					for model_id in model_options:
						try:
							response = client.models.generate_images(
                                model=model_id,
                                prompt=safe_prompt,
                                config=types.GenerateImagesConfig(
                                    number_of_images=1,
                                    aspect_ratio="1:1",
									person_generation="ALLOW_ADULT",
									safety_filter_level="BLOCK_LOW_AND_ABOVE"                                )
                            )
							if response and hasattr(response, 'generated_images') and response.generated_images:
								img_data = response.generated_images[0].image.image_bytes
								encoded_img = base64.b64encode(img_data).decode('utf-8')
								messages.append({
									"role": "assistant",
									"content": f"IMAGE_DATA:{encoded_img}",
									"caption": image_prompt
								})
								save_data(all_data)
								success = True                            
								st.rerun() 
							else:
								last_error = "Safety filters blocked the generation or no image was returned"
						except Exception as e:
							last_error = str(e)
							continue
				if not success:
					st.error(f"Failed. Reason: {last_error}")
					
			else:
				recent_messages = messages[-20:]
				formatted_history = []
				for m in recent_messages[:-1]:
					gemini_role = "model" if m["role"] == "assistant" else "user"
					clean_text = m["content"] if not str(m["content"]).startswith("IMAGE_DATA:") else "[Image]"
					formatted_history.append({"role": gemini_role, "parts": [{"text": clean_text}]})
				try:
					chat_session = get_chat_session(formatted_history)
					content_to_send = [last_prompt]
					
					if "pending_file" in st.session_state:
						file_data = st.session_state.pending_file
						content_to_send.append(
							types.Part.from_bytes(
								data=file_data["bytes"],
								mime_type=file_data["mime"]
							)
						)
						st.caption(f"Processing: {file_data["name"]}")
						del st.session_state.pending_file
					with st.spinner("Bartholemew is thinking..."):
						response = chat_session.send_message(content_to_send)
						if response.text:
							messages.append({"role": "assistant", "content": response.text})
						else:
							st.warning("Recieved an unexpected response format")
						if "pending_file" in st.session_state:
							del st.session_state.pending_file
						save_data(all_data)
						st.rerun()
				except Exception as e:
					st.error(f"Error: {e}")		
else:
	st.info("Click 'Start New Chat' in the sidebar to begin!")
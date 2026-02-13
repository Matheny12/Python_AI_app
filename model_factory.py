from ai_models import AIModel
from GeminiBartModel import GeminiModel
from BartBotModel import BartBotModel
import streamlit as st

def get_model(model_type: str = "gemini") -> AIModel:
    if model_type == "GeminiBart":
        api_key = st.secrets.get("GEMINI_KEY")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")
    elif model_type == "BartBot":
        return BartBotModel("meta-llama/Llama-3.1-8B-Instruct")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
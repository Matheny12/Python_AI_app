from ai_models import AIModel
from GeminiBartModel import GeminiModel
import streamlit as st
import os

def get_model(model_type: str = "GeminiBart") -> AIModel:
    if model_type == "GeminiBart":
        api_key = st.secrets.get("GEMINI_KEY")
        if not api_key:
            raise ValueError("GEMINI_KEY not found in environment or secrets")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")
    elif model_type == "BartBot":
        try:
            from BartBotModel import BartBotModel
            return BartBotModel()
        except Exception as e:
            st.error(f"BartBot failed to load: {str(e)}")
            st.info("Falling back to Gemini...")
            api_key = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")
            return GeminiModel(api_key=api_key, bot_name="Bartholemew")
    else:
        api_key = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")
        return GeminiModel(api_key=api_key, bot_name="Bartholemew")
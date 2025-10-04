""" 
A mushroom expert chatbot that responds to user queries about mushrooms.
"""
from dotenv import load_dotenv
import os
from google import genai
import gradio as gr
import random
import mimetypes
import json
from google.genai.types import Part  

print("🚀 Mushroom chatbot started")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

gemini_model = "gemini-2.5-flash"   

# Latest mushroom type 
latest_mushroom_data = {}

import json

latest_mushroom_data = None  # global state for memory

def gemini_response(message, history):
    global latest_mushroom_data

    # --- Parse input (text vs multimodal)
    if isinstance(message, str):
        user_text = message
        user_files = []
    elif isinstance(message, dict):
        user_text = message.get("text", "")
        user_files = message.get("files", [])
    else:
        user_text, user_files = "", []

    has_images = bool(user_files)
    user_question = user_text.strip() if user_text else ""

    # --- Build system prompt
    if has_images:
        if user_question:
            explanation_mode = (
                "After the JSON, insert a separator line: ---\n"
                "If the user asked a question, ONLY answer that question directly. "
                "Do NOT summarize the identification."
            )
        else:
            explanation_mode = (
                "After the JSON, insert a separator line: ---\n"
                "If no question was asked, summarize the mushroom identification using the JSON data."
            )

        system_prompt = (
            "You are a mycology expert with deep knowledge of mushrooms. "
            "When analyzing mushroom images, always return your output in two parts:\n\n"
            "1. First, a JSON block with the following fields:\n"
            "{\n"
            "  \"common_name\": \"...\",\n"
            "  \"genus\": \"...\",\n"
            "  \"confidence\": 0.0-1.0,\n"
            "  \"visible\": [\"cap\", \"hymenium\", \"stipe\"],\n"
            "  \"color\": \"...\",\n"
            "  \"edible\": true/false/null\n"
            "}\n\n"
            f"2. {explanation_mode}\n\n"
            "If the image has no mushrooms, return null values in JSON and say so politely. "
            "If you are unsure, lower the confidence score and suggest verification."
        )
    else:
        system_prompt = (
            "You are a mycology expert with deep knowledge of mushrooms. "
            "Answer in an encyclopedic, factual style. "
            "Answer only questions about mushrooms and fungi. "
            "If the user previously uploaded an image and now asks a follow-up, "
            "you may reference the stored identification. "
            "If the user asks about other mushrooms, just answer factually. "
            "If asked about other topics, politely respond: "
            "'Sorry, but I can only answer mushroom-related questions.' "
            "If unsure about an identification, suggest verification from a field expert."
        )

    # --- Build request contents
    contents = [{"text": system_prompt}]
    if latest_mushroom_data and not has_images:
        context = f"Previously identified mushroom data: {json.dumps(latest_mushroom_data, indent=2)}"
        contents.append({"text": context})
    if user_text:
        contents.append({"text": user_text})

    for f in user_files:
        mime_type, _ = mimetypes.guess_type(f)
        if not (mime_type and mime_type.startswith("image/")):
            return "I can only analyze images of mushrooms."
        with open(f, "rb") as file_data:
            contents.append(Part.from_bytes(file_data.read(), mime_type=mime_type))

    # --- Streaming + safety + JSON parsing
    final_text = ""
    explanation_text = ""
    try:
        stream = client.models.generate_content_stream(
            model=gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                ],
            ),
        )

        for chunk in stream:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]

            # --- Safety filter
            if candidate.safety_ratings:
                for rating in candidate.safety_ratings:
                    if rating.probability not in ("NEGLIGIBLE", "LOW"):
                        print("[Gemini Safety Filter Triggered]")
                        yield "Sorry, I cannot answer that question due to safety restrictions."
                        return

            if candidate.content and candidate.content.parts:
                part = candidate.content.parts[0].text
                if not part:
                    continue

                final_text += part
                yield final_text  # ✅ live streaming to UI

        # --- Post-process JSON if image mode
        if has_images and "---" in final_text:
            json_part, explanation = final_text.split("---", 1)
            cleaned = json_part.strip()

            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            try:
                json_data = json.loads(cleaned)
                latest_mushroom_data = json_data
                print("🍄 Mushroom Identification Data:")
                print(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                print("[ERROR] Failed to parse JSON from model output.")
                print("Raw JSON candidate:\n", cleaned)

            explanation_text = explanation.strip()

        # --- Final yields depending on mode
        if has_images and explanation_text:
            yield explanation_text
        elif not has_images:
            yield final_text

    except Exception as e:
        print(f"[ERROR] Streaming failed: {str(e)}")
        yield "Sorry, the conversation cannot continue due to a streaming error."

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.ChatInterface(
        fn=gemini_response,
        title="🍄 Your Personal Mushroom Expert 🍄‍🟫",
        multimodal=True,
        type="messages"
    )

if __name__ == "__main__":
    demo.launch()

"""A mushroom expert chatbot that responds to user queries about mushrooms.
"""
from dotenv import load_dotenv
import os
from google import genai
import gradio as gr
import random
import mimetypes
import json
from google.genai.types import Part 
from google.genai import types


# Exercise 1: Add the Gemini chatbot to the interface.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

gemini_model = "gemini-2.5-flash"   

# Latest mushroom identification data for memory
latest_mushroom_data = None

def gemini_response(message, history):
    global latest_mushroom_data

    # Exercise 2: Make the chatbot multimodal so that it can read mushroom images.
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

    # Exercise 3: Add instructions to the chatbot, with enhanced JSON processing for images
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
            "If you are unsure about an identification, suggest verification "
            "from a field expert and lower the confidence score."
        )
    else:
        system_prompt = (
            "You are a mycology expert with deep knowledge of mushrooms. "
            "Answer in an encyclopedic, factual style. "
            "Answer only questions about mushrooms and fungi. "
            "If the user previously uploaded an image and now asks a follow-up, "
            "you may reference the stored identification data. "
            "If the user asks about other mushrooms, just answer factually. "
            "If asked about other topics, politely respond: "
            "'Sorry, but I can only answer mushroom-related questions.' "
            "If you are unsure about an identification, suggest verification "
            "from a field expert."
        )
    
    contents = [{"text": system_prompt}]
    
    # Add context from previous identification if available and no new images
    if latest_mushroom_data and not has_images:
        context = f"Previously identified mushroom data: {json.dumps(latest_mushroom_data, indent=2)}"
        contents.append({"text": context})
    
    # Add text if provided
    if user_text:
        contents.append({"text": user_text})

    # Process image files
    for f in user_files:
        mime_type, _ = mimetypes.guess_type(f)
        if not (mime_type and mime_type.startswith("image/")):
            return "I can only analyze images of mushrooms."
        with open(f, "rb") as file_data:
            contents.append(
                Part.from_bytes(
                    data=file_data.read(),
                    mime_type=mime_type
                )
            )
    
    final_text = "" 

    try:
        # Exercise 4: Stream the answers and display the chunks of text as they are being processed.
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
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ],
            ),
        )

        for chunk in stream:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]

            # Exercise 5: Print a message when the chatbot's safety filter is triggered.
            if candidate.safety_ratings:
                for rating in candidate.safety_ratings:
                    if rating.probability not in ("NEGLIGIBLE", "LOW"):
                        print("[Gemini Safety Filter Triggered]")
                        yield "Sorry, I cannot answer that question due to safety restrictions. Please ask me something related to mushrooms."
                        return  # stop streaming immediately
    
            # Case 2: Log if Gemini refused
            if candidate.content and candidate.content.parts:
                part = candidate.content.parts[0].text
                if not part:
                    continue
                
                if part and (part.startswith("I cannot") or part.startswith("Sorry")):
                    print("[Gemini Refusal Filter Triggered]")
    
                final_text += part
                
                # Only stream for text-only mode, buffer for image mode
                if not has_images:
                    yield final_text

        # Post-process JSON if image mode
        if has_images and "---" in final_text:
            json_part, explanation = final_text.split("---", 1)
            cleaned = json_part.strip()

            # Clean up JSON formatting
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()

            try:
                json_data = json.loads(cleaned)
                latest_mushroom_data = json_data
                print("Mushroom Identification Data:")
                print(json.dumps(json_data, indent=2))
            except json.JSONDecodeError:
                print("[ERROR] Failed to parse JSON from model output.")
                print("Raw JSON candidate:\n", cleaned)

            explanation_text = explanation.strip()
            
            # Final yield with explanation only for image mode
            if explanation_text:
                yield explanation_text
        elif has_images:
            # If no separator found but has images, yield the full text
            yield final_text

    # Exercise 6: If the streaming fails, it is not possible to keep the conversation going. Add a try catch block to handle this situation.
    except Exception as e:
        print(f"[ERROR] Streaming failed: {str(e)}")
        yield "Sorry, the conversation cannot continue due to a streaming error."

        
# Add a theme to the chat interface.
mushroom_theme = gr.themes.Base(
    primary_hue="stone",    
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    # Backgrounds
    body_background_fill="linear-gradient(160deg, #f6f5f3 0%, #ebe7df 100%)", # light cream to off-white
    block_background_fill="linear-gradient(135deg, #e4dacd 0%, #c9b7a6 100%)", # warm beige

    # Text
    body_text_color="#2f2f2f",

    # Buttons
    button_primary_background_fill="#8b5e3c",  # brown
    button_primary_background_fill_hover="#6b4226",
    button_primary_text_color="white",

    button_secondary_background_fill="#8b5e3c", # brown
    button_secondary_background_fill_hover="#6b4226",
    button_secondary_text_color="white",
)


css = """
/* User messages */
.message.user {
    background-color: #d8b89c !important; /* soft beige-brown */
    color: #2f2f2f !important;            /* dark gray */
    border-radius: 16px !important;       /* rounded bubble corners */
    border: none !important;              /* no border outline */
}

/* Bot messages */
.message.bot {
    background-color: #f5ede6 !important; /* light cream */
    color: #2f2f2f !important;            
    border-radius: 16px !important;
    border: none !important;
}

"""


with gr.Blocks(fill_height=True, theme=mushroom_theme, css=css) as demo:
    chatbot = gr.ChatInterface(
        fn=gemini_response,
        title="🍄 Your Personal Mushroom Expert 🍄",
        # Add a description to the chat interface.
        description="Ask me anything about mushrooms! I'll answer your questions, identify features in your images, and share mycological insights.",
        multimodal=True,
        type="messages"
    )

if __name__ == "__main__":
    demo.launch()
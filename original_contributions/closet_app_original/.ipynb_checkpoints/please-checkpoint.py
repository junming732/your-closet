
"""A mushroom expert chatbot that responds to user queries about mushrooms.
"""
from dotenv import load_dotenv
import os
from google import genai
import gradio as gr
import random
import mimetypes
from google.genai.types import Part 
from google.genai import types


# Exercise 1: Add the Gemini chatbot to the interface.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

gemini_model = "gemini-2.5-flash"   

def gemini_response(message, history):

# Exercise 2: Make the chatbot multimodal so that it can read mushroom images.

    if isinstance(message, str):
        user_text = message
        user_files = []
    elif isinstance(message, dict):
        user_text = message.get("text", "")
        user_files = message.get("files", [])
    else:
        user_text, user_files = "", []

# Exercise 3: Add instructions to the chatbot, so that it behaves like a mushroom expert, and that it stirs the conversation in the right direction (i.e. mushrooms related).
    
    contents = [{
        "text": (
                "You are a mycology expert with deep knowledge of mushrooms. "
                    "Answer in an encyclopedic, factual style. "
                    "Answer only questions about mushrooms and fungi. "
                    "If asked about other topics, politely respond: "
                    "'Sorry, but I can only answer mushroom-related questions.' "
                    "If given an image, describe any mushrooms seen: note their color, size, "
                    "environment, and visible parts (cap, hymenium, stipe). "
                    "If the image has no mushrooms, say so politely. "
                    "If you are unsure about an identification, suggest verification "
                    "from a field guide or a local expert."

        )
    }]
    
    # Add text if provided
    if user_text:
        contents.append({"text": user_text})

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
            if chunk.candidates:
                candidate = chunk.candidates[0]

                # Exercise 5: Print a message when the chatbot's safety filter is triggered.
                # Case 1: Safety ratings on this chunk
                if candidate.safety_ratings:
                    for rating in candidate.safety_ratings:
                        if rating.probability not in ("NEGLIGIBLE", "LOW"):
                            print("[Gemini Safety Filter Triggered]")
                            yield "Sorry, I cannot answer that question due to safety restrictions. Please ask me something related to mushrooms."
                            return  # stop streaming immediately
        
                # Case 2: Log if Gemini refused
                if candidate.content and candidate.content.parts:
                    part = candidate.content.parts[0].text
                    if part and (part.startswith("I cannot") or part.startswith("Sorry")):
                        print("[Gemini Safety Filter Triggered]")
        
                    if part:
                        final_text += part
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
        description="Ask me anything about mushrooms!’ll answer your questions, identify features in your images, and share mycological insights.",
        multimodal=True,
        type="messages"
    )

if __name__ == "__main__":
    demo.launch()


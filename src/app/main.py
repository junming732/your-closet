import gradio as gr
import pandas as pd
from src.app.ui_config import fashion_theme, custom_css, occasions, seasons


from src.app.wardrobe_app import (
    add_item_to_wardrobe, upload_csv, enter_edit_mode, exit_edit_mode,
    update_item_choices, delete_selected_items, clear_wardrobe,
    export_wardrobe_csv, generate_outfit, chat_response, fetch_weather
)

from src.retrieval.gemini_rag import (
    make_client as gemini_make_client,
    GeminiEmbeddings, load_pdf_as_documents, chunk_docs, get_vectorstore
)
from src.retrieval.fashion_theory_rag import (
    make_client as fashion_make_client,
    generate_fashion_theory_advice
)

# Set up
embedding_client = gemini_make_client()
embeddings = GeminiEmbeddings(embedding_client)
pdf_docs = load_pdf_as_documents("original_contributions/BeginnerGuide_howtodress_original.pdf")
chunks = chunk_docs(pdf_docs)
db = get_vectorstore(chunks, embeddings)


# --- Gradio app state ---
# UI
fashion_theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    body_background_fill="linear-gradient(160deg, #f7f9f7 0%, #e8f0e8 100%)",
    block_background_fill="#f5f3f0",
    # Primary button colors (darker green for main actions)
    button_primary_background_fill="#8a9a8a",
    button_primary_background_fill_hover="#6b7c6b",
    button_primary_text_color="white",

    # Secondary button colors (lighter green for secondary actions)
    button_secondary_background_fill="#b8c8b8",
    button_secondary_background_fill_hover="#9ab09a",
    button_secondary_text_color="white",
    body_text_color="#4a4540",
    block_label_text_color="#5a534d",
    block_label_background_fill="#e8e4df",
    input_background_fill="#faf8f5",
    input_border_color="#d4cdc4",
    block_border_color="#e0d8d0",
    block_title_text_color="#4a4540",
    link_text_color="#6b6460",
    link_text_color_hover="#8a9a8a",
)

with gr.Blocks(
    theme=fashion_theme,
    title="Fashion Assistant",
    css=custom_css) as demo:

    gr.Markdown("# Your Personal Fashion Stylist")

    with gr.Tabs():
        # TAB 1: Wardrobe Management
        with gr.Tab("Your Wardrobe"):
            # Input fields in a 2x2 grid
            with gr.Row():
                item_name = gr.Textbox(label="Item (e.g., Dress, Jeans)", placeholder="e.g., Dress, Jeans", max_lines=1, container=True, scale=1, elem_classes="centered-input")
                item_color = gr.Textbox(label="Color", placeholder="e.g., Blue, Black", max_lines=1, container=True, scale=1, elem_classes="centered-input")

            with gr.Row():
                item_pattern = gr.Textbox(label="Pattern", placeholder="e.g., Striped, Floral", max_lines=1, container=True, scale=1, elem_classes="centered-input")
                item_material = gr.Textbox(label="Material", placeholder="e.g., Cotton, Leather", max_lines=1, container=True, scale=1, elem_classes="centered-input")

            # Add Item button
            # Primary action button
            with gr.Row():
                add_btn = gr.Button("+ Add Item", variant="primary", size="sm", scale=0, min_width=150)

            # My Closet title with CSV upload and edit buttons
            with gr.Row():
                gr.Markdown("# My Closet")
                with gr.Column(scale=0, min_width=300):
                    with gr.Row():
                        csv_upload = gr.UploadButton(
                            "Upload CSV",
                            file_types=[".csv"],
                            file_count="single",
                            variant="secondary",
                            size="sm",
                            scale=0
                        )
                        edit_icon_btn = gr.Button("✎ Edit", variant="secondary", size="sm", scale=0)
                        done_editing_btn = gr.Button("Done", variant="primary", size="sm", scale=0, visible=False) # Hidden by default
                        delete_selected_btn = gr.Button("Delete", variant="stop", size="sm", scale=0, visible=False) # Hidden by default

            gr.Markdown("*In edit mode, click the checkboxes in the 'Select' column, then click Delete button*")

            # Interactive table displaying wardrobe items
            # Starts as non-interactive (read-only); becomes editable in edit mode
            wardrobe_display = gr.Dataframe(
                value=pd.DataFrame(columns=["Item", "Color", "Pattern", "Material"]),
                headers=["Item", "Color", "Pattern", "Material"],
                label="",
                interactive=False,
                row_count=(3, "dynamic")
            )

            # Action buttons below table
            with gr.Row():
                clear_table_btn = gr.Button("Clear Table", variant="stop", size="sm")
                export_csv_btn = gr.Button("Export CSV", variant="secondary", size="sm")

            # Hidden file component for CSV download
            # Becomes visible when export button is clicked
            export_file = gr.File(label="Download Wardrobe CSV", visible=False)

        # TAB 2: Interactive Outfit Builder
        with gr.Tab("Build Outfit"):
            gr.Markdown("Get styling advice based on your wardrobe")

            with gr.Row():
                with gr.Column():
                    occasion = gr.Dropdown(
                        choices=occasions,
                        label="Occasion",
                        value="Casual"
                    )
                with gr.Column():
                    season = gr.Dropdown(
                        choices=seasons,
                        label="Season",
                        value="Spring"
                    )

            # Custom occasion input (shown only when "Other" is selected)
            custom_occasion_input = gr.Textbox(
                label="Specify your occasion",
                placeholder="e.g., Job Interview, Wedding Reception, Concert",
                value="",
                max_lines=1,
                visible=False
            )

            # City input field with live weather option
            with gr.Row():
                city_input = gr.Textbox(
                    label="(Optional) City",
                    placeholder="e.g., New York, London, Tokyo",
                    value="",
                    max_lines=1,
                    scale=3
                )
                use_live_weather = gr.Checkbox(
                    label="Use Live Weather",
                    value=False,
                    scale=1,
                    container=True
                )

            weather_display = gr.Textbox(
                label="Current Weather",
                value="",
                interactive=False,
                visible=False,
                max_lines=1
            )

            # Hidden state to store weather data for prompt
            weather_prompt_state = gr.State(value="")

            selected_items = gr.Dropdown(
                choices=[], # Populated dynamically from wardrobe
                label="(Optional) Choose items from your wardrobe",
                multiselect=True, # Allow selecting multiple items
                value=[] # No items selected by default
            )

            # State to track previous outfits to avoid repetition
            previous_outfits_state = gr.State(value=[])

            # Button row with primary generate and secondary regenerate
            with gr.Row():
                generate_btn = gr.Button("Generate Outfit", variant="primary", scale=2)
                regenerate_btn = gr.Button(" Generate Different Outfit", variant="secondary", scale=1, visible=False)

            outfit_output = gr.Markdown(label="Your Outfit", elem_classes="outfit-markdown")


        # TAB 3: Chat Mode
        with gr.Tab("Chat with Stylist"):
            gr.Markdown("Ask questions or request outfit recommendations")

            chatbot = gr.Chatbot(label="Fashion Chat", type='messages', height=300)
            
            # Suggestion prompts - visible only when chat is empty
            with gr.Column(visible=True) as suggestions_group:
                gr.Markdown("### Try asking me about:")
                
                
                suggestion_1 = gr.Button(
                    "What should I wear for a first date?",
                    variant="secondary",
                    size="sm"
                )
                suggestion_2 = gr.Button(
                    "What are the key principles of garment construction?",
                       variant="secondary",
                    size="sm"
                )
                
                suggestion_3 = gr.Button(
                    "How has fashion evolved from Victorian to modern times?",
                    variant="secondary",
                    size="sm"
                )
                suggestion_4 = gr.Button(
                    "How can I build a capsule wardrobe?",
                    variant="secondary",
                    size="sm"
                )
                
                suggestion_5 = gr.Button(
                    "What colors complement each other best?",
                    variant="secondary",
                    size="sm"
                )

            msg = gr.Textbox(label="Message", placeholder="What should I wear today?", lines=1, max_lines=2)

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_chat_btn = gr.Button("Clear")

    def user_msg(message, history):
        return "", history + [{"role": "user", "content": message}]

    def bot_msg(history, wardrobe, occ, seas):
        user_message = history[-1]["content"]
        for response in chat_response(user_message, history[:-1], wardrobe, occ, seas):
            if len(history) > 0 and history[-1]["role"] == "user":
                history.append({"role": "assistant", "content": response})
            else:
                history[-1] = {"role": "assistant", "content": response}
            yield history

    # Events

    # Add item button: add to wardrobe -> clear inputs -> update dropdown
    add_btn.click(
        fn=add_item_to_wardrobe,
        inputs=[item_name, item_color, item_pattern, item_material, wardrobe_display],
        outputs=wardrobe_display
    ).then(
        fn=lambda: ("", "", "", ""), # Clear all input fields
        outputs=[item_name, item_color, item_pattern, item_material]
    ).then(
        fn=update_item_choices, # Update item selector dropdown
        inputs=[wardrobe_display],
        outputs=[selected_items]
    )

    # CSV upload: merge with wardrobe -> update dropdown
    csv_upload.upload(
        fn=upload_csv,
        inputs=[csv_upload, wardrobe_display],
        outputs=wardrobe_display
    ).then(
        fn=update_item_choices,
        inputs=[wardrobe_display],
        outputs=[selected_items]
    )

    # Enter edit mode: add Select column, show edit buttons
    edit_icon_btn.click(
        fn=enter_edit_mode,
        inputs=[wardrobe_display],
        outputs=[wardrobe_display, edit_icon_btn, done_editing_btn, delete_selected_btn]
    )

    # Exit edit mode: remove Select column, hide edit buttons, update dropdown
    done_editing_btn.click(
        fn=exit_edit_mode,
        inputs=[wardrobe_display],
        outputs=[wardrobe_display, edit_icon_btn, done_editing_btn, delete_selected_btn]
    ).then(
        # After exiting edit mode, refresh the "Choose items" dropdown with updated wardrobe
        fn=update_item_choices,
        inputs=[wardrobe_display],
        outputs=[selected_items]
    )

    # Delete selected items from wardrobe table
    delete_selected_btn.click(
        fn=delete_selected_items,
        inputs=[wardrobe_display],
        outputs=wardrobe_display
    ).then(
        # Refresh dropdown after deletion
        fn=update_item_choices,
        inputs=[wardrobe_display],
        outputs=[selected_items]
    )

    # Clear entire wardrobe table
    clear_table_btn.click(
        fn=clear_wardrobe,
        outputs=wardrobe_display
    ).then(
        # Refresh dropdown to show empty list
        fn=update_item_choices,
        inputs=[wardrobe_display],
        outputs=[selected_items]
    )

    # Export wardrobe table to CSV file
    export_csv_btn.click(
        fn=export_wardrobe_csv,
        inputs=[wardrobe_display],
        outputs=[export_file]
    )

    # Fetch weather when checkbox is enabled
    def handle_weather_toggle(use_weather, city):
        """Fetch weather when checkbox is checked, clear when unchecked."""
        if use_weather and city:
            return fetch_weather(city)
        else:
            return "", "", gr.update(visible=False)

    use_live_weather.change(
        fn=handle_weather_toggle,
        inputs=[use_live_weather, city_input],
        outputs=[weather_display, weather_prompt_state, weather_display]
    )

    # Also fetch weather when city changes if checkbox is already enabled
    city_input.change(
        fn=handle_weather_toggle,
        inputs=[use_live_weather, city_input],
        outputs=[weather_display, weather_prompt_state, weather_display]
    )

    # Generate outfit suggestion - NOW INCLUDING WEATHER DATA
    generate_btn.click(
        fn=generate_outfit,
        inputs=[wardrobe_display, occasion, season, city_input, selected_items, custom_occasion_input, weather_prompt_state],
        outputs=outfit_output
    )
    

    # Show/hide custom occasion input based on occasion dropdown selection
    def toggle_custom_occasion(occasion_value):
        if occasion_value == "Other":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)

    occasion.change(
        fn=toggle_custom_occasion,
        inputs=[occasion],
        outputs=[custom_occasion_input]
    )

    def handle_generate_outfit(wardrobe, occ, seas, city, items, custom_occ, weather, prev_outfits):
        """Wrapper to handle initial outfit generation and show regenerate button"""
        outfit_gen = generate_outfit(wardrobe, occ, seas, city, items, custom_occ, weather, prev_outfits)
        final_outfit = ""
        
        for chunk in outfit_gen:
            final_outfit = chunk
            yield chunk, gr.update(visible=False), prev_outfits  # Hide regenerate during generation
        
        # After generation completes, show regenerate button and update history
        new_history = prev_outfits + [final_outfit]
        yield final_outfit, gr.update(visible=True), new_history

    generate_btn.click(
        fn=lambda: ([], gr.update(visible=False)),  # Reset history and hide regenerate button
        outputs=[previous_outfits_state, regenerate_btn]
    ).then(
        fn=handle_generate_outfit,
        inputs=[wardrobe_display, occasion, season, city_input, selected_items, custom_occasion_input, weather_prompt_state, previous_outfits_state],
        outputs=[outfit_output, regenerate_btn, previous_outfits_state]
    )

    # Regenerate outfit with different suggestions
    regenerate_btn.click(
        fn=handle_generate_outfit,
        inputs=[wardrobe_display, occasion, season, city_input, selected_items, custom_occasion_input, weather_prompt_state, previous_outfits_state],
        outputs=[outfit_output, regenerate_btn, previous_outfits_state]
    )
    

    # Helper function to handle suggestion clicks
    def handle_suggestion_click(suggestion_text, history):
        """When user clicks a suggestion, add it as a user message, hide suggestions"""
        new_history = history + [{"role": "user", "content": suggestion_text}]
        return "", new_history, gr.update(visible=False)
    
    # Helper to handle user messages and hide suggestions
    def user_msg_with_hide(message, history):
        """Add user message and hide suggestions"""
        if not message.strip():
            return message, history, gr.update(visible=len(history) == 0)
        new_history = history + [{"role": "user", "content": message}]
        return "", new_history, gr.update(visible=False)
    
    # Suggestion button handlers
    suggestion_1.click(
        fn=lambda h: handle_suggestion_click("What should I wear for a first date?", h),
        inputs=[chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )
    
    suggestion_2.click(
        fn=lambda h: handle_suggestion_click("What are the key principles of garment construction?", h),
        inputs=[chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )
    
    suggestion_3.click(
        fn=lambda h: handle_suggestion_click("How has fashion evolved from Victorian to modern times?", h),
        inputs=[chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )
    
    suggestion_4.click(
        fn=lambda h: handle_suggestion_click("How can I build a capsule wardrobe?", h),
        inputs=[chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )
    
    suggestion_5.click(
        fn=lambda h: handle_suggestion_click("What colors complement each other best?", h),
        inputs=[chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )
    
    # Handle chat: when user presses Enter in text box
    msg.submit(
        fn=user_msg_with_hide,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )

    # Handle chat: when user clicks Send button
    send_btn.click(
        fn=user_msg_with_hide,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, suggestions_group]
    ).then(
        fn=bot_msg,
        inputs=[chatbot, wardrobe_display, occasion, season],
        outputs=chatbot
    )

    # Clear chat and show suggestions again
    clear_chat_btn.click(
        fn=lambda: ([], gr.update(visible=True)),
        outputs=[chatbot, suggestions_group]
    )


    # --- FOOTER BANNER (local image) ---
    banner_path = "src/assets/bla.drawio (1).png"
    with gr.Row(elem_classes="footer-banner"):
        gr.Image(
            value=banner_path,
            show_label=False,
            container=False,
            interactive=False
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)
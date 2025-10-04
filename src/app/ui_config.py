
# src/app/ui_config.py
import gradio as gr

# Theme
fashion_theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    body_background_fill="linear-gradient(160deg, #f7f9f7 0%, #e8f0e8 100%)",
    block_background_fill="#f5f3f0",
    button_primary_background_fill="#8a9a8a",
    button_primary_background_fill_hover="#6b7c6b",
    button_primary_text_color="white",
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

# Custom CSS
custom_css = """
 /* Footer banner styling - full width image at bottom */
.footer-banner {
    margin-top: 30px;
    margin-left: calc(-1 * var(--block-padding));
    margin-right: calc(-1 * var(--block-padding));
    width: calc(100% + 2 * var(--block-padding));
}
.footer-banner img {
    width: 100%;
    height: auto;
    display: block;
    object-fit: cover;
}
/* Center text in input fields */
.centered-input input {
    text-align: center !important;
}
/* Center table headers and add light background */
.dataframe thead th {
    text-align: center !important;
    background-color: #e8f0e8 !important;
    font-weight: 600 !important;
}
.dataframe thead th > * {
    text-align: center !important;
    justify-content: center !important;
}
table.table thead th {
    text-align: center !important;
}
table.table thead th span {
    display: block !important;
    text-align: center !important;
}
/* Center table cell content - all methods */
.dataframe tbody td {
    text-align: center !important;
}
.dataframe tbody td > * {
    text-align: center !important;
    justify-content: center !important;
}
.dataframe .cell-wrap {
    text-align: center !important;
    justify-content: center !important;
}
/* Center text in editable cells */
.dataframe input[type="text"],
.dataframe textarea {
    text-align: center !important;
}
/* Target Gradio's specific table structure */
table.table tbody td {
    text-align: center !important;
}
table.table tbody td span {
    display: block !important;
    text-align: center !important;
}
/* Make Your Wardrobe tab more compact */
#component-0 h1 {
    font-size: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}
#component-0 .label-wrap label {
    font-size: 0.85rem !important;
}
#component-0 input, #component-0 textarea {
    font-size: 0.9rem !important;
    padding: 0.4rem !important;
}
#component-0 button {
    font-size: 0.9rem !important;
    padding: 0.4rem 1rem !important;
}
"""

# You can also put dropdown choices here:
occasions = ["Work", "Interview", "Casual", "Date", "Formal", "Gym", "Brunch", "Travel", "Party"]
seasons = ["Spring (10-20°C)", "Summer (20-30°C)", "Fall (10-20°C)", "Winter (<10°C)"]

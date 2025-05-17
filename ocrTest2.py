# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import base64
import os
import streamlit as st
import json # Import the json library to parse the LLM's output
import pandas as pd # Import pandas to display items nicely

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure API key is available
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please set it.")
    st.markdown("Create a `.env` file in the same directory as this script with the line `GEMINI_API_KEY='YOUR_API_KEY_HERE'`")
    st.stop() # Stop the app if the key is missing

# Initialize the LLM
# Using gemini-1.5-flash as requested, suitable for multimodal tasks
# Lower temperature for more consistent, structured output (like JSON)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)

# --- Prompt Definition ---
# System Prompt: Sets the AI's role and overall goal
system_prompt_text = "You are an expert receipt data extractor and categorizer. Your task is to analyze a receipt image, accurately extract key financial and item details, perform calculations, and structure the output in a precise JSON format."

# Human Prompt Text: Provides detailed instructions for the task
human_prompt_text = """
Analyze the following receipt image.

**Task:**
1.  Perform Optical Character Recognition (OCR) to extract all visible text from the receipt.
2.  Identify individual purchased items and their corresponding pre-tax prices. Focus on distinct products or services listed, excluding subtotals, totals, discounts, payment information, change due, store details, dates, times, etc.
3.  Locate the total tax amount on the receipt. If multiple tax lines are present (e.g., state tax, local tax), sum them up to get the total tax. If no explicit tax line is found, assume the total tax is 0.
4.  Calculate the total pre-tax price of all identified items.
5.  Distribute the total tax proportionally among the identified items based on each item's pre-tax price relative to the total pre-tax price of all items. The formula is: `Item Tax = Total Tax * (Item Price / Total Pre-Tax Price)`. If the total pre-tax price is zero or the total tax is zero, the allocated tax for each item is 0. Round the allocated tax and final price to 2 decimal places.
6.  Categorize each identified item into one of the following categories. Use the most appropriate category, and use 'Other' if none fit well:
    *   `Food` (e.g., groceries, restaurant meals, snacks)
    *   `Drink` (e.g., beverages, coffee, soda)
    *   `Household` (e.g., cleaning supplies, toiletries, paper goods)
    *   `Apparel` (e.g., clothing, shoes, accessories)
    *   `Electronics` (e.g., gadgets, components, batteries)
    *   `Entertainment` (e.g., movies, books, games, event tickets)
    *   `Service` (e.g., car wash, haircut, consulting fee)
    *   `Bill/Utility` (e.g., phone bill payment, electricity bill payment - *less common on standard retail receipts, but include if applicable*)
    *   `Other` (Use for anything that doesn't fit the above)

**Output Format:**
Provide the result as a JSON object with the following structure. **Ensure the output is ONLY the JSON object, with no introductory or concluding text.**

```json
{{
  "raw_text": "...", // All extracted text from the receipt as a single string
  "total_tax_found": ..., // The total tax amount identified on the receipt (float)
  "items": [
    {{
      "name": "...", // Name of the item (string)
      "price_pre_tax": ..., // Price of the item before tax (float)
      "tax_allocated": ..., // The portion of the total tax allocated to this item (float, rounded to 2 decimal places)
      "price_final": ..., // The final price including allocated tax (float, rounded to 2 decimal places)
      "category": "..." // Category (string, must be one of the specified categories)
    }},
    // ... include all identified items here
  ],
  "notes": "..." // Optional: Add any notes about extraction difficulties, assumptions made (e.g., tax calculation), or items that couldn't be processed.
}}
Constraints & Considerations:


Ensure prices and tax amounts are parsed as numerical values (floats).
Be careful to distinguish items from quantities, discounts, or other non-item lines.
If an item or its price cannot be confidently identified, it should be excluded from the items list.
If no items are found, the items list should be empty [].
If no tax is found, total_tax_found should be 0.0.
Use the specified category names exactly.
Round tax_allocated and price_final to 2 decimal places.

Here is the receipt image:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_text),
        (
            "human",
            [
                {"type": "text", "text": human_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {
                        # This placeholder will be replaced by the actual base64 image data during invoke
                        "url": "data:image/jpeg;base64,{image_data}",
                        "detail": "low", # 'low' is faster/cheaper, 'high' provides more detail
                    },
                },
            ],
        ),
    ]
)

chain = prompt | llm

def encode_image(image_file):
    """Encodes an image file uploaded via Streamlit to a base64 string."""
    # Ensure the file pointer is at the beginning before reading
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode()

st.title("Receipt Data Extractor")


st.markdown("""
Upload a receipt image and the AI will extract item details, prices,
calculate proportional tax, and categorize each item.
""")

uploaded_file = st.file_uploader("Upload your receipt image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.subheader("Uploaded Receipt Image:")
    st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)


# Button to trigger the analysis
if st.button("Analyze Receipt"):
    # Encode the image to base64 after the button is clicked
    image_base64 = encode_image(uploaded_file)

    # Display a loading message while processing
    with st.spinner("Analyzing receipt..."):
        try:
            # Invoke the chain, passing the base64 image data to the placeholder
            res = chain.invoke({
                "image_data": image_base64 # Pass the base64 string here
            })

            # The LLM is instructed to return JSON, so we parse the content
            json_output_string = res.content

            # Attempt to clean up potential markdown formatting (like ```json ... ```)
            # This helps if the LLM wraps the JSON in markdown code blocks
            if json_output_string.startswith("```json"):
                 json_output_string = json_output_string[7:]
            if json_output_string.endswith("```"):
                 json_output_string = json_output_string[:-3]

            # Parse the cleaned JSON string
            extracted_data = json.loads(json_output_string)

            st.subheader("Extraction Results:")

            # Display Raw Text extracted by the LLM
            st.write("**Raw Text Extracted:**")
            # Use .get() with a default value for robustness
            st.text(extracted_data.get("raw_text", "N/A - Raw text not found in output"))

            # Display Total Tax found
            # Use .get() and format as currency
            total_tax = extracted_data.get('total_tax_found', 0.0)
            st.write(f"**Total Tax Found:** ${total_tax:.2f}")

            # Display Identified Items
            items = extracted_data.get("items", []) # Get the list of items, default to empty list
            if items:
                st.write("**Identified Items:**")
                # Convert list of dictionaries to pandas DataFrame for a nice table display
                df_items = pd.DataFrame(items)

                # Reorder columns for better readability if the DataFrame is not empty
                if not df_items.empty:
                     # Ensure all expected columns exist before reordering
                     expected_cols = ['name', 'price_pre_tax', 'tax_allocated', 'price_final', 'category']
                     # Add missing columns with default values if necessary (though prompt should ensure they exist)
                     for col in expected_cols:
                         if col not in df_items.columns:
                             df_items[col] = None # Or appropriate default like 0.0 for numbers

                     df_items = df_items[expected_cols]

                     # Format currency columns to 2 decimal places with dollar sign
                     # Use .apply(pd.to_numeric, errors='coerce') to handle potential non-numeric values gracefully
                     # Then format
                     for col in ['price_pre_tax', 'tax_allocated', 'price_final']:
                         df_items[col] = pd.to_numeric(df_items[col], errors='coerce').apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")


                st.dataframe(df_items, use_container_width=True)
            else:
                st.info("No items were identified in the receipt.")

            # Display Notes from the LLM
            notes = extracted_data.get("notes", "") # Get notes, default to empty string
            if notes:
                st.write("**Notes:**")
                st.info(notes)

            # Optional: Display the raw JSON output for debugging purposes
            with st.expander("View Raw JSON Output"):
                 st.json(extracted_data)

        except json.JSONDecodeError:
            # Handle cases where the AI's response is not valid JSON
            st.error("Failed to parse JSON response from the AI. The output might not be in the expected format.")
            st.text("Raw AI Output:")
            st.text(res.content) # Display the raw output for debugging
        except Exception as e:
            # Handle any other unexpected errors
            st.error(f"An error occurred during analysis: {e}")

else:
    # Message displayed when no file is uploaded
    st.info("Please upload a receipt image to begin.")


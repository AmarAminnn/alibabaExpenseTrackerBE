# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import base64
import os
import json
import logging # Import logging for better error reporting on the backend

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration & Initialization ---
# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure API key is available on startup
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables. Please set it.")
    # In a real app, you might want to exit or raise an exception here
    # For this example, we'll let it run but the LLM calls will fail.
    # A better approach is to check before initializing the LLM.

# Initialize the LLM (This happens once when the app starts)
# Using gemini-1.5-flash, lower temperature for more consistent output
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)
    logger.info("Gemini LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini LLM: {e}")
    llm = None # Set llm to None if initialization fails

# --- Prompt Definition ---
# System Prompt: Sets the AI's role and overall goal
system_prompt_text = "You are an expert receipt data extractor and categorizer. Your task is to analyze a receipt image, accurately extract key financial and item details, perform calculations, and structure the output in a precise JSON format."

# Human Prompt Text: Provides detailed instructions for the task
# **IMPORTANT:** Escaped curly braces {{}} for the literal JSON structure
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


try:
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
                            # This is the placeholder for the base64 image data
                            "url": "data:image/jpeg;base64,{image_data}",
                            "detail": "low", # 'low' is faster/cheaper, 'high' provides more detail
                        },
                    },
                ],
            ),
        ]
    )
    # Chain the prompt and the language model
    if llm: # Only create chain if LLM initialized successfully
        chain = prompt | llm
        logger.info("LangChain prompt and chain created.")
    else:
        chain = None
        logger.warning("LangChain chain not created because LLM initialization failed.")


except Exception as e:
    logger.error(f"Failed to create LangChain prompt/chain: {e}")
    chain = None


app = FastAPI(
    title="Receipt Data Extractor API",
    description="API to upload a receipt image and extract structured data using Gemini LLM.",
    version="1.0.0",
)

@app.post("/process-receipt/")
async def process_receipt(file: UploadFile = File(...)):
    """
    Receives a receipt image file, processes it using a Gemini LLM,
    and returns extracted item details, tax, and categories in JSON format.
    """
    if not chain:
        logger.error("API called but LangChain chain is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend service is not fully initialized (LLM or Chain failed to load)."
        )

    # 1. Receive the image file from the front-end
    # FastAPI's UploadFile handles the file upload stream

    # 2. Read the image data
    try:
        # Read the file content asynchronously
        image_bytes = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {e}"
        )

    # 3. Encode the image data to Base64
    # Gemini API expects image data in Base64 format within the prompt
    try:
        image_base64 = base64.b64encode(image_bytes).decode()
        logger.info("Image successfully encoded to Base64.")
    except Exception as e:
        logger.error(f"Error encoding image to Base64: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not encode image for processing: {e}"
        )

    # 4. Prepare data and send to LLM via LangChain
    try:
        logger.info("Invoking LangChain chain with image data...")
        # Invoke the chain, passing the base64 image data to the placeholder
        res = chain.invoke({
            "image_data": image_base64 # Pass the base64 string here
        })
        logger.info("Received response from LLM.")

        # The LLM is instructed to return JSON, so we parse the content
        json_output_string = res.content

        # 5. Parse the JSON response from the LLM
        # Attempt to clean up potential markdown formatting (like ```json ... ```)
        if json_output_string.startswith("```json"):
            json_output_string = json_output_string[7:]
        if json_output_string.endswith("```"):
            json_output_string = json_output_string[:-3]

        extracted_data = json.loads(json_output_string)
        logger.info("Successfully parsed JSON response from LLM.")

        # 6. Return the extracted data as a JSON response
        return JSONResponse(content=extracted_data)

    except json.JSONDecodeError:
        logger.error(f"LLM response was not valid JSON: {json_output_string}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI response was not valid JSON. Please try again or check the image quality."
        )
    except Exception as e:
        logger.error(f"An error occurred during AI processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during analysis: {e}"
        )
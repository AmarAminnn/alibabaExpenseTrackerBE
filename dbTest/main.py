# combined_api.py

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import base64
import os
import json
import logging
import models
from models import Expenses
from database import engine, SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# --- Database Setup ---
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

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

# Initialize the LLM (This happens once when the app starts)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY, temperature=0.1)
    logger.info("Gemini LLM initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini LLM: {e}")
    llm = None

# --- Prompt Definition ---
system_prompt_text = "You are an expert receipt data extractor and categorizer. Your task is to analyze a receipt image, accurately extract key financial and item details, perform calculations, and structure the output in a precise JSON format."

human_prompt_text = """
Analyze the following receipt image.

**Task:**
1.  Perform Optical Character Recognition (OCR) to extract all visible text from the receipt.
2.  Identify individual purchased items and their corresponding pre-tax prices. Focus on distinct products or services listed, excluding subtotals, totals, discounts, payment information, change due, store details etc.
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
                            "url": "data:image/jpeg;base64,{image_data}",
                            "detail": "low",
                        },
                    },
                ],
            ),
        ]
    )
    if llm:
        chain = prompt | llm
        logger.info("LangChain prompt and chain created.")
    else:
        chain = None
        logger.warning("LangChain chain not created because LLM initialization failed.")
except Exception as e:
    logger.error(f"Failed to create LangChain prompt/chain: {e}")
    chain = None


app = FastAPI(
    title="Combined Expense Tracker API",
    description="API for receipt processing and expense tracking",
    version="1.0.0",
)


@app.get("/expenses")
async def read_all(db: db_dependency):
    """Get all expenses from the database"""
    return db.query(Expenses).all()


@app.post("/expenses/")
async def create_expense(expense_data: dict, db: db_dependency):
    """Create a new expense in the database"""
    try:
        # Create a new Expenses object from the provided data
        new_expense = Expenses(
            item_name=expense_data.get("name"),
            price=expense_data.get("amount"),
            category=expense_data.get("category"),
            # Add other fields as needed based on your Expenses model
        )


        # Add to database and commit
        db.add(new_expense)
        db.commit()
        db.refresh(new_expense)
        
        return new_expense
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating expense: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create expense: {e}"
        )

@app.post("/process-receipt/")
async def process_receipt(db: db_dependency, file: UploadFile = File(...)):
    """
    Receives a receipt image file, processes it using a Gemini LLM,
    extracts items, adds them to the database, and returns all expenses.
    """
    if not chain:
        logger.error("API called but LangChain chain is not initialized.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Backend service is not fully initialized (LLM or Chain failed to load)."
        )


    # 1. Read the image data
    try:
        image_bytes = await file.read()
        logger.info(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {e}"
        )

    # 2. Encode the image data to Base64
    try:
        image_base64 = base64.b64encode(image_bytes).decode()
        logger.info("Image successfully encoded to Base64.")
    except Exception as e:
        logger.error(f"Error encoding image to Base64: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not encode image for processing: {e}"
        )

    # 3. Process with LLM
    try:
        logger.info("Invoking LangChain chain with image data...")
        res = chain.invoke({
            "image_data": image_base64
        })
        logger.info("Received response from LLM.")

        # Parse the JSON response
        json_output_string = res.content
        if json_output_string.startswith("```json"):
            json_output_string = json_output_string[7:]
        if json_output_string.endswith("```"):
            json_output_string = json_output_string[:-3]

        extracted_data = json.loads(json_output_string)
        logger.info("Successfully parsed JSON response from LLM.")

        # 4. Add extracted items to the database
        added_items = []
        if "items" in extracted_data and extracted_data["items"]:
            logger.info(f"Adding {len(extracted_data['items'])} items to database")
            
            for item in extracted_data["items"]:
                try:
                    # Create a new expense for each item
                    new_expense = Expenses(
                        item_name=item["name"],
                        price=item["price_final"],  # Use the final price (including tax)
                        category=item["category"]
                        # Add other fields as needed
                    )
                    
                    # Add to database
                    db.add(new_expense)
                    added_items.append(item)
                except Exception as e:
                    logger.error(f"Error adding item {item['name']} to database: {e}")
                    # Continue with other items even if one fails
            
            # Commit all additions at once
            db.commit()
            logger.info(f"Successfully added {len(added_items)} items to database")
        else:
            logger.info("No items found in receipt to add to database")

        # 5. Get all expenses (including the newly added ones)
        all_expenses = db.query(Expenses).all()
        logger.info(f"Retrieved {len(all_expenses)} expenses from database")
        
        # Convert SQLAlchemy objects to dictionaries for JSON serialization
        expenses_list = []
        for expense in all_expenses:
            # Adjust this based on your Expenses model attributes
            expenses_list.append({
                "id": expense.id,
                "item_name": expense.item_name,
                "price": expense.price,
                "category": expense.category
                # Add other fields as needed
            })

        # 6. Combine the data
        combined_response = {
            "receipt_data": extracted_data,
            "added_items": added_items,
            "all_expenses": expenses_list
        }

        return JSONResponse(content=combined_response)

    except json.JSONDecodeError:
        logger.error(f"LLM response was not valid JSON: {json_output_string}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI response was not valid JSON. Please try again or check the image quality."
        )
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during analysis: {e}"
        )

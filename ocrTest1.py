from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import base64
import os
import streamlit as st
from datetime import date # Import date for st.date_input default value

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure API key is available
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables.")
    st.stop() # Stop the app if the key is missing

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)

# Define the prompt template with a placeholder for the image data
# We'll use '{image_data}' as the placeholder name
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can verify identification documents by comparing provided text details with the image."),
        (
            "human",
            [
                {"type": "text", "text": "Verify the identification details based on the image."},
                {"type": "text", "text": "Provided Name: {user_name}"},
                {"type": "text", "text": "Provided DOB: {user_dob}"},
                {
                    "type": "image_url",
                    "image_url": {
                        # Use the placeholder here instead of the variable 'image'
                        "url": "data:image/jpeg;base64,{image_data}",
                        "detail": "low",
                    },
                },
                {"type": "text", "text": "Does the name and date of birth provided match the information visible in the document image? Respond with 'Match' or 'No Match' and a brief explanation."},
            ],
        ),
    ]
)

# Chain the prompt and the language model
chain = prompt | llm

def encode_image(image_file):
    # Ensure the file pointer is at the beginning
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode()

st.title("KYC Verification Application")

col1, col2 = st.columns(2)

uploaded_file = None # Initialize uploaded_file outside the column block

with col1:
    uploaded_file = st.file_uploader("Upload your ID document", type=["jpg", "png", "jpeg"]) # Added jpeg type

with col2:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Document", use_column_width=True)
    else:
        st.write("Image Preview")

user_name = st.text_input("Enter your name")
# Set a default value for date_input, e.g., today's date or a common past date
user_dob = st.date_input("Enter your date of birth", value=date(2000, 1, 1)) # Example default date

# Only run the chain when all inputs are available
if uploaded_file is not None and user_name and user_dob:
    # Encode the image *after* it's uploaded
    image_base64 = encode_image(uploaded_file)

    # Display a loading message while processing
    with st.spinner("Verifying document..."):
        # Invoke the chain, passing the actual base64 image data
        # as the value for the 'image_data' placeholder defined in the prompt
        res = chain.invoke({
            "user_name": user_name,
            "user_dob": user_dob.strftime("%Y-%m-%d"), # Format date as string
            "image_data": image_base64 # Pass the base64 string here
        })

    st.subheader("Verification Result:")
    st.write(res.content)

# Optional: Add instructions or notes
st.markdown("""
<small>
**Note:** This is a demo application using AI for document verification.
Always follow official KYC procedures for real-world applications.
</small>
""", unsafe_allow_html=True)
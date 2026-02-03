🤖 AI Ticket Triage System

This Streamlit application uses Google Gemini (via the Agno/Phidata framework) to automatically categorize customer support tickets from a CSV file based on a provided Support Policy PDF.

## 🚀 Quick Start for Graders

To run this application locally or in a cloud environment, follow these steps:

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt

2. Configuration (API Key)
​This app requires a Google Gemini api key
 Create a folder named .streamlit in the root directory.
 Create a file inside it named secrets.toml.
Add your key like this:
 GOOGLE_API_KEY = "your_actual_api_key_here"

 3. Launch the App
 streamlit run Streamlit_app.py

 🛠️ Key Features & Regional Optimizations
​During development, this app was specifically optimized to handle Regional API Constraints (India) and modern v1 Stable requirements:

​API v1 Migration: The app has been migrated from the deprecated v1beta to the stable v1 endpoint using the google-genai library to prevent 404 errors common in the South Asia region.

​Model Discovery Suite: Added a sidebar tool to 🔎 List Available Models and 🧪 Test Selected Model. This allows the user to verify if a specific model ID (like gemini-1.5-flash) is active and has remaining quota before running the full triage.

​Auto-Select Logic: Includes a "Smart Recovery" feature that can automatically test and find the best available working model for your specific API key.

​Agent Resilience: Resolved a NameError in the agent constructor to ensure the Triage Agent initializes correctly with the selected model parameters.

​📁 File Requirements
​Support Policy: Upload a .pdf file containing the rules for categorization.
​Tickets Data: Upload a .csv file with a Ticket Description or Subject column.
​🧑‍💻 Technical Stack

​Frontend: Streamlit
​Orchestration: Agno (Phidata)
​Model: Google Gemini 1.5 Flash (v1 Stable)
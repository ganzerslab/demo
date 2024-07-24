import streamlit as st
from openai import OpenAI
import re
import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import json

# Set page config at the very beginning
st.set_page_config(page_title="Fraud Analysis Platform", layout="wide")

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def clear_analysis_results():
    st.session_state.checklist_items = []
    st.session_state.ai_responses = {}
    if 'ai_findings' in st.session_state:
        del st.session_state.ai_findings
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text, num_lines=20):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_lines:
        return ' '.join(sentences)

    step = max(1, len(sentences) // num_lines)
    summary = ' '.join(sentences[::step][:num_lines])

    return summary

@st.cache_resource
def process_knowledge_base():
    knowledge_base = ""
    knowledge_base_dir = 'knowledge_base'

    if not os.path.exists(knowledge_base_dir):
        print(f"Warning: '{knowledge_base_dir}' folder not found. Proceeding with empty knowledge base.")
        return knowledge_base

    pdf_files = [f for f in os.listdir(knowledge_base_dir) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"Warning: No PDF files found in '{knowledge_base_dir}'. Proceeding with empty knowledge base.")
        return knowledge_base

    for pdf_file in pdf_files[:5]:
        pdf_path = os.path.join(knowledge_base_dir, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            summary = summarize_text(text, num_lines=20)
            knowledge_base += f"Summary of {pdf_file}:\n{summary}\n\n"
            print(f"Processed: {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    return knowledge_base

@st.cache_data
def load_transaction_data():
    transaction_data = {}
    transaction_dir = 'transactions'
    if os.path.exists(transaction_dir):
        for file in os.listdir(transaction_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(transaction_dir, file)
                df = pd.read_csv(file_path)
                transaction_data[file] = df
    return transaction_data

# Process knowledge base and load transaction data once at startup
KNOWLEDGE_BASE = process_knowledge_base()
TRANSACTION_DATA = load_transaction_data()

def get_llm_response(api_key, goals, analysis_type, task=None, transaction_data=None):
    if task:
        prompt = f"Perform the following task related to fraud analysis: {task}\n\nProvide a response indicating if the task was completed, and any follow-up tasks if they exist."
    else:
        if analysis_type == "Image Analysis":
            prompt = f"""As a fraud analyst, break down the following automation goals into high-level findings and detailed, actionable checklist tasks for analyzing a suspicious check image. Refer to the provided example of fraud vectors and key challenges to guide your breakdown:

Automation Goals: {goals}

Example Fraud Vectors and Key Challenges:
- Checks: Counterfeit and stolen checks, lack of mature consortium solutions, outdated breach of warranty rules.
- ACH: Originators being hit with ACH returns, WSUD adoption by Pay by Bank providers.
- ATO: End users being compromised, phishing up since the launch of Chat GPT in Nov 22.
- Biz Email Compromise: Exposure to ACH fraud, not limited to ACH - payment form factor agnostic.
- Elder Exploitation: APP fraud - romance and investment scams, use of AI to present 'family at risk' scenarios.
- Synthetic Identity Fraud: Creation and use of fictitious identities combining real and fake information.
- Card-Not-Present Fraud: Unauthorized transactions in e-commerce and over-the-phone purchases.
- Money Mule Schemes: Use of intermediary accounts to launder fraudulently obtained funds.
- First-Party Fraud: Customers making purchases or taking out loans with no intention to pay.
- Insider Threats: Employees exploiting their access to commit or facilitate fraud.
- Crypto-Related Fraud: Scams and fraudulent activities involving cryptocurrencies and blockchain technology.
- Real-Time Payment Fraud: Exploiting the speed of instant payment systems for fraudulent transactions.

Additional Knowledge:
{KNOWLEDGE_BASE}

Your tasks should address these challenges using tools like image analysis, ML for anomaly detection, rule creation for standard fraud schemes, monitoring RDI ratios, and leveraging consortium data.
Your tasks should also be tasks that an AI can run.

Provide your response in the following format:
INSIGHTS:
[List high-level insights here]

CHECKLIST:
[List detailed, actionable checklist items here. Ensure all tasks are actionable and avoid empty items. Use bullet points for tasks.]
"""
        else:
            prompt = f"""As a fraud analyst, break down the following automation goals into high-level findings and detailed, actionable checklist tasks for analyzing transaction data. Refer to the provided example of fraud vectors and key challenges to guide your breakdown:

Automation Goals: {goals}

Transaction Data Summary:
{transaction_data.describe().to_string() if transaction_data is not None else "No transaction data provided"}

Example Fraud Vectors and Key Challenges:
- Unusual Transaction Patterns: Sudden changes in transaction frequency or amounts.
- Money Laundering: Series of transactions designed to obscure the source of funds.
- Account Takeover: Unexpected changes in account behavior or access from new locations.
- First Party Fraud: Customers making purchases or taking out loans with no intention to pay.
- Structuring: Multiple transactions just below reporting thresholds.
- Insider Threats: Unusual employee account activities or access patterns.

Your tasks should address these challenges using tools like statistical analysis, machine learning for anomaly detection, pattern recognition, and leveraging historical transaction data.
Your tasks should also be tasks that an AI can run on the transaction data.

Provide your response in the following format:
INSIGHTS:
[List high-level insights here]

CHECKLIST:
[List detailed, actionable checklist items here. Ensure all tasks are actionable and avoid empty items. Use bullet points for tasks.]
"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in fraud analysis."},
        {"role": "user", "content": prompt}
    ]

    # Print raw prompt to terminal
    print("\n--- RAW PROMPT ---")
    print(json.dumps(messages, indent=2))
    print("--- END RAW PROMPT ---\n")

    response = client.chat.completions.create(model="gpt-4o",
    messages=messages)

    # Print raw response to terminal
    print("\n--- RAW RESPONSE ---")
    print(json.dumps(response.model_dump(), indent=2))
    print("--- END RAW RESPONSE ---\n")

    return response.choices[0].message.content

def parse_checklist(checklist_text):
    lines = checklist_text.split('\n')
    parsed_items = []
    current_category = None
    current_subcategory = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('CHECKLIST:', 'Checklist:')):
            if re.match(r'^\d+\.', line) or (not line.startswith('-') and not line.startswith('•') and ':' in line):  # Main category
                current_category = re.sub(r'^\d+\.\s*', '', line)
                current_subcategory = None
                parsed_items.append({"text": current_category, "is_category": True, "level": 1})
            elif line.startswith('-') or line.startswith('•'):  # Task or subcategory
                item = re.sub(r'^[-•]\s*', '', line)
                if ':' in item:  # Subcategory
                    current_subcategory = item
                    parsed_items.append({"text": item, "is_category": True, "level": 2, "parent": current_category})
                else:  # Task
                    parsed_items.append({
                        "text": item,
                        "completed": False,
                        "sent_to_agent": False,
                        "is_category": False,
                        "category": current_category,
                        "subcategory": current_subcategory
                    })
    return parsed_items

# Initialize session state
if 'checklist_items' not in st.session_state:
    st.session_state.checklist_items = []
if 'ai_responses' not in st.session_state:
    st.session_state.ai_responses = {}

# Sidebar
st.sidebar.title("Configuration")

# API key input
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
client = OpenAI(api_key=api_key)

# Analysis type selection
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = "Image Analysis"

analysis_type = st.sidebar.radio("Select Analysis Type", ["Image Analysis", "Transaction Analysis"])

if analysis_type != st.session_state.current_analysis_type:
    clear_analysis_results()
    st.session_state.current_analysis_type = analysis_type

if analysis_type == "Image Analysis":
    uploaded_file = st.sidebar.file_uploader("Upload Check Image", type=["png", "jpg", "jpeg"])
else:
    if TRANSACTION_DATA:
        selected_file = st.sidebar.selectbox("Select Transaction File", list(TRANSACTION_DATA.keys()))
    else:
        st.sidebar.warning("No transaction data files found in the 'transactions' folder.")

# Default automation goals
if analysis_type == "Image Analysis":
    default_goals = """I have a suspicious counterfeit check that I want to analyze, using the following automation goals:
    -process the check image for anomalies.
    -monitor RDI ratios.
    -use consortium data."""
else:
    default_goals = """Analyze the transaction data for potential fraud, considering the following goals:
    -identify unusual transaction patterns.
    -detect potential money laundering activities.
    -flag high-risk transactions based on amount and frequency."""

goals = st.sidebar.text_area("Enter Automation Goals", value=default_goals)

# Display knowledge base status
st.sidebar.subheader("Knowledge Base Status")
if KNOWLEDGE_BASE:
    st.sidebar.success("Knowledge base loaded successfully.")
else:
    st.sidebar.warning("No knowledge base found. Using default information only.")

# Main content
st.title("giniX Fraud Analysis Demo")

if st.sidebar.button("Generate Analysis"):
    if api_key and goals:
        with st.spinner("Analyzing..."):
            if analysis_type == "Image Analysis":
                if uploaded_file is not None:
                    response = get_llm_response(api_key, goals, analysis_type)
                else:
                    st.warning("Please upload an image for analysis.")
                    st.stop()
            else:
                if TRANSACTION_DATA and selected_file:
                    transaction_data = TRANSACTION_DATA[selected_file]
                    response = get_llm_response(api_key, goals, analysis_type, transaction_data=transaction_data)
                else:
                    st.warning("No transaction data available for analysis.")
                    st.stop()

            # Split response into insights and checklist
            try:
                insights, checklist = response.split("CHECKLIST:")
                insights = insights.replace("INSIGHTS:", "").strip()
                checklist = checklist.strip()

                # Store AI findings in session state
                st.session_state.ai_findings = insights

                # Parse and store checklist
                parsed_items = parse_checklist(checklist)
                if parsed_items:
                    st.session_state.checklist_items = parsed_items
                else:
                    st.warning("No checklist items were parsed. Check the raw response for formatting issues.")

            except ValueError:
                st.error("Unable to split response into insights and checklist. Check the raw response above.")
                st.session_state.ai_findings = response
                st.session_state.checklist_items = []

    else:
        st.warning("Please enter both API key and automation goals.")

# Display AI Findings
if 'ai_findings' in st.session_state and st.session_state.ai_findings:
    st.subheader("AI Findings")
    st.write(st.session_state.ai_findings)
else:
    st.info("No AI findings available. Generate an analysis to see insights.")

# Display checklist
if st.session_state.checklist_items:
    st.subheader("Checklist")
    current_category = None
    current_subcategory = None
    for i, item in enumerate(st.session_state.checklist_items):
        if item['is_category']:
            if item['level'] == 1:
                st.markdown(f"### {item['text']}")
                current_category = item['text']
            else:
                st.markdown(f"#### {item['text']}")
                current_subcategory = item['text']
        else:
            col1, col2, col3 = st.columns([1, 18, 1])

            completed = col1.checkbox("", key=f"checkbox_{i}", value=item['completed'])
            if completed != item['completed']:
                st.session_state.checklist_items[i]['completed'] = completed
                st.experimental_rerun()

            status = ""
            if item['completed']:
                status = "✅ Completed"
            elif item['sent_to_agent']:
                status = "🔄 Sent to agent"

            col2.write(f"{item['text']} {status}")

            if col3.button("AI", key=f"ai_{i}"):
                st.session_state.checklist_items[i]['sent_to_agent'] = True
                with st.spinner("Processing task..."):
                    context = f"{current_category}"
                    if current_subcategory:
                        context += f" - {current_subcategory}"
                    response = get_llm_response(api_key, goals, analysis_type, f"{context}: {item['text']}")
                st.session_state.ai_responses[i] = response
                st.experimental_rerun()

        if i in st.session_state.ai_responses:
            with st.expander(f"AI Response for: {item['text']}"):
                st.markdown(f"<div style='background-color: #ffffd0; padding: 10px; border-radius: 5px;'>{st.session_state.ai_responses[i]}</div>", unsafe_allow_html=True)
else:
    st.info("No checklist items available. Generate an analysis to create a checklist.")

# Display uploaded image or transaction data summary
if analysis_type == "Image Analysis":
    if uploaded_file is not None:
        st.subheader("Uploaded Check Image")
        st.image(uploaded_file, caption="Uploaded Check", use_column_width=True)
    else:
        st.info("No check image uploaded. You can upload an image for analysis in the sidebar.")
else:
    if TRANSACTION_DATA and selected_file:
        st.subheader("Transaction Data Summary")
        st.write(TRANSACTION_DATA[selected_file].describe())
    else:
        st.info("No transaction data available. Please add CSV files to the 'transactions' folder.")

# Action buttons
st.subheader("Actions")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.button("Flag as Suspicious")
with col2:
    st.button("Mark as False Positive")
with col3:
    st.button("Clear Transaction")
with col4:
    st.button("Escalate to Senior Analyst")

# Add some space
st.write("")

# Demo disclaimer
st.info("This is a demo version of the Fraud Analysis Platform.")
import streamlit as st
import openai
import re

# Function to get LLM response
def get_llm_response(api_key, goals, task=None):
    openai.api_key = api_key
    if task:
        prompt = f"Perform the following task related to fraud analysis: {task}\n\nProvide a response indicating if the task was completed, and any follow-up tasks if they exist."
    else:
        prompt = f"""As a fraud analyst, break down the following automation goals into high-level findings and detailed, actionable checklist tasks. Refer to the provided example of fraud vectors and key challenges to guide your breakdown:

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

Your tasks should address these challenges using tools like image analysis, ML for anomaly detection, rule creation for standard fraud schemes, monitoring RDI ratios, and leveraging consortium data.
Your tasks should also be tasks that an AI can run.

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

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )

    return response.choices[0].message['content']

# Function to parse checklist items
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

# Streamlit app
st.set_page_config(page_title="Fraud Analysis Platform", layout="wide")

# Initialize session state
if 'checklist_items' not in st.session_state:
    st.session_state.checklist_items = []
if 'ai_responses' not in st.session_state:
    st.session_state.ai_responses = {}

# Sidebar
st.sidebar.title("Configuration")

# API key input
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Default automation goals
default_goals = """I have a suspicious counterfeit check that I want to analyze, using the following automation goals:
-process the check image for anomalies.
-monitor RDI ratios.
-use consortium data."""
goals = st.sidebar.text_area("Enter Automation Goals", value=default_goals)

uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Main content
st.title("giniX Fraud Analysis Demo")

if st.sidebar.button("Generate Analysis"):
    if api_key and goals:
        with st.spinner("Analyzing..."):
            response = get_llm_response(api_key, goals)

            # Split response into insights and checklist
            try:
                insights, checklist = response.split("CHECKLIST:")
                insights = insights.replace("INSIGHTS:", "").strip()
                checklist = checklist.strip()
            except ValueError:
                st.error("Unable to split response into insights and checklist. Check the raw response above.")
                insights = response
                checklist = ""

            # Display insights
            st.subheader("AI Findings")
            st.write(insights)

            # Parse and display checklist
            st.subheader("Checklist")
            parsed_items = parse_checklist(checklist)
            if parsed_items:
                st.session_state.checklist_items = parsed_items
            else:
                st.warning("No checklist items were parsed. Check the raw response for formatting issues.")

    else:
        st.warning("Please enter both API key and automation goals.")

# Display checklist
if st.session_state.checklist_items:
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
                    response = get_llm_response(api_key, goals, f"{context}: {item['text']}")
                st.session_state.ai_responses[i] = response
                st.experimental_rerun()

        if i in st.session_state.ai_responses:
            with st.expander(f"AI Response for: {item['text']}"):
                st.markdown(f"<div style='background-color: #ffffd0; padding: 10px; border-radius: 5px;'>{st.session_state.ai_responses[i]}</div>", unsafe_allow_html=True)
else:
    st.warning("No checklist items available. Try generating the analysis again.")

# Display uploaded image
if uploaded_file is not None:
    st.subheader("Uploaded Check Image")
    st.image(uploaded_file, caption="Uploaded Check", use_column_width=True)
else:
    st.info("No check image uploaded. You can upload an image for analysis in the sidebar.")

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
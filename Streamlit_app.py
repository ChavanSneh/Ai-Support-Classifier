import streamlit as st
import pandas as pd
import pdfplumber
import os
from agno.agent import Agent
from agno.models.google import Gemini

# Required helper: use the newer v1 client `google.genai` only (no fallback)

# This app targets API version v1 and will stop if `google.genai` is not installed.
try:
    import google.genai as genai_v1  # v1 client
    HAVE_GENAI_V1 = True
except Exception:
    genai_v1 = None
    HAVE_GENAI_V1 = False

# API version used in this app
PREFERRED_API_VERSION = 'v1'  # google.genai (required)


def choose_best_model(models):
    """Choose the best model from a models iterable returned by genai.list_models().

    Preference is given to:
      1) models whose id/name matches common production names (gemini-1.5-pro*, gemini-1.5, text-bison, chat-bison)
      2) models that advertise supported methods like 'generateContent' or 'generate'

    Returns the best model id string, or None if none found.
    """
    preferred_names = [
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-1.5",
        "chat-bison@001",
        "text-bison@001",
        "chat-bison",
        "text-bison",
    ]

    # lowercased priorities for robust matching

    method_priority = ["generatecontent", "generate", "chat", "text"]

    name_hits = []
    method_hits = []

    for m in models:
        mid = getattr(m, "name", None) or getattr(m, "model", None) or str(m)
        mid_l = str(mid).lower()
        display = (getattr(m, 'display_name', '') or getattr(m, 'description', '') or '').lower()
        supported = getattr(m, "supported_methods", None) or []
        try:
            # Normalize supported methods to str list for membership checks
            supported = [str(x).lower() for x in supported]
        except Exception:
            supported = [str(x).lower() for x in supported] if supported else []

        # check for strong method matches first
        if any(pr in s for s in supported for pr in method_priority):
            method_hits.append(mid)
            continue

        # name-based matching (includes resource-name formats)
        if any(pref in mid_l for pref in preferred_names) or any(pref in display for pref in preferred_names):
            name_hits.append(mid)
            continue

        # fallback: model name contains common engine words
        if any(token in mid_l for token in ["bison", "gemini", "chat", "text"]):
            name_hits.append(mid)
            continue

    if method_hits:
        return method_hits[0]
    if name_hits:
        return name_hits[0]
    return None


def make_triage_agent(model_id):
    """Construct the triage agent using the selected model id."""
    return Agent(
        name="YNC Triage Agent",
        model=Gemini(id=model_id),
        description="Automated Triage Agent for YNC E-commerce support.",
        instructions=[
            "Categorize tickets into: Refund, Shipping, Account, Product, or Spam.",
            "Extract Sentiment (Positive/Negative) and Urgency (High/Medium/Low).",
            "Suggest a polite draft response based on company policy.",
            "Highlight cases needing escalation to a human supervisor."
        ],
        markdown=True
    )


def test_selected_model(model_id, timeout_seconds=10):
    """Quickly validate that a model can be used by creating an Agent and making a short call.

    This function prefers Google GenAI v1 client when available (explicitly using API v1).
    It falls back to using the existing Agent approach if direct v1 calls aren't possible.

    Returns a tuple: (True, message) on success or (False, error_message) on failure.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")

    # First try using the newer google.genai v1 client if available
    if HAVE_GENAI_V1 and api_key:
        try:
            client = genai_v1.Client(api_key=api_key)
            # Try chat-based generation if available
            try:
                if hasattr(client.chats, 'create'):
                    resp = client.chats.create(model=model_id, messages=[{"role": "user", "content": "Triage this issue: Quick connectivity test."}], temperature=0.0)
                    # Best-effort to extract textual content
                    preview = getattr(resp, 'candidates', None) or getattr(resp, 'output', None) or str(resp)
                    return True, str(preview)[:400]
            except Exception:
                pass

            # Try generic generate/predict if available
            try:
                if hasattr(client, 'generate'):
                    resp = client.generate(model=model_id, input="Triage this issue: Quick connectivity test.")
                    return True, str(resp)[:400]
            except Exception:
                pass

        except Exception as e:
            # fall through to agent-based testing
            fallback_err = str(e)

    # Fallback: use the Agent (which will use whatever supported client is configured)
    try:
        agent = make_triage_agent(model_id)
    except Exception as e:
        return False, f"Failed to construct agent: {e}"

    try:
        resp = agent.run("Triage this issue: Quick connectivity test.")
        preview = getattr(resp, 'content', str(resp))
        return True, str(preview)[:400]
    except Exception as e:
        # include fallback error if present to help debugging
        detail = f" ({fallback_err})" if 'fallback_err' in locals() else ''
        return False, f"Model call failed: {e}{detail}"

# --- 1. MUST BE FIRST (Fixes StreamlitAPIException) ---
st.set_page_config(page_title="YNC Triage Dashboard", layout="wide")

# --- 2. SECURITY & API KEY (Fixes KeyError) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set the GOOGLE_API_KEY in .streamlit/secrets.toml")
    st.stop() # Stops execution here so the app doesn't crash later

# Enforce that the v1 client (google.genai) is installed — no fallbacks allowed for v1 deployments
if not HAVE_GENAI_V1:
    st.error("This app requires the 'google.genai' package (v1). Install it with `pip install google-genai` and restart the app.")
    st.stop()

# --- 3. AGENT CONFIGURATION ---
# Allow choosing a model from the sidebar so users can select a supported model.
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_id = st.selectbox(
        "Model ID",
        options=[
            "gemini-1.5-flash-002",
            "gemini-2.0-flash",
        ],
        index=0,
        key="model_id",
        help="Default is 'gemini-1.5-flash-002'. These models are stable (v1) in your region—use 'List available models' if you get a 404."
    )

    st.caption(f"Using API version: {PREFERRED_API_VERSION} (requires google.genai v1)")

    if st.button("🧪 Test selected model"):
        mid = st.session_state.get('model_id')
        ok, msg = test_selected_model(mid)
        if ok:
            st.success(f"Model {mid} responded: {msg}")
        else:
            st.error(f"Model {mid} failed test: {msg}")
            st.info("Try listing available models and use the discovered models or auto-select to find a compatible one.")

    if st.button("🔎 List available models"):
        # List models using google.genai (v1) client only
        def _list_in_ui():
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                st.error("GOOGLE_API_KEY is not set. Add it to .streamlit/secrets.toml or export it in the environment.")
                return
            try:
                client = genai_v1.Client(api_key=api_key)
                rows = []
                for m in client.models.list():
                    model_id_val = getattr(m, 'name', None) or getattr(m, 'id', None) or str(m)
                    display = getattr(m, 'display_name', '') or getattr(m, 'description', '') or ''
                    # Some v1 model objects may include supported methods info
                    methods = getattr(m, 'supported_methods', None) or []
                    rows.append({"id": model_id_val, "display_name": display, "supported_methods": methods})

                if not rows:
                    st.info("No models returned for this project/key.")
                    return

                st.table(rows)

                # Save discovered models for later use (auto-select fallback, manual choose)
                discovered = [r['id'] for r in rows]
                st.session_state['available_models'] = discovered

                if discovered:
                    st.info(f"Discovered {len(discovered)} model(s). You can auto-select or pick one below to use.")
                    choice = st.radio("Choose a discovered model to test", discovered, key='discovered_choice')
                    if st.button("Use selected discovered model"):
                        ok, msg = test_selected_model(choice)
                        if ok:
                            st.session_state['model_id'] = choice
                            st.success(f"Selected model: {choice} (test succeeded)")
                            st.experimental_rerun()
                        else:
                            st.error(f"Selected model failed test: {msg}")
                            st.info("Try another discovered model or use auto-select to attempt several models automatically.")

                            # Keep the available_models in session_state for retrying later
                            
            except Exception as e:
                st.error(f"Failed to list models using google.genai: {e}")
                st.info("Check that the API key has Generative AI access and that the key is for the correct project and region.")

        _list_in_ui()

    if st.button("🤖 Auto-select best model"):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY is not set. Add it to .streamlit/secrets.toml or export it in the environment.")
        else:
            try:
                # Auto-select using google.genai v1 client (required)
                client = genai_v1.Client(api_key=api_key)
                models = list(client.models.list())

                # Try the preferred best model first (if any), but validate it with a test call.
                best = choose_best_model(models)
                tried = []
                failures = []
                if best:
                    ok, msg = test_selected_model(best)
                    if ok:
                        st.session_state['model_id'] = best
                        st.success(f"Auto-selected model: {best} (validated)")
                        st.experimental_rerun()
                    else:
                        failures.append((best, msg))
                        tried.append(best)

                # If we got here, try discovered models sequentially until one works (limit to first 8)
                max_attempts = 8
                count = 0
                for m in models:
                    mid = getattr(m, 'name', None) or getattr(m, 'model', None) or getattr(m, 'id', None) or str(m)
                    if mid in tried:
                        continue
                    if count >= max_attempts:
                        break
                    ok, msg = test_selected_model(mid)
                    count += 1
                    if ok:
                        st.session_state['model_id'] = mid
                        st.success(f"Auto-selected model by testing: {mid}")
                        st.experimental_rerun()
                    else:
                        failures.append((mid, msg))

                # If none succeeded, persist discovered ids and summarize failures
                discovered_ids = [getattr(m, 'name', None) or getattr(m, 'model', None) or getattr(m, 'id', None) or str(m) for m in models]
                st.session_state['available_models'] = discovered_ids
                st.warning("Auto-selection: no tested models succeeded. See failures below and consider trying a discovered model manually.")
                for fid, ferr in failures[:10]:
                    st.info(f"Tried {fid}: {ferr}")
                st.info("You can pick a discovered model from the list shown by 'List available models' or try the discovered choices below.")
                if discovered_ids:
                    sel = st.selectbox("Pick a discovered model to try", discovered_ids, key='fallback_choice')
                    if st.button("Use this model"):
                        ok, msg = test_selected_model(sel)
                        if ok:
                            st.session_state['model_id'] = sel
                            st.success(f"Selected model: {sel} (validated)")
                            st.experimental_rerun()
                        else:
                            st.error(f"Model {sel} failed test: {msg}")
                            st.info("Try another discovered model or check your API key/project permissions.")
            except Exception as e:
                st.error(f"Auto-selection failed: {e}")
                st.info("Ensure the API key has access to Generative AI for your project.")


def make_triage_agent(model_id):
    """Construct the triage agent using the selected model id."""
    return Agent(
        name="YNC Triage Agent",
        model=Gemini(id=model_id),
        description="Automated Triage Agent for YNC E-commerce support.",
        instructions=[
            "Categorize tickets into: Refund, Shipping, Account, Product, or Spam.",
            "Extract Sentiment (Positive/Negative) and Urgency (High/Medium/Low).",
            "Suggest a polite draft response based on company policy.",
            "Highlight cases needing escalation to a human supervisor."
        ],
        markdown=True
    )

# --- 4. DATA PREPROCESSING ---
def process_policy_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

# --- 5. UI SETUP & WORKFLOW ---
st.title("🤖 YNC Customer Support Triage")

with st.sidebar:
    st.header("📂 Data Ingestion")
    support_logs = st.file_uploader("Upload Support Logs (CSV)", type=["csv"])
    policy_doc = st.file_uploader("Upload Policy (PDF)", type=["pdf"])

if support_logs:
    df = pd.read_csv(support_logs)
    
    # Simple spam filter logic
    df['is_spam'] = df.iloc[:, 0].astype(str).str.contains('spam_', case=False, na=False)
    valid_tickets = df[df['is_spam'] == False]
    
    st.subheader("📋 Incoming Tickets")
    st.dataframe(valid_tickets.head())

    if st.button("🚀 Run AI Triage Analysis"):
        st.header("🔍 AI Triage Results")
        # Create the agent lazily so model selection can be changed without breaking the app on load
        try:
            triage_agent = make_triage_agent(model_id)
        except Exception as e:
            st.error(f"Failed to create agent with model '{model_id}': {e}")
            st.info("Tip: Try a different model (e.g., 'text-bison@001') or check that your GOOGLE_API_KEY has access to the model.")
            st.stop()

        for index, row in valid_tickets.head(5).iterrows():
            issue_text = str(row.values[0])
            with st.spinner(f"Analyzing Ticket #{index+1}..."):
                try:
                    response = triage_agent.run(f"Triage this issue: {issue_text}")
                except Exception as e:
                    st.error(f"Model call failed: {e}")
                    st.info("Common fixes: choose a different model in the sidebar, ensure the GOOGLE_API_KEY is valid and enabled for Generative AI, or list available models in your project.")
                    break
                with st.expander(f"Ticket Analysis: {issue_text[:50]}..."):
                    st.markdown(response.content)

        st.divider()
        st.header("📊 Supervisor Insights")
        try:
            summary = triage_agent.run("Summarize the top 3 customer pain points from the analyzed tickets.")
            st.info(summary.content)
        except Exception as e:
            st.error(f"Summary generation failed: {e}")
            st.info("If this persists, try choosing another model in the sidebar.")
else:
    st.info("Upload a CSV to start triaging tickets.")
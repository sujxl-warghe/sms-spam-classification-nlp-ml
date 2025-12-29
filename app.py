import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import imaplib
import email
import re
from bs4 import BeautifulSoup

# --- Page Config ---
st.set_page_config(
    page_title="💌 Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; }
    .stTextArea textarea { border-radius: 10px; border: 1px solid #ccc; }
    .stButton button { background-color: #ff4b4b; color: white; border-radius: 10px; padding: 10px 20px; font-weight: bold; }
    .stButton button:hover { background-color: #ff2e2e; }
    .prediction-good { padding: 15px; border-radius: 10px; background-color: #d4edda; color: #155724; font-size: 18px; font-weight: bold; }
    .prediction-bad { padding: 15px; border-radius: 10px; background-color: #f8d7da; color: #721c24; font-size: 18px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# --- Detect mobile ---
try:
    width = st.runtime.scriptrunner.script_run_context.session_data.width
    is_mobile = width < 600
except:
    is_mobile = False  # fallback if detection fails

# --- Clean email body ---
def clean_email_body(body):
    body = BeautifulSoup(body, "html.parser").get_text()
    body = re.sub(r'\[image.*?\]', '', body)
    body = re.sub(r'Content-Type.*?\n', '', body)
    body = re.sub(r'http\S+', ' ', body)
    body = re.sub(r'\s+', ' ', body).strip()
    return body

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()
    y = [ps.stem(i) for i in text]
    return " ".join(y)

# Load model/vectorizer
etc = pickle.load(open('vectorizer-1.pkl', 'rb'))
rfc = pickle.load(open('model_rfc.pkl', 'rb'))

# --- Session State ---
if 'spam_emails' not in st.session_state: st.session_state['spam_emails'] = []
if 'not_spam_emails' not in st.session_state: st.session_state['not_spam_emails'] = []
if 'last5_emails' not in st.session_state: st.session_state['last5_emails'] = []
if 'summary' not in st.session_state: st.session_state['summary'] = {}

# --- Title ---
st.markdown("<h1 style='text-align: center;'>💌 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>🚀 Detect spam emails & SMS instantly using Machine Learning!</h5>", unsafe_allow_html=True)

# --- Manual input ---
st.subheader("✍️ Test a Custom Message")
input_sms = st.text_area('Enter your message here...')
if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = etc.transform([transformed_sms])
    proba = rfc.predict_proba(vector_input)[0][1]
    result = rfc.predict(vector_input)[0]

    if result == 1:
        st.markdown('<div class="prediction-bad">⚠️ This looks like SPAM!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-good">✅ This message is SAFE (Not Spam).</div>', unsafe_allow_html=True)
    st.write(f"🔍 Spam Confidence: **{proba:.2f}**")

# --- Gmail Fetch ---
st.markdown("<h1 style='text-align:center;'>📧 Fetch & Classify Gmail</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>🚀 Check your latest Emails through Login..</h5>", unsafe_allow_html=True)

# --- App Password link ---
st.markdown("""
    <p style='text-align:center; font-size:14px;'>
    Don't have an App Password? 
    <a href='https://support.google.com/accounts/answer/185833?hl=en' target='_blank' style='color:#ff4b4b;'>Click here to create one</a>
    </p>
""", unsafe_allow_html=True)

# --- Email login ---
email_user = st.text_input("Email ID")
email_pass = st.text_input("App Password", type="password")

if st.button("Fetch & Analyze Emails"):
    try:
        with st.spinner("📩 Fetching emails from Gmail..."):
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(email_user, email_pass)
            mail.select('inbox')

            status, messages = mail.search(None, 'ALL')
            email_ids = messages[0].split()
            total_emails = len(email_ids)

            spam_count = 0
            not_spam_count = 0
            email_data = []
            spam_emails = []
            not_spam_emails = []

            for idx, i in enumerate(email_ids):
                status, msg_data = mail.fetch(i, '(RFC822)')
                msg = email.message_from_bytes(msg_data[0][1])

                # Extract plain text
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")

                clean_body = clean_email_body(body)
                transformed = transform_text(clean_body)
                vector_input = etc.transform([transformed])
                prediction = rfc.predict(vector_input)[0]
                proba = rfc.predict_proba(vector_input)[0][1]

                if prediction == 1:
                    spam_count += 1
                    spam_emails.append({"from": msg['from'], "subject": msg['subject'], "raw": body, "cleaned": clean_body, "proba": proba})
                else:
                    not_spam_count += 1
                    not_spam_emails.append({"from": msg['from'], "subject": msg['subject'], "raw": body, "cleaned": clean_body, "proba": proba})

                if idx >= total_emails - 5:
                    email_data.append({"from": msg['from'], "subject": msg['subject'], "raw": body, "cleaned": clean_body, "prediction": prediction, "proba": proba})

            st.session_state['spam_emails'] = spam_emails
            st.session_state['not_spam_emails'] = not_spam_emails
            st.session_state['last5_emails'] = email_data
            st.session_state['summary'] = {'total': total_emails, 'spam': spam_count, 'not_spam': not_spam_count}

            mail.logout()

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Summary ---
if st.session_state['summary']:
    s = st.session_state['summary']
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align:center; font-size:18px; padding:10px;'>
            📊 <b>Summary Report</b><br><br>
            📬 Total Emails in Inbox: <b>{s['total']}</b><br>
            🟥 Spam Emails: <b style='color:#ff4b4b;'>{s['spam']}</b><br>
            🟩 Not Spam Emails: <b style='color:green;'>{s['not_spam']}</b>
        </div>
    """, unsafe_allow_html=True)

# --- Last 5 Emails Preview ---
if st.session_state['last5_emails']:
    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>📩 Preview of Last 5 Emails</h3>", unsafe_allow_html=True)
    for e in st.session_state['last5_emails']:
        st.markdown("---")
        st.write(f"**From:** {e['from']}")
        st.write(f"**Subject:** {e['subject']}")

        # Mobile-friendly: expanders with previews
        if is_mobile:
            with st.expander("📜 Raw Email (preview)"):
                st.write(e['raw'][:300])
            with st.expander("🧹 Cleaned Email (preview)"):
                st.write(e['cleaned'][:300])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("📜 Raw Email", e['raw'], height=200)
            with col2:
                st.text_area("🧹 Cleaned Email", e['cleaned'], height=200)

        if e['prediction'] == 1:
            st.markdown('<div class="prediction-bad">⚠️ Classified as SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-good">✅ Classified as SAFE (Not Spam)</div>', unsafe_allow_html=True)
        st.write(f"🔍 Spam Confidence: **{e['proba']:.2f}**")

# --- Optional Checkboxes for all emails ---
if st.session_state['spam_emails'] or st.session_state['not_spam_emails']:
    st.markdown("---")
    st.subheader("📌 Optional: View All Emails")
    show_spam = st.checkbox("Show All Spam Emails")
    show_not_spam = st.checkbox("Show All Not Spam Emails")

    if show_spam and st.session_state['spam_emails']:
        st.markdown("### 🟥 All Spam Emails")
        for e in st.session_state['spam_emails']:
            st.markdown("---")
            st.write(f"**From:** {e['from']}")
            st.write(f"**Subject:** {e['subject']}")
            st.text_area("Email Content", e['raw'][:500], height=150)

    if show_not_spam and st.session_state['not_spam_emails']:
        st.markdown("### 🟩 All Not Spam Emails")
        for e in st.session_state['not_spam_emails']:
            st.markdown("---")
            st.write(f"**From:** {e['from']}")
            st.write(f"**Subject:** {e['subject']}")
            st.text_area("Email Content", e['raw'][:500], height=150)

# --- Footer ---
# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align: center; padding-top: 10px; font-size: 16px; color: #555;'>
        🚀 Built with ❤ by <b style='color:#ff4b4b;'>S.you.jalll</b><br>
        📩 For any suggestions, contact us at 
        <a href='mailto:sujalwarghe@gmail.com' style='color:#ff4b4b; text-decoration:none;'>sujalwarghe@gmail.com</a>
    </div>
""", unsafe_allow_html=True)
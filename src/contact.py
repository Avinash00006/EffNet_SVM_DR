import requests
import streamlit as st

def send_contact_message(name, email, message):
    """
    Sends a message via Web3Forms API using the access key in Streamlit secrets.
    Ensures that the recipient's personal email is kept entirely private and off the client code.
    """
    # Look for the secret in Streamlit secrets
    access_key = st.secrets.get("web3forms_access_key")
    if not access_key:
        return False, "Message system not configured: 'web3forms_access_key' is missing in Streamlit secrets."
        
    # Clean whitespace and quotes from the key to prevent configuration issues
    access_key = access_key.strip().replace('"', '').replace("'", "")
        
    payload = {
        "access_key": access_key,
        "name": name,
        "email": email,
        "message": message,
        "subject": "Retinal Diagnostics AI Workstation - Developer Message"
    }
    
    # Emulate standard browser headers to prevent WAF bot-detection blocks (403 errors)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://effnet-svm-dr.streamlit.app/"
    }
    
    try:
        response = requests.post(
            "https://api.web3forms.com/submit", 
            data=payload, 
            headers=headers, 
            timeout=12
        )
        if response.status_code == 200:
            res_json = response.json()
            if res_json.get("success", False):
                return True, "Message sent successfully! The developer will contact you shortly."
            else:
                return False, res_json.get("message", "API returned failure response.")
        else:
            # Log response body snippet to aid diagnostics
            return False, f"Server error {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return False, f"Failed to connect to email gateway: {str(e)}"

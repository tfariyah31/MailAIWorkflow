from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import os
import json

def authenticate():
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
    
    creds = None
    if os.path.exists('../credentials/token.json'):
        creds = Credentials.from_authorized_user_file('../credentials/token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '../credentials/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('../credentials/token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds
if __name__ == "__main__":
    creds = authenticate()
    if creds and creds.valid:
        print("Authentication successful.")
    else:
        print("Authentication failed.")
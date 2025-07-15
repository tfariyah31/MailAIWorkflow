from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import base64
import os
import logging

logger = logging.getLogger(__name__)

class GmailClient:
    def __init__(self, creds):
        """Initialize with credentials and ensure they're fresh"""
        self.creds = creds
        self._refresh_credentials_if_needed()
        self.service = build('gmail', 'v1', credentials=self.creds)
    
    def _refresh_credentials_if_needed(self):
        """Refresh credentials if expired"""
        if self.creds.expired and self.creds.refresh_token:
            try:
                self.creds.refresh(Request())
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                raise

    def fetch_unread_emails(self):
        """Fetch all unread emails with error handling"""
        try:
            results = self.service.users().messages().list(
                userId='me',
                q='is:unread',
                maxResults=50  # Limit results to prevent timeout
            ).execute()
            return results.get('messages', [])
        except Exception as e:
            logger.error(f"Error fetching unread emails: {e}")
            return []

    def get_email_details(self, msg_id):
        """Get complete email details including full body"""
        try:
            msg = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'  # Changed from 'metadata' to get full content
            ).execute()
            
            headers = {h['name']: h['value'] for h in msg['payload']['headers']}
            
            return {
                'id': msg['id'],
                'from': headers.get('From', ''),
                'subject': headers.get('Subject', ''),
                'snippet': msg.get('snippet', ''),
                'body': self._extract_email_body(msg)
            }
        except Exception as e:
            logger.error(f"Error getting email details: {e}")
            return None

    def _extract_email_body(self, msg):
        """Extract plain text body from email"""
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    return base64.urlsafe_b64decode(
                        part['body']['data'].encode('ASCII')
                    ).decode('utf-8')
        return msg.get('snippet', '')

    def create_draft(self, to_email: str, subject: str, body: str, resume_path: str = None):
        """Create a draft email with proper PDF attachment handling"""
        try:
            msg = MIMEMultipart()
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            if resume_path and os.path.exists(resume_path):
                with open(resume_path, 'rb') as f:
                    part = MIMEApplication(
                        f.read(),
                        Name=os.path.basename(resume_path)
                    )
                part['Content-Disposition'] = f'attachment; filename="{os.path.basename(resume_path)}"'
                msg.attach(part)

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            return self.service.users().drafts().create(
                userId='me',
                body={'message': {'raw': raw}}  # Fixed key from 'msg' to 'message'
            ).execute()
        except Exception as e:
            logger.error(f"Error creating draft: {e}")
            raise

    def mark_as_read(self, msg_id):
        """Marks an email as read with error handling"""
        try:
            return self.service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
        except Exception as e:
            logger.error(f"Error marking email as read: {e}")
            raise
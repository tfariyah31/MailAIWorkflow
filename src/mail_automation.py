from auth import authenticate
from classifier import Classifier
from gmail_client import GmailClient
import logging
import os
from skills_matcher import extract_resume_text
from semantic_matcher import compute_resume_job_similarity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mail_automation.log'
)
logger = logging.getLogger(__name__)

RESUME_PATH = "../assets/resume.pdf"

def clean_recruiter_name(email_field: str) -> str:
    # Extract name part before email
    name_part = email_field.split('<')[0].strip()
    
    # Remove special characters and duplicates
    name = ''.join(char for char in name_part if char.isalpha() or char.isspace())
    return ' '.join(name.split()).title()  # Remove extra spaces

# Usage:
recruiter_name = clean_recruiter_name("Tasnim Fariyah <Fariyah@company.com>")
# Returns: "Tasnim Fariyah"

def process_emails():
    creds = authenticate()
    gmail = GmailClient(creds)
    classifier = Classifier(fine_tune_path="./fine_tuned_model")

    unread_emails = gmail.fetch_unread_emails()
    logger.info(f"Found {len(unread_emails)} unread emails")

    for email in unread_emails:
        try:
            details = gmail.get_email_details(email['id'])
            logger.debug(f"Email details keys: {details.keys()}")

            logger.debug(f"Processing email: {details['subject']}")

            classification = classifier.classify_subject(details['subject'])
            logger.info(f"Classification result: {classification} for '{details['subject']}'")

            if classification == "recruiter":
                resume_text = extract_resume_text(RESUME_PATH)
                job_text = details["body"]
                similarity_score = compute_resume_job_similarity(resume_text, job_text)
                logger.info(f"Resume-Job semantic similarity score: {similarity_score}")

                #recruiter_name = details['from'].split('@')[0].split('.')[0].title()
                if similarity_score >= 0.5:
                    logger.info(f"High semantic match – creating draft for RECRUITER email: {details['subject']}")
                    recruiter_name = clean_recruiter_name(details['from'])
                    formatted_body = details['body'].replace('\n', '\n')
                    date = details.get('date', 'an earlier date')
                    reply_body = f"""Hi {recruiter_name},

Thank you for reaching out regarding the {details['subject']} position. 
I'm very interested and have attached my resume for your review.

Best regards,
TF

On {date}, {details['from']} wrote:
> {formatted_body}"""
            
                    gmail.create_draft(
                        to_email=details['from'],
                        subject=f"Re: {details['subject']}",
                        body=reply_body,
                        resume_path=RESUME_PATH if os.path.exists(RESUME_PATH) else None
                    )
                    logger.info(f"Created draft for RECRUITER email: {details['subject']}")
                    gmail.mark_as_read(email['id'])


                else:
                    logger.info("Low semantic match – skipping draft.")
            else:    
                logger.info(f"Skipping NON-RECRUITER email: {details['subject']}")
                
                #print(f"Draft created for: {details['subject']}")
            
        except Exception as e:
            logger.error(f"Failed processing email {email['id']}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    process_emails()
from imap_tools import MailBox, AND
import joblib
import pandas as pd
import logging
import time
from datetime import datetime
import os
import json
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spam_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpamDetector:
    def __init__(self, model_path: str = "spam_model.joblib"):
        """Initialize the spam detector with a trained model"""
        try:
            self.model = joblib.load(model_path)
            self.metadata = joblib.load("model_metadata.joblib")
            logger.info(f"✅ Loaded model: {self.metadata.get('model_type', 'Unknown')}")
            logger.info(f"📊 Model accuracy: {self.metadata.get('accuracy', 'Unknown'):.4f}")
        except FileNotFoundError:
            logger.error("❌ Model file not found. Please train the model first.")
            raise
    
    def preprocess_email_text(self, subject: str, body: str) -> str:
        """Preprocess email text for prediction"""
        # Combine subject and body
        full_text = f"{subject or ''} {body or ''}"
        
        # Basic preprocessing (should match training preprocessing)
        full_text = full_text.strip()
        
        return full_text
    
    def predict_spam(self, subject: str, body: str) -> tuple:
        """Predict if email is spam and return confidence"""
        text = self.preprocess_email_text(subject, body)
        
        # Get prediction and probability
        prediction = self.model.predict([text])[0]
        
        # Get confidence if available
        try:
            proba = self.model.predict_proba([text])[0]
            confidence = max(proba)
        except:
            confidence = 0.0
        
        return prediction, confidence

class EmailProcessor:
    def __init__(self, email: str, password: str, confidence_threshold: float = 0.7):
        self.email = email
        self.password = password
        self.confidence_threshold = confidence_threshold
        self.detector = SpamDetector()
        
        # Statistics
        self.stats = {
            'total_checked': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'emails_moved': 0,
            'errors': 0
        }
    
    def connect_to_mailbox(self) -> MailBox:
        """Connect to the email server"""
        try:
            mailbox = MailBox('imap.gmail.com').login(self.email, self.password)
            logger.info("🔐 Successfully connected to Gmail")
            return mailbox
        except Exception as e:
            logger.error(f"❌ Failed to connect: {str(e)}")
            raise
    
    def process_folder(self, mailbox: MailBox, folder: str = 'INBOX', 
                      action: str = 'scan') -> List[Dict]:
        """Process emails in a specific folder"""
        results = []
        
        try:
            # Set folder
            mailbox.folder.set(folder)
            logger.info(f"📂 Processing folder: {folder}")
            
            # Fetch messages
            messages = list(mailbox.fetch(bulk=True))
            
            if not messages:
                logger.info(f"📭 No emails found in {folder}")
                return results
            
            logger.info(f"📬 Found {len(messages)} emails in {folder}")
            
            for i, msg in enumerate(messages, 1):
                try:
                    result = self.process_single_email(mailbox, msg, action)
                    results.append(result)
                    self.stats['total_checked'] += 1
                    
                    # Progress update
                    if i % 10 == 0:
                        logger.info(f"⏳ Processed {i}/{len(messages)} emails")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing email {i}: {str(e)}")
                    self.stats['errors'] += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing folder {folder}: {str(e)}")
            self.stats['errors'] += 1
        
        return results
    
    def process_single_email(self, mailbox: MailBox, msg, action: str) -> Dict:
        """Process a single email"""
        # Extract email content
        subject = msg.subject or ""
        sender = msg.from_
        body = msg.text or ""
        date = msg.date
        
        # Predict spam
        prediction, confidence = self.detector.predict_spam(subject, body)
        is_spam = prediction == 1
        
        # Create result
        result = {
            'from': sender,
            'subject': subject,
            'date': date,
            'prediction': 'spam' if is_spam else 'ham',
            'confidence': confidence,
            'action_taken': 'none',
            'uid': msg.uid
        }
        
        # Update statistics
        if is_spam:
            self.stats['spam_detected'] += 1
        else:
            self.stats['ham_detected'] += 1
        
        # Log prediction
        logger.info(f"\n📨 From: {sender}")
        logger.info(f"📌 Subject: {subject[:50]}...")
        logger.info(f"🧠 Prediction: {'SPAM' if is_spam else 'HAM'} (confidence: {confidence:.2f})")
        
        # Take action based on prediction and confidence
        if action == 'move' and is_spam and confidence >= self.confidence_threshold:
            try:
                # Move to spam folder
                mailbox.move(msg.uid, '[Gmail]/Spam')
                result['action_taken'] = 'moved_to_spam'
                self.stats['emails_moved'] += 1
                logger.info("🗑️ Moved to spam folder")
            except Exception as e:
                logger.error(f"❌ Failed to move email: {str(e)}")
                result['action_taken'] = 'move_failed'
        
        elif action == 'delete' and is_spam and confidence >= self.confidence_threshold:
            try:
                # Delete the email
                mailbox.delete(msg.uid)
                result['action_taken'] = 'deleted'
                self.stats['emails_moved'] += 1
                logger.info("🗑️ Deleted email")
            except Exception as e:
                logger.error(f"❌ Failed to delete email: {str(e)}")
                result['action_taken'] = 'delete_failed'
        
        return result
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save processing results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"spam_detection_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'statistics': self.stats,
                'results': results
            }, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filename}")
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "="*50)
        print("📊 SPAM DETECTION STATISTICS")
        print("="*50)
        print(f"Total emails checked: {self.stats['total_checked']}")
        print(f"Spam detected: {self.stats['spam_detected']}")
        print(f"Ham detected: {self.stats['ham_detected']}")
        print(f"Emails moved/deleted: {self.stats['emails_moved']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats['total_checked'] > 0:
            spam_rate = (self.stats['spam_detected'] / self.stats['total_checked']) * 100
            print(f"Spam rate: {spam_rate:.1f}%")
        
        print("="*50)

def main():
    # Configuration
    EMAIL = "jazzykay142006@gmail.com"  # Replace with your email
    PASSWORD = "arzz pqsb wkfd mumy"  # Replace with your app password
    CONFIDENCE_THRESHOLD = 0.7      # Minimum confidence for action
    
    # Choose action: 'scan', 'move', or 'delete'    
    ACTION = 'scan'  # Change to 'move' or 'delete' when ready
    
    # Choose folder: 'INBOX', '[Gmail]/Spam', etc.
    FOLDER = 'INBOX'
    
    print("🚀 Starting Enhanced Spam Detection System")
    print(f"📧 Email: {EMAIL}")
    print(f"📂 Folder: {FOLDER}")
    print(f"🎯 Action: {ACTION}")
    print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    # Create processor
    processor = EmailProcessor(EMAIL, PASSWORD, CONFIDENCE_THRESHOLD)
    
    # Process emails
    with processor.connect_to_mailbox() as mailbox:
        results = processor.process_folder(mailbox, FOLDER, ACTION)
    
    # Save results and print statistics
    processor.save_results(results)
    processor.print_statistics()
    
    print("\n✅ Processing complete!")

if __name__ == "__main__":
    main()
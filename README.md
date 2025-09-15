# 📧 Spam Filter with Machine Learning

An end-to-end spam email detection system that integrates a trained ML model with Gmail via IMAP. The project automatically scans, classifies, and (optionally) moves or deletes spam emails, while maintaining logs and statistics.

## 🚀 Features

* **Machine Learning Model**: Trained using labeled spam/ham datasets (`spam.csv`).
* **Prediction with Confidence**: Classifies emails as spam/ham with probability scores.
* **Email Integration**: Connects to Gmail using `imap-tools` and processes emails from any folder (e.g., `INBOX`, Spam).
* **Automated Actions**:
  * *Scan* – log results only
  * *Move* – relocate spam to Gmail’s Spam folder
  * *Delete* – permanently remove spam emails
* **Logging & Statistics**: Saves detailed logs (`spam_detection.log`) and JSON reports with results, error counts, and spam rates.
* **Customizable Threshold**: Control how confident the model must be before taking action.

## 🛠️ Tech Stack

* **Python 3**
* **Libraries**:
  * `scikit-learn` / `joblib` – model training & loading
  * `imap-tools` – email fetching & folder management
  * `pandas` – dataset handling
  * `logging` / `json` – logs & results export

## 📂 Project Structure

```
.
├── spam.csv                 # Training dataset (spam/ham labeled messages)
├── train_model.py           # Script to train and save ML model
├── spam_model.joblib        # Trained ML model
├── model_metadata.joblib    # Metadata (accuracy, model type)
├── read_emails.py           # Main script: connects to Gmail & classifies emails
├── spam_detection.log       # Log file of processing
├── spam_detection_results_*.json # Saved JSON results
```

## ⚙️ How It Works

1. **Train** the model with `train_model.py` (generates `spam_model.joblib` + metadata).
2. **Run** `read_emails.py` to:
   * Connect to Gmail
   * Fetch emails
   * Predict spam/ham
   * Log actions and stats
3. **Choose Action** (`scan`, `move`, `delete`) with a confidence threshold.

## 📊 Example Output

```
📨 From: example@scammer.com
📌 Subject: Congratulations! You've won a prize...
🧠 Prediction: SPAM (confidence: 0.94)
🗑️ Moved to spam folder
```

## 🔒 Security Note

Use a Gmail **App Password** (not your main password) for authentication. Never hardcode credentials in public repos—store them securely in environment variables.

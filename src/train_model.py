import pandas as pd
from classifier import Classifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load your dataset
        logger.info("Loading dataset...")
        df = pd.read_csv("dataset.csv")
        texts = df["subject"].tolist()
        labels = df["label"].tolist()  # 1=recruiter, 0=non-recruiter

        # Initialize classifier
        logger.info("Initializing classifier...")
        classifier = Classifier()

        # Fine-tune the model
        logger.info("Starting training...")
        classifier.train(
            train_texts=texts,
            train_labels=labels,
            output_dir="./fine_tuned_model"
        )

        logger.info("Fine-tuning complete! Model saved to ./fine_tuned_model")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
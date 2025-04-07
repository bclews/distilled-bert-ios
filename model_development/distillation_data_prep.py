import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import nltk
from nltk.corpus import wordnet
import random
import logging
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DistillationDatasetPreparator:
    def __init__(
        self,
        csv_path,
        teacher_model_name="bert-base-uncased",
        max_length=128,
        batch_size=8,
        random_seed=42,
    ):
        """
        Initialize the dataset preparator for knowledge distillation

        Args:
            csv_path: Path to the CSV file containing the training data
            teacher_model_name: Pre-trained BERT model to use as teacher
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for DataLoaders
            random_seed: Random seed for reproducibility
        """
        self.csv_path = csv_path
        self.teacher_model_name = teacher_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        # Download NLTK resources if needed
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

    def load_and_preprocess_data(self):
        """Load and preprocess the dataset from CSV"""
        logging.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        # Check for column names that might be similar to label_id
        if "label_id" in df.columns:
            id_column = "label_id"
        elif "label id" in df.columns:
            id_column = "label id"
        else:
            id_column = None

        # Filter out rows without labels or with NaN labels
        if id_column and id_column in df.columns:
            filtered_df = df[df[id_column] != -1].copy()
            filtered_df = filtered_df[filtered_df["label"].notna()].copy()
        else:
            # If no label ID column is found, filter based on label column
            filtered_df = df[df["label"].notna()].copy()

        logging.info(
            f"Found {len(filtered_df)} labeled examples out of {len(df)} total rows"
        )

        # Create label mapping if needed
        unique_labels = filtered_df["label"].unique()
        # Remove NaN from labels if present
        unique_labels = [label for label in unique_labels if not pd.isna(label)]
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        logging.info(f"Label mapping: {self.label_map}")

        # Extract text and labels from filtered dataframe
        valid_indices = []
        texts = []
        labels = []

        for idx, row in filtered_df.iterrows():
            if pd.isna(row["label"]):
                continue

            # Get text content
            if "tagless statement" in filtered_df.columns and not pd.isna(
                row["tagless statement"]
            ):
                text = row["tagless statement"]
            elif "full statement" in filtered_df.columns and not pd.isna(
                row["full statement"]
            ):
                text = row["full statement"].replace(r"\[.*?\]", "", regex=True)
            else:
                continue  # Skip rows without valid text

            # Skip rows with NaN text
            if pd.isna(text) or not isinstance(text, str):
                continue

            # Only add valid examples
            if row["label"] in self.label_map:
                texts.append(text)
                labels.append(self.label_map[row["label"]])
                valid_indices.append(idx)

        logging.info(f"Extracted {len(texts)} valid text examples with labels")

        return texts, labels

    def augment_data(self, texts, labels, augmentation_factor=2):
        """
        Augment the dataset using synonym replacement and back-translation simulation

        Args:
            texts: List of text examples
            labels: List of corresponding labels
            augmentation_factor: Number of augmented examples to create per original example

        Returns:
            Augmented texts and labels
        """
        logging.info(f"Augmenting data by factor of {augmentation_factor}")
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()

        for idx, (text, label) in enumerate(zip(texts, labels)):
            # Skip any non-string or empty text
            if not isinstance(text, str) or not text.strip():
                continue

            for _ in range(augmentation_factor):
                # Choose augmentation technique
                technique = random.choice(["synonym_replacement", "random_deletion"])

                if technique == "synonym_replacement":
                    augmented_text = self._synonym_replacement(text)
                else:
                    augmented_text = self._random_deletion(text)

                augmented_texts.append(augmented_text)
                augmented_labels.append(label)

            if idx % 100 == 0 and idx > 0:
                logging.info(f"Augmented {idx}/{len(texts)} examples")

        logging.info(
            f"Data augmentation complete. New dataset size: {len(augmented_texts)}"
        )
        return augmented_texts, augmented_labels

    def _synonym_replacement(self, text, replace_prob=0.2):
        """Replace random words with synonyms"""
        words = text.split()
        num_to_replace = max(1, int(len(words) * replace_prob))
        indexes = random.sample(range(len(words)), min(num_to_replace, len(words)))

        for idx in indexes:
            word = words[idx]
            synonyms = []

            # Skip very short words and special characters
            if len(word) <= 3 or not word.isalpha():
                continue

            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().replace("_", " "))

            if synonyms:
                words[idx] = random.choice(synonyms)

        return " ".join(words)

    def _random_deletion(self, text, delete_prob=0.1):
        """Randomly delete words from text"""
        if not isinstance(text, str) or not text.strip():
            return "empty text"  # Return a placeholder for empty or invalid text

        words = text.split()
        if not words:
            return "empty text"

        new_words = []

        for word in words:
            if random.random() > delete_prob:
                new_words.append(word)

        # Ensure we don't delete all words
        if not new_words:
            return random.choice(words)

        return " ".join(new_words)

    def tokenize_dataset(self, texts, labels):
        """
        Tokenize the texts using BERT tokenizer

        Args:
            texts: List of text examples
            labels: List of corresponding labels

        Returns:
            input_ids, attention_masks, and labels as tensors
        """
        logging.info("Tokenizing dataset")
        tokenizer = BertTokenizer.from_pretrained(self.teacher_model_name)

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_masks = encodings["attention_mask"]
        labels_tensor = torch.tensor(labels)

        return input_ids, attention_masks, labels_tensor

    def create_dataloaders(self, input_ids, attention_masks, labels, val_ratio=0.1):
        """
        Create training and validation DataLoaders

        Args:
            input_ids: Tokenized input IDs
            attention_masks: Attention masks
            labels: Labels
            val_ratio: Validation set ratio

        Returns:
            Training and validation DataLoaders
        """
        logging.info(f"Creating DataLoaders with validation ratio {val_ratio}")

        # Split into train and validation sets
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = (
            train_test_split(
                input_ids,
                attention_masks,
                labels,
                test_size=val_ratio,
                stratify=labels,
                random_state=self.random_seed,
            )
        )

        # Create TensorDatasets
        train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
        val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        logging.info(f"Created train_dataloader with {len(train_dataset)} examples")
        logging.info(f"Created val_dataloader with {len(val_dataset)} examples")

        return train_dataloader, val_dataloader

    def generate_soft_labels(self, input_ids, attention_masks, teacher_model_path=None):
        """
        Generate soft labels (probability distributions) from the teacher model

        Args:
            input_ids: Tokenized input IDs
            attention_masks: Attention masks
            teacher_model_path: Path to saved teacher model (if None, load from Hugging Face)

        Returns:
            Soft labels for the inputs
        """
        logging.info("Generating soft labels from teacher model")

        # Load teacher model
        if teacher_model_path and os.path.exists(teacher_model_path):
            teacher_model = BertForSequenceClassification.from_pretrained(
                teacher_model_path
            )
            logging.info(f"Loaded teacher model from {teacher_model_path}")
        else:
            # If no saved model exists, we'll need to use a pre-trained one
            num_labels = len(self.label_map)
            teacher_model = BertForSequenceClassification.from_pretrained(
                self.teacher_model_name, num_labels=num_labels
            )
            logging.info(
                f"Loaded pre-trained model {self.teacher_model_name} as teacher"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher_model.to(device)
        teacher_model.eval()

        # Create DataLoader for batch processing
        dataset = TensorDataset(input_ids, attention_masks)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        soft_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch_input_ids, batch_attention_masks = tuple(
                    t.to(device) for t in batch
                )

                outputs = teacher_model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_masks
                )

                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                soft_labels.append(probs.cpu())

        # Concatenate all batches
        all_soft_labels = torch.cat(soft_labels, dim=0)
        logging.info(f"Generated soft labels with shape {all_soft_labels.shape}")

        return all_soft_labels

    def save_prepared_data(
        self,
        input_ids,
        attention_masks,
        labels,
        soft_labels,
        output_dir="data/distillation_data",
    ):
        """
        Save the prepared data for distillation

        Args:
            input_ids: Tokenized input IDs
            attention_masks: Attention masks
            labels: Hard labels
            soft_labels: Soft labels from teacher
            output_dir: Directory to save prepared data
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info(f"Saving prepared data to {output_dir}")

        torch.save(input_ids, os.path.join(output_dir, "input_ids.pt"))
        torch.save(attention_masks, os.path.join(output_dir, "attention_masks.pt"))
        torch.save(labels, os.path.join(output_dir, "labels.pt"))
        torch.save(soft_labels, os.path.join(output_dir, "soft_labels.pt"))

        # Save label mapping
        pd.DataFrame(
            {"label": list(self.label_map.keys()), "id": list(self.label_map.values())}
        ).to_csv(os.path.join(output_dir, "label_map.csv"), index=False)

        logging.info("Data preparation complete!")

    def prepare_for_distillation(self, augmentation_factor=3, teacher_model_path=None):
        """
        Main method to prepare data for knowledge distillation

        Args:
            augmentation_factor: Number of augmented examples to create per original example
            teacher_model_path: Path to saved teacher model

        Returns:
            train_dataloader, val_dataloader, and soft_labels
        """
        # Load and preprocess data
        texts, labels = self.load_and_preprocess_data()

        if not texts or not labels:
            raise ValueError("No valid text examples with labels found in the dataset")

        # Augment data
        augmented_texts, augmented_labels = self.augment_data(
            texts, labels, augmentation_factor
        )

        # Tokenize dataset
        input_ids, attention_masks, labels_tensor = self.tokenize_dataset(
            augmented_texts, augmented_labels
        )

        # Generate soft labels from teacher
        soft_labels = self.generate_soft_labels(
            input_ids, attention_masks, teacher_model_path
        )

        # Create DataLoaders
        train_dataloader, val_dataloader = self.create_dataloaders(
            input_ids, attention_masks, labels_tensor
        )

        # Save prepared data
        self.save_prepared_data(input_ids, attention_masks, labels_tensor, soft_labels)

        return train_dataloader, val_dataloader, soft_labels


if __name__ == "__main__":
    # Example usage
    preparator = DistillationDatasetPreparator(
        csv_path="data/train_test_splits.csv",
        teacher_model_name="/Users/cle126/Developer/github/bclews/distilled-bert-ios/model_development/models/teacher",
        max_length=128,
        batch_size=8,
    )

    try:
        train_dataloader, val_dataloader, soft_labels = (
            preparator.prepare_for_distillation(
                augmentation_factor=3,
                teacher_model_path="/Users/cle126/Developer/github/bclews/distilled-bert-ios/model_development/models/teacher",
            )
        )
        logging.info("Dataset preparation complete!")
    except Exception as e:
        logging.error(f"Error during dataset preparation: {str(e)}")
        # Print detailed information about the error
        import traceback

        traceback.print_exc()

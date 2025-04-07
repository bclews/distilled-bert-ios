import argparse
import os

import coremltools as ct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_cosine_schedule_with_warmup,
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
)


class PrecomputedDistillationDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels, soft_labels=None):
        """
        Dataset for knowledge distillation using precomputed tensors.
        
        Args:
            input_ids: Tokenized input sequences
            attention_masks: Attention masks for the inputs
            labels: Ground truth labels
            soft_labels: Teacher model's predictions (logits)
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }
        
        if self.soft_labels is not None:
            item["teacher_logits"] = self.soft_labels[idx]
            
        return item


def load_precomputed_data(data_dir, val_split=0.2, random_seed=42):
    """Load precomputed tensors and split into train/validation sets"""
    print(f"Loading precomputed data from: {data_dir}")
    
    # Load all tensor files
    input_ids = torch.load(os.path.join(data_dir, "input_ids.pt"))
    attention_masks = torch.load(os.path.join(data_dir, "attention_masks.pt"))
    labels = torch.load(os.path.join(data_dir, "labels.pt"))
    soft_labels = torch.load(os.path.join(data_dir, "soft_labels.pt"))
    
    # Load label mapping
    label_map_df = pd.read_csv(os.path.join(data_dir, "label_map.csv"))
    num_labels = len(label_map_df)
    
    print(f"Loaded {len(input_ids)} examples with {num_labels} classes")
    
    # Create train/validation split
    indices = torch.randperm(len(input_ids))
    val_size = int(len(indices) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create train datasets
    train_input_ids = input_ids[train_indices]
    train_attention_masks = attention_masks[train_indices]
    train_labels = labels[train_indices]
    train_soft_labels = soft_labels[train_indices]
    
    # Create validation datasets
    val_input_ids = input_ids[val_indices]
    val_attention_masks = attention_masks[val_indices]
    val_labels = labels[val_indices]
    val_soft_labels = soft_labels[val_indices]
    
    print(f"Split into {len(train_labels)} training and {len(val_labels)} validation examples")
    
    return (
        (train_input_ids, train_attention_masks, train_labels, train_soft_labels),
        (val_input_ids, val_attention_masks, val_labels, val_soft_labels),
        num_labels,
        label_map_df
    )


def calculate_class_weights(labels):
    """Calculate class weights inversely proportional to class frequencies"""
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
    class_counts = np.bincount(labels.astype(int))
    total_samples = len(labels)

    # Compute inverse of frequency
    weights = total_samples / (len(class_counts) * class_counts)

    # Normalize weights to prevent extremely high values
    weights = weights / weights.sum() * len(class_counts)

    print(f"Class distribution: {class_counts}")
    print(f"Calculated weights: {weights}")

    # Convert to tensor
    return torch.FloatTensor(weights)


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=1.0,
    alpha=0.5,
    class_weights=None,
    use_focal_loss=True,
    focal_gamma=2.0,
):
    """
    Compute the knowledge distillation loss with optional class weights or focal loss.

    Args:
        student_logits: logits from the student model
        teacher_logits: logits from the teacher model
        labels: ground truth labels
        temperature: softmax temperature for distillation
        alpha: weight for balancing distillation loss and CE/focal loss
        class_weights: optional weights for handling class imbalance
        use_focal_loss: whether to use focal loss instead of cross-entropy
        focal_gamma: focusing parameter for focal loss

    Returns:
        combined loss (distillation loss + hard loss)
    """
    # Compute soft targets
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)

    # Compute distillation loss (KL divergence)
    kl_div_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / temperature, dim=1), soft_targets
    ) * (temperature**2)

    # Compute hard loss (either focal loss or weighted cross-entropy)
    if use_focal_loss:
        hard_loss = focal_loss(
            student_logits, labels, gamma=focal_gamma, alpha=class_weights
        )
    else:
        # Use standard cross-entropy loss with optional class weights
        if class_weights is not None:
            # Move weights to same device as labels
            weights = class_weights.to(labels.device)
            hard_loss = F.cross_entropy(student_logits, labels, weight=weights)
        else:
            hard_loss = F.cross_entropy(student_logits, labels)

    # Combine losses
    return alpha * kl_div_loss + (1 - alpha) * hard_loss


def create_student_model(student_type, num_labels):
    """Create the student model based on specified type"""
    if student_type == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    elif student_type == "mobilebert":
        model = MobileBertForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased", num_labels=num_labels
        )
        tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
    else:
        raise ValueError(f"Unsupported student model type: {student_type}")

    return model, tokenizer


class ModelWrapper(nn.Module):
    """Wrapper for the student model to make it compatible with torch.jit.trace"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # Return only the logits instead of a dictionary


def get_device():
    """Get the best available device (MPS for M1/M2 Macs, CUDA for NVIDIA GPUs, or CPU)"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_distillation(
    student_model,
    train_dataloader,
    val_dataloader,
    device,
    training_args,
    class_weights=None,
):
    """Train the student model using knowledge distillation with precomputed teacher logits"""
    # Set student to training mode
    student_model.train()

    # Group parameters with different learning rates
    parameters = []

    # Lower learning rate for embeddings to preserve language knowledge
    params_group1 = {
        "params": [p for n, p in student_model.named_parameters() if "embeddings" in n],
        "lr": training_args.learning_rate / 10.0,
    }

    # Regular learning rate for other layers
    params_group2 = {
        "params": [
            p for n, p in student_model.named_parameters() if "embeddings" not in n
        ],
        "lr": training_args.learning_rate,
    }

    parameters.append(params_group1)
    parameters.append(params_group2)

    # Create optimizer with parameter groups
    optimizer = AdamW(parameters)

    # Add this after optimizer creation
    total_steps = len(train_dataloader) * training_args.num_epochs
    warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup

    # Create scheduler with cosine annealing and warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(training_args.num_epochs):
        student_model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)

            # Forward pass through student model
            student_outputs = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            # Compute loss
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=training_args.temperature,
                alpha=training_args.alpha,
                class_weights=class_weights,
                use_focal_loss=training_args.use_focal_loss,
                focal_gamma=training_args.focal_gamma,
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        student_model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = student_outputs.logits

                # Use class weights for evaluation loss too if provided
                if class_weights is not None:
                    loss = F.cross_entropy(
                        student_logits, labels, weight=class_weights.to(device)
                    )
                else:
                    loss = F.cross_entropy(student_logits, labels)

                val_loss += loss.item()

                preds = torch.argmax(student_logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Print metrics
        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        print(
            f"Epoch {epoch+1}/{training_args.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        report = classification_report(all_labels, all_preds, zero_division=0)
        print(report)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(training_args.output_dir, exist_ok=True)
            student_model.save_pretrained(
                os.path.join(training_args.output_dir, "best_model")
            )
        else:
            patience_counter += 1
            if patience_counter >= training_args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return student_model


def quantize_and_convert_to_coreml(student_model, student_tokenizer, output_dir):
    """Quantize the model and convert to Core ML format"""
    # Create a wrapper for the model to simplify the output structure
    wrapped_model = ModelWrapper(student_model)
    wrapped_model = wrapped_model.cpu()

    # Create a sample input for tracing
    sample_text = "This is a sample input for tracing the model."
    inputs = student_tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    # Make sure inputs are on CPU
    input_ids = inputs["input_ids"].cpu()
    attention_mask = inputs["attention_mask"].cpu()

    # Trace the model with sample input
    wrapped_model.eval()
    try:
        os.makedirs("./models/coreml/weights", exist_ok=True)
        traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))

        print("Model successfully traced!")

        # First save the PyTorch model to create weight files
        torch.jit.save(traced_model, "./models/coreml/weights/traced_model.pt")

        # Convert to Core ML model
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=input_ids.shape),
                ct.TensorType(name="attention_mask", shape=attention_mask.shape),
            ],
            minimum_deployment_target=ct.target.iOS15,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        # Save the model
        os.makedirs(output_dir, exist_ok=True)
        mlmodel_path = os.path.join(output_dir, "distilled_model.mlpackage")
        mlmodel.save(mlmodel_path)

        # Save the tokenizer
        student_tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    except Exception as e:
        print(f"Error during model tracing or conversion: {e}")
        print("Saving PyTorch model and tokenizer only (without CoreML conversion)")
        os.makedirs(output_dir, exist_ok=True)
        student_model.save_pretrained(os.path.join(output_dir, "pytorch_model"))
        student_tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))


def focal_loss(student_logits, labels, gamma=2.0, alpha=None):
    """
    Focal Loss for handling class imbalance better than weighted cross-entropy.

    Args:
        student_logits: logits from the student model
        labels: ground truth labels
        gamma: focusing parameter (higher values increase focus on hard examples)
        alpha: optional weighting factor for each class

    Returns:
        focal loss
    """
    # Get device
    device = student_logits.device

    # Get number of classes
    num_classes = student_logits.size(1)

    # Convert labels to one-hot encoding
    one_hot = torch.zeros(labels.size(0), num_classes, device=device)
    one_hot.scatter_(1, labels.view(-1, 1), 1)

    # Compute softmax probabilities
    probs = F.softmax(student_logits, dim=1)

    # Get the probability corresponding to the target class
    pt = (one_hot * probs).sum(1)

    # Compute the focal term: (1-pt)^gamma
    focal_term = (1 - pt).pow(gamma)

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, labels, reduction="none")

    # Apply alpha weighting if provided
    if alpha is not None:
        # Convert alpha to tensor and move to the same device
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, device=device)
        else:
            # If alpha is already a tensor, make sure it's on the right device
            alpha = alpha.to(device)

        # Apply alpha weighting
        alpha_weight = alpha[labels]
        focal_term = alpha_weight * focal_term

    # Combine focal term with cross-entropy loss
    loss = focal_term * ce_loss

    # Return mean loss
    return loss.mean()


def main(args):
    # Set device
    device = get_device()
    print(f"Using device: {device}")

    # Load pre-computed data
    (train_data, val_data, num_labels, label_map_df) = load_precomputed_data(
        args.data_dir, val_split=args.val_split, random_seed=args.random_seed
    )
    
    train_input_ids, train_attention_masks, train_labels, train_soft_labels = train_data
    val_input_ids, val_attention_masks, val_labels, val_soft_labels = val_data

    # Calculate class weights from training set
    class_weights = calculate_class_weights(train_labels) if args.use_class_weights else None

    # Create student model
    student_model, student_tokenizer = create_student_model(
        args.student_model_type, num_labels
    )
    student_model.to(device)

    # Create datasets
    train_dataset = PrecomputedDistillationDataset(
        train_input_ids, train_attention_masks, train_labels, train_soft_labels
    )
    val_dataset = PrecomputedDistillationDataset(
        val_input_ids, val_attention_masks, val_labels, val_soft_labels
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Train with distillation
    student_model = train_distillation(
        student_model,
        train_dataloader,
        val_dataloader,
        device,
        args,
        class_weights=class_weights,
    )

    # Load best model
    student_model = type(student_model).from_pretrained(
        os.path.join(args.output_dir, "best_model")
    )
    student_model.to(device)

    # Evaluate final model
    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = student_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    print("Final model performance:")
    print(classification_report(all_labels, all_preds, zero_division=0))

    # Convert to Core ML
    quantize_and_convert_to_coreml(
        student_model, student_tokenizer, args.coreml_output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for BERT models with precomputed data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing precomputed tensor files (input_ids.pt, attention_masks.pt, labels.pt, soft_labels.pt, label_map.csv)",
    )
    parser.add_argument(
        "--student_model_type",
        type=str,
        default="distilbert",
        choices=["distilbert", "mobilebert"],
        help="Type of student model to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./distilled_model",
        help="Directory to save the distilled model",
    )
    parser.add_argument(
        "--coreml_output_dir",
        type=str,
        default="./coreml_model",
        help="Directory to save the Core ML model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Temperature for distillation (lower = harder probabilities)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Weight for balancing distillation and CE losses",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Patience for early stopping"
    )
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights to handle class imbalance",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=True,
        help="Use focal loss instead of weighted cross-entropy",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=3.0,
        help="Focusing parameter for focal loss",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
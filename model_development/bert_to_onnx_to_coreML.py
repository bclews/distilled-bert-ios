#!/usr/bin/env python3
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import onnx
import coremltools as ct

def main():
    # Define model path and output file names
    model_path = "/Users/cle126/Developer/github/bclews/distilled-bert-ios/model_development/models/teacher"
    onnx_model_path = "/Users/cle126/Developer/github/bclews/distilled-bert-ios/model_development/models/onnx/bert.onnx"
    coreml_model_path = "/Users/cle126/Developer/github/bclews/distilled-bert-ios/model_development/models/coreml_2/BERT.mlmodel"

    # Load the pre-trained model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Create dummy input using the tokenizer (adjust the input text as needed)
    dummy_text = "Hole sizes for BACB30AU blind bolts must be in accordance with the Engineering drawing."
    dummy_input = tokenizer(dummy_text, return_tensors="pt")

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_model_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"ONNX export complete. Model saved at: {onnx_model_path}")

    # Verify the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

    # Convert the ONNX model to Core ML using coremltools
    coreml_model = ct.convert(onnx_model, minimum_deployment_target=ct.target.iOS13)

    # Save the converted Core ML model
    coreml_model.save(coreml_model_path)
    print(f"Core ML model conversion complete. Model saved at: {coreml_model_path}")

if __name__ == "__main__":
    main()
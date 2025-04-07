//
//  TextClassifier.swift
//  auto_tagger
//
//  Created by Ben Clews on 26/2/2025.
//
import UIKit
import CoreML

class TextClassifier {
    private let model: MLModel
    private let tokenizer: BERTTokenizer
    private var labels: [String]
    
    struct ClassificationResult {
        let labelIndex: Int
        let label: String
        let confidence: Double
        let allProbabilities: [(label: String, value: Double)]
        let rawOutput: [String: Double]
    }
    
    init() throws {
        // Load the Core ML model
        let mlModel = try distilled_model()
        model = mlModel.model
        
        // Initialize the tokenizer
        tokenizer = try BERTTokenizer()
        
        // Try to extract class labels from model metadata
        if let metadata = model.modelDescription.metadata as? [String: Any],
           let classLabels = metadata["MLModelDescriptionKey.classLabels"] as? [String],
           !classLabels.isEmpty {
            self.labels = classLabels
            print("Found model labels: \(classLabels)")
        } else {
            // Fallback to default labels
            self.labels = ["condition", "constraint", "notice", "process"]
            print("Using default labels: \(labels)")
        }
    }
    
    func classify(text: String) throws -> ClassificationResult {
        // Tokenize the input text
        let tokens = try tokenizer.tokenize(text: text, maxLength: 128)
        
        // Create input features for the model
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": tokens.ids,
            "attention_mask": tokens.mask
        ])
        
        // Get predictions from the model
        let prediction = try model.prediction(from: inputFeatures)
        
        // Debug: Print out all available feature names in the prediction
        print("Available output features: \(prediction.featureNames.joined(separator: ", "))")
        
        // Try to get the logits or other output feature
        // Common output names in transformer models: "logits", "output", "probabilities", "scores"
        let possibleOutputNames = ["logits", "output", "probabilities", "scores", "identity", "classLabel"]
        
        var outputFeature: MLMultiArray? = nil
        var outputFeatureName: String? = nil
        
        for name in possibleOutputNames {
            if let feature = prediction.featureValue(for: name)?.multiArrayValue {
                outputFeature = feature
                outputFeatureName = name
                print("Found output feature: \(name) with shape: \(feature.shape)")
                break
            }
        }
        
        // If we still don't have an output feature, try to use any available numerical feature
        if outputFeature == nil {
            for name in prediction.featureNames {
                if let feature = prediction.featureValue(for: name)?.multiArrayValue {
                    outputFeature = feature
                    outputFeatureName = name
                    print("Using alternative output feature: \(name) with shape: \(feature.shape)")
                    break
                }
            }
        }
        
        guard let logits = outputFeature else {
            print("Failed to find any usable output feature")
            throw ClassifierError.predictionFailed
        }
        
        print("Using output feature: \(outputFeatureName ?? "unknown")")
        
        // Convert logits to probabilities using softmax
        let probabilities = softmax(logits: logits)
        
        // Find the index with the highest probability
        var maxIndex = 0
        var maxValue = probabilities[0]
        
        for i in 1..<probabilities.count {
            if probabilities[i] > maxValue {
                maxValue = probabilities[i]
                maxIndex = i
            }
        }
        
        // Make sure we don't exceed the labels array bounds
        let safeIndex = min(maxIndex, labels.count - 1)
        
        // Create a dictionary of all probabilities
        var rawOutput = [String: Double]()
        var labeledProbabilities = [(label: String, value: Double)]()
        
        for i in 0..<min(probabilities.count, labels.count) {
            let label = labels[i]
            let probability = probabilities[i]
            rawOutput[label] = probability
            labeledProbabilities.append((label: label, value: probability))
        }
        
        // Add any remaining probabilities with generic labels
        if probabilities.count > labels.count {
            for i in labels.count..<probabilities.count {
                let label = "Class_\(i)"
                let probability = probabilities[i]
                rawOutput[label] = probability
                labeledProbabilities.append((label: label, value: probability))
            }
        }
        
        // Sort probabilities in descending order
        labeledProbabilities.sort { $0.value > $1.value }
        
        return ClassificationResult(
            labelIndex: safeIndex,
            label: labels[safeIndex],
            confidence: probabilities[safeIndex],
            allProbabilities: labeledProbabilities,
            rawOutput: rawOutput
        )
    }
    
    private func softmax(logits: MLMultiArray) -> [Double] {
        var values = [Double]()
        let count = logits.count
        
        // Extract all values
        for i in 0..<count {
            values.append(logits[i].doubleValue)
        }
        
        // Find max for numerical stability
        let maxValue = values.max() ?? 0.0
        
        // Apply exp to each value after shifting by max
        var expValues = [Double]()
        var sumExp = 0.0
        
        for value in values {
            let expValue = exp(value - maxValue)
            expValues.append(expValue)
            sumExp += expValue
        }
        
        // Normalize by sum
        return expValues.map { $0 / sumExp }
    }
    
    enum ClassifierError: Error {
        case predictionFailed
        case modelNotFound
    }
}

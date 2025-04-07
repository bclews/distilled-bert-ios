//
//  BERTTokenizer.swift
//  auto_tagger
//
//  Created by Ben Clews on 26/2/2025.
//
import UIKit
import CoreML


class BERTTokenizer {
    private let vocabulary: [String: Int]
    private let unkToken = "[UNK]"
    private let clsToken = "[CLS]"
    private let sepToken = "[SEP]"
    private let padToken = "[PAD]"
    
    struct TokenizedInput {
        let ids: MLMultiArray
        let mask: MLMultiArray
    }
    
    init() throws {
        // Load the vocabulary from the saved tokenizer
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "txt") else {
            throw TokenizerError.vocabNotFound
        }
        
        let vocabString = try String(contentsOf: vocabURL, encoding: .utf8)
        let tokens = vocabString.components(separatedBy: .newlines)
        
        var vocab = [String: Int]()
        for (index, token) in tokens.enumerated() {
            if !token.isEmpty {
                vocab[token] = index
            }
        }
        
        self.vocabulary = vocab
    }
    
    func tokenize(text: String, maxLength: Int) throws -> TokenizedInput {
        // Basic tokenization (would need to be replaced with a proper WordPiece tokenizer)
        var tokens = [clsToken]
        
        // Split by whitespace and add each token
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        for word in words {
            if !word.isEmpty {
                if vocabulary[word] != nil {
                    tokens.append(word)
                } else {
                    tokens.append(unkToken)
                }
            }
        }
        
        tokens.append(sepToken)
        
        // Convert tokens to IDs
        var ids = tokens.map { vocabulary[$0] ?? vocabulary[unkToken]! }
        
        // Pad or truncate to maxLength
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        } else if ids.count < maxLength {
            let padding = Array(repeating: vocabulary[padToken]!, count: maxLength - ids.count)
            ids.append(contentsOf: padding)
        }
        
        // Create attention mask (1 for real tokens, 0 for padding)
        let mask = ids.map { $0 == vocabulary[padToken]! ? 0 : 1 }
        
        // Create MLMultiArray with proper shape (batch_size=1, sequence_length=maxLength)
        let idMultiArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let maskMultiArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        
        // Fill the arrays with our values
        for i in 0..<maxLength {
            idMultiArray[[0, i] as [NSNumber]] = NSNumber(value: ids[i])
            maskMultiArray[[0, i] as [NSNumber]] = NSNumber(value: mask[i])
        }
        
        return TokenizedInput(
            ids: idMultiArray,
            mask: maskMultiArray
        )
    }
    
    enum TokenizerError: Error {
        case vocabNotFound
    }
}

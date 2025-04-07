//
//  ContentView.swift
//  auto_tagger
//
//  Created by Ben Clews on 26/2/2025.
//

import SwiftUI

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var resultText: String = "Classification result will appear here"
    @State private var debugText: String = ""
    @State private var classificationResults: [(label: String, value: Double)] = []
    @State private var isAnalyzing: Bool = false
    @State private var showDebug: Bool = false
    @State private var showRawData: Bool = false
    
    // Create an instance of the classifier
    private var classifier: TextClassifier? = {
        do {
            return try TextClassifier()
        } catch {
            print("Error initializing classifier: \(error)")
            return nil
        }
    }()
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("Text Classification")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                TextEditor(text: $inputText)
                    .frame(height: 150)
                    .padding(4)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                    )
                    .padding(.horizontal)
                
                Button(action: analyzeText) {
                    Text("Analyze Text")
                        .fontWeight(.semibold)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(inputText.isEmpty || isAnalyzing)
                
                if isAnalyzing {
                    ProgressView("Analyzing...")
                        .padding()
                }
                
                Text(resultText)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    .padding(.horizontal)
                
                if !classificationResults.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Button(action: { showRawData.toggle() }) {
                            HStack {
                                Text(showRawData ? "Hide Raw Data" : "Show Raw Data")
                                    .fontWeight(.medium)
                                Image(systemName: showRawData ? "chevron.up" : "chevron.down")
                            }
                        }
                        .padding(.horizontal)
                        
                        if showRawData {
                            VStack(alignment: .leading, spacing: 5) {
                                ForEach(classificationResults, id: \.label) { result in
                                    HStack {
                                        Text(result.label)
                                            .font(.system(.body, design: .monospaced))
                                        
                                        Spacer()
                                        
                                        Text("\(String(format: "%.2f", result.value * 100))%")
                                            .font(.system(.body, design: .monospaced))
                                    }
                                    
                                    ProgressView(value: result.value)
                                        .progressViewStyle(LinearProgressViewStyle(tint: colorForConfidence(result.value)))
                                        .frame(height: 4)
                                }
                            }
                            .padding()
                            .background(Color.gray.opacity(0.05))
                            .cornerRadius(8)
                            .padding(.horizontal)
                        }
                    }
                }
                
                if !debugText.isEmpty {
                    VStack {
                        Button(action: { showDebug.toggle() }) {
                            HStack {
                                Text(showDebug ? "Hide Debug Info" : "Show Debug Info")
                                    .font(.footnote)
                                Image(systemName: showDebug ? "chevron.up" : "chevron.down")
                            }
                        }
                        
                        if showDebug {
                            ScrollView {
                                Text(debugText)
                                    .font(.system(.footnote, design: .monospaced))
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.black.opacity(0.05))
                                    .cornerRadius(8)
                            }
                            .frame(height: 200)
                        }
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .padding()
        }
        .onTapGesture {
            hideKeyboard()
        }
    }
    
    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
    
    private func colorForConfidence(_ value: Double) -> Color {
        switch value {
        case 0..<0.3:
            return .red
        case 0.3..<0.6:
            return .orange
        case 0.6..<0.9:
            return .blue
        default:
            return .green
        }
    }
    
    private func analyzeText() {
        guard let classifier = classifier, !inputText.isEmpty else {
            resultText = "Please enter some text to analyze"
            return
        }
        
        isAnalyzing = true
        debugText = ""
        classificationResults = []
        
        // Use a background task to prevent UI freezing during analysis
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try classifier.classify(text: inputText)
                
                // Update UI on the main thread
                DispatchQueue.main.async {
                    resultText = "Classification: \(result.label) (Confidence: \(String(format: "%.2f", result.confidence * 100))%)"
                    classificationResults = result.allProbabilities
                    
                    // Add to debug text
                    debugText += "Raw output feature values:\n"
                    for (key, value) in result.rawOutput {
                        debugText += "  \(key): \(String(format: "%.4f", value))\n"
                    }
                    
                    isAnalyzing = false
                }
            } catch {
                DispatchQueue.main.async {
                    resultText = "Classification error: \(error)"
                    debugText = "Error during classification: \(error)"
                    isAnalyzing = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}

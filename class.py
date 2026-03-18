# Taka Irizarry
# Final Project - Student Performance MLP Classifier
import numpy as np
import csv
import pandas as pd

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Softmax activation function for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# Initialize weights randomly in [-0.05, 0.05]
def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    # Add 1 to input size for bias
    W_input_hidden = np.random.uniform(-0.05, 0.05, (hidden_size, input_size + 1))
    W_hidden_output = np.random.uniform(-0.05, 0.05, (output_size, hidden_size + 1))
    return W_input_hidden, W_hidden_output

# Load and preprocess student performance data
def load_student_data(filename):
    data = []
    df = pd.read_csv(filename)
    
    print(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"Columns: {list(df.columns)}")
    
    # Use GradeClass as the target variable (already encoded as 0-4 for A-F)
    # Features: exclude StudentID, GPA, and GradeClass
    feature_cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
    
    # Check if all expected columns exist
    missing_cols = [col for col in feature_cols + ['GradeClass'] if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Use available columns instead
        feature_cols = [col for col in df.columns if col not in ['StudentID', 'GPA', 'GradeClass']]
        print(f"Using available feature columns: {feature_cols}")
    
    # Get features and normalize them
    features_data = df[feature_cols].values.astype(np.float32)
    
    # Normalize features by dividing each column by its maximum value
    features_max = np.max(features_data, axis=0)
    features_max[features_max == 0] = 1  # Avoid division by zero
    features_normalized = features_data / features_max
    
    print("Feature normalization (max values used):")
    for i, (col, max_val) in enumerate(zip(feature_cols, features_max)):
        print(f"  {col}: max = {max_val:.2f}")
    
    # Prepare data samples
    for i, (_, row) in enumerate(df.iterrows()):
        # Use GradeClass as label (should be 0-4 for A-F)
        label = int(row['GradeClass'])
        
        # Get normalized features for this sample
        features = features_normalized[i]
        
        # Add bias term
        features = np.insert(features, 0, 1.0)
        data.append((label, features))
    
    # Print class distribution
    class_counts = [0] * 5
    for label, _ in data:
        class_counts[label] += 1
    
    grade_labels = ['A', 'B', 'C', 'D', 'F']
    print("Class distribution:")
    for i, (grade, count) in enumerate(zip(grade_labels, class_counts)):
        print(f"  Grade {grade}: {count} samples ({count/len(data)*100:.1f}%)")
    
    return data

# Function to split data into train and test sets
def split_data(data, test_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples")
    return train_data, test_data

# Create one-hot encoded target vector for softmax
def create_targets(label, num_classes=5):
    target = np.zeros(num_classes)
    target[label] = 1.0
    return target

# Training function
def train_network(train_data, test_data, hidden_size, learning_rate=0.1, momentum=0.9, epochs=100):
    # Determine input size from first sample
    input_size = len(train_data[0][1]) - 1  # Subtract 1 for bias
    output_size = 5  # 5 classes 
    
    # Initialize weights and momentum terms
    W_input_hidden, W_hidden_output = initialize_weights(input_size, hidden_size, output_size)
    delta_input_hidden = np.zeros_like(W_input_hidden)
    delta_hidden_output = np.zeros_like(W_hidden_output)
    
    # Track accuracies
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        np.random.shuffle(train_data)  # Shuffle training data
        
        for label, features in train_data:
            # Forward pass
            hidden_input = np.dot(W_input_hidden, features)
            hidden_output = relu(hidden_input)
            hidden_output = np.insert(hidden_output, 0, 1.0)  # Add bias for hidden layer
            
            final_input = np.dot(W_hidden_output, hidden_output)
            final_output = softmax(final_input.reshape(-1, 1)).flatten()
            
            # Create target vector
            target = create_targets(label, output_size)
            
            # Backward pass
            # Output layer error (cross-entropy with softmax)
            output_error = target - final_output
            
            # Hidden layer error
            hidden_error = np.dot(W_hidden_output[:, 1:].T, output_error) * relu_derivative(hidden_output[1:])
            
            # Update weight deltas with momentum
            delta_hidden_output = learning_rate * np.outer(output_error, hidden_output) + momentum * delta_hidden_output
            delta_input_hidden = learning_rate * np.outer(hidden_error, features) + momentum * delta_input_hidden
            
            # Apply weight updates
            W_hidden_output += delta_hidden_output
            W_input_hidden += delta_input_hidden
        
        # Evaluate network performance
        train_acc = evaluate_network(train_data, W_input_hidden, W_hidden_output)
        test_acc = evaluate_network(test_data, W_input_hidden, W_hidden_output)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    
    return W_input_hidden, W_hidden_output, train_accs, test_accs

# Evaluation function
def evaluate_network(data, W_input_hidden, W_hidden_output):
    correct = 0
    for label, features in data:
        pred = predict(features, W_input_hidden, W_hidden_output)
        if pred == label:
            correct += 1
    return correct / len(data)

# Prediction function
def predict(features, W_input_hidden, W_hidden_output):
    hidden_input = np.dot(W_input_hidden, features)
    hidden_output = relu(hidden_input)
    hidden_output = np.insert(hidden_output, 0, 1.0)  # Add bias
    
    final_input = np.dot(W_hidden_output, hidden_output)
    final_output = softmax(final_input.reshape(-1, 1)).flatten()
    
    return np.argmax(final_output)  # Return class with highest probability

# Compute confusion matrix
def compute_confusion_matrix(data, W_input_hidden, W_hidden_output):
    matrix = np.zeros((5, 5), dtype=int)  # 5x5 for grades A-F
    for true_label, features in data:
        pred = predict(features, W_input_hidden, W_hidden_output)
        matrix[true_label][pred] += 1  # Row: true label, Column: predicted label
    return matrix

# Calculate precision, recall, and F1-score for each class
def calculate_metrics(confusion_matrix):
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        # Precision
        tp = confusion_matrix[i][i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall 
        fn = np.sum(confusion_matrix[i, :]) - tp
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score (precision * recall) / (precision + recall)
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1_score

if __name__ == "__main__":
    # Load your student performance dataset
    print("Loading student performance dataset...")
    # Load the complete dataset
    all_data = load_student_data("Student_performance_data.csv")
    # Split into train and test sets
    train_data, test_data = split_data(all_data, test_ratio=600/len(all_data), random_seed=42)
    # Grade labels for output
    grade_labels = ['A', 'B', 'C', 'D', 'F']
    # Experiment 1: Vary hidden layer size
    hidden_sizes = [10, 25, 50]
    fixed_momentum = 0.9
    
    print("\n" + "="*60)
    print("EXPERIMENT 1: Varying Hidden Layer Size")
    print("="*60)
    
    best_test_acc = 0
    best_hidden_size = 0
    
    for hidden_size in hidden_sizes:
        print(f"\nTraining with hidden_size={hidden_size}, momentum={fixed_momentum}")
        print("-" * 50)
        
        W_input_hidden, W_hidden_output, train_accs, test_accs = train_network(
            train_data, test_data, hidden_size, learning_rate=0.1, momentum=fixed_momentum, epochs=100)
        
        final_test_acc = test_accs[-1]
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_hidden_size = hidden_size
        
        # Compute and display confusion matrix
        matrix = compute_confusion_matrix(test_data, W_input_hidden, W_hidden_output)
        print(f"\nConfusion Matrix (Hidden Units = {hidden_size}):")
        print("Pred " + "  ".join(f"{grade:>3}" for grade in grade_labels))
        print("True")
        for i, row in enumerate(matrix):
            print(f"{grade_labels[i]:>3}  ", "  ".join(f"{val:>3}" for val in row))
        
        # Calculate and display metrics
        precision, recall, f1_score = calculate_metrics(matrix)
        print(f"\nDetailed Metrics for Hidden Size {hidden_size}:")
        print(f"{'Grade':<5} {'Precision':<9} {'Recall':<9} {'F1-Score':<9}")
        print("-" * 40)
        for i, grade in enumerate(grade_labels):
            print(f"{grade:<5} {precision[i]:<9.3f} {recall[i]:<9.3f} {f1_score[i]:<9.3f}")
        
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
    
    print(f"\nBest Hidden Size: {best_hidden_size} (Test Accuracy: {best_test_acc:.4f})")
    
    # Experiment 2: Vary momentum
    momenta = [0.3, 0.6, 0.9]
    fixed_hidden_size = best_hidden_size  # Use best hidden size from experiment 1
    
    print("\n" + "="*60)
    print("EXPERIMENT 2: Varying Momentum")
    print("="*60)
    
    best_test_acc = 0
    best_momentum = 0
    
    for momentum in momenta:
        print(f"\nTraining with hidden_size={fixed_hidden_size}, momentum={momentum}")
        print("-" * 50)
        
        W_input_hidden, W_hidden_output, train_accs, test_accs = train_network(
            train_data, test_data, fixed_hidden_size, learning_rate=0.1, momentum=momentum, epochs=100)
        
        final_test_acc = test_accs[-1]
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_momentum = momentum
        
        matrix = compute_confusion_matrix(test_data, W_input_hidden, W_hidden_output)
        print(f"\nConfusion Matrix (Momentum = {momentum}):")
        print("Pred " + "  ".join(f"{grade:>3}" for grade in grade_labels))
        print("True")
        for i, row in enumerate(matrix):
            print(f"{grade_labels[i]:>3}  ", "  ".join(f"{val:>3}" for val in row))
        
        precision, recall, f1_score = calculate_metrics(matrix)
        print(f"\nDetailed Metrics for Momentum {momentum}:")
        print(f"{'Grade':<5} {'Precision':<9} {'Recall':<9} {'F1-Score':<9}")
        print("-" * 40)
        for i, grade in enumerate(grade_labels):
            print(f"{grade:<5} {precision[i]:<9.3f} {recall[i]:<9.3f} {f1_score[i]:<9.3f}")
        
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
    
    print(f"\nBest Momentum: {best_momentum} (Test Accuracy: {best_test_acc:.4f})")
    
    # Experiment 3: Vary learning rate (ETA)
    learning_rates = [0.001, 0.01, 0.1]
    fixed_momentum = best_momentum  # Use best momentum from experiment 2
    
    print("\n" + "="*60)
    print("EXPERIMENT 3: Varying Learning Rate")
    print("="*60)
    
    best_test_acc = 0
    best_lr = 0
    
    for lr in learning_rates:
        print(f"\nTraining with hidden_size={fixed_hidden_size}, learning_rate={lr}, momentum={fixed_momentum}")
        print("-" * 50)
        
        W_input_hidden, W_hidden_output, train_accs, test_accs = train_network(
            train_data, test_data, fixed_hidden_size, learning_rate=lr, momentum=fixed_momentum, epochs=100)
        
        final_test_acc = test_accs[-1]
        if final_test_acc > best_test_acc:
            best_test_acc = final_test_acc
            best_lr = lr
        
        matrix = compute_confusion_matrix(test_data, W_input_hidden, W_hidden_output)
        print(f"\nConfusion Matrix (Learning Rate = {lr}):")
        print("Pred " + "  ".join(f"{grade:>3}" for grade in grade_labels))
        print("True")
        for i, row in enumerate(matrix):
            print(f"{grade_labels[i]:>3}  ", "  ".join(f"{val:>3}" for val in row))
        
        precision, recall, f1_score = calculate_metrics(matrix)
        print(f"\nDetailed Metrics for Learning Rate {lr}:")
        print(f"{'Grade':<5} {'Precision':<9} {'Recall':<9} {'F1-Score':<9}")
        print("-" * 40)
        for i, grade in enumerate(grade_labels):
            print(f"{grade:<5} {precision[i]:<9.3f} {recall[i]:<9.3f} {f1_score[i]:<9.3f}")
        
        print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
    
    print(f"\nBest Learning Rate: {best_lr} (Test Accuracy: {best_test_acc:.4f})")
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Best Configuration:")
    print(f"Hidden Size: {best_hidden_size}")
    print(f"Momentum: {best_momentum}")
    print(f"Learning Rate: {best_lr}")
    print(f"Final Test Accuracy: {best_test_acc:.4f}")
    
    # Train final model with best parameters
    print(f"\nTraining final model with best parameters")
    final_W_input_hidden, final_W_hidden_output, final_train_accs, final_test_accs = train_network(
        train_data, test_data, best_hidden_size, learning_rate=best_lr, momentum=best_momentum, epochs=100)
    
    # Final confusion matrix and metrics
    final_matrix = compute_confusion_matrix(test_data, final_W_input_hidden, final_W_hidden_output)
    print(f"\nFinal Confusion Matrix:")
    print("Pred= " + "  ".join(f"{grade:>3}" for grade in grade_labels))
    print("True")
    for i, row in enumerate(final_matrix):
        print(f"{grade_labels[i]:>3}  ", "  ".join(f"{val:>3}" for val in row))
    
    final_precision, final_recall, final_f1_score = calculate_metrics(final_matrix)
    print(f"\nFinal Model Performance:")
    print(f"{'Grade':<5} {'Precision':<9} {'Recall':<9} {'F1-Score':<9}")
    print("-" * 40)
    for i, grade in enumerate(grade_labels):
        print(f"{grade:<5} {final_precision[i]:<9.3f} {final_recall[i]:<9.3f} {final_f1_score[i]:<9.3f}")
    
    # Overall metrics
    overall_precision = np.mean(final_precision)
    overall_recall = np.mean(final_recall)
    overall_f1 = np.mean(final_f1_score)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {final_test_accs[-1]:.4f}")
    print(f"  Macro-Avg Precision: {overall_precision:.4f}")
    print(f"  Macro-Avg Recall: {overall_recall:.4f}")
    print(f"  Macro-Avg F1-Score: {overall_f1:.4f}")

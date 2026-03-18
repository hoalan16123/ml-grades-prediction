#Alan Ho
#hoalan@pdx.edu
#CS445 Final Project MLP Regression Model
import numpy as np

def load_data(filename):
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        print(f"Loaded {len(data)} student records")
        
        features = data[:, 1:13]  
        gpa_values = data[:, 13]  
        labels = gpa_to_grade(gpa_values)
        return features, gpa_values, labels
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None




def split_data(features, gpa, labels, test_size=600):

    #split the data so we have 600 test examples
    num_data = len(labels)
    
    indices = np.arange(num_data)
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    return (features[train_indices], features[test_indices], 
            gpa[train_indices], gpa[test_indices],
            labels[train_indices], labels[test_indices])
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0)


def gpa_to_grade(gpa):
    #convert gpa to grade for classification purposes
    labels = np.zeros(len(gpa), dtype=int)
    labels[gpa >= 3.5] = 0                  #A
    labels[(gpa >= 3.0) & (gpa < 3.5)] = 1  #B
    labels[(gpa >= 2.5) & (gpa < 3.0)] = 2  #C
    labels[(gpa >= 2.0) & (gpa < 2.5)] = 3  #D
    labels[gpa < 2.0] = 4                   #F
    return labels

def confusion_matrix(true_labels, predicted_labels):
    num_classes = 5
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(true_labels)):
        matrix[true_labels[i], predicted_labels[i]] += 1
    return matrix

def print_confusion_matrix(matrix):
    grade_names = ['A', 'B', 'C', 'D', 'F']
    print("True | Predicted")
    print("      A    B    C    D    F")
    print("---------------------------")
    for i in range(5):
        row = f"{grade_names[i]:>4} |"
        for j in range(5):
            row += f"{matrix[i, j]:5d}"
        print(row)

class StudentNN:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        #initialize weights 
        self.weights1 = np.random.normal(0, np.sqrt(2.0/input_size), (input_size + 1, hidden_size)) * 0.1
       
        #gpa prediction is a single neuron
        self.weights2 = np.random.normal(0, np.sqrt(2.0/hidden_size), (hidden_size + 1, 1)) * 0.1
        
        #previous weight update initialization for momentum
        self.prev_update_w1 = np.zeros_like(self.weights1)
        self.prev_update_w2 = np.zeros_like(self.weights2)
    
    def forward(self, inputs):
        """Forward propagation"""
        #adding bias term
        inputs_and_bias = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
        
        #hidden layer calculations
        self.hidden_input = np.dot(inputs_and_bias, self.weights1)
        self.hidden = relu(self.hidden_input)
        
        #add hidden layer bias
        self.hidden_bias = np.hstack((self.hidden, np.ones((self.hidden.shape[0], 1))))
        
        #output layer (linear activation)
        self.output_input = np.dot(self.hidden_bias, self.weights2)
      
        self.outputs = self.output_input
        return self.outputs, inputs_and_bias
    
    def backprop(self, inputs_bias, targets, eta, momentum):
        
        targets = targets.reshape(-1, 1)
        
        #calculate delta for output layer
        delta_o = targets - self.outputs 
        
        #hidden layer delta
        delta_h = np.dot(delta_o, self.weights2.T) * np.hstack([relu_derivative(self.hidden), np.ones((self.hidden.shape[0], 1))])
       
        #remove bias
        delta_h = delta_h[:, :-1]
        
        #weight update calculations
        update_w2 = eta * np.dot(self.hidden_bias.T, delta_o) + momentum * self.prev_update_w2
        update_w1 = eta * np.dot(inputs_bias.T, delta_h) + momentum * self.prev_update_w1
        
        #storing weight update calculation for the momentum (for next calculation)
        self.prev_update_w2 = update_w2
        self.prev_update_w1 = update_w1
        
        #adding weight update for new weights
        self.weights2 += update_w2
        self.weights1 += update_w1
        
    
    
    def train_network(self, train_features, train_gpa, test_features, test_gpa, test_labels, max_epochs=100):
        num_samples = train_features.shape[0]
        prev_train_acc = 0.0
        convergence_threshold = 0.0001  #converges if training set is 0.01% different (or less)
        
        for epoch in range(max_epochs + 1):
            indices = np.random.permutation(num_samples)
            train_features_shuffled = train_features[indices]
            train_gpa_shuffled = train_gpa[indices]
            
            #batch size of 1
            for i in range(num_samples):
                train_vector = train_features_shuffled[i:i+1]
                train_gpa_i = train_gpa_shuffled[i:i+1]
                
                
                _, inputs_bias = self.forward(train_vector)
                
                #make it so that we can still see results for epoch 0 (before any training)
                if epoch > 0:
                    self.backprop(inputs_bias, train_gpa_i, self.learning_rate, self.momentum)
            
            
            
            #predict gpa
            train_pred_gpa = self.predict_gpa(train_features)
            test_pred_gpa = self.predict_gpa(test_features)
            
            #convert gpa to the grade letters for classification
            train_pred_labels = gpa_to_grade(train_pred_gpa.flatten())
            test_pred_labels = gpa_to_grade(test_pred_gpa.flatten())
            
            #calculaing accuracy
            train_labels = gpa_to_grade(train_gpa)
            train_acc = np.sum(train_pred_labels == train_labels) / len(train_labels)
            test_acc = np.sum(test_pred_labels == test_labels) / len(test_labels)
            
            print(f"Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            if epoch > 0:
                acc_diff = abs(train_acc - prev_train_acc)
                if acc_diff <= convergence_threshold:
                    print(f"Converged at epoch {epoch} (accuracy difference: {acc_diff:.6f})")
                    break
            
            prev_train_acc = train_acc
    
    def predict_gpa(self, input_data):
        outputs, _ = self.forward(input_data)
        return outputs
    
    def predict_grades(self, input_data):
        gpa_predictions = self.predict_gpa(input_data)
        return gpa_to_grade(gpa_predictions.flatten())
    
    def evaluate(self, input_data, true_gpa, true_labels):
        pred_gpa = self.predict_gpa(input_data)
        
        pred_labels = gpa_to_grade(pred_gpa.flatten())
        accuracy = np.sum(pred_labels == true_labels) / len(true_labels)
        
        return accuracy, pred_labels, pred_gpa.flatten()

def main():
    np.random.seed(42)
    
    features, gpa, labels = load_data("Student_performance_data.csv")
    
    if features is None:
        print("Failed to load data. Exiting.")
        return
    
    #normalize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    #avoid division by zero
    std[std == 0] = 1
    features = (features - mean) / std

    #split data
    train_features, test_features, train_gpa, test_gpa, train_labels, test_labels = split_data(
    features, gpa, labels, test_size=600)


    print(f"\nTraining samples: {len(train_features)}")
    print(f"Testing samples: {len(test_features)}")
    
    print(f"\n{'='*80}")
    print("Training Regression Neural Network ---- Converts GPA to Grades")

    print(f"{'='*80}")
    
    input_size = train_features.shape[1]

    #hyperparameters
    hidden = 25
    epochs = 100  
    
    nn = StudentNN(input_size, hidden, learning_rate=0.01, momentum=0.6)
    
    nn.train_network(train_features, train_gpa, test_features, test_gpa, test_labels, epochs)
    
    print(f"\n{'='*80}")
    print("Final Results")
    print(f"{'='*80}")
    

    #get data for the final accuracies and metrics
    train_acc, train_pred_labels, train_pred_gpa = nn.evaluate(train_features, train_gpa, train_labels)
    test_acc, test_pred_labels, test_pred_gpa = nn.evaluate(test_features, test_gpa, test_labels)
    
    
    print(f"\nFinal Accuracies\n")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    
    
    conf_matrix = confusion_matrix(test_labels, test_pred_labels)
    print("\nConfusion Matrix - GPA predictions converted to grades:")
    print_confusion_matrix(conf_matrix)
    
 
    grade_names = ['A', 'B', 'C', 'D', 'F']
    print(f"{'Grade':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 40)
    
    #calculate classification metrics
    for i, grade in enumerate(grade_names):
        true_p = conf_matrix[i, i]
        false_p = np.sum(conf_matrix[:, i]) - true_p
        false_n = np.sum(conf_matrix[i, :]) - true_p
        
        precision = true_p / (true_p + false_p) if (true_p + false_p) > 0 else 0
        recall = true_p / (true_p + false_n) if (true_p + false_n) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{grade:<6} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f}")

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class Network:
    def __init__(self, input_nodes, hidden_nodes1, hidden_nodes2, output_nodes):
        self.weights_ih = np.random.randn(hidden_nodes1, input_nodes) * 0.1
        self.weights_hh = np.random.randn(hidden_nodes2, hidden_nodes1) * 0.1
        self.weights_ho = np.random.randn(output_nodes, hidden_nodes2) * 0.1

        self.bias_h1 = np.zeros((hidden_nodes1, 1))
        self.bias_h2 = np.zeros((hidden_nodes2, 1))
        self.bias_o = np.zeros((output_nodes, 1))
        self.learning_rate = 0.025


    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def forward_propagation(self, X):
        input_to_hidden1 = np.dot(X, self.weights_ih.T)
        self.hidden1 = self.relu(input_to_hidden1 + self.bias_h1.T)
    
        hidden1_to_hidden2 = np.dot(self.hidden1, self.weights_hh.T)
        self.hidden2 = self.relu(hidden1_to_hidden2 + self.bias_h2.T)
    
        hidden2_to_output = np.dot(self.hidden2, self.weights_ho.T)
        self.outputs = self.softmax(hidden2_to_output + self.bias_o.T)
    
        return self.outputs

    def back_propagation(self, X, y):
        count = X.shape[0]
    
        # Derivative of cross-entropy loss with softmax
        outputs_error = y - self.outputs

        d_weights_ho = np.dot(outputs_error.T, self.hidden2) / count
        d_bias_o = np.sum(outputs_error, axis=0, keepdims=True).T / count
    
        hidden2_error = np.dot(outputs_error, self.weights_ho) * self.relu_derivative(self.hidden2)
      
        d_weights_hh = np.dot(hidden2_error.T, self.hidden1) / count
        d_bias_h2 = np.sum(hidden2_error, axis=0, keepdims=True).T / count
    
        hidden1_error = np.dot(hidden2_error, self.weights_hh) * self.relu_derivative(self.hidden1)
        d_weights_ih = np.dot(hidden1_error.T, X) / count
        d_bias_h1 = np.sum(hidden1_error, axis=0, keepdims=True).T / count

        self.weights_ho += self.learning_rate * d_weights_ho
        self.weights_hh += self.learning_rate * d_weights_hh
        self.weights_ih += self.learning_rate * d_weights_ih
        self.bias_o += self.learning_rate * d_bias_o
        self.bias_h2 += self.learning_rate * d_bias_h2
        self.bias_h1 += self.learning_rate * d_bias_h1
        
    

    def train(self, X_train, y_train, X_validation, y_validation, epochs):
        validation_accuracies = []  
        for epoch in range(epochs):
            self.forward_propagation(X_train)
            self.back_propagation(X_train, y_train)
            validation_accuracy = self.evaluate(X_validation, y_validation)
            validation_accuracies.append(validation_accuracy) 
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy * 100}%')

        plt.plot(validation_accuracies)
        plt.title('Validation Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.savefig('validation_accuracy.png')  
        plt.show()
    
    def evaluate(self, X, y):
        outputs = self.forward_propagation(X)
        predictions = np.argmax(outputs, axis=1)
        targets = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == targets)
        return accuracy

    def test(self, X_test, y_test):
        outputs = self.forward_propagation(X_test)
        predictions = np.argmax(outputs, axis=1)
        targets = np.argmax(y_test, axis=1)
    
        test_accuracy = np.mean(predictions == targets)
        print(f'Test Accuracy: {test_accuracy * 100}%')
    
        for i in range(10):
            correct = np.sum((predictions == i) & (targets == i))
            total = np.sum(targets == i)
            correctness = correct / total if total > 0 else 0
            print(f'Class {i}: Correctness {correctness * 100}%')

def train_network():
    df = pd.read_csv('data.csv')

    y = df['label'].values
    X = df.drop('label', axis=1).values / 255.0

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

    y_train_one_hot = np.eye(10)[y_train]
    y_validation_one_hot = np.eye(10)[y_validation]
    y_test_one_hot = np.eye(10)[y_test]

    network = Network(784, 100, 50, 10)

    print("Started training..")
    network.train(X_train, y_train_one_hot, X_validation, y_validation_one_hot, 500)
    print("Finished training")

    network.test(X_test, y_test_one_hot)

train_network()
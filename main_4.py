#Initializing x and y with numpy
import numpy as np
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

# sigmoid  function
def sigmoid(t):
    '''This will return the sigmoid value of the function'''
    return 1/(1+np.exp(-t))

# derivative sigmoid
def sigmoid_derivative(d):
    return d * (1 - d)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],4) #initializing random weights
        self.weights2 = np.random.rand(4,1)#considering we have 4 nodes in the hidden layer
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''
        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += derv_weights1#updation of weight matrix of layer1
        self.weights2 += derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation

model=NeuralNetwork(X,y)
for i in range(1500):
    if i % 100 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        print ("\n")
  
    model.train(X, y)

#Initializing x and y with numpy
import numpy as np
X=np.array(([0,1],[1,1],[1,0],[0,0]), dtype=float)
y=np.array(([1],[0],[1],[0]), dtype=float)

#Question 1
# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],4) #initializing random weights
        self.weights2 = np.random.rand(4,1)#considering we have 4 nodes in the hidden layer
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''
        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += derv_weights1#updation of weight matrix of layer1
        self.weights2 += derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation

model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()


# Class definition
#Question 2
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],6) #initializing random weights
        self.weights2 = np.random.rand(6,1)#considering we have 6 nodes in the hidden layer
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''
        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += derv_weights1#updation of weight matrix of layer1
        self.weights2 += derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation

model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()


# Class definition
#Question 3
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],6) #initializing random weights
        self.weights2 = np.random.rand(6,1)#considering we have 6 nodes in the hidden layer
        self.bias1=np.random.uniform(size=(1, 6))
        self.bias2= np.random.uniform(size=(1, 1))
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        self.learning_rate = 0.5
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''

        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
        derv_bias2=2*(self.y -self.output)*sigmoid_derivative(self.output)
        derv_bias1=np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)
        derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)#taking columnwise sum for dimension compatibility
        derv_bias2=(1/4) * np.sum(derv_bias2.T, axis=1, keepdims=True)
        
        #derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)
        self.bias1+=derv_bias1.T
        self.bias2+=derv_bias2.T
        self.weights1 += derv_weights1#updation of weight matrix of layer1
        self.weights2 += derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation
        
        
        
        



model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()


#Question 4
def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return np.where(x > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],6) #initializing random weights
        self.weights2 = np.random.rand(6,1)#considering we have 6 nodes in the hidden layer
        self.bias1=np.random.uniform(size=(1, 6))
        self.bias2= np.random.uniform(size=(1, 1))
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        self.learning_rate = 0.1
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = relu(np.dot(self.input, self.weights1) + self.bias1)#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''

        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*derivative_relu(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
        derv_bias2=2*(self.y -self.output)*sigmoid_derivative(self.output)
        derv_bias1=np.dot(2*(self.y -self.output)*derivative_relu(self.output), self.weights2.T)
        derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)#taking columnwise sum for dimension compatibility
        derv_bias2=(1/4) * np.sum(derv_bias2.T, axis=1, keepdims=True)
        
        #derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)
        self.bias1+=derv_bias1.T
        self.bias2+=derv_bias2.T
        self.weights1 += (0.01)*derv_weights1#updation of weight matrix of layer1
        self.weights2 += (0.01)*derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation
        
        
        
        




model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()


#Question 5

class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],6) #initializing random weights
        self.weights2 = np.random.rand(6,1)#considering we have 6 nodes in the hidden layer
        self.bias1=np.random.uniform(size=(1, 6))
        self.bias2= np.random.uniform(size=(1, 1))
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        self.learning_rate = 0.1
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = relu(np.dot(self.input, self.weights1) + self.bias1)#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''

        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*derivative_relu(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
        derv_bias2=2*(self.y -self.output)*derivative_relu(self.output)
        derv_bias1=np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)
        derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)#taking columnwise sum for dimension compatibility
        derv_bias2=(1/4) * np.sum(derv_bias2.T, axis=1, keepdims=True)
        
        #derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)
        self.bias1+=derv_bias1.T
        self.bias2+=derv_bias2.T
        self.weights1 += (0.01)*derv_weights1#updation of weight matrix of layer1
        self.weights2 += (0.01)*derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation
        
        
        
        




model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()

#Question 6



class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x #initializing x
        self.weights1= np.random.rand(self.input.shape[1],6) #initializing random weights
        self.weights2 = np.random.rand(6,1)#considering we have 6 nodes in the hidden layer
        self.bias1=np.random.uniform(size=(1, 6))
        self.bias2= np.random.uniform(size=(1, 1))
        self.y = y#initializing y
        self.output = np. zeros(y.shape)#initializing the output
        self.learning_rate = 0.1
        
    def feedforward(self):
        '''This will perform the forward propagation for the next 2 layers'''
        self.layer1 = relu(np.dot(self.input, self.weights1) + self.bias1)#calcuation of w_T*X+b for layer1
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)#calcuation of w_T*X+b for layer2
        return self.layer2
        
    def backprop(self):
        '''Back propagation of the final hidden layers to initial layers'''

        derv_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*derivative_relu(self.output))#backpropagation of layer2
        derv_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*derivative_relu(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
        derv_bias2=2*(self.y -self.output)*derivative_relu(self.output)
        derv_bias1=np.dot(2*(self.y -self.output)*derivative_relu(self.output), self.weights2.T)
        derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)
        derv_bias2=(1/4) * np.sum(derv_bias2.T, axis=1, keepdims=True)#taking columnwise sum for dimension compatibility
        
        #derv_bias1=(1/4) * np.sum(derv_bias1.T, axis=1, keepdims=True)
        self.bias1+=derv_bias1.T
        self.bias2+=derv_bias2.T
        self.weights1 += (0.01)*derv_weights1#updation of weight matrix of layer1
        self.weights2 += (0.01)*derv_weights2#updation of weight matrix of layer2

    def train(self, X, y):
        self.output = self.feedforward()#Forward Propagation
        self.backprop()#Backward Propagation
        
        
        
        





model=NeuralNetwork(X,y)
import matplotlib.pyplot as plt
iterations =100
cost_list = []
for i in range(100):
    if i % 10 ==0:#For each 100 epochs output will come 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(model.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - model.feedforward())))) # mean sum squared loss
        cost=np.mean(np.square(y - model.feedforward()))
        cost_list.append(np.mean(np.square(y - model.feedforward())))

        
  
    model.train(X, y)
print(cost_list)
plt.plot(cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations for XOR Problem")
plt.show()

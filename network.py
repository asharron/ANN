from node import node
class network():

    #Constructor for the network
    def __init__(self, arch, start_weights):
        self.neurons = [] # Two dimensional array to hold all the neurons
        self.eta = 0.01 #Standard learning rate

        self.dimensions = arch # What the dimensions of each layer will be
        nextLayer = 1 # Easy access to next layer

        for i in range(len(self.dimensions)): # For the number of layers 
            nodeArray = [] # Create new row
            for j in range(self.dimensions[i]): #Each node in that layer
                nodeWeights = [] #Weights for the node
                count = 0 #Keep track of which weight we are on
                if i < len(self.dimensions)-1: #Only do this if it isn't the output layer
                    for k in range(self.dimensions[nextLayer]):
                        nodeWeights.append(start_weights[j + (self.dimensions[i] * count)]) #Grab the right weights for the node
                        count += 1
                newNode = node() #Create new node and give it its weights
                newNode.set_weights(nodeWeights) #Set the weights of the node
                nodeArray.append(newNode) #Add it to the array for this layer
            nextLayer += 1
            self.neurons.append(nodeArray) #Add the row to the neurons matrix

    #Helper function to add ouput times the weights
    def calc(self,neuron, weightIndex):
        return neuron.output * neuron.weights[weightIndex]

    #Feed the input forward through the network
    def feedforward(self, xin):
        layers = len(self.dimensions)
        for i in range(self.dimensions[0]): #Set the output of the input layer to the input
            self.neurons[0][i].output = xin[i]

        #Calculate the output of each node in the network
        for layer in range(1, layers): #For each layer
            for node in range(self.dimensions[layer]): #For each node in that layer
                #Calculate the sum of output * weight for each neuron in the previous layer
                summation = sum([self.calc(neuron,node) for neuron in self.neurons[layer-1]])
                self.neurons[layer][node].fire(summation) #Fire that node and save the output for it
        fun = lambda x : x.output #Quick function to return the output of a node
        results = [ fun(x) for x in self.neurons[-1]] #Make a list of all the outputs of the last layer
        return results #Return the results of the output layer

    def backprop(self, xin, y):
        layers = len(self.dimensions)
        results = self.feedforward(xin)
        #Set the Error of the output layer
        for output_node in range(self.dimensions[-1]):
            curr_node = self.neurons[-1][output_node]
            #Calculate the error
            curr_node.error = self.cost(curr_node.output, y[output_node]) * self.sig_prime(curr_node.zsum)


        #Go Backwards and calculate the error for the rest of the layers
        #Loop for each layer
        for layer in range(layers-2,-1,-1):
            #Loop for each node in that layer
            for node in range(self.dimensions[layer]):
                #Loop for each node in the next layer
                for next_node in range(self.dimensions[layer+1]):
                    self.neurons[layer][node].error = (self.neurons[layer][node].weights[next_node] * self.neurons[layer+1][next_node].error) * \
                            self.sig_prime(self.neurons[layer][node].zsum) #Calculate the error of the node
                    #Calculate the error of the weight connecting to the next node
                    self.neurons[layer][node].weight_error.append(self.neurons[layer][node].output * self.neurons[layer+1][next_node].error)

        #Changed the weights in the network based on the weight_error
        for layer in range(layers-1):
            for node in range(self.dimensions[layer]):
                for weight in range(len(self.neurons[layer][node].weights)):
                    self.neurons[layer][node].weights[weight] = (self.neurons[layer][node].weights[weight] - self.eta) * \
                            self.neurons[layer][node].weight_error[weight]

        trained_weights = [] #Newly trained weights
        weight_error = [] #Error of each weight
        #Grab weights and weight error
        for layer in range(layers-1):
            for connection in range(self.dimensions[layer+1]):
                for node in range(self.dimensions[layer]):
                    trained_weights.append(self.neurons[layer][node].weights[connection])
                    weight_error.append(self.neurons[layer][node].weight_error[connection])

        #Grab the output error
        output_error = [n.error for a in self.neurons for n in a] #double list comprehension

        return (output_error,weight_error,trained_weights)

    #Cost function
    def cost(self, activation, correct):
        return activation * correct

    #Quick grab for sigmoid
    def sigmoid(self, x):
        return x / (1 + abs(x))

    #Quick grab for sigmoid derivative
    def sig_prime(self, z):
        return self.sigmoid(z) * (1-self.sigmoid(z))

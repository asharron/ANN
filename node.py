class node():

    def __init__(self):
        self.weights = [] #Each weight it has for the layer below it
        self.output = 0 #The output after activation has been applied
        self.weight_error = [] #Error for each wieght
        self.error = 0 #Error of the node itself
        self.zsum = 0 #The sum before it is put through activation function

    #Sets the weights 
    def set_weights(self,in_weights):
        self.weights = in_weights
    
    #Activation function for this particular node
    def activation(self,x):
        return x / (1 + abs(x))

    #Derivative of the activation function
    def activation_deriv(self,x):
        return self.activation(x) * (1-self.activation(x))

    #Does activation on a summed number and then stores the output
    def fire(self, x):
        self.zsum = x #Appends the z vector to its list
        self.output = self.activation(x) #Activate on the sum
        return self.output

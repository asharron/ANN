from node import node
from network import network


def read_weights():
    startweights = []

    #Read the f
    with open("weights.in") as f:
        arch = f.readline().split()
        arch = list(map(int, arch)) #Turn the list from strings to integers
        numRows = sum(arch) - arch[0] #Add the entire list to know how many rows we have
        for i in range(numRows):
            row = f.readline().split()
            row = list(map(float,row)) #Turn the row into floats
            startweights += row #I don't want the weights to be a double array, so just add the contents
    return (startweights, arch)

def read_patterns():
    #Read the patterns
    with open("patterns.in") as f:
        numPatterns = int(f.readline().split()[0])
        xin = []
        for i in range(numPatterns):
            xin.append(f.readline().split())
            xin[i] = list(map(float,xin[i])) #Make sure the input is a float not string
    return xin

def correct_pat():
    yout = []
    with open("correct.out") as f:
        numPatterns = int(f.readline().split()[0])
        for i in range(numPatterns):
            yout.append(f.readline().split())
            yout[i] = list(map(float,yout[i])) #Convert to floats
    return yout

def output_file():
    #Write output file
    with open("output.out","a") as f:
        results = list(map(str, results)) #Convert results to string 
        f.write(" ".join(results)) #Write the results

startweights,arch = read_weights()
net = network(arch, startweights) #Create the network
xin = read_patterns()
yout = correct_pat()

for xi,yo in zip(xin,yout):
    print("#######################")
    results = net.feedforward(xi) #Feed forward the network
    output_error,weight_error,trained_weights = net.backprop(xi, yo) #Do backprop and obtain the error while adjusting weights
    print("You input the patterns of ",xi, "\n")
    print("Your results for these patterns were: ", results, "!\n")
    print("The Error of each node is ",output_error,"\n")
    print("The Error of each weight is ",weight_error,"\n")
    print("The New Trained weights are ", trained_weights, "\n")
    print("#######################")

import numpy as np
class NeuralNetwork:
    def __init__(self,sizes,learning_rate):
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # Initialize & randomize biases
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # Initialize & randomize weights
        self.learning_rate = learning_rate
    def feedforward(self,inputs):
        a = np.transpose([inputs])
        for b,w in zip(self.biases,self.weights):
            a = self.sigmoid(np.dot(w,a)+b)
        return a
    def train(self,inputs,supervised_outputs_array):
        x = np.transpose([inputs])
        supervised_outputs_array = np.transpose([supervised_outputs_array])
        a = [x]
        for i in range(0,len(self.sizes)-1): # Feedforward and append all activations to array a
            a.append(self.sigmoid(np.dot(self.weights[i],a[i]) + self.biases[i]))
        e = a[-1] - supervised_outputs_array # Error rate
        for l in range(1,len(self.sizes)): # Backpropagation , l = index from end
            delta = self.learning_rate*e*a[-l]*(1-a[-l])*(-1)
            e = np.dot(np.transpose(self.weights[-l]),e) # Next e
            self.biases[-l] += delta
            self.weights[-l] += np.dot(delta,np.transpose(a[-l-1]))
            
    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

nn = NeuralNetwork([2,3,1],0.1)
train_data = [([1,1],0),([1,0],[1]),([0,1],[1]),([0,0],[0])]
print("Untrained:\n1 xor 1 =",nn.feedforward([1,1]))
for i in range(100000):
    r = train_data[np.random.randint(0,len(train_data))]
    nn.train(r[0],r[1])
print("\nTrained:\n1 xor 1 =",nn.feedforward([1,1]))
print("1 xor 0 =",nn.feedforward([1,0]))
print("0 xor 1 =",nn.feedforward([0,1]))
print("0 xor 0 =",nn.feedforward([0,0]))

#debug
'''c=30000
    if(d%c==0): #in feedforward loop
        print("\nIteration #{0}:\ninputs:\n{1}\nweights:\n{2}\nbiases:\n{3}\noutputs:\n{4}\nerror:\n{5}".format(i,inputs,self.weights,self.biases,a[-1],e))
    if(d%c==0): #in backprop loop
        p = -l+len(self.sizes)-1
        print("delta_b{0}:\n{1}\ndelta_w{0}:\n{2}".format(p,delta,np.dot(delta,np.transpose(a[-l-1]))))
        if(p!=0):
            print("e{0}:\n{1}".format(p,e))'''
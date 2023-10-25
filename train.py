from autograd import Value
import perceptron as pr
import numpy as np


nn = pr.MLP(3, [4, 4, 1])
print(nn)
print("number of parameters", len(nn.parameters()))



for k in range(20):

    # forward and calculate loss (mean square error)
    total_loss, acc = pr.loss(nn, 4)
    ...
    # backward (zero_grad + backward)
    nn.zero_grad()
    total_loss.backward()

    # update
    learning_rate = 0.15
    for p in nn.parameters():
        p.data -= learning_rate * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
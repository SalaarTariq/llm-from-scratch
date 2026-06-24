import math
import numpy as np
import matplotlib.pyplot as plt
"""
def f(X):
    return 3*X**2 -4*X + 5

#print(f(3.0))

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
#plt.plot(xs, ys)
#plt.show()

# Complex system 
a = 2.0
b = -3
c = 10
d = a*c + b


h = 0.00001
d1 = a*b+c
a +=h
d2 = a*b+c
print("d1",d1)
print("d2",d2)
print('slope', (d2-d1)/h)"""


class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0.0
        self._children = _children
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, _children=(self,), _op=f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    def __radd__(self, other):
        return self + other
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), _children=(self,), _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, _children=(self,), _op='tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    
a = Value(2.0, label ='A')
b = Value(-3.0, label='B')
c = Value(10.0, label='C')
e = a * b ; e.label = 'E'
d = e+c; d.label = 'D'
f = Value(-2.0, label='F')
l = d * f; l.label = 'L'




from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot



l.grad = 1.0
f.grad = 4.0
d.grad = -2.0
c.grad = -2.0
e.grad = -2.0
a.grad = -2.0 * b.data
b.grad = -2.0 * a.data

#dot = draw_dot(l)
#dot.render('Micrograd_from_scratch/graph', view=True)



a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

e = a * b
d = e + c
l = d * f 

#print(l.data)





def lol():
    h= 0.00001


    a = Value(2.0, label ='A')
    b = Value(-3.0, label='B')
    c = Value(10.0, label='C')
    e = a * b ; e.label = 'E'
    d = e+c; d.label = 'D'
    f = Value(-2.0, label='F')
    l = d * f; l.label = 'L'
    L1 = l.data


    a = Value(2.0 +h, label ='A')
    b = Value(-3.0, label='B')
    c = Value(10.0, label='C')
    e = a * b ; e.label = 'E'
    d = e+c; d.label = 'D'
    f = Value(-2.0, label='F')
    l = d * f; l.label = 'L'
    L2 = l.data


    print((L2-L1)/h)

#lol()


#inputes x1 and x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

#weights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

#bias b
bi = Value(6.8813735870195432, label='bi')

#x1*w1 + x2w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + bi; n.label = 'n'


e = (2*n).exp()
o = (e-1)/(e+1);



o.label = 'o'




"""o.grad = 1.0
n.grad=  1 - o.data ** 2
x1w1x2w2.grad = n.grad
bi.grad = n.grad
x2w2.grad = x1w1x2w2.grad
x1w1.grad = x1w1x2w2.grad
x2.grad = x2w2.grad * w2.data
x1.grad = x1w1.grad * w1.data
w2.grad = x2w2.grad * x2.data
w1.grad = x1w1.grad * x1.data"""


"""
o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
w2._backward()
w1._backward()
x1._backward()
x2._backward()
draw_dot(o).render('Micrograd_from_scratch/NN', view=False)

topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)
build_topo(o)
print(topo)


a= Value(-4.0, label='a')
b= Value(2.0, label='b')
print(a+b)
print(a*b)
print(a/b)
print(a-b)
"""




o.backward()
#draw_dot(o).render('Micrograd_from_scratch/NN', view=False)



import torch
"""""
x1 = torch.Tensor([2.0]).double()              ;x1.requires_grad = True
x2 = torch.Tensor([0.0]).double()              ;x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double()             ;w1.requires_grad = True
w2 = torch.Tensor([1.0]).double()              ;w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double() ;b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()

print("____")
print("x2", x2.grad.item())
print("w2", w2.grad.item())
print("x1", x1.grad.item())
print("w1", w1.grad.item())

torch.tensor([2.0]).double().dtype"""


class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

    

"""x= [2.0,3.0,-1.0]
n = MLP(3, [4,4,1])
print(n(x))
draw_dot(n(x)).render('Micrograd_from_scratch/Neuron', view=False)"""


xs = [
    [2.0, 3.0, -1.0],
    [0.0, 1.0, 2.0],
    [1.0, -1.0, 0.0],
    [1.0,1.0,-1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
model = MLP(3, [4,4,1])
for k in range(400):
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    for p in model.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)


print("ypred", ypred)






"""
print(ypred)
print(len(model.parameters()))
print(model.layers[0].neurons[0].w[0].data) 
draw_dot(loss).render('Micrograd_from_scratch/Neuron', view=False)"""



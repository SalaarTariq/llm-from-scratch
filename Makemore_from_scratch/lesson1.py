import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline


words = open(os.path.join(os.path.dirname(__file__), 'names.txt'), 'r').read().splitlines()
#print(words[:10])

b = {}
for w in words:
    chs = ['<S>'] + list(w)+ ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram, 0) + 1
        #print(ch1, ch2)

#print(sorted(b.items(), key=lambda x: x[1]))




N = torch.zeros((27,27), dtype=torch.int32)




chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


p = (N[0]+1).float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
#print(p)

ix =torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#print(itos[ix])
#print(p.sum)


g = torch.Generator().manual_seed(2147483647)
ix=0

P = N.float()
P = P/P.sum(1, keepdim=True)
for i in range(20):
    out = []
    while True:
        p = P[ix].float()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break   
#    print(''.join(out))


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.savefig(os.path.join(os.path.dirname(__file__), 'bigram_counts.png'), bbox_inches='tight')
# plt.show()

#print('stoi =', stoi)
#print('itos =', itos)



n=0
log_likelihood = 0
for w in words[:3]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2} {prob:.4f} {logprob:.4f}')
"""print(f'{log_likelihood=}')
nll = -log_likelihood 
print(f'{nll=}')
print(f'{nll/n=}')
"""


xs, ys = [], []
for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)


print('xs =', xs)
print('ys =', ys)


g = torch.Generator().manual_seed(2147483647)
W = torch.randn(27,27, generator=g)

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
print('xenc =', xenc)
#print('xenc.shape =', xenc.shape)
plt.imshow(xenc)
plt.savefig(os.path.join(os.path.dirname(__file__), 'one_hot_encoding.png'), bbox_inches='tight')




W = torch.randn(27,27)
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)




print('xenc @ W =', xenc @ W)
print('xenc @ W.shape =', (xenc @ W).shape)






print("____________________________________________________")

nlls = torch.zeros(5)
for i in range(5):
    x = xs[i].item()
    y = ys[i].item()
    print("-------")
    print(f'bigram example {i+1}: {itos[x]}{itos[y]}(indexes {x}, {y})')
    print("input to the neural net : ", x)
    print("output probabilities from the neural net: ", probs[i])
    print(f"laberl (actual next character): ",y)
    p = probs[i,y]
    print("probability assigned by the net to the correct character: ", p.item())
    logp = torch.log(p)
    print("log likelihood: ", logp.item())
    nll = -logp
    print('negative log likelihood: ', nll.item())
    nlls[i] = nll
    print("================")
    print("average negative log likelihood, i.e. loss = ", nlls.mean().item())


g = torch.Generator().manual_seed(2147483647)
W = torch.randn(27, 27, generator=g, requires_grad=True)
xenc = F.one_hot(xs, num_classes=27).float()

logits = xenc @ W
counts = logits.exp()
probs = counts/ counts.sum(1,keepdim=True)
loss = -probs[torch.arange(5), ys].log().mean()

W.grad = None
loss.backward()

# -------------------- ADD THESE --------------------
print("Old loss:", loss.item())
print("Gradient mean:", W.grad.abs().mean().item())
print("Gradient max :", W.grad.abs().max().item())

old_W = W.clone()

W.data += -0.1 * W.grad

print("Weights changed:", not torch.allclose(old_W, W))

# Recompute the loss after updating W
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
new_loss = -probs[torch.arange(5), ys].log().mean()

print("New loss:", new_loss.item())
print("Loss difference:", (new_loss - loss).item())
# ---------------------------------------------------




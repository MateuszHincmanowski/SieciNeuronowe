import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn  as sk

zero = np.array([
    [-1, -1, 1, 1, 1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [1, 1, -1, -1, -1, -1, 1, 1],
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [-1, -1, 1, 1, 1, 1, -1, -1],
])
jeden = np.array([
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1, -1],
    [-1,  1,  1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1,  1, -1],
])
dwa = np.array([
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1,  1,  1, -1, -1, -1, -1],
    [-1,  1,  1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1,  1, -1],
])
trzy = np.array([
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1,  1,  1,  1, -1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
])
cztery = np.array([
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1, -1,  1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1,  1,  1, -1, -1],
    [ 1,  1, -1, -1,  1,  1, -1, -1],
    [ 1,  1,  1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
])
piec = np.array([
    [-1,  1,  1,  1,  1,  1,  1, -1],
    [-1,  1,  1, -1, -1, -1, -1, -1],
    [-1,  1,  1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1, -1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
])
szesc = np.array([
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1, -1, -1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1, -1, -1, -1],
    [ 1,  1, -1, -1,  1,  1, -1, -1],
    [ 1,  1, -1, -1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
])
siedem = np.array([
    [-1,  1,  1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1, -1,  1,  1, -1, -1, -1],
    [-1, -1,  1,  1, -1, -1, -1, -1],
    [-1, -1,  1,  1, -1, -1, -1, -1],
])
osiem = np.array([
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1, -1, -1],
])
dziewiec = np.array([
    [-1, -1,  1,  1,  1,  1, -1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1,  1,  1, -1, -1,  1,  1, -1],
    [-1, -1,  1,  1,  1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1, -1,  1,  1, -1],
    [-1, -1, -1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1,  1, -1, -1, -1],
])

digits = [zero,jeden,dwa,trzy,cztery,piec,szesc,siedem,osiem,dziewiec]
patterns = np.array([d.flatten() for d in digits])
print(patterns.shape)

fig,axes = plt.subplots(2,5,figsize=(10,4))
for i , ax in enumerate(axes.flatten()):
    ax.imshow(digits[i],cmap="grey_r")
    ax.set_title(str(i))
    ax.axis("off")
plt.show()

def add_noise(vec,level):
    k = round(level*64)
    idx = np.random.choice(64, k, replace=False)
    noisy = vec.copy()
    noisy[idx] = noisy[idx]*-1
    return noisy

def show_noise(original,noisy,level):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(original.reshape(8,8), cmap="grey_r")
    axes[0].set_title("Oryginał")

    axes[1].imshow(noisy.reshape(8,8), cmap="grey_r")
    axes[1].set_title(f"Szum {int(level * 100)}%")

    plt.show()

for level in (0.1, 0.2, 0.3):
    show_noise(patterns[0], add_noise(patterns[0], level), level)

noisy_10 = np.array([add_noise(p, 0.1) for p in patterns])
noisy_20 = np.array([add_noise(p, 0.2) for p in patterns])
noisy_30 = np.array([add_noise(p, 0.3) for p in patterns])

print(noisy_10.shape, noisy_20.shape, noisy_30.shape)


class Hopfield:
    def __init__(self, n=64):
        self.n = n
        self.W = np.zeros((n, n))

    def local_field(self, s, i):
       return np.dot(self.W[i], s)

    def update_neuron(self, s, i):
        h = self.local_field(s, i)
        if h>0:
            s[i]=1
        elif h<0 :
            s[i]=-1
    def train_hebb(self,patterns):
        self.W = np.zeros((self.n, self.n))
        for p in patterns:
            self.W += np.outer(p, p)
        self.W = self.W / self.n
        np.fill_diagonal(self.W, 0)

    def recall(self, s0, max_iter=5000):
        s = s0.copy()
        no_change = 0
        for it in range(max_iter):
            i = np.random.randint(self.n)
            old = s[i]
            self.update_neuron(s, i)
            if s[i] == old:
                no_change += 1
            else:
                no_change = 0
            if no_change >= 4 * self.n:
                break
        return s


net = Hopfield()
net.train_hebb(patterns)

# czyste wzorce
for k, p in enumerate(patterns):
    print(k, np.array_equal(net.recall(p), p))

# zaszumione
for level, data in ((0.1, noisy_10), (0.2, noisy_20), (0.3, noisy_30)):
    for k in range(10):
        out = net.recall(data[k])
        print(level, k, np.sum(out != patterns[k]))

net = Hopfield()
net.train_hebb(patterns[[0, 1]])      # tylko 0 i 1
for k in (0, 1):
    print(k, np.array_equal(net.recall(patterns[k]), patterns[k]))

for m in range(2, 11):
    net = Hopfield()
    net.train_hebb(patterns[:m])
    ok = sum(np.array_equal(net.recall(patterns[k]), patterns[k]) for k in range(m))
    print(m, "wzorców, odtworzono czystych:", ok)

net = Hopfield()
net.train_hebb(patterns[:3])

for level, data in ((0.1, noisy_10), (0.2, noisy_20), (0.3, noisy_30)):
    for k in range(3):
        out = net.recall(data[k])
        print(level, k, np.sum(out != patterns[k]))   # 0 = naprawiony


def shift(img, dy, dx):
    out = np.full((8, 8), -1)
    for y in range(8):
        for x in range(8):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 8 and 0 <= nx < 8:
                out[ny, nx] = img[y, x]
    return out


digits_30 = []
for d in digits:
    digits_30.append(d)              # oryginał
    digits_30.append(shift(d, 0, 1)) # w prawo
    digits_30.append(shift(d, 0, -1))  # w lewo

patterns_30 = np.array([d.flatten() for d in digits_30])
print(patterns_30.shape)

fig, axes = plt.subplots(3, 10, figsize=(14, 5))
for n, ax in enumerate(axes.flatten()):
    k, v = n % 10, n // 10           # k = cyfra, v = wariant
    ax.imshow(digits_30[k * 3 + v], cmap="gray_r")
    ax.axis("off")
plt.show()


noisy30_10 = np.array([add_noise(p, 0.1) for p in patterns_30])
noisy30_20 = np.array([add_noise(p, 0.2) for p in patterns_30])
noisy30_30 = np.array([add_noise(p, 0.3) for p in patterns_30])

net30 = Hopfield()
net30.train_hebb(patterns_30)

ok = sum(np.array_equal(net30.recall(p), p) for p in patterns_30)
print("czyste odtworzone:", ok, "/ 30")

for level, data in ((0.1, noisy30_10), (0.2, noisy30_20), (0.3, noisy30_30)):
    errs = [np.sum(net30.recall(x) != p) for x, p in zip(data, patterns_30)]
    print(level, "średnio błędnych pikseli:", np.mean(errs))
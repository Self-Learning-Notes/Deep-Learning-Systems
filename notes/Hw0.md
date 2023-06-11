# Hw0

*DATE: 2022/10/14*

---

#### Q2: Loading [MNIST Data](http://yann.lecun.com/exdb/mnist/)

Write a loader that will read files of **MNIST formart** and return **numpy arrays**.

**The basic format for labels**

|Offset | Type               | Value           |   Description                   |
|-------|--------------------|-----------------|---------------------------------|
|0000   |4 byte integer      |0x00000801(2049) |magic number (MSB first)         |
|0004   |4 byte integer      |10000 or 60000   |number of items (test or train)  |
|0008   |unsigned byte       |??               |label                            |
|0009   |unsigned byte       |??               |label                            |
|...    |...                 |...              |...                              |
|xxxx   |unsigned byte       |??               |

**The basic format for images**

|Offset | Type               | Value           |   Description                   |
|-------|--------------------|-----------------|---------------------------------|
|0000   |4 byte integer      |0x00000801(2051) |magic number (MSB first)         |
|0004   |4 byte integer      |10000 or 60000   |number of images (test or train) |
|0008   |4 byte integer      |28               |number of rows                   |
|0012   |4 byte integer      |28               |number of columns                |
|0016   |unsigned byte       |??               |pixel intensity (0-255)          |
|0017   |unsigned byte       |??               |pixel intensity (0-255)          |
|...    |...                 |...              |...                              |
|xxxx   |unsigned byte       |??               |pixel intensity (0-255)          |

*Modules* used: `struct`, `gzip` and `numpy`

---

#### Q3: Softmax Regression

For a single training input $x \in \mathbb{R}^{n\times 1}$, matrix $\mathbf{\theta} \in \mathbb{R}^{k\times n}$ 

where $n = \textbf{input_dim}$ and $k=\textbf{num_labels}$  

- **Linear hypothesis function**: $h_\theta(x) = \theta^{T}x$
- **Softmax function**: $z\equiv \text{normalize} (exp(h))$

$$
z_i= p(label=i)= \frac{\exp \left(h_i(x)\right)}{\sum_{j=1}^k \exp \left(h_j(x)\right)}
$$

- The loss function: **cross-entropy loss**

$$
\ell_{ce}(h(x), y) = -\log p(label=y) = -h_y(x) + \log \sum_{j=1}^{k} exp(h_j(x))
$$

**Matrix batch notation** version $X \in \mathbb{R}^{m \times n}$

- X - **design matrix**, $m= \textbf{num_examples}$,  $n = \textbf{input_dim}$ and $k=\textbf{num_labels}$ 

$$
X \in \mathbb R^{m \times n} =\left[\begin{array}{c}
-x^{(1)^T}- \\
\vdots \\
-x^{(m)^T}-
\end{array}\right], \quad y \in\{1, \ldots, k\}^m=\left[\begin{array}{c}
y^{(1)} \\
\vdots \\
y^{(m)}
\end{array}\right]
$$

- **Linear hypothesis** applied to this batch $h(X)\in \mathbb{R}^{m\times k}$ 

$$
h(X)=\left[\begin{array}{c}
-h_\theta\left(x^{(1)}\right)^T-\\
\vdots \\
-h_\theta\left(x^m\right)^T-
\end{array}\right]=\left[\begin{array}{c}
-x^{(1) T} \theta- \\
-x^{(2) T} \theta- \\
\vdots \\
-x^{(m)} \theta-
\end{array}\right]=X \theta
$$

- **Average softmax loss** over the dataset

  ```python
  log_sum_exp = np.log(np.sum(np.exp(h(X)), axis=1)) # size: (m,1)
  h_y = h(X) * y_one_hot # element-wise mul size:(m,1)
  average_loss = np.mean(log_sum_exp - h_y)
  ```


---

#### Q4: SGD for softmax regression
**Optimization** for softmax regression
$$
\min_{\theta} \frac{1}{m} \sum_{i=1}^{m} \ell_{ce}(\theta^{T} x^{(i)}, y^{(i)})
$$

**Gradient Descent:** 

For a matrix-input, scalar-output function $\mathcal{f}: \mathbb{R}^{n\times k} \rightarrow \mathbb{R}$, the gradient is defined as the matrix of **partial derivatives**.
$$
\nabla_\theta f(\theta) \in \mathbb{R}^{n \times k}
$$

where $(\nabla_\theta f(\theta))_{ij} = \frac{\partial f(\theta)}{\partial \theta_{ij}}$

The gradient descent algorithm proceeds by iteratively taking steps in the direction of the <u>negative gradient</u>

$$
\theta \leftarrow \theta-\alpha \nabla_{\theta}f(\theta)
$$
Where $\alpha$ is a step size or learning rate.

**Minibatch Stochastic Gradient Descent**                                                                                 

Repeat until loss converges
    Sample a **minibatch ** of data $X \in \mathbb{R}^{B \times n}, y \in\{1, \ldots, k\}^B$ 
    Update parameters $\theta:=\theta-\frac{\alpha}{B}\sum_{i=1}^{B} \nabla_\theta \ell\left(h_\theta\left(x^{(i)}\right), y^{(i)}\right)$



**Calculating the gradient of the softmax objective ("Cheating way")**

1. deriving the gradient of the softmax loss itself: for vector $h \in \mathbb{R}^k$

$$
\begin{aligned}
\frac{\partial \ell_{c e}(h, y)}{\partial h_i} &=\frac{\partial}{\partial h_i}\left(-h_y+\log \sum_{j=1}^k \exp h_j\right) \\
&=-1\{i=y\}+\frac{\exp h_i}{\sum_{j=1}^k \exp h_j}
\end{aligned}
$$
​		So, in vector form: $\nabla_h \ell_{c e}(h, y)=z-e_y$, where $z=\operatorname{normalize}(\exp (\mathrm{h}))$

2. deriving the gradient with respect to $\theta$  with chain rule

$$
\begin{aligned}
\frac{\partial}{\partial \theta} \ell_{c e}\left(\theta^T x, y\right) &=\frac{\partial \ell_{c e}\left(\theta^T x, y\right)}{\partial \theta^T x} \frac{\partial \theta^T x}{\partial \theta} \\
&=\left(z-e_y\right)(x), \quad\left(\text { where } z=\text { normalize }\left(\exp \left(\theta^T x\right)\right)\right)
\end{aligned}
$$

3. So to make the dimensions work... $\rightarrow$ outer product

$$
\nabla_\theta \ell_{c e}\left(\theta^T x, y\right) \in \mathbb{R}^{n \times k}=x\left(z-e_y\right)^T
$$
**"Matrix batch" form **
$$
\nabla_\theta \ell_{c e}(X \theta, y) \in \mathbb{R}^{n \times k}=\frac{1}{m} X^T\left(Z-I_y\right)
$$
​		  $X \in \mathbb{R}^{m \times n} \text{ is the design matrix}$ 

​		  $Z \in \mathbb{R}^{m \times k} \equiv \text { normalize }(\exp (X \theta)) \quad \text{(normalization applied row-wise)}$ 
 		 $I_y \in \mathbb{R}^{m \times k} \text{ represents a concatenation of one-hot bases for the labels in $y$.}$ 

---

#### Q5: SGD for a two-layer neural network

![image-20221016221244255](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221016221244255.png)Generic form of a 𝐿-layer neural network
$$
\begin{aligned}
&Z_{i+1}=\sigma_i\left(Z_i W_i\right), i=1, \ldots, L \\
&Z_1=X \\
&h_\theta(X)=Z_{L+1} \\
&{\left[Z_i \in \mathbb{R}^{m \times n_i}, W_i \in \mathbb{R}^{n_i \times n_{i+1}}\right]} \\
& \sigma_i: \mathbb{R} \rightarrow \mathbb{R} \text { nonlinearities applied elementwise}\\
& \theta=\left\{W_1, \ldots, W_L\right\} \text{: parameters}
\end{aligned}
$$

**Forward pass**
$$
Z_{i+1}=\sigma_i\left(Z_i W_i\right), \quad i=1, \ldots, L
$$

Then define $G_{i+1}=\frac{\partial \ell\left(Z_{L+1}, y\right)}{\partial Z_{i+1}}$ by derivation with <u>chain rule</u>: 
$$
\frac{\partial \ell\left(Z_{L+1}, y\right)}{\partial W_i}=\underbrace{\frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_{L-1}}{\partial Z_{L-2}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} }_{G_{i+1}=\frac{\partial \ell\left(Z_{L+1}, y\right)}{\partial Z_{i+1}}}\cdot \frac{\partial Z_{i+1}}{\partial W_i}
$$

**Backward pass**

Iteration to compute the ${G_i}^{'}s$
$$
G_i=G_{i+1} \cdot \frac{\partial Z_{i+1}}{\partial Z_i}=G_{i+1} \cdot \frac{\partial \sigma_i\left(Z_i W_i\right)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial Z_i}=G_{i+1} \cdot \sigma^{\prime}\left(Z_i W_i\right) \cdot W_i
$$
Computing the **real gradients** considering matrix sizes  
$$
G_i=\frac{\partial \ell\left(Z_{L+1}, y\right)}{\partial Z_i}=\nabla_{Z_i} \ell\left(Z_{L+1}, y\right) \in \mathbb{R}^{m \times n_i}
$$
so with "real" matrix operations
$$
G_i=G_{i+1} \cdot \sigma^{\prime}\left(Z_i W_i\right) \cdot W_i=\left(G_{i+1} \circ \sigma^{\prime}\left(Z_i W_i\right)\right) W_i^T
$$
Where $\circ$ denotes the elementwise multiplication.

**Actual parameter gradients**

Similar formula for  $\nabla_{W_i} \ell\left(Z_{L+1}, y\right) \in \mathbb{R}^{n_i \times n_{i+1}}$
$$
\begin{aligned}

&\frac{\partial \ell\left(Z_{L+1}, y\right)}{\partial W_i}=G_{i+1} \cdot \frac{\partial \sigma_i\left(Z_i W_i\right)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial W_i}=G_{i+1} \cdot \sigma^{\prime}\left(Z_i W_i\right) \cdot Z_i \\

&\Longrightarrow \nabla_{W_i} \ell\left(Z_{L+1}, y\right)=Z_i^T\left(G_{i+1} \circ \sigma^{\prime}\left(Z_i W_i\right)\right)

\end{aligned}
$$

**Summary**

Putting it all together, we can efficiently compute all the gradients we need for a neural network by following the procedure below

$\left.\begin{array}{l}\text { 1. Initialize: } Z_1=X\\ \text { Iterate: }  Z_{i+1}=\sigma_i\left(Z_i W_i\right), \quad i=1, \ldots, L-1 \text { (no $\sigma$ for L+1 } )\end{array} \right] $ Forward pass

$\left.\begin{array}{l}\text { 2. Initialize: } G_{L+1}=\nabla_{Z_{L+1}} \ell\left(Z_{L+1}, y\right)=S-I_y \\ \text { Iterate: } G_i=\left(G_{i+1} \circ \sigma_i^{\prime}\left(Z_i W_i\right)\right) W_i^T, \quad i=L, \ldots, 1\end{array}\right]$ Backward pass

And we can compute all the needed gradients along the way
$$
\nabla_{W_i} \ell\left(Z_{k+1}, y\right)=Z_i^T\left(G_{i+1} \circ \sigma_i^{\prime}\left(Z_i W_i\right)\right)
$$



**Simple two-layer neural network case**

Specifically, for input $x \in \mathbb{R}^n$, we'll consider a two-layer neural network (or one hidden layer)
$$
z = W_2^T \mathrm{ReLU}(W_1^T x)
$$
where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), and where $z \in \mathbb{R}^k$ represents the logits output by the network.  

- We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem

$$
\min_{W_1, W_2} \;\; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(W_2^T \mathrm{ReLU}(W_1^T x^{(i)}), y^{(i)}).
$$
- Batch form with matrix $X \in \mathbb{R}^{m \times n}$, this can also be written 

$$
\min_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y)
$$
- Using the chain rule, we can derive the backpropagation updates for this network

$$
\begin{aligned}

& Z_1 \in \mathbb{R}^{m \times n} = X\\

& Z_2 \in \mathbb{R}^{m \times d}  = \mathrm{ReLU}(Z_1 W_1) \\

& G_2 \in \mathbb{R}^{m \times d}  = \mathrm{1}\{Z_2 > 0\} \circ (G_3 W_2^T) \\

& G_3 \in \mathbb{R}^{m \times k}  = \text{normalize}(\exp(Z_2 W_2)) - I_y 

\end{aligned}
$$
​		where $\mathrm{1}\{Z_1 > 0\}$ is a binary matrix with entries equal to zero or one depending on 		whether each term in $Z_1$ is strictly positive 

- Gradients of the objective are given by

$$
\begin{split}

\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2  \\

\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_2^T G_3.  \\

\end{split}
$$





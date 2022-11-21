# Lecture 9

*DATE: 2022/11/ 10*
---

## **Normalization**

**Why should you normalize inputs in a neural network**

TL;DR Reduce training time and difficulty 

**Consider the matrix form of forward updates**
$$
\hat{Z}_{i+1} = \sigma_i(Z_iW_i+b_i^{T})
$$
Assumption: the input to this layer is a 2D tensor, with batches in the first dimension and features on the second.

![image-20221114224357238](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221114224357238.png)

**Layer normalization**

- $$
  \begin{equation}
  
  y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b
  
  \end{equation}
  $$

  where $\textbf{E}[x]$ denotes the empirical mean of the inputs, $\textbf{Var}[x]$ denotes their empirical variance (not that here we are using the "unbiased" estimate of the variance, i.e., dividing by $N$ rather than by $N-1$), and $w$ and $b$ denote learnable scalar weights and biases respectively. 

**Batch normalization**

- *Internal covariate shift*: distribution of each layer’s inputs changes during training, which slows down the training. 

- allows much higher **learning rates** and be less careful about **initialization**. Also acts as a regularizer, eliminating the need for dropout.

- <u>Oddity about BN</u>: introduce a dependency between all the examples in the batch.

- <u>Solution</u>: The function also computes a **running estimates** of mean/variance for all features at each layer $\hat{\mu}_{i+1}, \hat{\sigma}^2_{i+1}$ instead of batch statistics,  and at test time normalizes by these quantities:
  $$
  \begin{aligned}
  &\left.(z_{i+1}\right)_j=\frac{\left(\hat{z}_{i+1}\right)_j-\left(\hat{\mu}_{i+1}\right)_j}{\left(\left(\hat{\sigma}_{i+1}^2\right)_j+\epsilon\right)^{1 / 2}}\\
  
  & \hat{\mu}_{i+1} := (1-m) \hat{\mu}_{i} + m E[\hat{z}_{i+1}] \\
  
  & \text{where $m$ is momentum}.
  
  \end{aligned}
  $$

**Regularization**

- Why we use regularization in deep networks?
  - Typically deep networks are *overparameterized* models: contain more parameters than the number of training examples. This means that they are capable of fitting the training data exactly
  - This should imply that the models will *overfit* the training set, and not *generalize* well

- The process of “limiting the complexity of the function class” in order to ensure that networks will generalize better to new data.

  - *Implicit* regularization:  refers to the manner in which our existing algorithms (namely SGD) or architectures already limit functions considered
  - *Explicit* regularization:  modifications made to the network and training procedure explicitly intended to regularize the network 


- **$\ell_2$ regularization(weight decay)**: the magnitude of a model’s parameters are often a reasonable proxy for complexity, so we can minimize loss while also keeping parameters small.
  
  -     At each iteration we shrink the weights by a factor $(1-\alpha \lambda)$ before taking the gradient step.
    $$
    \underset{W_{1: D}}{\operatorname{minimize}} \frac{1}{m} \sum_{i=1}^m \ell\left(h_{W_{1: D}}\left(x^{(i)}\right), y^{(i)}\right)+\frac{\lambda}{2} \sum_{i=1}^D\left\|W_i\right\|_2^2
    $$
    Results in the gradient descent updates:
    $$
    W_i:=W_i-\alpha \nabla_{W_i} \ell(h(X), y)-\alpha \lambda W_i=(1-\alpha \lambda) W_i-\alpha \nabla_{W_i} \ell(h(X), y)
    $$
  
- **Dropout**: randomly set some fraction of the activations at each layer to zero.

$$
\begin{aligned}

& \hat{z}_{i+1} = \sigma_i (W_i^T z_i + b_i) \\

& (z_{i+1})_j = 
  \begin{cases}
  (\hat{z}_{i+1})_j /(1-p) & \text{with probability } 1-p \\

  0 & \text{with probability } p \\
  \end{cases}

\end{aligned}
$$
​     Dropout as stochastic approximation
$$
\begin{aligned}
\quad \frac{1}{m} \sum_{i=1}^m \ell\left(h\left(x^{(i)}\right), y^{(i)}\right) & \Longrightarrow \frac{1}{|B|} \sum_{i \in B} \ell\left(h\left(x^{(i)}\right), y^{(i)}\right) \\
z_{i+1}=\sigma_i\left(\sum_{j=1}^n W_{:, j}\left(z_i\right)_j\right) & \Longrightarrow z_{i+1}=\sigma_i\left(\frac{n}{|\mathcal{P}|} \sum_{j \in \mathcal{P}}^n W_{:, j}\left(z_i\right)_j\right)
\end{aligned}
$$

**Interaction: many design choices to ease optimization ability of deep networks**

- Optimizer learning rate/ momentum
- Weight initialization
- Normalization layer
- Regularization

Summary:

In many cases, it seems to be possible to get similarly good results with wildly different architectural and methodological choices.(different tips, training techniques and different variants).


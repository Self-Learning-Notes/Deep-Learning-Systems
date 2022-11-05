# Automatic Differentiation

*DATE: 2022/10/ 17*

---

Recall the three gradients of a deep learning system:

- Hypothesis class $h_\theta(x)$
- Loss function $\ell(h_\theta(x), y)$
- Optimization method $\theta:= \frac{\alpha}{B}\sum_{i=1}^{B}\nabla_{\theta}\ell(h_\theta(x^{(i)}), y^{(i)})$

This section focus on  **loss function gradient** w.r.t hypothesis class parameters.

#### Numerical differentiation

- By definition

$$
\frac{f(\boldsymbol{\theta})}{\partial \theta_i} = \lim_{\epsilon \rightarrow 0}\frac{f(\boldsymbol{\theta} + \epsilon\boldsymbol{e}_i) - f(\boldsymbol{\theta})}{\epsilon} \tag{1}
$$

​			where $\boldsymbol{e}_{i}$ has $i^{th}$ entry = 1, else = 0.  

- A more numerically accurate way

$$
\frac{\partial f(\boldsymbol{\theta})}{\partial \theta_i} = \frac{f(\boldsymbol{\theta}+\epsilon\boldsymbol{e}_i) - f(\boldsymbol{\theta}-\epsilon\boldsymbol{e}_i)}{2\epsilon} + o(\epsilon^2) \tag{2}
$$

​		Reason: $f(\theta+\delta) = f(\theta) + f'(\theta)\delta + \frac{1}{2}f''(\theta)\delta^2 + o(\delta^2)$, therefore Eq.(1) has $o(\epsilon)$ error. 

- Suffer from numerical error ($\epsilon$ has to be small) and less efficient to compute (requires two forward computation).

- However, a powerful tool to **check** in unit test cases. If $\nabla_{\boldsymbol{\theta}}f(\boldsymbol{\theta})$ is correct:
  $$
  \boldsymbol{\delta}^{\mathsf{T}}\nabla_{\boldsymbol{\theta}}f(\boldsymbol{\theta}) = \frac{f(\boldsymbol{\theta}+\epsilon\boldsymbol{\delta}) - f(\boldsymbol{\theta}-\epsilon\boldsymbol{\delta})}{2\epsilon} + o(\epsilon^2)
  $$
  where $\delta$ is picked from unit ball

#### Computational graph

$$
f = (x_1, x_2) = \ln(x_1) + x_1x_2 - \sin(x_2)
$$

![image-20221019163536181](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221019163536181.png)

A form of Directed Acyclic Graph(DAG)

- Each node represents an intermediate value in the computation.

- Edges represent input and output relations.

#### Forward mode automatic differentiation

- Define $\dot{v_i} = \frac{\partial v_i}{\partial x_1}$, compute $\dot{v_i}$ in the forward topological order of the computational graph.
- Limitation: $f:\mathbb{R}^n \rightarrow \mathbb{R}^k$, efficient only when $n$ is small. While in the case of deep learning, the input dimension is usually large.

#### Reverse mode automatic differentiation

- Define **adjoint**  $\bar{v_i} = \frac{\partial y}{\partial v_i}$, iteratively in the reverse topological order of the computational graph. 

- Derivation for the <u>multiple pathway case</u>: define **partial adjoint** $\overline{v_{i\rightarrow j}} = \bar{v_j}\frac{\partial v_j}{\partial v_i}$, then we have
  $$
  \overline{v_i} = \sum_{j \in {\rm next}(i)}\overline{v_{i \rightarrow j}}
  $$

- Reverse AD algorithm

  ```python
  def gradient(out):
      # Dictionary that records a list of partial adjoints of each node
      node_to_grad = {out: [1]}
      
      for i in reverse_topo_order(out):
          # Sum up partial adjoints to compute adjoint vi
          adjoint_i = sum(partial_adjoint_i_j) = sum(node_to_grad[i])
          for k in inputs(i):
              partial_adjoint_k_i = adjoin_i * partial_grad(v_i, v_k)
              # “Propagates” partial adjoint to its input
              node_to_grad[k].append(partial_adjoint_k_i)
      return adjoint_input
  ```

- Reverse AD  by extending computational graph

  An example: $y = exp(v_1) \cdot (exp(v_1)+1) $, extend the left computation graph to the right.

  Outer loop $i = 4,3,2,1:$ kind of saving all of <u>the additional exponential computations</u>.

  ![image-20221019205442100](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221019205442100.png)![image-20221019205512734](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221019205512734.png)

- Reverse mode AD vs Backprop

  - Backprop used in first generation deep learning frameworks.

  - Reverse mode AD

    - construct separate graph nodes for adjoints.
    - First used in **Theano (MILA)**, adopted by modern deep learning frameworks.

  - Why Reverse mode AD?

    If we are interested in handling gradient of gradient, only need to extend that graph further by composing more operations and run reverse mode AD again on the gradient

- Reverse mode AD on Tensors

  ![image-20221019233240247](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221019233240247.png)

  - **Matrix size**: $X - m\times n$; $W - n\times p$; $Z - m\times p$ 
  
  - **Define adjoint** for tensor values
    $$
    \bar{\boldsymbol{Z}} = \left[ \begin{matrix} \frac{\partial y}{\partial Z_{1,1}} & \cdots & \frac{\partial y}{\partial Z_{1,n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial y}{\partial Z_{m,1}} & \cdots & \frac{\partial y}{\partial Z_{m,n}} \end{matrix} \right]
    $$
    
  - ![image-20221019234128120](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221019234128120.png)
  
  
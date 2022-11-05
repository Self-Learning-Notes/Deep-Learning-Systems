# Hw1 

*DATE: 2022/10/ 27*

---

#### **Q1: Implementing forward computation**

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```

Consider the `EWiseAdd` operator in the `ops.py` as example

- The `compute()` function computes the "forward" pass(the operation itself).
- Arguments are both `NDArray` instead of `Tensor` objects within the automatic differentiation.

---

#### **Q2:  Implementing backward computation**

- Trick: size matching
  
  - The size of `out_grad` will always be the size of the *output* of the operation.
  
  - The sizes of the `Tensor` objects *returned* by `gradient()` have to always be the same as the original *inputs* to the operator.
  
- `Element-wise` operation on $\mathbf{w}$ and $\mathbf{x}$ mathematically
  $$
  \begin{aligned}
  & \text { Op Partial with respect to $\mathbf{w}$ } \\
   & + \quad  \frac{\partial(\mathbf{w}+\mathbf{x})}{\partial \mathbf{w}}=\operatorname{diag}\left(\ldots \frac{\partial\left(w_i+x_i\right)}{\partial w_i} \ldots\right)=\operatorname{diag}(\overrightarrow{1})=I
  \\
  &-\quad \frac{\partial(\mathbf{w}-\mathbf{x})}{\partial \mathbf{w}}=\operatorname{diag}\left(\ldots \frac{\partial\left(w_i-x_i\right)}{\partial w_i} \ldots\right)=\operatorname{diag}(\overrightarrow{1})=I\\
  &\otimes \quad \frac{\partial(\mathbf{w} \otimes \mathbf{x})}{\partial \mathbf{w}}=\operatorname{diag}\left(\ldots \frac{\partial\left(w_i \times x_i\right)}{\partial w_i} \ldots\right)=\operatorname{diag}(\mathbf{x})\\
  &\oslash \quad \frac{\partial(\mathbf{w} \oslash \mathbf{x})}{\partial \mathbf{w}}=\operatorname{diag}\left(\ldots \frac{\partial\left(w_i / x_i\right)}{\partial w_i} \ldots\right)=\operatorname{diag}\left(\ldots \frac{1}{x_i} \ldots\right)\\
  &\text { Op Partial with respect to } \mathbf{x}\\
  &+ \quad \frac{\partial(\mathbf{w}+\mathbf{x})}{\partial \mathbf{x}}=I\\
  &- \quad \frac{\partial(\mathbf{w}-\mathbf{x})}{\partial \mathbf{x}}=\operatorname{diag}\left(\ldots \frac{\partial\left(w_i-x_i\right)}{\partial x_i} \ldots\right)=\operatorname{diag}(-\overrightarrow{1})=-I\\
  &\otimes \quad \frac{\partial(\mathbf{w} \otimes \mathbf{x})}{\partial \mathbf{x}}=\operatorname{diag}(\mathbf{w})\\
  &\oslash \quad \frac{\partial(\mathbf{w} \oslash \mathbf{x})}{\partial \mathbf{x}}=\operatorname{diag}\left(\ldots \frac{-w_i}{x_i^2} \ldots\right)
  \end{aligned}
  $$

- `MatMul` 

  - For normal matrix multiplication $Z = XW$, we have

    $$\bar{X} = \bar{Z}W^{T} \quad \bar{W}= X^{T} \bar{Z}$$

  - In `test_matmul_batched_backward()` of `test_autograd_hw.py`  

    Test Case: input `Tensors` A with shape of (6,6,5,4) and B (4,3) 

    - The input matrix B of shape (4,3) first broadcasts to (6,6,4,3). 
    - (6,6,5,4) @ (6,6,4,3) -> The output matrix of shape(6,6,5,3)
    - `transpose` of A (6,6,5,4) @ `out_grad` (6,6,5,3) -> Derivative w.r.t. B (6,6,4,3)
    - (6,6,4,3) sum at axis = (0,1) ->  (4, 3)

- `BroadcastTo`

  - **Example**: $f(\mathbf{b}) = \mathbf{b} \mathbf{M}$, where $\mathbf{M} = [1,1,\dotsc,1]$ is $1 \times N$ matrix and $\mathbf{b}$ is $N \times 1$ matrix. This example broadcasts $\mathbf{b}$ from $(N,1)$ to $(N,N)$.
  -  $\frac{\partial \ell}{\partial \mathbf{b}}=\frac{\partial \ell}{\partial \mathbf{f}}\mathbf{M}^{T} $ , this is actually **summation of the upstream gradient** - we need to sum `` back to size $(N,1)$ in the original broadcast direction.

- `Summation`
  
  - Kind of "Duality" with `BroadcastTo`
  - Find the axes of input which are summed at, **Broadcast  upstream gradient** at the summed axes 
  

---

#### **Q3:Topological Sort **

- Recall graph $G=(V,E)$
  - $V = \text{set of vertice}$ 
  - $E= \text{set of edges}$
    - $\text{Directed e=(v,w) ordered pairs} $ 

- We actually follow the DFS [tree edge](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/recitation-14-depth-first-search-dfs/)

- Postorder (Left, Right, Root)

- Test case: $c_1 = 3\times a_1 \times a_1 + 4 \times b_1 \times a_1 - a_1$

  ![image-20221030113302228](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221030113302228.png)

---

#### **Q4: Reverse mode automatic differentiation**

![image](https://global.discourse-cdn.com/standard10/uploads/dlsyscourse/optimized/1X/e7b81cd3919e5026c3a36169f6b919e60b618a97_2_690x164.png)

<img src="C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221030174338005.png" alt="image-20221030174338005" style="zoom:50%;" />



- Nodes $a$ and $b$ are the leaf nodes, therefore no inputs to them

- Reversed topological order: $f,d,e,c,b,a$

- `out_grad`: the gradient of loss w.r.t node $f$

  ```python
  node_to_output_grads_list[output_tensor] = [out_grad]
  ```

- For each node in the reverse_topo_order

  - Sum up the all partial joints
  - Compute "downstream partial joints" by `op.gradient_as_tuple`.
  - Append each partial joint to the corresponding `node_to_output_grads_list[input_node]`


---

#### Q5&6: Re-implement the simple two-layer neural network 

**Softmax loss** with needle

```python
# Use needle operation
log_sum_exp = ndl.ops.log(ndl.ops.summation(ndl.ops.exp(Z), axes=1))
Z_y = ndl.ops.summation(Z * y_one_hot, axes =1) 
```

**Neural network computation graph**

```python
loss = softmax_loss(Z2, Iy)
loss = loss.backward()
# update parameters directly with its gradient
W1 = (W1 - lr * W1.grad).detach()
W2 = (W2 - lr * W2.grad).detach()
```


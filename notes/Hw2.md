# Hw 2

*Date: 2022/11/04*

---

Q1: Weight initialization

- Xaiver uniform/normal
- Kaiming uniform/normal

Q2： Additional modules

`LogSumExp` $(z) = \log (\sum_{i} \exp (z_i - \max{z})) + \max{z}$

- Understand axis in numpy array

  ```python
  z  = np.array([
          [[3.35, 3.25, 2.8 ],
          [2.3 , 3.75, 3.75],
          [3.35, 2.45, 2.1 ]],
  
         [[1.65, 0.15, 4.15],
          [2.8 , 2.1 , 0.5 ],
          [2.6 , 2.25, 3.25]],
  
         [[2.4 , 4.55, 4.75],
          [0.75, 3.85, 0.05],
          [4.7 , 1.7 , 4.7 ]]], dtype="float32")
  
  max_z0 = np.max(z, axis=0)
  max_z1 = np.max(z, axis=1)
  max_z2 = np.max(z, axis=2)
  # result of max_z0
  array([[3.35, 4.55, 4.75],
         [2.8 , 3.85, 3.75],
         [4.7 , 2.45, 4.7 ]], dtype=float32)
  # result of max_z1
  array([[3.35, 3.75, 3.75],
         [2.8 , 2.25, 4.15],
         [4.7 , 4.55, 4.75]], dtype=float32)
  # result of max_z2
  array([[3.35, 3.75, 3.35],
         [4.15, 2.8 , 3.25],
         [4.75, 3.85, 4.7 ]], dtype=float32)
  ```
- gradient
  $$
  \begin{aligned}
  & \frac{\part{f}}{\part{z_i}} = \frac{exp(z_i-maxz)}{\sum_i (z_i-maxz)} \\
  & \frac{\part{f}}{\part{z}} = \text{softmax}(z-maxz)
  \end{aligned}
  $$

return $\text{out_grad element-wise multiply with softmax(z-max)}$, we need to broadcast and reshape both

`out_grad` and $\sum_i z_i-maxz$ to the input_shape.



**SoftmaxLoss**(Module)

Rewrite hw1 simple_ml.py `softmax_loss`


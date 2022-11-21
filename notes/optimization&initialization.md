# Lecture 6

*DATE: 2022/11/ 01*

---

#### Fully connected networks

A **multi-layer perceptron (MLP)** with an explicit bias term is defined by the iteration.

$$
\begin{aligned}
&Z_{i+1}=\sigma_i\left(Z_i W_i + \textcolor{red}{b_i^{T}}\right), i=1, \ldots, L \\
&Z_1 \equiv X \\
&h_\theta(X) \equiv Z_{L+1} \\
& \\
&{Z_i \in \mathbb{R}^{m \times n_i}, W_i \in \mathbb{R}^{n_i \times n_{i+1}}}, \textcolor{red}{b_i} \in \mathbb{R}^{n_{i+1} \times1}\\
& \sigma_i: \text{nonlinear activation with } \sigma_L = I\\
& \theta=\left\{W_1, \ldots, W_L\right\} \text{: parameters}
\end{aligned}
$$
In practice,  we perform operation via *broadcasting*

**Key questions** of how-to

- width and depth of the network
- optimize the objective
-  initialize the weights of the network

#### Optimization

The goal is to $\min_{\theta} \frac{1}{m} \sum_{i=1}^{m} \ell(\theta^{T} x^{(i)}, y^{(i)})$ or simply $\min_{\theta} f(\theta)$

- **Gradient descent**
  $$
  \theta_{t+1}=\theta_t-\alpha \nabla_\theta f\left(\theta_t\right)
  $$
   $t: \text{iteration}$ , $\alpha>0 \text{ step size}$ 

  $\nabla_\theta f\left(\theta_t\right)$ is gradient evaluated at the parameters $\theta_t$

  Illustration of gradient descent with different step sizes:

  ![image-20221101162444510](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101162444510.png)

- **Newton's method**: scales gradient according to inverse of the Hessian
  $$
  \theta_{t+1}=\theta_t-\alpha\left(\nabla_\theta^2 f\left(\theta_t\right)\right)^{-1} \nabla_\theta f\left(\theta_t\right)
  $$
  Full step given by $\alpha=1$, otherwise called a damped Newton method

   ![image-20221101163415382](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101163415382.png)

- **Momentum**: moving average of multiple *previous gradients* 
  $$
  \begin{aligned}
  
  &u_{t+1}=\beta u_{t} + (1-\beta) \nabla_\theta f\left(\theta_t\right) \\
  &= (1-\beta)\nabla_\theta f\left(\theta_t\right) + \beta(1-\beta)\nabla_\theta f\left(\theta_{t-1}\right) + \beta^{2}(1-\beta)\nabla_\theta f\left(\theta_{t-2}\right) + \dotsc
  \\
  
  &\theta_{t+1}=\theta_t-\alpha u_{t+1}
  
  \end{aligned}
  $$
  ![image-20221101190244099](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101190244099.png)

  **unbiasing** momentum terms:
  $$
  \theta_{t+1}=\theta_t-\frac{\alpha u_{t+1}}{1-\beta^{t+1}}
  $$
  ![image-20221101192232601](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101192232601.png)

  **Nesterov Momentum**: evaluate the gradient at the updated point
  $$
  \begin{aligned}
  &u_{t+1}=\beta u_t+(1-\beta) \nabla_\theta f\left(\theta_t\right) \\
  &\theta_{t+1}=\theta_t-\alpha u_{t+1}
  \end{aligned} \Longrightarrow \begin{aligned}
  &u_{t+1}=\beta u_t+(1-\beta) \nabla_\theta f\left(\theta_t- \textcolor{red}{\alpha u_t}\right) \\
  &\theta_{t+1}=\theta_t-\alpha u_{t+1}
  \end{aligned}
  $$
  ![image-20221101192251757](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101192251757.png)

- **Adam**: adaptive gradient methods attempt to estimate this scale over iterations
  and then re-scale the gradient update accordingly
  $$
  \begin{aligned}
  
  u_{t+1} &=\beta_1 u_t+\left(1-\beta_1\right) \nabla_\theta f\left(\theta_t\right) \\
  
  v_{t+1} &=\beta_2 v_t+\left(1-\beta_2\right)\left(\nabla_\theta f\left(\theta_t\right)\right)^2 \\
  
  \theta_{t+1} &=\theta_t- \frac{\alpha u_{t+1}}{\left(v_{t+1}^{1 / 2}+\epsilon\right)}
  
  \end{aligned}
  $$
  ![image-20221101194314557](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221101194314557.png)

**Takeaways**:

- experiment to see how different methods affect deep networks of different types
- All the optimization methods presented **only** used in their stochastic form

#### Weight initialization

Standard method in convex optimization: initializing parameters to zero

- Choice of initialization matters(initialize weights with $W_i \sim N(0, \sigma^2I)$)

  - The norm of the forward activations $Z_i$

  - The norm of the gradients $\nabla_{W_i} \ell(h_{\theta}(X),y)$

The variance of Gaussian random variables matter a lot. A deep network with poorly-chosen weights will never train. The graph below shows a 50-layer deep network applying to MNIST (activation norm is $||Z_i||_{2}$ and gradient norm is $||\nabla_{W_i} \ell||_{2}$).

![image-20221110231656256](C:\Users\Steve\AppData\Roaming\Typora\typora-user-images\image-20221110231656256.png)

**Insights** about choice of $\sigma^2$ from the above figure: $W_i \sim N(0,\frac{c}{n})$, where $c=2$



Reference: 

- Xavier
- residual neural network
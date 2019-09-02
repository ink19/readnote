### Contextual Bandits
Context vector $x$

动作 $a$

在对应的$x$，动作$a$产生的收益$r(x, a)$

代理需要最大化收益，但是不知道收益函数。

为了平衡探索收益函数和最大化总收益，将算法分为了两个部分。

### 优化对象
$$
\begin{aligned}
    min_{a_1, ..., a_T} &\quad \frac{1}{T} \sum_{t-1}^T{FIM_t} \\
    s.t. &\quad \frac{1}{T} \sum_{t=1}^T \frac{PT_t - v_0}{v_0} \ge d
\end{aligned}
$$

### 策略网络

输入：$x$

输出：摇臂选择策略$\pi(a|x)$

神经网络定义为$\pi(a|x;\theta)$


对于任意的随机变量$f(a)$，任何两个临近的策略$\pi(a|x, \theta_1)$和$\pi(a|x;\theta_2)$满足
$$
\mathbb{E}_{\pi(\theta_2)}[f(a)] - \mathbb{E}_{\pi(\theta_1)}[f(a)] \approx g_f^T(\theta_1)*(\theta_2 - \theta_1)\\
g_f(\theta_1) = \mathbb{E}_q[\frac{\nabla_\theta \pi(a|x;\theta_1)}{q(a|x)} f(a)]
$$

其中$q(a|x)$可以为任意的策略。

$$
\mathbb{E}_\pi(\theta_1)[f(a)] = \mathbb{E}_q[q^{-1}(a|x) \pi(a|x;\theta_1)f(a)]
$$


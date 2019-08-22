### 问题
上下文集合 $\mathcal{X} = {1, 2, ..., J}$

操作集合 $\mathcal{A}={1, 2, ..., K}$

在每一轮$t$中，会有一个$X_t$出现，其出现的概率为$\mathbb{P}\{X_t = j\} = \pi_j, j\in \mathcal{X}$，而每一个动作$k \in \mathcal{A}$都会产生一个非负的激励$Y_{k,t}$。

在给予的上下文$X_t = j$之下激励，$Y_{k,t}$是一个独立随机的变量介于$[0, 1]$。对于代理来说条件期望$\mathbb{E}[Y_{k,t}|X_t=j]=u_{j,k}$是不可知的。

如果动作$k$在上下文$j$的时候发生的会，也会有一个消耗$c_{i,j} > 0$。对于一定的$k$和$j$，$c_{i,j}$也是一定的。

在每一轮的开始$X_t$是已知的。但是在采取行动后获得的奖励只会在每一轮的结束时得到。

$\Gamma$是一个映射在$\mathcal{H}_{t-1}={X_1,A_1,Y_1;X_2,A_2,Y_2; ...; X_{t-1},A_{t-1},Y_{t-1}}$的函数。算法的目标就是在给定时间$T$和预算$B$的情况下最大化总利润$U_{\Gamma}(T,B)$

$$
\begin{aligned}
maximize_{\Gamma} \qquad & U_{\Gamma}(T,B) = \mathbb{E}_{\Gamma}[\sum_{t=1}^{T}{Y_t}]
\\
subject \ \ to \qquad & \sum_{t=1}^{T}{Z_t} \le B
\end{aligned}
$$

oracle 算法得到的结果为$U^*(T,B)$。则损失函数可以表示为
$$
R_{\Gamma}(T,B)=U^*(T,B) - U_{\Gamma}(T,B)
$$

定义$\rho = B/T$

### Oracle算法（神的算法）
已知$u_{j,k}$。由于在$X_j$到达的时候，该算法就可以很正确的得到对于$X_j$的最佳操作。而最重要的问题就是确定是否进行该操作。这取决于剩余时间和剩余的预算限制。

这里将问题规划成LP问题来进行求解。
$$
\begin{aligned}
(\mathcal{LP}_{T,B}) maximize_p \qquad & \sum_{j=1}^{J}{p_j \pi_j u_j^*}
\\
subject \ \ to \qquad & \sum_{j=1}^{J}{p_j \pi_j} \le B/T, 
\\
&\mathcal{p} \in [0,1]^J
\end{aligned}
$$

定义
$$
\tilde{j}(\rho) = \max{\{j:\sum_{j'=1}^j{\pi_{j'}} \le \rho\}}
$$

当$\pi_1 > \rho$，$\tilde{j}(\rho)=0$。

我们可以得到以下最优的解决方案
$$
p_j(\rho) = \left\{\begin{aligned}

& 1, & \mathrm{if}\ 1 \le j \le \tilde{j}(p),\\
& \frac{\rho-\sum_{j'=1}^{\tilde{j}(\rho)}{\pi_{j'}}}{\pi_{\tilde{j}(\rho)+1}},\ & \mathrm{if}\ j = \tilde{j}(\rho)+1, \\
& 0, &\mathrm{if}\ j > \tilde{j}(\rho) + 1.

\end{aligned}
\right.
$$
相应的，可以得到优化的结果为
$$
v(\rho) = \sum_{j=1}^{\tilde{j}(\rho)}{\pi_j u_j^*} + p_{\tilde{j}(\rho) + 1}(\rho) \pi_{\tilde{j}(\rho) + 1} u^*_{\tilde{j}(\rho) + 1}
$$

可以得到$U^*(T,B)$的上限为$\widehat{U}(T,B) = Tv(\rho)$

ALP算法的概率为$p_{X_t}(\frac{b}{\tau})$

ALP算法只需要预期收益的排序，而不需要其准确值。

#### 定理1 
给予任意一个定值 $\rho \in (0, 1)$，ALP算法的误差可以满足：
1. （没有约束的例子）如果$\rho \not ={q_j}$对于任意的$j\in\{1,2, ..., J-1\}$，那么$R_{ALP} \le \frac{u_1^* - u_J^*}{1-e^{-2\delta^2}}$，其中$\delta = min\{\rho - q_{\tilde{j}(\rho)}, q_{\tilde{j}(\rho) + 1} - \rho\}$
2. (有约束的例子)如果存在$j\in \{1, 2, ..., J-1\}$，使得$\rho = q_j$，那么$R_{ALP}(T,B) \le \Theta^{(o)}\sqrt{T} + \frac{u_1^* - u_J^*}{1-e^{-2{\delta'}^2}}$

### UCB-ALP算法
#### UCB

使$C_{j,k}(t)$为在$t$轮下，动作$k$在$j$发生的次数。

如果$C_{j,k}(t-1) > 0$，则使$\bar{u}_{j,k}(t)$为动作$k$在$j$的一个经验，比如：均值。

当$C_{j,k}(t) > 0$定义$\hat{u}_{j,k}(t) = \bar{u}_{j,k}(t) + \sqrt{\frac{\log{t}}{2C_{j,k}(t-1)}}$，而当$C_{j,k}(t) = 0$时，$\hat{u}_{j,k}(t) = 1$。

由于论文$24$，我们在探测期使用了一个较小的探测系数。

**引理3** 对于两个对$(j,k)$和$(j', k')$，如果$u_{j,k} < u_{j',k'}$，那么对于任意时间$t \le T$
$$
\mathbb{P}\{\hat{u}_{j,k}(t) \ge \hat{u}_{j',k'}(t) | C_{j,k}(t-1) \ge l_{j,k}\} \le 2t^{-1}
$$

其中$l_{j,k} = \frac{2\log{T}}{(u_{j',k'} - u_{j,k})^2}$，证明位于13和25

#### 算法

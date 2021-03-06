### Uncertainty Quantification of Kevlar Potential Energy Surface Using Bayesian Optimization
Aramid fibers, such as Kevlar and Twaron, are of great interest due to their outstanding ratio of strength per weight and thermal resistance. There are numerous applications of these fibers that include bulletproof vests and helmets, turbine engine fragment, containment structures, cut resistant gloves, etc. Compositional and structural studies of aramid fibers suggest that these materials derive their exceptional properties from crystalline domains of poly(p-phenylene terephthalamide) (PPTA) polymers, highly oriented along the fiber axis.  

<p align="center">
  <img src="https://github.com/AnkitMish/Kevlar-BayesOpt/blob/master/images/Figure1a.png">
  <br><br>
  <b>Figure above illustrates the Kevlar aramid Fibres</b>
  <br><br>
</p>

<br>

While aramid fibers display exceptional performance, their development and improvement have been historically empirical and based on trial and error. Recent investigations on the effect of defects on the performance of fibers have been instrumental in understanding intrinsic deformations and failure mechanisms. However, there is a startling lack of shock investigations on PPTA and aramid fibers. Since the main application of these fibers is in shock and impact
protection, such studies are required for understanding the intrinsic stress release mechanisms of PPTA. Atomistic simulations are key to understanding shock damage microscopically and to enable atomistically informed continuum simulations. Recent simulations have highlighted the influence of defects, and the arrangement of chain structures, on the mechanical properties. Nevertheless, in order to understand the intrinsic ability of aramid fabrics to withstand shock loading, the response of its most basic constituent, PPTA crystals, needs to be investigated explicitly first. The shock response of the intricate bondhierarchy-stabilized PPTA structure can be modeled accurately by first-principles methods.

***
However, the initial random search showed that the potential energy surface of the system is too flat, making it challenging to find a global minimum.

<p align="center">
  <img src="https://github.com/AnkitMish/Kevlar-BayesOpt/blob/master/images/Figure4a.png">
  <br><br>
  <b>Figure above shows the energy landscape as a function polymer chains’ displacement, which shows multiple minima.</b>
  <br><br>
</p>

<br>
Furthermore, computation on each structure is extremely costly. Thus, we employ Bayesian optimization technique to efficiently find the optimal solution. 

Bayesian optimization (BO) optimizes a black box objective function which is typically expensive
to compute due to the amount of time required, monetary cost or an opportunity cost. This class of
machine learning problem is suitable for solving problems in continuous domain of maximum 20
dimensions. BO builds a surrogate model for the objective function and quantifies the uncertainty
in the surrogate using gaussian process regression.

We perform Bayesian optimization of smaller system consisting 224 atoms. Further, we only allow
the chain as a whole to move in [010] and [001] direction while preserving the hydrogen-bond.
These constraints allow us to reduce the phase space from 448 (224 atoms in [010] and [001]
direction) to 8-dimensional space. The feasible set space in the present problem is define as a set
of 8 tuple values . The polymer chains are allowed to displace {𝑦𝑖, 𝑧𝑖} 𝑤ℎ𝑒𝑟𝑒 𝑖 = 1,2,3,4 ± 0.5𝑦𝑖
, ± 0.5𝑧 in the y and z directions. In Bayesian optimization method, acquisition function selects 𝑖
the next point to be search based on trade-off between exploration and exploitation. We use
Expected improvement (EI) as acquisition function. After each iteration of BO, a new structure is
created and minimized based on the suggestion. Each computed structure is added to the dataset
and maximum EI is computed for next structure. 
<br>

<p align="center">
  <img src="https://github.com/AnkitMish/Kevlar-BayesOpt/blob/master/images/Table.png">
  <br><br>
  <b>Pseudo-code for Bayesian optimization process is shown above.</b>
  <br><br>
</p>

***

We performed 40 Bayesian optimization runs to sample the input space and found several possible
structures. The Gaussian process regression is fitted initially to ground truth values and augmented
each step to include the new data point and newest evaluated values. The model converges to
optimum in nearly 40 iterations as shown in figure S3 and successfully exploit as well as explore
the function space.

<p align="center">
  <img src="https://github.com/AnkitMish/Kevlar-BayesOpt/blob/master/images/BayesianGlobalMinima.png">
  <br><br>
  <b>Bayesian optimization to find global minimum</b>
  <br><br>
</p>






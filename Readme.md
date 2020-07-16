# Kevlar Shock Loading
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
in the surrogate using gaussian process regression




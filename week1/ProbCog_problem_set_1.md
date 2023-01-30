University of Helsinki, Master's Programme in Data Science  
DATA20047 Probabilistic Cognitive Modelling - Spring 2023  
Luigi Acerbi  

# Problem Set 1: Bayesian inference in perception

- This homework problem set focuses on **Week 1 and 2** of the course.
- This problem set is worth **20 points** in total (out of 100 for the full course).
- Check the submission deadline on Moodle!


## Submission instructions

Submission must be perfomed entirely on Moodle (**not** by email).
1. When you have completed the exercises, save the notebook.
2. Report your solutions and answers on Moodle ("*Problem set 1 answer return*").
3. Submit two files on Moodle ("*Problem set 1 notebook return*"): 
  - The notebook as `.ipynb`.
  - The same notebook downloaded as `.pdf` (there are various ways to save the file as PDF, the most general is "File" > "Print Preview" and then print the page to PDF using your browser - remember to enter the Print Preview first).

## IMPORTANT

1. Do not share your code and answers with others. Contrary to the class exercises, which you can do with others, these problems are *not* group work and must be done individually.
2. It is allowed to use snippets of code from the lecture exercises and model solutions.
3. It is your responsibility to ensure that the notebook has fully finished running all the cells, all the plots view properly etc. before submitting it. However, the notebook should be runnable from scratch if needed ("Kernel > Restart & Run All").
4. Submit your work by the deadline.
5. Unless stated otherwise, please report your numerical answers in Moodle with full numerical precision (~14-15 digits), unless the answer is an integer.
6. If you are confused, think there is a mistake or find things too difficult, please ask on Moodle.

## References

- \[**MKG22**\] Ma WJ, Körding K, and Goldreich D. "Bayesian Models of Perception and Action: An Introduction". MIT Press, 2022.
- *Acknowledgements*: Question 1.1 and 1.2 of this notebook are adapted from problems in \[**MKG22**\].


```python
# set-up -- do not change
import numpy as np
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt
np.random.seed(1)
```

# Question 1.1 (5 pts)

> This question is about performing Bayesian inference in an "everyday" scenario, with some simplifying assumptions. Related material was covered in Week 1 of the course.


You are one of 80 passengers waiting for your bag at an airport luggage carousel (see Section 2.5 of \[**MKG22**\]). We assume each passenger has one and only one bag. In general, your bag looks the same as 6% of all bags. In formulas:
$$
p(\text{looks like your bag}|\text{it is your bag}) = 1, \qquad p(\text{looks like your bag}|\text{it is not your bag}) = 0.06.
$$

Derive a general expression for the probability that the bag you are viewing (which matches your bag visually) is your own, $$p(\text{it is your bag} | \text{looks like your bag}),$$ 
as a function of the number of bags $b$ you have viewed so far (before the current one). 

- a) What is $p(\text{it is your bag} | \text{looks like your bag})$ after 40 bags have gone by, none of which was yours (that is, $b = 40$)?
- b) How many bags must you view (without finding your own) before the posterior probability $p(\text{it is your bag}|\text{looks like your bag})$ is equal or greater than 70%?

Report your results in Moodle.


```python
# code here...
(1/40)/(1/40 + 0.06 * 39 /40)
```




    0.29940119760479045



# Question 1.2 (5 pts)

> This question deals with how perception about the world is influenced by the statistics of the environment. See Chapter 2 and particularly Section 2.6 of \[**MKG22**\].


Imagine you live in a very boring world consisting of a 2 x 10 grid of squares:

```
▢▢▢▢▢▢▢▢▢▢
▢▢▢▢▢▢▢▢▢▢
```
Only two things ever happen in this world: 
- $H1$ ("vertical bar"): With a probability of 30%, a vertical bar will appear in this world, consisting of two black squares in a column, chosen so that each possible column is equally probable. 
- $H2$ ("independent dots"): With a probability of 70%, one black square will appear in a random position in the top row (uniformly chosen), and another black square will appear in a random position in the bottom row (uniformly chosen, independently from the first row). 

When doing inference, we will refer to these possibilities as Hypotheses 1 and 2 ($H1$ and $H2$), respectively.

- a) Suppose that as an observer in this world, you see the following retinal image ($\text{obs}_a$):
```
▢▢▢▢▢▢■▢▢▢
▢▢▢▢▢■▢▢▢▢
```  
  Calculate the posterior probability of $H1$ and $H2$ and report your results in Moodle.
  
- b) Suppose in another scenario you have the following retinal image ($\text{obs}_b$):
```
▢▢▢▢▢▢■▢▢▢
▢▢▢▢▢▢■▢▢▢
```  
  Calculate the posterior probability of $H1$ and $H2$ and report your results in Moodle.

- c) Write out a brief explanation of your reasoning for parts (a) and (b), and report them in Moodle. Add a brief explanation for how your answer to (b) may explain why observers in this world may tend to perceive the second image as containing a *single object*, as opposed to two separate dots. (max 200 words)

### Answers

Write your extended answers here if needed, and report a summary in Moodle (max 200 words).


```python
# code here...
# b)
ph1 = 0.03/(0.03+0.007)
ph2 = 1 - ph1
ph1
```




    0.8108108108108109




```python
ph2
```




    0.18918918918918914



# Question 1.3 (5 pts)

> In this question, we examine how an observer would estimate a continuous quantity under a noisy measurement.

An observer is estimating the horizontal location of a visual stimulus on a screen (for simplicity, we assume a 1D problem). 

We assume a Bayesian observer with prior $p(s_\text{hyp}) = \mathcal{N}\left(s| \mu_s = 2, \sigma_s^2 = 5^2 \right)$ and likelihood function $p(x_\text{obs}| s_\text{hyp}) = \mathcal{N}\left(x_\text{obs}| s_\text{hyp}, \sigma^2 = 2^2 \right)$ with observed noisy measurement $x_\text{obs} = -3$, in arbitrary units.

- a) What's the value of the posterior mean estimate $\hat{s}_\text{PM}$?
- b) What's the value of the maximum-likelihood estimate $\hat{s}_\text{ML}?$
- c) What's the probability density of the posterior at $s_\text{hyp} = 2.5$?

Report your results in Moodle.


```python
# code here...
x_obs = -3
mu_s = 2
sigma_s = 5
sigma = 2

s_pm = (x_obs * sigma_s**2 + mu_s * sigma**2)/(sigma**2 + sigma_s**2)
s_psigma = np.sqrt((sigma**2 * sigma_s**2)/(sigma**2 + sigma_s**2))
posterior_pdf = sps.norm.pdf(2.5, s_pm, s_psigma)
posterior_pdf
```




    0.007498207939080527



# Question 1.4 (5 pts)

> In this question, we examine a more complex inference scenario under a noisy measurement and a complex prior.


A Bayesian observer is estimating the value of a stimulus (e.g., horizontal location of a sound source, in arbitrary units).
The observer is told that there are two potential sound sources (e.g., two speakers hidden behind a screen), one to the right and one to the left (0 is straight ahead), but the observer is not told the exact location of these sound sources, just a vague position.

Thus, we represent the observer's prior over the potential sound location as a mixture of $K = 2$ Gaussians:
$$p(s_\text{hyp}) = \sum_{k=1}^K w_k \mathcal{N}\left(s_\text{hyp}| \mu_k, \sigma_k^2\right)$$
where 
$$w_1 = w_2 = \frac{1}{2}, \qquad \mu_1 = -3, \mu_2 = 3, \qquad \sigma_1 = \sigma_2 = 1.$$
Each mixture component represents one of the two potential locations of the sound (each component is Gaussian, and not a single point, because the location of the source itself is not exactly known).

After the sound is played (heard as noisy measurement $x_\text{obs}$), the likelihood is Gaussian, $p(x_\text{obs}| s_\text{hyp}) = \mathcal{N}\left(x_\text{obs}| s_\text{hyp}, \sigma^2 \right)$, with $\sigma = 1$.

- a) Compute the posterior mean for $x_\text{obs} = 1$ via numerical integration.
- b) Compute $p(x_\text{obs})$ for $x_\text{obs} = 5$ via numerical integration.
- c) Given that the prior is a mixture of Gaussians and the likelihood is Gaussian, this is a case in which we could still perform all computations analytically. Write the analytical expression for $p(x_\text{obs})$. Double-check the validity of your expression by computing $p(x_\text{obs})$ for $x_\text{obs} = 5$ and that it is the same (up to a small numerical error) as what you got in part (b).

Report your numerial results in Moodle, and write the analytical expression for $p\left(x_\text{obs}\right)$ below.

### Answer:

Write your expression for $p(x_\text{obs})$ here.

\begin{align*}
p(x_{obs})   &= \int p(s_{hyp})p(x_{obs} \mid s_{hyp})ds \\
             &= \frac{1}{2}(\int N(s_{hyp} \mid -3,1)N(s_{hyp} \mid x,1)ds + \int N(s_{hyp} \mid 3,1)N(s_{hyp} \mid x,1)ds) \\
             &= \frac{1}{2}(\int N(x_{obs} \mid -3,\sqrt{2})N(s_{hyp} \mid 1,\frac{\sqrt{2}}{2})ds + \int N(x_{obs} \mid 3,\sqrt{2})N(s_{hyp} \mid 4,\frac{\sqrt{2}}{2})ds) \\
             &= \frac{1}{2}(N(x_{obs} \mid -3,\sqrt{2}) + N(x_{obs} \mid 3,\sqrt{2}))

\end{align*}



```python
# code here...
w1 = 0.5
w2 = 0.5
mu1 = -3
mu2 = 3
sigma1 = 1
sigma2 = 1
N = int(2**9 + 1)
xobs1 = 1
xobs2 = 5
lb = mu1 - 5 * sigma1
hb = mu2 + 5 * sigma2
s_grid = np.linspace(lb, hb, N)

prior_pdf = w1 * sps.norm.pdf(s_grid, mu1, sigma1) + w2 * sps.norm.pdf(s_grid, mu2, sigma2)
likelihood1 = sps.norm.pdf(s_grid, xobs1, sigma1)
ds = s_grid[1] - s_grid[0]
protoposterior = prior_pdf * likelihood1
normalization_constant = sp.integrate.romb(protoposterior, dx=ds)
posterior_pdf = protoposterior / normalization_constant
posterior_mean = sp.integrate.romb(posterior_pdf * s_grid, dx=ds)
posterior_mean
```




    1.8577223804673217




```python
likelihood2 = sps.norm.pdf(s_grid, xobs2, sigma1)
protoposterior1 = prior_pdf * likelihood2
normalization_constant1 = sp.integrate.romb(protoposterior1, dx=ds)
normalization_constant1
```




    0.05188845265037266




```python
xobs5 = (sps.norm.pdf(xobs2, -3, np.sqrt(2)) + sps.norm.pdf(xobs2, 3, np.sqrt(2)))/2
xobs5
```




    0.05188845305036769



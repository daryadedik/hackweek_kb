---
title: Multiple correction investigation
authors:
- Darya Dedik
tags:
- Python
- Statistics
- Data Science
- Multiple correction
created_at: 2017-12-18 00:00:00
updated_at: 2017-12-18 14:52:19.568874
tldr: Jupyter-Notebook for multiple correction problem investigation in statistics
path: projects/multiple-correction-advanced
---
```python
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as st
import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
```
Multiple correction method families:

* FWER: probability of making 1 or more false discoveries (Type 1 error or false positives or conclusion about the existence of effect or relationship when in fact there is none). FWER methos limit the probability of at least 1 false discovery. It does not treat multiple simultaneous false discoveries as any worse than 1 false discovery.
* FDR: limits the expected portion of false discoveries.

In our a/b testing model we would like to avoid to make a lot of false positive decisions, but at the same time, we wouldn't like to increase the probability of false negatives or not detecting an effect or relationship when there is one. We need a balance between false positives and false negatives or more robust multiple correction strategy.

### Simulated data


```python
alpha = 0.05
size1 = 5
size2 = 5
numtests = size1+size2

np.random.seed(0)
```

```python
def print_error_rates(test, size1, size2):
    print("True: %d" % sum([t for t in test[:size1]]))
    print("False: %d" % sum([not t for t in test[:size1]]))

    print("True: %d" % sum([t for t in test[size1:]]))
    print("False: %d" % sum([not t for t in test[size1:]]))
    
    print("Type I error rate (false positives): %.4f" % (sum([not t for t in test[:size1]]) / float(size1)))
    print("Type II error rate (false negatives): %.4f" % (sum([t for t in test[size1:]]) / float(size2)))
```
We perform 250 hypothesis tests and exploit normals in such way that we will know the results of each hypothesis will be. 

Hypothesis:

* <b>H0: value of x is not different from 0, given the entries are drawn from a standard normal distribution
* <b>H1: value of x is larger than 0



```python
data1= st.norm.rvs(0, 1, size=size1) # random values from normal distribution with mean 0 and std 1
data2 = st.norm.rvs(2, 1, size=size2) # random values from normal distribution with mean 2 and std 1
data = np.concatenate((data1, data2), axis=0)
print("Data points: \n{}".format(data))

pvals = 2*(1-st.norm.cdf(abs(data))) # p-values
print("Probabilities: \n{}".format(pvals))
```
    Data points: 
    [ 1.76405235  0.40015721  0.97873798  2.2408932   1.86755799  1.02272212
      2.95008842  1.84864279  1.89678115  2.4105985 ]
    Probabilities: 
    [ 0.07772317  0.68904073  0.32770946  0.02503299  0.06182371  0.30643925
      0.00317683  0.06450941  0.05785683  0.01592637]



```python
test = pvals > alpha
print(test[:50])
```
    [ True  True  True False  True  True False  True  True False]


$error_I = \frac{sum(pvals_{false})}{size_{data1}}$

$error_{II} = \frac{sum(pvals_{true})}{size_{data2}}$


```python
print_error_rates(test, size1, size2)
```
    True: 4
    False: 1
    True: 3
    False: 2
    Type I error rate (false positives): 0.2000
    Type II error rate (false negatives): 0.6000


### P-values and confidence intervals relationship

Conclusions based on P-values and zero in/out of confidence intervals are interchangable and used to conclude on deciding the significance of the results during hypothesis testing.

If a corresponding hypothesis test is performed, the confidence level is the complement of the level of significance; for example, a 95% confidence interval reflects a significance level $\alpha$ of 0.05. If it is hypothesized that a true parameter value is 0 but the 95% confidence interval does not contain 0, then the estimate is significantly different from zero at the 5% significance level.

$p$ values themselves are the smallest significance levels ($\alpha$) at which the null hypothesus is rejected. Meaning p value is not alpha even it can be equal to alpha in some cases. $p$ value, in its different definition, is the probability of finding observed, or even extreme, result, when null hypothesis is true. While confidence interval is an estimate of your data as if we have an unknown value in the population and we want to get a good estimate of its value, $p$ value is a property of data itself.

Both metrics can be used in hypothesis testing for defining statistical significance of the result. Note that significance level (0.05) is set before data collection and it is the probability of the study rejecting the null hypothesis, given that it were true.

If the $p$ value is less than 0.05, then the 95% confidence interval will not contain zero (when comparing two means) and result will be statistically significant, if $p$ value is larger than 0.05, then the 95% confidence interval will contain zero and the result will be statisticaly insignificant. Given confidence interval 95% we know that the significance level is 0.05. Using this information we compute $p$ values and compare them with 0.05. 

### Bonferroni

1. Denote by $p_{i}$ the p-value for testing $H_{i}$
2. Reject $H_{i}$ if $p_{i} \le \frac{\alpha}{m}$


```python
bonf_test = pvals > alpha/numtests
print("Alpha: %.5f"%(alpha/numtests))
```
    Alpha: 0.00500



```python
print_error_rates(bonf_test, size1, size2)
```
    True: 5
    False: 0
    True: 4
    False: 1
    Type I error rate (false positives): 0.0000
    Type II error rate (false negatives): 0.8000


Bonferroni correction controls FWER: family-wise error rate or type I error. The correction comes at the cost of increasing the probability of producing type II error rate (false negatives), i.e., reducing statistical power.

In our particular case Bonferroni correction makes the confidence intervals wider so that 0 may fall into the CI and produce non-significant results.


```python
# Confidence interval for Bonferroni

percentls_bonf = [2.5, 35.5, 97.5]
percentls_bonf_adj = [float(p)/numtests if p < 50.0 else 100 - (100 - float(p)) / numtests if p > 50.0 
                      else p for p in percentls_bonf]

print("Percentiles: {}".format(alpha / numtests))
print("Percentiles: {}".format(percentls_bonf_adj))
```
    Percentiles: 0.005
    Percentiles: [0.25, 3.55, 99.75]


1. Testing each hypothesis at level $\alpha_{SID}=1-(1-\alpha )^\frac{1}{m}$ is Sidak's multiple testing procedure.
2. This procedure is more powerful than Bonferroni but the gain is small.
3. This procedure can fail to control the FWER when the tests are negatively dependent.


```python
rej_sidak, pval_corr_sidak = smm.multipletests(pvals, alpha=alpha, method='s')[:2]
accept_sidak = [not i for i in rej_sidak]
print_error_rates(accept_sidak, size1, size2)
```
    True: 5
    False: 0
    True: 4
    False: 1
    Type I error rate (false positives): 0.0000
    Type II error rate (false negatives): 0.8000



```python
# Confidence interval for Sidak

percentls_sidak = [2.5, 97.5]
percentls_sidak_adj = [float(p)**(1/numtests) if p < 50.0 else 100 - (100 - float(p))**(1/numtests) 
                       if p > 50.0 else p for p in percentls_sidak]

print("Percentiles: {}".format(percentls_sidak_adj))
```
    Percentiles: [1.0, 99.0]


### Benjamini-Hochberg and other FRD control methods

Other FDR controlling methods are step procedures of adjusting the number of true null hypotheses. More detailed examples can be found in the previous investigation: https://github.bus.zalan.do/axolotl/experimentation-library/blob/master/multiple-correction-problem/multiple-correction.ipynb


### FCR-controlling procedure of Benjamini-Hochberg

#### Selective inference

When we talk about the multiplicity in confidence intervals we should distinguish between simultaneous inference and selective inference. Simultaneous confidence intervals mean that all the parameters are covered with 1-$\alpha$ confidence.
Selective confidence intervals mean that a subset of selected parameters are covered. Benjamini-Hochberg, Bonferroni and many other multiple correction procedures are procedures with simultaneous inference. Selective inference means that you construct intervals only on parameters for which you rejected the null hypothesis.



#### FCR-Adjusted BH-Selected CIs

One of the limitations of all FDR and some FWER methods is that they are unapplicable to adjustment of confidence intervals (CIs) which we want to use in A/B testing. Corrected $p$ values, as explained above, can't be translated to the corresponding corrected CIs.


One of the known methods of adjusting CIs for multiple correction is based on controlling the special generalization of false discovery rate (FDR) or the ratio of false discoveries among all discoveries - false coverage-statement rate (FCR) suggested by Y.Benjamini and D.Yekutieli in the paper: "False Discovery Rate - Adjusted Multiple Confidence Intervals for Selected Parameters". FCR is the average rate of false coverage, i.e. not covering the true parameters, among the selected intervals (read more: https://en.wikipedia.org/wiki/False_coverage_rate). Paper produces more sophisticated method of controlling the FCR, but we will stick to more simple version of controlling FCR for CIs: Benjamini-Hochberg FCR controlling procedure which tends to be better in terms of limiting the false positives without drastical increase of false negatives.

In the BH procedure, after sorting the $p$ values $P_{(1)}\ \le \dots \le P_{(m)}$ and calculating $R = max\{j:P_{(j)} \le j\cdot \frac{q}{m} \}$, the $R$ null hypothesis for which $P_{(i)} \le R \cdot \frac{q}{m}$ are rejected. The suggested method of adjusting for FCR at level $q$ is to construct a marginal CI with confidence level $1-R\cdot\frac{q}{m}$ for the $R$ parameters selected. They consider it to be the best possible general procedure.
The steps are the following:

1. Sort the $p$ values used for testing the $m$ hypothesis regarding the parameters, $P_{(1)}\ \le \dots \le P_{(m)}$.
2. Calculate $R = max\{i:P_{(i)} \le i\cdot \frac{q}{m} \}$
3. Select the $R$ parameters for which $P_{(i)} \le R\cdot\frac{q}{m}$, corresponding to the rejected hypotheses.
4. Construct a $1 - R\cdot\frac{q}{m}$ CI for each parameter selected.


```python
pvals_sorted = np.sort(pvals)

fdr_rate = 0.05 #5% of false false discoveries, this parameter is defined by researcher

if len([i for i,val in enumerate(pvals_sorted, 1) if val <= i*fdr_rate/numtests]) != 0:
    R = np.max([i for i,val in enumerate(pvals_sorted, 1) if val <= i*fdr_rate/numtests])
else:
    R = 0
    
print("R = %d" % (R))
```
    R = 1



```python
rejected_hypos = [(i,val) for i, val in enumerate(pvals_sorted) if val <= R*fdr_rate/numtests]
print("Rejected hypothesis: {}".format(rejected_hypos))
```
    Rejected hypothesis: [(0, 0.0031768300141499228)]



```python
ind = np.argsort(pvals)
fdr_bh = [True] * len(pvals)

for i, ix in enumerate(ind):
    if pvals[ix] > R*fdr_rate/numtests:
        break
    fdr_bh[ix] = False
```

```python
ind = np.argsort(pvals)
fdr_bh_2 = [True] * len(pvals)

for i, ix in enumerate(ind):
    if pvals[ix] > (i+1.0) * alpha/numtests:
        break
    fdr_bh_2[ix] = False
```

```python
print("BH selected")
print_error_rates(fdr_bh, size1, size2)
print("\nBH standard")
print_error_rates(fdr_bh_2, size1, size2)
```
    BH selected
    True: 5
    False: 0
    True: 4
    False: 1
    Type I error rate (false positives): 0.0000
    Type II error rate (false negatives): 0.8000
    
    BH standard
    True: 5
    False: 0
    True: 4
    False: 1
    Type I error rate (false positives): 0.0000
    Type II error rate (false negatives): 0.8000



```python
# Confidence intervals
for _,v in enumerate(pvals[0:len(rejected_hypos)]):
    print("Confidence interval for param {}: {}".format(v, 1 - R*fdr_rate/numtests))
for _,v in enumerate(pvals[len(rejected_hypos):]):
    print("Confidence interval for param {}: {}".format(v, 1 - alpha))
```
    Confidence interval for param 0.077723166533: 0.995
    Confidence interval for param 0.689040730125: 0.95
    Confidence interval for param 0.327709458876: 0.95
    Confidence interval for param 0.0250329938834: 0.95
    Confidence interval for param 0.0618237066533: 0.95
    Confidence interval for param 0.306439249638: 0.95
    Confidence interval for param 0.00317683001415: 0.95
    Confidence interval for param 0.064509409405: 0.95
    Confidence interval for param 0.0578568284099: 0.95
    Confidence interval for param 0.0159263700494: 0.95


### Benjamini-Krieger-Yekutieli

The Benjamini–Hochberg–Yekutieli procedure controls the false discovery rate under positive dependence assumptions.


```python
rej_bky, pval_corr_bky = smm.multipletests(pvals, alpha=alpha, method='fdr_tsbky')[:2]
accept_bky= [not i for i in rej_bky] 
print_error_rates(accept_bky, size1, size2)
```
    True: 5
    False: 0
    True: 4
    False: 1
    Type I error rate (false positives): 0.0000
    Type II error rate (false negatives): 0.8000


Cons:
1. Is difficult to implement (use of smm packages).
2. Less studies, not applicable to CI correction.


### Multiple correction for p values

Since there are quite a few methods for handling multiple correction for CIs and this topic is currently under research, it is a common practice to correct $p$ values instead of correcting CIs. This means that in our set-up, it make sense to implement $p$ values computation and exploit Benjamini-Hochberg correction strategy. CIs in this case may represent non-multiple hypothesis testing in the results. All other recommendations can be done with comparing $p$ values with $\alpha$: if $p$ values is less than alpha there is a high evidence of rejecting null hypothesis, otherwise, there is no evidence of rejecting it.

### Comparison table of type I error and type II errror for different number of tests and different strategies 

Type I error and type II error for the different number of tests. Split is 50(control)/50(treatment).

|Number of tests|10|20|30|50|80|100|200||10|20|30|50|80|100|200|
|------|------|
|<b>Type I error</b>||||||||<b>Type II error</b>||||||||
|<b>Control Family-wise Error Rate</b>||||||||||||||||
|Uncorrected|0.2|0.1|0.0667|0.12|0.1|0.08|0.04||0|0|0.1333|0.28|0.225|0.14|0.17|
|Bonferroni|0|0|0|0|0|0|0||0.2|0.2|0.4667|0.72|0.85|0.78|0.68|
|Sidak|0|0|0|0|0|0|0||0.2|0.2|0.4667|0.72|0.85|0.78|0.68|
|<b>Control False Discovery Rate</b>||||||||||||||||
|<i>Benjamini-Hochberg</i>|0|0|0|0.04|0.025|0.02|0.01||0.2|0.1|0.2667|0.32|0.4|0.26|0.34|
|Benjamini-Krieger-Yekutieli|0.2|0.1|0.0667|0.12|0.075|0.06|0.03||0|0|0.1333|0.32|0.325|0.18|0.22|

1. From the results above, for the small number of tests by correction we reduce false positives (0.2) to 0 but at the same time we increase false negatives from 0 to 0.2. This effect keeps appearing till the number of tests reaches 30. That is why, with the small number of tests, the correction may not be needed.

2. Bonferroni and Sidak methods are extremely strict and, by reducing true positive, also increase false negatives. 

3. The better choice is to use Benjamini-Hochberg correction or Benjamini-Krieger-Yekutieli: the first reduces both errors simultaneously. Benjamini-Krieger-Yekutieli seem to be rather robust method: it do not correct for multiple correction when the number of tests is small and begin to correct when the correction is needed, but at the same time it produces a bit more false positives than Benjamini-Hochberg.

<b>NOTE</b>: you may notice that "Uncorrected" results could seem even better than with correction (false negative rates for Benjamini-Hochberg or Benjamini-Krieger-Yekutieli), but this statement is actually not 100% true. This happens because all methods aim to reduce type I error or false positive rate. Some of methods do that very strictly (Bonferroni, Sidak) by increasing false negatives, others - by balancing between false positives and false negative. When we agree to do correction we should accept that this may produce more false negatives in the end. Having 0 both in false positives and false negatives is impossible scenario, because there is always a tradeoff between false positives and false negatives and only we can think about which error we are more prone to accept.

Type I error and type II error for the different number of tests and smaller effect size (Uncorrected, Bonferroni, Sidak, Benjamini-Hochberg,Benjamini-Krieger-Yekutieli). Split is 50(control)/50(treatment)

|Number of tests|10|20|30|50|80|100|200||10|20|30|50|80|100|200|
|------|------|
|<b>Type I error</b>||||||||<b>Type II error</b>||||||||
|<b>Control Family-wise Error Rate</b>||||||||||||||||
|Uncorrected|0.2|0.1|0.0667|0.12|0.1|0.08|0.04||0.6|0.2|0.4|0.6|0.625|0.44|0.48|
|Bonferroni|0|0|0|0|0|0|0||0.8|0.8|0.73|0.88|0.975|0.94|0.9|
|Sidak|0|0|0|0|0|0|0||0.8|0.8|0.73|0.88|0.975|0.94|0.9|
|<b>Control False Discovery Rate</b>||||||||||||||||
|<i>Benjamini-Hochberg</i>|0|0|0|0|0|0|0||0.8|0.7|0.6|0.76|0.975|0.82|0.68|
|Benjamini-Krieger-Yekutieli|0|0|0|0|0|0|0||0.8|0.7|0.6|0.76|0.975|0.82|0.64|

When we decreased the effect (the mean of the second data half) we made that part of observations to be more close to the first part in terms of distribution.
By that we produced an increase in false negatives (when we accept null hypothesis but in fact it's wrong) meaning that x is actually different from 0 (mean 2 produce a difference) but we failed to accept that. The effect in this particular simulation influences only how much true positives and true negatives we produce, but it does not change the state of correction methods: they decrease false positives in any case which now becomes simpler (values become closer to each other and we don't reject null hypothesis less and less). In fact by not rejecting null hypothesis we statistically increase the chances of false negatives (failing to reject false null hypothesis).

### Update on the correction of boundary percentiles

It was found out that it is possible to correct the upper and lower boundary of the percentiles namely 2.5% and 97.5% having corrected $p$ value and 95% confidence interval.

The procedure is the following:
* given new $p$ value we can compute $z$ statistic by: $z = st.norm.ppf(1-p/2)$
* given $z$ statistic that corresponds to the new $p$ value we can compute new confidence interval by $ci = st.norm.cdf(z)$
* the 97.5% percentile for 95% confidence interval becomes $ci\cdot100$
* since the normal distribution is symmetric the lower percentile is become $100-ci\cdot100$

<b>Remarks</b>:
1. Unfortunately I can't validate whether this approach is 100% correct in a sense that it produces correct corrected CIs, because multiple correction methods which correct $p$ values do not propose or mention any such correction translated to confidence intervals.
2. Other percentiles can't be corrected in the same way, so, only boundaries of CIs can be presented.


```python
p = 0.05
print("Initial uncorrected p: {}".format(p))
t_stat = st.t.ppf(1-p/2, df=size1 + size2 - 2)
print("t score: {}\n".format(t_stat))

print("P value corrected: {}".format(pval_corr_bky[0]))
t_stat = st.t.ppf(1-pval_corr_bky[0]/2, df=size1 + size2 - 2)
print("Corrected t score: {}\n".format(t_stat))
print("New upper percentile: %.2f" % (st.norm.cdf(t_stat)))
print("New lower percentile: %.2f\n" % (1-st.norm.cdf(t_stat)))
```
    Initial uncorrected p: 0.05
    t score: 2.30600413503
    
    P value corrected: 0.10492627482
    Corrected t score: 1.82819872729
    
    New upper percentile: 0.97
    New lower percentile: 0.03
    


So, we can find the boundaries of the condifence interval, e.g. for 95% we have 2.5% and 97.5% corrected percentiles. Other percentiles cannot be obtained by this method. 

It makes sense to implement both $p$ values and CI strategies since they both agree on statistical significance of the results:
* If the $p$ value is less than your significance ($\alpha$) level, the hypothesis test is statistically significant.
* If the confidence interval does not contain the null hypothesis value, the results are statistically significant.
* If the $p$ value is less than $\alpha$, the confidence interval will not contain the null hypothesis value.

This makes sense because we then can validate whether our approach is correct: if our corrected $p$ value is less than $\alpha$ then our corrected CIs should not contain null hypothesis value (or 0).

### Further questions

The main question that follows is: how to combine the results of all hypothesis tests to one decision of whether treatment actually has an affected on control or not. If let say we test 50 kpi, and each of them produces a decision about significance/non-significance having in total 50 hypothesis decisions, what is the strategy of produce a single recommendation. This question is a research question and should be studied. There are some proposed ideas in the article: http://egap.org/methods-guides/10-things-you-need-know-about-multiple-comparisons

### Resources

1. [False Discovery Rate–Adjusted Multiple Confidence
Intervals for Selected Parameters](http://www.math.tau.ac.il/~yekutiel/papers/JASA%20FCR%20prints.pdf)

2. http://blog.minitab.com/blog/adventures-in-statistics-2/understanding-hypothesis-tests%3A-confidence-intervals-and-confidence-levels

3. http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html

4. Cox, D. R. “A Remark on Multiple Comparison Methods.”

5. Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures.

6. http://egap.org/methods-guides/10-things-you-need-know-about-multiple-comparisons - nice read about 10 things to know about multiple correction problem

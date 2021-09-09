#!/usr/bin/env python
# coding: utf-8

# ## Chapter 01  - Dogs: loglinear model for binary data
# 
# 
# **Background:** 
# 
# Solomon-Wynne in 1953 conducted an experiment on avoidance learning in dogs from traumatic experiences in past such as those from electric shocks.
# The apparatus of experiment holds a dog in a closed compartment with steel flooring, open on side with a small barrier for dog to jump over to the other side. A high-voltage electric shock is discharged into the steel floor intermittently to stimulate the dog; Thus the dog is effectively left with an option to either get the shock for that trial or jump over the barrier to other side & save himself. Several dogs were subjected to similar experiment for many consecutive trials.
# This picture elicits the apparatus
# 
# ![dog_setup](./data/avoidance_learning.png)
# 
# The elaborate details of the experiment can be found at
# http://www.appstate.edu/~steelekm/classes/psy5300/Documents/Solomon&Wynne%201953.pdf
# 
# The hypothesis is that most of the dogs learnt to avoid shocks by jumping over barrier to the other side after suffering the trauma of shock in previous trials. That inturn sustain dogs in future encounters with electric shocks.

# Since the experiment aims to study the avoidance learning in dogs from past traumatic experiences and reach a plausible model where dogs learn to avoid scenerios responsible for causing trauma, we describe the phenomenon using expression
# $$
# \pi_j   =   A^{xj} B^{j-xj}
# $$
# Where :
#    * $\pi_j$ is the probability of a dog getting shocked at trial $j$
#    * A & B both are random variables drawing values from Normal distribution
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
# In the subsequent 
# 
# The hypothesis is thus corroborated by Bayesian modelling and comprehensive analysis of dogs data available from Solomon-Wynne experiment in Pyro.
# 

# The data is analysed step by step in accordance with Bayesian workflow as described in "Bayesian Workflow", Prof. Andrew Gelman [http://www.stat.columbia.edu/~gelman/research/unpublished/Bayesian_Workflow_article.pdf].
# 
# Import following dependencies.

# In[1]:


import torch
import pyro
import pandas as pd
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import pyro.distributions as dist
import seaborn as sns
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from chapter01 import base
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')


# ### 1. Model Specification: Dogs Model definition
# ________
# $$
# \pi_j   =   A^{xj} B^{j-xj}
# $$
# 
# We intend to find most probable values for parameters $\alpha$ & $\beta$ (dubbed as random variable A & B respectively) in the expression to compute likelihood ($\pi_j$) of dogs getting shocked.
# 
# **Generative model for resulting likelihood of shock:**
# 
# $\pi_j$  ~   $bern\ (\exp \ (\alpha.XAvoidance + \beta.XShocked)\ )$,  $prior\ \alpha$ ~ $N(0., 316.)$,  $\beta$ ~ $N(0., 316.)$
# 
# The above expression is used as a generalised linear model with log-link function in WinBugs implementation
# 
#   **BUGS model**
#   
# $\log\pi_j = \alpha\ x_j + \beta\ ( $j$-x_j )$
# 
#    **Here**
#    * $\log\pi_j$ is log probability of a dog getting shocked at trial $j$
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
#    *  $\alpha$ is the coefficient corresponding to number of success, $\beta$ is the coefficient corresponding to number of failures.
# 
#   
#   ____________________
# The same model when implemented in PyStan
#   
#   **Equivalent Stan model** 
#   
#       {
#   
#       alpha ~ normal(0.0, 316.2);
#   
#       beta  ~ normal(0.0, 316.2);
#   
#       for(dog in 1:Ndogs)
#   
#         for (trial in 2:Ntrials)  
# 
#           y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * 
#           xs[dog, trial]));
#       
#       }  
# 

# **Model implementation**
# 
# The model is defined using Pyro as per the expression of generative model for this dataset as follows

# In[2]:


DogsModel= base.DogsModel
DogsModel


# **Dogs data**
# 
# Following holds the Dogs data in the pystan modelling format

# In[3]:


dogs_data = base.load_data()
dogs_data


# **Following processes target label `y` to obtain input data `x_avoidance` & `x_shocked` where:**
# * `x_avoidance` :  number of shock avoidances before current trial.
# * `x_shocked` :  number of shocks before current trial.

# **Here the py-stan format data (python dictionary) is passed to the function above, in order to preprocess it to tensor format required for pyro sampling**

# In[4]:


x_avoidance, x_shocked, y= base.transform_data(**dogs_data)
print("x_avoidance: %s, x_shocked: %s, y: %s"%(x_avoidance.shape, x_shocked.shape, y.shape))

print("\nSample x_avoidance: %s \n\nSample x_shocked: %s"%(x_avoidance[1], x_shocked[1]))


# ### 2. Prior predictive checking
# 
# These checks help to understand the implications of a prior distributions of underlying parameters (random variables) in the context of a given generative model by simulating from the model rather than observed data.
# 

# In[5]:


priors_list= [(pyro.sample("alpha", dist.Normal(0., 316.)).item(), 
               pyro.sample("beta", dist.Normal(0., 316.)).item()) for index in range(1100)]# Picking 1100 prior samples

prior_samples = {"alpha":list(map(lambda prior_pair:prior_pair[0], priors_list)), "beta":list(map(lambda prior_pair:prior_pair[1], priors_list))}


# Sampled output of prior values for alpha & beta is stored in `prior_samples` above, and is plotted on a KDE plot as follows:

# In[6]:



fig = ff.create_distplot(list(prior_samples.values()), list(prior_samples.keys()))
fig.update_layout(title="Prior distribution of parameters", xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Prior alpha Q(0.5) :%s | Prior beta Q(0.5) :%s"%(np.quantile(prior_samples["alpha"], 0.5), np.quantile(prior_samples["beta"], 0.5)))


# ### 3. Posterior estimation
# 
# In the parlance of probability theory, Posterior implies the probability of updated beliefs in regard to a quantity or parameter of interest, in the wake of given evidences and prior information.
# 
# $$Posterior = \frac {Likelihood x Prior}{Probability \ of Evidence}$$
# 
# 
#  
# For the parameters of interest $\alpha,\beta$ & evidence y; Posterior can be denoted as $P\ (\alpha,\beta\ /\ y)$.
# 
# 
# $$P\ (\alpha,\beta\ /\ y) = \frac {P\ (y /\ \alpha,\beta) P(\alpha,\beta)}{P(y)}$$
# 
# Posterior, $P\ (\alpha,\beta\ /\ y)$ in regard to this experiment is the likelihood of parameter values (i.e., Coefficient of instances avoided dubbed retention ability & Coefficient of instances shocked dubbed learning ability) given the observed instances `y` of getting shocked. Where $P(\alpha,\beta)$ is prior information/likelihood of parameter values.

# The following intakes a pyro model object with defined priors, input data and some configuration in regard to chain counts & chain length prior to launching a `MCMC NUTs sampler` and outputs MCMC chained samples in a python dictionary format.

# In[7]:


hmc_sample_chains= base.get_hmc_n_chains(DogsModel, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# `hmc_sample_chains` holds sampled MCMC values as `{"Chain_0": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}, "Chain_1": {alpha	[-0.20020795, -0.1829252, -0.18054989 . .,], "beta": {}. .,}. .}`

# ### 4. Diagnosing model fit
# 
# Model fit diagnosis consists of briefly obtaining core statistical values from sampled outputs and assess the convergence of various chains from the output, before moving onto inferencing or evaluating predictive power of model.

# Following plots **Parameter vs. Chain matrix** and optionally saves the dataframe.

# In[8]:


beta_chain_matrix_df = pd.DataFrame(hmc_sample_chains)
beta_chain_matrix_df.to_csv("dogs_log_regression_hmc_sample_chains.csv", index=False)
beta_chain_matrix_df


# **Key statistic results as dataframe**
# 
# **Following outputs the summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, given a list of statistic names**

# In[9]:


key_metrics= ["mean", "std", "25%", "50%", "75%"]

summary_stats_df_= base.summary_stats_df(beta_chain_matrix_df, key_metrics)
summary_stats_df_


# **Obtain 5 point Summary statistics (mean, Q1-Q4, Std, ) as tabular data per chain and save the dataframe.**
# 

# In[10]:


fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

# fit_df.to_csv("data/dogs_classification_hmc_samples.csv", index=False)
fit_df


# In[11]:


# Use/Uncomment following once the results from pyro sampling operation are saved offline
# fit_df= pd.read_csv("data/dogs_classification_hmc_samples.csv")

fit_df.head(3)


# Following outputs the similar summary of required statistics such as `"mean", "std", "Q(0.25)", "Q(0.50)", "Q(0.75)"`, **But in a slightly different format**, given a list of statistic names**

# In[12]:


summary_stats_df_2= base.summary_stats_df_2(fit_df, key_metrics)
summary_stats_df_2


# **Following plots sampled parameters values as Boxplots with `M parameters` side by side on x axis for each of the `N chains`**

# In[13]:


parameters= ["alpha", "beta"]# All parameters for given model
chains= fit_df["chain"].unique()# Number of chains sampled for given model


# **Pass the list of `M parameters` and list of `N chains`, with `plot_interactive` as `True or False` to choose between Plotly or Seaborn**

# In[14]:


# Use plot_interactive=False for Normal seaborn plots offline

base.plot_parameters_for_n_chains(fit_df, chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=True)


# **Following plots the `joint distribution` of `pair of each parameter` sampled values for all chains**

# In[15]:



base.plot_joint_distribution(fit_df, parameters)


# **Following plots the `Pairplot distribution` of each parameter with every other parameter's sampled values**

# In[16]:


sns.pairplot(data=fit_df, hue= "chain");


# **Following intakes the list of parameters say `["alpha", "beta"]` and plots hexbins for each interaction pair for all possible combinations of parameters `alpha & beta`.**

# In[17]:


#launch docstring for plot_interaction_hexbins


# **Here parameters `["alpha", "beta"]` are passed to plot all possible interaction pair Hexbin plots in between**

# In[18]:



base.plot_interaction_hexbins(fit_df, parameters=parameters)


# ### 5. Model evaluation: Posterior predictive checks
# 
# Posterior predictive checking helps examine the fit of a model to real data, as the parameter drawn for simulating conditions & regions of interests come from the posterior distribution.

# **Pick samples from one particular chain of HMC samples say `chain_3`**

# In[19]:


import torch
for chain, samples in hmc_sample_chains.items():
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))


# **Plot density for parameters from `chain_3` to visualise the spread of sample values from that chain**

# In[20]:


title= "parameter distribution for : %s"%(chain)
fig = ff.create_distplot(list(map(lambda x:x.numpy(), samples.values())), list(samples.keys()))
fig.update_layout(title=title, xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))


# **Plot density & contours for both parameters from `chain_3` to visualise the joint distribution & region of interest**

# In[21]:


#Choosing samples from chain 3
chain_samples_df= fit_df[fit_df["chain"]==chain].copy()# chain is 'chain_3' 

alpha= chain_samples_df["alpha"].tolist()
beta= chain_samples_df["beta"].tolist()
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(alpha, beta, colorscale=colorscale, hist_color='rgb(255, 255, 150)', point_size=4, title= "alpha beta joint density plot")
fig.update_layout( xaxis_title="x (alpha)", yaxis_title="y (beta)")

fig.show()


# **Note:** The distribution of alpha values are significantly offset to the left from beta values, by almost 13 times; Thus for any given input observation of avoidances or shocked, the likelihood of getting shocked is more influenced by small measure of avoidance than by getting shocked.

# **Observations:**
# 
# **On observing the spread of alpha & beta values, the parameter beta being less negative & closer to zero can be interpreted as `learning ability`, i.e., the ability of dog to learn from shock experiences. The increase in number of shocks barely raises the probability of non-avoidance (value of 𝜋𝑗) with little amount. Unless the trials & shocks increase considerably large in progression, it doesn't mellow down well and mostly stays around 0.9.**
# 
# **Whereas its not the case with alpha, alpha is more negative & farthest from zero. It imparts a significant decline in non-avoidance (𝜋𝑗) even for few instances where dog avoids the shock; therefore alpha can be interpreted as `retention ability` i.e., the ability to retain the learning from previous shock experiences.**

# In[22]:


print(chain_samples_df["alpha"].describe(),"\n\n", chain_samples_df["beta"].describe())


# **From the contour plot above following region in posterior distribution seems highly plausible for parameters:**
# 1. For alpha, `-0.2 < alpha < -0.19`
# 2. For beta `-0.0075 < beta < -0.0055`
# 
# Following selects all the pairs of `alpha, beta` values between the range mentioned above.

# In[23]:


select_sample_df= chain_samples_df[(chain_samples_df["alpha"]<-0.19)&(chain_samples_df["alpha"]>-0.2)&(chain_samples_df["beta"]<-0.0075)&(chain_samples_df["beta"]<-0.0055)]

# print(select_sample_df.set_index(["alpha", "beta"]).index)
print("Count of alpha-beta pairs of interest, from mid region with high desnity in contour plot above (-0.2 < alpha < -0.19, -0.0075 < beta < -0.0055): ", select_sample_df.shape[0])

select_sample_df.head(3)


# **Picking a case of 3 trials with Y [0,1,1], i.e. Dog is shocked in 1st, Dogs avoids in 2nd & thereafter, effectively having an experience of 1 shock & 1 avoidance. `Considering all values of alpha & beta in range -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055`**

# In[24]:


Y_li= []
Y_val_to_param_dict= defaultdict(list)

# Value -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055
for rec in select_sample_df.iterrows():# for -0.2 < alpha < -0.19, -0.0075 < beta < -0.0055
    a,b = float(rec[1]["alpha"]), float(rec[1]["beta"])
    res= round(np.exp(a+b), 4)
    Y_li.append(res)
    Y_val_to_param_dict[res].append((round(a,5),round(b,5)))# Sample-- {0.8047: [(-0.18269378, -0.034562342), (-0.18383412, -0.033494473)], 0.8027: [(-0.18709463, -0.03263992), (-0.18464606, -0.035114493)]}


# In above `Y_val_to_param` is a dictionary that holds value $\exp^{\alpha +\beta}$ as key and tuple of corresponding $(\alpha, \beta)$ as value.
# 
# The following plots the histogram of $\exp^{\alpha +\beta}$ values obtained as an interaction of selected $\alpha$ and $\beta$ values from region of interest.

# In[25]:


Y_for_select_sample_df = pd.DataFrame({"Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055": Y_li})
fig = px.histogram(Y_for_select_sample_df, x= "Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055")
title= "observed values distribution for params Y_for -0.2 < alpha < -0.19 & -0.0075 < beta < -0.0055"

fig.update_layout(title=title, xaxis_title="observed values", yaxis_title="count", legend_title="dogs")
fig.show()
print("Mean: %s | Median: %s"%(np.mean(Y_li), np.quantile(Y_li, 0.5)))
print("Sorted observed values: \n", sorted(Y_li))


# **For given experiment of 3 trials, from all the `Ys` with corresponding alpha-beta pairs of interest, pick 3  lower most values of `Y` for instance; Thus selecting its corresponding alpha-beta pairs**
# 
# **Note:** Can add multiple observed values from histogram for comparison.

# Corresponding to `lowest_obs` values of `Y`, obtain `select_pairs` as list of correspoding alpha, beta pairs from  `Y_val_to_param_dict`.

# In[26]:


lowest_obs = sorted(Y_li)[:3]#[0.8085, 0.8094, 0.8095]# Pick values from above histogram range or sorted list

selected_pairs= list(itertools.chain.from_iterable(map(lambda obs: Y_val_to_param_dict.get(obs), lowest_obs)))
selected_pairs


# **Following stores a dictionary of `observed y` values for pair of alpha-beta parameters**

# In[27]:


obs_y_dict= base.get_obs_y_dict(selected_pairs, x_avoidance, x_shocked)

print("Alpha-beta pair values as Keys to access corresponding array of inferred observations: \n", list(obs_y_dict.keys()))


# **Following plots scatterplots of `observed y` values for all 30 dogs for each alpha-beta pair of interest**

# In[28]:


#launch docstring for plot_observed_y_given_parameters


# **Also Optionally pass the `True observed y` values to `original_obs` argument for all 30 dogs to plot alongside the `observed y` from alpha-beta pairs of interest.**
# 
# **_Note_**: `True observed y` are marked with legends in format `*Dog_X`

# In[29]:


base.plot_observed_y_given_parameters(lowest_obs, selected_pairs, obs_y_dict, chain, original_obs= y)


# **Following plots a single scatterplots for comparison of `observed y` values for all alpha-beta pairs of interest from dense region in contourplot above, that is `-0.2 < alpha < -0.19`, `-0.0075 < beta < -0.0055`**
# 

# In[30]:



#launch docstring for compare_dogs_given_parameters


# **Also Optionally pass the `True observed y` values to `original_obs` argument for all 30 dogs to plot alongside the `observed y` from alpha-beta pairs of interest.**
# 
# **_Note_**: `True observed y` are marked with legends in format `*Dog_X`

# In[31]:


base.compare_dogs_given_parameters(selected_pairs, obs_y_dict, original_obs= y)


# **Observations:** The 3 individual scatter plots above correspond to 3 most optimum alpha-beta pairs from 3rd quadrant of contour plot drawn earlier; Also the scatterplot following them faciliates comparing obeserved y values for all 3 pairs at once:
# 
# Data for almost all dogs in the experiment favours m1 parameters (-0.19852, -0.0156), over m3 & m2; With exceptions of Dog 6, 7 showing affinity to m3 parameters (-0.19804, -0.01568), over m2 & m1 at all levels of 30 trials.

# **Plotting observed values y corresponding to pairs of alpha-beta with with `mean, minmum, maximum value of` $\frac{alpha}{beta}$**

# **Following computes $\frac{alpha}{beta}$ for each pair of alpha, beta and outputs pairs with `mean, maximum & minimum values`; that can therefore be marked on a single scatterplots for comparison of observed y values for all alpha-beta pairs of interest**

# In[32]:



#launch docstring for get_alpha_by_beta_records


# In[33]:


alpha_by_beta_dict = base.get_alpha_by_beta_records(chain_samples_df, metrics=["max", "min", "mean"])# outputs a dict of type {(-0.2010, -0.0018): 107.08}
print("Alpha-beta pair with value as alpha/beta: ", alpha_by_beta_dict)


alpha_by_beta_selected_pairs= list(alpha_by_beta_dict.keys())
alpha_by_beta_obs_y_dict = base.get_obs_y_dict(alpha_by_beta_selected_pairs, x_avoidance, x_shocked)# Outputs observed_values for given (alpha, beta)


# **Following is the scatter plot for `observed y` values corresponding to pairs of `alpha, beta` yielding `minimum, maximum & mean` value for $\frac{alpha}{beta}$.**
# 
# **_Note_**: The y i.e., original observations are simultaneously plotted side by side.

# In[34]:



base.compare_dogs_given_parameters(alpha_by_beta_selected_pairs, alpha_by_beta_obs_y_dict, 
                              original_obs= y, alpha_by_beta_dict= alpha_by_beta_dict)


# **Observations:** The scatter plots above corresponds to 3 pairs of alpha-beta values from contour plot drawn earlier, which correspond to maxmimum, minimum & mean value of 𝛼/𝛽. Plot faciliates comparing `obeserved y` values for all pairs with `True observed y` at once:
# 
#     1. Data for for first 7 dogs in the experiment favours m1 parameters (-0.184, -0.0015) with highest 𝛼/𝛽 around 116, followed by m3 & m2 at all levels of 30 trials. Avoidance learning in these 7 dogs is captured suitablely by model 2 but most of the instances for which they are shocked, are modelled well with m1 parameters.
#     
#     2. Data for for rest 23 dogs in the experiment showed affinity for m2 parameters (-0.197, -0.016) with lowest 𝛼/𝛽 around 11, followed by m3 & m1 at all levels of 30 trials; Likewise Avoidance learning in these 23 dogs is captured suitablely by model 2 but most of the instances for which they are shocked, are modelled well with m1 parameters only.
#     
#     3. Data for Dogs 18-20 fits model 2 increasingly well after 10th trial; Whereas for Dogs 21-30 model 2 parameters fit the original data exceptionally well after 6th trial only.

# ### 6. Model Comparison
# **Compare Dogs model with Normal prior & Uniform prior using Deviance Information Criterion (DIC)**

# **DIC is computed as follows**
# 
# $D(\alpha,\beta) = -2\ \sum_{i=1}^{n} \log P\ (y_{i}\ /\ \alpha,\beta)$
# 
# $\log P\ (y_{i}\ /\ \alpha,\beta)$ is the log likehood of shocks/avoidances observed given parameter $\alpha,\beta$, this expression expands as follows:
# 
# $$D(\alpha,\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\alpha Xa_{i}\ +\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\alpha Xa_{i}\ +\beta\ Xs_{i})})]$$
# 
# 
# #### Using $D(\alpha,\beta)$ to Compute DIC
# 
# $\overline D(\alpha,\beta) = \frac{1}{T} \sum_{t=1}^{T} D(\alpha,\beta)$
# 
# $\overline \alpha = \frac{1}{T} \sum_{t=1}^{T}\alpha_{t}\\$
# $\overline \beta = \frac{1}{T} \sum_{t=1}^{T}\beta_{t}$
# 
# $D(\overline\alpha,\overline\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i})})]$
# 
# 
# **Therefore finally**
# $$
# DIC\ =\ 2\ \overline D(\alpha,\beta)\ -\ D(\overline\alpha,\overline\beta)
# $$
# 
# 

# **Following method computes deviance value given parameters `alpha & beta`**

# In[35]:



#launch docstring for calculate_deviance_given_param

#launch docstring for calculate_mean_deviance


# **Following method computes `deviance information criterion` for a given bayesian model & chains of sampled parameters `alpha & beta`**

# In[36]:


#launch docstring for DIC

#launch docstring for compare_DICs_given_model


# **Define alternate model with different prior such as uniform distribution**
# 
# The following model is defined in the same manner using Pyro as per the following expression of generative model for this dataset, just with modification of prior distribution to `Uniform` rather than `Normal` as follows:
# 
# $\pi_j$  ~   $bern\ (\exp \ (\alpha.XAvoidance + \beta.XShocked)\ )$,  $prior\ \alpha$ ~ $U(0., 316.)$,  $\beta$ ~ $U(0., 316.)$

# In[37]:


# # Dogs model with uniform prior

#launch docstring for DogsModelUniformPrior


# In[38]:


DogsModelUniformPrior= base.DogsModelUniformPrior
DogsModelUniformPrior


# In[39]:



hmc_sample_chains_uniform_prior= base.get_hmc_n_chains(DogsModelUniformPrior, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# **compute & compare `deviance information criterion` for a multiple bayesian models**

# In[40]:


base.compare_DICs_given_model(x_avoidance, x_shocked, y, Dogs_normal_prior= hmc_sample_chains, Dogs_uniform_prior= hmc_sample_chains_uniform_prior)


# **Since the Model `Dogs_normal_prior` has DIC lower than Model `Dogs_uniform_prior`, therefore Model `Dogs_uniform_prior` is more likely to fit the observed data successfully**

# _______________

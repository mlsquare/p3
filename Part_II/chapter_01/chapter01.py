import numpy as np
import os
import torch
import pyro
import random
import time
import numpy as np
import pandas as pd
import re
from io import StringIO
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict

from scipy import stats
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS


import plotly
import plotly.express as px
import plotly.figure_factory as ff
from statsmodels.graphics.tsaplots import plot_acf
import ipywidgets as widgets
from IPython.display import display, clear_output


slide_counter=0

class base(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_original_y(original_obs,ylabel=None):
        """

        Input
        -------
        original_obs: original observations/ labels from given data

        returns  plotly scatter plots with number of trials on X axis & corresponding probability of getting
        shocked for each pair of (alpha, beta) passed in 'selected_pairs_list'.

        Output
        --------
        Plots scatter plot of all observed values of y corresponding to each given pair of alpha, beta

        """
        # num_dogs = base.load_data()["Ndogs"] if original_obs.ndim!=1 else 1
        num_dogs = original_obs.shape[0] if original_obs.ndim!=1 else 1
        obs_column_names = [f'Dog_{ind+1}' for ind in range(num_dogs)] if num_dogs!=1 else ["Dogs"]
        obs_y_df= pd.DataFrame(original_obs.T, columns=obs_column_names)

        if ylabel is None:
            ylabel = "Probability of shock at trial j (𝜋𝑗)"

        obs_y_title= "Original observed value distribution for all dogs"
        fig = px.scatter(obs_y_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title=ylabel, legend_title="Dog identifier")
        fig.show()
    


    @staticmethod
    def DogsModel(x_avoidance, x_shocked, y, alpha_prior= None, beta_prior= None, activation= "exp"):
        """
        Input
        -------
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25)
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        alpha_prior: pyro distribution for sampling
        beta_prior: pyro distribution for sampling
        activation: activation function to use inside likelihood function, default: "exp"

        Output
        --------
        Implements pystan model: {
                alpha ~ normal(0.0, 316.2);
                beta  ~ normal(0.0, 316.2);
                for(dog in 1:Ndogs)  
                    for (trial in 2:Ntrials)  
                    y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}

        Note: Implements model with variations of prior distribution & sampling activation function
        and output variant models as 1A, 1B, 2A, 2B.

        """
        beta_prior= beta_prior if beta_prior else alpha_prior
        
        activation_function = lambda name, value: {"sigmoid": 1/(1+torch.exp(-value)), "exp":torch.exp(-value)}.get(name)
        alpha = pyro.sample("alpha", alpha_prior)#10
        beta = pyro.sample("beta", beta_prior)
        with pyro.plate("data"):
            pyro.sample("obs", dist.Bernoulli(activation_function(activation, alpha*x_avoidance + beta * x_shocked)), obs=y)

    @staticmethod
    def init_priors(prior_dict={"default":dist.Normal(0., 316.)}):
        """
        Input
        -------


        
        Output
        --------
        """
        prior_dict["names"]= prior_dict.get("names") if prior_dict.get("names") else ["alpha", "beta"] 
        # if not prior_dict.get("names"):
        #     raise Exception("pass 'parameters name as list' to key 'names', example: {'names': ['alpha', 'beta']}")
        if not prior_dict.get("default"):
            raise Exception("pass a deafault distribution to key 'default' for parameters ['alpha', 'beta']}")
        prior_list = []
        [prior_list.append(prior_dict.get(param, prior_dict.get("default"))) for param in prior_dict.get("names")];
        return prior_list

    @staticmethod
    def get_prior_samples(alpha_prior, beta_prior, num_samples=1100):
        """
        Input
        -------


        
        Output
        --------
        """
        priors_list= [(pyro.sample("alpha", alpha_prior).item(), 
                pyro.sample("beta", beta_prior).item()) for index in range(num_samples)]# Picking 1100 prior samples
        prior_samples = {"alpha":list(map(lambda prior_pair:prior_pair[0], priors_list)), 
                        "beta":list(map(lambda prior_pair:prior_pair[1], priors_list))}

        return prior_samples

    @staticmethod
    def plot_prior_distributions(**kwargs):
        """
        Input
        -------


        
        Output
        --------
        """
        for model_name, prior_samples in kwargs.items():
            print("For model '%s' Prior alpha Q(0.5) :%s | Prior beta Q(0.5) :%s"%(model_name, np.quantile(prior_samples["alpha"], 0.5), np.quantile(prior_samples["beta"], 0.5)))
            fig = ff.create_distplot(list(prior_samples.values()), list(prior_samples.keys()))
            fig.update_layout(title="Prior distribution of '%s' parameters"%(model_name), xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
            fig.show()
    
    @staticmethod
    def DogsModel_(x_avoidance, x_shocked, y, alpha_prior= None, beta_prior= None, activation= "exp"):
        """
        Input
        -------
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25)
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        alpha_prior: pyro distribution for sampling
        beta_prior: pyro distribution for sampling
        activation: activation function to use inside likelihood function, default: "exp"

        Output
        --------
        Implements pystan model: {
                alpha ~ normal(0.0, 316.2);
                beta  ~ normal(0.0, 316.2);
                for(dog in 1:Ndogs)  
                    for (trial in 2:Ntrials)  
                    y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}

        Note: Implements model with variations of prior distribution offset by certain value & sampling activation function
        and output variant models as 1A, 1B, 2A, 2B.

        """
        beta_prior= beta_prior if beta_prior else alpha_prior

        activation_function = lambda name, value: {"sigmoid": 1/(1+torch.exp(-value)), "exp":torch.exp(-value)}.get(name)
        alpha = pyro.sample("alpha", alpha_prior)#10
        beta = pyro.sample("beta", beta_prior)
        alpha_ = alpha + 4# offset by 4
        beta_ = beta + 4# offset by 4

        with pyro.plate("data"):
            pyro.sample("obs", dist.Bernoulli(activation_function(activation, alpha_*x_avoidance + beta_*x_shocked)), obs=y)

    @staticmethod
    def load_data():
        dogs_data = {"Ndogs":30, 
             "Ntrials":25, 
             "Y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 
                  1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 
                  1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 
                  0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 
                  0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 
                  1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((30,25))}
        return  dogs_data

    @staticmethod
    def transform_data(Ndogs=30, Ntrials=25, Y= np.array([0, 0, 0, 0])):
        """
        Input
        -------
        Ndogs: Total number of Dogs i.e., 30
        Ntrials: Total number of Trials i.e., 25
        Y: Raw responses from data, example: np.array([0, 0, 0, 0])
        
        Outputs
        ---------
        xa: tensor holding avoidance count for all dogs & all trials
        xs: tensor holding shock count for all dogs & all trials
        y: tensor holding response observation for all dogs & all trials
        
        """
        y= np.zeros((Ndogs, Ntrials))
        xa= np.zeros((Ndogs, Ntrials))
        xs= np.zeros((Ndogs, Ntrials))

        for dog in range(Ndogs):
            for trial in range(1, Ntrials+1):
                xa[dog, trial-1]= np.sum(Y[dog, :trial-1]) #Number of successful avoidances uptill previous trial
                xs[dog, trial-1]= trial -1 - xa[dog, trial-1] #Number of shocks uptill previous trial
        for dog in range(Ndogs):
            for trial in range(Ntrials):
                y[dog, trial]= 1- Y[dog, trial]
        xa= torch.tensor(xa, dtype=torch.float)
        xs= torch.tensor(xs, dtype=torch.float)  
        y= torch.tensor(y, dtype=torch.float)

        return xa, xs, y

    @staticmethod
    def simulate_observations_given_param(init_obs=None, num_dogs=1, num_trials=30, 
                                        parameter_pair_list= [(-2.490809, -1.642264)], 
                                        activation = "exp", prior_offset= 0, print_logs=0, print_exceptions=0):
        """
        Input
        -------



        Output
        --------
        """
        t1= time.time()
        simulated_data_given_pair = []
        activation_function = lambda name, value: {"sigmoid": 1/(1+np.exp(-value)), "exp":np.exp(-value)}.get(name)
        for alpha, beta in parameter_pair_list:
            simulated_y= []# Dogs level list
            try:
                for dog in range(num_dogs):
                    Y_list = [0]# Trials level list, This holds P(Y=1), where Y is avoidance
                    random_trial = init_obs if init_obs else np.random.randint(2)
                    for trial in range(num_trials):
                        Y_list.append(random_trial)
                        test_data = {'Y': np.array([Y_list])}
                        test_data["Ndogs"]= len(test_data["Y"])#len(Y_list) 
                        test_data["Ntrials"]= len(Y_list)

                        test_x_avoided, test_x_shocked, _ = base.transform_data(**test_data)
                        if print_logs:
                            print("____\n\nFor Priors alpha: %s | beta: %s | trial: %s"%(alpha, beta, trial))
                            print("test_data: ", test_data)
                            print("trial number %s observations -- %s, %s"%(trial, test_x_avoided, test_x_shocked))

                        alpha+=prior_offset# Add offset value if any
                        beta+=prior_offset# Add offset value if any
                        calculate_pij = lambda alpha, beta, test_x_avoided, test_x_shocked: activation_function(activation, alpha*np.array(test_x_avoided) + 
                                                                                        beta*np.array(test_x_shocked))

                        Pij_arr= calculate_pij(alpha, beta, test_x_avoided, test_x_shocked)# 𝜋𝑖𝑗=exp(𝛼𝑋𝑎+𝛽𝑋𝑠),𝜋𝑖𝑗 is P(getting shocked)
                        Pij= Pij_arr[0][-1]# Pij, P(getting shocked) for last trial

                        P_avoidance_ij = 1-Pij

                        obs_y= np.random.binomial(1, P_avoidance_ij)# obs_y =1 indicates avoidance as success
                        random_trial = obs_y

                        if print_logs:
                            print("Pij for trial number %s --  %s"%(trial, Pij))
                            print("Next observed trial: %s"%random_trial)


                    simulated_y.append(Y_list)
                simulated_data_given_pair.append(simulated_y)# Contains 'avoidances' as obs/success
            except Exception as error:
                if print_exceptions:
                    print("Issue '%s' with parameter pair: %s"%(error, [alpha, beta]))

        simulated_data_given_pair= 1- np.array(simulated_data_given_pair)# Contains 'shocked' as obs/success
        total_time= time.time()- t1
        print("Total execution time: %s\n"%total_time)

        return simulated_data_given_pair

    @staticmethod
    def simulate_observations_given_prior_posterior_pairs(original_data, init_obs=None, num_dogs=30, 
                                                        num_trials=24, activation_type= "exp", 
                                                        prior_simulations= None,  prior_offset= 0, **kwargs):
        """
        Input
        -------



        Output
        --------
        """
        simulated_arr_list= []
        flag = "posterior" if prior_simulations is not None else "prior"
        note_text= "Here "
        for idx, content in enumerate(kwargs.items()):
            model_name, prior_samples = content
            print("___________\n\nFor model '%s' %s"%(model_name, flag))
            parameters_pairs = list(zip(*list(prior_samples.values())))
            print("total samples count:", len(parameters_pairs), " sample example: ", parameters_pairs[:2])

            simulated_data_given_pair= base.simulate_observations_given_param(init_obs=init_obs, num_dogs=num_dogs, num_trials=num_trials, 
                                                                        parameter_pair_list= parameters_pairs, activation=activation_type, 
                                                                        prior_offset= prior_offset)#, print_logs=1

            print("Number of datasets/%s pairs generated: %s"%(flag, simulated_data_given_pair.shape[0]))

            simulated_data_given_pair_flattened = np.reshape(simulated_data_given_pair, (-1, 25))

            simulated_arr=np.mean(simulated_data_given_pair_flattened, axis=0)
            print("Shape of data simulated from %s for model '%s' : %s"%(flag, model_name, simulated_arr.shape))
            simulated_arr_list.append(simulated_arr.reshape((1, -1)))
            note_text+="'Dog_%s' corresponds to observations simulated from %s of '%s', "%(idx+1, flag, model_name)

        l_idx = idx+1#last index
        original_data_arr = 1- np.mean(original_data, axis=0)# Contains 'shocked' as obs/success

        if prior_simulations is not None:
            original_plus_simulated_data= np.concatenate(simulated_arr_list)
            original_plus_simulated_data= np.concatenate([original_plus_simulated_data, prior_simulations])
    #         for idx in range(1, len(prior_simulations)+1):
            for idx, content in enumerate(kwargs.items()):
                model_name, _ = content
                note_text+="'Dog_%s' corresponds to observations simulated from prior of '%s', "%(l_idx+idx+1, model_name)
            ylabel_text= 'Probability of shock at trial j [original & simulated priors, posteriros]'
            l_idx+=idx+1
        else:
            simulated_arr_list.append(original_data_arr.reshape((1, -1)))
            original_plus_simulated_data= np.concatenate(simulated_arr_list)
            ylabel_text= 'Probability of shock at trial j [original & simulated priors]'
        print("Respective shape of original data: %s and concatenated arrays of data simulated for different priors/posteriors: %s"%(original_data_arr.shape, original_plus_simulated_data.shape))

        note_text = note_text[:-2]
        note_text+=" & 'Dog_%s' corresponds to 'Original data'."%(l_idx+1)
        print("\n%s\n%s\n%s"%("_"*55, "_"*70, note_text))

        base.plot_original_y(original_plus_simulated_data, ylabel=ylabel_text)

        return original_plus_simulated_data
    
    @staticmethod
    def compute_grubin(param_chains_sample_dict):
        """
        Input
        -------
        param_chains_sample_dict: dictionary with alpha, beta as keys and
                                array of chains of sample parameters values.
                                example: {'alpha': array([[-0.18649854, ..,-0.19441406]]), 
                                            'beta': array([[-0.18322189, ..,-0.19441406]])}

        Output
        -------
        Returns gelman-rubin statistics value.
        """
        grubin_dict= {}
        for param, chain_list in param_chains_sample_dict.items():
            L = float(min(list(map(len, chain_list))))# find minimum of the chain
            num_chains_J = float(len(chain_list))
            chain_mean = np.mean(chain_list, axis=1).reshape((-1,1))# shape (2, 1)

            grand_chain_mean = np.mean(chain_mean)# constant

            B= L*np.reciprocal(num_chains_J-1)*np.sum(np.square(chain_mean-grand_chain_mean))# constant

            Sj_square= np.reciprocal(L-1)*np.sum(np.square(chain_list - chain_mean))# constant

            W= np.mean(Sj_square)

            grubin = round(((L-1)*np.reciprocal(L)*W + np.reciprocal(L)*B)/W, 4)
            grubin_dict[param]= grubin
            print("\nGelmen-rubin for 'param' %s all chains is: %s"%(param, grubin))
        
        return grubin_dict

    
    @staticmethod
    def gelman_rubin_stats(pruned_hmc_sample_chains):
        """
        Input
        -------
        pruned_hmc_sample_chains: dictionary with alpha, beta as keys and
                                array of chains of sample parameters values.
                                example: {'alpha': array([[-0.18649854, ..,-0.19441406]]), 
                                            'beta': array([[-0.18322189, ..,-0.19441406]])}

        Output
        -------
        Returns gelman-rubin statistics value given hmcs samples.
        """
        param_chains_sample_dict_= {}
        param_chain_list_ = list(map(lambda chain: (tuple(pruned_hmc_sample_chains[chain].keys()), chain), 
                                    list(pruned_hmc_sample_chains)))

        def starighten_up(records):
            return list(map(lambda param:(param, records[1]), records[0]))

        param_chain_list= list(itertools.chain.from_iterable(map(starighten_up, param_chain_list_)))

        param_chains_sample_dict= defaultdict(list)
        list(map(lambda val: param_chains_sample_dict[val[0]].append(val[1]), param_chain_list));

        for param, chain_list in param_chains_sample_dict.items():
            param_chain_list=[]
            L = min(list(map(lambda chain: len(pruned_hmc_sample_chains[chain][param]), chain_list)))# find minimum of the chain
            for chain in chain_list:
                samples_arr = pruned_hmc_sample_chains[chain][param]
                param_chain_list.append(samples_arr[:L].reshape((1, -1)))

            param_chains_sample_dict_[param]= np.concatenate(param_chain_list, axis=0)# Contains dict of arrays of alpha/beta chains
        
    #     grubin_dict= compute_grubin(param_chains_sample_dict_)
        grubin_dict= base.compute_grubin(param_chains_sample_dict_)

        return grubin_dict
    
    @staticmethod
    def get_hmc_n_chains(pyromodel, xa, xs, y, num_chains=4, sample_count = 1000, 
                     burnin_percentage = 0.1, thining_percentage =0.9, 
                     alpha_prior= None, beta_prior= None, activation= "exp"):
        """
        Input
        -------
        pyromodel: Pyro model object with specific prior distribution
        xa: tensor holding avoidance count for all dogs & all trials
        xs: tensor holding shock count for all dogs & all trials
        y: tensor holding response observation for all dogs & all trials
        num_chains: Count of MCMC chains to launch, default 4
        sample_count: count of samples expected in a MCMC chains , default 1000
        burnin_percentage:
        thining_percentage: 

        Outputs
        ---------
        hmc_sample_chains: a dictionary with chain names as keys & dictionary of parameter vs sampled values list as values 
        hmc_chain_diagnostics: a dictionary with chain names as keys & dictionary of chain diagnostic metric values from hmc sampling. 

        """
        hmc_sample_chains =defaultdict(dict)
        hmc_chain_diagnostics =defaultdict(dict)

        net_sample_count= round(sample_count/((1- burnin_percentage)*(1- thining_percentage)))

        t1= time.time()
        for idx in range(num_chains):
            num_samples, burnin= net_sample_count, round(net_sample_count*burnin_percentage)
            nuts_kernel = NUTS(pyromodel)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin)
            mcmc.run(xa, xs, y, alpha_prior= alpha_prior, beta_prior= beta_prior, activation= activation)
            hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
            hmc_chain_diagnostics['chain_{}'.format(idx)]= mcmc.diagnostics()

        print("\nTotal time: ", time.time()-t1)
        hmc_sample_chains= dict(hmc_sample_chains)
        hmc_chain_diagnostics= dict(hmc_chain_diagnostics)

        return hmc_sample_chains, hmc_chain_diagnostics

    
    @staticmethod
    def prune_hmc_samples(hmc_sample_chains, thining_dict):
        """
        Input
        -------
        hmc_sample_chains: a dictionary with chain names as keys & dictionary of parameter vs sampled values list as values 
        thining_dict: a dictionary with chain names as keys & dictionary of parameter vs thining factor list as values 
                      example:  {"chain_0": {"alpha":6, "beta":3}, "chain_1": {"alpha":7, "beta":3}}


        Outputs
        ---------
        Outputs a pruned version of hmc_sample_chains in accordance with respective thining factors.

        """
        pruned_hmc_sample_chains= defaultdict(dict)
        for chain, params_dict in hmc_sample_chains.items():
            pruned_hmc_sample_chains[chain]= dict(map(lambda val: (val[0], params_dict[val[0]][0:-1:val[1]]), list(thining_dict[chain].items())))
            
            original_sample_shape_dict= dict(map(lambda val: (val[0], val[1].shape), list(params_dict.items())))
            pruned_sample_shape_dict = dict(map(lambda val: (val[0], val[1].shape), list(pruned_hmc_sample_chains[chain].items())))
            
            print("%s\nOriginal sample counts for '%s' parameters: %s"%("-"*25, chain, original_sample_shape_dict))
            print("\nThining factors for '%s' parameters: %s "%(chain, thining_dict[chain]))
            print("Post thining sample counts for '%s' parameters: %s\n\n"%(chain, pruned_sample_shape_dict))

        pruned_hmc_sample_chains= dict(pruned_hmc_sample_chains)

        return pruned_hmc_sample_chains
    
    @staticmethod
    def get_chain_diagnostics(hmc_chain_diagnostics):
        """
        Input
        -------
        hmc_chain_diagnostics: dictionary holding chain diagnostic metric values from hmc sampling 
                                (ex: {'chain_0': {'alpha': OrderedDict([('n_eff', tensor(320.6277)),('r_hat', tensor(0.9991))]),
                                'beta': OrderedDict([('n_eff', tensor(422.8024)), ('r_hat', tensor(0.9991))]),'divergences': {'chain 0': []},
                                'acceptance rate': {'chain 0': 0.986}}}).
        
        Outputs
        ---------
        pandas dataframe holding hmc chain diagnostic results.
        
        """
        diagnostics_df= pd.DataFrame()

        for chain, diag_di in hmc_chain_diagnostics.items():
            parameters= list(set(diag_di.keys())- {'acceptance rate', 'divergences'})
            diag_params= list(diag_di.get(parameters[0]).keys())

            diag_func = lambda param: (param, list(map(lambda d_param: diag_di[param][d_param].item(), diag_params)))

            diagnostics_dict = dict(map(diag_func, parameters))
            diagnostics_dict.update({"metric": diag_params, "chain":chain, 
                        "acceptance rate":diag_di.get("acceptance rate", {}).get("chain 0")})

            diagnostics_dict_df = pd.DataFrame(diagnostics_dict)
            diagnostics_dict_df["divergences"]= str(diag_di.get("divergences", {}).get("chain 0", []))
        #     diagnostics_dict_df["divergences"]=[diag_di.get("divergences", {}).get("chain 0", []) for i in diagnostics_dict_df.index]
            diagnostics_df = pd.concat([diagnostics_df, diagnostics_dict_df], axis=0)

        diagnostics_df= diagnostics_df.melt(id_vars=["chain", "metric", "acceptance rate", "divergences"], var_name="parameters", value_name="metric_values")
        diagnostics_df.set_index(["parameters", "chain", "metric"], inplace=True)

        return diagnostics_df

    @staticmethod
    def summary_stats_df(beta_chain_matrix_df, key_metrics):
        all_metric_func_map = lambda metric, vals: {"mean":np.mean(vals), "std":np.std(vals), 
                                            "25%":np.quantile(vals, 0.25), 
                                            "50%":np.quantile(vals, 0.50), 
                                            "75%":np.quantile(vals, 0.75)}.get(metric)
        summary_stats_df= pd.DataFrame()
        for metric in key_metrics:
            final_di = {}
            for column in beta_chain_matrix_df.columns:
                params_per_column_di = dict(beta_chain_matrix_df[column].apply(lambda x: all_metric_func_map(metric, x)))
                final_di[column]= params_per_column_di
            metric_df_= pd.DataFrame(final_di)
            metric_df_["parameter"]= metric
            summary_stats_df= pd.concat([summary_stats_df, metric_df_], axis=0)

        summary_stats_df.reset_index(inplace=True)
        summary_stats_df.rename(columns= {"index":"metric"}, inplace=True)
        summary_stats_df.set_index(["parameter", "metric"], inplace=True)

        return summary_stats_df
    
    @staticmethod
    def summary_stats_df_2(fit_df, key_metrics):
        summary_stats_df= pd.DataFrame()
        parameters= list(set(fit_df.columns) - {"chain"})
        for param in parameters:
            for name, groupdf in fit_df.groupby("chain"):
                groupdi = dict(groupdf[param].describe())

        #         values = dict(map(lambda key:(key, [groupdi.get(key)]), ['mean', 'std', '25%', '50%', '75%']))
                values = dict(map(lambda key:(key, [groupdi.get(key)]), key_metrics))

                values.update({"parameter": param, "chain":name})
                summary_stats_df_= pd.DataFrame(values)
                summary_stats_df= pd.concat([summary_stats_df, summary_stats_df_], axis=0)
        summary_stats_df.set_index(["parameter", "chain"], inplace=True)

        return summary_stats_df
    
    @staticmethod
    def plot_parameters_for_n_chains(fit_df, chains=["chain_0"], parameters=["beta0", "beta1", "beta2", "beta3", "sigma"], plotting_cap=[4, 5], plot_interactive=False):
        """
        Input
        --------
        chains: list of valid chain names, example - ["chain_0"].
        
        parameters: list of valid parameters names, example -["beta0", "beta1", "beta2", "beta3", "sigma"].
        
        plotting_cap: list of Cap on number of chains & Cap on number of parameters to plot, example- [4, 5] 
                    means cap the plotting of number of chains upto 4 & number of parameters upto 5 ONLY,
                    If at all the list size for Chains & parameters passed increases.
        
        plot_interactive: Flag for using Plotly if True, else Seaborn plots for False.
        
        
        output
        -------
        Plots box plots for each chain from list of chains with parameters on x axis.
        
        """
        func_all_params_per_chain = lambda param, chain: (param, fit_df[fit_df["chain"]==chain][param].tolist())

        try:
            chain_cap, param_cap = plotting_cap#
            assert len(chains)<=chain_cap, "Cannot plot Number of chains greater than %s!"%chain_cap
            assert len(parameters)<=param_cap, "Cannot plot Number of parameters greater than %s!"%param_cap
            
            for chain in chains:
                di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, chain), parameters))
                df_all_params_per_chain = pd.DataFrame(di_all_params_per_chain)
                if df_all_params_per_chain.empty:
    #                 raise Exception("Invalid chain number in context of model!")
                    print("Note: Chain number [%s] is Invalid in context of this model!"%chain)
                    continue
                if plot_interactive:
                    df_all_params_per_chain= df_all_params_per_chain.unstack().reset_index(level=0)
                    df_all_params_per_chain.rename(columns={"level_0":"parameters", 0:"values"}, inplace=True)
                    fig = px.box(df_all_params_per_chain, x="parameters", y="values")
                    fig.update_layout(height=600, width=900, title_text=f'{chain}')
                    fig.show()
                else:
                    df_all_params_per_chain.plot.box()
                    plt.title(f'{chain}')
        except Exception as error:
            if type(error) is AssertionError:
                print("Note: %s"%error)
                chains = np.random.choice(chains, chain_cap, replace=False)
                parameters=np.random.choice(parameters, param_cap, replace=False)
                plot_parameters_for_n_chains(chains, parameters)
            else: print("Error: %s"%error)

    @staticmethod
    def plot_joint_distribution(fit_df, parameters):
        all_combination_params = list(itertools.combinations(parameters, 2))
        for param_combo in all_combination_params:
            param1, param2= param_combo
            print("\nPyro -- %s"%(f'{param1} Vs. {param2}'))
            sns.jointplot(data=fit_df, x=param1, y=param2, hue= "chain")
            plt.title(f'{param1} Vs. {param2}')
            plt.show()

    
    # ACF plots widgets
    @staticmethod
    def plot_autocorrelation(beta_chain_matrix_df, param, chain_list):
        for chain in chain_list:
            data= beta_chain_matrix_df.loc[param][chain]
            fig = plot_acf(data, title="Sample autocorrelation for '%s' from '%s'"%(param, chain))
            fig.set_figwidth(9)
            fig.set_figheight(5)
            plt.show()


    # ACF plots widgets
    @staticmethod
    def autocorrelation_plots(beta_chain_matrix_df):
        """
        Input:
        -------
        
        Output:
        --------
        
        
        """
        
        parameters= list(beta_chain_matrix_df.index)
        chains = list(beta_chain_matrix_df.columns)

        def radio_but_eventhandler(tick):
            print("Use 'Shift / Ctrl or Cmd' + Arrow keys to select chains")
            select_multiple= widgets.SelectMultiple(options=chains,
                                                    value=[np.random.choice(chains)], description='Select Chains', disabled=False)
            select_multiple_output = widgets.Output()

            def select_multiple_eventhandler(change):
                select_multiple_output.clear_output()
                param = tick.owner.value
                with select_multiple_output:
                    # display(plot_autocorrelation(beta_chain_matrix_df, param, list(change.owner.value)))
                    display(base.plot_autocorrelation(beta_chain_matrix_df, param, list(change.owner.value)))

            clear_output()
            display(radio_but)
            select_multiple.observe(select_multiple_eventhandler, names='value')
            display(select_multiple)
            display(select_multiple_output)

        radio_but= widgets.RadioButtons(options=parameters, description='Parameters:', disabled=False)
        radio_but_output = widgets.Output()

        radio_but.observe(radio_but_eventhandler, names='value')
        display(radio_but)


    #Save & load button widgets
    @staticmethod
    def toggle_status(any_button, flag= 0):
        time.sleep(0.3)
        if flag:
            any_button.icon = "hourglass-half"
            any_button.button_style='warning'
        else:
            any_button.icon = any_button.description.lower()
            any_button.button_style="info"

    @staticmethod
    def save_main(save_button, param_chain_matrix_df, filepath):
        clear_output()
        param_chain_matrix_df.to_csv(filepath, index=False)
        display(save_button)
        base.toggle_status(save_button, 1)
        base.toggle_status(save_button, 0)
        print("Saved at '%s'"%filepath)

    @staticmethod
    def build_save_button():
        save_button = widgets.Button(
            description='Save',
            disabled=False,
            button_style='info',
            tooltip='save',
            icon="save"
        )
        return save_button

    @staticmethod
    def save_parameter_chain_dataframe(param_chain_matrix_df, filepath):
        save_button= base.build_save_button()
        save_func = lambda x: base.save_main(save_button, param_chain_matrix_df, filepath)
        save_button.on_click(save_func)
        display(save_button)

    @staticmethod
    def load_main(load_button):
        clear_output()
        for filename in load_button.value:
            string_representation=str(load_button.value[filename]["content"],'utf-8')
            string_transformed = StringIO(string_representation)
        
            param_chain_matrix_df=pd.read_csv(string_transformed)
    
        base.toggle_status(load_button, 1)
        base.toggle_status(load_button, 0)
        print("Loaded '%s'"%filename)
        return param_chain_matrix_df

    @staticmethod
    def build_upload_button():
        load_button = widgets.FileUpload(accept='.csv', 
                                    button_style='info',
                                    icon="upload",
                                    multiple=False)

        display(load_button)
        return load_button

    @staticmethod
    def load_parameter_chain_dataframe(load_button):
        if load_button.value:
            param_chain_matrix_df= base.load_main(load_button)
        else: 
            param_chain_matrix_df= pd.DataFrame()
        return param_chain_matrix_df

    
    @staticmethod
    def summary(beta_chain_matrix_df, layout = 1):
        reset_slider= lambda x: reset_slider_main()

        def reset_slider_main():
            clear_output()
            slider.observe(slider_eventhandler, names='value')
            display(slider)


        def slider_eventhandler(tick):
            global slide_counter
            if tick.owner.value=="View 1":
                slide_counter+=1
                print("Use 'Shift / Ctrl or Cmd' + Arrow keys to select")
                select_multiple= widgets.SelectMultiple(options=["mean", "std", "25%", "50%", "75%"],
                                                        value=["mean"], description='Summarise', disabled=False)
                select_multiple_output = widgets.Output()

                def select_multiple_eventhandler(change):
                    select_multiple_output.clear_output()
                    with select_multiple_output:
                        if layout!=2:
                            display(base.summary_stats_df(beta_chain_matrix_df, list(change.owner.value)))
                        else: display(base.summary_stats_df_2(beta_chain_matrix_df, list(change.owner.value)))

                select_multiple.observe(select_multiple_eventhandler, names='value')
                display(select_multiple)
                display(select_multiple_output)
            elif tick.owner.value=="View 2":
                slide_counter+=1
                print("Select any value")
                dropdown = widgets.Dropdown(options =["mean", "std", "25%", "50%", "75%", "ALL"])
                dropdown_output = widgets.Output()
                def dropdown_eventhandler(change):
                    dropdown_output.clear_output()
                    with dropdown_output:
                        if (change.new == "ALL"):
                            if layout!=2:
                                display(base.summary_stats_df(beta_chain_matrix_df, ["mean", "std", "25%", "50%", "75%"]))
                            else: display(base.summary_stats_df_2(beta_chain_matrix_df, ["mean", "std", "25%", "50%", "75%"]))

                        else:
                            if layout!=2:
                                display(base.summary_stats_df(beta_chain_matrix_df, [change.new]))#summary_stats_df_[summary_stats_df_ == change.new])
                            else: display(base.summary_stats_df_2(beta_chain_matrix_df, [change.new]))#summary_stats_df_[summary_stats_df_ == change.new])

                dropdown.observe(dropdown_eventhandler, names='value')
                display(dropdown)
                display(dropdown_output)
            else:
                clear_output()
                print("Results cleared!")
                slide_counter=0
                reset_button.icon = "hourglass-half"#"battery-half"#"hourglass-half"#
                reset_button.button_style='warning'
                display(reset_button)
            if slide_counter>3:
                clear_output()
                print("Attempts Exhausted!")
                slide_counter=0
                reset_button.icon = "hourglass-end"#"battery-empty"#"hourglass-end"#
                reset_button.button_style='danger'
                display(reset_button)


        reset_button = widgets.Button(
            description='Show slider',
            disabled=False,
            button_style='warning',
            tooltip='reset',
            icon="hourglass-half"#"battery-half"
        )
        reset_button.on_click(reset_slider)

        slider = widgets.SelectionSlider(
            options=['View 1', 'View 2', 'Clear'],
            value='View 1',
            description='Slide to',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        slider_output = widgets.Output()

        slider.observe(slider_eventhandler, names='value')
        display(slider)

    
    @staticmethod
    def plot_chains(param_chain_matrix_df):
        """
        Input
        -------
        param_chain_matrix_df: Dataframe holding samples of parameters (alpha & beta)
                            with parameter names across rows chain names across columns.
        
        Output
        -------
        Plot intermixing chains for each parameter.

        """
        for param in param_chain_matrix_df.index:
            plt.figure(figsize=(10,8))
            for chain in param_chain_matrix_df.columns:
                plt.plot(param_chain_matrix_df.loc[param, chain], label=chain)
                plt.legend()
            plt.title("Chain intermixing for '%s' samples"%param)
            plt.show()
    
    @staticmethod
    def hexbin_plot(x, y, x_label, y_label):
        """
        
        Input
        -------
        x: Pandas series or list of values to plot on x axis.
        y: Pandas series or list of values to plot on y axis.
        x_label: variable name x label. 
        y_label: variable name x label. 
        
        
        Output
        -------
        Plot Hexbin correlation density plots for given values.
        
        
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        min_x = min(list(x)+list(y)) - 0.1
        max_x = max(list(x)+list(y)) + 0.1
        ax.plot([min_x, max_x], [min_x, max_x])
        
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_x, max_x])
        
        ax.set_title('{} vs. {} correlation scatterplot'.format(x_label, y_label))
        hbin= ax.hexbin(x, y, gridsize=25, mincnt=1, cmap=plt.cm.Reds)
        cb = fig.colorbar(hbin, ax=ax)
        cb.set_label('occurence_density')
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()
    
    @staticmethod
    def plot_interaction_hexbins(fit_df, parameters=["alpha", "beta"]):
        """
        
        Input
        -------
        fit_df: Pandas dataframe containing sampled values across columns with parameter names as column headers
        parameters: List of parameters for which all combination of hexbins are to be plotted, defaults to ["alpha", "beta"]
        
        
        Output
        -------
        Plots hexbin correlation density plots for each pair of parameter combination.
            
        """
        all_combination_params = list(itertools.combinations(parameters, 2))
        for param1, param2 in all_combination_params:#Plots interaction between each of two parameters
            base.hexbin_plot(fit_df[param1], fit_df[param2], param1, param2)
        
    @staticmethod
    def get_obs_y_dict(select_pairs, x_a, x_s):
        """
        
        Input
        -------
        select_pairs: pairs of (alpha, beta) selected
        x_a: array holding avoidance count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
        x_s: array holding shock count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
        
        Output
        -------
        
        Outputs a dictionary with tuple of alpha, beta as key & observerd values of y corresponding to alpha, beta in key
        
        """
        sigmoid = lambda value: 1/(1+np.exp(-value))
        y_dict = {}
        for alpha, beta in select_pairs:# pair of alpha, beta
            y_dict[(alpha, beta)] = sigmoid(alpha*x_a + beta* x_s)
        
        return y_dict

    @staticmethod
    def plot_observed_y_given_parameters(observations_list, selected_pairs_list, observed_y, chain, original_obs= []):
        """
        
        Input
        -------
        observations_list:list of observated 'y' values from simulated 3 trials experiment computed corresponding 
                        to selected pairs of (alpha, beta)
        selected_pairs_list: list of alpha, beta pair tuples, example :  [(-0.225, -0.01272), (-0.21844, -0.01442)]
        
        observed_y: dict holding observed values correspodning to pair of alpha, beta tuple as key, 
                    example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])}
        chain: name of the chain from sampler
        original_obs: original observations/ labels from given data

        returns  plotly scatter plots with number of trials on X axis & corresponding probability of getting
        shocked for each pair of (alpha, beta) passed in 'selected_pairs_list'.
        
        Output
        --------
        Plots scatter plot of all observed values of y corresponding to each given pair of alpha, beta
        
        """
        # obs_column_names = [f'Dog_{ind+1}'for ind in range(dogs_data["Ndogs"])]
        obs_column_names = [f'Dog_{ind+1}'for ind in range(base.load_data()["Ndogs"])]
        
        for record in zip(observations_list, selected_pairs_list):
            sim_y, select_pair = record
            print("\nFor simulated y value: %s & Selected pair: %s"%(sim_y, select_pair))

            obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=obs_column_names)
            if not original_obs is base.plot_observed_y_given_parameters.__defaults__[0]:
                original_obs_column_names = list(map(lambda name:f'*{name}', obs_column_names))
                
                original_obs_df= pd.DataFrame(original_obs.numpy().T, columns=original_obs_column_names)
                obs_y_df= pd.concat([obs_y_df, original_obs_df], axis=1)
                print("Note: Legend *Dog_X corresponds to 'y' i.e.,original observation values")
            
            obs_y_title= "Observed values distribution for all dogs given parameter %s from %s"%(select_pair, chain)
            fig = px.scatter(obs_y_df, title=obs_y_title)
            fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
            fig.show()

    @staticmethod
    def compare_dogs_given_parameters(pairs_to_compare, observed_y, original_obs=[], alpha_by_beta_dict= {}):
        """
        
        Input
        --------
        
        pairs_to_compare: list of alpha, beta pair tuples to compare, 
                        example :  [(-0.225, -0.0127), (-0.218, -0.0144)]
        observed_y: dict holding observed values correspodning to pair of alpha,
                        beta tuple as key, example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])} 
        alpha_by_beta_dict: holds alpha, beta pair tuples as keys & alpha/beta as value, example:
                            {(-0.2010, -0.0018): 107.08}
        
        
        Output
        --------
        returns a plotly scatter plot with number of trials on X axis & corresponding probability of getting
        shocked for each pair of (alpha, beta) passed for comparison.
        
        """
        combined_pairs_obs_df= pd.DataFrame()
        title_txt = ""
        additional_txt = ""
        # obs_column_names = [f'Dog_{ind+1}'for ind in range(dogs_data["Ndogs"])]
        obs_column_names = [f'Dog_{ind+1}'for ind in range(base.load_data()["Ndogs"])]
        for i, select_pair in enumerate(pairs_to_compare):
            i+=1
            title_txt+=f'Dog_X_m_{i} corresponds to {select_pair}, '

            obs_column_names_model_x =list(map(lambda name:f'{name}_m_{i}', obs_column_names))

            if alpha_by_beta_dict:
                additional_txt+=f'𝛼/𝛽 for Dog_X_m_{i} {round(alpha_by_beta_dict.get(select_pair), 2)}, '
            
            obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=obs_column_names_model_x)

            combined_pairs_obs_df= pd.concat([combined_pairs_obs_df, obs_y_df], axis=1)

        print(title_txt)
        print("\n%s"%additional_txt)

        if not original_obs is base.compare_dogs_given_parameters.__defaults__[0]:
            original_obs_column_names = list(map(lambda name:f'*{name}', obs_column_names))

            original_obs_df= pd.DataFrame(original_obs.numpy().T, columns=original_obs_column_names)
            combined_pairs_obs_df= pd.concat([combined_pairs_obs_df, original_obs_df], axis=1)
            print("\nNote: Legend *Dog_X_ corresponds to 'y' i.e.,original observation values")
            
        obs_y_title= "Observed values for all dogs given parameter for a chain"
        fig = px.scatter(combined_pairs_obs_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
        fig.show()

    @staticmethod
    def get_alpha_by_beta_records(chain_df, metrics=["max", "min", "mean"]):
        """
        
        Input
        --------
        chain_df: dataframe holding sampled parameters for a given chain
        
        returns an alpha_by_beta_dictionary with alpha, beta pair tuples as keys & alpha/beta as value,
        example: {(-0.2010, -0.0018): 107.08}
        
        Output
        -------
        Return a dictionary with values corresponding to statistics/metrics asked in argument metrics, computed
        over alpha_by_beta column of passed dataframe.
        
        
        """
        alpha_beta_dict= {}
        
        chain_df["alpha_by_beta"] = chain_df["alpha"]/chain_df["beta"]
        min_max_values = dict(chain_df["alpha_by_beta"].describe())
        alpha_beta_list= list(map(lambda key: chain_df[chain_df["alpha_by_beta"]<=min_max_values.get(key)].sort_values(["alpha_by_beta"]).iloc[[-1]].set_index(["alpha", "beta"])["alpha_by_beta"].to_dict(), metrics))

        [alpha_beta_dict.update(element) for element in alpha_beta_list];
        return alpha_beta_dict

    @staticmethod
    def calculate_deviance_given_param(parameters, x_avoidance, x_shocked, y):
        """

        Input
        -------
        parameters : dictionary containing sampled values of parameters alpha & beta
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25)
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        

        Output
        -------
        Computes deviance as D(Bt)
        D(Bt)   : Summation of log likelihood / conditional probability of output, 
                given param 'Bt' over all the 'n' cases.

        Returns deviance value for a pair for parameters, alpha & beta.
        
        """
        sigmoid = lambda value: 1/(1+torch.exp(-value))
        D_bt_ = []
        p = parameters["alpha"]*x_avoidance + parameters["beta"]*x_shocked# alpha * Xai + beta * Xsi
        #  𝛼𝑋𝑎𝑖 +𝛽 𝑋𝑠𝑖= -log(1/sigmoid(𝛼𝑋𝑎𝑖 +𝛽 𝑋𝑠𝑖)- 1)
        p= -torch.log((1/sigmoid(p)) - 1)
        p=p.double()
        p= torch.where(p<-0.0001, p, -0.0001).float()
        
        Pij_vec = p.flatten().unsqueeze(1)# shapes (750, 1)
        Yij_vec= y.flatten().unsqueeze(0)# shapes (1, 750)
        
        # D_bt = -2 * Summation_over_i-30 (yi.(alpha.Xai + beta.Xsi)+ (1-yi).log (1- e^(alpha.Xai + beta.Xsi)))

        # exp(𝛼𝑋𝑎𝑖 +𝛽 𝑋𝑠𝑖) = sigmoid(𝛼𝑋𝑎𝑖 +𝛽 𝑋𝑠𝑖)

        D_bt= torch.mm(Yij_vec, Pij_vec) + torch.mm(1-Yij_vec, torch.log(1- sigmoid(Pij_vec)))
        D_bt= -2*D_bt.squeeze().item()
        return D_bt

    @staticmethod
    def calculate_mean_deviance(samples, x_avoidance, x_shocked, y):
        """
        
        Input
        -------
        samples : dictionary containing mean of sampled values of parameters alpha & beta.
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25).
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).

        
        Output
        -------
        Computes mean deviance as D(Bt)_bar
        D(Bt)_bar: Average of D(Bt) values calculated for each 
                    Bt (Bt is a single param value from chain of samples)
        Returns mean deviance value for a pair for parameters, alpha & beta.
        
        
        """
        samples_count = list(samples.values())[0].size()[0]
        all_D_Bts= []
        for index in range(samples_count):# pair of alpha, beta
            samples_= dict(map(lambda param: (param, samples.get(param)[index]), samples.keys()))
            
            D_Bt= base.calculate_deviance_given_param(samples_, x_avoidance, x_shocked, y)
            all_D_Bts.append(D_Bt)
        
        D_Bt_mean = torch.mean(torch.tensor(all_D_Bts))
        
        D_Bt_mean =D_Bt_mean.squeeze().item()
        
        return D_Bt_mean

    @staticmethod
    def DIC(sample_chains, x_avoidance, x_shocked, y):
        """
            
        Input
        -------
        sample_chains : dictionary containing multiple chains of sampled values, with chain name as
                        key and sampled values of parameters alpha & beta.
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25).
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
                
        Output
        -------
        Computes DIC as follows
        D_mean_parameters: 𝐷(𝛼_bar,𝛽_bar), Summation of log likelihood / conditional probability of output, 
                    given average of each param 𝛼, 𝛽, over 's' samples, across all the 'n' cases.
        D_Bt_mean: 𝐷(𝛼,𝛽)_bar, Summation of log likelihood / conditional probability of output, 
                    given param 𝛼, 𝛽, across all the 'n' cases.
            
        𝐷𝐼𝐶 is computed as 𝐷𝐼𝐶 = 2 𝐷(𝛼,𝛽)_bar − 𝐷(𝛼_bar,𝛽_bar)
        
        returns Deviance Information Criterion for a chain alpha & beta sampled values.


        """
        dic_list= []
        for chain, samples in sample_chains.items():
            samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors

            mean_parameters = dict(map(lambda param: (param, torch.mean(samples.get(param))), samples.keys()))
            D_mean_parameters = base.calculate_deviance_given_param(mean_parameters, x_avoidance, x_shocked, y)

            D_Bt_mean = base.calculate_mean_deviance(samples, x_avoidance, x_shocked, y)
            dic = round(2* D_Bt_mean - D_mean_parameters,3)
            dic_list.append(dic)
            print(". . .DIC for %s: %s"%(chain, dic))
        print("\n. .Mean Deviance information criterion for all chains: %s\n"%(round(np.mean(dic_list), 3)))

    @staticmethod
    def compare_DICs_given_model(x_avoidance, x_shocked, y, **kwargs):
        """
        
        Input
        --------
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25).
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        kwargs: dict of type {"model_name": sample_chains_dict}
        
        Output
        --------
        Compares Deviance Information Criterion value for a multiple bayesian models.
        
        
        """
        for model_name, sample_chains in kwargs.items():
            print("%s\n\nFor model : %s"%("_"*30, model_name))
            base.DIC(sample_chains, x_avoidance, x_shocked, y)

                    




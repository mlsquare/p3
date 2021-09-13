import numpy as np
import os
import torch
import pyro
import random
import time
import numpy as np
import pandas as pd
import re
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
        num_dogs = base.load_data()["Ndogs"] if original_obs.ndim!=1 else 1
        obs_column_names = [f'Dog_{ind+1}' for ind in range(num_dogs)] if num_dogs!=1 else ["Dogs"]
        obs_y_df= pd.DataFrame(original_obs.T, columns=obs_column_names)

        if ylabel is None:
            ylabel = "Probability of shock at trial j (𝜋𝑗)"

        obs_y_title= "Original observed value distribution for all dogs"
        fig = px.scatter(obs_y_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title=ylabel, legend_title="Dog identifier")
        fig.show()
    


    @staticmethod
    def DogsModel(x_avoidance, x_shocked, y):
        """
        Input
        -------
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25)
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        
        Output
        --------
        Implements pystan model: {
                alpha ~ normal(0.0, 316.2);
                beta  ~ normal(0.0, 316.2);
                for(dog in 1:Ndogs)  
                    for (trial in 2:Ntrials)  
                    y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}
        
        """

        alpha = pyro.sample("alpha", dist.Normal(0., 316.))
        beta = pyro.sample("beta", dist.Normal(0., 316))
        with pyro.plate("data"):
            pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)

    @staticmethod
    def DogsModelUniformPrior(x_avoidance, x_shocked, y):
        """
        Input
        -------
        x_avoidance: tensor holding avoidance count for all dogs & all trials, example for 
                    30 dogs & 25 trials, shaped (30, 25)
        x_shocked:   tensor holding shock count for all dogs & all trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        y:           tensor holding response for all dogs & trials, example for 30 dogs
                    & 25 trials, shaped (30, 25).
        
        Output
        --------
        Implements pystan model: {
                alpha ~ uniform(0.0, 316.2);
                beta  ~ uniform(0.0, 316.2);
                for(dog in 1:Ndogs)  
                    for (trial in 2:Ntrials)  
                    y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));}
        
        """
        alpha = pyro.sample("alpha", dist.Uniform(-10, -0.00001))
        beta = pyro.sample("beta", dist.Uniform(-10, -0.00001))
        with pyro.plate("data"):
            pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)

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
    def get_hmc_n_chains(pyromodel, xa, xs, y, num_chains=4, base_count = 900):
        """
        Input
        -------
        pyromodel: Pyro model object with specific prior distribution
        xa: tensor holding avoidance count for all dogs & all trials
        xs: tensor holding shock count for all dogs & all trials
        y: tensor holding response observation for all dogs & all trials
        num_chains: Count of MCMC chains to launch, default 4
        base_count:Minimum count of samples in a MCMC chains , default 900
        
        Ouputs
        ---------
        hmc_sample_chains: a dictionary with chain names as keys & dictionary of parameter vs sampled values list as values 
        
        """
        hmc_sample_chains =defaultdict(dict)
        possible_samples_list= random.sample(list(np.arange(base_count, base_count+num_chains*100, 50)), num_chains)
        possible_burnin_list= random.sample(list(np.arange(100, 500, 50)), num_chains)

        t1= time.time()
        for idx, val in enumerate(list(zip(possible_samples_list, possible_burnin_list))):
            num_samples, burnin= val[0], val[1]
            nuts_kernel = NUTS(pyromodel)
            mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin)
            mcmc.run(xa, xs, y)
            hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        print("\nTotal time: ", time.time()-t1)
        hmc_sample_chains= dict(hmc_sample_chains)
        return hmc_sample_chains

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
        for param in ["alpha", "beta"]:
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

    @staticmethod
    def summary(beta_chain_matrix_df):
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
                        display(base.summary_stats_df(beta_chain_matrix_df, list(change.owner.value)))

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
                            display(base.summary_stats_df(beta_chain_matrix_df, ["mean", "std", "25%", "50%", "75%"]))
                        else:
                            display(base.summary_stats_df(beta_chain_matrix_df, [change.new]))#summary_stats_df_[summary_stats_df_ == change.new])

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
        y_dict = {}
        for alpha, beta in select_pairs:# pair of alpha, beta
            y_dict[(alpha, beta)] = torch.exp(alpha*x_a + beta* x_s)
        
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

        D_bt_ = []
        p = parameters["alpha"]*x_avoidance + parameters["beta"]*x_shocked# alpha * Xai + beta * Xsi
        p=p.double()
        p= torch.where(p<-0.0001, p, -0.0001).float()
        
        Pij_vec = p.flatten().unsqueeze(1)# shapes (750, 1)
        Yij_vec= y.flatten().unsqueeze(0)# shapes (1, 750)
        
        # D_bt = -2 * Summation_over_i-30 (yi.(alpha.Xai + beta.Xsi)+ (1-yi).log (1- e^(alpha.Xai + beta.Xsi)))
        D_bt= torch.mm(Yij_vec, Pij_vec) + torch.mm(1-Yij_vec, torch.log(1- torch.exp(Pij_vec)))
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

                    




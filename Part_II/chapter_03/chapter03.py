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
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output



class base(object):
    def __init__(self):
        pass

    @staticmethod
    def load_data():
        """
        Input
        -------

        Output
        --------

        """
        transform_data= lambda x:torch.tensor(stats.zscore(x), dtype=torch.float)# standardises Input data
        stacks_data = {"p":3, "N":21, "Y":[43, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 
            7, 8, 8, 9, 15, 15], "x":[80, 80, 75, 62, 62, 62, 62, 62, 59, 58, 58, 58, 58, 
            58, 50, 50, 50, 50, 50, 56, 70, 27, 27, 25, 24, 22, 23, 24, 24, 
            23, 18, 18, 17, 18, 19, 18, 18, 19, 19, 20, 20, 20, 89, 88, 90, 
            87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89, 86, 72, 79, 80, 
            82, 91]}
        
        x_data= torch.tensor(np.array(stacks_data["x"]).reshape((stacks_data["N"], stacks_data["p"])), dtype=torch.float)
        y_data= torch.tensor(stacks_data["Y"], dtype=torch.float)
        stacks_data_ = []
        for idx in range(stacks_data["p"]):
            stacks_data_.append(transform_data(x_data[:,idx]))
        
        stacks_data_.append(y_data)
        return stacks_data_

    @staticmethod
    def StackModel(X1, X2, X3, Y, beta_prior = None, sigma_prior= None, **kwargs):
        beta0 = pyro.sample("beta0", kwargs.get("beta0", beta_prior))
        beta1 = pyro.sample("beta1", kwargs.get("beta1", beta_prior))
        beta2 = pyro.sample("beta2", kwargs.get("beta2", beta_prior))
        beta3 = pyro.sample("beta3", kwargs.get("beta3", beta_prior))
        sigma = pyro.sample("sigma", sigma_prior)
        sigma = torch.sqrt(sigma)

        mu = beta0 + beta1 * X1 + beta2 * X2 + beta3*X3
        with pyro.plate("data", len(X1)):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)
    
    @staticmethod
    def get_hmc_n_chains(pyromodel, x1, x2, x3, y, num_chains=4, sample_count = 1000, 
                     burnin_percentage = 0.1, thining_percentage =0.9, 
                     beta_prior = None, sigma_prior= None, **kwargs):
        """
        Input
        -------
        pyromodel: Pyro model object with specific prior distribution
        x1: tensor holding 
        x2: tensor holding 
        x3: tensor holding 
        y: tensor holding response observation 
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
            mcmc.run(x1, x2, x3, y, beta_prior= beta_prior, sigma_prior= sigma_prior, **kwargs)
            hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
            hmc_chain_diagnostics['chain_{}'.format(idx)]= mcmc.diagnostics()

        print("\nTotal time: ", time.time()-t1)
        hmc_sample_chains= dict(hmc_sample_chains)
        hmc_chain_diagnostics= dict(hmc_chain_diagnostics)

        return hmc_sample_chains, hmc_chain_diagnostics

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
    def plot_autocorrelation(beta_chain_matrix_df, parameters=None, chains= None, msize=8, lags= 40, plot_pacf=False):
        lags= int(lags)
        chains_list= chains if chains else list(beta_chain_matrix_df.columns)
        parameters_list = parameters if parameters else list(beta_chain_matrix_df.index)

        for param in parameters_list:
            print("Autocorrelation for '%s'"%param)
            fig= make_subplots()
            for chain in chains_list:#["chain_0", "chain_1"]:
    #             plot_pacf= False
                beta_chain_df_slice= pd.DataFrame(beta_chain_matrix_df.loc[param][chain])
                corr_array = pacf(beta_chain_df_slice.dropna(), alpha=0.05, fft=False, nlags= lags) if plot_pacf else acf(beta_chain_df_slice.dropna(), alpha=0.05, fft=False, nlags= lags)

                lower_y = corr_array[1][:,0] - corr_array[0]
                upper_y = corr_array[1][:,1] - corr_array[0]

                r, g, b= random.sample(range(0, 255), 3)
                for x in range(len(corr_array[0])):
                    fig.add_trace(go.Scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='rgba(%s,%s,%s,0.9)'%(r, g,b), showlegend=False))

            #     fig.add_trace(go.Scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='rgba(%s,%s,%s,0.9)'%(r, g,b), name=chain))# To add legend for even line

                fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='rgba(%s, %s,%s,0.8)'%(r, g,b),
                                marker_size=msize, name=chain))
                fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(%s, %s,%s,0)'%(r, g,b), showlegend=False))
                fig.add_trace(go.Scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(%s,%s,%s,0.2)'%(r, g,b),
                            fill='tonexty', line_color='rgba(255,255,255,0)', name=chain))
            fig.update_layout(title="ACF plot for '%s' (all chains)\n\n"%param, legend_title="Chains")
            fig.show()


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
    def plot_joint_distribution(fit_df, parameters):
        all_combination_params = list(itertools.combinations(parameters, 2))
        for param_combo in all_combination_params:
            param1, param2= param_combo
            print("\nPyro -- %s"%(f'{param1} Vs. {param2}'))
            sns.jointplot(data=fit_df, x=param1, y=param2, hue= "chain")
            plt.title(f'{param1} Vs. {param2}')
            plt.show()

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
    def plot_interaction_contours(fit_df, parameters=None, 
                              chain="chain_0", plot_3d=False):
        pair_count= 2
        if plot_3d:
            pair_count= 3
        
        chain_samples_df= fit_df[fit_df["chain"]==chain].copy()# chain is 'chain_3' 
        parameters= parameters if parameters else list(set(chain_samples_df.columns)- {"chain"})
        
        all_combination_params= list(itertools.combinations(parameters, pair_count))
        for param_combination in all_combination_params:
            param_combination= list(param_combination)
            print("\nFor parameters: '%s' & '%s'"%(param_combination, chain))
            chain_samples_df_slice= chain_samples_df[param_combination].values
            if plot_3d:
                name1, name2, name3 = param_combination
                fig = go.Figure(data=[go.Surface(z=chain_samples_df_slice)])
                fig.update_layout(title="3D Contour plot for '%s'"%(str(param_combination)), autosize=False,
                            xaxis_title="x (%s)"%name1, yaxis_title="y (%s)"%name2,
                            width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
            else:
                param1, param2 = chain_samples_df_slice.T
                name1, name2 = param_combination
                colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
                fig = ff.create_2d_density(param1, param2, colorscale=colorscale, hist_color='rgb(255, 255, 150)', 
                    point_size=4)
                fig.update_layout(title= "%s-%s joint density plot for model"%(name1, name2), autosize=False,
                                xaxis_title="x (%s)"%name1, yaxis_title="y (%s)"%name2)

            fig.show()
    
    @staticmethod
    def plot_original_y(original_obs,ylabel=None):
        """

        Input
        -------
        original_obs: original observations/ labels from given data

        returns  plotly scatter plots with Nth day observation on X axis & corresponding stack loss
        for each pair of (beta0, beta1, beta2, beta3, sigma) passed in 'selected_pairs_list'.

        Output
        --------
        Plots scatter plot of all observed values of y corresponding to each given pair of 
        beta0, beta1, beta2, beta3, sigma

        """
        num_obs = original_obs.shape[0] if original_obs.ndim!=1 else 1
        obs_column_names = [f'Observation_{ind+1}' for ind in range(num_obs)] if num_obs!=1 else ["Observations"]
        obs_y_df= pd.DataFrame(original_obs.T, columns=obs_column_names)

        if ylabel is None:
            ylabel = "Stack loss"

        obs_y_title= "stack-loss distribution"
        fig = px.scatter(obs_y_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="N-th Day", yaxis_title=ylabel, legend_title="Observation palette")
        fig.show()

    @staticmethod
    def plot_y_violin(obs_y, original_obs, ylabel=None, prior_simulations=None):
        """

        Input
        -------
        obs_y: observations / labels 'Y' from given data.
        original_obs: original ovbservation for given data.

        returns  plotly violin plots with Nth day observation on X axis & corresponding stack loss
        for each pair of (beta0, beta1, beta2, beta3, sigma) passed in 'selected_pairs_list'.

        Output
        --------
        Plots violin plot of all observed values of y corresponding to each given pair of 
        beta0, beta1, beta2, beta3, sigma

        """
        flag = "posterior" if prior_simulations is not None else "prior"
        if ylabel is None:
            ylabel = "Stack loss [original & simulated values]"
        obs_y_with_original_data_df = pd.DataFrame()

        obs_y_df= pd.DataFrame(obs_y)
        obs_y_df= obs_y_df.unstack().reset_index(level=0)
        obs_y_df.rename(columns={"level_0":"parameters", 0:"values"}, inplace=True)
        
        note_text= "'%s_day_X' corresponds to observations simulated from %s of the given model, "%(flag, flag)
        if prior_simulations is not None:#Df with original Y, prior & posterior
            ylabel= 'Stack loss [original, simulated priors & posteriros]'
            note_text+="'prior_day_X' corresponds to observations simulated from prior of the given model"
            for param, groupdf in obs_y_df.groupby("parameters"):
                index = param+1
                prior_obs_df = pd.DataFrame({"parameters": "prior_day_%s"%(index), "values":prior_simulations[:, param]})
                groupdf["parameters"] = "posterior_day_%s"%(index)
                groupdf= groupdf.append({"parameters":"day_%s *"%(index), "values":original_obs[param].item()}, ignore_index=True)
                obs_y_with_original_data_df = pd.concat([obs_y_with_original_data_df, prior_obs_df, groupdf])
        else:#Df with Only original Y & posterior
            for param, groupdf in obs_y_df.groupby("parameters"):
                index = param+1
                groupdf["parameters"] = "day_%s"%(index)
                groupdf= groupdf.append({"parameters":"day_%s *"%(index), "values":Y[param].item()}, ignore_index=True)
                obs_y_with_original_data_df = pd.concat([obs_y_with_original_data_df, groupdf])
        
        note_text+=", 'day_X *' corresponds to the original observations for N-th day."
        print("\n%s\n%s\n%s"%("_"*55, "_"*70, note_text))
        obs_y_title= "stack-loss distribution"
        # fig = px.box(obs_y_with_original_data_df, x="parameters", y="values")
        fig = px.violin(obs_y_with_original_data_df, x="parameters", y="values", box=True, # draw box plot inside the violin
                        points='all', # can be 'outliers', or False
                    )
        fig.update_layout(title=obs_y_title, height=600, width=900, xaxis_title="N-th Day observation", 
                        yaxis_title=ylabel, legend_title="Observation palette")
        fig.show()

    return obs_y_with_original_data_df


    @staticmethod
    def simulate_observations_given_param(parameter_pair_list= [(72.2, 293.2, 506, 17.92, 8.5)]):
        """
        Input
        -------



        Output
        --------
        """
        t1= time.time()
        simulated_data_given_pair = []
        
        for parameters in parameter_pair_list:
            beta0, beta1, beta2, beta3, sigma= parameters

            mu= beta0 + beta1 * X1 + beta2 * X2 + beta3 * X3# (21,)
            simulated_y= torch.normal(mu, np.sqrt(sigma))

            simulated_data_given_pair.append(simulated_y.numpy().reshape((1,-1)))

        total_time= time.time()- t1
        print("Total execution time: %s\n"%total_time)
        
        simulated_data_arr= np.concatenate(simulated_data_given_pair, axis=0)
        simulated_data_arr= np.clip(simulated_data_arr, -10, 10)
        return simulated_data_arr

    @staticmethod
    def init_priors(prior_dict={"default":dist.Normal(0., 316.)}):
        """
        Input
        -------



        Output
        --------
        """
        prior_dict["names"]= prior_dict.get("names") if prior_dict.get("names") else ["beta0", "beta1", "beta2", "beta3", "sigma"]
        if not prior_dict.get("default"):
            raise Exception("pass a deafault distribution to key 'default' for parameters ['beta0', 'beta1', 'beta2', 'beta3']}")
        prior_list = []
        [prior_list.append(prior_dict.get(param, prior_dict.get("default"))) for param in prior_dict.get("names")];
        return prior_list

    @staticmethod
    def get_prior_samples(num_samples=1100, **kwargs):
        """
        Input
        -------



        Output
        --------
        """
        prior_samples={}
        for param, param_prior in kwargs.items():
            sample_list = list(map(lambda k: pyro.sample(param, param_prior).item(), range(num_samples)))
            prior_samples[param]=sample_list
        return prior_samples
    
    @staticmethod
    def simulate_observations_given_prior_posterior_pairs(original_data, prior_simulations= None, **kwargs):
        """
        Input
        -------



        Output
        --------
        """
        flag = "posterior" if prior_simulations is not None else "prior"
        for idx, content in enumerate(kwargs.items()):
            model_name, samples_dict = content
            print("___________\n\nFor model '%s' %s"%(model_name, flag))
            parameters_pairs = list(zip(*list(samples_dict.values())))
            print("total samples count:", len(parameters_pairs), " sample example: ", parameters_pairs[:2])
            simulated_arr= base.simulate_observations_given_param(parameters_pairs)# shaped (1100, 21)
            
        
        # obs_y_with_original_data_df = plot_y_violin(simulated_arr, original_data, prior_simulations = prior_simulations)
        obs_y_with_original_data_df = base.plot_y_violin(simulated_arr, original_data, prior_simulations = prior_simulations)

        return simulated_arr, obs_y_with_original_data_df

    @staticmethod
    def pyro_pystan_chain_intermixing(fit_df_A, fit_df_B):
        sns.set_style("darkgrid")

        all_params_chains_df = pd.DataFrame()
        parameters = list(set(fit_df_A.columns)- {"chain"})# All params names
        chains_pyro_pystan = list(fit_df_A["chain"].unique())# All chain_names

        for param in parameters:
            fig= plt.figure(figsize=(10,10))
            all_chains_df= pd.DataFrame()
            print("\nFor pyro_parameter: %s & pystan_parameter: %s"%(param, param))
            for chain in chains_pyro_pystan:
                all_chains_df_a = pd.DataFrame({"chain":"pyro_%s"%chain,
                                            param:fit_df_A[fit_df_A["chain"]==chain][param].tolist()})
                plt.plot(fit_df_A[fit_df_A["chain"]==chain][param], label="pyro_%s"%chain)

                all_chains_df_b = pd.DataFrame({"chain":"pystan_%s"%chain,
                                            param:fit_df_B[fit_df_B["chain"]==chain][param].tolist()})
                plt.plot(fit_df_B[fit_df_B["chain"]==chain][param], label="pystan_%s"%chain)# Plots pystan chain
                all_chains_df =pd.concat([all_chains_df, all_chains_df_a, all_chains_df_b], axis=0)
            all_params_chains_df= pd.concat([all_params_chains_df, all_chains_df], axis=1)

            plt.legend()
            plt.title("Sampled values for %s"%(param))
            plt.show()

        return all_params_chains_df

    @staticmethod
    def pyro_pystan_param_histograms(parameter= "beta0", **kwargs):
        fit_A_B_combined= pd.DataFrame()
        for method, fit_df in kwargs.items():
            fit_df["method"] = method
            fit_A_B_combined= pd.concat([fit_A_B_combined, fit_df])
        
        fit_A_B_combined= fit_A_B_combined[["chain", parameter, "method"]]
        fig = px.histogram(fit_A_B_combined, x=parameter, color="method", marginal="rug", # can be `box`, `violin`
                            hover_data=fit_A_B_combined.columns)
        fig.update_layout(title_text="Histogram comparison between %s for parameter '%s'"%(list(kwargs.keys()), parameter))
        fig.show()


        
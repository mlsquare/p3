#!/usr/bin/env python
# coding: utf-8

# ## Foreword

# ### Objective:
# 
# - Make Bayesian Analysis tools accessible to ML Practitioners.
# - Make Scalable computing tools accessible to Statisticians.
# 
# ### Gaps in ML Science
# Problems with Machine Learning and Deep Learning:
# 
# Majority of both classical and modern ML and DL methods rarely venture into asking the “why” and “so what” questions. These paradigms are obsessed with achieving prediction accuracy, and no other aspect of modeling is considered -- such as robustness, uncertainty quantification, interpretability, explainability, low data paradigms.
# 
# There is a strong and urgent need for Holistic Machine Learning, where all the aspects are given the right importance, instead of obsessively focussing on achieving state-of-the-art performances. Of course, the situation is rapidly changing. But, the field is being reinvented. Many known concepts that were established in 1950s-60s are simply being resurrected, albeit with new marketing terms, and most importantly, not recognizing the work done by some of the brightest minds of the bygone era. One of the goals in assembling these data journals is to make those ideas and techniques to the ML community, which for some reason singularly focused on loss functions without looking at their connections to statistics.
#  
# ### Gaps in Bayesian Tooling
# 
# There are three main ingredients to doing Bayesian Analysis 1) data 2) models 3) fit. For the frequentist counterpart, usually Analysts will spend time iterating between (1) and (2) and use routine commercial software for (3). However, a Bayesian Analyst often needs to implement even the fitting mechanism (MCMC sampler, for example). If something is not working, the Analyst is fraught between figuring out what is wrong among the three 1) Does the model a wrong one?  Model is OK, but the technical derivation of the sampling is wrong or finally, is the problem just too hard for the sampler. Debugging and testing is incredibly hard due to these confounding factors.
# 
# Fortunately, with tools like winBUGS, and most recently, Stan, have made fitting Bayesian models very declarative. General purpose samplers take care of approximating the posterior draws. Nevertheless, if something fails, like in the case of winBUGs, there is very little the Analyst could do. There is no control. Also, with the exception of Stan, perhaps, these tools do not scale well for the modern world. The Deep Learning community has made tremendous contributions on the technology side with frameworks like PyTorch, TensorFlow etc, which can handle data and models in sizes not comparable. Therefore, Bayesians, and Statisticians need to embrace these modern technologies, and build their science on the top.
# 
# 
# ### Approach
# 
# Alright we have vented out the frustration. What is the way out. Majority of the books/ teaching material on the Statistics side are concept-centric. Even though many books have companion codes, datasets, and notebooks, they are still concept centric. Use the data or the model to make a point selectively, without the context. 
# 
# #### Data-first, not Model-first:
# In this compendium, we take a dataset first view. We are given a dataset, ask ourselves, what are the objectives of the experiment? What modeling is required? 
# Andrew Ng[1], talks about Data-centric, not Model-centric AI, albeit in a different context.
# 
# 
# 
# #### Case-based, not Concept-based:
# Instead of cherry picking a technique first, and datafirst, we pick the dataset, and solve. And most importantly, solve in an end-to-end fashion. As a result, the flow may not be linear, and the reader is taken through the journey of discovering and fixing problems. 
# Both Profs. Gelman and Betancourt et talk about principled ways to approach a problem, particularly from a Bayesian stand-point [2,3]
# 
# #### Exploratory, not confirmatory
# Due to legacy pedagogy, every problem has an answer or needs to have an answer. Sometimes, problems are created with an end in sight. Not with a data-first, case-based approach.  However, It is not necessary that a problem be solved in the usual sense. It can end up in a negative result. The conclusion could be, the scientific question can not be answered with the experimental data available, or better tools/ approaches are required or the model is plain junk.
# 
# Premier ML conferences like NuerIPS are encouraging pre-registration to beat confirmation bias. This article in science  and encourage [4,5]
# 
# #### Build-up, not ground-up
# There are many datasets available such as UCI, WinBUGs, along with codes, which were discussed in various formats (books, blogs, etc..). Take those datasets, and translate the models and analysis, where possible, into one of the many Probabilistic Programming tools. The benefit is that, those familiar in one domain, can relate to the translated counterparts
# 
# #### Learn-by-Doing, not Learn-by-Reading
# Data Science is a Learn-by-Doing type of discipline. Traditional format, decoupled the doing from the learning part. By placing the code, data, models, analysis -- all side-by-side, the reader can learn by executing the notebooks. Where possible, we use dynamic visualizations to allow interaction with data. This is largely missing with  traditional publishing material.
# 
# <Br>
# [1] Andrew Ng: MLOPs: From Model-centric to Data-centric AI
# https://www.youtube.com/watch?v=06-AZXmwHjo 
# <Br>
# [2] Gelman et al, Bayesian workflow
# https://arxiv.org/abs/2011.01808 
# <Br>
# [3] Schad, D.J, Betancourt, M,. Vasisth, S, Towards a principled Bayesian workflow in cognitive science, 2020, 
# https://arxiv.org/abs/1904.12765 
# <Br>
# [4] https://preregister.science/ 
# <Br>
# [5] Alison Ledgerwood, The preregistration revolution needs to distinguish between predictions and analyses, PNAS, November 6, 2018 115 (45) E10516-E10517; first published October 19, 2018
# https://www.pnas.org/content/115/45/E10516 
# <Br>
# 
# ### Output
# 
# A collection 50+ of case studies, covering various domains, concepts, techniques, testing ideas
# 
# ### Organization
# #### Part-1: Foundations:
# All the prereqs needed will be covered such as Probability Theory, Estimation Theory..  Lot of material is out there on the internet.
# 
# #### Part-2: Small-Scale Bayesian Analysis:
# Majority of the classical Bayesian analysis will be covered here. We hope to devot this part to analyzing datasets available in winBUGs, Stan, and many open sources. 
# 
# #### Part-3: Large-Scale Bayesian Analysis:
# Here we will make a transition from small data sets and tiny interpretable models, to big data and bigger models. Bayesian Deep Learning is one such area.
# 
# #### Part-4: Advanced Methods
# Advanced topics such as Bayesian nonparametrics, Bayesian networks, Graphical models, will be covered. Any new models for problems covered in Part-2 also go here.
# 
# #### Part-5: Bayesian counterparts to ML Classics
# Introduction to Elements of Statistical Learning by Gareth James et is popular for ML enthusiasts learning Statistical Theory. We can translate the examples/ notebooks into Python, as well as develop Bayesian analogues. 
# 
# ##### Part-6: Contributed Chapters
# Anybody that is interested in sharing their Bayesian Analysis (more than a blog) are welcome.
# 
# 
# ### Evolution & Contribution:
# 
# Publishing books or scholarly works is no longer the same. Writing books is now more like writing software. Books can be incrementally developed, they can be versioned, and can be made accessible in a wide variety of formats. 
# 
# We embrace this continuous delivery and continuous improvement aspect of agile software development and apply it to writing this book. As a result, material will evolve, without ever having to wait for the perfect, near-complete material. It will evolve in quality and quantity, with time.
# 

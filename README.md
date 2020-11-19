# Flu-Shot-Learning

## Collaborators

Camila Mattos dos Santos, Eduardo Yuki Yada, Marco Antonio Occhialini Filho, Giovanna Vendeiro Vilar, Renato Yoshio Eguti.

## Objective 

This project is the final assignment for the Statistical Learning course held at University of São Paulo and ministered by Florencia Leonardi. The purpose of this assignment is to put the techniques covered in the discipline into practice in a real world problem. Given the global pandemic situation we chose a competition regarding vaccination. 

Vaccination is a key public health measure used to fight infectious diseases. As of November 2020, vaccines for the COVID-19 virus are still not available. Therefore, this competition aims to revisit the public health response to another recent respiratory disease pandemic caused by the H1N1 influenza virus. A survey was conducted in the United States asking respondents whether they had received the H1N1 and seasonal flu vaccines, in conjunction with questions about themselves. For more details check out the [competition link](https://www.drivendata.org/competitions/66/flu-shot-learning/page/210/). The goal then is to predict how likely individuals are to receive their H1N1 and seasonal flu vaccines.

## Methodology

This is a multilabel binary classification problem as we have two labels (`h1n1_vaccine` and `seasonal_vaccine`). The dataset contains 35 features and all of them are categorical (all features are described [here](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/). Some of these features are nominal (e.g. `race`) and some are ordinal (e.g. `opinion_h1n1_vacc_effective` scaled from 1 to 5). The performance is evaluated according to the average AUC regarding the two target variables. 

An Exploratory Data Analysis was done to begin investigating the problem. With a better grasp of what the data looks like, we fitted a several different models. On the one hand models with great interpretability might be more suitable to understand what features are most impactful in the model given the objective of revisiting the public health response. But on the other hand the competition only evaluates the AUC, therefore it is also important to choose the model with best predictive power.  

## Conclusion

**WIP**
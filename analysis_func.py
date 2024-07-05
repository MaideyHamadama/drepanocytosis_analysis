# Import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import statistics
import seaborn as sns
from math import *
from matplotlib import pyplot as plt
from scipy.stats import iqr,kstest,shapiro,chi2_contingency
from numpy.random import seed
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Charging analysis file
def load_data():
    """ Charge the data source for analysis"""
    data_file = "data.xlsx"
    df = pd.read_excel(data_file)
    return df

# Load cleaned data
def load_cleaned_data():
    """ Load cleaned data """
    file = "data_cleaned.xlsx"
    df_cleaned = pd.read_excel(file)
    return df_cleaned

# Scatter plot diagram
def scatter_plot(df, ind_var, dep_var, title, var3 = ""):
    """ Used to visualize relations between two continuous variables
    - Points on the graph show how a variable change in relation to another variable
    - Coloring the points in function of a third variable can help in the visualization of sub groups
    - Generaly x axis is the independent variable and y axis the dependent variable"""
    plt.figure(figsize=(10, 6))
    if var3 != "":
        plt.scatter(df[ind_var], df[dep_var], c=df[var3], cmap='viridis', label=ind_var + ' vs ' + dep_var)
    else:
        plt.scatter(df[ind_var], df[dep_var], label=ind_var + ' vs ' + dep_var)
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    plt.title(title)
    if var3 != "":
        plt.legend([var3])
        plt.colorbar(label = var3)
    plt.show()
    
# Histogram diagram
def histogram(df, var, title, color='blue'):
    """ Used to visualize the distribution of one continuous variable 
    - Show observations frequency in different classes
    - Useful to identify the form of distribution (normal, asymetric, etc...) and outliners"""
    plt.figure()
    plt.hist(df[var], bins=10, alpha=0.7, color=color, label=var + " Distribution")
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()
    
# Box plots
def box_plot(df, main_var, secondary_var):
    """ Used to visualize distribution of one continuous variable and identify outliers in the dataset
    - It is used to summarize the distribution of a continuous variable
    - Display the mean, quartiles and outliers
    - Compare the box plots between groups (for e.g Vaccinated and non vaccinated) helps identify the differences in distributions"""
    plt.figure(figsize=(10,6))
    df.boxplot(column=main_var, by=secondary_var, grid=False)
    plt.xlabel(secondary_var)
    plt.ylabel(main_var)
    plt.title("Box plot : " + main_var + " by " + secondary_var)
    plt.suptitle('')
    plt.show()
    
# Pair plot
def seaborn(df, key_var):
    """ Used to visualize relations by pairs in an entire dataframe """
    sns.pairplot(df, hue=key_var)
    plt.show()
    
# Heatmap plot
def heatmap(df):
    """ Used to visualize correlations between continuous variables"""
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap of correlations')
    plt.show()

# Density plot
def density_plot(df, var):
    """ Used to estimate the probability distribution of a continuous variable"""
    plt.figure(figsize=(10,6))
    sns.kdeplot(df[var], fill=True)
    plt.xlabel(var)
    plt.title('Density Plot: ' + var)
    plt.show()

# Regression plot
def regression_plot(df, ind_var, dep_var):
    """ Diplay a regression line on a scatter graph. It helps in visualizing the linear trend between variables """
    plt.figure(figsize=(10,6))
    sns.regplot(x=ind_var, y=dep_var, data=df)
    plt.xlabel(ind_var)
    plt.ylabel(dep_var)
    plt.title('Regression Plot : ' + ind_var + ' vs ' + dep_var + '')
    plt.show()
    
# Categorical Scatter plot
def cat_scatter_plot(df, cat_var, cont_var, var3):
    """ Used to visualize how a continuous variable is related to a categorical variable.
    Sometimes with colored categories or different forms"""
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=cont_var, y=cat_var, hue=var3, data=df)
    plt.xlabel(cont_var)
    plt.ylabel(cat_var)
    plt.title('Scatter Plot with ' + var3)
    plt.show()

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (vif)
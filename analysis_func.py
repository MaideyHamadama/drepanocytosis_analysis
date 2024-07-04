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
    
# Creating autocpt arguments
def autocpt_func(pct, allvalues):
    """
    Creating autocpt arguments
    """
    absolute = int(pct/100.*np.sum(allvalues))
    return "{:.1f}%".format(pct,absolute)

# Creating camembert, section plot
def sector_plot(dataset, title):
    """
    Creating camembert or section plots
    """
    data = list(dataset.values())
    labels = list(dataset.keys())
    
    # Figure size
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Creating plot
    wedges, texts, autotexts = ax.pie(data,
                                autopct = lambda pct: autocpt_func(pct, data),
                                labels = labels)
    
    # Setting title to plot
    ax.set_title(title)
    
    # Show plot
    plt.show()

# Creating bar plot
def bar_plot(dataset, title):
    """
    Creating bar plot
    """
    data = list(dataset.values())
    total = sum(data)
    labels = list(dataset.keys())
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Horizontal bar plot
    ax.barh(labels, data)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
        
    # Remove x,y ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Adding padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad= 10)
    
    # Add x,y gridlines
    ax.grid(color = 'grey', linestyle = '-.', linewidth = 0.5, alpha = 0.2)
    
    # Show top values
    ax.invert_yaxis()
    
    # Add annotations to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round(i.get_width(),2)) + "(" + str(round(((i.get_width()/total)*100), 2)) + "%)", 
                fontsize = 10, fontweight = 'bold', color = 'grey')
        
    # Add Plot Title
    ax.set_title(title, loc = 'center', )
        
    # Show plot
    plt.show()

# Creating two bars plot
def bibar_plot(dataset):
    """
    Creating two bars plot
    """
    barwidth = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))
    data = list(dataset.values())
    labels = list(dataset.keys())
    
    # Set position of bar on X axis
    br1 = np.arange(len(data))
    br2 = [x+barwidth for x in br1]
    br3 = [x+barwidth for x in br2]
    
    # Make the plot
    splot = plt.bar(br1, data[0], color='r', width=barwidth, edgecolor='grey', label=labels[0])
    plt.bar(br2, data[1], color='g', width=barwidth, edgecolor='grey', label=labels[1])
    plt.bar(br3, data[2], color='y', width=barwidth, edgecolor='grey', label=labels[2])
    
    # Annotate bars 
    # for p in splot.patches: 
    #     splot.annotate(format(p.get_height()), 
    #                (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                ha='center', va='center', 
    #                xytext=(0, 9), 
    #                textcoords='offset points') 
 
    for i in ax.patches:
        plt.text(i.get_x() + barwidth/2, i.get_height() + 2, i.get_height(), fontsize=15, fontweight='bold', color='black')
        
    # Adding Xticks
    plt.xlabel('Différentes activités d\'engagement communautaire', fontweight='bold', fontsize=15)
    plt.ylabel('Effectif', fontweight='bold', fontsize=15)
    plt.xticks([r+barwidth for r in range(len(data[0]))], ['Pris part à une activité de lutte contre une épidémie\n(COVID-19, Choléra, MPox)', 'Participation aux activités de santé généralement' ,'Vacciner contre la COVID-19'])
    plt.legend()
    plt.show()

# Deaggregate single column multiple choice 
def deaggregation_mcq(series, df_index, question, response_list):
    """
    Deaggregate single column multiple choice question into several columns of single responses
    """
    elements_df = series.to_frame()
    for ind in df_index:       
        if type(series[ind]) == float:
            for i in range (0, len(response_list)):
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "0")
                else:
                    elements_df[question+str(i+1)][ind] = "0"
            continue
        for i in range (0, len(response_list)):
            if response_list[i] in series[ind]:
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "1")
                else:
                    elements_df[question+str(i+1)][ind] = "1"
            else:
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "0")
                else:
                    elements_df[question+str(i+1)][ind] = "0"
    writer = pd.ExcelWriter(os.getcwd()+"output"+question+".xlsx")
    elements_df.to_excel(writer)
    writer._save()

# Deaggregate communication tool and channel 
def deaggregation_comm_supp(series, df_index, question, comm_tool_list):
    """
    Deaggregate communication tool and channel into several columns
    """
    elements_df = series.to_frame()
    for ind in df_index:       
        if type(series[ind]) == float:
            for i in range (0, len(comm_tool_list)):
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "0")
                else:
                    elements_df[question+str(i+1)][ind] = "0"
            continue
        for i in range (0, len(comm_tool_list)):
            if comm_tool_list[i] in series[ind]:
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "1")
                else:
                    elements_df[question+str(i+1)][ind] = "1"
            else:
                if question+str(i+1) not in elements_df:
                    elements_df.insert(0+i, question+str(i+1), "0")
                else:
                    elements_df[question+str(i+1)][ind] = "0"
    writer = pd.ExcelWriter(os.getcwd()+"output"+question+".xlsx")
    elements_df.to_excel(writer)
    writer._save()
    return elements_df
    
# Calculate score of mcq and scq
def score_cal(score_serie, combination):
    """
    Calculate score of mcq and scq.
    First paramenter is the score serie having all scores of a mcq or scq.
    Second parameter is the right combination of the mcq or scq.
    """
    individual_score = []
    count = 0
    for element in score_serie:
        for i in range(0, len(combination)):
            if element[i] == combination[i] and element[i] == "1":
                count += 1
        individual_score.append(count)
        count = 0
    return individual_score

# Returns the desciption of a serie of discrete and numerical values
def data_description(serie):
    """
    Returns the desciption of a serie of discrete and numerical values
    """
    mean = serie.mean()
    std = serie.std()
    min = serie.min()
    max = serie.max()
    q25 = serie.quantile(q=0.25)
    q75 = serie.quantile(q=0.75)
    median = serie.median()
    int_qr = iqr(serie, rng=(25,75),interpolation='midpoint')

    return {"mean" : mean, "std" : std, "min" : min, "max" : max, "25 %" : q25, "75 %" : q75, "median" : median, "iqr" : int_qr}

# Draw the normal distribution curve of a serie
def draw_normal_distribution(serie):
    """
    Draw the normal distribution curve of a serie passed in parameter
    """
    description = data_description(serie)
    pdf = stats.norm.pdf(serie.sort_values(), description['mean'], description['std'])
    plt.plot(serie.sort_values(), pdf)
    plt.xlim([description['min'], description['max']])
    plt.xlabel('Title', size=12)
    plt.ylabel('Frequency', size=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()

# Draw the normal distribution curve of a serie with unequal mean,mode and median
def draw_anormal_distribution(serie):
    """
        Draw the normal distribution curve of a serie with unequal mean, mode and median passed in parameter
    """
    description = data_description(serie)
    pdf = stats.norm.pdf(serie.sort_values(), description['median'], description['iqr'])
    plt.plot(serie.sort_values(), pdf)
    plt.xlim([description['min'], description['max']])
    plt.xlabel('Title', size=12)
    plt.ylabel('Frequency', size=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.show()

# Kolmogorov_smirnov_test using scipy module
def kolmogorov_smirnov_test(dataset):
    seed(1)
    print(kstest(dataset, 'norm'))

# Other version of kolmogorov_smirnov_test using mathematical logic and functions
def kolmogorov_smirnov_test_2(df):
    n = len(df)
    df = df.sort_values(by="final_scores")
    df['Frequency'] = df['final_scores'].map(df['final_scores'].value_counts())
    df = df.drop_duplicates()
    df['count'] = np.arange(1, len(df)+1)
    df['observed relative cumulative frequency'] = df['count'] / len(df)
    normal = statistics.NormalDist(data_description(df['final_scores'])['mean'], data_description(df['final_scores'])['std'])
    df['expected relative cumulative frequency'] = df['final_scores'].apply(lambda x: normal.cdf(x))
    df['difference'] = abs(df['observed relative cumulative frequency']-df['expected relative cumulative frequency'])
    dn = max(df['difference'])
    cv = 1.35810/sqrt(n)
    print("The value of Dn is ", dn)
    if (dn <= cv):
        print("The data fits with a normal distribution")
    else:
        print("The data DO NOT fits with a normal distribution")
    
def shapirot_wilk_test(dataset):
    seed(1)
    print(shapiro(dataset))
    
def crosstable1(df, varx, vary):
    # Get the name of variables (columns)
    colx = varx.name
    coly = vary.name
    # Get the modalities and total frequencies of variables
    resx = varx.value_counts()
    resy = vary.value_counts()
    resx_modalities = sorted(resx.index.to_list())
    resy_modalities = sorted(resy.index.to_list())
    one_dim = []
    two_dim = []
    print(resy_modalities)
    for x in resx_modalities:
        print(x)
        for y in resy_modalities:
            one_dim.append(sum((df[colx] == x) & (df[coly] == y)))
        two_dim.append(one_dim)
        one_dim = []
    
    return two_dim

def crosstable1_adj(df, varx, vary, mod):
    size = len(df)
    # Get the name of variables (columns)
    colx = varx.name
    coly = vary.name
    resy = vary.value_counts()
    resy_modalities = sorted(resy.index.to_list())
    one_dim = []
    two_dim = []
    for y in resy_modalities:
        one_dim.append(sum((df[colx] == mod) & (df[coly] == y)))
    two_dim.append(one_dim)
    one_dim = []
    for y in resy_modalities:
        one_dim.append(sum((df[colx] != mod) & (df[coly] == y)))
    two_dim.append(one_dim)
    return two_dim

def crosstable2(df, varx, vary):
    # Get the name of variables (columns)
    colx = varx.name
    coly = vary.name
    # Get the modalities and total frequencies of variables
    resx = varx.value_counts()
    resy = vary.value_counts()
    resx_modalities = [1, 0]
    resy_modalities = sorted(resy.index.to_list())
    resx_sum = resx.to_list()
    resy_sum = resx.to_list()
    one_dim = []
    two_dim = []
    print(resy_modalities)
    print(resx_modalities)
    for x in resx_modalities:
        for y in resy_modalities:
            one_dim.append(sum((df[colx] == x) & (df[coly] == y)))
        two_dim.append(one_dim)
        one_dim = []
    return two_dim

def translate(dict_to_translate, terms):
    french_dict = dict(zip(terms, list(dict_to_translate.values())))
    french_dict = sorted(french_dict.items(), key=lambda x:x[1], reverse=True)
    french_dict = dict(french_dict)
    return french_dict

def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (vif)
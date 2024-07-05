from analysis_func import *

def run():
    # Charging analysis file
    df = load_cleaned_data()

    # Filter key variables
    df = df[['CODE', 'AGE', 'SEXE', 'NIVEAUSCOLAIRE', 'STATUTMARITAL', 'ETHNIE', 'AGE_DE_DIAGNOSTIC', 'DECEDES', 'VACCINSAJOUR',
             'AGEAUDECES']]
    
    # Preliminary exploration of dataset
    
    # Summarize the statistic analysis
    # print(df.describe())

    # Draw a scatter plot to visualize the distribution of 2 continuous variables (diagnostic age and death age)
    # scatter_plot(df, 'AGE_DE_DIAGNOSTIC', 'AGEAUDECES', "Age at diagnostic vs Age at death", var3="VACCINSAJOUR")
        
    # Draw a box plot to visualize the distribution of one variable (death age) and identify outliers
    # box_plot(df, 'AGEAUDECES', 'VACCINSAJOUR')
    
    # Draw a pair plot
    # seaborn(df, 'VACCINSAJOUR')
    
    # Draw a Heatmap plot
    # heatmap(df.drop(columns='ETHNIE'))
    
    # Univariate analysis
    
    # Descriptive analysis
    # print("Skewness:\n ", df.drop(columns='ETHNIE').skew())
    # print("\nKurtosis:\n ", df.drop(columns='ETHNIE').kurtosis())
    
    # Draw a histogram to visualize the distribution of a continuous variable like age
    # histogram(df, 'AGE', "Histogram: Age distribution")
    
    # Draw a density plot to estimate the probility distribution of continuous variables like age and death age
    # density_plot(df, 'AGE')
    # density_plot(df, 'AGEAUDECES')

    # Bivariate analysis
    
    # Coefficient of Correlation
    # Coefficient of correlation of Pearson
    pearson_corr = df[['AGE', 'AGEAUDECES']].corr(method='pearson')
    print("Pearson correlation:\n", pearson_corr)
    df = df.drop(columns='AGEAUDECES')
    
    # Regression plot
    # Draw a regression plot to visualize linear trend between marital status and death age cause the heatmap had shown a slight relation between these variables
    # regression_plot(df, 'AGEAUDECES', 'STATUTMARITAL')
    # regression_plot(df, 'AGE', 'AGEAUDECES')
    
if __name__ == '__main__':
    run()


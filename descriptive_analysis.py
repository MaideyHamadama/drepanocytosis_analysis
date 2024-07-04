from analysis_func import *

def run():
    # Charging analysis file
    df = load_cleaned_data()

    # Filter key variables
    df = df[['CODE', 'AGE', 'SEXE', 'NIVEAUSCOLAIRE', 'STATUTMARITAL', 'ETHNIE', 'AGE_DE_DIAGNOSTIC', 'DECEDES', 'VACCINSAJOUR',
             'AGEAUDECES']]
    
    # Preliminary exploration of dataset
    
    # Summarize the descriptive analysis
    print(df.describe())

    # Draw a scatter plot to visualize the distribution of 2 continuous variables
    scatter_plot(df, 'AGE_DE_DIAGNOSTIC', 'AGEAUDECES', "Age at diagnostic vs Age at death", var3="VACCINSAJOUR")
    
    # Draw a histogram to visualize the distribution of a continuous variable
    histogram(df, 'AGE', "Histogram: Age distribution")
    
    # Draw a box plot to visualize the distribution of one variable and identify outliers
    box_plot(df, 'AGEAUDECES', 'VACCINSAJOUR')
    
    # Draw a pair plot
    seaborn(df, 'VACCINSAJOUR')
    
    # Draw a Heatmap plot
    heatmap(df.drop(columns='ETHNIE'))
    
    # Count Male and Female sex in the dataset
    # sex_count = dict(df['sex'].value_counts())
    # print(sex_count)
    
    # Evaluate distribution of age variable

    # print(data_description(df['age']))
    # age_groups_df = pd.DataFrame({"age_groups" : age_groups(df['age'])})
    # print(age_groups_df.value_counts())

    # Drawing normal distribution

    # draw_normal_distribution(df['age'])

    # Drawing anormal distribution

    # draw_anormal_distribution(df['age'])

    # Count educational level in the dataset
    # edu_level_count = dict(df['study_level'].value_counts())
    # print(edu_level_count)
    
    # Count religion in the dataset
    # religion_count = dict(df['religion'].value_counts())

    # Count region in the dataset
    # region_count = dict(df['sociodemo/region'].value_counts())

    # Count marital status in the dataset
    # marital_status_count = dict(df['marital_status'].value_counts())
        
    # Creating plot
    # creating_plot(sex_count)
    # creating_plot(edu_level_count)
    # creating_plot(religion_count)
    # creating_plot(region_count)
    # creating_plot(marital_status_count)

if __name__ == '__main__':
    run()


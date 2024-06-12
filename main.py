import requests
import bs4
from bs4 import BeautifulSoup
import discord
import os
from pycoingecko import CoinGeckoAPI
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from newsapi import NewsApiClient
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Variable for Discord Token
discord_token = os.environ['discord_token']

#S&P Price
snp_link = requests.get('https://finance.yahoo.com/quote/%5EGSPC/')

soup = bs4.BeautifulSoup(snp_link.text, "html.parser")

text = soup.find_all('div',
                     {'class': 'container svelte-mgkamr'})[0].find('span')


#Function to get stock prices
def get_stock_price(symbol):
    try:
        link = requests.get(f'https://finance.yahoo.com/quote/{symbol}')
        soup = bs4.BeautifulSoup(link.text, "html.parser")
        text = soup.find_all(
            'div', {'class': 'container svelte-mgkamr'})[0].find('span')
        return text.text
    except Exception as e:
        print(f"Error fetching stock price for {symbol}: {e}")
        return None


#Weather
weather_link = requests.get(
    'https://weather.com/ms-MY/weather/today/l/MYXX0008:1:MY')
weather_soup = bs4.BeautifulSoup(weather_link.text, "html.parser")
weather_text = weather_soup.find_all(
    'div', {'class': 'CurrentConditions--primary--2DOqs'})[0].find('span')


#Quotes
def get_random_quotes(url):
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()  # Assuming the response is JSON
        return data[0]['q']
    else:
        return f"Error:{response.status_code}"


#CoinGecko
cg = CoinGeckoAPI()

#News API
news_api_key = os.environ['news_api_key']
newsapi = NewsApiClient(api_key=news_api_key)


#Malaysia
def get_malaysia_news(country):
    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(language='en', country=country)

    return top_headlines['articles'][0]['url']


#Specific
def get_specific_news(topic):
    # /v2/top-headlines
    top_headlines = newsapi.get_everything(q=topic)

    return top_headlines['articles'][0]['url']


#CSV Files


#Write
def write_csv(data):
    with open('main.csv', 'a', newline='') as csvfile:
        list = []
        list.append(data)
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(list)
    return 'Written!'


#Read
def read_csv():
    data = []
    with open('main.csv', 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
    return '\n'.join(sublist[0] for sublist in data)


#Analyse
def analyse_csv():
    df = pd.read_csv('ratings.csv', names=['Place', 'Votes'])

    max_votes_index = df['Votes'].idxmax()
    highest_voted_place = df.loc[max_votes_index]
    place = highest_voted_place['Place']
    votes = highest_voted_place['Votes']
    return f'The place with the highest votes is {place} with {votes} votes'


#Show list
def show_list():
    df = pd.read_csv('vote.csv')
    return '\n'.join(df['Restaurant'].unique())


#Update
def update_votes(restaurant):
    df = pd.read_csv("vote.csv")
    df['Votes'] = pd.to_numeric(df['Votes'],
                                errors='coerce').fillna(0).astype(int)
    df.loc[df['Restaurant'] == restaurant, 'Votes'] += 1
    df.to_csv("vote.csv", index=False)
    return df.to_string(index=False)


def show_votes():
    df = pd.read_csv("vote.csv")
    return df.to_string(index=False)


#Project
def get_data():
    URL_DATA_POPULATION = 'https://storage.dosm.gov.my/population/population_state.parquet'
    URL_DATA_INCOME = 'https://storage.dosm.gov.my/hies/hies_state.parquet'
    URL_DATA_PERCENTILE = 'https://storage.dosm.gov.my/hies/hies_state_percentile.parquet'

    population_df = pd.read_parquet(URL_DATA_POPULATION)
    income_df = pd.read_parquet(URL_DATA_INCOME)
    percentile_df = pd.read_parquet(URL_DATA_PERCENTILE)

    return population_df, income_df, percentile_df


#Data Formatting
def date_formatting(df):
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    df = df[df['date'].dt.year == 2022]
    return df


# Define a function to assign tier based on percentile
def assign_tier(percentile):
    if percentile <= 40:
        return 'B-40'
    elif percentile <= 80:
        return 'M-40'
    else:
        return 'T-20'


# Function to map ethnicity
def map_ethnicity(ethnicity):
    # Group mappings
    group_mapping = {
        'bumi': ['bumi', 'bumi_malay', 'bumi_other'],
        'others': ['other', 'other_citizen', 'other_noncitizen'],
        'chinese': ['chinese', 'chinese_other'],
        'indian': ['indian']
    }

    for group, values in group_mapping.items():
        if ethnicity in values:
            return group
    return None


# Function to map age group
def map_age_group(age):

    age_groups = {
        '0-19': ['0-4', '5-9', '10-14', '15-19'],
        '20-39': ['20-24', '25-29', '30-34', '35-39'],
        '40-59': ['40-44', '45-49', '50-54', '55-59'],
        '60+': ['60-64', '65-69', '70-74', '75-79', '80-84', '85+']
    }
    for group, values in age_groups.items():
        if age in values:
            return group
    return None


def table_formatting(percentile_df, population_df, income_df):
    # Filter the DataFrame by the 'variable' column to only consider 'minimum' and 'maximum'
    percentile_df = percentile_df[percentile_df['variable'].isin(
        ['minimum', 'maximum'])]

    percentile_df = percentile_df[(percentile_df['percentile'] >= 2)
                                  & (percentile_df['percentile'] <= 99)]

    # Apply the function to create the 'tier' column
    percentile_df['tier'] = percentile_df['percentile'].apply(assign_tier)

    # Group by state and tier, and calculate the minimum and maximum incomes separately
    min_B40 = percentile_df[percentile_df['tier'] == 'B-40'].groupby(
        'state')['income'].min()
    max_B40 = percentile_df[percentile_df['tier'] == 'B-40'].groupby(
        'state')['income'].max()
    min_M40 = percentile_df[percentile_df['tier'] == 'M-40'].groupby(
        'state')['income'].min()
    max_M40 = percentile_df[percentile_df['tier'] == 'M-40'].groupby(
        'state')['income'].max()
    min_T20 = percentile_df[percentile_df['tier'] == 'T-20'].groupby(
        'state')['income'].min()
    max_T20 = percentile_df[percentile_df['tier'] == 'T-20'].groupby(
        'state')['income'].max()

    # Merge the results into a single DataFrame
    group_percentile_df = pd.DataFrame({
        'state': min_B40.index,
        'min_B40': min_B40.values,
        'max_B40': max_B40.values,
        'min_M40': min_M40.values,
        'max_M40': max_M40.values,
        'min_T20': min_T20.values,
        'max_T20': max_T20.values
    })

    df_filtered = population_df[~((population_df['sex'] == 'both') |
                                  (population_df['age'] == 'overall') |
                                  (population_df['ethnicity'] == 'overall'))]

    # Apply mapping functions to create new columns
    df_filtered['ethnicity_group'] = df_filtered['ethnicity'].apply(
        map_ethnicity)
    df_filtered['age_group'] = df_filtered['age'].apply(map_age_group)

    # Group by 'state', 'sex', 'ethnicity_group', and 'age_group', and sum up the 'population' column
    df_summed = df_filtered.groupby(
        ['state', 'sex', 'ethnicity_group',
         'age_group'])['population'].sum().reset_index()

    # Calculate total population by state
    total_population = df_summed.groupby(
        'state')['population'].sum().reset_index()

    # Filter DataFrame for 'bumi' ethnicity and '20-39' age group
    bumi_df = df_summed[df_summed['ethnicity_group'] == 'bumi']
    age_20_39_df = df_summed[df_summed['age_group'] == '20-39']

    # Calculate 'bumi' penetration by state
    bumi_penetration = bumi_df.groupby('state')['population'].sum(
    ) / total_population.set_index('state')['population']
    bumi_penetration = bumi_penetration.reset_index(name='bumi_penetration')

    # Calculate '20-39' age group penetration by state
    age_20_39_penetration = age_20_39_df.groupby('state')['population'].sum(
    ) / total_population.set_index('state')['population']
    age_20_39_penetration = age_20_39_penetration.reset_index(
        name='age_20_39_penetration')

    # Merge the calculated values into a single DataFrame
    result_df = pd.merge(total_population, bumi_penetration, on='state')
    result_df = pd.merge(result_df, age_20_39_penetration, on='state')

    # Rename columns
    group_population_df = result_df.rename(
        columns={'population': 'total_population'})

    #Drop Date
    income_df.drop(columns=['date'])

    # Merge df1, df2, and df3 based on the 'key' column, performing a left join
    merged_df = pd.merge(group_percentile_df,
                         group_population_df,
                         on='state',
                         how='left')

    merged_df = pd.merge(merged_df, income_df, on='state', how='left')
    return merged_df


def correlation(df):
    # Normalizing the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)

    # Renaming columns
    df.rename(columns={'min_B40': 'Min B40',
                       'max_B40': 'Max B40',
                       'min_M40': 'Min M40',
                        'max_M40': 'Max M40',
                       'min_T20': 'Min T20',
                        'max_T20': 'Max T20',
                       'total_population': 'Population',
                          'bumi_penetration': 'Bumi %',
                          'age_20_39_penetration': '20-39 %',
                           'income_mean': 'Income Mean',
                          'income_median': 'Income Median',
                           'expenditure_mean': 'Expenditure',
                       'gini' : 'Gini Index',
                       'poverty' : 'Poverty %'
                      }, inplace=True)

    
    # Creating a DataFrame from the normalized data
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

    # Calculating correlation matrix
    correlation_matrix = normalized_df.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig('correlation.png')

def table_display(merged_df):
    #Cut off df here if you wanna do correlation/stats regression

    # Create B40, M40, and T20 range columns
    merged_df['B40 Range'] = '$' + merged_df['min_B40'].astype(
        str) + ' - ' + '$' + merged_df['max_B40'].astype(str)
    merged_df['M40 Range'] = '$' + merged_df['min_M40'].astype(
        str) + ' - ' + '$' + merged_df['max_M40'].astype(str)
    merged_df['T20 Range'] = '$' + merged_df['min_T20'].astype(
        str) + ' - ' + '$' + merged_df['max_T20'].astype(str)

    # Drop the min/max columns, income_mean, date, and poverty columns
    columns_to_drop = [
        'min_B40', 'max_B40', 'min_M40', 'max_M40', 'min_T20', 'max_T20',
        'income_mean', 'date', 'poverty'
    ]
    merged_df.drop(columns=columns_to_drop, inplace=True)

    merged_df['bumi_penetration'] = (merged_df['bumi_penetration'] *
                                     100).round(0).astype(int)
    merged_df['age_20_39_penetration'] = (merged_df['age_20_39_penetration'] *
                                          100).round(0).astype(int)

    merged_df['total_population'] = merged_df['total_population'].apply(
        lambda x: '{:,.0f}'.format(x))
    merged_df['income_median'] = merged_df['income_median'].apply(
        lambda x: '{:,.0f}'.format(x))
    merged_df['expenditure_mean'] = merged_df['expenditure_mean'].apply(
        lambda x: '{:,.0f}'.format(x))

    merged_df['gini'] = (merged_df['gini']).round(2)

    # Rename columns
    merged_df.rename(columns={
        'bumi_penetration': 'Bumi Penetration(%)',
        'age_20_39_penetration': 'Age 20-39 (%)',
        'state': 'State',
        'total_population': "Population ('000)",
        'income_median': 'Median Income',
        'expenditure_mean': 'Average Spending',
        'gini': 'Gini Index'
    },
                     inplace=True)

    #Function to get population charts
    sns.set_style("darkgrid")
    sns.set_palette(["#004c6d", "#007aa0", "#00add1", "#00e2ff"])

    # Create a table plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=merged_df.values,
                     colLabels=merged_df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=['skyblue'] * len(merged_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(6)

    # Adjust layout
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig('dataframe_image.png')


def income_charts(state, income_df):
    state_data = income_df[income_df['state'] == state]

    # Bar chart
    # Bar chart
    plt.figure(figsize=(8, 6))
    categories = ['Income Mean', 'Expenditure Mean']
    income_mean = state_data['income_mean'].values[0]
    expenditure_mean = state_data['expenditure_mean'].values[0]
    values = [income_mean, expenditure_mean]
    colors = ['#004c6d', '#00e2ff']  # Different colors for each bar
    plt.bar(categories, values, color=colors)
    plt.xlabel('Category')
    plt.ylabel('Amount (in MYR)')
    plt.title(f'Income and Expenditure Comparison in {state}',
              fontsize=16,
              pad=20)
    plt.tight_layout()
    plt.savefig('incomevsexpenditure.png')

    # Pie chart
    plt.figure(figsize=(8, 6))
    plt.pie([
        state_data['poverty'].values[0], 100 - state_data['poverty'].values[0]
    ],
            labels=['Poverty', ''],
            autopct='%1.0f%%',
            startangle=100,
            textprops={
                'fontsize': 12,
                'color': 'white',
            })
    plt.axis('equal')
    plt.title(f'Poverty Distribution in {state}', fontsize=16, pad=20)
    plt.savefig('poverty.png')


def population_charts(state, population_df):
    # Set Seaborn style
    state_data = population_df[population_df['state'] == state]
    # Pie chart for age groups

    # Filter data for sex = 'both' and age != 'overall'
    filtered_df = state_data[(state_data['sex'] == 'both')
                             & (state_data['age'] != 'overall') &
                             (state_data['ethnicity'] != 'overall')]

    filtered_df = pd.DataFrame(filtered_df)

    # Group by ethnicity and sum up population
    grouped_df_ethicity = filtered_df.groupby(
        'ethnicity')['population'].sum().reset_index()

    # Define a mapping dictionary for grouping
    group_mapping = {
        'bumi': ['bumi', 'bumi_malay', 'bumi_other'],
        'others': ['other', 'other_citizen', 'other_noncitizen'],
        'chinese': ['chinese', 'chinese_other'],
        'indian': ['indian']
    }

    # Group population by new higher-level age groups
    grouped_data = {}
    for group, ages in group_mapping.items():
        grouped_data[group] = grouped_df_ethicity[
            grouped_df_ethicity['ethnicity'].isin(ages)]['population'].sum()

    # Create a new DataFrame with the grouped data
    grouped_df_ethicity = pd.DataFrame(list(grouped_data.items()),
                                       columns=['ethnicity', 'population'])

    grouped_df_age = filtered_df.groupby(
        'age')['population'].sum().reset_index()

    age_groups = {
        '0-19': ['0-4', '5-9', '10-14', '15-19'],
        '20-39': ['20-24', '25-29', '30-34', '35-39'],
        '40-59': ['40-44', '45-49', '50-54', '55-59'],
        '60+': ['60-64', '65-69', '70-74', '75-79', '80-84', '85+']
    }

    # Group population by new higher-level age groups
    grouped_data = {}
    for group, ages in age_groups.items():
        grouped_data[group] = grouped_df_age[grouped_df_age['age'].isin(
            ages)]['population'].sum()

    # Create a new DataFrame with the grouped data
    grouped_df_age = pd.DataFrame(list(grouped_data.items()),
                                  columns=['Age Group', 'Population'])

    plt.figure(figsize=(8, 6))
    plt.pie(grouped_df_ethicity['population'],
            labels=None,
            autopct='%1.0f%%',
            startangle=90,
            shadow=False,
            textprops={
                'fontsize': 12,
                'color': 'white',
            })
    plt.title('Ethnicity Distribution', fontsize=16,
              pad=20)  # Increase the distance between the title and the plot
    plt.legend(labels=grouped_df_ethicity['ethnicity'],
               loc='upper right',
               fontsize=10)  # Add a legend outside the plot
    plt.axis(
        'equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('ethicity_group.png')

    # Pie chart for sex
    plt.figure(figsize=(8, 6))
    plt.pie(grouped_df_age['Population'],
            labels=None,
            autopct='%1.0f%%',
            startangle=90,
            shadow=False,
            textprops={
                'fontsize': 12,
                'color': 'white',
            })
    plt.title('Age Distribution', fontsize=16,
              pad=20)  # Increase the distance between the title and the plot
    plt.legend(labels=grouped_df_age['Age Group'],
               loc='upper right',
               fontsize=10)  # Add a legend outside the plot
    plt.axis(
        'equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('age_group.png')


def cluster(merged_df):

    test_df = merged_df.copy()
    test_df.drop(columns=['state', 'date','max_B40', 'min_M40', 'max_M40', 'min_T20','income_mean','gini','age_20_39_penetration'], inplace=True)
    # correlation(merged_df)
    # Renaming columns
    test_df.rename(columns={'min_B40': 'Min B40',
                       'max_B40': 'Max B40',
                       'min_M40': 'Min M40',
                        'max_M40': 'Max M40',
                       'min_T20': 'Min T20',
                        'max_T20': 'Max T20',
                       'total_population': 'Population',
                          'bumi_penetration': 'Bumi %',
                          'age_20_39_penetration': '20-39 %',
                           'income_mean': 'Income Mean',
                          'income_median': 'Income Median',
                           'expenditure_mean': 'Expenditure',
                       'gini' : 'Gini Index',
                       'poverty' : 'Poverty %'
                      }, inplace=True)
    # Step 1: Scale Your Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(test_df) 
    # # Step 2: Elbow Method
    # inertia_values = []
    # silhouette_scores = []
    # max_clusters = 10  # Maximum number of clusters to try

    # for k in range(2, max_clusters + 1):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(scaled_data)
    #     inertia_values.append(kmeans.inertia_)
    #     silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

    # Based on the Elbow Method and Silhouette Score, choose the optimal number of clusters
    optimal_num_clusters = 3 # Adjust this based on the plots above

    # Step 3: Apply KMeans Clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    kmeans.fit(scaled_data)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Assign cluster labels to DataFrame
    test_df['Cluster'] = cluster_labels
    # Step 4: Analyze Cluster Profiles
    # Group the DataFrame by the 'Cluster' column and calculate the mean for each cluster

    # Multiply 'Bumi' column by 100
    test_df['Bumi %'] *= 100

    # Round all columns except 'Poverty %' to 0 decimals
    columns_to_round = test_df.columns.difference(['Poverty %'])
    test_df[columns_to_round] = test_df[columns_to_round].round(decimals=0)
    
    cluster_profiles = test_df.groupby('Cluster').mean()

    cluster_profiles.reset_index(inplace=True)
    cluster_profiles = cluster_profiles.sort_values(by='Cluster')
    cluster_profiles = cluster_profiles.round(decimals=1)
    cluster_profiles.to_csv('cluster_profiles.csv', index=False)

    # Retrieve the 'state' column from one of the original DataFrames
    state_column = merged_df['state']  # Assuming 'state' column exists in population_df



    # Concatenate 'state' column with test_df
    final_df = pd.concat([state_column, test_df], axis=1)
    final_df = final_df.sort_values(by='Cluster')
    final_df[columns_to_round] = final_df[columns_to_round].round(decimals=0)

    # Save final_df to CSV
    final_df.to_csv('final_df_with_clusters.csv', index=False)

    #Function to get population charts
    sns.set_style("darkgrid")
    sns.set_palette(["#004c6d", "#007aa0", "#00add1", "#00e2ff"])

    # Create a table plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cluster_profiles.values,
                     colLabels=cluster_profiles.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=['skyblue'] * len(cluster_profiles.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(6)

    # Adjust layout
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig('cluster_profiles.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=final_df.values,
                     colLabels=final_df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=['skyblue'] * len(final_df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(6)

    # Adjust layout
    plt.tight_layout()
    # Save the plot as an image
    plt.savefig('final_df_with_clusters.png')



#States list for command later
states_list = '''
List of states: 
1. Johor
2. Kedah
3. Kelantan
4. Melaka
5. Negeri Sembilan
6. Pahang
7. Perak
8. Perlis
9. Pulau Pinang
10. Sabah
11. Sarawak
12. Terengganu
13. Selangor
14. W.P. Kuala Lumpur
15. W.P. Labuan
16. W.P. Putrajaya
'''

#Discord Functions
intents = discord.Intents.default()
intents.message_content = True
# Create a new Discord client
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.reply('Hello!')

    elif message.content.startswith('$greet'):
        await message.reply(f'Hello {message.author.mention}!')

    elif message.content.startswith('$echo'):
        await message.reply(f'{message.content[1:]}')

    elif message.content.startswith('pedro'):
        await message.reply('https://imgur.com/gallery/pedro-loop-a1SXTx2')

    elif message.content.startswith('$s&p'):
        await message.reply(f'The S&P Price is {text.text}')

    elif message.content.startswith('$stock'):
        stock_symbol = message.content[6:].upper().strip()
        if get_stock_price(stock_symbol) == None:
            await message.reply('There is no such stock available')
        else:
            await message.reply(
                f'The {stock_symbol} Price is {get_stock_price(stock_symbol)}')

    elif message.content.startswith('$weather'):
        await message.reply(
            f'The Temperature at your area is: {weather_text.text}')

    elif message.content.startswith('$quote'):
        await message.channel.send(
            get_random_quotes('https://zenquotes.io/api/random'))

    elif message.content.startswith('$bitcoin'):
        # Convert the dictionary keys into a list
        dictionary = cg.get_price(ids='bitcoin', vs_currencies='usd')
        keys_list = list(dictionary.keys())
        coin_name = keys_list[0]
        # Access the inner dictionary using the outer key
        inner_dict = dictionary[coin_name]
        # Access the value using an index
        bitcoin_usd = list(inner_dict.values())[0]
        await message.channel.send(
            f"Master, the price of bitcoin is as of now is ${bitcoin_usd}")

    elif message.content.startswith('$crypto'):
        crypto_symbol = message.content[8:].lower().strip()
        # Convert the dictionary keys into a list
        dictionary = cg.get_price(ids=crypto_symbol, vs_currencies='usd')
        keys_list = list(dictionary.keys())
        coin_name = keys_list[0]
        # Access the inner dictionary using the outer key
        inner_dict = dictionary[coin_name]
        # Access the value using an index
        value_usd = list(inner_dict.values())[0]
        await message.channel.send(
            f"Master, the price of {crypto_symbol} is as of now is ${value_usd}"
        )

    elif message.content.startswith('$malaysia_news'):
        await message.channel.send(get_malaysia_news('my'))

    elif message.content.startswith('$news'):
        content = message.content[6:].lower().strip()
        await message.channel.send(get_specific_news(content))

    elif message.content.startswith('$write'):
        content = message.content[7:].lower().strip()
        await message.channel.send(write_csv(content))

    elif message.content.startswith('$read'):
        await message.channel.send(read_csv())

    elif message.content.startswith('$analyse'):
        await message.channel.send(analyse_csv())

    elif message.content.startswith('$vote_list'):
        await message.channel.send(show_list())

    elif message.content.startswith('$vote'):
        restaurant = message.content[6:].strip()
        await message.channel.send(update_votes(restaurant))

    elif message.content.startswith('$show_list'):
        await message.channel.send(show_votes())

    elif message.content.startswith('$malaysia_news'):
        await message.channel.send(get_malaysia_news('my'))

    elif message.content.startswith('$news'):
        content = message.content[6:].lower().strip()
        await message.channel.send(get_specific_news(content))

    elif message.content.startswith('$command'):
        await message.channel.send('''
        List of bot commands: 
        $hello - Say hello to the bot
        $greet - Greet the bot
        $echo - Repeat the user's message
        $quote - Get a random quote
        $weather - Get the current weathe
        $malaysia_news - Get the latest malaysian news
        $news [user input] - Get the latest news
        $bitcoin - Get the current price of bitcoin
        $crypro [user input] - Get the current price of cryptocurrency (must be crypto name)
        $s&p - Get the current price of S&P 500
        $stock [user input] - Get the current price of stock
        $read - Read the csv file
        $write [user input] - Write to the csv file
        $analyse - Analyse the ratings csv file
        $vote_list - Get the list of restaurants that have been voted
        $vote [user input] - Vote for a restaurant
        $current_vote - Get the current vote
        $state - get State list you can get demography of
        $demography [State] - Get the demography of a state
        $correlation - Get the correlation between the relevant metrics
        $clustering - Get files and tables for clustering analysis
        $clustersummary - get a summary of the clustering analysis in image form
        ''')

    elif message.content.startswith('$demography'):
        state = message.content[12:].strip()
        population_df, income_df, percentile_df = get_data()
        population_df = date_formatting(population_df)
        income_df = date_formatting(income_df)
        percentile_df = date_formatting(percentile_df)

        table_formatting(percentile_df, population_df, income_df)
        population_charts(state, population_df)
        income_charts(state, income_df)
        # List of image file paths
        image_files = [
            'age_group.png', 'ethicity_group.png', 'incomevsexpenditure.png',
            'poverty.png'
        ]

        # Send each image in a separate message
        for image_file in image_files:
            with open(image_file, 'rb') as f:
                image = discord.File(f)
                await message.channel.send(file=image)

    elif message.content.startswith('$state'):
        # Send each image in a separate message
        await message.channel.send(states_list)

    elif message.content.startswith('$marketsummary'):
        population_df, income_df, percentile_df = get_data()
        population_df = date_formatting(population_df)
        income_df = date_formatting(income_df)
        percentile_df = date_formatting(percentile_df)

        merged_df = table_formatting(percentile_df, population_df, income_df)

        table_display(merged_df)

        image = ['dataframe_image.png']
        # Convert DataFrame to string

        for table in image:
            with open(table, 'rb') as f:
                image = discord.File(f)
                await message.channel.send(file=image)

    elif message.content.startswith('$correlation'):
        population_df, income_df, percentile_df = get_data()
        population_df = date_formatting(population_df)
        income_df = date_formatting(income_df)
        percentile_df = date_formatting(percentile_df)

        merged_df = table_formatting(percentile_df, population_df, income_df)
        merged_df.drop(columns=['state', 'date'], inplace=True)
        correlation(merged_df)
        image = ['correlation.png']
        # Convert DataFrame to string

        for table in image:
            with open(table, 'rb') as f:
                image = discord.File(f)
                await message.channel.send(file=image)

    elif message.content.startswith('$clustering'):
        population_df, income_df, percentile_df = get_data()
        population_df = date_formatting(population_df)
        income_df = date_formatting(income_df)
        percentile_df = date_formatting(percentile_df)
    
        merged_df = table_formatting(percentile_df, population_df, income_df)
        cluster(merged_df)
        image = ['final_df_with_clusters.png','cluster_profiles.png']
        # Convert DataFrame to string
        csvs = ['cluster_profiles.csv','final_df_with_clusters.csv']
        
        for table in image:
            with open(table, 'rb') as f:
                image = discord.File(f)
                await message.channel.send(file=image)

        for table in csvs:
            with open(table, 'rb') as f:         
                file = discord.File(f)       
                await message.channel.send(file=file)     

    elif message.content.startswith('$clustersummary'):
        image = ['cluster_summary.png']
    
        for table in image:
            with open(table, 'rb') as f:
                image = discord.File(f)
                await message.channel.send(file=image)

    

@client.event
async def on_member_join(member):
    # Greet the new member
    channel = member.guild.system_channel
    if channel is not None:
        await channel.send(f'Welcome to the server, {member.mention}!')


client.run(discord_token)


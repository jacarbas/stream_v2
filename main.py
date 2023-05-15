import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from widgets import *
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from joblib import load

coches = pd.read_csv(r'data/coches_clean.csv', sep=',')
coches['Year'] = coches['Year'].astype('str')
coches.drop(['Numeration'],axis=1, inplace = True)

rentabilidad = pd.read_csv(r'data/rentabilidad.csv', sep=',')
rentabilidad.drop(['Unnamed: 0'],axis = 1, inplace = True)

electricos = pd.read_csv(r'data/rentabilidad_electricos.csv', sep=',')
electricos.drop(['Unnamed: 0'],axis = 1, inplace = True)
electricos = electricos.sort_values(by=['average_profitability'], ascending = False)

pd.set_option('display.max_rows', 10)
#coches.drop(['Unnamed: 0'],axis = 1, inplace = True)
coches['Horsepower'] = coches['Horsepower'].fillna(0).round(0).astype(int)

st.set_page_config(
    page_title="Luxury brand used cars",
    page_icon="::car",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "My first Exploratory Data Analysis!"
    }
)
#titulo pagina web
st.title('From Deutschland to Spain: A Data-Driven Exploration of the Profitability of Importing Audi, Mercedes, and BMW Models')

#creamos un selector de paginas
option = st.sidebar.selectbox(
    '',
    ('Data exploration', 'Analysis', 'Conclusions', 'Price predictor'),
    index=0)

if option == 'Data exploration':
    st.subheader('Author: Javier Carrascosa Basterra')
    st.image(r'images/Audi-BMW-Mercedes.jpg',width=600)
    with st.expander('''Project's target''',expanded=True):
        st.write("""
        This project aims to analyze the profitability of importing various models of BMW, Audi, and Mercedes cars from Germany to Spain. 
        The goal is to determine which brand, specific models and characteristics are the most profitable to import, taking into account various factors such as the market offer in Germany or the 
        market demand in Spain.
        Through a comprehensive analysis of data, this project seeks to provide 
        insights and recommendations to businesses interested in importing luxury cars to the Spanish market. The ultimate goal is to help businesses make informed decisions that maximize 
        profits and minimize risks when importing cars from Germany to Spain.
        """)

    with st.expander('''Data acquisition'''):
        st.write("""
        To collect the data for this project, I performed web scraping on www.autoscout24.com using Python's library Beautiful Soup. I applied various filters, such as brand, model, 
        fuel type, year, mileage, horsepower, body, and country, to obtain selling prices on Audi, BMW, and Mercedes cars that are 10 years old or newer available for sale in both countries. In total, I collected data on 125.000 car advertisements, 
        of which 100.000 were unique.
        Overall, the data collection process involved a combination of web scraping techniques, data cleaning, and filtering 
        to ensure that the data obtained was accurate, complete, and relevant to the research goals.
        """)
    
    st.write('Pandas dataframes are key in this analysis. This is the car dataframe after scrapping and cleaning. Widgets on the left panel can be used to filter and explore the data')
    
    widgets(coches)
    #st.table(coches.head(10))
    st.subheader('Amount of cars per brand and country')

    # Agrupar los datos y contar la cantidad de elementos en cada grupo
    coches_por_marca_pais = coches.groupby(['Brand', 'Country']).count().reset_index()

    # Crear el objeto de gráfica de barras
    fig = px.bar(coches_por_marca_pais, x='Brand', y='Model', color='Country',barmode='group')

    # Grafico marcas por pais
    fig.update_layout(yaxis_title='Amount')
    fig.update_layout(xaxis_title='')
    fig.update_layout(legend_title_text='Country')
    st.plotly_chart(fig)
    st.write('Upon analyzing the data, it is evident that the number of cars in Germany significantly exceeds the offer in Spain. Furthermore, the data shows that Audi, followed by Mercedes and BMW, are the most popular brands in both countries')

    st.subheader("Variables correlation")
    st.image(r'graphs/heatmap.png')
    st.write('From the graph it is appreciated that the horsepower has an important relevance in the price. More powerful models usually are more expensive. In addition we can see that newer models frequently have smaller mileage.')
    
    st.subheader('Data description')
    st.table(coches.describe())
    st.write('Cars whose price is over 300.000 euros will be kept out of the analysis due to the small group of people who can afford this type of vehicles. Same happens to cars with more than 800 horsepower because either these are outlayers or super expensive modified cars. In Horsepower we can appreciate some outlayers too, as the minimun horsepower for a car equals to 0.')

    #years graph
    st.subheader('Distribution of cars age')
    counts_by_year = coches.groupby('Year').size()
    fig = px.bar(counts_by_year, x=counts_by_year.index, y=counts_by_year.values,
                labels={'x': 'Year', 'y': 'Number of Cars'},
                title='Number of Cars by Year')
    st.plotly_chart(fig)
    st.write('The amount of newer cars is over the amount of old cars, with more than 20.000 cars manufactured in 2018.')
    st.write('Now change to the Analysis tab!')

elif option == 'Analysis':
    st.subheader('Geospatial distribution of car advertisements and mean price of all data')
    st.image(r"maps/Distribucion anuncios coches germany.png")
    st.write('This map displays the average car prices and offer for cars in Germany, highlighting spatial variations across different regions. The map reveals that the west and south regions of Germany tend to have higher average car prices and offer compared to other regions. This pattern can be attributed to the concentration of car factories in these areas, which may drive up prices due to proximity to supply sources and increase availability of cars due to higher accessibility to dealerships and showrooms. The positive correlation between car prices and offer in these regions suggests that the high demand for cars in these areas can also contribute to the high prices. Overall, this map provides valuable insights into the spatial dynamics of the car market in Germany and can help inform strategic decisions for car importers and dealerships operating in the region.')

    st.image(r"maps/Distribucion anuncios coches spain.png")
    st.write('Map of average car prices and offer in Spain. High offer observed in Valencia, Barcelona, Madrid, and the Mediterranean coast. This region is well-known for being the place where foreign and national people go for retirement. Many of these are potential buyers of brand-used cars. Moreover, high demand in big metropolitan areas is appreciated.')

    #rsq3
    st.write('''
                The strategy is to find groups of cars that share the country, brand, model, fuel type, range of mileage, years and horsepower, and then calculate the average price for that group. This grouping enables the analysis of car prices and 
                price difference based on the same characteristics. The next dataframe also contains the number of cars in each group and the average selling price for each country. I considered 6 the minimum number of cars per group so that the average price is reliable. 
                Overall, this approach allows for a more detailed analysis of the car market and helps identify the most profitable cars to import from Germany to Spain.
                ''')
    st.table(rentabilidad.head(8))
    st.image(r'graphs/profitability.png')
    
    st.write('The data shows that the Audi Q3 in its RS version, with 400 horsepower, is the most profitable car to import, with an average difference of 10.000 euros aproximately between countries. However, the Mercedes GLC 350 diesel is an interesting option too, easier to sell due to a more affordable price for customers. ')
    st.image(r'images/rsq3.jpg', width=800)

    st.write('''However, what is the current status of electric cars? Are they considered to be the future of automobiles? Let's look at their profitability''')
    st.table(electricos)
    st.write('It appears that certain electric vehicle models are potentially lucrative to import. Nevertheless, I cannot deem this information as reliable due to the insufficient number of cars in Spain for such cases, which makes it difficult to calculate a trustworthy price estimation.')

    st.write('Given that we have established the Audi RSQ3 as the most financially lucrative car for importation, let us proceed to examine its geospatial dispersion and its prices for those cars with mileages between 15.000 and 60.000 kms and from the years 2019, 2020 and 2021.')
    
    #rsq3 germany
    st.image(r"maps/rsq3_germany.png")
    st.write('The german map illustrates that states with lower average prices, seen in the pevious map, typically offer the most affordable RSQ3 cars.')

    #rsq3 spain
    st.image(r"maps/rsq3_spain.png")
    st.write('The spanish map indicates that prices tend to be higher in provinces with major cities and higher incomes, such as Bizkaia, Madrid, Valencia, and Barcelona. Additionally, prices are also elevated in Malaga and Seville, which are popular tourist destinations.')

    st.write('Both countries exhibit significant regional variations. By purchasing in the least expensive regions of Germany and selling in the most costly areas of Spain, we can potentially maximize our profits to approximately 15.000 - 20.000 euros')

elif option == 'Conclusions':
    st.write('- The primary deduction drawn from the analysis is that importing the Audi RSQ3 with a mileage ranging between 15,000 and 60,000 kilometers and produced in the years 2019, 2020, or 2021 would be the most financially advantageous option. The profit can vary between 10.000 and 20.000 euros aproximately. Nevertheless, a profitable investment. This finding is partially explained by the bar chart in the data exploration, which demonstrates that Audi is the prominent car brand, comprising approximately 40% of the dataset. This can explain why an Audi is the most profitable brand to import with this data.')
    st.write('')

    st.write('- Does this imply that importing BMW vehicles to Spain is not a viable option? Not necessarily. To obtain a more accurate understanding of the car market in both countries, it would be imperative to explore and do web scrapping on some additional online car marketplaces.')
    st.write('')

    st.write('- When searching for the most cost-effective car to purchase or deciding where to sell an imported vehicle, location is a crucial factor to consider. The price of a car varies significantly based on the region within the country it is located.')
    st.write('')

    st.write('- Attaining more accurate and detailed data can prevent potential issues during the analysis process. Therefore, it is imperative to exercise extreme care and precision when performing web scraping.')
    st.write('')

    st.write('All the data can be downloaded from my GitHub repository at https://github.com/jacarbas/jacarbas')

elif option == 'Price predictor':
    st.subheader('Select the car brand, model, year of production, fuel type, mileage, horsepower, country and region of trade.')
    
    brand = coches['Brand'].unique()
    brand = st.selectbox('Brand',(np.sort(brand)))

    model = coches[coches['Brand'] == brand]['Model'].unique()
    model = st.selectbox('Model',(np.sort(model)))

    year = coches[(coches['Brand'] == brand) & (coches['Model'] == model)]['Year'].unique()
    year = st.selectbox('Year',(np.sort(year)))

    fuel = coches[(coches['Brand'] == brand) & (coches['Model'] == model) & (coches['Year'] == year)]['Fuel type'].unique()
    fuel = st.selectbox('Fuel type',(np.sort(fuel)))

    mileage = st.text_input('Mileage in kms')
    if mileage != '':
        mileage = int(mileage)

    horsepower = st.text_input('Horsepower')
    if horsepower != '':
        horsepower = int(horsepower)

    country = coches['Country'].unique()
    country = st.selectbox('Country',(np.sort(country)))
    
    if st.button("Predict"):
        model1 = pickle.load(open('dec_tree.sav', 'rb'))
        
        df = pd.DataFrame({'Marca':[brand], 'Modelo':[model], 'Kms':[mileage], 
                           'Año':[year], 'Combustible':[fuel], 'Potencia':[horsepower], 'Pais':[country]})
        
        encoder = LabelEncoder()
        encoder.classes_ = np.load("encoders/marca_encoder.npy", allow_pickle= True)
        df['Marca'] = encoder.transform(df['Marca'])

        encoder = LabelEncoder()
        encoder.classes_ = np.load("encoders/modelo_encoder.npy", allow_pickle= True)
        df['Modelo'] = encoder.transform(df['Modelo'])

        encoder = LabelEncoder()
        encoder.classes_ = np.load("encoders/combustible_encoder.npy", allow_pickle= True)
        df['Combustible'] = encoder.transform(df['Combustible'])

        encoder = LabelEncoder()
        encoder.classes_ = np.load("encoders/pais_encoder.npy", allow_pickle= True)
        df['Pais'] = encoder.transform(df['Pais'])

        sc=load('encoders/std_scaler.bin')
        df = sc.transform(df)
        
        df2 = pd.DataFrame(data=df, columns=['Marca','Modelo','Kms','Año','Combustible','Potencia','Pais'])
        
        prediction = model1.predict(df2)
        st.subheader(int(prediction[0]))
        st.image(r'images/lambo.png',width=600)

import streamlit as st
import pandas as pd

def widgets(df):
    # Create a list of unique brands
    brands = df['Brand'].unique()
    fuel = df['Fuel type'].unique()
    year = df['Year'].unique()
    year = sorted(year)
    country = df['Country'].unique()

# Create a dictionary of models for each brand
    models_by_brand = {}
    for brand in brands:
        models_by_brand[brand] = df[df['Brand'] == brand]['Model'].unique()

    # Get the selected brand from the user
    selected_brand = st.sidebar.selectbox('Brand', brands)

    # Get the list of models for the selected brand
    selected_brand_models = models_by_brand[selected_brand]

    # Get the selected model from the user
    selected_model = st.sidebar.selectbox('Model', selected_brand_models)
    selected_fuel = st.sidebar.selectbox('Fuel', fuel)
    selected_year = st.sidebar.selectbox('Year', year)
    selected_country = st.sidebar.selectbox('Country', country)

    filtered_df = df[(df['Brand'] == selected_brand) & (df['Model'] == selected_model) & (df['Fuel type'] == selected_fuel) & (df['Year'] == selected_year) & (df['Country'] == selected_country)]

    # Display the filtered dataframe
    st.write(filtered_df)
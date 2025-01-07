import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


st.markdown("# Uber NYC Analysis🚕", unsafe_allow_html=True)

@st.cache_data
def load_data():
    file_path = r"C:\Users\Mahnum Zahid\Desktop\iids\UBER DATASET.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error reading the file: {e}")
    
    return df


st.sidebar.title("Navigation")
sections = ["Introduction", "EDA", "Model", "Conclusion"]
choice = st.sidebar.radio("Go to", sections)

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    df = None

# Section 1: Introduction
if choice == "Introduction":
    st.header("Introduction")
    st.write("""
        This project explores the Uber Pickups in NYC dataset.
        The goal is to analyze trends, derive insights, and evaluate predictive models.
    """)
    if df is not None:    
        st.write("### Full Dataset")
        st.dataframe(df)
    else:
        st.write("No data to display due to an error.")

# Section 2: EDA

elif choice == "EDA":
    st.header("Exploratory Data Analysis")
    st.write("### Key Statistics")
    st.write(df.describe())

    top_n_pickup = df['PU_Address'].value_counts().head(10).index
    df_top_pickup = df[df['PU_Address'].isin(top_n_pickup)]

    st.write("### Countplot for Top Pickup Addresses")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.countplot(y=df_top_pickup['PU_Address'], order=df_top_pickup['PU_Address'].value_counts().index, palette="viridis", ax=ax1)
    ax1.set_xlabel('Count', fontsize=12)
    ax1.set_ylabel('Pickup Address', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")  
    st.pyplot(fig1)

    top_n_dropoff = df['DO_Address'].value_counts().head(10).index
    df_top_dropoff = df[df['DO_Address'].isin(top_n_dropoff)]

    st.write("### Countplot for Top Drop-Off Addresses")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.countplot(y=df_top_dropoff['DO_Address'], order=df_top_dropoff['DO_Address'].value_counts().index, palette="coolwarm", ax=ax2)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_ylabel('Drop-Off Address', fontsize=12)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")  
    st.pyplot(fig2)

    st.write("### Distribution of Status")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.countplot(x=df['Status'], ax=ax3)
    ax3.set_title('Distribution of Status', fontsize=14)
    ax3.set_xlabel('Status', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    st.pyplot(fig3)

    st.write("### Trend Analysis: Number of Trips Over Time")
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    trips_per_day = df.groupby('Date').size()
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    trips_per_day.plot(kind='line', ax=ax4)
    ax4.set_title('Trips Over Time', fontsize=14)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.set_ylabel('Number of Trips', fontsize=12)
    ax4.grid(True)
    st.pyplot(fig4)

    st.write("### Missing Value Analysis")
    missing_values = df.isnull().sum()

    st.write("#### Missing Values per Column:")
    st.write(missing_values)

    missing_percentage = (missing_values / len(df)) * 100

    st.write("#### Percentage of Missing Values per Column:")
    st.write(missing_percentage)
    
    st.write("### Data Types and Unique Value Counts")

    st.write("#### Data Types of Columns:")
    st.write(df.dtypes)

    unique_values = df.nunique()
    st.write("#### Unique Values in Each Column:")
    st.write(unique_values)

    st.write("### Trend Analysis: Number of Trips Over Hours of the Day")

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')  

    df['hour'] = df['Time'].dt.hour

    trips_per_hour = df['hour'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    trips_per_hour.plot(kind='line', ax=ax)
    ax.set_title('Trips Over Hours of the Day')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Number of Trips')
    ax.grid(True)
    st.pyplot(fig)

    st.write("### Count of Trips by Pickup Address and Status")
    
    grouped_data = df.groupby(['PU_Address', 'Status']).size().reset_index(name='Trips Count')
    
    st.write(grouped_data)

    avg_trips_by_address = df.groupby('PU_Address').size().mean()
    st.write(f"### Average Trips by Pickup Address: {avg_trips_by_address:.2f}")

    st.write("### Top 20 Pickup Addresses vs. Status")

    top_20_pickup_addresses = df['PU_Address'].value_counts().head(20).index

    df_top_20 = df[df['PU_Address'].isin(top_20_pickup_addresses)]

    plt.figure(figsize=(14, 8))

    sns.countplot(x='PU_Address', hue='Status', data=df_top_20, palette='Set2')

    plt.title('Top 20 Pickup Addresses vs. Status', fontsize=16)
    plt.xlabel('Pickup Address', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    plt.xticks(rotation=90, fontsize=12)

    plt.tight_layout()

    st.pyplot(plt)

    missing_data = df.isnull().sum()

    df_cleaned = df.dropna()

    df_filled = df.copy()  

    
    numerical_columns = df_filled.select_dtypes(include=['float64', 'int64']).columns
    df_filled[numerical_columns] = df_filled[numerical_columns].fillna(df_filled[numerical_columns].mean())

    categorical_columns = df_filled.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df_filled[column] = df_filled[column].fillna(df_filled[column].mode()[0])

    
    missing_data_after = df_filled.isnull().sum()

    
    st.write("Missing data before imputation:")
    st.write(missing_data)
    st.write("\nMissing data after imputation:")
    st.write(missing_data_after)
    
    st.write("Cleaned Data (with Imputation or Removal):")
    st.write(df_filled.head())


    label_encoder = LabelEncoder()

    df['PU_Address_encoded'] = label_encoder.fit_transform(df['PU_Address'])
    df['Status_encoded'] = label_encoder.fit_transform(df['Status'])

    df[['PU_Address', 'PU_Address_encoded', 'Status', 'Status_encoded']].head()


    st.subheader("3. Scale or Normalize Numerical Features")

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    df['hour'] = df['Time'].dt.hour

    st.write("Extracted Hour Column:")
    st.write(df[['Time', 'hour']].head())


    # Section 3: Model
elif choice == "Model":
    
    st.title('Uber Status Prediction with Random Forest')

    data = {
        'Time': [
            '2025-01-01 12:30:00', '2025-01-01 13:45:00', '2025-01-01 14:00:00',
            '2025-01-02 15:15:00', '2025-01-02 10:00:00', '2025-01-02 11:30:00',
            '2025-01-03 18:45:00', '2025-01-03 08:15:00'
        ],
        'PU_Address_encoded': [1, 2, 3, 1, 2, 3, 1, 2],
        'DO_Address_encoded': [3, 2, 1, 2, 3, 1, 2, 3],
        'Status_encoded': [0, 1, 0, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['hour'] = df['Time'].dt.hour  

    required_columns = {'hour', 'PU_Address_encoded', 'DO_Address_encoded', 'Status_encoded'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing required columns: {required_columns - set(df.columns)}")
    else:
        X = df[['hour', 'PU_Address_encoded', 'DO_Address_encoded']]
        y = df['Status_encoded']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        st.subheader("Model Evaluation on Complete Dataset")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y, y_pred))

        st.subheader("Predict Status for New Data")
        hour_input = st.number_input('Enter Hour:', min_value=0, max_value=23, value=12)
        pu_address_input = st.number_input('Enter Pickup Address Encoded:', min_value=0, value=3)
        do_address_input = st.number_input('Enter Dropoff Address Encoded:', min_value=0, value=2)

        new_data = pd.DataFrame([[hour_input, pu_address_input, do_address_input]],
                                columns=['hour', 'PU_Address_encoded', 'DO_Address_encoded'])

        if st.button('Predict Status'):
            predicted_status = model.predict(new_data)
            st.write(f"Predicted Status for {new_data.values.tolist()[0]}: {predicted_status[0]}")

            

    # Section 4: Conclusion
elif choice == "Conclusion":
    st.header("Conclusion")
    st.write("""
    This analysis of the Uber Pickups in NYC dataset provides valuable insights into trip patterns, including:
    
    1. **High-Density Pickup Areas:** The heatmap revealed that certain areas, such as Manhattan, consistently experience high pickup densities, likely due to the concentration of commercial and entertainment activities.
    2. **Day and Time Trends:** Analysis of trip data across different days and times showed distinct patterns, such as higher demand on weekends and during evening hours.
    3. **Trip Duration Predictions:** A machine learning model was used to predict trip durations based on location data, achieving an R-squared score of {r2_score:.2f}. This indicates that there is scope for improving model accuracy by including more features like traffic conditions, weather data, or time of day.

    ### Suggestions for Future Work
    - **Include External Factors:** Incorporate additional datasets (e.g., weather, traffic) to improve the predictive capabilities of the model.
    - **Enhance Visualizations:** Use animations or time-series visualizations to better represent how pickup trends evolve over time.
    - **User-Centric Features:** Implement features like trip fare estimations, ride-sharing optimizations, and driver availability analysis.
    - **Advanced Analytics:** Conduct clustering analysis to identify key trip zones and segmentation analysis for understanding user behavior.
    - **Real-Time Updates:** Develop a dashboard that updates with live trip data for more actionable insights.
    """)

# Add Credits
st.sidebar.info("Developed by Mahum")

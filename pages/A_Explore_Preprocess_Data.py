import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from pandas.plotting import scatter_matrix
from itertools import combinations


#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <Mall Customers Analysis ML Model>")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################


def app_state_init():
    if 'data' not in st.session_state:
        st.session_state['data'] = None

def fetch_dataset():
#sesion state
    if st.session_state['data'] is None:
        data = st.file_uploader('Upload a Dataset', type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.session_state['data'] = df
    return st.session_state['data']

def visualize_missing_data(df): 
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if missing_data.empty:
        st.write("No missing values found.")
    else:
        missing_data.sort_values(inplace=True)
        fig, ax = plt.subplots()
        missing_data.plot.bar(ax=ax)
        ax.set_title('Missing data in each column')
        st.pyplot(fig)

def handle_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].median())
    return df

def visualize_distributions(df):
    #Visualization of data
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        fig = px.histogram(df, x=column, marginal='rug', title=f'Histogram of {column}')
        st.plotly_chart(fig)

def one_hot_encode_feature(df, features):
    #One-hot encode
    df = df.dropna()
    dataset = df.copy()
    for feat in features:
        dataset = pd.get_dummies(df, columns=[feat], prefix=feat)
    st.session_state['data'] = dataset
    return dataset

def integer_encode_feature(df, features):
   #Integer encode 
    df = df.dropna()
    dataset = df.copy()
    for feat in features:
        encoder = OrdinalEncoder()
        dataset[feat] = encoder.fit_transform(dataset[[feat]])
    st.session_state['data'] = dataset
    return dataset

def remove_irrelevant_features(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop)
    return df


def remove_outliers(df, features, outlier_removal_method=None):
    df=df.dropna()
    dataset = df.copy()

    for feature in features:
        lower_bound = dataset[feature].max()
        upper_bound = dataset[feature].min()

        if(outlier_removal_method =='IQR'): # IQR method
            if (feature in df.columns):
                dataset = dataset.dropna()
                Q1 = np.percentile(dataset[feature], 25, axis=0)
                Q3 = np.percentile(dataset[feature], 75, axis=0)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5*IQR
                lower_bound = Q1 - 1.5*IQR
        else: # Standard deviation methods
            upper_bound = dataset[feature].mean() + 3* dataset[feature].std() #mean + 3*std
            lower_bound = dataset[feature].mean() - 3* dataset[feature].std() #mean - 3*std
        dataset_size1 = dataset.size
        dataset = dataset[dataset[feature] > lower_bound]
        dataset = dataset[dataset[feature] < upper_bound]
        dataset_size2 = dataset.size
        st.write('%s: %d outliers were removed from feature %s in the dataset' % (outlier_removal_method,dataset_size1-dataset_size2, feature))

    st.session_state['data'] = dataset
    return dataset

def normalize_data(df):
    """Normalize numerical data using StandardScaler."""
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

def scale_features(df, features, scaling_method): 
    df = df.dropna()
    X = df.copy()
    for f in features:
        if(scaling_method == 'Standardarization'):
            X[f+'_std'] = (X[f] - X[f].mean()) / X[f].std()
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        elif(scaling_method == 'Normalization'):
            X[f+'_norm'] = (X[f] - X[f].min()) / (X[f].max() - X[f].min())  
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        elif(scaling_method == 'Log'):
            X[f+'_log'] = np.log2(X[f])
            X[X[f+'_log']<0] = 0 # Check for -inf
            st.write('Feature {} is scaled using {}'.format(f, scaling_method))
        else:
            st.write('scaling_method is invalid.')

    st.session_state['house_df'] = X
    return X

def compute_descriptive_stats(df, stats_feature_select, stats_select):
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    df=df.dropna()
    X = df.copy()
    for f in stats_feature_select:
        output_str = str(f)
        for s in stats_select:
            if(s=='Mean'):
                mean = round(X[f].mean(), 2)
                output_str = output_str + ' mean: {0:.2f}    |'.format(mean)
                out_dict['mean'] = mean
            elif(s=='Median'):
                median = round(X[f].median(), 2)
                output_str = output_str + ' median: {0:.2f}    |'.format(median)
                out_dict['median'] = median
            elif(s=='Max'):
                max = round(X[f].max(), 2)
                output_str = output_str + ' max: {0:.2f}    |'.format(max)
                out_dict['max'] = max
            elif(s=='Min'):
                min = round(X[f].min(), 2)
                output_str = output_str + ' min: {0:.2f}    |'.format(min)
                out_dict['min'] = min
        st.write(output_str)
    return output_str, out_dict

# Helper Function
def compute_correlation(df, features):
    correlation = df[features].corr()
    feature_pairs = combinations(features, 2)
    cor_summary_statements = []

    for f1, f2 in feature_pairs:
        cor = correlation[f1][f2]
        summary = '- Features %s and %s are %s %s correlated: %.2f' % (
            f1, f2, 'strongly' if cor > 0.5 else 'weakly', 'positively' if cor > 0 else 'negatively', cor)
        st.write(summary)
        cor_summary_statements.append(summary)

    return correlation, cor_summary_statements

#############################################
# Streamlit 
app_state_init()
df = fetch_dataset()

if df is not None:
    st.markdown('## View Initial Data')
    st.dataframe(df)
    
    st.markdown('## View missing data')
    visualize_missing_data(df)

    st.markdown('## Visualize Features')
    visualize_distributions(df)

    st.markdown('## Remove Features')
    columns_to_drop = st.multiselect('Select Features to Remove', df.columns)
    if st.button('Remove Selected Features'):
        df = remove_irrelevant_features(df, columns_to_drop)
        st.session_state['data'] = df
        st.markdown('### Data after Removing Selected Features')
        st.dataframe(df)
        
        

    st.markdown('## Feature Encoding')
    string_columns = list(df.select_dtypes(['object']).columns)

    int_col, one_hot_col = st.columns(2)

    with (int_col):
        text_feature_select_int = st.multiselect(
            'Select text features for Integer encoding',
            string_columns,
        )
        if (text_feature_select_int and st.button('Integer Encode feature')):
            df = integer_encode_feature(df, text_feature_select_int)
            st.dataframe(df)
            #st.write(df)
    
    # Perform One-hot Encoding
    with (one_hot_col):
        text_feature_select_onehot = st.multiselect(
            'Select text features for One-hot encoding',
            string_columns,
        )
        if (text_feature_select_onehot and st.button('One-hot Encode feature')):
            df = one_hot_encode_feature(df, text_feature_select_onehot)
            st.dataframe(df)
            #st.write(df)

    # Show updated dataset


    
    st.markdown('## Remove outliers')
    outlier_feature_select = None
    numeric_columns = list(df.select_dtypes(include='number').columns)

    outlier_method_select = st.selectbox(
        'Select statistics to display',
        ['IQR', 'STD']
    )

    outlier_feature_select = st.multiselect(
        'Select a feature for outlier removal',
        numeric_columns,
    )
    if (outlier_feature_select and st.button('Remove Outliers')):
        df = remove_outliers(df, outlier_feature_select, outlier_method_select)
        st.dataframe(df)


    st.markdown('## Normalize Data')
    if st.button('Normalize Data'):
        df = normalize_data(df)
        st.session_state['data'] = df
        st.markdown('### Data after Normalization')
        st.dataframe(df)
    
    
    # Sacling features
    st.markdown('## Feature Scaling')
    st.markdown('Use standardarization or normalization to scale features')

    # Use selectbox to provide impute options {'Standardarization', 'Normalization', 'Log'}
    scaling_method = st.selectbox(
        'Select feature scaling method',
        ('Standardarization', 'Normalization', 'Log')
    )

    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    scale_features_select = st.multiselect(
        'Select features to scale',
        numeric_columns,
    )

    if (st.button('Scale Features')):
        # Call scale_features function to scale features
        if(scaling_method and scale_features_select):
            df = scale_features(df, scale_features_select, scaling_method)

    # Display updated dataframe
    st.dataframe(df)


    # Descriptive Statistics 
    st.markdown('## Descriptive Statistics')

    stats_numeric_columns = list(df.select_dtypes(['float','int']).columns)
    stats_feature_select = st.multiselect(
        'Select features for statistics',
        stats_numeric_columns,
    )

    stats_select = st.multiselect(
        'Select statistics to display',
        ['Mean', 'Median','Max','Min']
    )
    display_stats, _ = compute_descriptive_stats(df, stats_feature_select, stats_select)
    
    st.markdown("## Correlation Analysis")
    # Collect features for correlation analysis using multiselect
    numeric_columns = list(df.select_dtypes(['float','int']).columns)


    select_features_for_correlation = st.multiselect(
        'Select features for visualizing the correlation analysis (up to 4 recommended)',
        numeric_columns,
    )

    # Compute correlation between selected features
    correlation, correlation_summary = compute_correlation(
        df, select_features_for_correlation)
    st.write(correlation)

    # Display correlation of all feature pairs
    if select_features_for_correlation:
        try:
            fig = scatter_matrix(
                df[select_features_for_correlation], figsize=(12, 8))
            st.pyplot(fig[0][0].get_figure())
        except Exception as e:
            print(e)

    st.markdown('## Preprocessed Data')
    st.dataframe(df)
    st.write('Continue to **Train Model**')
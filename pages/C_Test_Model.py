import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <Mall Customers Analysis ML Model>")

#############################################

st.title('Test Model')

#############################################

df = None
df = fetch_dataset()

if df is not None:
    st.markdown("### Get Performance Metrics")
    metric_options = ['']

    model_options = ['']

    trained_models = ['K-means Clustering', 'Regression Analysis', 'Model Performance Comparison']

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained models for evaluation',
        options=trained_models
    )

    if 'K-means Clustering' in model_select:
        st.markdown("### K-means Clustering")

        # Select the features of annual income and spending score to cluster
        X = df.iloc[:, [3,4]].values

        # Standardize the data for k-means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=5)
        y_kmeans = kmeans.fit_predict(X_scaled)

        # Create a trace for each cluster
        cluster_traces = []
        colors = ['red', 'blue', 'green', 'cyan', 'magenta']

        for i in range(5):
            cluster_trace = go.Scatter(
                x=X[y_kmeans == i, 0],
                y=X[y_kmeans == i, 1],
                mode = 'markers',
                marker=dict(size=10, color=colors[i]),
                name=f'Cluster {i+1}'
            )
            cluster_traces.append(cluster_trace)

        # Create a trace for centroids
        centroid_trace = go.Scatter(
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            mode='markers',
            marker=dict(size=12, color='yellow', symbol='cross'),
            name='Centroids'
        )

        # Create the layout
        layout = go.Layout(
            title='Clusters of Customers',
            xaxis=dict(title='Annual Income (k$)'),
            yaxis=dict(title='Spending Score (1-100)'),
            legend=dict(x=0.7, y=0.95),
            font=dict(size=14, color="black"),
            paper_bgcolor="lightgray",
            plot_bgcolor="white",
        )
        fig = go.Figure(data=cluster_traces + [centroid_trace], layout=layout)
        st.plotly_chart(fig)

        # Initialize a list to score the WCSS values
        wcss = []

        # Try different values of K (1-10) and calculate WCSS for each K
        for k in range(1,11):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        # Plot the Elbow Method graph
        st.pyplot(plt.figure(figsize=(8, 6)))
        plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.grid()

        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('### Review Model Performance')

            review_options = ['plot', 'metrics']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=review_options
            )

            if 'plot' in review_plot:
                pass

            if 'metrics' in review_plot:
                pass

    # Select a model to deploy from the trained models
    st.markdown("### Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=trained_models,
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to **Deploy Model**')

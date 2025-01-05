import pandas as pd
import plotly.express as pl
import matplotlib.pyplot as plt
import seaborn as sns
import io
from io import BytesIO
import streamlit as st
import numpy as np
st.set_page_config(page_title="Advanaced EDA",
    page_icon="https://cdn-icons-png.flaticon.com/512/3090/3090011.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
hide_menu = """
    <style>
    #MainMenu{
        visibility:hidden;
    }
    footer{
        visibility:hidden;
    }
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
st.sidebar.markdown("")
if 'df' in st.session_state:
    df = st.session_state.df
    @st.cache_data(persist=True)
    def generate_frequency_table(df, column):
        return pd.crosstab(index=df[column], columns="count")

    @st.cache_data(persist=True)
    def generate_histogram(df, column, custom_colors):
        return pl.histogram(df, x=column, color=column, barmode='group', color_discrete_sequence=custom_colors)

    @st.cache_data(persist=True)
    def generate_histogram_for_univariate(df, column, nbins):
        return pl.histogram(df[column], nbins=nbins)

    @st.cache_data(persist=True)
    def generate_box_plot(df, column):
        return pl.box(df[column])

    @st.cache_data(persist=True)
    def generate_scatter_plot(df, column1, column2):
        return pl.scatter(x=df[column1], y=df[column2])

    @st.cache_data(persist=True)
    def generate_regression_plot(df, column1, column2):
        reg_size, ax = plt.subplots(figsize=(12,4))
        ax.set_title(f"Regression Plot of {column1} vs {column2}")
        sns.regplot(x=column1, y=column2, data=df)
        return reg_size

    @st.cache_data(persist=True)
    def generate_box_plot_bivariate(df, column1, column2):
        fig = pl.box(df, x=df[column2], y=df[column1], color=column2)
        fig.update_layout(title='Box Plot', xaxis_title=column2, yaxis_title=column1)
        return fig

    @st.cache_data(persist=True)
    def generate_bar_plot_bivairate(df, column1, column2):
        fig = pl.bar(df, x=column2, y=column1, color=column2)
        fig.update_layout(title='Bar Plot', xaxis_title=column2, yaxis_title=column1)
        return fig

    @st.cache_data(persist=True)
    def generate_violin_plot_bivariate(df, column1, column2):
        fig = pl.violin(df, x=column2, y=column1, color=column2, box=True)
        fig.update_layout(title='Violin Plot', xaxis_title=column2, yaxis_title=column1)
        return fig

    @st.cache_data(persist=True)
    def generate_line_plot_univariate(df, column1, column2):
        fig = pl.line(df, x=column1, y=column2)
        fig.update_layout(title='Line Plot', xaxis_title=column1, yaxis_title=column2)
        return fig

    @st.cache_data(persist=True)
    def generate_heatmap_univariate(df, column1, column2):
        correlation = df[[column1, column2]].corr()
        fig, ax = plt.subplots(figsize=(12,4))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title(f'Heatmap of {column1} and {column2}')
        return fig

    @st.cache_data(persist=True)
    def generate_stacked_bar_plot_bivariate(df, column1, column2):
        fig = pl.bar(df, x=column2, color=column1, barmode='stack')
        fig.update_layout(title="Stacked Bar Plot", xaxis_title=column2, yaxis_title="Count")
        return fig

    @st.cache_data(persist=True)
    def generate_grouped_bar_plot_bivariate(df, column1, column2):
        fig = pl.bar(df, x=column2, color=column1, barmode='group')
        fig.update_layout(title="Grouped Bar Plot", xaxis_title=column2, yaxis_title="Count")
        return fig

    @st.cache_data(persist=True)
    def generate_heatmap_contingency_bivariate(contingency_table):
        fig, ax = plt.subplots(figsize=(12,4))
        sns.heatmap(contingency_table, annot=True, cmap="Greens", fmt='.2f')
        for _, spine in ax.spines.items():
            spine.set_visible(True)  # Show the spine
            spine.set_color('black')  # Set color of the spine
            spine.set_linewidth(2)
        plt.title('Contingency Table Heatmap')
        return fig
    st.markdown("""
        <h1 style='text-align: center;'>Advanced
            <span style='color: red;'> EDA Analysis</span> on Dataset</h1>""",
        unsafe_allow_html=True
    )
    st.markdown("---")
    tabs=st.tabs(['Feature Extraction','Handling Missing Value','Outlier Detection and Handle',
    'Univariate Aanlysis', 'Bivariate Aanlysis'])
    with tabs[0]:
        st.write("#### :green[Original Dataset :]")
        st.write(df)
        st.write("No. of :green[Row] and :green[Columns] in DataSet: ")
        st.write(df.shape)
        st.markdown("----")
        st.write("#### :green[After Feature Extraction :]")
        columns_to_drop = [
            'Site Num',
            'Datum',
            'Metric Used',
            'Method Name',
            'Event Type',
            'Completeness Indicator',
            'Primary Exceedance Count',
            'Secondary Exceedance Count',
            'Arithmetic Mean',
            'Arithmetic Standard Dev',
            '1st Max Value',
            '2nd Max Value',
            '3rd Max Value',
            '4th Max Value',
            '1st Max Non Overlapping Value',
            '1st NO Max DateTime',
            '2nd Max Non Overlapping Value',
            '2nd NO Max DateTime',
            '99th Percentile',
            '98th Percentile',
            '95th Percentile',
            '90th Percentile',
            '75th Percentile',
            '50th Percentile',
            '10th Percentile',
            'Address',
            'CBSA Name',
            '1st Max DateTime',
            '2nd Max DateTime',
            '3rd Max DateTime',
            '4th Max DateTime',
            'Date of Last Change',
            'Local Site Name'
        ]
        feature_extraction_data=df.drop(columns=columns_to_drop, errors='ignore')
        st.write(feature_extraction_data)
        st.write("No. of :green[Row] and :green[Columns] in DataSet: ")
        st.write(feature_extraction_data.shape)
    with tabs[1]:
        st.write("## :green[Handling Missing Value Using] :red[Mean :]")
        st.write(":green[After Filling the data, there is no missing records present in the dataset]")
        new_updated_data = feature_extraction_data.copy()
        numeric_column = new_updated_data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_column:
            if new_updated_data[col].dtype == 'int64':
                new_updated_data[col] = new_updated_data[col].fillna(new_updated_data[col].mean().astype(int))
            else:
                new_updated_data[col] = new_updated_data[col].fillna(new_updated_data[col].mean())
        categoric_column = new_updated_data.select_dtypes(include=['object']).columns
        for col in categoric_column:
            new_updated_data[col] = new_updated_data[col].fillna(new_updated_data[col].mode()[0])
        st.write(new_updated_data)
        st.download_button(
            ":green[Download Data File as .CSV]",
            new_updated_data.to_csv(),
            file_name='new_dataset.csv',
            mime='csv',
            key="slider00010"
        )
        st.write("None Missing Records:")
        st.write(new_updated_data.isnull().sum())
    with tabs[2]:
        new_outlier_dataset=new_updated_data.copy()
        numeric_columns = new_outlier_dataset.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
        iqr_column = st.selectbox(":green[ Select a Column]", numeric_columns,key="10_key")
        @st.cache_data(persist=True)
        def calculate_iqr(new_outlier_dataset, iqr_column):
            q1 = new_outlier_dataset[iqr_column].quantile(0.25)
            q3 = new_outlier_dataset[iqr_column].quantile(0.75)
            iqr = q3 - q1
            return iqr, q1, q3

        iqr, q1, q3 = calculate_iqr(new_outlier_dataset, iqr_column)
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers_iqr = new_outlier_dataset[(new_outlier_dataset[iqr_column] < lower_bound) | (new_outlier_dataset[iqr_column] > upper_bound)][iqr_column]
        right_cols, left_col, center, last_left = st.columns((5,5,5,5))
        with right_cols:
            st.write(f"Feature Name: :green[{iqr_column}]")
            st.write(new_outlier_dataset[iqr_column])
        with left_col:
            st.write(f"Five-Number Summary for :green[{iqr_column}]:")
            st.write("Q1(25th Percentile): ",q1)
            st.write("Q3(75th Percentile): ",q3)
            st.write("IQR Value: ", iqr)
            st.write("Minimum: ",new_outlier_dataset[iqr_column].min())
            st.write("Maximun: ",new_outlier_dataset[iqr_column].max())
            st.write("Median: ",new_outlier_dataset[iqr_column].median())
        with center:
            st.write(":green[Outlier's] detected with index number.")
            if isinstance(outliers_iqr, pd.Series):
                outliers_df = outliers_iqr.to_frame()
                if not outliers_df.empty:
                    st.write(outliers_df)
                else:
                    st.write(f"There is no such :green[Outlier's] detected in :green[{iqr_column}] Column.")
            else:
                st.write("Outliers data is not in the expected format.")
        with last_left:
            st.write("Total No. of :green[Outlier's] are: ")
            st.write(len(outliers_iqr))
        inner_zscore = st.tabs(['Box IQR Plot', 'Scatter IQR Plot', "Violin IQR Plot"])
        with inner_zscore[0]:
            fig=pl.box(new_outlier_dataset[iqr_column])
            fig.update_layout(
            title='Box Plot'
            )
            st.plotly_chart(fig,key='max-1')
        with inner_zscore[1]:
            fig=pl.scatter(new_outlier_dataset, x=range(len(new_outlier_dataset)), y=iqr_column)
            fig.add_scatter(x=outliers_iqr.index, y=outliers_iqr, marker=dict(color='red', size=10),
            name='Outliers',  text='Outlier' , textposition='top center',mode='markers')
            fig.update_layout(
            title='Scatter Plot',
            xaxis_title=(f"length of {iqr_column}")
            )
            st.plotly_chart(fig)
        with inner_zscore[2]:
            df_outliers = new_outlier_dataset.copy()
            df_outliers['Outlier'] = new_outlier_dataset[iqr_column].apply(lambda x: 'Outlier' if x < lower_bound or x > upper_bound else 'Not Outlier')
            fig = pl.violin(df_outliers, y=iqr_column, color='Outlier', box=True, points='all',title='Violin Plot',color_discrete_map={'Outlier': 'red', 'Not Outlier': 'blue'})
            st.plotly_chart(fig)
        st.markdown("---")
        st.markdown("""
            <h1 style='text-align: center; font-size: 45px; color: green;'>Outlier Handle</h1>""",
            unsafe_allow_html=True
        )
        @st.cache_data(persist=True)
        def impute_outliers_with_median(new_outlier_dataset, _numeric_columns):
            df_median_imputed = new_outlier_dataset.copy()
            for column in numeric_columns:
                q1 = df_median_imputed[column].quantile(0.25)
                q3 = df_median_imputed[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                median_value = df_median_imputed[column].median()
                df_median_imputed[column] = np.where((df_median_imputed[column] > upper_bound) | (df_median_imputed[column] < lower_bound),median_value, df_median_imputed[column])
            return df_median_imputed
        st.success("The New Updated Dataset Imputed :green[Outlier's] With :green[Median] .")
        df_median_imputed = impute_outliers_with_median(new_outlier_dataset, numeric_columns)
        st.write(df_median_imputed)
        shapes,download=st.columns((27,7))
        with shapes:
            st.write("Total no. of Rows: ",df_median_imputed.shape[0])
            st.write("Total no. of Columns: ",df_median_imputed.shape[1])
        with download:
            st.download_button(":green[Download Data File as .CSV]", df_median_imputed.to_csv(index=False), file_name='median_outlier_dataset.csv', mime='csv', key='slid4')
        st.markdown("---")
        st.markdown("<h1 style='text-align: center; color: #54B254;'>Visualization of Original and Removed Outiler's</ h1>",unsafe_allow_html=True)
        st.markdown("---")

        select_columns = st.selectbox("Select a Column for Comparison Between :green[Outliers]", numeric_columns,key="select_columns_median")
        q1 = new_outlier_dataset[select_columns].quantile(0.25)
        q3 = new_outlier_dataset[select_columns].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers_detected = new_outlier_dataset[(new_outlier_dataset[select_columns] < lower_bound) | (new_outlier_dataset[select_columns] > upper_bound)][select_columns]

        multiple_graphs = st.tabs(["Box Plot", "Scatter Plot", "Violin Plot"])
        with multiple_graphs[0]:
            original_outlier, handled_outlier = st.columns((5, 5))
            with original_outlier:
                fig_original = pl.box(new_outlier_dataset[select_columns])
                fig_original.update_layout(title="Box Plot Having Outliers")
                st.plotly_chart(fig_original,key="key_1")
            with handled_outlier:
                fig_handled = pl.box(df_median_imputed[select_columns])
                fig_handled.update_layout(title="Box Plot After Imputation with Median")
                st.plotly_chart(fig_handled)

        with multiple_graphs[1]:
            original_outlier, handled_outlier = st.columns((5, 5))
            with original_outlier:
                fig = pl.scatter(new_outlier_dataset, x=range(len(new_outlier_dataset)), y=select_columns)
                fig.add_scatter(x=outliers_detected.index, y=outliers_detected, marker=dict(color='red', size=10),
                                name='Outliers', text='Outlier', textposition='top center', mode='markers')
                fig.update_layout(title="Scatter Plot Having Outliers", xaxis_title=f"length of {select_columns}")
                st.plotly_chart(fig,key="key_2")
            with handled_outlier:
                fig = pl.scatter(df_median_imputed, x=df_median_imputed.index, y=select_columns)
                fig.update_layout(title="Scatter Plot After Imputation with Median", xaxis_title=f"length of {select_columns}")
                st.plotly_chart(fig)

        with multiple_graphs[2]:
            original_outlier, handled_outlier = st.columns((5, 5))
            with original_outlier:
                new_outlier_dataset['Outlier'] = new_outlier_dataset[select_columns].apply(lambda x: 'Outlier' if x < lower_bound or x > upper_bound else 'Not Outlier')
                fig = pl.violin(new_outlier_dataset, y=select_columns, box=True, color='Outlier', points='all', color_discrete_map={'Outlier': 'red'})
                fig.update_layout(title="Violin Plot Having Outliers")
                st.plotly_chart(fig,key="key_3")
            with handled_outlier:
                fig = pl.violin(df_median_imputed, y=select_columns, box=True, points='all')
                fig.update_layout(title="Violin Plot After Imputation with Median")
                st.plotly_chart(fig)
    with tabs[3]:
        inner_analysis = st.tabs(['Univariate Analysis for Categorical Data', 'Univariate Analysis for Numerical Data'])
        with inner_analysis[0]:
            categorical_columns = df_median_imputed.select_dtypes(include=['object']).columns
            if categorical_columns.empty:
                st.warning(":red[There is no Categoriacal Column's in the Dataset. This Operation will not be able to Perform.]")
            else:
                if len(categorical_columns) > 0:
                    column = st.selectbox("**:green[Select Column ]**", categorical_columns,key="4_key")
                    if column:
                        right_col, left_col=st.columns((2,6))
                        with right_col:
                            st.write(f"Feature Name: :green[{column}]")
                            st.write(df_median_imputed[[column]])
                        with left_col:
                            st.write("Frequency Table")
                            freq_table = generate_frequency_table(df_median_imputed, column)
                            st.write(freq_table)
                        st.markdown("---")
                        custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
                        fig = generate_histogram(df_median_imputed, column, custom_colors)
                        fig.update_layout(title="Count Plot")
                        st.plotly_chart(fig)

        with inner_analysis[1]:
            numeric_columns = df_median_imputed.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
            if len(numeric_columns) > 0:
                column = st.selectbox("**:green[Select Column ]**", numeric_columns,key="5_key")
                if column:
                    right_col, left_col = st.columns((2,5))
                    with right_col:
                        st.write(f"Feature Name: :green[{column}]")
                        st.write(df_median_imputed[[column]])
                    with left_col:
                        st.write("Summary Statistics")
                        st.write(df_median_imputed[column].describe())
                    inner_plot = st.tabs(['Histogram Plot', 'Box Plot'])
                    with inner_plot[0]:
                        slid = st.slider(label="Number of Plots Bins", min_value=5, max_value=25, value=9, key='slider007')
                        fig=generate_histogram_for_univariate(df_median_imputed, column,slid)
                        fig.update_layout(
                        title="Histogram Plot"
                        )
                        st.plotly_chart(fig)
                    with inner_plot[1]:
                        fig=generate_box_plot(df_median_imputed, column)
                        fig.update_layout(
                        title='Box Plot'
                        )
                        st.plotly_chart(fig)

    with tabs[4]:
        inner_ba_analysis = st.tabs(['Numerical Data V/S Numerical Data', 'Numerical Data V/S Categorical Data', 'Categorical Data V/S Categorical Data'])
        with inner_ba_analysis[0]:
            numeric_columns = df_median_imputed.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
            left_col, right_col=st.columns((5,5))
            with left_col:
                column1 = st.selectbox("**:green[Select X-Axis Column]**", numeric_columns,key="6_key")
                name,summary=st.columns((3,4))
                with name:
                    st.write(f"Feature Name : :green[{column1}]")
                    st.write(df_median_imputed[column1])
                with summary:
                    st.write(f"Summary of :  :green[{column1}]")
                    st.write(df_median_imputed[column1].describe())
            with right_col:
                column2 = st.selectbox("**:green[Select Y-Axis Column]**", numeric_columns,key="7_key")
                name,summary=st.columns((3,4))
                with name:
                    st.write(f"Feature Name : :green[{column1}]")
                    st.write(df_median_imputed[column2])
                with summary:
                    st.write(f"Summary of :  :green[{column1}]")
                    st.write(df_median_imputed[column2].describe())
            graphs_for_plot=st.tabs(["Scatter Plot","Regression Plot", "Heatmap Plot"])
            with graphs_for_plot[0]:
                if len(numeric_columns) > 1:
                    if column1 and column2:
                        fig = generate_scatter_plot(df_median_imputed, column1, column2)
                        fig.update_layout(title='Scatter Plot', xaxis_title=column1, yaxis_title=column2)
                        st.plotly_chart(fig)

            with graphs_for_plot[1]:
                if len(numeric_columns)>1:
                    if column1 and column2:
                        fig=generate_regression_plot(df_median_imputed, column1, column2)
                        st.pyplot(fig)
                        buffer = BytesIO()
                        fig.savefig(buffer, format='png')
                        buffer.seek(0)
                        st.download_button(
                            label=':green[Download Graph as .PNG]',
                            data=buffer.getvalue(),
                            file_name='regression_chart.png',
                            mime='image/png'
                        )

            with graphs_for_plot[2]:
                if len(numeric_columns) > 1:
                    if column1 and column2:
                        fig = generate_heatmap_univariate(df_median_imputed, column1, column2)
                        st.pyplot(fig)
                        buffer = BytesIO()
                        fig.savefig(buffer, format='png')
                        buffer.seek(0)
                        st.download_button(
                            label=':green[Download Graph as .PNG]',
                            data=buffer.getvalue(),
                            file_name='Heatmap.png',
                            mime='image/png',
                            key="don_key"
                        )

        with inner_ba_analysis[1]:
            categorical_columns = df_median_imputed.select_dtypes(include=['object']).columns
            numeric_columns = df_median_imputed.select_dtypes(['float64', 'float32', 'int32', 'int64']).columns
            if categorical_columns.empty:
                st.warning(":red[There is no Categoriacal Column's in the Dataset. This Operation will not be able to Perform.]")
            else:
                left_column,right_column=st.columns((5,5))
                with left_column:
                    column1 = st.selectbox("**:green[Select Y-Axis Column]**", numeric_columns,key="8_key")
                with right_column:
                    column2 = st.selectbox("**:green[Select X-Axis Column]**", categorical_columns,key="89_key")
                left_col,center_col,right_col=st.columns((2,5,3))
                with left_col:
                    st.write(f"Feature Name: :green[{column1}] (Numerical Feature)")
                    st.write(df_median_imputed[column1])
                with center_col:
                    st.write("Summary Statistics by Category")
                    summary = df_median_imputed.groupby(column2)[column1].describe()
                    st.write(summary)
                with right_col:
                    st.write(f"Feature Name: :green[{column2}] (Categorical Feature)")
                    st.write(df_median_imputed[column2])
                graphs_visual=st.tabs(["Box Plot", "Bar Plot", "Violin Plot"])
                with graphs_visual[0]:
                    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
                        if column1 and column2:
                            fig = generate_box_plot_bivariate(df_median_imputed, column1, column2)
                            st.plotly_chart(fig)
                with graphs_visual[1]:
                    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
                        if column1 and column2:
                            fig= generate_bar_plot_bivairate(df_median_imputed, column1, column2)
                            st.plotly_chart(fig)
                with graphs_visual[2]:
                    if len(numeric_columns) > 0 and len(categorical_columns) > 0:
                        if column1 and column2:
                            fig=generate_violin_plot_bivariate(df_median_imputed, column1, column2)
                            st.plotly_chart(fig)
        with inner_ba_analysis[2]:
            categorical_columns1 = df_median_imputed.select_dtypes(include=['object']).columns
            categorical_columns2 = df_median_imputed.select_dtypes(include=['object']).columns
            if categorical_columns1.empty and categorical_columns2.empty:
                st.warning(":red[There is no Categoriacal Column's in the Dataset. This Operation will not be able to Perform.]")
            else:
                left_column,right_column=st.columns((5,5))
                with left_column:
                    column1 = st.selectbox("**:green[Select Y-Axis Column]**", categorical_columns2,key="88_key")
                with right_column:
                    column2 = st.selectbox("**:green[Select X-Axis Column]**", categorical_columns1,key="87_key")
                left_col,center_col,right_col=st.columns((3,4,4))
                with left_col:
                    st.write(f"Feature Name: :green[{column1}]")
                    st.write(df_median_imputed[column1])
                with center_col:
                    st.write("Contingency Table")
                    contingency_table = pd.crosstab(index=df_median_imputed[column1], columns=df_median_imputed[column2])
                    st.write(contingency_table)
                with right_col:
                    st.write(f"Feature Name: :green[{column2}]")
                    st.write(df_median_imputed[column2])
                cat_graphs=st.tabs(["Stacked Bar Plot","Grouped Bar Plot","Heatmap Plot (Contingency)"])
                with cat_graphs[0]:
                    if len(categorical_columns1) > 0 and len(categorical_columns2) > 0:
                        if column1 and column2:
                            fig=generate_stacked_bar_plot_bivariate(df_median_imputed, column1, column2)
                            st.plotly_chart(fig)
                with cat_graphs[1]:
                    if len(categorical_columns1) > 0 and len(categorical_columns2) > 0:
                        if column1 and column2:
                            fig=generate_grouped_bar_plot_bivariate(df_median_imputed, column1, column2)
                            st.plotly_chart(fig)
                with cat_graphs[2]:
                    if len(categorical_columns1) > 0 and len(categorical_columns2) > 0:
                        if column1 and column2:
                            fig=generate_heatmap_contingency_bivariate(contingency_table)
                            st.pyplot(fig)
                            buffer = BytesIO()
                            fig.savefig(buffer, format='png')
                            buffer.seek(0)
                            st.download_button(
                                label=':green[Download Graph as .PNG]',
                                data=buffer.getvalue(),
                                file_name='HeatMap_contegency.png',
                                mime='image/png',
                                key="do_key"
                            )
    st.session_state['df_median_imputed'] = df_median_imputed
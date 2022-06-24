import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from scipy.optimize import minimize
from scipy import stats

import seaborn as sns
from matplotlib import pyplot as plt

import streamlit as st

st.set_page_config(
    page_title="A/B Test App (CUPED version)", page_icon="📈"
)

def generate_data(treatment_effect, size): 
    # generate y from a normal distribution
    df = pd.DataFrame({'y': np.random.normal(loc=0, scale=1, size=size)})
    # create a covariate that's corrected with y 
    df['x'] = minimize(
        lambda x: 
        abs(0.95 - pearsonr(df.y, x)[0]), 
        np.random.rand(len(df.y))).x
    # random assign rows to two groups 0 and 1 
    df['group'] = np.random.randint(0, 2, df.shape[0])
    # for treatment group add a treatment effect 
    df.loc[df["group"] == 1, 'y'] += treatment_effect
    return df

def mean_t_test(df, group_column, group_name_a, group_name_b, y_col): 
    a = df[df[group_column] == group_name_a][y_col]
    b = df[df[group_column] == group_name_b][y_col]
    diff = a.mean() - b.mean()
    t, p_value = stats.ttest_ind(a, b)
    return diff, t, p_value

def cuped_mean_t_test(df, group_column, group_name_a, group_name_b, x_col, y_col): 
    theta = df.cov()[x_col][y_col] / df.cov()[x_col][x_col]
    df['metric_cuped'] = df[y_col] - theta * df[x_col]
    a = df.loc[df[group_column] == group_name_a, 'metric_cuped']
    b = df.loc[df[group_column] == group_name_b, 'metric_cuped']
    diff = a.mean() - b.mean()
    t, p_value = stats.ttest_ind(a, b)
    return diff, t, p_value

st.write(
    """
# 📈 A/B Test App (CUPED version)
"""
)

uploaded_file = st.file_uploader("📁 Upload CSV", type=".csv")


generate_random_data = st.checkbox(
    "Use random data", False, help="Use random data to demo the app"
)

covariate_default = None
metric_default = None
group_default = None

df = pd.DataFrame()

if generate_random_data:

    treatment_effect = st.number_input("treatment effect", value = 0.5)   # 加默认值
    size = st.number_input("data size", value = 100, step = 1)   # 加默认值

    if treatment_effect and size:
        df = generate_data(treatment_effect, size)
        covariate_default = ["x"]
        metric_default = ["y"]
        group_default = ["group"]

elif uploaded_file:
     df = pd.read_csv(uploaded_file)


if not df.empty:

    st.markdown("### 🔢 Data preview")
    st.dataframe(df.head())

    st.markdown("### 📍 Select columns for analysis")
    with st.form(key="my_form"):
        ab = st.multiselect(
            "Grouping column",
            options = df.columns,
            help = "Select which column refers to your A/B testing labels.",
            default= group_default,
        )
        
        if ab:
            control = df[ab[0]].unique()[0]
            treatment = df[ab[0]].unique()[1]
            decide = st.radio(
                f"Is *{treatment}* Group B?",
                options = ["Yes", "No"],
                help = "Select yes if this is group B (or the treatment group) from your test.",
            )
            if decide == "No":
                control, treatment = treatment, control
        
        covariate = st.multiselect(
            "Covariate column",
            options = df.columns,
            help = "Select which column is covariate.",
            default = covariate_default,
        )

        metric = st.multiselect(
            "Metric column",
            options = df.columns,
            help = "Select which column is metric.",
            default = metric_default,
        )
        submit_button = st.form_submit_button("Submit")

    if ab and covariate and metric:
        theta = df.cov()[covariate[0]][metric[0]] / df.cov()[covariate[0]][covariate[0]]
        df['CUPED'] = df[metric[0]] - theta * df[covariate[0]]

        st.markdown("### 🆚 Metric vs. Metric CUPED")
        
        st.markdown('#### 🔹 Metric')
        fig_y, ax_y = plt.subplots()
        sns.kdeplot(data=df, x=metric[0], hue=ab[0], fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
        st.pyplot(fig_y)

        st.markdown('#### 🔹 Metric CUPED')
        fig_cuped, ax_cuped = plt.subplots(figsize=(8,6))
        sns.kdeplot(data=df, x="CUPED", hue=ab[0], fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
        st.pyplot(fig_cuped)

        diff, t, p_value = mean_t_test(df, ab[0], control, treatment, metric[0])
        cuped_diff, cuped_t, cuped_p_value = cuped_mean_t_test(df, ab[0], control, treatment, covariate[0], metric[0])

        table = pd.DataFrame(
            {
                "diff": [diff, cuped_diff],
                "t": [t, cuped_t],
                "p-value": [p_value, cuped_p_value]
            },
            index = pd.Index(["Metric", "Metric CUPED"])
        )
        st.table(table)


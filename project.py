# ============================================================
# PART 1 : IMPORTS, CONFIGURATION, DATA LOADING & SIDEBAR FILTERS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM User Interaction Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("projectfile.csv")

df = load_data()
filtered_df = df.copy()

# ============================================================
# TITLE
# ============================================================

st.title("📊 LLM User Interaction Analytics Dashboard")

st.markdown("""
Analyze user interactions with Large Language Models using
interactive filters, statistical summaries and visualizations.
""")

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard Filters")
st.sidebar.markdown("### Categorical Filters")

# ----------------------------
# MODEL
# ----------------------------

selected_models = st.sidebar.multiselect(
    "🤖 Model Name",
    sorted(df["model_name"].dropna().unique()),
    default=sorted(df["model_name"].dropna().unique())
)

# ----------------------------
# DOMAIN
# ----------------------------

selected_domains = st.sidebar.multiselect(
    "🌍 Application Domain",
    sorted(df["application_domain"].dropna().unique()),
    default=sorted(df["application_domain"].dropna().unique())
)

# ----------------------------
# TASK TYPE
# ----------------------------

selected_tasks = st.sidebar.multiselect(
    "📄 Task Type",
    sorted(df["task_type"].dropna().unique()),
    default=sorted(df["task_type"].dropna().unique())
)

# ----------------------------
# RESPONSE
# ----------------------------

selected_response = st.sidebar.multiselect(
    "✅ Successful Response",
    sorted(df["successful_response"].dropna().unique()),
    default=sorted(df["successful_response"].dropna().unique())
)

# ----------------------------
# RAG
# ----------------------------

selected_rag = st.sidebar.multiselect(
    "📚 RAG Enabled",
    sorted(df["rag_enabled"].dropna().unique()),
    default=sorted(df["rag_enabled"].dropna().unique())
)

# ----------------------------
# HALLUCINATION
# ----------------------------

selected_hallucination = st.sidebar.multiselect(
    "⚠ Hallucination Flag",
    sorted(df["hallucination_flag"].dropna().unique()),
    default=sorted(df["hallucination_flag"].dropna().unique())
)

st.sidebar.divider()
st.sidebar.markdown("### Numerical Filters")

# ----------------------------
# PROMPT LENGTH
# ----------------------------

prompt_range = st.sidebar.slider(
    "Prompt Length",
    int(df["prompt_length"].min()),
    int(df["prompt_length"].max()),
    (
        int(df["prompt_length"].min()),
        int(df["prompt_length"].max())
    )
)

# ----------------------------
# TOKENS
# ----------------------------

token_range = st.sidebar.slider(
    "Total Tokens",
    int(df["total_tokens"].min()),
    int(df["total_tokens"].max()),
    (
        int(df["total_tokens"].min()),
        int(df["total_tokens"].max())
    )
)

# ----------------------------
# LATENCY
# ----------------------------

latency_range = st.sidebar.slider(
    "Latency (Seconds)",
    float(df["latency_sec"].min()),
    float(df["latency_sec"].max()),
    (
        float(df["latency_sec"].min()),
        float(df["latency_sec"].max())
    )
)

# ----------------------------
# COST
# ----------------------------

cost_range = st.sidebar.slider(
    "Estimated Cost ($)",
    float(df["estimated_cost_usd"].min()),
    float(df["estimated_cost_usd"].max()),
    (
        float(df["estimated_cost_usd"].min()),
        float(df["estimated_cost_usd"].max())
    )
)

# ----------------------------
# SATISFACTION
# ----------------------------

rating_range = st.sidebar.slider(
    "User Satisfaction",
    float(df["user_satisfaction"].min()),
    float(df["user_satisfaction"].max()),
    (
        float(df["user_satisfaction"].min()),
        float(df["user_satisfaction"].max())
    )
)

# ----------------------------
# TEMPERATURE
# ----------------------------

temperature_range = st.sidebar.slider(
    "Temperature",
    float(df["temperature"].min()),
    float(df["temperature"].max()),
    (
        float(df["temperature"].min()),
        float(df["temperature"].max())
    )
)

# ----------------------------
# TOP P
# ----------------------------

top_p_range = st.sidebar.slider(
    "Top P",
    float(df["top_p"].min()),
    float(df["top_p"].max()),
    (
        float(df["top_p"].min()),
        float(df["top_p"].max())
    )
)

# ============================================================
# APPLY FILTERS
# ============================================================

filter_values = {
    "models": selected_models,
    "domains": selected_domains,
    "tasks": selected_tasks,
    "response": selected_response,
    "rag": selected_rag,
    "hallucination": selected_hallucination,
    "prompt": prompt_range,
    "tokens": token_range,
    "latency": latency_range,
    "cost": cost_range,
    "rating": rating_range,
    "temperature": temperature_range,
    "top_p": top_p_range,
}

if st.sidebar.button("Apply Filters"):
    st.session_state.applied_filters = filter_values

applied_filters = st.session_state.setdefault("applied_filters", filter_values)

filtered_df = df[
    (df["model_name"].isin(applied_filters["models"])) &
    (df["application_domain"].isin(applied_filters["domains"])) &
    (df["task_type"].isin(applied_filters["tasks"])) &
    (df["successful_response"].isin(applied_filters["response"])) &
    (df["rag_enabled"].isin(applied_filters["rag"])) &
    (df["hallucination_flag"].isin(applied_filters["hallucination"])) &
    (df["prompt_length"].between(*applied_filters["prompt"])) &
    (df["total_tokens"].between(*applied_filters["tokens"])) &
    (df["latency_sec"].between(*applied_filters["latency"])) &
    (df["estimated_cost_usd"].between(*applied_filters["cost"])) &
    (df["user_satisfaction"].between(*applied_filters["rating"])) &
    (df["temperature"].between(*applied_filters["temperature"])) &
    (df["top_p"].between(*applied_filters["top_p"]))
]

st.sidebar.divider()
st.sidebar.success(f"Filtered Records : {len(filtered_df)}")

# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

st.header("Executive Summary")

st.write("""
The LLM User Interaction Analytics Dashboard is designed to analyze and
visualize user interactions with Large Language Models (LLMs). As the use
of AI-powered applications continues to grow across different domains,
understanding user behavior, model performance, response quality, latency,
and operational cost has become increasingly important.

This dashboard provides an interactive platform for exploring LLM interaction
data using dynamic filters, statistical summaries, and professional
visualizations. Users can evaluate model performance, compare application
domains, analyze response latency, monitor token usage, estimate API costs,
measure user satisfaction, and identify successful and unsuccessful responses.

Developed using Python, Streamlit, Pandas, NumPy, Matplotlib, Seaborn,
and Plotly Express, the dashboard enables efficient data exploration and
supports data-driven decision-making. It offers an intuitive interface
that simplifies complex analytical tasks and provides meaningful insights
into Large Language Model usage across different applications.
""")


st.markdown("---")

# ==========================================================
# PROJECT DESCRIPTION
# ==========================================================

st.header("Project Description")

st.write("""
The LLM User Interaction Analytics Dashboard is an interactive data
visualization and analytics application developed to analyze user
interactions with Large Language Models (LLMs). The dashboard provides
a comprehensive overview of user activity, model performance, response
quality, latency, token consumption, API cost, and user satisfaction.

The project enables users to explore the dataset through interactive
categorical and numerical filters, making it easier to compare different
LLM models, application domains, and task types. It also includes
professional charts and statistical summaries that help identify
performance trends, usage patterns, and operational insights.

The dashboard has been developed using Streamlit, Pandas, NumPy,
Plotly Express, Matplotlib, and Seaborn. These tools provide an
interactive interface for efficient data exploration and support
data-driven decision-making by transforming raw interaction data
into meaningful business insights.
""")

st.markdown("---")


# ============================================================
# PART 2 : DASHBOARD OVERVIEW & DATASET INFORMATION
# ============================================================

st.header("📋 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Total Columns",
        filtered_df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        filtered_df.isnull().sum().sum()
    )

with col4:
    st.metric(
        "Duplicate Records",
        filtered_df.duplicated().sum()
    )

st.divider()

# ============================================================
# DATASET PREVIEW
# ============================================================

st.header("🗂 Dataset Preview")

rows = st.selectbox(
    "Number of Records",
    [5,10,15,20,25,50],
    index=1
)

st.dataframe(
    filtered_df.head(rows),
    use_container_width=True,
    height=350
)

st.divider()

# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📑 Dataset Information")

info = pd.DataFrame({
    "Column Name": filtered_df.columns,
    "Data Type": filtered_df.dtypes.astype(str),
    "Missing Values": filtered_df.isnull().sum().values,
    "Unique Values": filtered_df.nunique().values
})

st.dataframe(
    info,
    use_container_width=True
)

st.divider()

# ============================================================
# NUMERICAL SUMMARY
# ============================================================

st.header("🔢 Numerical Data Summary")

numeric_columns = filtered_df.select_dtypes(include=np.number).columns

st.dataframe(
    filtered_df[numeric_columns].describe().T,
    use_container_width=True
)

st.divider()

# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

st.header("📂 Categorical Data Summary")

categorical_columns = filtered_df.select_dtypes(include="object").columns

cat_summary = pd.DataFrame({
    "Column": categorical_columns,
    "Unique Values":[filtered_df[col].nunique() for col in categorical_columns],
    "Most Frequent":[
        filtered_df[col].mode()[0]
        if not filtered_df[col].mode().empty
        else ""
        for col in categorical_columns
    ]
})

st.dataframe(
    cat_summary,
    use_container_width=True
)

st.divider()

# ============================================================
# MISSING VALUES
# ============================================================

st.header("❗ Missing Values")

missing_df = pd.DataFrame({
    "Column": filtered_df.columns,
    "Missing Values": filtered_df.isnull().sum().values
})

fig = px.bar(
    missing_df,
    x="Column",
    y="Missing Values",
    color="Missing Values",
    text="Missing Values",
    title="Missing Values by Column"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ============================================================
# DUPLICATE RECORDS
# ============================================================

st.header("📄 Duplicate Records")

duplicates = filtered_df[filtered_df.duplicated()]

st.write("Total Duplicate Records :", len(duplicates))

if len(duplicates) > 0:
    st.dataframe(
        duplicates,
        use_container_width=True
    )
else:
    st.success("No duplicate records found.")

st.divider()

# ============================================================
# PART 3 : DATA VISUALIZATIONS
# ============================================================

st.header("📊 Data Visualizations")

# ============================================================
# BAR CHART
# ============================================================

st.subheader("📌 Bar Chart:Model-wise Session Count")

bar_data = filtered_df.groupby(
    ["model_name", "successful_response"]
).size().reset_index(name="Count")

fig = px.bar(
    bar_data,
    x="model_name",
    y="Count",
    color="successful_response",
    barmode="group",
    text="Count",
    title="Sessions by LLM Model"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Mistral Large recorded the highest number of sessions among the available models.
- Llama 3.1 recorded comparatively fewer sessions than the other models.
- The chart shows noticeable differences in session volume across LLM models.
""")
st.divider()

# ============================================================
# PIE CHART
# ============================================================

st.subheader("🥧 Pie Chart:Application Domain Distribution")

fig = px.pie(
    filtered_df,
    names="application_domain",
    title="Application Domain Distribution"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Customer Support has the highest share of application-domain interactions.
- Education and Retail also contribute a significant proportion of total sessions.
- Legal records the lowest share among the major application domains.
""")
st.divider()

# ============================================================
# HISTOGRAM
# ============================================================

st.subheader("📈 Histogram: Response Latency ")

fig = px.histogram(
    filtered_df,
    x="latency_sec",
    color="model_name",
    nbins=30,
    title="Latency Distribution"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Most responses are concentrated within the middle latency range.
- Response latency varies noticeably across different LLM models.
- The distribution helps identify both typical and higher-latency interactions.
""")
st.divider()

# ============================================================
# BOX PLOT
# ============================================================

st.subheader("📦 Box Plot: Estimated API Cost by Model")

fig = px.box(
    filtered_df,
    x="model_name",
    y="estimated_cost_usd",
    color="model_name",
    title="Estimated Cost Distribution"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Claude 3.7 shows the highest average estimated API cost among the models.
- Gemma 3 records comparatively lower average interaction costs.
- Cost variation across models highlights differences in operational efficiency.
""")
st.divider()

# ============================================================
# VIOLIN PLOT
# ============================================================

st.subheader("🎻 Violin Plot: User Satisfaction Distribution")

fig = px.violin(
    filtered_df,
    x="model_name",
    y="user_satisfaction",
    color="model_name",
    box=True,
    points="all"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Claude 3.7 records the highest average user satisfaction.
- GPT-4o also shows a relatively high satisfaction level.
- Llama 3.1 has the comparatively lowest average satisfaction score.
""")
st.divider()

# ============================================================
# BUBBLE CHART
# ============================================================

st.subheader("🫧 Bubble Chart: Latency Vs User Satisfaction")

fig = px.scatter(
    filtered_df,
    x="latency_sec",
    y="user_satisfaction",
    size="total_tokens",
    color="model_name",
    hover_data=["application_domain"]
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Higher response latency is generally associated with lower user satisfaction.
- The chart indicates a moderate negative relationship between latency and satisfaction.
- Bubble size shows token usage, helping compare processing volume with performance.
""")
st.divider()

# ============================================================
# LINE CHART
# ============================================================

st.subheader("📉 Line Chart")

line_df = (
    filtered_df
    .groupby(["prompt_length", "model_name"])["latency_sec"]
    .mean()
    .reset_index()
)

fig = px.line(
    line_df,
    x="prompt_length",
    y="latency_sec",
    color="model_name",
    markers=True,
    title="Average Latency vs Prompt Length"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Average response latency generally increases as prompt length increases.
- Longer prompts require comparatively more processing time.
- Latency patterns vary across the different LLM models.
""")
st.divider()

# ============================================================
# SUNBURST CHART
# ============================================================

st.subheader("☀️ Sunburst Chart:Application Domain and Task Type Distribution")

fig = px.sunburst(
    filtered_df,
    path=[
        "application_domain",
        "task_type"
    ]
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Customer Support represents one of the largest application-domain segments.
- Code Generation, QA, Summarization and Translation appear across multiple domains.
- The chart highlights how task types are distributed within each application domain.
""")
st.divider()

# ============================================================
# TREEMAP
# ============================================================

st.subheader("🌳 Treemap: Model Usage Across Application Domain")

fig = px.treemap(
    filtered_df,
    path=[
        "model_name",
        "application_domain"
    ],
    color="estimated_cost_usd"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Mistral Large shows strong usage across multiple application domains.
- Model usage varies noticeably between different application domains.
- Estimated cost differences are visible through the treemap structure.
""")
st.divider()

# ============================================================
# DENSITY HEATMAP
# ============================================================

st.subheader("🔥 Density Heatmap")

fig = px.density_heatmap(
    filtered_df,
    x="prompt_length",
    y="total_tokens"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Prompt length and total token usage show a strong positive relationship.
- Longer prompts generally result in higher total token consumption.
- The densest regions represent the most frequently occurring combinations.
""")
st.divider()

# ============================================================
# FUNNEL CHART
# ============================================================

st.subheader("🎯 Funnel Chart")

funnel_df = (
    filtered_df["successful_response"]
    .value_counts()
    .reset_index()
)

funnel_df.columns = [
    "successful_response",
    "Count"
]

fig = px.funnel(
    funnel_df,
    x="Count",
    y="successful_response"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Unsuccessful responses account for a much larger share of recorded sessions.
- Successful responses represent a smaller proportion of the overall interactions.
- The distribution highlights an important opportunity to improve response success.
""")
st.divider()

# ============================================================
# 3D SCATTER PLOT
# ============================================================

st.subheader("🌐 3D Scatter Plot:3D Analysis of Prompt Length, Total Tokens and Latency")

fig = px.scatter_3d(
    filtered_df,
    x="prompt_length",
    y="total_tokens",
    z="latency_sec",
    color="model_name",
    hover_data=[
        "application_domain",
        "task_type"
    ]
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🔑 Key Insights")
st.write("""
- Prompt length and total tokens increase together across the dataset.
- Higher input and token volumes are generally associated with increased latency.
- The 3D view combines input size, token consumption and response performance.
""")
st.divider()

# ============================================================
# PART 4 : KEY INSIGHTS & ADVANCED ANALYTICS
# ============================================================

st.header("🔍 Key Insights")

# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sessions",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Unique Models",
        filtered_df["model_name"].nunique()
    )

with col3:
    st.metric(
        "Application Domains",
        filtered_df["application_domain"].nunique()
    )

with col4:
    success_rate = (
        filtered_df["successful_response"]
        .value_counts(normalize=True)
        .get(True, 0) * 100
    )

    st.metric(
        "Success Rate",
        f"{success_rate:.2f}%"
    )

st.divider()


# ============================================================
# NUMERICAL INSIGHTS
# ============================================================

st.subheader("📈 Numerical Insights")

num_summary = filtered_df.describe().transpose()

st.dataframe(
    num_summary,
    use_container_width=True
)

st.divider()

# ============================================================
# CATEGORICAL INSIGHTS
# ============================================================

st.subheader("📂 Categorical Insights")

cat_summary = pd.DataFrame({
    "Column": filtered_df.select_dtypes(include="object").columns,
    "Unique Values":[
        filtered_df[col].nunique()
        for col in filtered_df.select_dtypes(include="object").columns
    ],
    "Most Frequent":[
        filtered_df[col].mode()[0]
        for col in filtered_df.select_dtypes(include="object").columns
    ]
})

st.dataframe(
    cat_summary,
    use_container_width=True
)

st.divider()


# ============================================================
# PART 5 : EXTRA ANALYTICS & PROJECT SUMMARY
# ============================================================

st.header("📌 Extra Analytics")

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Model Performance")

model_summary = filtered_df.groupby("model_name").agg(
    Total_Sessions=("model_name", "count"),
    Avg_Latency=("latency_sec", "mean"),
    Avg_Cost=("estimated_cost_usd", "mean"),
    Avg_Satisfaction=("user_satisfaction", "mean"),
    Avg_Tokens=("total_tokens", "mean")
).round(2)

st.dataframe(
    model_summary,
    use_container_width=True
)

st.divider()

# ============================================================
# APPLICATION DOMAIN SUMMARY
# ============================================================

st.subheader("🌍 Application Domain Summary")

domain_summary = filtered_df.groupby(
    "application_domain"
).agg(
    Sessions=("application_domain", "count"),
    Avg_Satisfaction=("user_satisfaction", "mean"),
    Avg_Latency=("latency_sec", "mean")
).round(2)

st.dataframe(
    domain_summary,
    use_container_width=True
)

st.divider()

# ============================================================
# TASK TYPE SUMMARY
# ============================================================

st.subheader("📄 Task Type Summary")

task_summary = filtered_df.groupby(
    "task_type"
).agg(
    Sessions=("task_type", "count"),
    Avg_Cost=("estimated_cost_usd", "mean"),
    Avg_Tokens=("total_tokens", "mean")
).round(2)

st.dataframe(
    task_summary,
    use_container_width=True
)


st.divider()

# ============================================================
# RAG ENABLED SUMMARY
# ============================================================

st.subheader("📚 RAG Enabled Summary")

rag_summary = filtered_df.groupby(
    "rag_enabled"
).size().reset_index(name="Sessions")

fig = px.pie(
    rag_summary,
    names="rag_enabled",
    values="Sessions"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ============================================================
# HALLUCINATION SUMMARY
# ============================================================

st.subheader("⚠ Hallucination Summary")

hall_summary = filtered_df.groupby(
    "hallucination_flag"
).size().reset_index(name="Sessions")

fig = px.bar(
    hall_summary,
    x="hallucination_flag",
    y="Sessions",
    color="hallucination_flag",
    text="Sessions"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ============================================================
# PROJECT SUMMARY
# ============================================================

st.header("📖 Project Summary")

st.info("""
This dashboard analyzes LLM User Interaction Analytics using
interactive filters, statistical summaries and professional
visualizations.

The dashboard helps users compare different LLM models,
application domains, response latency, token usage,
estimated API cost and user satisfaction.

Interactive charts allow easy identification of trends,
patterns and performance differences among language models.
""")
# ============================================================
# CONCLUSION
# ============================================================

st.header("✅ Conclusion")

st.write("""
This dashboard provides a comprehensive analysis of LLM user interactions.
It enables users to explore model performance, response latency, token usage,
API cost, user satisfaction, application domains, and task types through
interactive filters, professional visualizations, and statistical insights.
The dashboard supports data-driven decision-making by presenting meaningful
patterns and trends in Large Language Model usage.
""")

st.divider()



st.caption(
    "Developed using Streamlit • Pandas • NumPy • Plotly Express • Matplotlib • Seaborn"
)
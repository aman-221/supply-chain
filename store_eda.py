import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.mosaicplot import mosaic
import streamlit as st
import datetime

st.set_option("deprecation.showPyplotGlobalUse", False)


def main():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Year"] = df["Order Date"].dt.year
    df["Order day"] = df["Order Date"].dt.day

    st.title("Module 2 Assignment #1: Predicting Sales Data - Aman Gupta")
    st.write("Raw data from Kaggle")
    st.dataframe(df)
    cats = [
        "Ship Mode",
        "Segment",
        "State",
        "Region",
        "Category",
        "Sub-Category",
    ]
    st.write("Yearly sales breakdown by categorical feature")
    option = st.selectbox("What categorical feature would you like to visualize", cats)

    t = (
        df.groupby([str(option), "Order Year"])["Sales"]
        .sum()
        .reset_index()
        .sort_values(by="Sales", ascending=False)
    )

    ax = sns.lineplot(data=t, x="Order Year", y="Sales", hue=option)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot()

    st.write("Overall Sales breakdown by date")
    cats2 = [2015, 2016, 2017, 2018]
    option2 = st.selectbox("What year would you like to see sales", cats2)

    c_df = df[df["Order Year"] == option2]

    p = c_df.groupby(["Order Date"])["Sales"].sum().reset_index()

    ax2 = sns.lineplot(data=p, x="Order Date", y="Sales")

    plt.tight_layout()
    st.pyplot()

    d1 = st.date_input("Date Start Value", datetime.date(2015, 1, 3))

    # d2 = st.date_input("Date End Value", datetime.date(2018, 12, 30))

    concat_df = df[(df["Order Date"] >= pd.to_datetime(d1))]

    st.write("Historical Sales & SMA forecast by date and sub-category")
    option3 = st.selectbox("Select a sub-category", df["Sub-Category"].unique())
    concat_df = concat_df[concat_df["Sub-Category"] == option3]

    window_size = st.number_input("Select a window size")
    p2 = concat_df.groupby(["Order Date"])["Sales"].sum().reset_index()
    numbers_series = pd.Series(p2["Sales"])
    p2["SMA"] = numbers_series.rolling(int(window_size), min_periods=1).mean()

    df2 = pd.melt(p2, "Order Date", var_name="Measure", value_name="Value")

    sns.lineplot(data=df2, x="Order Date", y="Value", hue="Measure")

    plt.tight_layout()
    st.pyplot()


if __name__ == "__main__":
    main()

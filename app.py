from scipy import stats
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.title('Data Analyzer')
st.write('Load and visualize run data from driving swarm')

def create_df_last(data):
    return data.sort_values(["experiment","mode", "algorithm_variant", "robot", "t"]).groupby(["experiment","mode", "algorithm_variant", "robot"], observed=True).tail().reindex()

def aggregation(data):
    ret = data.groupby("goal_completed").size().reset_index()[:-1].reindex()
    ret["ttg"] = ret[0]
    del ret[0]
    #st.dataframe(ret.head())
    return ret

def create_df_time(data):
    return data.groupby(["db3","mode", "algorithm_variant", "n","robot"], observed=True).apply(aggregation).reset_index()

@st.cache_data
def load_data(filename=None):
    if filename is None:
        filename = '~/data/few_long_runs.prq'
    st.write(filename)
    df = pd.read_parquet(filename)
    df["algorithm_variant"] = df.algorithm +':'+ df.algorithm_params
    df = df.sort_values(by=['experiment', 'robot', 'timestamp'])
    df.robot_id = df.robot_id.astype("category")
    df.pair_id = df.pair_id.astype("category")
    df.experiment = df.experiment.astype("category")
    #df.db3 = df.db3.astype("category")
    #df.algorithm = df.algorithm.astype("category")
    #df.algorithm_params = df.algorithm_params.astype("category")
    #df.algorithm_variant = df.algorithm_variant.astype("category")

    
    return df, create_df_last(df), create_df_time(df)

datafile = st.file_uploader('Upload data', accept_multiple_files=False)


df_all, df_last, df_time = load_data(datafile)
st.markdown(f'''
## Loading Data
Loaded data.

We got a total of {len(df_all.db3.unique())} experiments.
With columns:\n
{df_all.columns}
''')
st.table(df_all.columns)
st.dataframe(df_last.head())

st.markdown(f'''
### Overall goals completed

Over the dataset we have {df_last.goal_completed.sum():_} goals completed.

Goals completed by mode (simulation and discrete simulation).
''')

st.dataframe(df_last.groupby(["mode", "algorithm_variant"]).goal_completed.median())

fig = px.box(df_last, color="algorithm_variant", x='n', y="goal_completed", facet_col="mode")
st.plotly_chart(fig)

st.write("full dataframe with goals completed:")
st.dataframe(df_last.head())


##########################ü

st.markdown(f'''
### Time to goal

Over the dataset we have a median of {df_time.ttg.median()} seconds to goal.

''')

#st.dataframe(df_time.groupby(["mode", "algorithm_variant"]).ttg.median())

st.write("full dataframe with time to goal:")
st.dataframe(df_time)

fig = px.box(df_time, color="algorithm_variant", y="ttg", facet_col='mode', x="n")
st.plotly_chart(fig)

for n in df_time.n.unique():
    for mode in df_time["mode"].unique():
        group_labels = [a for a in df_time.algorithm_variant.unique()]
        groups = [df_time.loc[df_time.n.eq(n) & df_time['mode'].eq(mode) & df_time.algorithm_variant.eq(algorithm_variant), 'ttg'] for algorithm_variant in group_labels]
        groups = [group for group in groups if len(group) > 0]
        #fig = ff.create_distplot(groups, group_labels)
        fig = px.histogram(df_time.loc[df_time.n.eq(n) & df_time['mode'].eq(mode)], x="ttg", color="algorithm_variant", marginal="box", nbins=500)
        st.write(f"TTG: n={n}, mode: {mode}")
        st.plotly_chart(fig)

## HGT
def hgt_column(data, reference="global_planner:"):
    
    def man_whitney_u(group):
        if not len(group):
            return np.nan
        if not len(data.loc[data.algorithm_variant.eq(reference), "ttg"]):
            return np.nan
        p = stats.mannwhitneyu(group, data.loc[data.algorithm_variant.eq(reference), "ttg"])[1]
        return p
    
    def significance(group):
        return "*" if man_whitney_u(group) < 0.05 else ""

    column = data.groupby("algorithm_variant").ttg.agg([("significance", significance), ("mean", np.mean), ("std", np.std), ("median", np.median), ("iqr", stats.iqr), ("p", man_whitney_u)]).reset_index()
    
    
    return column

def hgt(data):
    return data.groupby(["mode", "n"]).apply(hgt_column).reset_index()

st.title("HGT")
st.dataframe(hgt(df_time))

st.markdown('''
### Analysing autocorrelation

$\\forall$ experiment, we compute an ACV and PACV plot.
        
The autocorrelation works as follows:
''')

def autocorr(x):
    x.sort_values(["db3","robot","goal_completed"], inplace=True)
    return x.ttg.autocorr(lag=1)

st.dataframe(df_time.groupby(["mode", "n", "algorithm_variant"]).apply(autocorr).reset_index())

with st.expander("Autocorrelation plots"):
    for mode in df_time["mode"].unique():
        for algorithm_variant in df_time.algorithm_variant.unique():
            for n in df_time.n.unique():
                df = df_time.loc[
                    df_time.n.eq(n) & df_time.algorithm_variant.eq(algorithm_variant) & df_time['mode'].eq(mode)
                    ]
                st.write(f"mode: {mode}, algorithm_variant: {algorithm_variant}, n: {n}")
                df.sort_values(["robot", "db3", "goal_completed"], inplace=True)
                plt.figure()
                fig = pd.plotting.autocorrelation_plot(df.ttg).figure
                plt.gca().set_xlim([1, 10])
                #.set_xlim([0,10])
                st.pyplot(fig)
                plt.close()
                plt.figure()
                fig = pd.plotting.lag_plot(df.ttg, alpha=0.1).figure
                st.pyplot(fig)
                plt.close()
            
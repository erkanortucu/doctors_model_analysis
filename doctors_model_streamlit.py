import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import os
import openpyxl
from scipy.stats import shapiro
from scipy.stats import kruskal


st.set_page_config(layout="wide")
#st.set_page_config(layout="centered")

fl = st.file_uploader(":file_folder: Upload a file", type=(["xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_excel(filename, engine="openpyxl")
else:
    os.chdir(r"C:\Users\erkan\Desktop\Upwork\Doctor vs model analysis")
    df = pd.read_excel("Doctor vs model analysis 20.12.2023.xlsx", engine="openpyxl")

#df = data.copy()

# streamlit run .\doctors_vs_model_analysis\doctors_model_streamlit.py


pd.set_option("display.max_columns", 500)
pd.set_option('display.expand_frame_repr', False)

#####  stremlit ########

st.markdown("<div style='text-align: center;background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'>Doctor vs Model Analysis - 20.12.2023 </h1></div>",
            unsafe_allow_html=True)

st.write(" ")
st.write(" ")

st.text("  In our experiment, we compared the gold standard diagnosis, the doctor's prediction, and the model prediction")
st.text("for a specific case.")
st.write(" ")
st.text("  Both the doctor and the model were given the patient's history and asked to provide their disease predictions.")
st.text("The gold standard diagnosis represents the actual diagnosis of the patient at the end of their diagnostic journey")
st.text("This experiment aimed to evaluate how well the doctor's prediction and the model's prediction aligned ")
st.text("with the gold standard diagnosis")
st.write(" ")
st.text("Its purpose is to compare the cancer prediction abilities of experts (doctors) and the model through various ")
st.text("statistical tests such as specificity and sensitivity")
st.write(" ")
st.text("-The doctor's prediction (DP1), (DP2), (DP3) compared to gold standard")
st.text("-The comparision of the model's prediction (MP1), (MP2), (MP3)  compared to gold standard")
st.text("-AI Prediction -1,-2,-3 : If doctor agree with model prediction (MP1), (MP2), (MP3)")
st.text("-AI Recommendation -1,-2,-3 :If doctor agree with model recommendation (MP1), (MP2), (MP3)")



st.write(" ")
st.write(" ")

#data = pd.read_excel(r"C:\Users\erkan\Desktop\Upwork\Doctor vs model analysis\Doctor vs model analysis 20.12.2023.xlsx")
#df = data.copy()
# last rows is Na value.
df.dropna(inplace=True)


# We edit the columns name
df.columns =['Sl_No', 'Case_No', 'User_name', 'Total_time_doctor',
       'Total_time_model', 'DP1vsGold_St', 'DP2vsGold_St',
       'DP3vsGold_St', 'MP1vsGold_St', 'MP2vsGold_St',
       'MP3vsGold_St', 'AI_Output_pre_1', 'AI_Output_pre_2',
       'AI_Output_pre_3', 'AI_Output_rec_1', 'AI_Output_rec_2',
       'AI_Output_rec_3']

# we changed the dtype from object to float  "Total_time_doctor", "Sl_No"

# Extract minutes and seconds using regular expressions
df[['Minutes', 'Seconds']] = df['Total_time_doctor'].str.extract(r'(\d+) min (\d+) sec')

df['Total_time_doctor'] = df['Minutes'].astype(str) + "." +df['Seconds'].astype(str)

# Format the result
df['Total_time_doctor'] = df['Total_time_doctor'].astype(float)

drop_cols = ["Minutes","Seconds"]
df.drop(drop_cols, axis=1, inplace=True)

df["Sl_No"] = df["Sl_No"].astype("int")
df["Sl_No"] = df["Sl_No"].astype("O")

# change "MissD" to "Mismatch"

df["DP1vsGold_St"] = df["DP1vsGold_St"].replace("MissD", "Mismatch")

#####  stremlit ########

st.markdown("<h1 style='text-align: left; color: black; font-size: 20px; font-weight: bold;'>DataSet  </h1>", unsafe_allow_html=True)
st.dataframe(df)

datasetinf1, datasetinf2 = st.columns((2))

with datasetinf1:
    st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Number of Observations : {}</h1>".format(
                df.shape[0]), unsafe_allow_html=True)
with datasetinf2:
    st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Number of Columns ( Variables ) : {}</h1>".format(
            df.shape[1]), unsafe_allow_html=True)

st.write("")

st.write(" ")

st.markdown("<div style='text-align: center;background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'>Total Time Doctor vs Model Summary Statistics </h1></div>",
            unsafe_allow_html=True)

st.write(" ")

st.write(df.describe([0.25, 0.50, 0.75, 0.95, 0.99]).T)

st.write("")

st.text("Doctors predict time mean is : 4.53 (4 min 53 sec)")
st.text("Model predict time mean is : 0.32 (32 sec) and medain is 0.32 (32 sec)")


st.write("")

st.write("")

datasetdistplot1, datasetdistplot2 , datasetdistplot3, datasetdistplot4= st.columns((4))

with datasetdistplot1:
    st.markdown("<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Time Doctor Histogram Plot  </h1>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(df['Total_time_doctor'], kde=True, ax=ax)
    ax.set_xlabel('Total Time  Doctors', fontsize=7)
    ax.set_ylabel('Frequency', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    st.pyplot(fig)

with datasetdistplot3:
    st.markdown( "<h1 style='text-align: left; color: black; font-size: 18px; font-weight: bold;'> Total Time Model Histogram Plot </h1>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(df["Total_time_model"], kde=True, ax=ax)
    ax.set_xlabel('Total Time  Model', fontsize=7)
    ax.set_ylabel('Frequency', fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    st.pyplot(fig)

# Shapiro-Wilks Test  ( The Shapiro–Wilk test is a test of normality  )



hypotcol1, hypotcol2 = st.columns((2))

with hypotcol1 :
    st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Test of normality(Shapiro–Wilk test) Doctors Total Time </h1>"
        , unsafe_allow_html=True)

    Test_statistic, P_value = shapiro(df["Total_time_doctor"])
    formatted_test_statistic = f"{Test_statistic:.3f}"
    formatted_p_value = f"{P_value:.3f}"

    st.markdown(f"<p style='font-size: 15px ; font-weight: bold;'>test statistic: {formatted_test_statistic}, p-value: {formatted_p_value}</p>",
        unsafe_allow_html=True)
    if P_value < 0.05:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Reject the (H0) null hypothesis. Not a normal distribution</h1>"
            , unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Null hypothesis (H0) cannot be rejected. Normal distribution</h1>"
            , unsafe_allow_html=True)

    st.text("Doctors time data is not normally distributed")

with hypotcol2 :
    st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Test of normality(Shapiro–Wilk test) Model Total Time </h1>"
        , unsafe_allow_html=True)
    Test_statistic, P_value = shapiro(df["Total_time_model"])
    formatted_test_statistic = f"{Test_statistic:.3f}"
    formatted_p_value = f"{P_value:.3f}"

    st.markdown(f"<p style='font-size: 15px ; font-weight: bold;'>test statistic: {formatted_test_statistic}, p-value: {formatted_p_value}</p>",
        unsafe_allow_html=True)
    if P_value < 0.05:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Reject the (H0) null hypothesis. Not a normal distribution</h1>"
            , unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Null hypothesis (H0) cannot be rejected. Normal distribution</h1>"
            , unsafe_allow_html=True)
    st.text("Model time data is normally distributed")


## DP1 vs MP1  #############################################################################

st.write("")
st.write("")
st.write("")
st.write("")

dp1_mp1_df = pd.DataFrame({"DP1 Count": df["DP1vsGold_St"].value_counts(),
                               "DP1 Ratio": df["DP1vsGold_St"].value_counts() / df.shape[0] * 100,

                               "MP1 Count": df["MP1vsGold_St"].value_counts(),
                               "MP1 Ratio": df["MP1vsGold_St"].value_counts() / df.shape[0] * 100})

aipre1_aiout1_df= pd.DataFrame({"AI Prediction-1 Count": df["AI_Output_pre_1"].value_counts(),
              "AI Prediction-1 Ratio": df["AI_Output_pre_1"].value_counts() / df.shape[0]*100,

              "AI Recommendation-1 Count": df["AI_Output_rec_1"].value_counts(),
              "AI Recommendation-1 Ratio": df["AI_Output_rec_1"].value_counts() / df.shape[0]*100})

st.markdown("<div style='text-align: center; background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'> DP1 vs MP1 (Doctors Prediction 1 vs Model Prediction 1) </h1></div>",
            unsafe_allow_html=True)
st.write("")
st.write("")
dp1_mp1_1,dp1_mp1_2, dp1_mp1_3 = st.columns(3)

with dp1_mp1_1:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp1_mp1_df["DP1 Count"], labels=dp1_mp1_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Doctors Prediction 1',fontsize = 7)
    st.pyplot()

with dp1_mp1_2:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp1_mp1_df["MP1 Count"], labels=dp1_mp1_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Model Prediction 1', fontsize=7)
    st.pyplot()


st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>DP1 vs MP1 Table </h1>"
        ,unsafe_allow_html=True)
st.dataframe(dp1_mp1_df)
st.write("")
st.text("- The doctors' predictions (DP1) Match rate is 37.93 % ,the model predictions (MP1)  Match rate is 63.79 %")

st.text("- The doctors' predictions (DP1) Mismatch rate is 28.74 % ,the model predictions (MP1) Mismatch rate is 13.22 %")

st.text("- Model predictions results have a higher Match rate")

st.write("")
st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)

st.write("")
st.text("With 95% reliability, the test result for the bellow variable :")
st.text("Test Statistic = 3.20, p-value = 0.53, Independent (H0 holds true) , p > 0.05 ")
st.text("* DP1 (doctors' predictions) and MP1 (model predictions) do not have a significant relation")

st.write("")

dp1_mp1_4,dp1_mp1_5, dp1_mp1_6 = st.columns(3)

with dp1_mp1_4:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre1_aiout1_df["AI Prediction-1 Count"], labels=aipre1_aiout1_df.index, colors=palette_color,
            autopct='%.2f%%', textprops={'fontsize': 7})
    plt.title('Distribution of  AI Prediction 1', fontsize=7)
    st.pyplot()

with dp1_mp1_5:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre1_aiout1_df["AI Recommendation-1 Count"], labels=aipre1_aiout1_df.index, colors=palette_color,
            autopct='%.2f%%', textprops={'fontsize': 7})
    plt.title('Distribution of AI Recommendation 1', fontsize=7)
    st.pyplot()




st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>AI Predictiın-1 vs "
            "AI Recommendation-1 Table </h1>"
                , unsafe_allow_html=True)
st.write("")
st.dataframe(aipre1_aiout1_df)
st.text("76.44% doctors are agree with model prediction-1 (MP1) and 87.36% agree with model recommendation-1")
st.write("")
st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)

st.write("")
st.text("With 95% reliability, the test result for the bellow variables :")
st.write("")
st.text("* AI prediction-1 and AI recommendation-1 have a significant relation. Cramer's phi = 0.20")
st.text("Test Statistic = 5.38, p-value = 0.02 , Dependent (reject H0) , p < 0.05")
st.text("There is a positive directional relation and a small effect size of 0.20")
st.write("")
st.text("* DP1 (doctors' predictions) and AI prediction-1 have a significant relation. Cramer's V = 0.30")
st.text("Test Statistic = 15.88, p-value = 0.00036 , Dependent (reject H0) , p < 0.05")
st.text("There is a positive directional relation and a small effect size of 0.30")
st.write("")
st.text("* DP1 (doctors' prediction) and AI recommendation-1 have a significant relation. Cramer's V = 0.23")
st.text("Test Statistic = 8.84, p-value = 0.01 , Dependent (reject H0) , p < 0.05")
st.text("There is a positive directional relation and a small effect size of 0.23")



st.write("")
st.write("")


## DP2 vs MP2 ##############################################################################################3


st.markdown("<div style='text-align: center;background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'>DP2 vs MP2 (Doctors Prediction 2 vs Model Prediction 2)</h1></div>",
            unsafe_allow_html=True)

st.write("")

dp2_mp2_df =pd.DataFrame({"DP2 Count":df["DP2vsGold_St"].value_counts(),
              "DP2 Ratio" : df["DP2vsGold_St"].value_counts()/ df.shape[0]*100,

              "MP2 Count": df["MP2vsGold_St"].value_counts(),
              "MP2 Ratio": df["MP2vsGold_St"].value_counts() / df.shape[0]*100})

aipre2_aiout2_df = pd.DataFrame({"AI Prediction-2 Count": df["AI_Output_pre_2"].value_counts(),
              "AI Prediction-2 Ratio": df["AI_Output_pre_2"].value_counts() / df.shape[0]*100,

              "AI Recommendation-2 Count": df["AI_Output_rec_2"].value_counts(),
              "AI Recommendation-2 Ratio": df["AI_Output_rec_2"].value_counts() / df.shape[0]*100})

dp2_mp2_1,dp2_mp2_2, dp2_mp2_3 = st.columns(3)

with dp2_mp2_1:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp2_mp2_df["DP2 Count"], labels=dp2_mp2_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Doctors Prediction 2', fontsize=7)
    st.pyplot()

with dp2_mp2_2:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp2_mp2_df["MP2 Count"], labels=dp2_mp2_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Model Prediction 2', fontsize=7)
    st.pyplot()


st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>DP2 vs MP2 Table </h1>"
                , unsafe_allow_html=True)
st.dataframe(dp2_mp2_df)
st.write("")
st.text("- The doctors' predictions (DP2) Match rate is 4.02 % ,the model predictions (MP2)  Match rate is 13.22 %")
st.text("- The doctors' predictions (DP2) Mismatch rate is 95.98 % ,the model predictions (MP2) Mismatch rate is 86.78 %")
st.text("- Model predictions results have a higher Match rate, but in general, the match rate is very low")
st.write("")

st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)

st.write("")
st.text("With 95% reliability, the test result for the bellow variable :")
st.text("Test Statistic = 0.42, p-value = 0.51, Independent (H0 holds true) , p > 0.05 ")
st.text("DP2 (doctors' predictions) and MP2 (model predictions) do not have a significant relation")

st.write("")

dp2_mp2_4,dp2_mp2_5, dp2_mp2_6 = st.columns(3)

with dp2_mp2_4:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre2_aiout2_df["AI Prediction-2 Count"], labels=aipre2_aiout2_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of  AI Prediction 2', fontsize=7)
    st.pyplot()

with dp2_mp2_5:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre2_aiout2_df["AI Recommendation-2 Count"], labels=aipre2_aiout2_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of AI Recommendation 2', fontsize=7)
    st.pyplot()


st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>AI Prediction-2 vs "
            "AI Recommendation-2 Table </h1>"
        , unsafe_allow_html=True)
st.dataframe(aipre2_aiout2_df)
st.write("")

st.text("- 83.91% doctors are agree with model predictions (MP2) and 86.21 % agree with model recommendation-2")
st.text("- Although the model prediction match rate is very low, doctors still accept model recommendation")
st.write("")

st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)
st.write("")

st.text("With 95% reliability, the test result for the bellow variables :")
st.write("")
st.text("*AI prediction-2 and AI recommendation-2 do not have a significant relation")
st.text("Test Statistic = 0.96, p-value = 0.32, Independent (H0 holds true) , p > 0.05 ")
st.write("")
st.text("*DP2 (doctors' predictions) and AI prediction-2 do not have a significant relation")
st.text("Test Statistic = 0.43, p-value = 0.51, Independent (H0 holds true) , p > 0.05 ")
st.write("")
st.text("*DP2 (doctors' predictions) and AI recommendation-2 do not have a significant relation")
st.text("Test Statistic = 0.27, p-value = 0.60, Independent (H0 holds true) , p > 0.05 ")

st.write("")
st.write("")



## DP3 vs MP3  ###################################################################

st.markdown("<div style='text-align: center;background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'>DP3 vs MP3 (Doctors Prediction 3 vs Model Prediction 3)</h1></div>",
            unsafe_allow_html=True)

st.write("")
st.write("")

dp3_mp3_df = pd.DataFrame({"DP3 Count" :df["DP3vsGold_St"].value_counts(),
              "DP3 Ratio" : df["DP3vsGold_St"].value_counts()/ df.shape[0]*100,

              "MP3 Count": df["MP3vsGold_St"].value_counts(),
              "MP3 Ratio": df["MP3vsGold_St"].value_counts() / df.shape[0]*100})

aipre3_aiout3_df = pd.DataFrame({"AI Prediction-3 Count": df["AI_Output_pre_3"].value_counts(),
              "AI Prediction-3 Ratio": df["AI_Output_pre_3"].value_counts() / df.shape[0]*100,

              "AI Recommendation-3 Count": df["AI_Output_rec_3"].value_counts(),
              "AI Recommendation-3 Ratio": df["AI_Output_rec_3"].value_counts() / df.shape[0]*100})
aipre3_aiout3_df['AI Recommendation-3 Count'].fillna(0, inplace=True)

dp3_mp3_1,dp3_mp3_2, dp3_mp3_3 = st.columns(3)

with dp3_mp3_1:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp3_mp3_df["DP3 Count"], labels=dp3_mp3_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Doctors Prediction 3', fontsize=7)
    st.pyplot()

with dp3_mp3_2:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(dp3_mp3_df["MP3 Count"], labels=dp3_mp3_df.index, colors=palette_color, autopct='%.2f%%',
            textprops={'fontsize': 7})
    plt.title('Distribution of Model Prediction 3',fontsize=7)
    st.pyplot()

st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>DP3vsMP3 Table </h1>"
                , unsafe_allow_html=True)
st.dataframe(dp3_mp3_df)
st.write("")
st.text("- The doctors's predictions (DP3) Match rate is 8.62 % ,the model predictions (MP3)  Match rate is 6.32 %")
st.text("- The doctors' predictions (DP3) Mismatch rate is 91.38 % ,the model predictions (MP3) Mismatch rate is 93.68 %")
st.text("- Doctors' predictions results have a higher Match rate,but in general the match rate is very low")


st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)
st.write("")
st.text("With 95% reliability, the test result for the bellow variable :")
st.write("")
st.text("*DP3 (doctors' predictions) and MP3 (model predictions) do not have a significant relation")
st.text("Test Statistic = 0.37, p-value = 0.54, Independent (H0 holds true), p > 0.05 ")

st.write("")
st.write("")

dp3_mp3_4,dp3_mp3_5, dp3_mp3_6 = st.columns(3)

with dp3_mp3_4:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre3_aiout3_df["AI Prediction-3 Count"], labels=aipre3_aiout3_df.index, colors=palette_color,
            autopct='%.2f%%', textprops={'fontsize': 7})
    plt.title('Distribution of  AI Prediction 3', fontsize=7)
    st.pyplot()

with dp3_mp3_5:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    palette_color = sns.color_palette('pastel')
    plt.pie(aipre3_aiout3_df["AI Recommendation-3 Count"], labels=aipre3_aiout3_df.index, colors=palette_color,
            autopct='%.2f%%', textprops={'fontsize': 7})
    plt.title('Distribution of AI Recommendation 3', fontsize=7)
    st.pyplot()


st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>AI Prediction-3 vs "
            "AI Recommendation-3 Table </h1>"
        , unsafe_allow_html=True)
st.dataframe(aipre3_aiout3_df)
st.write("")

st.text("- 67.24% doctors are agree with model prediction (MP3) and 100% agree with model recommendation-3")
st.text("- Although the doctors' prediction match rate is higher than model prediction, doctor's accept model")
st.text("recommendation-3  100% rate ")
st.write("")
st.text("-It should be noted that when the doctors' predictions are higher than the model predictions, we observe")
st.text("that the rate of agreement with the AI predictions decreases")

st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Measures of Association : Pearson’s Chi-Square </h1>"
        ,unsafe_allow_html=True)
st.write("")

st.text("With 95% reliability, the test result for the bellow variables :")
st.write("")
st.text("*AI prediction-3 and AI recommendation-3 do not have a significant relation")
st.write("")
st.text("*DP3 (doctors' predictions) and AI prediction-3 do not have a significant relation")
st.text("Test Statistic = 0.83, p-value = 0.36, Independent (H0 holds true) , p > 0.05 ")
st.write("")
st.text("DP3 (doctors' predictions) and AI recommendation-3 do not have a significant relation")

st.write("")
st.write("")



### Doctors prediction time avarage vs Gold standard diagnostic matches  #####################################3


st.markdown("<div style='text-align: center ;background-color: lightgray; padding: 5px; border-radius: 5px;"
            ";'><h1 style='font-size: 20px;color: navy;'>Doctors' Predictions Time Avarage vs Gold Standard Diagnostic Matches</h1></div>",
           unsafe_allow_html=True)
st.write(" ")
st.text("Is there a significant difference between doctors prediction time averages and gold standard diagnostic matches ?")

dpred_dtimecols1, dpred_dtimecols2, dpred_dtimecols3 = st.columns(3)

with dpred_dtimecols1:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    sns.barplot(x="DP1vsGold_St", y="Total_time_doctor", data=df)
    plt.xlabel("DP1 (doctors' predictions)")
    plt.ylabel("Total Time Doctors Mean", )
    plt.title("Doctors Total Time Mean by DP1 Gold Standart Diagnostic")
    st.pyplot()

    st.dataframe(df.groupby("DP1vsGold_St").agg({"Total_time_doctor": "mean"}))


    st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Non-Parametric Test (Kruskal test) </h1>"
        , unsafe_allow_html=True)
    Test_statistic, P_value = kruskal(df[(df["DP1vsGold_St"] == "Match")]["Total_time_doctor"],
                                      df[(df["DP1vsGold_St"] == "Mismatch")]["Total_time_doctor"],
                                      df[(df["DP1vsGold_St"] == "Similarity")]["Total_time_doctor"])
    formatted_test_statistic = f"{Test_statistic:.3f}"
    formatted_p_value = f"{P_value:.3f}"
    st.markdown(f"<p style='font-size: 15px ; font-weight: bold;'>test statistic: {formatted_test_statistic}, p-value: {formatted_p_value}</p>",
        unsafe_allow_html=True)
    if P_value < 0.05:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Reject the (H0) null hypothesis.There is a difference between group averages</h1>"
            , unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Null hypothesis (H0) cannot be rejected. There is no a difference between group averages</h1>"
            , unsafe_allow_html=True)

with dpred_dtimecols2:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    sns.barplot(x="DP2vsGold_St", y="Total_time_doctor", data=df)
    plt.xlabel("DP2 (doctors' predictions)")
    plt.ylabel("Total Time Doctors Mean")
    plt.title("Doctors Total Time Mean by DP2 Gold Standart Diagnostic")
    st.pyplot()

    st.dataframe(df.groupby("DP2vsGold_St").agg({"Total_time_doctor": "mean"}))

    from scipy.stats import kruskal

    st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Non-Parametric Test (Kruskal test) </h1>"
        , unsafe_allow_html=True)
    Test_statistic, P_value = kruskal(df[(df["DP2vsGold_St"] == "Match")]["Total_time_doctor"],
                                      df[(df["DP2vsGold_St"] == "Mismatch")]["Total_time_doctor"],
                                      )
    formatted_test_statistic = f"{Test_statistic:.3f}"
    formatted_p_value = f"{P_value:.3f}"
    st.markdown(
        f"<p style='font-size: 15px ; font-weight: bold;'>test statistic: {formatted_test_statistic}, p-value: {formatted_p_value}</p>",
        unsafe_allow_html=True)
    if P_value < 0.05:
        st.markdown(
            "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Reject the (H0) null hypothesis.There is a difference between group averages</h1>"
            , unsafe_allow_html=True)
    else:
        st.markdown(
            "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Null hypothesis (H0) cannot be rejected. There is no a difference between group averages</h1>"
            , unsafe_allow_html=True)


with dpred_dtimecols3:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.set(style="whitegrid")
    plt.figure(figsize=(3, 3))
    sns.barplot(x="DP3vsGold_St", y="Total_time_doctor", data=df)
    plt.xlabel("DP3 (doctors' predictions)")
    plt.ylabel("Total Time Doctors Mean")
    plt.title("Doctors Total Time Mean by DP3 Standart Diagnostic")
    st.pyplot()

    st.dataframe(df.groupby("DP3vsGold_St").agg({"Total_time_doctor": "mean"}))

    from scipy.stats import kruskal

    st.markdown(
        "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Non-Parametric Test (Kruskal test) </h1>"
        , unsafe_allow_html=True)
    Test_statistic, P_value = kruskal(df[(df["DP3vsGold_St"] == "Match")]["Total_time_doctor"],
                                      df[(df["DP3vsGold_St"] == "Mismatch")]["Total_time_doctor"],
                                      )
    formatted_test_statistic = f"{Test_statistic:.3f}"
    formatted_p_value = f"{P_value:.3f}"
    st.markdown(
        f"<p style='font-size: 15px ; font-weight: bold;'>test statistic: {formatted_test_statistic}, p-value: {formatted_p_value}</p>",
        unsafe_allow_html=True)
    if P_value < 0.05:
        st.markdown(
            "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'> Reject the (H0) null hypothesis.There is a difference between group averages</h1>"
            , unsafe_allow_html=True)
    else:
        st.markdown(
            "<h1 style='text-align: left; color: black; font-size: 15px; font-weight: bold;'>Null hypothesis (H0) cannot be rejected. There is no a difference between group averages</h1>"
            , unsafe_allow_html=True)

st.text("There is no significant difference between doctors' predictions time averages and gold standard diagnostic matches ")
st.write("")

###################################################################################################################

st.markdown("<div style='text-align: center;background-color: lightgray; padding: 5px; border-radius: 5px;"
            "'><h1 style='font-size: 20px;color: navy;'>Doctors (User_name) Predictions Success Comparison </h1></div>",
            unsafe_allow_html=True)

st.write("")

st.text("Assigned score to prediction mappings {'Match': 1, 'Mismatch': 0, 'Similarity': 0.5}")
st.text("Create new variable named New_Match_Score = DP1vsGold_St + DP2vsGold_St + DP3vsGold_St")
st.text("We divide the New_Match_Score by the total number of predictions made by each user to calculate the success score")

df1 = df.copy()

match_mappings = {'Match': 1, 'Mismatch': 0, 'Similarity': 0.5}

match_vars = ["DP1vsGold_St","DP2vsGold_St","DP3vsGold_St", "MP1vsGold_St","MP2vsGold_St","MP3vsGold_St"]

for var in match_vars:
    df1[var] = df1[var].map(match_mappings)

df1["New_Match_Score"] = df1["DP1vsGold_St"] + df1["DP2vsGold_St"] + df1["DP3vsGold_St"]


user_names = ['user291', 'user292', 'user293', 'user294', 'user295', 'user296',
              'user297', 'user298', 'user299', 'user300']

for user_name in user_names:
    success_score = df1[df1["User_name"] == user_name]["New_Match_Score"].sum() / df1[df1["User_name"] == user_name]["User_name"].count()
    st.write(f"Success score for {user_name}: {success_score:.3f}")

st.write("")
st.text("user296 is in first place with 0.808 success score")






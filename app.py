"""Run from the project directory with: streamlit run app.py"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import load_local_model, predict
from preprocessing import FEATURES

ROOT = Path(__file__).resolve().parent
NAMES = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest"}
FIELDS = {"temperature": "Temperature", "wind_speed": "Wind speed", "rain": "Rainfall",
          "relative_humidity": "Relative humidity"}
st.set_page_config(page_title="Wildfire Prediction System", page_icon="🌲", layout="wide")

st.html("""<style>
.block-container{max-width:1240px;padding-top:2.5rem;padding-bottom:3rem}
[data-testid="stSidebar"]{border-right:1px solid #cbd9d0}
[data-testid="stMetric"]{background:#fff;border:1px solid #cfdbd2;border-radius:14px;padding:14px 18px}
[data-testid="stMetricValue"]{color:#174f44;font-weight:650}
[data-testid="stForm"]{background:white;border-radius:16px}
.hero{background:radial-gradient(ellipse at 95% -80%,#639078 0%,transparent 63%),#123f36;
border-radius:18px;padding:clamp(20px,4vw,46px);margin:0 0 20px}
.hero h1{color:white;font-size:clamp(1.8rem,4vw,3.1rem);line-height:1.12;margin:0;padding:0;font-weight:650;letter-spacing:-.04em}
.hero p{color:#daece2;margin:12px 0 0;max-width:68ch}
.brand{color:#154e42;font-size:1.35rem;line-height:1.18;font-weight:700;margin-bottom:1rem}
.result{padding:18px 24px;border-left:6px solid;border-radius:10px;margin:8px 0 16px}
.result strong{display:block;font-size:1.7rem}.fire{background:#fae4d9;color:#913813}
.clear{background:#dfeee7;color:#165442}
</style>""")


@st.cache_data
def inputs():
    metadata = json.loads((ROOT/"model/metadata.json").read_text())
    path = ROOT/"data/wildfire.xlsx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["workbook_sha256"]:
        raise ValueError("Workbook does not match the model metadata.")
    return pd.read_excel(path, sheet_name="Clean data"), metadata


@st.cache_resource
def load_model():
    return load_local_model()


@st.cache_data
def table(name):
    return pd.read_csv(ROOT/"results"/(name+".csv"), index_col=None if name=="september_test" else 0)


def hero(title, summary):
    st.html(f'<section class="hero"><h1>{title}</h1><p>{summary}</p></section>')


def chart(section, filename, note):
    with st.container(border=True):
        st.image(str(ROOT/"assets"/section/(filename+".png")), width="stretch")
        st.caption(note)


def overview(d):
    hero("Wildfire Prediction System", "Explore Algerian forest-fire records and classify a region-day using four weather readings.")
    counts = d.fire.value_counts()
    for col,(title,value) in zip(st.columns(4), [("Total records",len(d)),("Fire observations",counts[1]),
                                                 ("No Fire observations",counts[0]),("Regions",d.region.nunique())]):
        col.metric(title, f"{value:,}")
    left,right=st.columns([3,2])
    with left:
        chart("eda","01_classes","Each observation represents one region on one recorded date.")
    with right:
        st.subheader("About the observations")
        st.write("The dataset covers June through September in 2012, 2017 and 2021. The project compares three classifiers for the recorded same-day Fire label.")
        st.write("Temperature, wind speed, rainfall and relative humidity are available to the prediction form. The six fire-weather indices appear in the exploratory analysis.")
        st.subheader("Coverage and limitations")
        st.write("Regions were sampled in different years. Five region-days are absent, and five inferred date corrections are excluded from temporal evaluation. The source leaves measurement units and label collection unverified.")


def eda():
    hero("Exploratory data analysis","Compare weather readings across the two classes and explore where and when Fire was recorded.")
    tabs=st.tabs(["Weather distributions","Relationships","Regions and months"])
    panels=[
        [("06_temperature","Temperature is typically higher in Fire records. The median rises from 31 to 35 recorded units."),
         ("03_boxplots","Relative humidity shows a 28-point median gap, from 67 in No Fire to 39 in Fire. The box plots retain outliers."),
         ("07_wind_rain","Wind distributions overlap. Rainfall has many zeros and a long upper tail; each curve shows the share at or below a reading.")],
        [("02_correlations","All six indices are included. FWI has the strongest positive correlation with Fire. Correlation alone does not establish a causal effect."),
         ("05_humidity_temperature","Hotter, drier conditions contain a greater share of Fire labels. Overlapping points illustrate why similar readings can yield either label.")],
        [("08_region","Observed Fire shares range from 24.6% in Tipaza to 71.3% in Tiaret. The two regions cover different years."),
         ("09_monthly","Bars use observed region-days in each year and month. Five absent August dates remain missing. Excluding the five inferred September dates gives 73 Fire records out of 205 in September 2021.")]]
    for tab,items in zip(tabs,panels):
        with tab:
            for file,note in items:
                chart("eda",file,note)


def performance(meta):
    scores=table("august_comparison")
    test=table("september_test").iloc[0]
    winner=meta["winner"]
    hero("ML model performance","August validation guided model selection. September provides the final chronological test.")
    st.subheader(f"Selected model · {NAMES[winner]}")
    st.write(f"The {NAMES[winner]} led on August Fire F1 ({scores.loc[winner,'f1']:.3f}). It produced "
             f"{scores.loc[winner,'fp']:.0f} false alarms and missed {scores.loc[winner,'fn']:.0f} Fire records. "
             "Logistic Regression caught more Fire records and achieved the best validation ROC-AUC and Brier score.")
    tabs=st.tabs(["Validation comparison","Final test","Model interpretation"])
    with tabs[0]:
        labels={"accuracy":"Accuracy","precision":"Fire precision","recall":"Fire recall","f1":"Fire F1","auc":"ROC-AUC",
                "brier":"Brier","training_f1":"Training F1","mean_test_score":"CV F1 mean","std_test_score":"CV F1 SD"}
        display=scores[list(labels)].rename(index=NAMES,columns=labels)
        st.dataframe(display.style.format("{:.3f}"),width="stretch")
        chart("evaluation","01_validation_metrics","All three tuned candidates used the same August observations. Higher bars indicate higher scores.")
        chart("evaluation","02_confusion_matrices","Rows are actual classes and columns are predictions. The upper-right cell counts false alarms.")
        chart("evaluation","03_roc","Curves compare Fire recall with the false-positive rate as the decision threshold varies.")
    with tabs[1]:
        for col,(key,label) in zip(st.columns(4),[("accuracy","Accuracy"),("precision","Fire precision"),("recall","Fire recall"),("f1","Fire F1")]):
            col.metric(label,f"{test[key]:.1%}")
        st.write(f"September had {int(test.tp+test.fn)} Fire labels. The model detected {int(test.tp)}, missed {int(test.fn)}, and gave "
                 f"{int(test.fp)} false alarms. ROC-AUC was {test.auc:.3f} and Brier score was {test.brier:.3f}.")
        chart("evaluation","04_reliability","The diagonal marks agreement between predicted and observed Fire fractions. Mid-range September scores overestimated the observed Fire share.")
    with tabs[2]:
        chart("evaluation","05_importance","Shuffling relative humidity reduced September Fire F1 the most. The bars measure how this fitted model used each input on this holdout.")


def prediction(d,meta):
    hero("Wildfire prediction","Enter four weather readings to estimate the recorded same-day class.")
    st.write("Use the dataset’s recorded scale. The form uses observed minimum and maximum values; physical units are unverified.")
    with st.form("weather"):
        cols=st.columns(2)
        values={}
        for i,key in enumerate(FEATURES):
            s=d[key]; cast=int if pd.api.types.is_integer_dtype(s) else float
            low,high=cast(s.min()),cast(s.max())
            values[key]=cols[i%2].number_input(FIELDS[key],min_value=low,max_value=high,value=None,placeholder="Enter reading",
                                step=1 if cast is int else 0.1,help=f"Observed range: {low:g} to {high:g}",key=key)
        submitted=st.form_submit_button("Predict",type="primary")
    if submitted:
        try:
            result = predict(load_model(), **values)
        except (ValueError, RuntimeError) as exc:
            st.session_state.pop("last_result", None)
            st.error(str(exc))
            return
        st.session_state["last_result"] = (values, result)
    if "last_result" in st.session_state:
        readings,result=st.session_state["last_result"]
        fire=result["class"]=="Fire"
        st.html(f'<div class="result {"fire" if fire else "clear"}">Predicted class<strong>{result["class"].upper()}</strong></div>')
        st.caption("Submitted readings · "+" · ".join(f"{FIELDS[k]} {v:g}" for k,v in readings.items()))
        for col,label,key in zip(st.columns(2),["Fire probability","No Fire probability"],["fire_probability","no_fire_probability"]):
            prob=result[key]; col.metric(label,f"{100*prob:.1f}%"); col.progress(float(prob))
        st.write(f"The {NAMES[meta['winner']]} assigned the higher probability to the {result['class']} label for these readings. "
                 "The percentages estimate dataset class membership; their reliability varies across score ranges.")
    st.caption("Academic demonstration only. This model is not an operational wildfire warning service. Its scores are not verified real-world ignition probabilities.")


data,metadata=inputs()
with st.sidebar:
    st.html('<div class="brand">Wildfire<br>Prediction System</div>')
    page=st.radio("Navigation",["Overview Dashboard","Exploratory Data Analysis","ML Model Performance","Wildfire Prediction"])
    st.divider()
    st.caption("Algerian forest-fire observations · 2012, 2017, 2021")
if page=="Overview Dashboard":
    overview(data)
elif page=="Exploratory Data Analysis":
    eda()
elif page=="ML Model Performance":
    performance(metadata)
else:
    prediction(data,metadata)

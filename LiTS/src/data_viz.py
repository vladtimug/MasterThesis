import streamlit as st
import pandas as pd
import pickle as pkl

st.set_page_config(
    page_title="Thesis Data Visualization",
    layout="wide"
)

st.write(
    """
    # Semantic Segmentation Model Metrics
    """
)

with st.expander("Upload experiment files"):
    metricsFile = st.file_uploader("Upload a metrics csv file")
    configFile = st.file_uploader("Upload a configuration file")

if metricsFile is not None:
    experimentMetrics = pd.read_csv(metricsFile)

    d1 = {
        "Train": "train",
        "Validation": "val"
    }

    d2 = {
        "Dice": "dice",
        "IoU": "iou",
        "Precision": "precision",
        "Accuracy": "accuracy",
        "Recall": "recall",
        "Specificity": "specificity",
        "AuROC": "auc_score",
        "Loss": "loss"
    }

    with st.sidebar:
        st.write("""
            # Filters
        """)

        splits = st.multiselect("Dataset Split", list(d1.keys()))

        metrics = st.multiselect("Metric", list(d2.keys()))

        filteredMetrics = {}
        
        for split in splits:
            for metric in metrics:
                col_name = d1[split] + "_" + d2[metric]
                filteredMetrics[col_name] = experimentMetrics[col_name]
                # columnsNames.append(col_name)
                # columnsToPlot.append(metricsFile[col_name])
        
        toPlot = pd.DataFrame.from_dict(filteredMetrics)
    st.line_chart(toPlot)

if configFile is not None:
    experimentConfig = pkl.loads(configFile.read())
    st.write(experimentConfig)
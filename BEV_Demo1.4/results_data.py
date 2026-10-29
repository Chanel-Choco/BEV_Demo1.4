"""
results_data.py -- static numbers copied from the final manuscript (FINALfinal_BEV_Manuscript), which matches the v6 notebook
runs, so the demo's Results tab matches the thesis. Table numbers refer to the manuscript. Nothing here is
computed live.
"""
import pandas as pd


def _df(rows, columns, index):
    return pd.DataFrame(rows, columns=columns).set_index(index)


def main_test_set():  # Table 6
    return _df([
        ["Accuracy", 0.9487, 0.9440, 0.9573],
        ["Precision", 0.9481, 0.9370, 0.9635],
        ["Recall", 0.9493, 0.9520, 0.9507],
        ["F1 score", 0.9487, 0.9444, 0.9570],
        ["ROC AUC", 0.9875, 0.9859, 0.9918],
        ["Misclassified (of 1,500)", 77, 84, 64],
    ], ["Metric", "Model A", "Model B", "Ensemble (A + B)"], "Metric")

def source_errors():  # Table 7
    return _df([
        ["Human (all styles)", 750, "39 (5.2%)", "48 (6.4%)"],
        ["SD 1.5", 150, "16 (10.7%)", "12 (8.0%)"],
        ["SDXL", 150, "16 (10.7%)", "10 (6.7%)"],
        ["Kandinsky 3", 150, "4 (2.7%)", "9 (6.0%)"],
        ["Kolors", 150, "1 (0.7%)", "3 (2.0%)"],
        ["SDXL Turbo", 150, "1 (0.7%)", "2 (1.3%)"],
    ], ["Data source", "Images", "Model A errors", "Model B errors"], "Data source")

def ablation():  # Table 9 (test set part, more metrics than the manuscript table)
    return _df([
        ["Accuracy", 0.9487, 0.9480, 0.9440, 0.9327],
        ["Precision", 0.9481, 0.9541, 0.9370, 0.9209],
        ["Recall", 0.9493, 0.9413, 0.9520, 0.9467],
        ["F1 score", 0.9487, 0.9477, 0.9444, 0.9336],
        ["ROC AUC", 0.9875, 0.9876, 0.9859, 0.9828],
        ["Misclassified (of 1,500)", 77, 78, 84, 101],
    ], ["Metric", "A with LFAB", "A without LFAB", "B with LFAB", "B without LFAB"], "Metric")

def distortions():  # Table 12
    return pd.DataFrame([
        ["A", "Clean", 0.9487, 0.9481, 0.9493, 0.9487, 0.9875],
        ["A", "Noise", 0.8367, 0.8654, 0.7973, 0.8300, 0.9135],
        ["A", "JPEG compression", 0.8120, 0.7623, 0.9067, 0.8283, 0.9147],
        ["A", "Resize", 0.8840, 0.8636, 0.9120, 0.8872, 0.9582],
        ["B", "Clean", 0.9440, 0.9370, 0.9520, 0.9444, 0.9859],
        ["B", "Noise", 0.8120, 0.8031, 0.8267, 0.8147, 0.8974],
        ["B", "JPEG compression", 0.8080, 0.8576, 0.7387, 0.7937, 0.9036],
        ["B", "Resize", 0.8873, 0.8931, 0.8800, 0.8865, 0.9559],
    ], columns=["Model", "Condition", "Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"]
    ).set_index(["Model", "Condition"])

def distortion_accuracy_chart():
    d = distortions().reset_index()
    out = d.pivot(index="Condition", columns="Model", values="Accuracy")
    out = out.loc[["Clean", "Noise", "JPEG compression", "Resize"]]
    out.columns = [f"Model {c}" for c in out.columns]
    return out




def seen_vs_heldout():  # Table 13
    return _df([
        ["Accuracy", 0.9487, 0.6980, 0.9440, 0.6820],
        ["ROC AUC", 0.9875, 0.7265, 0.9859, 0.7051],
        ["AI detection rate (recall)", 0.9493, 0.4480, 0.9520, 0.4280],
        ["Human specificity", 0.9480, 0.9480, 0.9360, 0.9360],
    ], ["Measure", "Model A seen", "Model A held out", "Model B seen", "Model B held out"], "Measure")


def unseen_generators():  # Table 14
    return _df([
        ["A with LFAB", 0.8480, 0.0480, 0.9480],
        ["A without LFAB", 0.8213, 0.0453, 0.9547],
        ["B with LFAB", 0.8000, 0.0560, 0.9360],
        ["B without LFAB", 0.8267, 0.0613, 0.9187],
    ], ["Model", "aMUSEd", "FLUX schnell", "Human specificity"], "Model")


def style_errors():  # Table 8
    return _df([
        ["Art Nouveau", 150, "11.3%", "14.7%", "8.0%", "11.3%", "14.7%", "8.0%"],
        ["Post Impressionism", 300, "8.0%", "6.7%", "9.3%", "8.0%", "8.7%", "7.3%"],
        ["Realism", 300, "5.0%", "2.7%", "7.3%", "5.0%", "3.3%", "6.7%"],
        ["Impressionism", 300, "3.7%", "6.0%", "1.3%", "6.0%", "9.3%", "2.7%"],
        ["Romanticism", 300, "2.7%", "2.7%", "2.7%", "2.3%", "2.7%", "2.0%"],
        ["Baroque", 150, "1.3%", "1.3%", "1.3%", "2.0%", "1.3%", "2.7%"],
    ], ["Style", "Images", "A error", "A human FP", "A AI miss", "B error", "B human FP", "B AI miss"], "Style")

def efficiency():  # Table 15
    return _df([
        ["Parameters", "1,125,714", "4,198,662"],
        ["Model size, 32 bit parameters (MB)", "4.29", "16.02"],
        ["Saved checkpoint (MB)", "4.48", "16.41"],
        ["After INT8 dynamic quantization (MB)", "3.97", "15.95"],
        ["GFLOPs, 256 x 256 input (thop)", "0.162", "1.087"],
        ["Latency, Tesla T4 GPU, batch size 1 (ms)", "5.96", "8.96"],
        ["Peak GPU memory (MB)", "66.6", "100.2"],
    ], ["Measure", "Model A", "Model B"], "Measure")

def benchmark():  # Table 16
    return _df([
        ["Zhang et al. SCADET (published)", 0.8100, 0.9620],
        ["Model A", 0.9487, 0.9875],
        ["Model B", 0.9440, 0.9859],
    ], ["Model", "Accuracy", "ROC AUC"], "Model")

def lfab_learned():  # Table 11 (selected rows)
    return _df([
        ["Final alpha (started at 0.1)", "0.0740", "0.3708"],
        ["Gain map, min to max (1.0 is neutral)", "0.952 to 1.100", "0.879 to 1.274"],
        ["Mean change LFAB makes to features", "7.44%", "5.74%"],
    ], ["Measure", "Model A", "Model B"], "Measure")

# Mean absolute SHAP per band at the final diagnosis (Table 5), in probability units.
BAND_SHAP_REFERENCE = {"Model A": 0.01257, "Model B": 0.00464}

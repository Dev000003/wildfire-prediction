"""Run with paths to the workbook and the Step 4 results directory."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             brier_score_loss, log_loss, confusion_matrix, ConfusionMatrixDisplay, roc_curve)
from sklearn.inspection import permutation_importance
from preprocessing import load_dataset, primary_split, FEATURES, SEED
from predict import predict

ROOT = Path("reproduced/evaluation")
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
NAMES = {"logistic_regression":"Logistic Regression","decision_tree":"Decision Tree","random_forest":"Random Forest"}
COLORS = ["#216a83","#a35434","#387753"]

def measure(model, X, y):
    p = model.predict_proba(X)[:,list(model.classes_).index(1)]
    labels = model.predict(X)
    tn,fp,fn,tp = confusion_matrix(y,labels,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,labels),precision=precision_score(y,labels),
                recall=recall_score(y,labels),f1=f1_score(y,labels),auc=roc_auc_score(y,p),
                brier=brier_score_loss(y,p),log_loss=log_loss(y,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp)),p

def reliability(y,p):
    frame = pd.DataFrame({"p":p,"y":np.asarray(y)})
    return frame.groupby(pd.cut(p,np.linspace(0,1,6),include_lowest=True),observed=True).agg(
        estimated=("p","mean"), observed=("y","mean"), n=("y","size"))

def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIGS / (name+".png"), dpi=170,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def calibrate(base,X,y,mask):
    fitted=clone(base).fit(X.loc[~mask],y.loc[~mask])
    return CalibratedClassifierCV(FrozenEstimator(fitted), method="sigmoid").fit(X.loc[mask],y.loc[mask])

def run(workbook,previous):
    RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(exist_ok=True)
    digest=hashlib.sha256(workbook.read_bytes()).hexdigest()
    assert digest==(previous/"workbook_sha256.txt").read_text().strip()
    d=load_dataset(workbook); role=primary_split(d.meta)
    X=d.X.loc[role.eq("train")]; y=d.y.loc[X.index]
    V=d.X.loc[role.eq("validation")]; v=d.y.loc[V.index]
    trained={k:joblib.load(previous/"models"/(k+"_tuned.joblib")) for k in NAMES}
    metrics={}; probs={}; training={}
    for k,m in trained.items():
        training[k],_=measure(m,X,y)
        metrics[k],probs[k]=measure(m,V,v)
    summary=pd.DataFrame(metrics).T
    cvs=pd.read_csv(previous/"grid_results.csv").query("selected").set_index("model")
    summary=summary.join(cvs[["mean_test_score","std_test_score","split0_test_score","split1_test_score","split2_test_score"]])
    summary["training_f1"]=[training[k]["f1"] for k in summary.index]
    summary.to_csv(RESULTS/"august_comparison.csv")
    pd.DataFrame(training).T.to_csv(RESULTS/"training_comparison.csv")
    leader=summary.sort_values(["f1","recall","precision"],ascending=False).index[0]
    base=trained[leader]
    # Test a deployable sigmoid calibration recipe using July for calibration only.
    trial=calibrate(base,X,y,d.meta.loc[X.index,"date"].ge("2021-07-01"))
    trial_metrics,_=measure(trial,V,v)
    raw=metrics[leader]
    use_cal=bool(trial_metrics["brier"]<raw["brier"] and trial_metrics["f1"]>=raw["f1"]
                 and trial_metrics["recall"]>=raw["recall"])
    pd.DataFrame([raw,trial_metrics],index=["raw","July sigmoid holdout"]).to_csv(RESULTS/"calibration_trial.csv")
    meta=dict(winner=leader,features=list(FEATURES),classes={"0":"No Fire","1":"Fire"},random_state=SEED,
              sklearn_version=sklearn_version,workbook_sha256=digest,calibrated=use_cal,
              selection="August Fire F1 first, then recall and precision for ties; default classifier decision.",
              calibration_rule="Accept sigmoid holdout recipe if August Brier improves and Fire F1 and recall do not fall.",
              parameters=base["classifier"].get_params(),train_rows=len(X),validation_rows=len(V),
              probability_meaning="Estimated probability of a recorded same-day class given the four inputs.",
              refit="Train through July and calibrate on August" if use_cal else "Refit on eligible train + August rows")
    print(summary.to_string(),trial_metrics,use_cal)
    combined=d.X.loc[role.isin(["train","validation"])]
    final=(calibrate(base,combined,d.y.loc[combined.index],role.loc[combined.index].eq("validation"))
           if use_cal else clone(base).fit(combined,d.y.loc[combined.index]))
    joblib.dump(final,ROOT/"best_model.joblib",compress=3)
    final=joblib.load(ROOT/"best_model.joblib")
    meta["classes_"]=list(map(int,final.classes_))
    (ROOT/"metadata.json").write_text(json.dumps(meta,indent=2))
    # Locked choice; September results are used only for reporting.
    T=d.X.loc[role.eq("test")]; t=d.y.loc[T.index]
    test,pt=measure(final,T,t)
    pd.DataFrame([test]).to_csv(RESULTS/"september_test.csv",index=False)
    print("September",test)
    output=predict(final,**T.iloc[0].to_dict())
    assert np.isclose(output["fire_probability"]+output["no_fire_probability"],1)
    for name,frame,target,prob,clf in [("september",T,t,pt,final)]+[
        ("august_"+k,V,v,probs[k],trained[k]) for k in NAMES]:
        pd.DataFrame(dict(actual=target,predicted=clf.predict(frame),p_fire=prob,
                          p_no_fire=clf.predict_proba(frame)[:,list(clf.classes_).index(0)])).to_csv(RESULTS/(name+"_predictions.csv"))
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    ax=summary[["accuracy","precision","recall","f1"]].rename(index=NAMES,columns=str.title).T.plot.bar(
        figsize=(9,5),color=COLORS,rot=0)
    ax.set(title="August validation · four-input classifiers",ylabel="Score",ylim=(0,1.17))
    ax.legend(loc="upper center",ncol=3,frameon=False);save(ax.figure,"01_validation_metrics")
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,(k,m) in zip(axes,trained.items()):
        ConfusionMatrixDisplay.from_estimator(m,V,v,labels=[0,1],display_labels=["No Fire","Fire"],ax=ax,cmap="Blues",colorbar=False)
        ax.set_title(NAMES[k])
    fig.suptitle("August validation · observed and predicted labels");save(fig,"02_confusion_matrices")
    fig,ax=plt.subplots(figsize=(7,5))
    for (k,p),c in zip(probs.items(),COLORS):
        x,z,_=roc_curve(v,p);ax.plot(x,z,label=f"{NAMES[k]}  {metrics[k]['auc']:.3f}",color=c)
    ax.plot([0,1],[0,1],"k--",alpha=.4);ax.legend();ax.set(title="August validation · ROC curves",xlabel="False positive rate",ylabel="True positive rate");save(fig,"03_roc")
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    bins=[]
    for ax,series,title,target in [(axes[0],probs,"August · candidate comparison",v),(axes[1],{leader:pt},"September · final pipeline",t)]:
        for k,p in series.items():
            rel=reliability(target,p);ax.plot(rel.estimated,rel.observed,"o-",label=NAMES[k],color=COLORS[list(NAMES).index(k)])
            bins.append(rel.assign(model=k,period=title))
        ax.plot([0,1],[0,1],"k--",alpha=.4);ax.set(title=title,xlabel="Mean estimated Fire probability",xlim=(-.03,1.03),ylim=(-.03,1.03));ax.legend()
    axes[0].set_ylabel("Observed Fire fraction");save(fig,"04_reliability")
    pd.concat(bins).to_csv(RESULTS/"reliability_bins.csv")
    imp=permutation_importance(final,T,t,scoring="f1",n_repeats=30,random_state=SEED)
    tab=pd.DataFrame({"feature":T.columns,"mean_drop":imp.importances_mean,"sd":imp.importances_std}).sort_values("mean_drop")
    tab.to_csv(RESULTS/"permutation_importance.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,4));ax.barh(tab.feature,tab.mean_drop,xerr=tab.sd,color=COLORS[2],capsize=3)
    ax.axvline(0,color="black",lw=.7);ax.set(title="September · final model feature importance",xlabel="Decrease in Fire F1 after shuffling (mean ± 1 SD, 30 repeats)");save(fig,"05_importance")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("workbook",type=Path);parser.add_argument("step4_results",type=Path)
    args=parser.parse_args();run(args.workbook,args.step4_results)

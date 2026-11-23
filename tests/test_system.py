import importlib
from importlib.metadata import version
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest
from predict import BOUNDS, load_local_model, predict
from preprocessing import FEATURES, form_row

ROOT = Path(__file__).resolve().parents[1]


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_local_model()
        cls.data = pd.read_excel(ROOT/"data/wildfire.xlsx", sheet_name="Clean data")
        cls.examples = [dict(zip(FEATURES, row)) for label in (0, 1)
                        for row in cls.data.loc[cls.data.fire.eq(label),list(FEATURES)].head(2).itertuples(index=False,name=None)]

    def test_artifact_rejections(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"best_model.joblib"
            with patch("predict.MODEL_FILE", file), patch("predict.joblib.load") as deserialize:
                with self.assertRaises(RuntimeError): load_local_model()
                file.write_bytes(b"incorrect artifact")
                with self.assertRaises(RuntimeError): load_local_model()
                deserialize.assert_not_called()

    def test_contract_and_fitted_preprocessing(self):
        self.assertEqual(tuple(self.model.feature_names_in_), FEATURES)
        self.assertEqual(set(self.model.classes_), {0,1})
        for k in FEATURES:
            self.assertEqual(BOUNDS[k], (self.data[k].min(),self.data[k].max()))
        row = form_row(**self.examples[0])
        self.assertEqual(tuple(row.columns), FEATURES)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(t) for t in row.dtypes))
        np.testing.assert_allclose(self.model["preprocess"].transform(row),row.to_numpy())
        self.assertTrue(hasattr(self.model["classifier"],"estimators_"))

    def test_invalid_inputs(self):
        for key in FEATURES:
            for value in [None,"bad",True,np.nan,np.inf,-np.inf,BOUNDS[key][0]-1,BOUNDS[key][1]+1]:
                with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                    predict(self.model,**(self.examples[0] | {key:value}))
        with self.assertRaises(ValueError): predict(self.model,**(self.examples[0] | {"temperature":31.5}))
        with self.assertRaises(TypeError): predict(self.model, temperature=33)

    def test_order_and_probability_mapping(self):
        class Reversed:
            classes_=np.array([1,0])
            def predict_proba(self,row): return np.array([[.8,.2]])
            def predict(self,row): return [1]
        model=Reversed()
        self.assertEqual(predict(model,**self.examples[0]),dict(fire_probability=.8,no_fire_probability=.2,**{"class":"Fire"}))
        with patch.object(model,"predict_proba",return_value=np.array([[.7,.8]])), self.assertRaises(RuntimeError):
            predict(model,**self.examples[0])

    def test_real_predictions(self):
        for values in self.examples + [dict(zip(FEATURES, v)) for v in [(43,50,0,11),(21,5,20.2,96)]]:
            output=predict(self.model,**dict(reversed(values.items())))
            row=form_row(**values)
            self.assertEqual(output["class"],"Fire" if self.model.predict(row)[0] else "No Fire")
            expected=dict(zip(self.model.classes_,self.model.predict_proba(row)[0]))
            self.assertAlmostEqual(output["fire_probability"],expected[1])
            self.assertAlmostEqual(output["no_fire_probability"],expected[0])
            self.assertAlmostEqual(100*(output["fire_probability"]+output["no_fire_probability"]),100)

    def test_assets_and_versions(self):
        files=list((ROOT/"assets").rglob("*.png"))
        self.assertEqual(len(files),13)
        for file in files:
            with Image.open(file) as image: image.load()
        for line in (ROOT/"requirements.txt").read_text().splitlines():
            package,pin=line.split("==")
            importlib.import_module("sklearn" if package=="scikit-learn" else package)
            self.assertEqual(version(package),pin)

    def test_pages_and_submissions(self):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric],["1,215","633","582","9"])
        self.assertEqual(len(app.get("imgs")),1)
        for page,count in [("Exploratory Data Analysis",7),("ML Model Performance",5),("Wildfire Prediction",0)]:
            app.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("imgs")),count)
            if page=="ML Model Performance":
                table=app.dataframe[0].value
                self.assertEqual(set(table.index),{"Logistic Regression","Decision Tree","Random Forest"})
                source=pd.read_csv(ROOT/"results/august_comparison.csv",index_col=0)
                self.assertAlmostEqual(table.loc["Random Forest","Fire F1"],source.loc["random_forest","f1"])
        self.assertEqual({i.key for i in app.number_input},set(FEATURES))
        outcomes=set()
        for values in self.examples:
            for key,value in values.items(): app.number_input(key=key).set_value(value)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            result=app.session_state["last_result"][1]
            self.assertEqual(result,predict(self.model,**values))
            outcomes.add(result["class"])
            self.assertEqual([m.value for m in app.metric],[f"{100*result[k]:.1f}%" for k in ("fire_probability","no_fire_probability")])
            self.assertEqual(len(app.get("progress")),2)
            print(values, result)
        self.assertEqual(outcomes,{"Fire","No Fire"})
        app.number_input(key="rain").set_value(None)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(app.metric)

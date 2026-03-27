from pipeline.risk_canonical import fetch_risk_main_frame 
from pipeline.runner import score_production_frame
from pi
import pandas as pd

y= fetch_risk_main_frame()
z= score_production_frame(y)
print(y.isnull().sum())
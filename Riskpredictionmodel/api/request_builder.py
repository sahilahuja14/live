from __future__ import annotations

import pandas as pd

from ..pipeline.risk_main import build_risk_main_manual_request_frame



def build_manual_request_frame(segment: str, payload) -> pd.DataFrame:
    return build_risk_main_manual_request_frame(segment, payload)

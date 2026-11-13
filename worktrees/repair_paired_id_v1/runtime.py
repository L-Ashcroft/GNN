"""Resolve only the preserved model implementation, inside the EDA workspace."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'versions/rgcn_calibrated_head_v1'));sys.path.insert(0,str(HERE))
import src.common as common
common.ROOT=ROOT

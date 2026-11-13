"""Selected RGCN checkpoint inference. Windows native, fresh Python process."""
from pathlib import Path
import sys
import argparse
ROOT=Path(__file__).resolve().parents[3];PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import torch
import src.common as common
common.ROOT=ROOT
from src.graph.builder import CircuitBatch
from src.gnn.accuracy_inference import load_model,predict_physical

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='artifacts/experiment1_accuracy_v2_cached_head/best.pt')
    parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args();target=(ROOT/args.output).resolve()
    if not target.is_relative_to(ROOT):raise ValueError('Output outside EDA')
    if target.exists():raise FileExistsError('Refusing prediction overwrite')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,saved=load_model(ROOT/args.checkpoint,args.device)
    batch=CircuitBatch(**torch.load(ROOT/args.batch,map_location='cpu',weights_only=True))
    prediction=predict_physical(model,batch)
    prediction={key:value.cpu() if isinstance(value,torch.Tensor) else value for key,value in prediction.items()}
    prediction['checkpoint_recipe_digest']=saved['recipe_digest'];target.parent.mkdir(parents=True,exist_ok=True);torch.save(prediction,target)
    print('Predicted',len(batch.sample_ids),'real samples; physical units; original mask preserved:',target)

if __name__=='__main__':main()

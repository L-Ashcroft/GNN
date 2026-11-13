"""Accuracy candidate: unchanged cached RGCN backbone, replaceable head only."""
from .network import EdgeCentricRGCN
from .contract_head import ContractResidualHead


class AccuracyRGCN(EdgeCentricRGCN):
    def __init__(self,schema,tasks,calibration,contract_residual=True,calibrated_targets=True,
                 adapter_hidden=32,adapter_min_train_samples=256,**kwargs):
        super().__init__(schema,tasks,**kwargs)
        self.head=ContractResidualHead(self.node_encoder.out_features,tasks,calibration,base=self.head,
            contract_residual=contract_residual,calibrated_targets=calibrated_targets,
            adapter_hidden=adapter_hidden,adapter_min_train_samples=adapter_min_train_samples)

    def initialize_from_legacy(self,legacy_state):
        """Explicit, weights-only comparison utility, never recipe/optimizer resume."""
        if self.head.calibrated_targets:raise ValueError('Legacy weights are not standardized-head weights; use a fresh recipe')
        current=self.state_dict();mapped={('head.base.'+k[5:] if k.startswith('head.') else k):v for k,v in legacy_state.items()}
        if not set(mapped)<=set(current):raise ValueError('Legacy model vocabulary differs')
        expected={k for k in current if not k.startswith(('head.adapters.','head.target_'))}
        if set(mapped)!=expected:raise ValueError('Legacy model keys differ')
        current.update(mapped);self.load_state_dict(current,strict=True)

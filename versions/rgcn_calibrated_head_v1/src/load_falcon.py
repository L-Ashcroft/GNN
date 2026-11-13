from .common import ROOT, rel
import yaml
from .netlist import evaluate


def discover():
    repo = ROOT / 'third_party/FALCON'
    metrics = yaml.safe_load((repo / 'config/data_config.yaml').read_text())['Performance']
    for csv in sorted((repo / 'dataset').glob('*/*/dataset.csv')):
        values = csv.parent / 'values.yaml'
        defaults = {}
        if values.exists():
            defaults = {k: evaluate(str(v), {}) for k, v in yaml.safe_load(values.read_text()).items()}
        yield {'source_dataset': 'FALCON', 'source_topology': csv.parent.name, 'circuit_family': csv.parent.parent.name,
               'csv': rel(csv), 'netlist': rel(csv.parent / 'netlist'), 'metric_units': metrics,
               'defaults': defaults, 'default_files': [rel(values)] if values.exists() else [],
               'temperature': None, 'process': 'CMOS_45nm_PDK_unspecified',
               'pdk_evidence': 'https://arxiv.org/html/2505.21923v2#S3.SS1',
               'testbench_status': 'not_published', 'testbench_files': []}


if __name__ == '__main__':
    from scripts.convert_source import main
    main('FALCON')

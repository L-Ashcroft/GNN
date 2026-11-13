import re
import yaml
from .common import ROOT, rel
from .netlist import evaluate


def discover():
    repo = ROOT / 'third_party/AICircuit'
    for csv in sorted((repo / 'Dataset').glob('*/*.csv')):
        name = csv.parent.name
        cfg = yaml.safe_load((repo / f'Config/Circuits/{name}.yaml').read_text())
        corrections = []
        if name == 'Mixer' and 'RF' in cfg['parameter_list']:
            # Official CSV, Spectre netlist and both Ocean scripts agree on WRF.
            cfg['parameter_list'] = ['WRF' if p == 'RF' else p for p in cfg['parameter_list']]
            corrections.append({'from': 'RF', 'to': 'WRF', 'reason': 'Config typo; CSV header, netlist w=WRF and Ocean desVar WRF agree.'})
        ocean = repo / f'Simulation/Ocean/{name}/oceanScript.ocn'
        text = ocean.read_text()
        defaults = {m[1]: evaluate(m[2], {}) for m in re.finditer(r'desVar\(\s*"([^"]+)"\s+([^\s)]+)\s*\)', text)}
        family = {'CSVA': 'VA', 'CVA': 'VA', 'TSVA': 'VA', 'Receiver': 'Receiver', 'Transmitter': 'Transmitter'}.get(name, name)
        units = {'PowerConsumption': 'W', 'Bandwidth': 'Hz', 'OscillationFrequency': 'Hz', 'TuningRange': 'Hz',
                 'OutputPower': 'dBm', 'PhaseNoise': 'dBc/Hz', 'PAE': '%', 'DrainEfficiency': '%',
                 'IFVoltageSwing': 'V', 'VoltageSwing': 'V', 'PSAT': 'source_label_dBm_compression_function'}
        metric_units = {}
        for metric in cfg['performance_list']:
            base = metric.rsplit('_', 1)[0] if '_' in metric else metric
            metric_units[metric] = units.get(base, 'dB')
        yield {'source_dataset': 'AICircuit', 'source_topology': name, 'circuit_family': family,
               'csv': rel(csv), 'netlist': rel(repo / f'Simulation/Netlists/{name}/netlist'),
               'parameters': cfg['parameter_list'], 'metric_units': metric_units, 'defaults': defaults,
               'config_corrections': corrections,
               'temperature': float(re.search(r'temp\(\s*([\d.-]+)', text)[1]),
               'process': 'PTM_LP_45nm_MetalGate_HighK_StrainedSi', 'testbench_status': 'published_reference_not_row_verified',
               'testbench_files': [rel(p) for p in sorted(ocean.parent.glob('*.ocn'))],
               'pdk_evidence': rel(repo / 'Simulation/Model/45n270AModels.sp')}


if __name__ == '__main__':
    from scripts.convert_source import main
    main('AICircuit')

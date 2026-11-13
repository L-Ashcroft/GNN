"""Bounded reference interface to original samples with separate quality metadata."""
from pathlib import Path
import hashlib,json
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/audit/loadbench_physical_v1'

class QualityView:
    def __init__(self):
        self.manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
        self._path=None;self._table=None

    def references(self,cohort='no_trigger',role='train',source=None,allow_test=False):
        if cohort not in {'all_original','no_trigger','review'}:raise ValueError('Unknown cohort')
        if role not in {'train','validation','test'}:raise ValueError('Explicit role required')
        if role=='test' and not allow_test:raise ValueError('Test access requires explicit allow_test=True')
        condition=ds.field('baseline_ID_role')==role
        if source is not None:condition=condition&(ds.field('source_dataset')==source)
        if cohort!='all_original':condition=condition&(ds.field('needs_review')==(cohort=='review'))
        data=ds.dataset(BASE/'quality_index.parquet',format='parquet')
        for batch in data.to_batches(filter=condition,batch_size=2048):
            yield from batch.to_pylist()

    def samples(self,cohort='no_trigger',role='train',source=None,limit=None,allow_test=False):
        for i,ref in enumerate(self.references(cohort,role,source,allow_test)):
            if limit is not None and i>=limit:return
            path=ROOT/ref['source_partition']
            if self._path!=path:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==self.manifest['source_partition_sha256'][ref['source_partition']]
                self._table=pq.read_table(path);self._path=path
            row=self._table.slice(ref['partition_row'],1).to_pylist()[0]
            assert row['sample_id']==ref['sample_id']
            row['quality_review']={'scope':ref['assessment_scope'],'needs_review':ref['needs_review'],'flags':ref['review_flags']}
            yield row

    def close(self):self._path=None;self._table=None

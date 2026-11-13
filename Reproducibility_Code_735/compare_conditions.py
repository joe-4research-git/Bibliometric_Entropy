"""Compare assignments after a shared initial clustering, with optional theme crosswalk."""
import argparse
from itertools import combinations
from pathlib import Path
import json
import pandas as pd

def main(output, crosswalk=None):
    output=Path(output)
    def get(name):
        return pd.read_csv(output/name/'bertopic_document_topics.csv',encoding='utf-8-sig').set_index('record_id',verify_integrity=True)
    raw=get('bertopic_retain')
    sel=get('bertopic_selective').reindex(raw.index)
    assert len(raw)==735 and sel['bertopic_raw_topic'].equals(raw['bertopic_raw_topic']), 'Base clustering changed; resolve before comparing'
    doc=raw[['ut','doi','title','bertopic_raw_topic','bertopic_max_probability']].copy()
    doc['retain']=raw['bertopic_topic'].astype(str)
    doc['exclude']=doc['retain'].replace('-1','EXCLUDED')
    doc['selective']=sel['bertopic_topic'].astype(str)
    assert (doc.loc[doc.bertopic_raw_topic.ne(-1),'retain']==doc.loc[doc.bertopic_raw_topic.ne(-1),'selective']).all()
    threshold=json.loads((output/'bertopic_selective'/'bertopic_run_metadata.json').read_text(encoding='utf-8'))['outlier_threshold']
    assert doc.loc[doc.bertopic_raw_topic.eq(-1),'selective'].ne('-1').equals(doc.loc[doc.bertopic_raw_topic.eq(-1),'bertopic_max_probability'].ge(threshold))
    conditions=['retain','exclude','selective']
    # Topic IDs share the same initial clustering. Map to final themes separately.
    template=[dict(condition=c,topic=t,theme='') for c in conditions for t in sorted(set(doc[c])-{'-1','EXCLUDED'},key=int)]
    pd.DataFrame(template).to_csv(output/'topic_to_theme_template.csv',index=False,encoding='utf-8-sig')
    levels={'topic':conditions}
    if crosswalk:
        allowed=set(json.loads(Path(__file__).with_name('ten_themes.json').read_text(encoding='utf-8')))
        m=pd.read_csv(crosswalk,dtype=str,keep_default_na=False)
        lookup=m.set_index(['condition','topic'],verify_integrity=True)['theme'].str.strip().to_dict()
        expected={(r['condition'],r['topic']) for r in template}
        assert set(lookup)==expected and set(lookup.values())<=allowed, 'Fill every topic with one of the ten canonical labels'
        for c in conditions:
            doc[c+'_theme']=doc[c].map(lambda t:lookup.get((c,t),t))
        levels['theme']=[c+'_theme' for c in conditions]
    rows=[]
    for level,cols in levels.items():
        for subset,df in [('all',doc),('initial_outliers',doc[doc.bertopic_raw_topic.eq(-1)])]:
            for a,b in combinations(cols,2):
                va=~df[a].isin(['-1','EXCLUDED']); vb=~df[b].isin(['-1','EXCLUDED'])
                both=va&vb; same=both&df[a].eq(df[b])
                rows.append(dict(level=level,subset=subset,condition_a=a,condition_b=b,n=len(df),both_assigned=int(both.sum()),same=int(same.sum()),changed=int((both&~same).sum()),agreement_among_both=float(same.sum()/both.sum()) if both.any() else None,only_a=int((va&~vb).sum()),only_b=int((vb&~va).sum()),neither=int((~va&~vb).sum()),a_outlier=int(df[a].eq('-1').sum()),b_outlier=int(df[b].eq('-1').sum()),a_excluded=int(df[a].eq('EXCLUDED').sum()),b_excluded=int(df[b].eq('EXCLUDED').sum())))
    doc.to_csv(output/'document_condition_comparison.csv',encoding='utf-8-sig')
    report=pd.DataFrame(rows)
    report.to_csv(output/'pairwise_agreement.csv',index=False,encoding='utf-8-sig')
    dest=output/'bertopic_exclude'
    dest.mkdir(exist_ok=True)
    raw.loc[raw.bertopic_raw_topic.ne(-1)].to_csv(dest/'bertopic_document_topics.csv',encoding='utf-8-sig')
    raw.loc[raw.bertopic_raw_topic.eq(-1)].to_csv(dest/'excluded_outliers.csv',encoding='utf-8-sig')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--crosswalk')
    a=p.parse_args();print(main(a.output,a.crosswalk).to_string(index=False))

"""Rebase active historical manifests without altering archived provenance."""

import json

from scripts.shared.paths import ROOT, RESULTS

RELOCATIONS = {
    str(ROOT/'analyses/sample_age_correction/outputs'):str(RESULTS/'analyses/sample_age_correction'),
    str(ROOT/'analyses/training_bucket_bias/outputs'):str(RESULTS/'analyses/training_bucket_bias'),
    str(ROOT/'analyses/brainage_gap/outputs'):str(RESULTS/'analyses/brainage_gap'),
    str(ROOT/'analyses/qq_plots/outputs'):str(RESULTS/'analyses/qq_plots'),
    str(ROOT/'Residual plotting'):str(RESULTS/'analyses/residual_history'),
    str(ROOT/'graph_outputs'):str(RESULTS/'analyses/distribution_history'),
}


def rebase(value):
    if isinstance(value,dict):
        return {k:rebase(v) for k,v in value.items()}
    if isinstance(value,list):
        return [rebase(v) for v in value]
    if isinstance(value,str):
        for old,new in RELOCATIONS.items():
            if value==old or value.startswith(old+'/'):
                return new+value[len(old):]
    return value


def main():
    count=0
    for path in (RESULTS/'analyses').rglob('*.json'):
        before=json.loads(path.read_text())
        after=rebase(before)
        if before!=after:
            path.write_text(json.dumps(after,indent=2)+'\n')
            count+=1
    print(f'Rebased {count} active manifests/reports; historical snapshot unchanged')


if __name__=='__main__':
    main()

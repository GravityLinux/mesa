#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Complete object and actual recording comparison across isolated policy arms."""
import argparse, csv, hashlib, json, os, re, subprocess, tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('binary',type=Path);p.add_argument('--out',type=Path)
a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='hk-native-stage-') as tmp:
    root=a.out or Path(tmp);root.mkdir(parents=True,exist_ok=True)
    arms={
        'on': {'HK_APPLE9_NATIVE_STAGES':'gs,tes'},
        'default': {}, 'shipping-default': {},
        'stages-only': {'HK_APPLE9_NATIVE_STAGES':'gs,tes'}, 'env': {'HK_APPLE9_NATIVE_STAGES':'0'},
        'invalid': {'HK_APPLE9_NATIVE_STAGES':'all'},
        'drirc': {'HK_APPLE9_NATIVE_STAGES':'gs,tes'},
        'index-kill': {'HK_APPLE9_NATIVE_STAGES':'gs,tes','HK_APPLE9_NATIVE_INDEX':'0'},
        'legacy-zero': {'HK_APPLE9_NATIVE_STAGES':'gs,tes','HK_APPLE9_NATIVE_MESH':'0'},
        'physical-negative': {'HK_APPLE9_NATIVE_STAGES':'gs,tes'},
        'logical-negative': {'HK_APPLE9_NATIVE_STAGES':'gs,tes'},
        'gs': {'HK_APPLE9_NATIVE_STAGES':'gs'}, 'tes': {'HK_APPLE9_NATIVE_STAGES':'tes'},
    }
    controls=('HK_APPLE9_NATIVE_STAGES','HK_APPLE9_NATIVE_INDEX','HK_APPLE9_NATIVE_MESH',
              'HK_APPLE9_GS_NATIVE','HK_APPLE9_TESS_NATIVE','HK_APPLE9_NATIVE_MESH_PACKED_SHAPES')
    env={k:v for k,v in os.environ.items() if k not in controls}
    results={};flags={}
    for arm, settings in arms.items():
        command=[str(a.binary.resolve()),arm,str(root/arm)]
        if arm!='on': command.append(str(root/'on'))
        arm_env=dict(env)
        if arm not in ('shipping-default','stages-only'):
            arm_env['HK_APPLE9_NATIVE_INDEX']='1'
        arm_env.update(settings)
        proc=subprocess.run(command,env=arm_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (root/(arm+'.log')).write_text(proc.stdout)
        assert proc.returncode==0,(arm,proc.returncode,proc.stdout[-3000:])
        assert 'GPU_execution=0' in proc.stdout
        assert proc.stdout.count('HK_APPLE9_NATIVE_STAGES: unknown selection') == (arm == 'invalid')
        rows=list(csv.DictReader((root/arm/'PROGRAMS.tsv').open(),delimiter='\t'))
        results[arm]={r['case']+'-'+r['object']:r for r in rows}
        match=re.search(r'POLICY stages=(\d+) index=(\d+) flags=([0-9a-f]+)',proc.stdout);assert match
        flags[arm]=dict(stages=int(match[1]),index=int(match[2]),compiler=match[3])
    off=results['default'];on=results['on'];assert off.keys()==on.keys()
    admitted={'gs-v3-r0','gs-v4-r0','tes-v3-r0','tes-v32-r0','tes-v3-r13'}
    native_objects=0
    for name,row in off.items():
        assert row['native']=='0'
        expected=row['case'] in admitted
        assert (on[name]['native']=='1') == expected
        old=(root/'default'/(name+'.bin')).read_bytes()
        new=(root/'on'/(name+'.bin')).read_bytes()
        assert (old!=new)==expected,('admission delta',name)
        native_objects+=expected
        for arm,rows in results.items():
            assert rows.keys()==off.keys()
            assert (root/arm/(name+'.fallback')).read_bytes()==(root/'default'/(name+'.fallback')).read_bytes(),('fallback program/state',arm,name)
            if arm in ('on','logical-negative'):
                assert (root/arm/(name+'.bin')).read_bytes()==new
            elif arm in ('gs','tes') and row['case'].startswith(arm+'-'):
                assert (rows[name]['native']=='1') == expected
                if expected:
                    assert (root/arm/(name+'.native')).read_bytes()==(root/'on'/(name+'.native')).read_bytes(),('native program/state',arm,name)
            else:
                assert (root/arm/(name+'.bin')).read_bytes()==old,('killed or unselected object',arm,name)
    for case in {row['case'] for row in off.values()}:
        packet=(root/'default'/(case+'.packet')).read_bytes()
        for arm in ('env','invalid','drirc','index-kill','legacy-zero','physical-negative','logical-negative','shipping-default','stages-only'):
            assert (root/arm/(case+'.packet')).read_bytes()==packet,('ordinary packet',case,arm)
    assert flags['shipping-default']['index']==flags['stages-only']['index']==0
    assert flags['shipping-default']['stages']==flags['stages-only']['stages']==0
    assert flags['default']['index']==flags['env']['index']==flags['drirc']['index']==1
    assert len({flags[x]['compiler'] for x in ('on','gs','tes','default','env','drirc')})==6
    result=dict(verdict='PASS',cases=len({r['case'] for r in off.values()}),objects=len(off),
                intended_native_objects=native_objects,arms=flags,masked_bytes=0,GPU_execution=False,
                fallback_variant_stream='complete hk_shader_serialize info/program/data per variant',
                native_variant_stream='complete hk_shader_serialize info/program/data per variant; pair IDs retain the full policy key',
                recording='real hk_apple9_record_draw with host allocations; no producer execution',
                artifact_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in root.rglob('*') if f.is_file()})
    (root/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifact_sha256'},indent=2))

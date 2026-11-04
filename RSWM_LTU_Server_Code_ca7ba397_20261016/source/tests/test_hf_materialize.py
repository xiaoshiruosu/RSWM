import importlib.util
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location('hf_materialize', Path(__file__).parents[1]/'scripts'/'hf_materialize.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_exact_pinned_shard_budget():
    assert sum(s[1] for s in m.SHARDS) == 1_316_127_801
    assert all(len(h)==64 for _,_,h in m.SHARDS)
    assert len(m.LEVIR_REVISION)==40 and len(m.INRIA_REVISION)==40


def test_smoke_profile_does_not_read_test_or_metric_splits():
    rows=[]
    for city in ['austin','chicago','vienna','kitsap','bellingham','bloomington','sfo','innsbruck']:
        for rank in range(1,13):rows.append({'city':city,'selection_rank':rank})
    a=m.select_profile({'worlds':rows},'smoke12')
    assert len(a)==12
    assert set(r['city'] for r in a)=={'austin','chicago','vienna','kitsap'}
    assert max(r['selection_rank'] for r in a)==3


def test_row_selection_is_order_independent_and_unique():
    counts={p:116 for p,_,_ in m.SHARDS}
    a=m.choose_levir_rows(counts)
    b=m.choose_levir_rows(dict(reversed(list(counts.items()))))
    assert a==b and len(a)==100 and len({(p,i) for h,p,i in a})==100
    assert all(0<=i<116 for h,p,i in a)


@pytest.mark.parametrize('counts,n',[({'a':2},3),({'a':-1},1),({},1),({'a':3},0),({'a':3},True)])
def test_row_selection_rejects_invalid(counts,n):
    with pytest.raises(ValueError):m.choose_levir_rows(counts,n)


@pytest.mark.parametrize('name,gap,reentry', [('exact',[34,42],[59,63]),('offset',[33,41],[58,62])])
def test_small_map_geometry(name,gap,reentry):
    q,l,s=m.levir_paths()[name]
    assert q.shape==(65,2) and s['min_adjacent_overlap']>=.65
    assert s['invisible_query_gap_range']==gap and s['reentry_event_range']==reentry
    assert s['fully_absent_calls_range']==[1,1]
    for x,y in q:
        assert 0<=512+x-208<512+x+208<=1024
        assert 0<=512+y-120<512+y+120<=1024
    # Independently inspect 25 fixed target pixels, without vectorized label logic.
    for iy in [0,7,64,95,127]:
        for ix in [0,7,64,95,127]:
            u=(int(l['x'][iy,ix]),int(l['y'][iy,ix]))
            events=[t for t,(x,y) in enumerate(q) if x-208<=u[0]<x+208 and y-120<=u[1]<y+120]
            first=min(t for t in events if t<=32)
            last=max(t for t in events if t<=32)
            back=min(t for t in events if t>32)
            assert (l['first_event'][iy,ix],l['last_prefix_visible_event'][iy,ix],l['reentry_event'][iy,ix])==(first,last,back)


def test_small_map_pair_prefix_identical():
    p=m.levir_paths()
    assert np.array_equal(p['exact'][0][:49],p['offset'][0][:49])
    assert np.array_equal(p['exact'][1]['first_event'],p['offset'][1]['first_event'])


def test_core_reference_pixel_worked_example():
    q=m.interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(480,0)],[16,16,32,16,16])
    l,s=m.geometry_labels(q,(480,0))
    assert s['first_event_range']==[7,12]
    assert s['reentry_event_range']==[92,95] and s['invisible_query_gap_range']==[72,76]
    assert [int(l[k][64,64]) for k in ['first_event','last_prefix_visible_event','reentry_event','invisible_query_gap']]==[10,20,93,72]
    assert list(q[10])==[300,0] and list(q[93])==[300,-90]


def test_image1_bytes_decode_without_resize():
    a=np.zeros((1024,1024,3),np.uint8);a[:,:,1]=203
    b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
    out,h,path=m.decode_image1({'bytes':b.getvalue(),'path':'A/train_1.png'})
    assert np.array_equal(out,a) and len(h)==64 and path=='A/train_1.png'


@pytest.mark.parametrize('mode,size',[('L',(1024,1024)),('RGB',(512,512)),('RGBA',(1024,1024))])
def test_invalid_image1_never_converted(mode,size):
    b=io.BytesIO();Image.new(mode,size).save(b,format='PNG')
    with pytest.raises(ValueError):m.decode_image1({'bytes':b.getvalue(),'path':None})


def test_external_image_paths_not_fetched():
    with pytest.raises(ValueError):m.decode_image1({'path':'https://example.invalid/image.png','bytes':None})


def test_fresh_output_and_examples(tmp_path):
    out=tmp_path/'examples'
    m.export_examples(out)
    assert len(list(out.glob('*.npz')))==6
    assert len(list(out.glob('*.json')))==7
    with pytest.raises(FileExistsError):m.export_examples(out)


@pytest.mark.parametrize('q',[np.array([[0,0]]*65,float),np.full((65,2),1000),np.zeros((3,2),int)])
def test_geometry_rejects_invalid_program(q):
    with pytest.raises(ValueError):m.geometry_labels(q,(480,0))

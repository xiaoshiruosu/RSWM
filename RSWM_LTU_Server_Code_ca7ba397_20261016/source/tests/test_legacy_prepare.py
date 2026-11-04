"""Source deployment and original loader fail-closed checks, no real weights."""
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from rswm_inria.geometry import file_hash
from rswm_inria.legacy_wan import strict_from_pretrained, strict_lora_from_pretrained

FIXTURE=Path(__file__).parent/"fixtures"/"legacy_sampler.py"
spec=importlib.util.spec_from_file_location("prepare_legacy_wan_test",Path(__file__).parents[1]/"scripts"/"prepare_legacy_wan.py")
prepare=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare)


def source_files():
    fixture=FIXTURE.read_text()
    inference="from wan.modules.model import WanModel\n" + prepare._function(fixture,"load_lora_model")[1] + "\n" + prepare._function(fixture,"generate_video")[1]
    return {"LICENSE":b"software fixture licensing only\n", "NOTICE":b"software fixture notice\n",
            prepare.SELECTED+"inference.py":inference.encode(),
            prepare.SELECTED+"evaluate_baseline.py":prepare._function(fixture,"load_base_model")[1].encode(),
            prepare.SELECTED+"backbone.py":b"# software fixture; not imported by bridge\n",
            prepare.SELECTED+"wan/modules/model.py":b"# software fixture\n",
            prepare.SELECTED+"wan/modules/tokenizers.py":b"class Tokenizer:\n    def __init__(self,name,**kwargs):\n        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)\n"}


def fixture_zip(tmp_path, extra=()):
    path=tmp_path/"source.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr(prepare.SOURCE_ROOT+"/",b"")
        for name,data in source_files().items(): archive.writestr(prepare.SOURCE_ROOT+"/"+name,data)
        for name,data in extra: archive.writestr(name,data)
    return path


def test_zip_root_directory_and_selected_mapping_readback(tmp_path):
    archive=fixture_zip(tmp_path)
    out=tmp_path/"prepared"
    result=prepare.prepare(source_zip=archive,expected_source_sha256=file_hash(archive),out=out)
    assert result["status"]=="PREPARED_SOURCE_ONLY" and result["real_model_validation"]=="NOT_RUN"
    inventory=json.loads((out/"source_inventory.json").read_text())
    assert all(file_hash(out/name)==sha for name,sha in inventory.items())
    assert "backend.py" not in inventory and "source_inventory.json" not in inventory
    assert not (out/"evaluate_baseline.py").exists() and not (out/"evaluate.py").exists()
    assert (out/"LICENSE").read_bytes()==source_files()["LICENSE"]
    provenance=json.loads((out/"source-provenance.json").read_text())
    assert not provenance["weights_copied"] and not provenance["license_approval_inferred"]
    assert provenance["selected_functions"]["generate_video"]["original_text_sha256"]=="74c905c7ec6ddc76aceb6386d4beb79d47d5fe6189d119799d4aa00d1a553453"
    assert provenance["base_function_original_text_sha256"]=="130edb8f8982627be151b41942c0afc1cb2cc425a159372a289b3c3158fbca13"
    ast.parse((out/"inference.py").read_text()); ast.parse((out/"base_loader.py").read_text())
    for name in ("inference.py","base_loader.py","wan/modules/tokenizers.py"):
        assert (out/name).read_text().startswith("# RSWM integration modifications:")
    assert (out/"inference.py").read_text().count("= strict_lora_from_pretrained(PeftModel,")==2
    with pytest.raises(FileExistsError): prepare.prepare(source_zip=archive,expected_source_sha256=file_hash(archive),out=out)


def test_unrelated_credential_benchmark_is_excluded_without_mutating_inputs(tmp_path):
    # Explicit synthetic secret-shaped input; this is never a real provider key.
    secret=hashlib.sha256(b"RSWM synthetic credential exclusion fixture").hexdigest()
    configuration=json.dumps({"provider":{"api_key":secret}}).encode()
    excluded=prepare.SOURCE_ROOT+"/"+prepare.EXCLUDED_SOURCE_SUBTREE
    archive=fixture_zip(tmp_path,[(excluded+"config.json",configuration),
                                 (excluded+"tools/train_lora.py",b"# unrelated old benchmark fixture\n")])
    before=file_hash(archive); out=tmp_path/"safe"
    prepare.prepare(source_zip=archive,expected_source_sha256=before,out=out)
    assert file_hash(archive)==before
    with zipfile.ZipFile(archive) as original: assert original.read(excluded+"config.json")==configuration
    inventory=json.loads((out/"source_inventory.json").read_text())
    provenance=json.loads((out/"source-provenance.json").read_text())
    assert not (out/"wan/remotesensebenchmark").exists()
    assert not any(name.startswith("wan/remotesensebenchmark/") for name in inventory)
    assert not any(name.startswith("wan/remotesensebenchmark/") for name in provenance["files"])
    assert provenance["excluded_source_subtrees"][0]["path"]==excluded
    assert provenance["excluded_source_subtrees"][0]["file_count"]==2
    assert all(secret.encode() not in path.read_bytes() for path in out.rglob("*") if path.is_file())
    # Excluding unrelated tooling leaves the real selected original sampler,
    # strict expert calls, and required vendor modules exactly as prepared.
    original=source_files()[prepare.SELECTED+"inference.py"].decode()
    assert (out/"inference.py").read_text()==prepare.patch_inference(original)
    assert (out/"wan/modules/model.py").read_bytes()==source_files()[prepare.SELECTED+"wan/modules/model.py"]
    assert "scheduler.step(" in prepare._function((out/"inference.py").read_text(),"generate_video")[1]


@pytest.mark.parametrize("entry", ["other/file.py",prepare.SOURCE_ROOT+"/../evil.py",prepare.SOURCE_ROOT+"//evil.py",prepare.SOURCE_ROOT+"/./evil.py",prepare.SOURCE_ROOT+"\\evil.py"])
def test_unsafe_zip_and_noncanonical_paths_fail_before_write(tmp_path,entry):
    archive=fixture_zip(tmp_path,[(entry,b"invalid")]); out=tmp_path/"absent"
    with pytest.raises(ValueError): prepare.prepare(source_zip=archive,expected_source_sha256=file_hash(archive),out=out)
    assert not out.exists()


def test_duplicate_zip_inventory_rejected(tmp_path):
    with pytest.warns(UserWarning,match="Duplicate name"):
        archive=fixture_zip(tmp_path,[(prepare.SOURCE_ROOT+"/LICENSE",b"duplicate")])
    with pytest.raises(ValueError,match="duplicate"): prepare.prepare(source_zip=archive,expected_source_sha256=file_hash(archive),out=tmp_path/"absent")


def test_zip_hash_symlink_and_directory_tamper_rejected(tmp_path):
    archive=fixture_zip(tmp_path)
    with pytest.raises(ValueError,match="SHA256"): prepare.prepare(source_zip=archive,expected_source_sha256="0"*64,out=tmp_path/"absent")
    link=tmp_path/"link.zip"; link.symlink_to(archive)
    with pytest.raises(ValueError,match="symlinks"): prepare.prepare(source_zip=link,expected_source_sha256=file_hash(archive),out=tmp_path/"absent")
    root=tmp_path/"directory"; root.mkdir()
    index=[]
    for name,data in source_files().items():
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        index.append({"path":prepare.SOURCE_ROOT+"/"+name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    inventory=tmp_path/"inventory.json"; inventory.write_text(json.dumps(index))
    (root/"LICENSE").write_text("tampered")
    with pytest.raises(ValueError,match="SHA256/size"): prepare.prepare(source_dir=root,source_inventory=inventory,out=tmp_path/"absent")


def test_local_tokenizer_has_no_network_fallback():
    source=source_files()[prepare.SELECTED+"wan/modules/tokenizers.py"].decode()
    calls=[]
    globals_={"AutoTokenizer":SimpleNamespace(from_pretrained=lambda name,**kwargs:calls.append((name,kwargs)))}
    exec(prepare.patch_tokenizer(source),globals_)
    globals_["Tokenizer"]("local-tokenizer")
    assert calls==[("local-tokenizer",{"local_files_only":True})]
    with pytest.raises(ValueError,match="network fallback"): globals_["Tokenizer"]("missing",local_files_only=False)
    assert len(calls)==1


def loader_doubles(monkeypatch,tmp_path,info_key=None,failed_expert="low"):
    calls={"loads":[],"peft":[],"merges":[],"frozen":[]}
    class Model:
        model_type="i2v"; patch_size=(1,2,2); in_dim=36; dim=5120
        ffn_dim=13824; freq_dim=256; num_heads=40; num_layers=40; text_len=512; out_dim=16
        patch_embedding_wancamctrl=SimpleNamespace(in_features=1536,out_features=5120)
        blocks=[SimpleNamespace(**{name:(lambda:None) for name in ("cam_injector_layer1","cam_injector_layer2","cam_scale_layer","cam_shift_layer")}) for _ in range(40)]
        def __init__(self,tag): self.tag=tag
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            calls["loads"].append((args,kwargs)); info={name:[] for name in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")}
            if info_key and kwargs["subfolder"]==failed_expert: info[info_key]=["software-missing-key"]
            return cls(kwargs["subfolder"]),info
        def eval(self): return self
        def requires_grad_(self,value): calls["frozen"].append((self.tag,value)); return self
        def cpu(self): return self
        def to(self,device): return self
    class Wrapper(Model):
        def merge_and_unload(self): calls["merges"].append(self.tag); return Model(self.tag)
    class Peft:
        @staticmethod
        def from_pretrained(model,path,**kwargs):
            calls["peft"].append((model.tag,path,kwargs))
            result=Wrapper(model.tag)
            if "actual_lora" in calls: result.adapter_state=calls["actual_lora"][model.tag]
            return result
    class Text:
        def __init__(self,**kwargs): self.model=Model("text")
    class VAE:
        def __init__(self,**kwargs): pass
    for module_name, attribute, value in [("wan.modules.model","WanModel",Model),("wan.modules.t5","T5EncoderModel",Text),("wan.modules.vae2_1","Wan2_1_VAE",VAE)]:
        module=ModuleType(module_name); setattr(module,attribute,value); monkeypatch.setitem(__import__("sys").modules,module_name,module)
    lora=tmp_path/"lora"; (lora/"low_noise").mkdir(parents=True); (lora/"high_noise").mkdir()
    c=SimpleNamespace(text_len=4,t5_dtype="bf16",t5_checkpoint="text.pth",t5_tokenizer="tokenizer",vae_checkpoint="vae.pth",low_noise_checkpoint="low",high_noise_checkpoint="high")
    globals_={"torch":SimpleNamespace(device=str,bfloat16="bf16"),"os":os,"Path":Path,"logger":logging.getLogger(__name__),
              "strict_from_pretrained":strict_from_pretrained,"strict_lora_from_pretrained":strict_lora_from_pretrained,
              "WanModel":Model,"Wan2_1_VAE":VAE,"T5EncoderModel":Text,"PeftModel":Peft}
    return calls,globals_,lora,c,Wrapper


def compile_function(source,name,namespace):
    fn=prepare._function(source,name)[0]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual-selected-loader","exec"),namespace)
    return namespace[name]


def local_lora_tensor_doubles(monkeypatch,calls,lora,corruption=None,expert="low"):
    torch=pytest.importorskip("torch")
    retained={tag:{"camera.lora_A.weight":torch.tensor([[.25,.5]],dtype=torch.float32),
                   "camera.lora_B.weight":torch.tensor([[.75],[1.]],dtype=torch.float32)} for tag in ("low","high")}
    actual={tag:{key:value.clone() for key,value in state.items()} for tag,state in retained.items()}
    target=actual[expert]
    key="camera.lora_A.weight"
    if corruption=="missing": del target[key]
    elif corruption=="extra": target["unplanned.lora_A.weight"]=torch.ones(1)
    elif corruption=="shape": target[key]=torch.ones(1)
    elif corruption=="nan": target[key][0,0]=float("nan")
    elif corruption=="value": target[key][0,0]=.125
    calls["actual_lora"]=actual
    helper=ModuleType("peft.utils.save_and_load")
    def weights(path,**kwargs):
        calls.setdefault("retained_lora",[]).append((path,kwargs))
        return retained[Path(path).name.removesuffix("_noise")]
    def state(model,**kwargs):
        assert kwargs=={"adapter_name":"default","save_embedding_layers":False}
        return model.adapter_state
    helper.load_peft_weights=weights; helper.get_peft_model_state_dict=state
    monkeypatch.setitem(__import__("sys").modules,"peft.utils.save_and_load",helper)
    for path in lora.iterdir():
        (path/"adapter_config.json").write_text("{}")
        (path/"adapter_model.safetensors").write_bytes(b"explicit software tensor-loader fixture")


@pytest.mark.parametrize("merge", [True,False,None])
def test_actual_restored_loader_preserves_explicit_merge_policy(monkeypatch,tmp_path,merge):
    calls,namespace,lora,c,Wrapper=loader_doubles(monkeypatch,tmp_path)
    local_lora_tensor_doubles(monkeypatch,calls,lora)
    inference=prepare.patch_inference(source_files()[prepare.SELECTED+"inference.py"].decode())
    load=compile_function(inference,"load_lora_model",namespace)
    loaded=load(str(tmp_path),str(lora),"cpu",c,**({} if merge is None else {"merge_lora":merge}))
    merge = True if merge is None else merge
    assert [x[0] for x in calls["peft"]]==["low","high"]
    assert all(x[2]=={"is_trainable":not merge,"local_files_only":True} for x in calls["peft"])
    assert all(x[1]=={"device":"cpu","local_files_only":True} for x in calls["retained_lora"])
    assert calls["merges"]==(["low","high"] if merge else [])
    assert all(isinstance(x,Wrapper) is (not merge) for x in loaded[2:])
    assert all(x[1]["local_files_only"] is True and x[1]["output_loading_info"] is True for x in calls["loads"])


@pytest.mark.parametrize("merge", [True,False])
@pytest.mark.parametrize("expert", ["low","high"])
@pytest.mark.parametrize("corruption", ["missing","extra","shape","nan","value"])
def test_actual_restored_loader_checks_exact_retained_lora_before_expert_merge(monkeypatch,tmp_path,merge,expert,corruption):
    calls,namespace,lora,c,_=loader_doubles(monkeypatch,tmp_path)
    local_lora_tensor_doubles(monkeypatch,calls,lora,corruption,expert)
    load=compile_function(prepare.patch_inference(source_files()[prepare.SELECTED+"inference.py"].decode()),"load_lora_model",namespace)
    with pytest.raises(ValueError,match="LoRA"): load(str(tmp_path),str(lora),"cpu",c,merge_lora=merge)
    assert expert not in calls["merges"]
    assert all(x[2]["local_files_only"] is True for x in calls["peft"])


@pytest.mark.parametrize("loader", ["base","restored_lora"])
@pytest.mark.parametrize("info_key", ["missing_keys","unexpected_keys","mismatched_keys","error_msgs"])
@pytest.mark.parametrize("failed_expert", ["low","high"])
def test_actual_loaders_fail_closed_on_incomplete_experts(monkeypatch,tmp_path,loader,info_key,failed_expert):
    calls,namespace,lora,c,_=loader_doubles(monkeypatch,tmp_path,info_key,failed_expert)
    if loader=="base":
        source=prepare.base_loader(FIXTURE.read_text()); name="load_base_model"; args=(str(tmp_path),"cpu",c)
    else:
        source=prepare.patch_inference(source_files()[prepare.SELECTED+"inference.py"].decode()); name="load_lora_model"; args=(str(tmp_path),str(lora),"cpu",c)
    load=compile_function(source,name,namespace)
    with pytest.raises(ValueError,match="checkpoint loading incomplete"): load(*args)
    assert not calls["peft"] and not calls["merges"]


def test_actual_isolated_base_loader_preserves_both_experts(monkeypatch,tmp_path):
    calls,namespace,_,c,_=loader_doubles(monkeypatch,tmp_path)
    load=compile_function(prepare.base_loader(FIXTURE.read_text()),"load_base_model",namespace)
    loaded=load(str(tmp_path),"cpu",c)
    assert [model.tag for model in loaded[2:]]==["low","high"]
    assert [item[1]["subfolder"] for item in calls["loads"]]==["low","high"]
    assert not calls["peft"]


@pytest.mark.parametrize("missing_expert", ["low_noise","high_noise"])
def test_actual_restored_loader_never_falls_back_to_base(monkeypatch,tmp_path,missing_expert):
    calls,namespace,lora,c,_=loader_doubles(monkeypatch,tmp_path)
    (lora/missing_expert).rmdir()
    load=compile_function(prepare.patch_inference(source_files()[prepare.SELECTED+"inference.py"].decode()),"load_lora_model",namespace)
    with pytest.raises(FileNotFoundError,match="LoRA"): load(str(tmp_path),str(lora),"cpu",c)
    assert not calls["peft"] and not calls["merges"]

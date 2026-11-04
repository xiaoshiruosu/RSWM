"""Local frozen feature inference and integrity-checked content-addressed caches.

A supplied TorchScript archive embeds the serialized graph and weights. No model
is downloaded or initialized as a fallback. Feature validity is not semantic or
real-model validation; test providers must declare diagnostic_features=True.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
from pathlib import Path
import re
import uuid
import copy
import numpy as np
from PIL import Image
from .geometry import canonical,digest,file_hash
from .replay import load_json

FEATURE_SCHEMA="rswm-frozen-feature-config-v1"


def validate_backend_manifest(manifest):
    """The diagnostic designation is part of the frozen backend identity."""
    if not isinstance(manifest,dict) or type(manifest.get("diagnostic_features")) is not bool:
        raise ValueError("FEATURE_BACKEND_DIAGNOSTIC_FLAG_REQUIRED")
    body={k:v for k,v in manifest.items() if k!="feature_identity_sha256"}
    if digest(body)!=manifest.get("feature_identity_sha256"):
        raise ValueError("FEATURE_BACKEND_IDENTITY_MISMATCH")
    if manifest.get("release_validated") is not False:
        raise ValueError("FEATURE_RELEASE_VALIDATION_NOT_ESTABLISHED")
    if not manifest["diagnostic_features"]:
        config=manifest.get("config",{})
        if (manifest.get("schema")!="rswm-feature-backend-manifest-v1" or
            config.get("schema")!=FEATURE_SCHEMA or config.get("backend")!="torchscript-local-v1" or
            manifest.get("config_sha256")!=digest(config) or
            manifest.get("weights_sha256")!=config.get("weights",{}).get("sha256")):
            raise ValueError("FEATURE_REAL_BACKEND_MANIFEST_REQUIRED")
    return manifest


class FeatureUnavailable(RuntimeError):
    def __init__(self,reason_code):super().__init__(reason_code);self.reason_code=reason_code


def validate_feature_config(config,base_dir=None):
    if not isinstance(config,dict) or config.get("schema")!=FEATURE_SCHEMA or config.get("backend")!="torchscript-local-v1":
        raise ValueError("FEATURE_CONFIG_SCHEMA_OR_BACKEND")
    allowed={"schema","backend","model_repo","model_revision","output_layer","feature_dimension","weights",
             "dtype","color_input","input_shape","output_reduction","output_selector","device","batch_size",
             "preprocessing","runtime_versions","diagnostic_features"}
    if set(config)-allowed:raise ValueError("FEATURE_CONFIG_UNKNOWN_FIELDS")
    if type(config.get("diagnostic_features",False)) is not bool:raise ValueError("FEATURE_DIAGNOSTIC_FLAG_INVALID")
    for key in ("model_repo","model_revision","output_layer"):
        if not isinstance(config.get(key),str) or not config[key]:raise ValueError("FEATURE_CONFIG_REQUIRED_"+key)
    if config["output_layer"]!="serialized_module_output":raise ValueError("FEATURE_OUTPUT_LAYER_NOT_SUPPORTED")
    if type(config.get("feature_dimension")) is not int or config["feature_dimension"]<1:raise ValueError("FEATURE_DIMENSION_INVALID")
    weights=config.get("weights")
    if not isinstance(weights,dict) or not isinstance(weights.get("path"),str) or not re.fullmatch(r"[0-9a-f]{64}",weights.get("sha256","")):
        raise ValueError("FEATURE_WEIGHT_MANIFEST_REQUIRED")
    if config.get("dtype")!="float32" or config.get("color_input")!="RGB" or config.get("input_shape")!=[128,128,3]:
        raise ValueError("FEATURE_INPUT_DTYPE_OR_COLOR")
    if config.get("output_reduction","none") not in {"none","global_mean_pool"}:raise ValueError("FEATURE_OUTPUT_REDUCTION_INVALID")
    if config.get("output_selector") is not None and type(config["output_selector"]) not in {str,int}:raise ValueError("FEATURE_OUTPUT_SELECTOR_INVALID")
    if not isinstance(config.get("device"),str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?",config["device"]):raise ValueError("FEATURE_DEVICE_INVALID")
    if type(config.get("batch_size")) is not int or config["batch_size"]<1:raise ValueError("FEATURE_BATCH_SIZE_INVALID")
    prep=config.get("preprocessing",{})
    if not isinstance(prep,dict) or set(prep)!={"resize","center_crop","resample","scale","mean","std"}:
        raise ValueError("FEATURE_PREPROCESSING_FIELDS_REQUIRED")
    size=prep.get("resize")
    if not isinstance(size,list) or len(size)!=2 or any(type(v) is not int or v<1 for v in size):raise ValueError("FEATURE_RESIZE_INVALID")
    crop=prep.get("center_crop")
    if crop is not None and (not isinstance(crop,list) or len(crop)!=2 or any(type(v) is not int or v<1 for v in crop) or any(v>s for v,s in zip(crop,size))):
        raise ValueError("FEATURE_CROP_INVALID")
    if prep.get("resample") not in {"bilinear","bicubic"} or prep.get("scale")!="uint8/255":raise ValueError("FEATURE_RESAMPLE_OR_SCALE_INVALID")
    for key in ("mean","std"):
        a=np.asarray(prep.get(key),dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all() or (key=="std" and (a<=0).any()):raise ValueError("FEATURE_NORMALIZATION_INVALID")
    path=Path(weights["path"])
    if not path.is_absolute():path=Path(base_dir or ".")/path
    if not path.exists():raise FeatureUnavailable("FEATURE_WEIGHTS_NOT_AVAILABLE")
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():raise ValueError("FEATURE_WEIGHT_PATH_INVALID")
    if file_hash(path)!=weights["sha256"]:raise ValueError("FEATURE_WEIGHTS_HASH_MISMATCH")
    return path.resolve()


class FrozenTorchScriptFeatures:
    diagnostic_features=False
    def __init__(self,config,base_dir=None):
        config=copy.deepcopy(config)
        path=validate_feature_config(config,base_dir)
        try:import torch
        except ImportError as exc:raise FeatureUnavailable("FEATURE_TORCH_NOT_INSTALLED") from exc
        self.torch=torch;self.config=config
        if config["device"].startswith("cuda") and not torch.cuda.is_available():raise FeatureUnavailable("FEATURE_CUDA_NOT_AVAILABLE")
        # The archive itself contains all parameters; no randomly initialized
        # architecture or implicit checkpoint fallback is ever constructed.
        self.model=torch.jit.load(str(path),map_location=config["device"]).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.feature_dimension=config["feature_dimension"]
        versions={"torch":torch.__version__,"numpy":np.__version__,"Pillow":importlib.metadata.version("Pillow")}
        for name,expected in config.get("runtime_versions",{}).items():
            if versions.get(name)!=expected:raise ValueError("FEATURE_RUNTIME_VERSION_MISMATCH_"+name)
        self.diagnostic_features=config.get("diagnostic_features",False)
        if type(self.diagnostic_features) is not bool:raise ValueError("FEATURE_DIAGNOSTIC_FLAG_INVALID")
        self.manifest={"schema":"rswm-feature-backend-manifest-v1","config":config,"config_sha256":digest(config),
                       "weights_sha256":config["weights"]["sha256"],"runtime_versions":versions,
                       "diagnostic_features":self.diagnostic_features,"release_validated":False,"real_model_validation":"NOT_RUN"}
        self.manifest["feature_identity_sha256"]=digest(self.manifest)
        validate_backend_manifest(self.manifest)

    def encode(self,images):
        torch=self.torch;prep=self.config["preprocessing"];processed=[]
        for image in images:
            if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(128,128,3):raise ValueError("FEATURE_PATCH_SHAPE_OR_DTYPE")
            im=Image.fromarray(image).resize(tuple(reversed(prep["resize"])),getattr(Image.Resampling,prep["resample"].upper()))
            if prep["center_crop"] is not None:
                h,w=prep["center_crop"];x=(im.width-w)//2;y=(im.height-h)//2;im=im.crop((x,y,x+w,y+h))
            a=np.asarray(im,dtype=np.float32)/np.float32(255.)
            a=(a-np.asarray(prep["mean"],np.float32))/np.asarray(prep["std"],np.float32)
            processed.append(np.ascontiguousarray(a.transpose(2,0,1)))
        results=[]
        for lo in range(0,len(processed),self.config["batch_size"]):
            inputs=torch.from_numpy(np.stack(processed[lo:lo+self.config["batch_size"]])).to(self.config["device"])
            with torch.inference_mode():
                output=self.model(inputs)
                selector=self.config.get("output_selector")
                if selector is not None:output=output[selector]
                if self.config.get("output_reduction","none")=="global_mean_pool":
                    if not hasattr(output,"ndim") or output.ndim<3:raise ValueError("FEATURE_POOL_SHAPE_INVALID")
                    output=output.mean(dim=tuple(range(2,output.ndim)))
                if not isinstance(output,torch.Tensor) or output.dtype!=torch.float32:raise ValueError("FEATURE_OUTPUT_DTYPE_INVALID")
                a=output.detach().cpu().numpy().copy()
            if a.shape!=(len(inputs),self.feature_dimension) or not np.isfinite(a).all():raise ValueError("FEATURE_OUTPUT_SHAPE_OR_FINITE")
            results.append(a)
        return np.concatenate(results) if results else np.empty((0,self.feature_dimension),np.float32)


def load_backend(config,base_dir=None):return FrozenTorchScriptFeatures(config,base_dir)


def _patch_integrity(patch):
    image,mask=patch["image"],patch["valid_mask"]
    if image.dtype!=np.uint8 or image.shape!=(128,128,3) or mask.dtype!=bool or mask.shape!=(128,128):raise ValueError("FEATURE_PATCH_SHAPE_OR_DTYPE")
    if patch.get("planned_pixels")!=128**2 or type(patch.get("complete")) is not bool or type(patch.get("valid_pixels")) is not int:
        raise ValueError("FEATURE_PATCH_DENOMINATOR_OR_COMPLETENESS")
    if (hashlib.sha256(image.tobytes()).hexdigest()!=patch["pixel_sha256"] or
        hashlib.sha256(mask.tobytes()).hexdigest()!=patch["mask_sha256"] or
        bool(mask.all())!=patch["complete"] or int(mask.sum())!=patch["valid_pixels"]):raise ValueError("FEATURE_PATCH_INTEGRITY")


def feature_cache_key(patch,backend):
    _patch_integrity(patch)
    return digest({"schema":"rswm-feature-cache-v1","pixel_sha256":patch["pixel_sha256"],"mask_sha256":patch["mask_sha256"],
                   "complete":patch["complete"],"first_source_sha256":patch["first_source_sha256"],
                   "feature_identity_sha256":backend.manifest["feature_identity_sha256"]})


def _atomic(path,data):
    tmp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    with tmp.open("xb") as f:f.write(data)
    try:
        import os
        os.link(tmp,path)  # Exclusive atomic publish; never overwrite another entry.
    finally:
        if tmp.exists():tmp.unlink()


def encode_patches(patches,backend,cache_dir):
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True)
    manifest=validate_backend_manifest(backend.manifest)
    records=[];pending={}
    for patch in patches:
        key=feature_cache_key(patch,backend)
        record={k:v for k,v in patch.items() if not isinstance(v,np.ndarray)}
        record.update(feature_cache_key=key,feature_identity_sha256=manifest["feature_identity_sha256"],
                      diagnostic_features=manifest["diagnostic_features"],feature_dimension=backend.feature_dimension,
                      release_validated=False)
        if not patch["complete"]:
            record.update(status="MISSING_PATCH",feature=None);records.append(record);continue
        data_path=root/(key+".npy");meta_path=root/(key+".json")
        if data_path.exists() or meta_path.exists():
            if not data_path.exists() or not meta_path.exists() or data_path.is_symlink() or meta_path.is_symlink():raise ValueError("FEATURE_CACHE_PARTIAL_OR_PATH")
            info=load_json(meta_path.read_text(encoding="utf-8"))
            if (info.get("schema")!="rswm-feature-cache-entry-v1" or info.get("cache_key")!=key or
                info.get("feature_sha256")!=file_hash(data_path) or
                info.get("feature_identity_sha256")!=manifest["feature_identity_sha256"] or
                info.get("feature_dimension")!=backend.feature_dimension or
                info.get("diagnostic_features") is not manifest["diagnostic_features"]):raise ValueError("FEATURE_CACHE_INTEGRITY")
            feature=np.load(data_path,allow_pickle=False)
        else:
            pending.setdefault(key,{"image":patch["image"],"data_path":data_path,"meta_path":meta_path,"records":[]})["records"].append(record)
            records.append(record);continue
        if feature.dtype!=np.float32 or feature.shape!=(backend.feature_dimension,) or not np.isfinite(feature).all():raise ValueError("FEATURE_CACHE_SHAPE_OR_FINITE")
        record.update(status="ENCODED",feature=feature,feature_file=str(data_path.resolve()),feature_sha256=file_hash(data_path))
        records.append(record)
    if pending:
        # The real provider executes its frozen batch_size, instead of receiving
        # one patch at a time despite recording a larger requested batch.
        encoded=np.asarray(backend.encode([item["image"] for item in pending.values()]))
        if encoded.shape!=(len(pending),backend.feature_dimension) or encoded.dtype!=np.float32 or not np.isfinite(encoded).all():
            raise ValueError("FEATURE_OUTPUT_SHAPE_OR_FINITE")
        import io
        for (key,item),feature in zip(pending.items(),encoded):
            buf=io.BytesIO();np.save(buf,feature,allow_pickle=False);_atomic(item["data_path"],buf.getvalue())
            feature_hash=file_hash(item["data_path"])
            _atomic(item["meta_path"],canonical({"schema":"rswm-feature-cache-entry-v1","cache_key":key,
                    "feature_sha256":feature_hash,"feature_identity_sha256":manifest["feature_identity_sha256"],
                    "feature_dimension":backend.feature_dimension,"diagnostic_features":manifest["diagnostic_features"]})+b"\n")
            for record in item["records"]:
                record.update(status="ENCODED",feature=feature.copy(),feature_file=str(item["data_path"].resolve()),feature_sha256=feature_hash)
    return records


def save_feature_bundle(records,backend_manifest,out,mosaic_bundle_sha256):
    validate_backend_manifest(backend_manifest)
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError("Feature output must be empty")
    rows=[]
    for i,row in enumerate(records):
        feature=row.get("feature");meta={k:v for k,v in row.items() if k!="feature"}
        if feature is not None:
            if row.get("feature_cache_key") and row.get("feature_file"):
                import os
                cache_file=Path(row["feature_file"]).resolve()
                meta.update(feature_cache_file=os.path.relpath(cache_file,root.resolve()),
                            feature_cache_metadata_file=os.path.relpath(cache_file.with_suffix(".json"),root.resolve()))
            name=f"feature-{i:06d}.npy"
            with (root/name).open("xb") as f:np.save(f,feature,allow_pickle=False)
            meta.update(feature_file=name,feature_sha256=file_hash(root/name))
        else:meta.update(feature_file=None,feature_sha256=None)
        rows.append(meta)
    bundle={"schema":"rswm-quality-features-v1","features":rows,"backend_manifest":backend_manifest,
            "mosaic_bundle_sha256":mosaic_bundle_sha256,"diagnostic_features":backend_manifest["diagnostic_features"],
            "release_validated":False,"complete_tiles":sum(r["status"]=="ENCODED" for r in rows),"planned_tiles":len(rows)}
    bundle["feature_bundle_sha256"]=digest(bundle)
    with (root/"features.json").open("xb") as f:f.write(canonical(bundle)+b"\n")
    return bundle


def load_feature_bundle(path):
    path=Path(path);bundle=load_json(path.read_text(encoding="utf-8"))
    if bundle.get("schema")!="rswm-quality-features-v1" or digest({k:v for k,v in bundle.items() if k!="feature_bundle_sha256"})!=bundle.get("feature_bundle_sha256"):
        raise ValueError("FEATURE_BUNDLE_INTEGRITY")
    manifest=validate_backend_manifest(bundle["backend_manifest"])
    if bundle.get("diagnostic_features") is not manifest["diagnostic_features"] or bundle.get("release_validated") is not False:
        raise ValueError("FEATURE_BUNDLE_DIAGNOSTIC_OR_RELEASE_FLAG")
    if bundle.get("planned_tiles")!=len(bundle["features"]) or bundle.get("complete_tiles")!=sum(r["status"]=="ENCODED" for r in bundle["features"]):
        raise ValueError("FEATURE_BUNDLE_DENOMINATOR_INVALID")
    rows=[];root=path.parent.resolve()
    for row in bundle["features"]:
        if (row.get("feature_identity_sha256")!=manifest["feature_identity_sha256"] or
            row.get("diagnostic_features") is not manifest["diagnostic_features"] or row.get("release_validated") is not False):
            raise ValueError("FEATURE_BUNDLE_ROW_IDENTITY_OR_DIAGNOSTIC_FLAG")
        if (row.get("status") not in {"ENCODED","MISSING_PATCH"} or type(row.get("complete")) is not bool or
            row["complete"]!=(row["status"]=="ENCODED") or row.get("planned_pixels")!=128**2 or
            type(row.get("valid_pixels")) is not int or not 0<=row["valid_pixels"]<=128**2 or
            row["complete"]!=(row["valid_pixels"]==128**2)):
            raise ValueError("FEATURE_BUNDLE_ROW_COMPLETENESS")
        feature=None
        if row["status"]=="ENCODED":
            p=root/row["feature_file"]
            if p.is_symlink() or not p.resolve().is_relative_to(root) or file_hash(p)!=row["feature_sha256"]:raise ValueError("FEATURE_BUNDLE_FILE_INTEGRITY")
            feature=np.load(p,allow_pickle=False)
            if feature.dtype!=np.float32 or feature.shape!=(row["feature_dimension"],) or not np.isfinite(feature).all():raise ValueError("FEATURE_BUNDLE_SHAPE_OR_FINITE")
        elif row.get("feature_file") is not None or row.get("feature_sha256") is not None:
            raise ValueError("FEATURE_MISSING_PATCH_HAS_FEATURE_FILE")
        rows.append({**row,"feature":feature})
    return bundle,rows


def validate_feature_controls(patches,backend,cache_dir,*,sample_caps=(2,4,8,16),bootstrap=200,seed=0):
    """Execute frozen feature inference on role-locked reference controls.

    Derived images are measurement controls, never benchmark submissions. Their
    order is reported without imposing a desired ranking on the chosen feature.
    """
    from PIL import ImageFilter
    from collections import defaultdict
    from .quality_evaluation import evaluate_distribution
    validate_backend_manifest(backend.manifest)
    if not sample_caps or any(type(n) is not int or n<1 for n in sample_caps):
        raise ValueError("FEATURE_CONTROL_SAMPLE_CAPS_INVALID")
    allowed={"metric_calibration","metric_validation","diagnostic_only"}
    if not patches or any(p.get("role") not in allowed for p in patches):
        raise ValueError("FEATURE_CONTROLS_REQUIRE_CALIBRATION_OR_VALIDATION_ROLE")
    if any(p.get("reference_kind") not in {"frozen-real-world","diagnostic-world"} for p in patches):
        raise ValueError("FEATURE_CONTROLS_REQUIRE_REFERENCE_MOSAICS")
    if not backend.manifest["diagnostic_features"] and any(p["role"]=="diagnostic_only" for p in patches):
        # Synthetic inputs may test a real loader, but cannot establish real data
        # validation. That distinction remains explicit in the returned report.
        real_data=False
    else:real_data=all(p["role"]!="diagnostic_only" for p in patches)
    controls=[]
    for patch in patches:
        image=patch["image"];x,y=patch["xy"];yy,xx=np.mgrid[y:y+128,x:x+128]
        token=np.uint64(int(digest(["feature-control-fixed-noise-map-v1",patch["source_identity"],seed])[:16],16))
        with np.errstate(over="ignore"):
            h=xx.astype(np.uint64)*np.uint64(0x9E3779B185EBCA87)^yy.astype(np.uint64)*np.uint64(0xC2B2AE3D27D4EB4F)^token
            h=(h^(h>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
            h=(h^(h>>np.uint64(27)))*np.uint64(0x94D049BB133111EB);h^=h>>np.uint64(31)
        noise=np.stack([(h>>np.uint64(8*c)).astype(np.uint8) for c in range(3)],axis=-1)
        seam=image.copy();seam[:,64:]=255-seam[:,64:]
        variants={"oracle_reference":image.copy(),"fixed_noise_map":noise,
                  "tiled":np.tile(image[:64,:64],(2,2,1)),
                  "blur":np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(4))).copy(),
                  "color_perturbation":np.clip(image[:,:,[1,2,0]].astype(np.int16)+20,0,255).astype(np.uint8),
                  "artificial_seam":seam}
        for name,derived in variants.items():
            row={**patch,"image":derived,"model_id":name,"control_name":name,"is_measurement_control":True,
                 "sample_id":digest([patch["sample_id"],"feature-control-v1",name]),
                 "pixel_sha256":hashlib.sha256(derived.tobytes()).hexdigest()}
            controls.append(row)
    encoded=encode_patches(controls,backend,cache_dir);groups=defaultdict(list)
    for row in encoded:groups[row["role"]].append(row)
    reports=[]
    for role,rows in sorted(groups.items()):
        references=[r for r in rows if r["control_name"]=="oracle_reference"]
        for name in sorted({r["control_name"] for r in rows}):
            generated=[r for r in rows if r["control_name"]==name]
            for cap in sample_caps:
                reports.append({"role":role,"control_name":name,"sample_cap":cap,
                                "distribution":evaluate_distribution(generated,references,cap=cap,bootstrap=bootstrap,seed=seed)})
        sources=sorted({r["source_identity"] for r in references},key=lambda s:digest(["feature-real-real-split-v1",role,s]))
        left=set(sources[::2]);right=set(sources[1::2])
        x=[r for r in references if r["source_identity"] in left];y=[r for r in references if r["source_identity"] in right]
        for cap in sample_caps:
            reports.append({"role":role,"control_name":"independent_real_real","sample_cap":cap,
                            "distribution":evaluate_distribution(x,y,kind="real-real",cap=cap,bootstrap=bootstrap,seed=seed)})
    report={"schema":"rswm-feature-control-validation-v1","feature_backend_manifest":backend.manifest,
            "sample_caps":list(sample_caps),"bootstrap":bootstrap,"seed":seed,"reports":reports,
            "planned_reference_tiles":len(patches),"complete_reference_tiles":sum(p["complete"] for p in patches),
            "diagnostic_features":backend.manifest["diagnostic_features"],"real_data":real_data,
            "real_validation":"NOT_RUN" if backend.manifest["diagnostic_features"] or not real_data else "EXECUTED_CANDIDATE_CONTROLS",
            "release_validated":False,"ranking_requirement":"none; inspect candidate sensitivity and narrow interpretation",
            "second_backbone_sensitivity":"NOT_RUN"}
    report["validation_sha256"]=digest(report)
    return report


def validation_main(argv=None):
    import argparse
    from .quality_protocol import load_mosaics
    parser=argparse.ArgumentParser(description="Validate a local frozen feature candidate on calibration/validation reference controls")
    parser.add_argument("--config",required=True);parser.add_argument("--reference-mosaics",required=True)
    parser.add_argument("--cache",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--sample-caps",type=int,nargs="+",default=[2,4,8,16]);parser.add_argument("--bootstrap",type=int,default=200)
    parser.add_argument("--seed",type=int,default=0);args=parser.parse_args(argv)
    try:
        backend=load_backend(load_json(Path(args.config).read_text(encoding="utf-8")),Path(args.config).parent)
        bundle,patches=load_mosaics(args.reference_mosaics)
        report=validate_feature_controls(patches,backend,args.cache,sample_caps=args.sample_caps,bootstrap=args.bootstrap,seed=args.seed)
        report.pop("validation_sha256");report["reference_bundle_sha256"]=bundle["bundle_sha256"];report["validation_sha256"]=digest(report)
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("xb") as f:f.write(canonical(report)+b"\n")
        print(canonical({"validation_sha256":report["validation_sha256"],"reports":len(report["reports"]),
                         "diagnostic_features":report["diagnostic_features"],"real_validation":report["real_validation"],"release_validated":False}).decode())
        return 0 if any(r["distribution"]["polynomial_mmd_u"] is not None for r in report["reports"]) else 1
    except FeatureUnavailable as exc:
        print(canonical({"status":"BLOCKED","reason_code":exc.reason_code,"real_validation":"NOT_RUN","release_validated":False}).decode());return 1
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print(canonical({"status":"FAILED","reason":str(exc),"error_type":type(exc).__name__}).decode());return 2


def main(argv=None):
    """Installed extraction entry point; source script delegates here."""
    import argparse
    from .quality_protocol import load_mosaics
    parser=argparse.ArgumentParser(description="Extract complete first-commit mosaics with explicitly frozen local feature weights")
    parser.add_argument("--config",required=True);parser.add_argument("--mosaics",required=True)
    parser.add_argument("--cache",required=True);parser.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    try:
        config=load_json(Path(args.config).read_text(encoding="utf-8"))
        backend=load_backend(config,Path(args.config).parent)
        bundle,patches=load_mosaics(args.mosaics)
        records=encode_patches(patches,backend,args.cache)
        result=save_feature_bundle(records,backend.manifest,args.out,bundle["bundle_sha256"])
        print(canonical({"feature_bundle_sha256":result["feature_bundle_sha256"],"complete_tiles":result["complete_tiles"],
                         "planned_tiles":result["planned_tiles"],"diagnostic_features":result["diagnostic_features"],
                         "release_validated":False}).decode())
        return 0 if result["complete_tiles"]==result["planned_tiles"] and result["complete_tiles"]>0 else 1
    except FeatureUnavailable as exc:
        print(canonical({"status":"BLOCKED","reason_code":exc.reason_code,"real_weight_validation":"NOT_RUN","release_validated":False}).decode());return 1
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print(canonical({"status":"FAILED","reason":str(exc),"error_type":type(exc).__name__}).decode());return 2


def validate_feature_release(receipt,feature_config_sha256,source_manifest_sha256):
    """Verify retained real calibration, preregistration and held-out release."""
    from .feature_release import validate_feature_release as validate
    try:return validate(receipt,feature_config_sha256,source_manifest_sha256)
    except (KeyError,TypeError,OSError,FeatureUnavailable) as exc:
        raise ValueError("BLOCKED_FEATURE_RELEASE_INCOMPLETE_OR_UNAVAILABLE_ARTIFACT: "+str(exc)) from exc


def release_main(argv=None):
    from .feature_release import main as release
    return release(argv)


if __name__=="__main__":raise SystemExit(main())

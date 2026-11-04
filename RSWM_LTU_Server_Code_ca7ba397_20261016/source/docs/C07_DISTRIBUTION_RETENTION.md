# C07 实际分发包留存和读回

此路径签收具体提交的内部软件交付。它不签收真实模型、GPU、数据、隔离或论文实验，不发布 tag，不合并 main。

完成所有实现修改和提交后，在干净的最终 HEAD 上执行。`--commit` 必须来自预期提交，不能用包内自报身份代替。构建前后均核对 Git HEAD、全部工作树 dirty 状态及包源码指纹。输出必须位于源码 checkout 外的新目录。

```bash
FINAL_COMMIT=$(git rev-parse HEAD)
python -m pip download --only-binary=:all: --dest /tmp/rswm-dependencies-new \
  'numpy>=1.24,<3' 'Pillow>=10,<13' 'setuptools>=68' wheel
python scripts/retain_distribution.py build --commit "$FINAL_COMMIT" \
  --dependencies /tmp/rswm-dependencies-new --out /tmp/rswm-retained-new
python scripts/retain_distribution.py verify --commit "$FINAL_COMMIT" \
  --retained /tmp/rswm-retained-new
python scripts/retain_distribution.py pack --commit "$FINAL_COMMIT" \
  --retained /tmp/rswm-retained-new --bundle /tmp/rswm-distribution-new.zip
```

留存目录包含实际 `dist/*.whl` 和 `dist/*.tar.gz`、离线安装依赖、构建 stdout/stderr、`build-manifest.json` 和 `SHA256SUMS`。构建清单记录 commit、dirty、包源码指纹、依赖环境和每个实际归档成员的大小、SHA256、原始源文件映射或生成元数据身份。sdist 的全部 authored inputs 必须逐项与实际成员字节相同；wheel 的包内 Python 文件必须与同一源身份相同。未记录的额外输入以及原始影像、权重、凭据、缓存会失败。

`pack` 返回独立的 ZIP SHA256。保存该 digest；ZIP hash 与内部 wheel/sdist hash 是不同对象。使用保存的 ZIP digest 实际读回到另一新目录，再在两个独立干净环境中分别安装 wheel 和从 sdist 离线重建安装：

```bash
python scripts/retain_distribution.py accept --commit "$FINAL_COMMIT" \
  --bundle /tmp/rswm-distribution-new.zip --bundle-sha256 "$RETAINED_ZIP_SHA256" \
  --readback /tmp/rswm-downloaded-new --work /tmp/rswm-installed-work-new \
  --out /tmp/rswm-installed-evidence-new
```

读回先验证外层 ZIP hash，再验证实际清单、所有保留文件和内部归档 hash。每个新 venv 在源码目录外执行已有安装验收：核对导入路径和源码身份、默认 F2/F7 的 192 个诊断输出、PNG replay、资源和表格，以及真实模型/隔离门的诚实 BLOCKED。sdist 安装明确使用保留归档及离线 setuptools/wheel，实际执行构建；没有用已保留 wheel 替它签收。默认模式明确为 software diagnostic，真实模型和 GPU 均 NOT_RUN。

CI `test` job 留存实际分发目录；`distribution-readback` job 通过 GitHub artifact API 下载该实际上传产物的 ZIP，并核对 `upload-artifact` 返回的 digest。下载器仅向 GitHub API 请求传递 token，不向跨域存储 redirect 传递。该 job 重新验证内部包，再分别验收 wheel/sdist，保留命令日志、归档输入清单、安装身份和 `readback-summary.json`。必要产物缺失、错误字节或安装失败使 job 失败。

本地从留存服务下载的 ZIP 也可走同一 `accept` 命令。仅本地打包读回不能冒充远端 CI 上传后的下载验收；尚未取得最终 HEAD 的远端 CI 终态时，报告应明确该边界。

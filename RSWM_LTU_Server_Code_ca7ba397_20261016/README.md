# LTU 最后运行版代码备份：ca7ba397

整理日期：2026-10-16，北京时间。

这份备份对应最后聊天中实际运行的目录：

```text
/data/group/zhaolab/project/rswm/work/rswm-code-v0.4.2-dynstate-na
```

运行包内全部 54 个 Python 文件的整体源码指纹为：

```text
ca7ba3977bba0f71d1165526e1caf02ada6d67c555086e3636679ece647161ca
```

该指纹与 LTU 记录一致。补丁链三个中间身份 f73597 → ec4495 → ca7ba397 均已逐字节重建并核对。恢复只发生在本次工作目录，原有源码和结果保留。

## 目录

| 目录／文件 | 内容 |
| --- | --- |
| `runtime/rswm-code-v0.4.2-dynstate-na/` | 对应服务器 PYTHONPATH 布局的运行包；54 个 Python 源文件整体身份与服务器记录一致。 |
| `source/` | 完整保留的源码工程、scripts、tests、docs 与 vendor；其中核心包覆盖为 ca7ba397，Motion24 helper 按实际服务器补丁恢复。 |
| `helpers/motion24-helper-3ec82/` | 从已交付服务器包和记录中的 inventory 常量修复恢复的 Motion24 脚本。 |
| `provenance/` | 服务器身份、恢复配方、源码清单与本次静态核验结果。 |
| `verify_package.py` | 只读检查文件哈希和两个核心包的源码指纹，只依赖 Python 标准库。 |
| `SHA256SUMS` | 包内逐文件 SHA256 清单，清单本身不作自引用。 |

## 本地核验

解压后，在此目录执行：

```bash
python -B verify_package.py
```

这条命令不加载权重，不启动训练，不提交调度任务。

在已配置依赖的环境中，可额外核对实际导入路径与程序自身指纹：

```bash
export PYTHONPATH="$PWD/runtime/rswm-code-v0.4.2-dynstate-na"
export PYTHONDONTWRITEBYTECODE=1
python -B -c 'import rswm_inria; from rswm_inria.suite import code_fingerprint; print(rswm_inria.__file__); print(code_fingerprint())'
```

这份备份没有自动提交或恢复服务器任务。服务器恢复后，应先回读既有作业和实际源码身份，再决定如何接续。

## 身份与范围

- 服务器维修期间未重新连接现场。本包由本地保留源码和原始会话中的热修复命令恢复；54 个 Python 文件的整体指纹与最后服务器记录完全一致。
- 服务器整体文件树、依赖环境、prepared Wan 后端、真实模型配置、计划 JSON、数据、权重和输出不在这项源码指纹的核验范围内。保留的 vendor、文档、测试和配置示例单独作为辅助源码，不冒充新的服务器实测。
- Motion24 helper 使用已交付包中的原脚本，加记录中的 inventory SHA 修复；其完整文件 SHA 已记录，但没有独立的最后远端脚本校验值。
- `source/src/rswm_inria/_build_info.json` 是历史构建记录。它保留原字节，不作为本次恢复身份；运行目录不复制这个历史元数据。恢复版本以 `provenance/SERVER_IDENTITY.json` 和实时计算的源码指纹为准。
- 本包包含代码与必要的源码说明，不包含模型权重、原始影像、生成结果、虚拟环境和完整私有聊天。
- 本次完成的是源码恢复、静态检查与归档完整性核对。历史完整 pytest、真实 GPU、模型性能与科研验收未在本次重跑。

# 版本核实

## 建议基线

- 源归档：`rsroll-bench-server-20260731c.tar.gz`
- 大小：1,568,993 bytes
- SHA-256：`FCA092BA5F294EE7FD3DA3A485484CA4716698237BCEE8ABBA9CB20B6591E9D6`
- 内部主要文档版本：v3.0/v3.1 系列；本次 manifest 升为 3.2.0。

归档不包含可用 `.git` 历史，因此 branch、HEAD、tag、remote commit 与未提交状态均无法从
源包核实。本次工作目录外层 Git 仓库不是原项目的上游历史，不能拿来冒充源项目 commit。

未发现能唯一确定官方上游仓库与 commit 的可靠 Git 元数据。本次没有 `git fetch`、没有
升级依赖、没有下载 checkpoint，也没有宣称远程最新版已核实。

## 源目录保护

只读源归档未被解包覆盖或修改。本次修改发生在独立工作副本，最终交付到
`C:\Users\25828\Desktop\lora`。验收时应再次计算源归档 SHA-256；只要仍等于上述值，
即可证明源包内容未变化。

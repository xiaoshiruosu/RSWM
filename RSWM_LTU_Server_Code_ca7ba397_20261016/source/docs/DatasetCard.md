# DatasetCard

来源是冻结revision的Inria RGB镜像；标签/预览/GT不参与源选择。inventory记录完整路径、字节数与可用LFS SHA；选择由版本namespace/path SHA排序与城市top-k决定。当前默认revision为`ad8315c82c8a3021b93a7d52922d19ea76201f60`。摘要证明内容身份，不能证明源真实性、许可或地理独立性。

| 城市 | 角色 | 核心数量 |
| --- | --- | ---: |
| Austin / Chicago / Vienna | train | 各20 |
| Kitsap | dev | 12 |
| Bellingham | metric_calibration | 12 |
| Bloomington | metric_validation | 12 |
| SFO / Innsbruck | test_unseen_city | 各12 |

真实单页TIFF必须为5000×5000×3无符号8-bit RGB原始样本。不能把RGB转换后的PNG、更高位深数据或多页文件称作合规源。物化中央4096²再用uint16求和`(sum+2)//4`得到2048²世界；独立审计以四个stride切片复算，逐像素比较世界、所有I0、任务几何和私有标签。

`rswm-materialize`默认读取本地源；显式`--download`受预先核对的字节预算限制。`smoke12`选冻结角色前缀：9 train和3 dev。首件验收不是全smoke12，synthetic单世界/单起点子集不是12真实世界。真实smoke12三诊断要求576 runs/50,688新输出，未获得真实源时不得生成该验收身份。

产物分为`public/initial`、`public/sessions`及`private/worlds`、`private/tasks`、`private/tasks/labels`、`private/index.json`、`private/training-clips.json`。在线公共session不包含未来轨迹。新标签相对路径在task内受限；旧`../labels`布局须在新目录重新构建。完整文件inventory检查会拒绝额外/缺失文件、symlink和坏hash。

训练固定60世界×64clip=3840clip，每clip33视图，共126,720视图事件。正式loader要求core120真实物化audit、Austin/Chicago/Vienna各20、真实任务hash与源/世界hash；仅修改role字段不能放行。dev验证只使用同audit的Kitsap dev世界。Bellingham/Bloomington和test世界不进入训练梯度。

LEVIR扩展在固定原Parquet身份上先选100行再解码，保留1024² native世界；不能把preview或放大到2048的图像冒充native1024。真实源当前未提供，相关实景验收保持BLOCKED。

精确重复像素记录并要求审查，不能静默换样；重复hash检查不证明地理分离。镜像许可未解决及代码许可待负责人确定，见[NOTICE](../NOTICE.md)。仓库/分发包不包含原影像、世界、标签NPZ、预测PNG或权重。构建命令与首件完整性见[G2指南](G2_DATA_ACCEPTANCE.md)。

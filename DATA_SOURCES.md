# 数据与第三方署名

本仓库是处理后的研究数据和代码快照，不是原项目的官方发布。完整实际来源记录、原始字段、单位、样本路径与哈希保存在发布附件中；不可仅因指标同名而合并监督。

- **FALCON**：Asal Mehradfar 等；官方仓库 https://github.com/AsalMehradfar/FALCON 。所用提交见 `reproducibility/upstream_commits.json`。原项目MIT版权声明保存在 `LICENSES/`。本项目对其数据进行解析、图表示统一和分层抽样，采用200,000条记录。
- **AICircuit**：Asal Mehradfar、Yue Niu、Sara Babakniya、Salman Avestimehr、Xuzhe Zhao、Mahdi Alesheikh、Hamidreza Aghasi；官方仓库 https://github.com/AvestimehrResearchGroup/AICircuit 。原项目MIT声明保留USC与UCI版权。处理后采用100,000条记录。
- **AnalogGym**：CODA-Team，https://github.com/CODA-Team/AnalogGym ，BSD-3-Clause。自建20,000条记录依据其电路拓扑，通过实际ngspice仿真获得；它们不是官方预先提供的同一批监督样本。工艺、测试条件、参数采样和质量策略保存在自建集。
- **LOADBench**：Markus Leibl、Filipe Azevedo、Stefan Uhlich、Andrea Bonetti、Arun Venkitaraman、Chia-Yu Hsieh、Phuoc Pham、Lorenzo Servadei、Helmut Graeb、Ricardo Martins。*LOADBench: A Large Open Analog Dataset and Benchmark for Machine Learning in IC Design – Dataset*，DOI https://doi.org/10.5281/zenodo.21759759 ，CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ 。保留原Zenodo记录（包含作者、版本与下载元数据）；本项目仅使用实际下载的优化结果／历史子集，经兼容性审查和抽样得到200,000条记录，并未下载完整优化历史。代码来源另固定在 `third_party/LOADBench-Scripts`。

转换包括端口感知图构建、SI字段与工程尺度记录、缺失Mask、监督契约审查、去重分组和固定划分。原始标签、极端记录及负向结果没有为改善指标而删除。来源作者不代表对本派生项目或结论的认可。

公开数据与自建数据分别用于不同实验。当前研究未确认可共享的跨数据集零样本监督目标；数据打包不表示标签已经跨来源等价。

可选AC诊断附件中的精简SKY130模型保留原文件的SkyWater PDK Authors版权及Apache-2.0声明，完整许可见 `LICENSES/SKY130_Apache-2.0.txt`。完整PDK与ngspice可执行环境没有打包；它们的来源与版本另由配置和原项目标识。

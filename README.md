# ego_data_process

一个基于 Hydra + Ray 的可配置数据处理框架，用于把第一视角手部/机械臂数据整理成统一中间格式，并继续执行 retarget、IK、可视化和渲染。

当前仓库已经包含两条主要处理链：

- 通用处理链：`load_data -> retarget -> inverse_kinematics -> write_data -> visualize`
- EgoDex 处理链：`load_egodex_data -> retarget -> inverse_kinematics -> write_data -> visualize`

渲染链单独运行：

- 渲染链：`processed_loader -> render_filter -> render`

## 1. 框架结构

目录约定：

- `config/`: Hydra 配置。包含 data loader、process、runtime 和 pipeline 入口。
- `data_loader/`: 数据发现层。负责枚举样本，生成 `sample_id / video_path / data_path / visualize` 等字段。
- `process/`: 处理节点。每个 process 接收一个 sample dict，输出更新后的 sample dict。
- `run/`: 运行入口。负责配置解析、并行调度、manifest 管理、meta 文件写回。
- `utils/`: 公共工具，包括 IK、路径映射、原子写文件、manifest、进度条等。
- `assets/`: MANO、URDF、mesh 等模型资源。
- `examples/`: 本地示例数据。
- `outputs/`: 默认输出目录。

运行时模型：

- DataLoader 先发现全部样本。
- `sample_id` 作为样本主键，同时决定输出相对路径。
- Pipeline 按配置顺序执行 enabled process。
- `run/run.py` 使用 Ray actor 并行执行样本。
- `ManifestStore` 记录 `pending / in_progress / completed`，支持 `resume` 和异常恢复。
- 处理完成后自动写出 `meta.json` 和 `render_meta.json`。

## 2. 主要 Pipeline

### 2.1 通用 Pipeline

入口配置：`[config/pipeline.yaml](/Users/cyxovo/ego_data_process/config/pipeline.yaml)`

默认 data loader：

- `[config/data/database_loader.yaml](/Users/cyxovo/ego_data_process/config/data/database_loader.yaml)`
- `[config/data/random_database_loader.yaml](/Users/cyxovo/ego_data_process/config/data/random_database_loader.yaml)`
- `[config/data/csv_loader.yaml](/Users/cyxovo/ego_data_process/config/data/csv_loader.yaml)`

默认 process 链：

- `[config/processes/process.yaml](/Users/cyxovo/ego_data_process/config/processes/process.yaml)`

输出：

- `outputs/data/*.parquet`
- `outputs/meta_data/*.json`
- `outputs/meta.json`
- `outputs/samples/*` 可视化结果

### 2.2 EgoDex Pipeline

入口配置：`[config/egodex_pipeline.yaml](/Users/cyxovo/ego_data_process/config/egodex_pipeline.yaml)`

默认 data loader：

- `[config/data/egodex_database_loader.yaml](/Users/cyxovo/ego_data_process/config/data/egodex_database_loader.yaml)`
- `[config/data/egodex_csv_loader.yaml](/Users/cyxovo/ego_data_process/config/data/egodex_csv_loader.yaml)`

本地目录版 data loader：

- `[config/data/egodex_glob_loader.yaml](/Users/cyxovo/ego_data_process/config/data/egodex_glob_loader.yaml)`

默认 process 链：

- `[config/processes/egodex_process.yaml](/Users/cyxovo/ego_data_process/config/processes/egodex_process.yaml)`

说明：

- EgoDex 的 `sample_id` 会把 `part/task/id.mp4` 或 `part/task/id.hdf5` 规范化成 `part_task_id`。
- 默认 database loader 也使用相同规则。

### 2.3 Render Pipeline

入口配置：`[config/pipeline_render.yaml](/Users/cyxovo/ego_data_process/config/pipeline_render.yaml)`

默认 data loader：

- `[config/data/processed_loader.yaml](/Users/cyxovo/ego_data_process/config/data/processed_loader.yaml)`

默认 process 链：

- `[config/processes/render.yaml](/Users/cyxovo/ego_data_process/config/processes/render.yaml)`

输出：

- `outputs/render/*.mp4`
- `outputs/render_meta.json`

## 3. 安装

推荐 Python 版本：`3.10`

### 3.1 基础环境

```bash
conda create -n ego python=3.10 -y
conda activate ego
python -m pip install -r requirements.txt
```

### 3.2 通用手部处理依赖

`load_data` 依赖 MANO、`manopth` 和 `chumpy`。

```bash
git clone https://github.com/hassony2/manopth.git ../manopth
python -m pip install -e ../manopth
python -m pip install chumpy --no-build-isolation
python scripts/patch_chumpy.py
```

### 3.3 EgoDex 依赖

`load_egodex_data` 额外需要 `h5py`。

```bash
python -m pip install h5py
```

### 3.4 IK 依赖

`inverse_kinematics` 依赖 Pinocchio 和 HPP-FCL。

```bash
conda install -c conda-forge pinocchio hpp-fcl -y
```

### 3.5 Render 依赖

`render` 依赖 Genesis。

```bash
python -m pip install genesis-world
```

### 3.6 数据库读取依赖

如果使用 `database_loader`，需要 `supabase`。`requirements.txt` 已包含，但也可以单独安装：

```bash
python -m pip install supabase
```

## 4. 运行命令

### 4.1 运行通用处理链

```bash
python -m run.run   data=database_loader   runtime.num_workers=256 data.params.dataset_name=Ego10k   processes.0.params.output_dir.remote=oss://ss-oss1/data/dataset/egocentric/Egocentric-10K/processed data.params.num_parts=8 data.params.part=5 runtime.resume=true
```

常用覆盖参数：

```bash
python -m run.run --config-name pipeline runtime.num_workers=1 runtime.limit=10
python -m run.run --config-name pipeline runtime.resume=true
python -m run.run --config-name pipeline data=random_database_loader data.params.random_threshold=0.01 data.params.query_limit=100
python -m run.run --config-name pipeline data=random_database_loader data.params.dataset_name=Ego10k data.params.random_threshold=0.01 data.params.query_limit=100
python -m run.run --config-name pipeline data=csv_loader
python -m run.run --config-name pipeline processes=process_no_ik
```

### 4.2 运行 EgoDex 处理链

默认使用数据库版 EgoDex loader：

```bash
python -m run.run --config-name egodex_pipeline
```

如果要改成本地目录版：

```bash
python -m run.run --config-name egodex_pipeline data=egodex_glob_loader
```

如果要改成读取 `/Users/cyxovo/Downloads/samples.csv` 里仅 `ml-egodex` 的样本：

```bash
python -m run.run --config-name egodex_pipeline data=egodex_csv_loader
```

如果只想处理少量样本：

```bash
python -m run.run --config-name egodex_pipeline runtime.limit=20 runtime.num_workers=1
```

### 4.3 运行渲染

渲染默认读取 `outputs/` 下已经生成的 processed 数据：

```bash
export PYGLET_HEADLESS=1
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
python -m run.run --config-name pipeline_render runtime.num_workers=128 processes.1.params.genesis_backend=gpu
```

如果 processed 数据不在默认目录：

```bash
python -m run.run --config-name pipeline_render data.params.input_dir=/path/to/processed_root
```

### 4.4 分片渲染

`processed_loader` 支持分片，`part` 是 0-based，且相同 `num_parts` 下不同 `part` 不会重复。

```bash
python -m run.run --config-name pipeline_render data.params.num_parts=4 data.params.part=0
python -m run.run --config-name pipeline_render data.params.num_parts=4 data.params.part=1
python -m run.run --config-name pipeline_render data.params.num_parts=4 data.params.part=2
python -m run.run --config-name pipeline_render data.params.num_parts=4 data.params.part=3
```

## 5. 输出约定

默认输出根目录是 `./outputs`。

处理链输出：

- `outputs/data/<sample_id>.parquet`
- `outputs/meta_data/<sample_id>.json`
- `outputs/meta.json`

渲染链输出：

- `outputs/render/<sample_id>.mp4`
- `outputs/render_meta.json`

说明：

- 当 `sample_id` 包含目录层级时，输出目录也会保留对应层级。
- EgoDex 现在默认使用扁平的 `part_task_id`，因此输出文件也会是扁平命名。

## 6. 配置说明

常用配置入口：

- 通用 pipeline: `[config/pipeline.yaml](/Users/cyxovo/ego_data_process/config/pipeline.yaml)`
- EgoDex pipeline: `[config/egodex_pipeline.yaml](/Users/cyxovo/ego_data_process/config/egodex_pipeline.yaml)`
- Render pipeline: `[config/pipeline_render.yaml](/Users/cyxovo/ego_data_process/config/pipeline_render.yaml)`
- 通用 process: `[config/processes/process.yaml](/Users/cyxovo/ego_data_process/config/processes/process.yaml)`
- EgoDex process: `[config/processes/egodex_process.yaml](/Users/cyxovo/ego_data_process/config/processes/egodex_process.yaml)`
- Render process: `[config/processes/render.yaml](/Users/cyxovo/ego_data_process/config/processes/render.yaml)`
- Runtime: `[config/runtime/default.yaml](/Users/cyxovo/ego_data_process/config/runtime/default.yaml)`

常用 Hydra override：

```bash
python -m run.run --config-name egodex_pipeline runtime.num_workers=8
python -m run.run --config-name egodex_pipeline runtime.resume=true
python -m run.run --config-name egodex_pipeline runtime.limit=100
python -m run.run --config-name egodex_pipeline processes.1.params.retarget_scheme=pinch_plane
python -m run.run --config-name pipeline_render data.params.part=0 data.params.num_parts=8
```

## 7. 注意事项

- `load_data` 和 `load_egodex_data` 依赖不同，不要只装一半。
- `pipeline_render` 依赖 processed 输出，不能直接对原始手部数据渲染。
- 当前 EgoDex pipeline 默认走 database loader；如果本地调试请显式使用 `data=egodex_glob_loader`。
- `requirements.txt` 里没有把所有可选依赖都列全，尤其是 `h5py`、`genesis-world`、`pinocchio`、`hpp-fcl`、`manopth`、`chumpy`。

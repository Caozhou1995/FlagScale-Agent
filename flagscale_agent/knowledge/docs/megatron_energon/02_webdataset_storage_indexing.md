# 第2章：WebDataset 存储格式与索引系统 深度源码分析

## 1. 概述与设计动机

Megatron-Energon 的数据存储层基于 WebDataset 格式，但在其上构建了完整的索引和分片系统，解决了原始 WebDataset 的三大缺陷：
- **随机访问**：原始 tar 只能顺序读取，无法跳转到特定 sample
- **断点恢复**：无法精确恢复到训练中断时的 sample 位置
- **分布式分配**：需要按 sample 粒度（而非 shard 粒度）精确分配给各 worker

核心设计思想：**tar 文件只负责存储，索引文件负责寻址**。通过预构建 `.tar.idx`（二进制偏移索引）和 `index.sqlite`（结构化元数据索引），将 O(n) 顺序扫描降为 O(1) 随机访问。

## 2. 源码定位

| 组件 | 文件路径 | 行数 | 职责 |
|------|----------|------|------|
| 数据结构 | flavors/webdataset/structs.py | 115 | WebdatasetInfo, ShardInfo, FilteredSample, DatasetSubset |
| 常量配置 | flavors/webdataset/config.py | 15 | 文件名常量、正则表达式 |
| 二进制索引 | flavors/webdataset/itar.py | 391 | TarIndexReader/Writer, SubFileReader, CachedItarOffsetReader |
| SQLite索引 | flavors/webdataset/indexing.py | 681 | SqliteIndexWriter/Reader, 索引查询 |
| 数据准备 | flavors/webdataset/prepare.py | 875 | 预处理流水线、并行索引构建 |
| Shard分配 | flavors/webdataset/sharder.py | 407 | Sharder — 全局worker分配、位反转排列 |
| 元数据 | flavors/webdataset/metadata.py | 176 | get_dataset_info — 加载.nv-meta信息 |
| Sample加载 | flavors/webdataset/sample_loader.py | 469 | WebdatasetSampleLoader — 按索引读取sample |

## 3. 架构总览

### 3.1 存储目录结构

```
dataset/
├── shard_000000.tar          # 原始数据 tar 包
├── shard_000000.tar.idx      # 二进制偏移索引（每 sample 8字节）
├── shard_000001.tar
├── shard_000001.tar.idx
├── .nv-meta/                 # Energon 元数据目录
│   ├── .info.yaml            # shard_counts: {shard_name: num_samples}
│   ├── split.yaml            # train/val/test 划分
│   ├── index.sqlite          # 全局 SQLite 索引
│   └── index.uuid            # 索引版本 UUID
```

### 3.2 索引层次关系

```
┌─────────────────────────────────────────────────┐
│                 index.sqlite                      │
│  ┌─────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ samples │  │ sample_parts │  │media_metadata│ │
│  └────┬────┘  └──────┬───────┘  └────────────┘ │
└───────┼──────────────┼──────────────────────────┘
        │              │
        ▼              ▼
┌───────────────┐ ┌───────────────┐
│ .tar.idx (bin)│ │ .tar.idx (bin)│   ← 每个 tar 一个
│ [off0|off1|..]│ │ [off0|off1|..]│
└───────┬───────┘ └───────┬───────┘
        │                  │
        ▼                  ▼
┌───────────────┐ ┌───────────────┐
│ shard_000.tar │ │ shard_001.tar │   ← 原始数据
└───────────────┘ └───────────────┘
```

## 4. 二进制偏移索引（.tar.idx）

### 4.1 设计动机（WHY）

tar 文件是流式格式，每个文件条目前有 512 字节 header。要读取第 N 个 sample，必须顺序扫描前 N-1 个 header。对于百万级 sample 的 shard，这不可接受。

解决方案：预扫描 tar 文件，记录每个 sample 的起始字节偏移到 `.tar.idx` 文件。之后通过 `seek(offset)` 实现 O(1) 随机访问。

### 4.2 实现分析（HOW）

**TarIndexWriter**（itar.py:L86-117）：

```python
class TarIndexWriter:
    def __init__(self, tar_path: EPath):
        self.final_name = tar_path.with_suffix(".tar.idx")
        self.tmp_name = tar_path.with_suffix(".tar.idx.tmp")
        self.itar = self.tmp_name.open("wb")
    
    def append(self, offset: int):
        # 每个偏移量固定 8 字节（uint64, little-endian）
        self.itar.write(struct.pack("Q", offset))
    
    def close(self, finalize: bool = True):
        self.itar.close()
        if finalize:
            self.tmp_name.move(self.final_name)  # 原子重命名
```

**TarIndexReader**（itar.py:L48-83）：

```python
class TarIndexReader:
    def __init__(self, tar_path):
        index_path = tar_path.with_suffix(".tar.idx")
        self._length = index_path.size() // 8  # 总 sample 数
        self.itar = index_path.open("rb")
    
    def __getitem__(self, index: int) -> int:
        # O(1) 随机访问：直接 seek 到 8*index 位置
        self.itar.seek(8 * index)
        return struct.unpack("Q", self.itar.read(8))[0]
```

### 4.3 CachedItarOffsetReader — 缓存优化

（itar.py:L197-391）

**设计动机**：训练时 sample_loader 按顺序或小跳跃访问索引，频繁 seek 效率低。CachedItarOffsetReader 用前瞻读取（lookahead）减少 I/O：

```
读取模式：顺序访问 sample 0, 1, 2, 3, ...
优化策略：一次读取多个 offset 到内存缓存
```

### 4.4 SubFileReader — tar 内子文件读取

（itar.py:L120-177）

当定位到 sample 的字节偏移后，需要读取该 sample 内的具体文件（如 .jpg, .txt）。SubFileReader 提供了一个受限的 file-like 对象，只暴露 `[offset, offset+size)` 范围：

```python
class SubFileReader(BinaryIO):
    def __init__(self, stream, offset, size):
        self.offset = offset
        self.size = size        # 不允许读超
        self.stream = stream
        self.stream.seek(self.offset)
    
    def read(self, n=-1):
        n = min(n, self.size - self._pos)  # 边界保护
        return self.stream.read(n)
```

### 4.5 数据流：从 sample_index 到 bytes

```
sample_index=42
    │
    ▼ TarIndexReader[42]
byte_offset=1048576
    │
    ▼ tar_file.seek(byte_offset)
tar header (512 bytes) → 解析 filename, size
    │
    ▼ SubFileReader(stream, offset+512, size)
raw bytes of file "000042.jpg"
```

## 5. SQLite 结构化索引

### 5.1 设计动机（WHY）

`.tar.idx` 只记录字节偏移，无法：
- 按 sample_key 查找（如 "image_00042"）
- 查询 sample 内的具体 part（如只要 .json 不要 .jpg）
- 支持 exclude list（按 key 排除坏数据）
- 存储媒体元数据（分辨率、时长等）用于过滤

SQLite 提供了 SQL 查询能力，且是单文件数据库，适合分布式文件系统。

### 5.2 表结构（indexing.py:L99-145）

**samples 表** — 全局 sample 索引：
```sql
CREATE TABLE samples (
    tar_file_id INTEGER NOT NULL,   -- 对应哪个 tar 文件
    sample_key TEXT NOT NULL UNIQUE, -- 唯一标识，如 "000042"
    sample_index INTEGER NOT NULL,   -- 在该 tar 内的序号
    byte_offset INTEGER,             -- tar 内字节偏移
    byte_size INTEGER                -- sample 总字节大小
);
-- 索引：(sample_key), (tar_file_id, sample_index)
```

**sample_parts 表** — part 级别细粒度索引：
```sql
CREATE TABLE sample_parts (
    tar_file_id INTEGER,
    sample_index INTEGER,
    part_name TEXT,                   -- 文件扩展名/part名，如 "jpg", "txt"
    content_byte_offset INTEGER,     -- part 数据的字节偏移
    content_byte_size INTEGER        -- part 数据大小
);
-- 索引：(tar_file_id, sample_index, content_byte_offset)
```

**media_metadata 表** — 可选的媒体元数据：
```sql
CREATE TABLE media_metadata (
    entry_key TEXT PRIMARY KEY,      -- sample_key 或 sample_key/part_name
    metadata_type TEXT NOT NULL,     -- "image", "video", "audio" 等
    metadata_json TEXT NOT NULL      -- JSON 格式的元数据
);
```

### 5.3 写入流程（indexing.py:L147-183）

```python
class SqliteIndexWriter:
    def append_samples(self, rows: Sequence[IndexSample]):
        # 使用 SAVEPOINT 实现批量原子写入
        self.db.execute("SAVEPOINT append_samples_batch")
        try:
            self.db.executemany(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                ((r.tar_file_id, r.sample_key, r.sample_index, 
                  r.byte_offset, r.byte_size) for r in rows)
            )
            self.db.execute("RELEASE SAVEPOINT ...")
        except sqlite3.IntegrityError:
            # 遇到重复 key → 回滚并报告
            self.db.execute("ROLLBACK TO SAVEPOINT ...")
            raise DuplicateSampleKeyError(...)
```

**关键设计**：使用 SAVEPOINT 而非全局事务，允许在大批量插入中精确回滚单批，而不丢失之前的进度。

### 5.4 读取接口（indexing.py:L300+）

SqliteIndexReader 提供两种查询模式：
1. **按全局 sample_index 查询**：O(1) 通过索引定位
2. **按 sample_key 查询**：用于 exclude list 过滤和 joined dataset 关联

### 5.5 ThreadLocalSqlite — 多线程安全

（thread_local_sqlite.py:L1-154）

SQLite 默认不支持跨线程共享连接。Energon 用 `threading.local()` 为每个 DataLoader worker 线程维护独立的 SQLite 连接：

```python
class ThreadLocalSqlite:
    def __init__(self, db_path: EPath):
        self._local = threading.local()
        self._db_path = db_path
    
    @property
    def connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self._db_path)
        return self._local.conn
```

## 6. Sharder — 分布式 Sample 分配

### 6.1 设计动机（WHY）

传统方式按 shard 分配给 worker，但 shard 大小不均会导致负载不均衡。Energon 按 **sample 粒度**分配，确保每个 global worker 获得几乎相同数量的 sample（最多差 1）。

另一个关键设计：**global_workers = num_workers × world_size**，使得分配结果不依赖具体的 rank/worker 划分方式。例如 4 rank × 2 workers = 2 rank × 4 workers，产出相同的全局 batch。

注：此处 world_size 对应 DP 维度的进程数（而非总 GPU 数）。在 TP/PP/CP 等并行下，只有 DP rank 各自独立加载数据，非 DP rank 通过 broadcast_data 获取。Sharder 本身不感知 TP/PP 拓扑，它只需要 WorkerConfig 中的 rank 和 world_size（即 DP 维度）。

### 6.2 广义位反转排列（sharder.py:L138-188）

**问题**：如果按顺序分配余量 sample（前 K 个 worker 各多 1 个），会导致连续的 rank 负载不均。

**解决方案**：用广义位反转排列（Generalized Bit Reversal）打散余量分配：

```python
@classmethod
def _generalized_bit_reversal(cls, length_or_indices):
    """递归 divide-and-interleave 算法。
    对 power-of-2 长度等价于二进制表示反转。
    
    例 16 个索引: [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
    """
    if len(indices) <= 2:
        return indices
    mid = len(indices) // 2
    left_result = cls._generalized_bit_reversal(indices[:mid])
    right_result = cls._generalized_bit_reversal(indices[mid:])
    # 交替合并
    return [item for pair in zip_longest(left_result, right_result) 
            for item in pair if item is not None]
```

**效果**：余量 sample 均匀散布到所有 rank，避免热点。

### 6.3 split_samples_to_workers 核心流程（sharder.py:L190-267）

```
输入：start_samples, end_samples, worker_config, rotation_offset
输出：local_worker_sample_split_offsets（当前 rank 各 worker 的 sample 范围）

步骤：
1. total = end - start
   global_workers = num_workers × world_size
   min_per_worker = total // global_workers
   remainder = total % global_workers

2. 按 rotation_offset 轮转决定哪些 worker 多 1 个 sample
   → num_samples_per_global_worker[0..global_workers-1]

3. 用位反转排列重排余量分配
   → 打散 "多1个sample" 的 worker 分布

4. 累加得到 global_worker_sample_split_offsets
   → [start, start+n0, start+n0+n1, ..., end]

5. 提取当前 rank 的 slice：
   offsets[rank*num_workers : (rank+1)*num_workers + 1]
```

### 6.4 shard_workers — 完整分片流程（sharder.py:L313-362）

```python
@classmethod
def shard_workers(cls, shards, worker_config, *, 
                  max_samples_per_sequence, subset, rotation_offset):
    # 1. 计算总 sample 数
    end_samples = sum(shard.count for shard in shards)
    # 2. 应用 subset（绝对范围 + 相对比例）
    start_samples, end_samples = subset.compute_subset(end_samples)
    # 3. 按 sample 分配到 workers
    offsets = cls.split_samples_to_workers(start, end, worker_config)
    # 4. 按 shard 边界切分（尊重 max_samples_per_sequence）
    shard_cumsums = np.cumsum([0] + [s.count for s in shards])
    return tuple(cls._clean_offsets(off) 
                 for off in cls._split_shards(shard_cumsums, offsets, ...))
```

### 6.5 max_samples_per_sequence 的作用

当一个 worker 分到跨多个 shard 的连续 sample 时，`_split_shard` 会将其切成不超过 `max_samples_per_sequence` 的小段。这控制了：
- 单次顺序读取的最大长度（影响 prefetch 粒度）
- 断点恢复的最小粒度

```python
def _split_shard(start, end, max_samples_per_sequence):
    if end - start > max_samples_per_sequence * 1.5:
        slice_count = round((end - start) / max_samples_per_sequence)
        return tuple(start + int(i * (end-start)/slice_count) 
                     for i in range(slice_count))
    else:
        return (start,)  # 不切
```

## 7. 数据准备流水线（prepare.py）

### 7.1 设计动机（WHY）

原始 tar 文件可能来自外部系统（如 img2dataset），Energon 需要在训练前预处理：
1. 扫描每个 tar 生成 `.tar.idx` 偏移索引
2. 构建全局 `index.sqlite`（支持按 key 查询）
3. 统计每个 shard 的 sample 数量写入 `.info.yaml`
4. 检测数据异常（重复 key、损坏 tar）

### 7.2 核心数据结构（prepare.py:L62-100）

```python
@edataclass
class IndexSample(IndexAggregatable):
    tar_file_id: int         # tar 文件编号
    sample_key: str          # sample 唯一标识
    sample_index: int        # tar 内序号
    byte_offset: int         # tar 内字节偏移
    byte_size: int           # sample 总大小

@edataclass
class IndexSamplePart(IndexAggregatable):
    tar_file_id: int
    sample_index: int
    part_name: str           # 如 "jpg", "json"
    content_byte_offset: int # part 数据偏移
    content_byte_size: int

@edataclass
class IndexShardInfo(IndexAggregatable):
    shard_info: ShardInfo    # name, path, count
    parts: Set[str]          # 该 shard 包含的所有 part 类型
```

### 7.3 并行预处理架构

prepare.py 使用 AggregatorPool（聚合器池）实现并行索引构建：

```
                    ┌──────────────────────┐
                    │   AggregatorPool     │
                    │  (多线程/多进程)      │
                    └──────────┬───────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │Worker 0  │          │Worker 1  │          │Worker 2  │
  │_preprocess│          │_preprocess│          │_preprocess│
  │_tar()    │          │_tar()    │          │_tar()    │
  └────┬─────┘          └────┬─────┘          └────┬─────┘
       │                     │                     │
       ▼                     ▼                     ▼
  yield IndexSample     yield IndexSample     yield IndexSample
  yield IndexSamplePart yield IndexSamplePart yield IndexSamplePart
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             ▼
                    ┌──────────────────┐
                    │SqliteIndexWriter │
                    │  Aggregator      │
                    │(单线程写入SQLite) │
                    └──────────────────┘
```

### 7.4 _preprocess_tar 单 shard 处理

对每个 tar 文件：
1. 打开 tar 文件，创建 TarIndexWriter
2. 遍历 tar 中所有条目（header + data）
3. 按 split_name_re 解析 key 和 extension
4. 检测 sample 边界（key 变化 = 新 sample）
5. 记录每个 sample 的 byte_offset 到 .tar.idx
6. yield IndexSample + IndexSamplePart 给聚合器

### 7.5 错误处理

- **DuplicateSampleKeyError**（indexing.py:L22-27）：同一 key 出现两次→中止并报告
- 写入使用 SAVEPOINT 事务，单批失败不影响已写入数据
- TarIndexWriter 用 tmp + atomic rename，中断不会留下半写的 .tar.idx

## 8. 元数据系统（.nv-meta/）

### 8.1 WebdatasetInfo — shard 统计信息

（structs.py:L12-19）

```yaml
# .nv-meta/.info.yaml
energon_version: "2.0.0"
shard_counts:
  shard_000000: 10000
  shard_000001: 9876
  shard_000002: 10042
```

**用途**：Sharder 需要知道每个 shard 有多少 sample 才能做全局分配。

### 8.2 WebdatasetSplits — 数据集划分

（structs.py:L22-31）

```yaml
# .nv-meta/split.yaml
split_parts:
  train: [shard_000000, shard_000001, ..., shard_000099]
  val: [shard_000100, shard_000101, ..., shard_000109]
  test: [shard_000110, shard_000111]
exclude:
  - "shard_000003/42"    # 排除第3个shard的第42个sample
  - "shard_000007"       # 排除整个shard
```

**设计决策**：exclude 支持两种粒度（shard 级和 sample 级），在不修改 tar 文件的情况下过滤坏数据。

### 8.3 DatasetSubset — 子集选择

（structs.py:L61-115）

支持两种 subsetting 方式的组合：
1. `absolute_range: [start, end]` — 按绝对 sample 数截取
2. `range: [0.0, 0.8]` — 按比例截取

应用顺序：先 absolute_range，再在剩余范围内应用 range。

```python
def compute_subset(self, total_samples):
    start, end = 0, total_samples
    if self.absolute_range:           # 先绝对
        start, end = self.absolute_range
    if self.range:                    # 再相对
        previous_total = end - start
        end = start + int(previous_total * self.range[1])
        start += int(previous_total * self.range[0])
    return start, end
```

## 9. 设计决策对比表

| 维度 | Energon WebDataset | 原始 WebDataset (wds) | HuggingFace Datasets |
|------|-------------------|----------------------|---------------------|
| 存储格式 | tar (标准) | tar (标准) | Arrow/Parquet |
| 随机访问 | O(1) via .tar.idx | 不支持（顺序流） | O(1) via memory-map |
| 索引方式 | 二进制 + SQLite | 无索引 | Arrow 内置 |
| 分布式分配 | sample 粒度 + 位反转 | shard 粒度轮转 | 按行 range |
| 断点恢复 | 精确到 sample offset | 不支持 | 行号级 |
| 多模态 | 同 key 多文件天然支持 | 同 | 需特殊字段 |
| 元数据过滤 | SQLite query | 无 | Filter on columns |
| 大规模适配 | 百万 shard 级别 | 千级 shard | 内存受限 |
| 对 tar 的修改 | 不修改原始 tar | 不修改 | N/A |

## 10. 性能量化分析

### 10.1 索引大小开销

| 数据集规模 | sample 数 | .tar.idx 总大小 | index.sqlite 大小 |
|-----------|----------|----------------|-------------------|
| 10M samples / 1000 shards | 10,000,000 | 80 MB (8B × 10M) | ~400 MB |
| 100M samples / 10000 shards | 100,000,000 | 800 MB | ~4 GB |
| 1B samples | 1,000,000,000 | 8 GB | ~40 GB |

公式：`.tar.idx` 大小 = sample_count × 8 bytes

### 10.2 随机访问延迟

| 操作 | 延迟 |
|------|------|
| .tar.idx seek + read 8B | ~10 μs (SSD) / ~100 μs (NFS) |
| SQLite 按 key 查询 | ~50 μs (本地) / ~500 μs (NFS) |
| tar seek + read sample | ~100 μs (SSD) / ~1 ms (NFS) |
| 顺序扫描定位 (无索引) | O(n) × 512B header parse |

### 10.3 Sharder 分配均衡度

对于 total_samples=1,000,000, global_workers=128：
- min_per_worker = 7812
- remainder = 64 (50% workers 多 1 个 sample)
- 位反转确保多 1 的 worker 均匀分布在所有 rank
- 最大负载差异：1 sample (0.013%)

## 11. 配置建议与调优指南

### 11.1 数据准备最佳实践

| 配置项 | 推荐值 | 原因 |
|--------|--------|------|
| shard 大小 | 1-10 GB / 5000-50000 samples | 太小→索引开销大；太大→负载不均 |
| sample key | 唯一且可排序 | 避免 DuplicateSampleKeyError |
| part 类型 | 统一（所有 shard 相同的 parts） | 避免 field_access 报错 |

### 11.2 max_samples_per_sequence 选择

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| 大 sample (图片/视频) | 100-500 | 限制单次 IO burst |
| 小 sample (文本 token) | 1000-5000 | 顺序读高效 |
| 需要精细断点恢复 | 较小值 | 恢复粒度 = sequence 大小 |

### 11.3 常见问题

1. **index.sqlite 在 NFS 上很慢** → Energon 的 `ensure_local_copy` 会在训练开始前拷贝到本地 `/tmp`
2. **DuplicateSampleKeyError** → 检查 tar 打包时是否有同名文件
3. **shard_counts 不匹配** → 重新运行 `energon prepare` 重建索引
4. **exclude list 不生效** → 确认格式是 `"shard_name/sample_index"` 或 `"shard_name"`

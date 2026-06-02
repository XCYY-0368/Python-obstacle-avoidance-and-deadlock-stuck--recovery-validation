# 测试指南 (TESTING.md)

本项目的本地运行与回归测试流程。涵盖:依赖安装、单次运行、成对扫描、确定性自检、黄金基线回归、聚合指标判定、以及可视化(轨迹图 + 动画)。

---

## 一、依赖安装

仿真本体依赖 `numpy`、`scipy`(地图标注的连通性分析);可视化额外需要 `matplotlib`、`pillow`(生成 GIF)。

推荐用虚拟环境(WSL2 / Linux):

```bash
./run_tests.sh setup          # 自动建 .venv 并装齐四个依赖
source .venv/bin/activate
```

或手动:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pillow
```

退出虚拟环境用 `deactivate`。

---

## 二、单次运行(手动)

代码通过环境变量控制,四个旋钮:

| 变量 | 含义 | 取值 |
|------|------|------|
| `SEED` | 随机种子(决定地图 + 初始布局 + 目标序列) | 任意整数 |
| `MAX_STEPS` | 最大步数 | 通常 5000 |
| `COORD` | 是否开协调层 | `0`(baseline) / `1` |
| `STEER_MODE` | 跟随器 | `arch1`(主力) / `arch2`(预留,未充分验证) |

复现 seed 12:

```bash
# baseline:纯 ORCA
SEED=12 MAX_STEPS=5000 STEER_MODE=arch1 python3 container_sim.py

# COORD:开协调层
COORD=1 SEED=12 MAX_STEPS=5000 STEER_MODE=arch1 python3 container_sim.py
```

输出格式:

```
SEED=12  steps=5000  stop=max_steps
tasks_done per robot = [12, 9, 20]  total=41
bottlenecks(unique-path)=2  reroutable_narrow_gaps=3
```

关注:`steps`(跑了多少步)、`stop`(`max_steps`=满程 / `collision`=碰撞 / `all_locked`=全员死锁)、`total`(完成任务总数)。

> **确定性**:同一 `SEED` + 同样的环境变量,结果应**完全可复现**。跑出来对不上,先确认分支(`git branch`)和依赖版本。

---

## 三、自动化测试:`run_tests.py`

核心测试引擎。一条命令完成:成对扫描 + 聚合判定 + 黄金基线回归 + 确定性自检,并存档 CSV/JSON。

### 设计要点:为什么用子进程

`container_sim.py` 在 **import 时**读取 `COORD`(模块顶层 `COORD = os.environ.get(...)`)。因此每个配置(baseline / COORD)都在**全新的子进程**里跑——这是翻转 `COORD` 最稳妥的方式,保证两次运行之间没有任何状态泄漏。这也是为什么扫描结果比手动 `for` 循环里反复 import 更干净。

### 默认运行

```bash
python3 run_tests.py                 # 默认 12 个 seed,约 5 分钟
```

默认 seed:`7 12 23 31 42 88 100 256 512 777 999 1234`。每个 seed 跑 baseline + COORD 两次,约 24 秒/seed。

### 常用参数

```bash
python3 run_tests.py --seeds 1 2 3          # 自定义 seed
python3 run_tests.py --max-steps 5000       # 改步数上限
python3 run_tests.py --update-golden        # 把本次结果存为黄金基线
python3 run_tests.py --no-determinism       # 跳过确定性自检(省时间)
python3 run_tests.py --min-survival 0.6     # COORD 存活率下限(0~1)
python3 run_tests.py --min-mean-tasks 25    # COORD 平均任务数下限
```

---

## 四、四种判定方式(逐一说明)

测试跑完会给一个 `OVERALL: PASS / FAIL`。它由下面几条共同决定。

### 1. 成对对比(每个 seed 一行表格)

同一张图,baseline vs COORD 并排。`verdict` 列给出一句话结论:

- `COORD wins (survives)` —— baseline 失败但 COORD 满程(协调层救活了系统)
- `COORD worse (fails)` —— baseline 满程但 COORD 失败(回归,要警惕)
- `both survive (+N tasks)` —— 都满程,COORD 比 baseline 多/少 N 个任务
- `both fail` —— 都没撑住

**意义**:这是协调层价值的直接体现。只看 COORD 的数字没有参照系,成对才能看出"协调层到底带来了什么"。

### 2. 聚合指标阈值

整体两个数:**COORD 存活率**(满程 seed 占比)和 **COORD 平均任务数**。低于 `--min-survival` 或 `--min-mean-tasks` 下限即判 FAIL。

**意义**:防止"个别 seed 好看、整体崩了"。默认下限较宽松(存活率 50%、平均 20 任务),你可以按当前水平收紧。

### 3. 黄金基线回归(逐 seed)

把某个**已知良好**的版本结果存成 `results/golden.json`,以后每次改完和它逐 seed 对比:

```bash
# 第一次:在你认可的版本上建立黄金基线
python3 run_tests.py --update-golden

# 以后每次改完:
python3 run_tests.py            # 自动和 golden 对比
```

输出会标:

- `REGRESSION` —— 某 seed 从满程退化成碰撞/死锁(**最该警惕**)
- `IMPROVED` —— 某 seed 从失败变满程
- `task drop` / `task gain` —— 任务数明显增减(阈值 ±3)
- `stable` —— 没明显变化

只要出现 `REGRESSION` 或大幅 `task drop`,`OVERALL` 即 FAIL。

**意义**:这是最严格的回归检测,能立刻抓出"修了 A 却弄坏了 B"。强烈建议每次改动前后都跑。

### 4. 确定性自检

对指定 seed 连跑两次,断言结果完全一致(步数 + 停止原因 + 每车任务数)。任何不一致都判 FAIL。

**意义**:这个项目设计上是确定性的。若自检失败,说明意外引入了随机性(比如某处用了未播种的随机、或依赖了字典/集合的遍历顺序),是必须修的 bug。

---

## 五、输出与存档

每次跑完写入 `results/`(已在 `.gitignore` 中,不进版本库):

- `results/last_run.csv` —— 逐 seed 表格,Excel/pandas 可直接读
- `results/last_run.json` —— 完整结构化结果(含 aggregate)
- `results/golden.json` —— (用 `--update-golden` 时)黄金基线快照

> 如果你想把某次黄金基线**纳入版本库**做长期对照,可以手动 `git add -f results/golden.json`(强制,绕过 .gitignore)。

---

## 六、可视化(轨迹图 + 动画)

可视化由 `viz.py` 提供。输出目录默认是 `./outputs/`(可用 `VIZ_OUT` 环境变量改)。

### 通过 wrapper(推荐)

```bash
./run_tests.sh viz 12      # seed 12 的 baseline-vs-COORD 轨迹对比图 -> outputs/coord_compare_seed12.png
./run_tests.sh anim 12     # seed 12 的 COORD 动画 GIF(较慢)        -> outputs/anim_coord_seed12.gif
```

### 直接调 viz.py

```python
import viz

viz.single_traj(12)              # 单图轨迹(默认配置,看 COORD 取决于环境变量)
viz.coord_compare([12])          # baseline(上) vs COORD(下) 对比图(单 seed 已修好)
viz.coord_compare([12, 42, 88])  # 多 seed 并排
viz.coord_animation(12)          # COORD 动画 GIF
viz.make_animation(12)           # 通用动画(按当前环境变量决定是否 COORD)
viz.grid_figure([1, 42, 5, 88])  # 多 seed 概览网格
```

图例:圆点=起点,方块=终点(红边=碰撞),编号星=完成的任务点,橙色圈/叉=标注的协调瓶颈及等待区。

> **注意动画很慢、文件大**:一个 5000 步的 GIF 可能要几分钟、几 MB。建议只对你关心的单个 seed 按需生成,不要批量。`*.gif` 默认被 `.gitignore` 排除(仅保留示例 `seed12_watchdog.gif`)。

---

## 七、`run_tests.sh` 子命令速查

```bash
./run_tests.sh                 # 默认完整扫描(12 seed,~5 分钟)
./run_tests.sh quick           # 2-seed 冒烟测试(~45 秒)
./run_tests.sh sweep 1 2 3     # 扫描指定 seed
./run_tests.sh golden          # 跑默认扫描并存为黄金基线
./run_tests.sh viz 12          # 单 seed 轨迹对比图
./run_tests.sh anim 12         # 单 seed COORD 动画(慢)
./run_tests.sh setup           # 建 .venv 并装依赖
```

---

## 八、典型工作流

改代码前后做一次完整回归:

```bash
# 1. (一次性)在你认可的当前版本上建立黄金基线
git checkout develop
python3 run_tests.py --update-golden

# 2. 开新功能分支改代码
git checkout -b feature/xxx
# ... 改 container_sim.py / coordination.py ...

# 3. 跑测试,自动和黄金基线对比
python3 run_tests.py
#    - 看有没有 REGRESSION / task drop
#    - 看 OVERALL 是 PASS 还是 FAIL
#    - 确定性自检是否通过

# 4. 想看具体哪辆车怎么走的:
./run_tests.sh viz <出问题的seed>

# 5. 满意了再合并
git checkout develop && git merge --no-ff feature/xxx
```

---

## 九、常见问题

**结果和文档对不上?** 先 `git branch` 确认在哪个分支(`main` 是 #14 版、未含 #15;`develop` 含 #15),再确认依赖版本。

**`run_tests.py` 报某个 seed 超时?** 默认子进程超时 600 秒,正常 5000 步约 12 秒,远不会触发;若触发说明那个 seed 卡进了异常循环,值得单独 debug。

**动画生成卡很久?** 正常现象(逐帧渲染)。`make_animation` 内部做了抽帧(最多 ~400 帧),但仍可能要几分钟。耐心等或减小 `max_steps`。

**想在 Windows 原生跑而非 WSL2?** 可以,但注意换行符(`git config --global core.autocrlf input`),且 `run_tests.sh` 需要 bash(用 Git Bash 或直接调 `python3 run_tests.py`)。推荐 WSL2。

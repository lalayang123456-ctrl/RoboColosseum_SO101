# 在新机器上复现 SO101 的 Colosseum 评测环境

这份文档写给要在另一台机器上重建整套环境的人（或 AI 助手）。照着做完后，新机器可以：

1. 用 Aaron 的 Colosseum 流程在真实 SO-ARM101 上跑 Open track 评测（A/B 盲测、录制、打分、上传）；
2. 用独立脚本直接跑三个通用模型，不经过 Router。

所有命令默认在工作区根目录执行，下文记作 `$ROOT`（原机器上是 `/home/magiclab/RoboColosseum`）。
文档里的数字都是在原机器上实测的：Ubuntu 24.04，RTX 5090 Laptop（24 GB），2026-10-06。

---

## 0. 先读这一节：整体结构和容易踩的坑

### 三个角色

| 角色 | 在哪 | 做什么 |
| --- | --- | --- |
| Router | Aaron 托管，`wss://191.222.219.43` | 分配 A/B 两个模型，保存分数和录像 |
| Client | 本机，`colosseum-client/` | 读相机和关节、执行动作、录制、打分、上传 |
| Policy Server | 本机，`colosseum-policy-server/` | 按 Router 指定的模型拉起对应 worker，做推理 |

我们用 **Local 模式**：推理在本机 GPU 上跑，Router 只告诉 Client 用哪个模型。

### SO101 Open track 的三个模型（Router 上注册的）

| Router 名称 | HF 仓库 | revision | `max_horizon` | 显存 |
| --- | --- | --- | --- | --- |
| `molmoact2-so101` | `allenai/MolmoAct2-SO100_101` | `152569fe57914d97be91055800035f54e250d009` | 30 | 约 11 GB |
| `pi05-so101` | `hqfang/pi05-so100_101` | `d3204e03ae84d232e2493a933fd460515904eb58` | **50** | 约 16 GB |
| `g05-so101` | `OpenGalaxea/G05`，子目录 `g05-so101` | `e312be81e90c56a55bcb26b57429bd39a335b449` | 30 | 约 17 GB |

契约都是 `joint_position`、6 维、30 Hz。Router 每轮从三个里挑两个，**三个都必须就绪**。

### 五个 Python 环境

依赖互相冲突，不能合并。

| 环境 | 路径 | Python | 用途 |
| --- | --- | --- | --- |
| Client | `colosseum-client/.venv` | 3.12 | 机械臂驱动、评测流程、独立脚本 |
| Policy Server | `colosseum-policy-server/.venv` | 3.12 | 本地推理服务（不加载模型） |
| MolmoAct2 | `envs/.venv` | 3.12 | MolmoAct2 worker |
| pi0.5 | `envs/pi05_so101/.venv` | 3.12 | pi0.5 worker（需要打过补丁的 LeRobot） |
| G05 | `vendor/GalaxeaVLA/.venv` | 3.10 | G05 worker |

### 最容易出错的五件事

1. **本地配置必须和 Router 逐字段一致**（URL、revision、动作维度、频率、`max_horizon`）。差一个字段，Client 报 `INVALID_REQUEST`，Policy Server 日志里是 `unregistered model`。pi0.5 的 `max_horizon` 是 50，另外两个是 30。
2. **关节坐标系**。三个模型都是在旧版 LeRobot 的 SO100 坐标系下训练的，和现在的标定差 `模型值 = 符号 × 当前值 + 偏移`，符号 `[1,-1,1,1,1,1]`，偏移 `[0,90,90,0,0,0]`（度）。转换已经写在 Policy Server 的适配器里，**不要在别处再转一次**。
3. **标定是每条机械臂各自的。** 只有换的是同一条机械臂时才能拷贝标定文件，否则必须重新标定，关节限位也要重算。
4. **一次只能加载一个模型。** 换模型前要先停掉上一个。停进程用 `pkill -f "[c]olosseum-policy-local"` 这种带方括号的写法，否则 `pkill -f` 会匹配到自己所在的 shell。
5. **`OpenGalaxea/G05` 是受限仓库**，下载用的 HF 账号必须先在网页上同意条款。

---

## 1. 前置条件

| 项目 | 要求 |
| --- | --- |
| 系统 | Ubuntu 22.04 / 24.04 |
| GPU | NVIDIA，显存 ≥ 20 GB（G05 用 17 GB）；`nvidia-smi` 能正常显示 |
| 磁盘 | 约 90 GB：权重 49 GB，五个环境 32 GB，其余缓存 |
| 工具 | `git`、`uv`（装法：`curl -LsSf https://astral.sh/uv/install.sh | sh`） |
| 网络 | 能访问 GitHub、PyPI、`download.pytorch.org`、Hugging Face、Router |
| 账号 | Colosseum 的 Client token 和 institution；一个已获 `OpenGalaxea/G05` 访问权限的 HF token |
| 硬件 | SO-ARM101 从动臂（USB 串口）、一个顶部/外部相机、一个腕部相机 |

说明：

- RTX 50 系（Blackwell）必须用 `cu128` 的 torch，并且只能用 NVIDIA 的 open 内核驱动，见 `scripts/fix_nvidia_driver.sh`。下面的命令统一装 `cu128`，旧一些的显卡也能用。
- 串口权限：当前用户要在 `dialout` 组里，见 `scripts/fix_serial_permissions.sh`（需要 sudo）。
- 网络慢时给 `uv` 加 `export UV_HTTP_TIMEOUT=1200 UV_CONCURRENT_DOWNLOADS=3`。原机器上第一次装 torch 就因为超时失败过，重试即可，已下载的会走缓存。

---

## 2. 获取代码

### 2.1 工作区

```bash
git clone https://github.com/lalayang123456-ctrl/RoboColosseum_SO101.git RoboColosseum
cd RoboColosseum
export ROOT=$PWD
```

工作区里已经带了：`vendor/GalaxeaVLA`（G05 的源码）、`envs/frozen/`（各环境的精确依赖清单）、`scripts/`、`configs/calibration/`（原机械臂的标定）和本目录 `colosseum/`。

### 2.2 Aaron 的两个仓库加上 SO101 改动

SO101 的支持是我们加的，在 `so101-support` 分支上。先试方式 A，分支不存在再用方式 B，两者得到的代码完全相同。

**方式 A：分支已经推到 GitHub**

```bash
cd $ROOT
for r in colosseum-client colosseum-policy-server; do
  git clone https://github.com/frodobots-org/$r.git
  git -C $r checkout so101-support
done
```

如果 PR 已经合并进 `main`，直接用 `main`，不用切分支。

**方式 B：用工作区里带的补丁**

```bash
cd $ROOT
git clone https://github.com/frodobots-org/colosseum-client.git
git -C colosseum-client checkout -b so101-support dc609dee437de67161d81940dee2590e6bfec33b
git -C colosseum-client am $ROOT/colosseum/patches/colosseum-client-so101.patch

git clone https://github.com/frodobots-org/colosseum-policy-server.git
git -C colosseum-policy-server checkout -b so101-support 92863a4a993fbc62a9a1396307ad2aed4d1fa417
git -C colosseum-policy-server am $ROOT/colosseum/patches/colosseum-policy-server-so101.patch
```

`git am` 需要先配置过 `git config user.name` 和 `user.email`。补丁是针对上面两个基准提交生成的，已验证能干净应用。想基于更新的上游时，应用后再 `git rebase origin/main`。

验证：`ls colosseum-client/docs/so101.md colosseum-policy-server/docs/so101.md` 两个文件都存在。

---

## 3. 建立五个环境

### 3.1 Client

```bash
cd $ROOT/colosseum-client
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,so101]' 'msgpack>=1,<2' imageio-ffmpeg
```

`so101` 这个 extra 会装 LeRobot 和 torch，下载约 2 GB。`msgpack` 是独立脚本跑 G05 时用的。

**ffmpeg**：Client 录制时要在 `PATH` 里找到 `ffmpeg`。有 sudo 就 `sudo apt install ffmpeg`；没有就用刚装的静态版本：

```bash
ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg
```

用后一种方式时，启动 Client 必须用 `uv run --no-sync ...`（它会把 `.venv/bin` 放进 `PATH`）。

注意：**不要在这个目录里运行不带 `--no-sync` 的 `uv sync` 或 `uv run`**，它会按 `pyproject.toml` 重新同步并卸掉上面额外装的包。

验证：

```bash
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest -q
```

预期 277 个通过。如果有 2 个失败且报错是找不到 `ffprobe`，那是上游已有的测试需要系统版 ffmpeg，可以忽略。

### 3.2 Policy Server

```bash
cd $ROOT/colosseum-policy-server
uv sync --python 3.12 --extra droid --extra dev
.venv/bin/python -m pytest -q        # 预期 160 passed, 1 skipped
```

### 3.3 MolmoAct2 环境（`envs/.venv`）

```bash
cd $ROOT
uv venv --python 3.12 envs/.venv
uv pip install --python envs/.venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/.venv/bin/python -r envs/frozen/molmoact2-py312.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python envs/.venv/bin/python --no-deps -e colosseum-policy-server
envs/.venv/bin/python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

预期输出 `2.11.0+cu128 True 5.5.4`。这个环境同时也是工作区里其他 LeRobot 脚本（`scripts/run_*_checkpoint.sh`）用的环境。

### 3.4 pi0.5 环境（`envs/pi05_so101/.venv`）

这个 checkpoint 需要它自带补丁的特定版本 LeRobot，所以单独一个环境。**这一步要在第 4 节下载完 pi0.5 权重之后做**，因为补丁文件在权重仓库里。

```bash
cd $ROOT
git clone https://github.com/huggingface/lerobot.git vendor/lerobot-pi05-so100_101
git -C vendor/lerobot-pi05-so100_101 checkout b6ec0060779550c0a157ae34feb89e0cf86012a8
git -C vendor/lerobot-pi05-so100_101 apply $ROOT/checkpoints/pi05-so100_101/code/lerobot.patch

uv venv --python 3.12 envs/pi05_so101/.venv
uv pip install --python envs/pi05_so101/.venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/pi05_so101/.venv/bin/python -e 'vendor/lerobot-pi05-so100_101[pi]' \
  msgpack 'websockets>=15,<16'
uv pip install --python envs/pi05_so101/.venv/bin/python --no-deps -e colosseum-policy-server
envs/pi05_so101/.venv/bin/python -c "import torch, lerobot, transformers, sentencepiece; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

预期 `2.11.0+cu128 True 5.5.4`。精确依赖清单在 `envs/frozen/pi05-so101-py312.txt`，版本解析出了偏差时可以对照。

### 3.5 G05 环境（`vendor/GalaxeaVLA/.venv`）

```bash
cd $ROOT/vendor/GalaxeaVLA
uv sync                                   # Python 3.10，torch 2.7.1+cu128，约 11 GB
uv pip install --python .venv/bin/python --no-deps -e $ROOT/colosseum-policy-server
.venv/bin/python -c "import g05, torch; print(torch.__version__, torch.cuda.is_available())"
```

G05 的运行配置按**相对于源码根目录**的路径找 tokenizer 和 processor，这两样在第 4 节下载的权重里，下载完后建链接：

```bash
mkdir -p $ROOT/vendor/GalaxeaVLA/checkpoints
ln -sfn $ROOT/checkpoints/G05/action_tokenizer.pt          $ROOT/vendor/GalaxeaVLA/checkpoints/action_tokenizer.pt
ln -sfn $ROOT/checkpoints/G05/qwen3_5_2b_base_processor    $ROOT/vendor/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor
```

RTX 50 系上 flash-attention 的内核编译不过；我们的 worker 启动时默认把它换成 SDPA，不需要额外设置。

---

## 4. 下载模型权重

约 49 GB，可断点续传。

```bash
cd $ROOT
HF_TOKEN=hf_你的token colosseum-client/.venv/bin/python colosseum/tools/download_weights.py checkpoints
```

结束后应有：

```
checkpoints/MolmoAct2-SO100_101/model-0000{1..5}-of-00005.safetensors   (共约 21 GB)
checkpoints/pi05-so100_101/model.safetensors                            (16.6 GB)
checkpoints/pi05-so100_101/code/lerobot.patch
checkpoints/G05/g05-so101/checkpoints/model_state_dict.pt               (11.4 GB)
checkpoints/G05/action_tokenizer.pt
checkpoints/G05/qwen3_5_2b_base_processor/
```

报 `GatedRepoError 403`：token 对应的账号没有 G05 的访问权限。用那个账号登录 https://huggingface.co/OpenGalaxea/G05 同意条款后重跑。

下载完后回去做 3.4 节和 3.5 节里依赖权重的两步。

---

## 5. 硬件

### 5.1 找设备

```bash
ls -l /dev/serial/by-id/          # 机械臂串口，用 by-id 路径，不要用 ttyACM0
ls /dev/v4l/by-id/                # USB 相机，用 ...-video-index0 那个
$ROOT/colosseum-client/.venv/bin/lerobot-find-cameras realsense   # RealSense 的序列号
```

相机一律用稳定标识：RealSense 用序列号，USB 相机用 `/dev/v4l/by-id/...` 路径。`/dev/videoN` 的编号重启后会变，原机器上就因此把 RealSense 的红外流当成过腕部相机。

### 5.2 标定

- **同一条机械臂搬过来**：拷贝原标定。

  ```bash
  mkdir -p ~/.cache/huggingface/lerobot/calibration
  cp -r $ROOT/configs/calibration/robots ~/.cache/huggingface/lerobot/calibration/
  ```

- **另一条机械臂**：必须重新标定，标定 id 要和配置里的 `robot_id` 一致。

  ```bash
  $ROOT/colosseum-client/.venv/bin/lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/你的串口 --robot.id=follower_arm
  ```

驱动不会在评测中途触发交互式标定：未标定、或电机里的标定和文件不一致，都会在启动时直接报错。

### 5.3 关节限位

```bash
python3 $ROOT/colosseum/tools/joint_limits.py \
  ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
```

输出两行 `joint_low` / `joint_high`，下一节填进配置。原机械臂的值是 `±[108.0, 103.4, 97.3, 101.3, 180.0]`。

---

## 6. 配置文件

两个文件都只存在于本机，不进 git。

### 6.1 Client

```bash
cp $ROOT/colosseum/configs/robot.so101.yaml.template $ROOT/colosseum-client/configs/robot.so101.yaml
chmod 600 $ROOT/colosseum-client/configs/robot.so101.yaml
```

填写里面大写的占位项：`token`、`institution`、两个相机标识、串口、`joint_low` / `joint_high`。

相机角色是 Router 规定的：`head_image` 是顶部/外部相机，`left_image` 是腕部相机。`control_hz` 必须是 30，图像尺寸保持 640×480。

### 6.2 Policy Server

```bash
cd $ROOT
sed "s#__ROOT__#$ROOT#g" colosseum/configs/local-runtime-so101.yaml.template \
  > colosseum-policy-server/configs/local-runtime-so101.yaml
```

只要目录结构和本文一致，这个文件不用再改。

### 6.3 核对 Router

```bash
cd $ROOT
colosseum-client/.venv/bin/python colosseum/tools/router_check.py colosseum-client/configs/robot.so101.yaml
```

预期：`health: {'status': 'ok'}`，`token + institution: HTTP 200 (OK)`，并列出三个 Open track 模型。**把列出的 URL 和 revision 与 6.2 的配置对一遍**；如果 Aaron 更新过注册的 revision，要重新下载权重并改配置。`max_horizon` 这个接口不显示，对不上时按第 10 节的办法读。

---

## 7. 逐级验证

按顺序做，每一级通过了再做下一级。第 7.2 和 7.5 级会让机械臂上力矩，**开始前把机械臂摆到静止姿态，人在旁边**。

### 7.1 单元测试

第 3.1 和 3.2 节里已经跑过。

### 7.2 机械臂驱动（不涉及模型和 Router）

```bash
cd $ROOT
colosseum-client/.venv/bin/python colosseum/tools/hw_check.py \
  colosseum-client/configs/robot.so101.yaml outputs/colosseum/probe
```

它会上力矩、读状态和两路相机、原地保持、让 wrist_roll 动 2° 和夹爪动 5、回位、松力矩。检查：

- 打开约 3 秒，`get_observation` 约 1 ms；
- 夹爪和 wrist_roll 朝目标方向动了（小位移有舵机死区，差 0.5–1° 正常）；
- **打开 `outputs/colosseum/probe/` 里的两张 PNG**：`head_image` 是顶部视角、`left_image` 是腕部视角，颜色正常（不是灰度、红蓝没反）。

### 7.3 每个模型走一遍完整的本地协议（不涉及机械臂）

终端 1 启动 Policy Server：

```bash
cd $ROOT/colosseum-policy-server
.venv/bin/colosseum-policy-local --config configs/local-runtime-so101.yaml
```

终端 2，三个模型各跑一次（用的是 7.2 存下来的画面和姿态）：

```bash
cd $ROOT
for m in molmoact2-so101 pi05-so101 g05-so101; do
  colosseum-policy-server/.venv/bin/python colosseum/tools/runtime_probe.py \
    colosseum-policy-server/configs/local-runtime-so101.yaml $m outputs/colosseum/probe "Pick up the red cube."
done
```

原机器上的结果，供对照：

| 模型 | `ready` 耗时 | 每次推理 | 返回 | 第一步与当前姿态的最大偏差 |
| --- | --- | --- | --- | --- |
| molmoact2-so101 | 约 16 s | 约 0.3–0.4 s | (30, 6) | 约 13°（elbow，见下） |
| pi05-so101 | 约 61 s | 约 0.6 s | (50, 6) | 约 6° |
| g05-so101 | 约 25–30 s | 约 0.8 s，首次约 8.6 s | (30, 6) | 约 1° |

MolmoAct2 那 13° 是正常的：静止姿态的 elbow 略超出它训练数据的范围，它会先拉回范围内。**如果第一步偏差到了几十上百度（尤其是 shoulder_lift），说明坐标系转换没生效或被重复应用了。**

某个模型失败时，看 `outputs/colosseum/policy-logs/` 里最新的 `<模型名>-时间.log`。

### 7.4 开环正确性检查（推荐）

用训练分布里的公开数据，对比模型预测和人类示教的真实动作。这一步能发现图像通道、归一化、关节顺序、坐标系方面的错误。Policy Server 保持运行。

```bash
cd $ROOT
envs/.venv/bin/python colosseum/tools/fetch_reference_samples.py outputs/colosseum/ref
colosseum-policy-server/.venv/bin/python colosseum/tools/openloop_check.py \
  colosseum-policy-server/configs/local-runtime-so101.yaml outputs/colosseum/ref outputs/colosseum \
  molmoact2-so101 pi05-so101 g05-so101
```

看每个模型最后的 `ALL` 行。原机器上的结果（模型有采样随机性，上下浮动 0.5° 属正常）：

| 模型 | 预测误差 | 对照：原地不动的误差 |
| --- | --- | --- |
| molmoact2-so101 | 2.2–3.2° | 9.1° |
| pi05-so101 | 2.5–2.8° | 13.3° |
| g05-so101 | 3.6–4.1° | 9.1° |

预测误差应明显小于"原地不动"的误差。两者接近或预测误差更大，就是推理链路有问题。

### 7.5 第一次正式评测

终端 1 保持 Policy Server 运行。终端 2：

```bash
cd $ROOT/colosseum-client
uv run --no-sync colosseum-robot configs/robot.so101.yaml --no-execute-action
```

`--no-execute-action` 会正常读机械臂、请求推理、打印每段动作，但不发给电机。看打印的动作贴近当前姿态、逐步变化平缓，再去掉这个参数正式跑。

注意：带 `--no-execute-action` 的这一轮打完分也会作为真实评测提交。不想计入就在打分前按 Ctrl-C，下次启动会自动关掉这条未完成的任务。

---

## 8. 日常使用

### 8.1 正式评测

```bash
# 终端 1
cd $ROOT/colosseum-policy-server
.venv/bin/colosseum-policy-local --config configs/local-runtime-so101.yaml

# 终端 2
cd $ROOT/colosseum-client
uv run --no-sync colosseum-robot configs/robot.so101.yaml
```

流程：输入任务指令（完整英文句子，会原样发给模型）和场景描述（只做记录）→ 摆好场景按 Enter → 模型加载完后输入 `y` 确认 → A 跑完打 0–100 分 → 恢复场景跑 B、打分 → 选 A / B / tie → 自动提交和上传 → 问是否继续。

要点：

- 每个 trial 开始前把机械臂摆到静止姿态，结束时它会回到这个姿态再松力矩。
- trial 中按 Enter 提前结束；默认上限 800 步。
- 同一种布置的 `scene` 用同一个字符串，方便以后筛选。可以写进配置的 `scene` / `evaluator`，或用 `--scene "..."` 临时覆盖。

### 8.2 上传失败

看到 `Dataset upload deferred`：分数已经提交成功，只是录像没传完（通常是网络）。补传：

```bash
cd $ROOT/colosseum-client
PATH=$PWD/.venv/bin:$PATH .venv/bin/colosseum-upload-dataset configs/robot.so101.yaml --evaluation-id ev_编号
```

不带 `--evaluation-id` 则补传 `eval_runs/` 里所有没传的。看到 `Uploaded and verified N LeRobot datasets.` 才算真的传完。

### 8.3 独立脚本（不经过 Router，不录制不打分）

先停掉 Policy Server（显存放不下两个模型）。

```bash
cd $ROOT
colosseum-client/.venv/bin/python scripts/run_open_model.py --model pi05          # 或 molmoact2 / g05
colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py --model pi05   # 无停顿版本
```

提示 `Task>` 时输入指令回车开始，再按 Enter 提前停，空指令或 `q` 退出。常用参数：`--dry-run`（只打印不执行）、`--duration 60`。

两个脚本的区别：

- `run_open_model.py` 和正式评测一样是同步推理，每执行完一段动作会停下来等下一段。
- `run_open_model_smooth.py` 在后台提前请求下一段，MolmoAct2 和 pi0.5 可以全程不停。G05 的推理耗时超过半段动作，做不到，脚本会自动退回同步方式并打印说明。

---

## 9. 已知现象（不是配置错误）

| 现象 | 说明 |
| --- | --- |
| 正式评测里动作一顿一顿 | Client 是同步推理，每段动作之间机械臂停着等。实测平均频率 MolmoAct2 约 22 Hz、G05 约 14 Hz。这是 Aaron 公共代码的设计 |
| 模型在真机上几乎不动、得 0 分 | 原机器的场景（正上方俯视相机 + 腕部相机，桌上一个红色方块）下，MolmoAct2 和 G05 都停在静止姿态附近。对照实验表明原因在画面：换成训练集的画面、配上我们机械臂的状态，模型就会动。推理链路经 7.4 验证是正确的 |
| G05 每轮刚开始很慢 | 它是慢启动的模型，第一次推理还有约 8 秒预热 |
| 回放视频比实际短 | 导出的数据集按名义 30 Hz 排帧，停顿期间没有帧；真实时间在 `observation.capture_timestamp` |

---

## 10. 排错

| 症状 | 原因和处理 |
| --- | --- |
| Client 报 `INVALID_REQUEST: Invalid local policy request`，Policy Server 日志有 `unregistered model` | 本地配置和 Router 发来的模型规格不一致。读 Router 的实际规格：Client 在 `eval_runs/<评测>/<run>/policy.json` 里存了已成功那一侧的；或者对照 `router_check.py` 的输出。改 `local-runtime-so101.yaml` 后**重启 Policy Server** |
| `MODEL_START_FAILED` | worker 没起来。看 `outputs/colosseum/policy-logs/` 最新日志。常见：权重路径不对、对应环境没装好、显存被别的进程占着 |
| `Local Policy evaluation currently supports robot_type: franka or yam` | Client 没有 SO101 改动，回第 2.2 节 |
| `SO101 hardware control is not implemented` | 同上 |
| `Install colosseum-client[so101] for LeRobot` | Client 环境没装 `so101` extra |
| `Install ffmpeg before starting a trial` | `PATH` 里没有 ffmpeg，见 3.1 节；用静态版本时要用 `uv run --no-sync` 启动 |
| `SO101 'follower_arm' is not calibrated` | 见 5.2 节 |
| 打开串口 `Permission denied` | 用户不在 `dialout` 组，见 `scripts/fix_serial_permissions.sh` |
| 腕部画面是灰度图或视角不对 | 相机标识指到了别的设备（比如 RealSense 的红外流），改用 `/dev/v4l/by-id/` 路径 |
| RealSense 打不开 | 确认没有别的进程占用；Client 环境里的 `pyrealsense2` 是 2.56.5（LeRobot 限制在 2.57 以下），原机器上工作正常 |
| CUDA out of memory | 有另一个模型或 Policy Server 还占着显存，`nvidia-smi` 看一下，停掉再来 |
| 第一步动作偏离当前姿态几十上百度 | 坐标系转换问题，见第 0 节第 2 点。检查配置里有没有误加 `joint_frame: current` |
| `uv` 下载超时 | `export UV_HTTP_TIMEOUT=1200 UV_CONCURRENT_DOWNLOADS=3` 后重试 |
| Client 测试有 2 个因 `ffprobe` 失败 | 上游已有测试需要系统版 ffmpeg，可忽略 |

**读取 Router 给某个模型注册的完整规格（含 `max_horizon`）**：`router_check.py` 用的公开接口不显示 `max_horizon`。Client 每跑成功一侧，就会把 Router 发来的完整规格存进 `eval_runs/<评测>/<run>/policy.json`，可以直接对照。

**只想验证连通性、不想留正式记录**：把 Client 配置里的 `test` 设成 `true` 再启动。这时用的是假机械臂和假画面，Router 上的记录带 Test 标记、不计入排名。验证完改回 `false`。

---

## 11. 哪些东西是机器专属的

换机器时这些必须重新确认，不能照搬：

- `colosseum-client/configs/robot.so101.yaml`：token、串口、相机标识、关节限位；
- `colosseum-policy-server/configs/local-runtime-so101.yaml`：全是绝对路径，用 6.2 节的命令重新生成；
- 机械臂标定文件（`~/.cache/huggingface/lerobot/calibration/`）；
- 各环境本身：不要拷贝 `.venv` 目录，里面写死了绝对路径，要按第 3 节重建。

不要提交进任何仓库的：`.env`、`command.md`、`colosseum-client/configs/robot.so101.yaml`（都含 token 或密码）。

---

## 12. 本目录的内容

```
colosseum/
├── README.md                              本文档
├── patches/
│   ├── colosseum-client-so101.patch           基于 dc609de
│   └── colosseum-policy-server-so101.patch    基于 92863a4
├── configs/
│   ├── robot.so101.yaml.template              Client 配置模板
│   └── local-runtime-so101.yaml.template      Policy Server 配置模板
└── tools/
    ├── download_weights.py                    下载三个模型
    ├── joint_limits.py                        从标定文件算关节限位
    ├── router_check.py                        Router 连通性和注册模型（只读）
    ├── hw_check.py                            机械臂驱动自检
    ├── runtime_probe.py                       单个模型走完整本地协议
    ├── fetch_reference_samples.py             下载开环检查用的参考样本
    └── openloop_check.py                      预测动作对比真实动作
```

相关文档：`colosseum-client/docs/so101.md`（驱动的契约和行为）、`colosseum-policy-server/docs/so101.md`（三个模型的 worker 和适配器）。

### 原机器上验证过什么

- 两个仓库的单元测试；三个模型各自的完整本地协议和开环检查（第 7.3、7.4 节的数字）；
- 机械臂驱动的真机自检；
- 一次完整的 Open track 正式评测（MolmoAct2 对 G05），包括提交和上传。

### 没有验证过的

- 本文档的步骤**没有在第二台机器上从头走过**。各步的命令是按原机器上实际执行过的整理的，但第 3.3 节（用依赖清单重建 MolmoAct2 环境）和第 3.5 节（`uv sync` 重建 G05 环境）在原机器上是更早以前用别的方式建的，这两步最可能需要调整；
- pi0.5 还没有在正式评测里被抽到过，只做过协议和开环验证；
- 合并上游最新代码之后，还没有重新跑过正式评测；
- 非 RTX 50 系显卡、其他型号的相机。

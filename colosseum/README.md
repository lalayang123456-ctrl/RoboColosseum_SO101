# 在新机器上复现 SO101 的 Colosseum 评测环境（5 个 VLA + 3 个 LLM）

这份文档写给要在另一台机器上从零重建整套环境的人（或 AI 助手）。照着做完，新机器上这两条命令可以直接跑通，Router 抽到 8 个模型里的任意一个都能正常执行：

```bash
# 终端 1：Policy Server
cd $ROOT/colosseum-policy-server
uv run --no-sync colosseum-policy-local --config configs/local-runtime-so101-mixed-joint.yaml

# 终端 2：Client
export OPENAI_API_KEY=sk-...      # GPT 用
export XAI_API_KEY=xai-...        # Grok 用
cd $ROOT/colosseum-client
uv run --no-sync colosseum-robot configs/robot.so101.joint.yaml
```

下文把工作区根目录记作 `$ROOT`（原机器上是 `/home/magiclab/RoboColosseum`）。所有命令都默认 `$ROOT` 已经 export。
文档里的实测数字来自原机器：Ubuntu 24.04，RTX 5090 Laptop（24 GB 显存），30 GB 内存，2026-10-21。

---

## 0. 先读这一节：整体结构和容易踩的坑

### 三个角色

| 角色 | 在哪 | 做什么 |
| --- | --- | --- |
| Router | Colosseum 托管，`wss://191.222.219.43` | 每轮分配 A/B 两个模型，保存分数和录像 |
| Client | 本机，`colosseum-client/` | 读相机和关节、执行动作、录制、打分、上传；把 LLM 的 API key 转给 Policy Server |
| Policy Server | 本机，`colosseum-policy-server/` | 按 Router 指定的模型拉起对应 worker（VLA）或在进程内跑 LLM agent |

用的是 **Local 模式**：推理在本机，Router 只告诉 Client 用哪个模型。

### 8 个模型（必须和 Router 注册的逐字段一致）

| # | Router 名称 | 类型 | 来源 | revision | `max_horizon` | `control_hz` | 运行位置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `molmoact2-so101` | VLA | `allenai/MolmoAct2-SO100_101` | `152569fe…` | 30 | 30 | `envs/.venv`，约 11 GB 显存 |
| 2 | `pi05-so101` | VLA | `hqfang/pi05-so100_101` | `d3204e03…` | 50 | 30 | `envs/pi05_so101/.venv`，约 16 GB |
| 3 | `g05-so101` | VLA | `OpenGalaxea/G05`（受限仓库） | `e312be81…` | 30 | 30 | `vendor/GalaxeaVLA/.venv`，约 17 GB |
| 4 | `groot-n17-so101` | VLA | `hqfang/groot-so100_101` + 基座 `nvidia/GR00T-N1.7-3B` | `f5e17af8…` | 40 | 30 | `envs/.venv`，约 14 GB |
| 5 | `lingbot-molmoact2-so101` | VLA | `RoboColosseum/LingBot-MolmoAct2-So101` + `Qwen/Qwen3-VL-4B-Instruct` 的 processor | `d360710b…` | 50 | 30 | `envs/.venv-lingbot`，约 12 GB 显存、峰值约 21 GB 内存 |
| 6 | `grok-4.7` | LLM | xAI 官方 API | `4.7` | 150 | 10 | Policy Server 进程内，需要 `XAI_API_KEY` |
| 7 | `gpt-6-astra` | LLM | OpenAI 官方 API | `rig-v1` | 150 | 10 | Policy Server 进程内，需要 `OPENAI_API_KEY` |
| 8 | `claude-opus-5-5` | LLM | Claude Code 账号登录 | `rig-v1` | 150 | 30 | Policy Server 进程内，需要本机 Claude CLI 已登录 |

完整 revision 见 [configs/local-runtime-so101-mixed-joint.yaml.template](configs/local-runtime-so101-mixed-joint.yaml.template)。

- 契约都是 `joint_position`、6 维。LLM 也用关节控制（`control_mode: joint_pos`），Client 里对应 `llm_control_mode: joint_pos`。
- 三个 LLM 在 Router 上登记的 `url` 是中转地址 `api.yhlxj.ai`，这只是身份标识；实际请求由 `api_base_url` 直接发到官方 API。
- Router 每轮从 8 个里挑两个，**8 个都必须就绪**，任何一个缺环境或缺权重，抽到它那一轮就会报 `MODEL_START_FAILED`。

### 七个 Python 环境 + 一个 CLI

依赖互相冲突，不能合并。**不要拷贝 `.venv` 目录**，里面写死了绝对路径，要在新机器上重建。

| 环境 | 路径 | Python | 用途 | 建法 |
| --- | --- | --- | --- | --- |
| Client | `colosseum-client/.venv` | 3.12 | 机械臂驱动、评测流程 | 3.1 |
| Policy Server | `colosseum-policy-server/.venv` | 3.12 | 本地推理服务 + 3 个 LLM agent（inspect-robots） | 3.2 |
| MolmoAct2 / GR00T | `envs/.venv` | 3.12 | MolmoAct2 和 GR00T 的 worker | 3.3 |
| pi0.5 | `envs/pi05_so101/.venv` | 3.12 | pi0.5 worker（打过补丁的 LeRobot） | 3.4 |
| G05 | `vendor/GalaxeaVLA/.venv` | 3.10 | G05 worker | 3.5 |
| LingBot | `envs/.venv-lingbot` | 3.12 | LingBot worker（torch 2.8） | 3.6 |
| Claude CLI | `envs/claude-cli` | — | Claude Code 二进制，Claude 模型通过它调用 | 3.7 |

### 最容易出错的几件事

1. **本地配置必须和 Router 逐字段一致**（name、URL、revision、动作维度、频率、`max_horizon`）。差一个字段，Client 报 `INVALID_REQUEST`，Policy Server 日志里是 `unregistered model`。
2. **关节坐标系。** 4 个通用 VLA 是在旧版 LeRobot 的 SO100 坐标系下训练的，转换已经写在 Policy Server 的适配器里，**不要在别处再转一次**。
3. **标定是每条机械臂各自的。** 只有搬的是同一条机械臂才能拷贝标定文件，否则必须重新标定，关节限位也要重算，而且两个配置文件里的限位都要改（Client 的 `adapter_config` 和 Policy Server 的 `backend.options.llm`）。
4. **一次只能加载一个 VLA。** Policy Server 会自动切换；但如果之前手动跑过别的脚本没退出，显存会不够。清显存见第 8.3 节。
5. **`OpenGalaxea/G05` 是受限仓库**，下载用的 HF 账号必须先在网页上同意条款。
6. **LingBot 的权重仓库现在在 Hub 上不公开了**（2026-10-21 用两个 HF 账号都查不到），新机器多半要从原机器拷贝，见第 4 节。
7. **不要在两个子仓库里运行不带 `--no-sync` 的 `uv run` / `uv sync`**，它会按 lock 重新同步，卸掉额外装的包。

---

## 1. 前置条件

| 项目 | 要求 |
| --- | --- |
| 系统 | Ubuntu 22.04 / 24.04，x86_64 |
| GPU | NVIDIA，显存 ≥ 20 GB（G05 用 17 GB）；`nvidia-smi` 能正常显示 |
| 内存 | ≥ 32 GB（LingBot 加载时峰值约 21 GB；原机器 30 GB 能跑） |
| 磁盘 | 约 170 GB：权重约 95 GB，七个环境约 55 GB，其余是缓存 |
| 工具 | `git`、`uv`（`curl -LsSf https://astral.sh/uv/install.sh \| sh`） |
| 网络 | 能访问 GitHub、PyPI、`download.pytorch.org`、Hugging Face、Router、`api.openai.com`、`api.x.ai`、`api.anthropic.com` |
| 账号 | Colosseum 的 Client token 和 institution；一个已获 `OpenGalaxea/G05` 访问权限的 HF token；OpenAI 和 xAI 的 API key；一个能用 Claude Code 的 Claude 账号 |
| 硬件 | SO-ARM101 从动臂（USB 串口）、一个前置/顶部相机（原机器是 RealSense D435）、一个腕部 USB 相机 |

说明：

- RTX 50 系（Blackwell）必须用 `cu128` 的 torch，并且只能用 NVIDIA 的 open 内核驱动，见 `scripts/fix_nvidia_driver.sh`。下面的命令统一装 `cu128`，旧一些的显卡也能用。
- 串口权限：当前用户要在 `dialout` 组里，见 `scripts/fix_serial_permissions.sh`（需要 sudo）。
- 网络慢时先 `export UV_HTTP_TIMEOUT=1200 UV_CONCURRENT_DOWNLOADS=3`。第一次装 torch 超时失败时重试即可，已下载的会走缓存。

---

## 2. 获取代码

### 2.1 工作区

```bash
git clone https://github.com/lalayang123456-ctrl/RoboColosseum_SO101.git RoboColosseum
cd RoboColosseum
export ROOT=$PWD
```

工作区里已经带了：

- `vendor/GalaxeaVLA`（G05 源码）和 `vendor/lingbot-vla-v2`（LingBot 源码）；
- `envs/frozen/`（各环境的精确依赖清单）和 `envs/pyheaders/`（LingBot 编译 Triton 内核用的 Python 3.12 头文件，新机器不用装 `python3.12-dev`）；
- `configs/calibration/`（原机械臂的标定）；
- 本目录 `colosseum/`：配置模板、两个子仓库的补丁、检查工具。

### 2.2 Client 和 Policy Server 两个子仓库

这两个是上游 `frodobots-org` 的仓库，SO101 的全部改动（含 LingBot worker、LLM 关节控制）在我们的 `so101-support` 分支上。两个目录在主仓库的 `.gitignore` 里，要单独获取。

**方式 A（推荐，只依赖主仓库）：上游代码 + 工作区里的补丁**

```bash
cd $ROOT
git clone https://github.com/frodobots-org/colosseum-client.git
git -C colosseum-client checkout -b so101-support c37afee3fb345cda0cc669b6566ba700895460f0
git -C colosseum-client apply $ROOT/colosseum/patches/colosseum-client-so101.patch

git clone https://github.com/frodobots-org/colosseum-policy-server.git
git -C colosseum-policy-server checkout -b so101-support 698b75d168da7b62073d6bec70608bb70e18160f
git -C colosseum-policy-server apply $ROOT/colosseum/patches/colosseum-policy-server-so101.patch
```

两个基准提交是上游 `main` 在 2026-10-19 的最新提交（合并了 `shailes-h` 的 PR #2），`so101-support` 已经合并了它们。补丁是 `git diff <基准> so101-support` 生成的。原机器上已经验证过：在全新 clone 上应用后，文件树和 `so101-support` 的 HEAD 完全一致。应用后想留一个提交，就在两个目录里各跑一次 `git add -A && git commit -m "SO101 support"`。

**方式 B：直接用我们的 fork**（前提是 `so101-support` 分支已经推到 fork 上）

```bash
cd $ROOT
for r in colosseum-client colosseum-policy-server; do
  git clone -b so101-support https://github.com/lalayang123456-ctrl/$r.git
  git -C $r remote add upstream https://github.com/frodobots-org/$r.git
done
```

验证：`ls colosseum-policy-server/src/colosseum_policy_server/so101_lingbot.py colosseum-client/src/colosseum_client/robots/so101.py`，两个文件都存在。

---

## 3. 建立环境

每一步最后都有一条验证命令，输出对得上再做下一步。

### 3.1 Client（`colosseum-client/.venv`）

```bash
cd $ROOT/colosseum-client
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,so101-eef]' 'msgpack>=1,<2' imageio-ffmpeg
```

`so101-eef` 包含 `so101` 的全部内容（LeRobot、Feetech、RealSense），另外加了 EEF 控制用的 `placo`；只跑关节控制时用 `so101` 也行。

**ffmpeg**：Client 录制时要在 `PATH` 里找到 `ffmpeg`。有 sudo 就 `sudo apt install ffmpeg`；没有就用刚装的静态版本：

```bash
ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg
```

所以 Client 一律用 `uv run --no-sync ...` 启动，它会把 `.venv/bin` 加进 `PATH`。

验证：

```bash
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest -q      # 原机器：347 passed, 1 skipped
```

### 3.2 Policy Server（`colosseum-policy-server/.venv`）

```bash
cd $ROOT/colosseum-policy-server
uv sync --locked --python 3.12 --extra agent --extra droid --extra dev
.venv/bin/python -m pytest -q          # 原机器：307 passed, 1 skipped
.venv/bin/python -c "import inspect_robots, inspect_robots_agent; print('agent ok')"
```

`agent` 这个 extra 装的是 3 个 LLM 要用的 inspect-robots，锁定在 `supernoooo/inspect-robots@22a4da6`（公开仓库）。**少了 `--extra agent`，3 个 LLM 都起不来。**

### 3.3 MolmoAct2 / GR00T 环境（`envs/.venv`）

```bash
cd $ROOT
uv venv --python 3.12 envs/.venv
uv pip install --python envs/.venv/bin/python -r envs/frozen/molmoact2-py312.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python envs/.venv/bin/python --no-deps -e colosseum-policy-server
envs/.venv/bin/python -c "import torch, transformers, lerobot; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

预期输出 `2.11.0+cu128 True 5.5.4`。依赖清单在原机器上用空环境 `--dry-run` 解析过，能完整解出全部 114 个包。这个环境同时也是 `scripts/run_*_checkpoint.sh` 用的环境。

### 3.4 pi0.5 环境（`envs/pi05_so101/.venv`）

这个 checkpoint 需要它自带补丁的特定版本 LeRobot。**这一步要在第 4 节下载完 pi0.5 权重之后做**，因为补丁文件在权重仓库里。

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

预期 `2.11.0+cu128 True 5.5.4`。版本解析出了偏差时，对照 `envs/frozen/pi05-so101-py312.txt`。

### 3.5 G05 环境（`vendor/GalaxeaVLA/.venv`）

```bash
cd $ROOT/vendor/GalaxeaVLA
uv sync --locked                          # Python 3.10，torch 2.7.1+cu128，约 11 GB
uv pip install --python .venv/bin/python --no-deps -e $ROOT/colosseum-policy-server
.venv/bin/python -c "import g05, torch; print(torch.__version__, torch.cuda.is_available())"
```

G05 的运行配置按**相对于源码根目录**的路径找 tokenizer 和 processor，这两样在第 4 节下载的权重里，下载完后建链接：

```bash
mkdir -p $ROOT/vendor/GalaxeaVLA/checkpoints
ln -sfn $ROOT/checkpoints/G05/action_tokenizer.pt          $ROOT/vendor/GalaxeaVLA/checkpoints/action_tokenizer.pt
ln -sfn $ROOT/checkpoints/G05/qwen3_5_2b_base_processor    $ROOT/vendor/GalaxeaVLA/checkpoints/qwen3_5_2b_base_processor
```

RTX 50 系上 flash-attention 的内核编译不过，worker 启动时默认改用 SDPA，不需要额外设置。

### 3.6 LingBot 环境（`envs/.venv-lingbot`）

```bash
cd $ROOT
uv venv --python 3.12 envs/.venv-lingbot
uv pip install --python envs/.venv-lingbot/bin/python --no-deps -r envs/frozen/lingbot-py312.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python envs/.venv-lingbot/bin/python --no-deps -e vendor/lingbot-vla-v2 -e colosseum-policy-server
envs/.venv-lingbot/bin/python -c "import torch, transformers, lingbotvla; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
```

预期 `2.8.0+cu128 True 4.57.3`。

**这里必须带 `--no-deps`**：清单里的 `lerobot==0.4.2` 声明要 `torchvision<0.23`，而 LingBot 实际要 0.23，正常解析会直接报 `No solution found`。这份清单是原环境的完整 freeze，`--no-deps` 能逐个包精确复现。原机器上已核对：`pip freeze` 和清单一致，`--dry-run` 能装上全部 131 个包。

worker 启动时设置了 `LINGBOT_ATTN_IMPL=sdpa`（没装 flash-attn），以及指向 `envs/pyheaders` 的 `C_INCLUDE_PATH`（Triton 编译时要用 `Python.h`），这些都写在配置模板的 launcher 里，不用手动设。

### 3.7 Claude CLI（`envs/claude-cli`）

`claude-opus-5-5` 用 Claude Code 账号登录来调用，Policy Server 需要找到 `claude` 可执行文件。这里装 `claude-agent-sdk` 里自带的那个，版本和原机器一致：

```bash
cd $ROOT
uv pip install --python 3.12 --target envs/claude-cli claude-agent-sdk==0.2.160
envs/claude-cli/claude_agent_sdk/_bundled/claude --version        # 原机器：2.1.283 (Claude Code)
envs/claude-cli/claude_agent_sdk/_bundled/claude auth status
```

`auth status` 显示 `"loggedIn": false` 时再登录（会打开浏览器，或者给一个链接让你粘贴回授权码）：

```bash
envs/claude-cli/claude_agent_sdk/_bundled/claude auth login
```

- 用**运行 Policy Server 的同一个 Linux 用户**登录。登录信息存在 `~/.claude`，和二进制放在哪里无关；本机装过别的 Claude Code，也是共用同一份登录。
- 不要同时设置 `ANTHROPIC_API_KEY`；Policy Server 在账号模式下会把它从子进程环境里剔掉，但设了容易混淆。

---

## 4. 下载模型权重

约 95 GB，可以断点续传，重复运行会接着下。

```bash
cd $ROOT
HF_TOKEN=hf_你的token colosseum-client/.venv/bin/python colosseum/tools/download_weights.py checkpoints
```

结束后应有：

```
checkpoints/MolmoAct2-SO100_101/model-0000{1..5}-of-00005.safetensors   (约 21 GB)
checkpoints/pi05-so100_101/model.safetensors                            (16.6 GB)
checkpoints/pi05-so100_101/code/lerobot.patch
checkpoints/G05/g05-so101/checkpoints/model_state_dict.pt               (11.4 GB)
checkpoints/G05/action_tokenizer.pt
checkpoints/G05/qwen3_5_2b_base_processor/
checkpoints/groot-so100_101/model.safetensors                           (11.7 GB)
checkpoints/GR00T-N1.7-3B/                                              (6.5 GB，GR00T 的基座)
checkpoints/LingBot-MolmoAct2-So101/model-0000{1..6}-of-00006.safetensors (约 24 GB)
checkpoints/Qwen3-VL-4B-Instruct/                                       (12 MB，只有 config/tokenizer/processor)
```

- 报 `GatedRepoError 403`：token 对应的账号没有 G05 的访问权限。用那个账号登录 https://huggingface.co/OpenGalaxea/G05 同意条款后重跑。
- 脚本最后打印 `missing: LingBot-MolmoAct2-So101`：这个仓库已经不公开了，其他模型照常下载。**从原机器拷贝**（约 24 GB）：

  ```bash
  rsync -a --info=progress2 magiclab@原机器IP:/home/magiclab/RoboColosseum/checkpoints/LingBot-MolmoAct2-So101 $ROOT/checkpoints/
  ```

  也可以用移动硬盘拷整个目录。拷完检查 `checkpoints/LingBot-MolmoAct2-So101/finetune_assets/` 和 `lingbotvla_cli.yaml` 都在；worker 启动时会校验这些文件。

下载完后回去做 3.4 节和 3.5 节里依赖权重的那两步。

---

## 5. 硬件

### 5.1 找设备

```bash
ls -l /dev/serial/by-id/          # 机械臂串口，用 by-id 路径，不要用 ttyACM0
ls /dev/v4l/by-id/                # USB 相机，用 ...-video-index0 那个
$ROOT/colosseum-client/.venv/bin/lerobot-find-cameras realsense   # RealSense 的序列号
```

相机一律用稳定标识：RealSense 用序列号，USB 相机用 `/dev/v4l/by-id/...` 路径。`/dev/videoN` 的编号重启后会变，原机器上就因此把 RealSense 的红外流当成过腕部相机。

原机器的值供对照：从动臂 `usb-1a86_USB_Single_Serial_5B79017804-if00`，前置 RealSense `262522073381`，腕部 `usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0`。

### 5.2 标定

Client 配置里的 `robot_id` 是 `follower_arm_open`，所以标定文件名必须是 `follower_arm_open.json`。

- **同一条机械臂搬过来**：拷贝原标定（两个文件内容相同，`scripts/` 里的工具用 `follower_arm` 这个 id）。

  ```bash
  mkdir -p ~/.cache/huggingface/lerobot/calibration
  cp -r $ROOT/configs/calibration/robots $ROOT/configs/calibration/teleoperators ~/.cache/huggingface/lerobot/calibration/
  ```

- **另一条机械臂**：必须重新标定，然后复制一份给另一个 id。

  ```bash
  $ROOT/colosseum-client/.venv/bin/lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/你的串口 --robot.id=follower_arm_open
  cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm_open.json \
     ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm.json
  ```

驱动不会在评测中途触发交互式标定：未标定，或者电机里存的标定和文件不一致，都会在启动时直接报 `not calibrated`。

### 5.3 关节限位

```bash
python3 $ROOT/colosseum/tools/joint_limits.py \
  ~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_arm_open.json
```

输出 `joint_low` / `joint_high` 两行，第 6 节里**两个配置文件都要填**。当前这条机械臂的值是 `±[119.2, 103.9, 97.2, 101.1, 180.0]`。

---

## 6. 配置文件

两个文件都只在本机，不进 git（子仓库的 `.gitignore` 已经排除了 `configs/*.yaml`）。

### 6.1 Policy Server：`local-runtime-so101-mixed-joint.yaml`

```bash
cd $ROOT
sed -e "s#__ROOT__#$ROOT#g" colosseum/configs/local-runtime-so101-mixed-joint.yaml.template \
  > colosseum-policy-server/configs/local-runtime-so101-mixed-joint.yaml
```

然后打开生成的文件，把 `backend.options.llm` 下的 `joint_low: [J1, ...]` / `joint_high: [J1, ...]` 换成 5.3 的输出。除此之外，只要目录结构和本文一致，就不用再改。

原机器上核对过：用模板生成的文件和原机器正在用的配置相比，只有两处路径不同，都是有意的改动。Claude CLI 从 `~/inspect-robots/.so101/claude-cli/...` 改到了 `envs/claude-cli/...`，Qwen processor 从 HF 缓存改到了 `checkpoints/Qwen3-VL-4B-Instruct`。

### 6.2 Client：`robot.so101.joint.yaml`

```bash
cp $ROOT/colosseum/configs/robot.so101.joint.yaml.template $ROOT/colosseum-client/configs/robot.so101.joint.yaml
chmod 600 $ROOT/colosseum-client/configs/robot.so101.joint.yaml
```

填写里面大写的占位项：`token`、`institution`、两个相机标识、串口、`joint_low` / `joint_high`。

- 相机角色是 Router 规定的：`head_image` 是前置/顶部相机，`left_image` 是腕部相机。
- `control_hz` 保持 30，图像保持 640×480。
- `deadline_ms: 180000` 不要删：LLM 带图规划一次要 15–40 秒，默认 30 秒的期限会让 trial 中途失败。
- API key **不写进文件**，文件里只有 `env:OPENAI_API_KEY` / `env:XAI_API_KEY`，启动 Client 的终端里 export 了才生效。

### 6.3 核对 Router

```bash
cd $ROOT
colosseum-client/.venv/bin/python colosseum/tools/router_check.py colosseum-client/configs/robot.so101.joint.yaml
```

预期：`health: {'status': 'ok'}`，`token + institution: HTTP 200 (OK)`，并列出 Open track 的模型。**把列出的 name、URL 和 revision 和 6.1 的配置逐一对照**；Router 更新过注册信息就要改配置（VLA 还要重新下载对应 revision 的权重）。这个接口不显示 `max_horizon`，对不上时按第 10 节的办法查。

---

## 7. 逐级验证

按顺序做，每一级通过再做下一级。7.2 和 7.4 会让机械臂上力矩，**开始前把机械臂摆到静止姿态，人在旁边**。

### 7.1 单元测试

3.1 和 3.2 节已经跑过。

### 7.2 机械臂驱动（不涉及模型和 Router）

```bash
cd $ROOT
colosseum-client/.venv/bin/python colosseum/tools/hw_check.py \
  colosseum-client/configs/robot.so101.joint.yaml outputs/colosseum/probe
```

它会上力矩、读状态和两路相机、原地保持、让 wrist_roll 动 2°、夹爪动 5、回位、松力矩。检查：

- 打开约 3 秒，`get_observation` 约 1 ms；
- 夹爪和 wrist_roll 朝目标方向动了（舵机有死区，小位移差 0.5–1° 正常）；
- **打开 `outputs/colosseum/probe/` 里的两张 PNG**：`head_image` 是前置视角、`left_image` 是腕部视角，颜色正常（不是灰度图，红蓝没有反）。

这一步同时把当前画面和姿态存成 `.npy`，7.3 要用。

### 7.3 8 个模型各走一遍完整的本地协议（不动机械臂）

终端 1 启动 Policy Server（就是正式评测的那条命令）：

```bash
cd $ROOT/colosseum-policy-server
uv run --no-sync colosseum-policy-local --config configs/local-runtime-so101-mixed-joint.yaml
```

终端 2：

```bash
cd $ROOT
CFG=colosseum-policy-server/configs/local-runtime-so101-mixed-joint.yaml
export OPENAI_API_KEY=sk-... XAI_API_KEY=xai-...     # 只给 gpt / grok 用
for m in molmoact2-so101 pi05-so101 g05-so101 groot-n17-so101 lingbot-molmoact2-so101 \
         grok-4.7 gpt-6-astra claude-opus-5-5; do
  echo "=== $m"
  colosseum-policy-server/.venv/bin/python colosseum/tools/runtime_probe.py $CFG $m outputs/colosseum/probe "Pick up the red cube."
done
```

`runtime_probe.py` 的行为和真正的 Client 一样：VLA 连续请求 3 段动作；LLM 只请求 1 段，并且照 Client 的方式带凭证（gpt/grok 用这个终端里的 key，claude 用 Claude Code 登录）。LLM 每个会真实调用一次 API，花费很少。

原机器 2026-10-21 用模板生成的配置跑出来的结果，供对照：

| 模型 | `ready` 耗时 | 返回 | 第一步与当前姿态的最大偏差 |
| --- | --- | --- | --- |
| molmoact2-so101 | 37 s | (30, 6) | 13.0°（elbow，见下） |
| pi05-so101 | 97 s | (50, 6) | 6.5° |
| g05-so101 | 42 s（首段推理约 26 s 预热） | (30, 6) | 1.1° |
| groot-n17-so101 | 75 s | (40, 6) | 6.5° |
| lingbot-molmoact2-so101 | 85 s，之后每段约 4 s | (50, 6) | 3.6° |
| grok-4.7 | < 1 s，规划约 40 s | (18, 6)，10 Hz | 4.9° |
| gpt-6-astra | < 1 s，规划约 4 s | (17, 6)，10 Hz | 4.9° |
| claude-opus-5-5 | < 1 s，规划约 6 s | (41, 6)，30 Hz | 1.4° |

- LLM 每次返回的段长不固定，取决于它规划了几步。
- MolmoAct2 那 13° 是正常的：静止姿态的 elbow 略超出它训练数据的范围，它会先拉回范围内。
- **如果第一步偏差到了几十上百度（尤其是 shoulder_lift），说明坐标系转换没生效或被重复应用了。**
- 某个模型失败时，看 `outputs/colosseum/policy-logs/` 里最新的 `<模型名>-时间.log`。

### 7.4 开环正确性检查（可选，只针对 4 个通用 VLA）

用训练分布里的公开数据，对比模型预测和人类示教的真实动作，能发现图像通道、归一化、关节顺序、坐标系方面的错误。Policy Server 保持运行。

```bash
cd $ROOT
envs/.venv/bin/python colosseum/tools/fetch_reference_samples.py outputs/colosseum/ref
colosseum-policy-server/.venv/bin/python colosseum/tools/openloop_check.py \
  colosseum-policy-server/configs/local-runtime-so101-mixed-joint.yaml outputs/colosseum/ref outputs/colosseum \
  molmoact2-so101 pi05-so101 g05-so101 groot-n17-so101
```

看每个模型最后的 `ALL` 行。原机器上的结果（有采样随机性，上下浮动 0.5° 正常）：

| 模型 | 预测误差 | 对照：原地不动的误差 |
| --- | --- | --- |
| molmoact2-so101 | 2.2–3.2° | 9.1° |
| pi05-so101 | 2.5–2.8° | 13.3° |
| g05-so101 | 3.6–4.1° | 9.1° |
| groot-n17-so101 | 6.7–6.9°（40 步） | 11.2° |

预测误差应明显小于"原地不动"的误差。两者接近或预测误差更大，就是推理链路有问题。

### 7.5 第一次正式评测

终端 1 保持 Policy Server 运行。终端 2：

```bash
export OPENAI_API_KEY=sk-... XAI_API_KEY=xai-...
cd $ROOT/colosseum-client
uv run --no-sync colosseum-robot configs/robot.so101.joint.yaml --no-execute-action
```

`--no-execute-action` 会正常读机械臂、请求推理、打印每段动作，但不发给电机。确认打印的动作贴近当前姿态、逐步变化平缓，再去掉这个参数正式跑。

注意：带 `--no-execute-action` 的这一轮打完分也会作为真实评测提交。不想计入就在打分前按 Ctrl-C，下次启动会自动关掉这条未完成的任务。

---

## 8. 日常使用

### 8.1 正式评测

```bash
# 终端 1
cd $ROOT/colosseum-policy-server
uv run --no-sync colosseum-policy-local --config configs/local-runtime-so101-mixed-joint.yaml

# 终端 2
export OPENAI_API_KEY=sk-...
export XAI_API_KEY=xai-...
cd $ROOT/colosseum-client
uv run --no-sync colosseum-robot configs/robot.so101.joint.yaml
```

流程：输入任务指令（完整英文句子，会原样发给模型）和场景描述（只做记录）→ 摆好场景按 Enter → 模型加载完后输入 `y` 确认 → A 跑完打 0–100 分 → 恢复场景跑 B、打分 → 选 A / B / tie → 自动提交和上传 → 问是否继续。

- 每个 trial 开始前把机械臂摆到静止姿态；结束时它会回到这个姿态再松力矩。G05 例外，它从配置里的 `start_poses` 起步。
- trial 中按 Enter 提前结束；默认上限 800 步。
- 同一种场景布置用同一个 `scene` 字符串，方便以后筛选。

### 8.2 上传失败

看到 `Dataset upload deferred`：分数已经提交成功，只是录像没传完（通常是网络问题）。补传：

```bash
cd $ROOT/colosseum-client
PATH=$PWD/.venv/bin:$PATH .venv/bin/colosseum-upload-dataset configs/robot.so101.joint.yaml --evaluation-id ev_编号
```

不带 `--evaluation-id` 就补传 `eval_runs/` 里所有没传完的。看到 `Uploaded and verified N LeRobot datasets.` 才算真的传完。

### 8.3 清显存

Policy Server 异常退出后，worker 有时会留着占显存，下一次启动报 `MODEL_START_FAILED` 或 CUDA OOM：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill; sleep 3
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

停 Policy Server 用 `pkill -f "[c]olosseum-policy-local"`。方括号不能省，否则 `pkill -f` 会匹配到执行它的 shell 本身。

### 8.4 只跑 VLA，不跑 LLM

`colosseum/configs/local-runtime-so101.yaml.template` 是只包含 5 个 VLA 的配置（`backend: vla`），配合 `colosseum/configs/robot.so101.yaml.template` 使用，生成方法和 6.1 节一样。Router 一旦分配到 LLM，这套配置会报 `unregistered model`。所以正式参加 Open track 时用 mixed-joint。

### 8.5 独立脚本（不经过 Router，不录制也不打分）

先停掉 Policy Server（显存放不下两个模型）。

```bash
cd $ROOT
colosseum-client/.venv/bin/python scripts/run_open_model.py --model pi05          # 或 molmoact2 / g05 / groot
colosseum-client/.venv/bin/python scripts/run_open_model_smooth.py --model pi05   # 无停顿版本
```

提示 `Task>` 时输入指令回车开始，再按 Enter 提前停，空指令或 `q` 退出。常用参数：`--dry-run`（只打印不执行）、`--duration 60`。

---

## 9. 已知现象（不是配置错误）

| 现象 | 说明 |
| --- | --- |
| 正式评测里动作一顿一顿 | Client 是同步推理，每段动作之间机械臂停着等下一段。这是上游公共代码的设计 |
| LLM 每一步之间停很久 | 带图规划一次要 15–40 秒（Grok 最慢）。只要不超过 `deadline_ms`（180 s）就正常 |
| 模型在真机上几乎不动、得 0 分 | 原机器的场景下 MolmoAct2 和 G05 都停在静止姿态附近。对照实验表明原因在画面：换成训练集的画面、配上我们机械臂的状态，模型就会动。推理链路经 7.4 验证是正确的 |
| G05、LingBot 每轮刚开始很慢 | 慢启动；G05 第一段推理还有十几到二十几秒的预热 |
| 回放视频比实际短 | 导出的数据集按名义 30 Hz 排帧，停顿期间没有帧；真实时间在 `observation.capture_timestamp` |

---

## 10. 排错

| 症状 | 原因和处理 |
| --- | --- |
| Client 报 `INVALID_REQUEST`，Policy Server 日志有 `unregistered model` | 本地配置和 Router 发来的模型规格不一致（常见于 Router 新加了模型或改了 revision）。读 Router 的实际规格：Client 在 `eval_runs/<评测>/<run>/policy.json` 里存了成功那一侧的完整规格；或者对照 `router_check.py` 的输出。改配置后**重启 Policy Server** |
| `MODEL_START_FAILED: Model startup failed` | worker 没起来。看 `outputs/colosseum/policy-logs/` 最新的日志。常见：权重缺失（尤其是 LingBot）、对应环境没装好、显存被别的进程占着（8.3）、LingBot 时内存不够 |
| LLM 报 `Client must supply its API key for this LLM` / `Assigned LLM has no matching Client API key` | 启动 Client 的终端里没 export 对应的 `OPENAI_API_KEY` / `XAI_API_KEY`。export 后重启 Client 即可，不用重启 Policy Server |
| LLM 报 401 / `invalid api key` | key 本身无效或没额度。`python3 scripts/check_openai_key.py` 可以单独检查 OpenAI key（不会打印 key） |
| Claude 报 `Claude Code CLI not found` | `claude_command` 路径不对，或者 3.7 节没做。确认 `envs/claude-cli/claude_agent_sdk/_bundled/claude --version` 能运行 |
| Claude 报未登录 / 认证失败 | 用运行 Policy Server 的那个用户执行 `auth status`，必要时 `auth login`（3.7） |
| Claude 报 `requires Anthropic without an API key` | Client 配置里同时设了 `llm_api_keys.anthropic`，删掉它，只保留 `llm_account_logins.anthropic: claude-code` |
| LLM trial 中途超时 | Client 配置里缺 `deadline_ms: 180000` |
| `Local Policy evaluation currently supports robot_type: franka or yam` / `SO101 hardware control is not implemented` | 子仓库没有 SO101 改动，回到 2.2 节 |
| `Install colosseum-client[so101] for LeRobot` | Client 环境没装 `so101`/`so101-eef` extra |
| `Install ffmpeg before starting a trial` | `PATH` 里没有 ffmpeg，见 3.1；用静态版本时要用 `uv run --no-sync` 启动 |
| `SO101 'follower_arm_open' is not calibrated` | 标定文件名和 `robot_id` 不一致，或者电机里存的标定和文件不同，见 5.2 |
| 打开串口 `Permission denied` | 用户不在 `dialout` 组，见 `scripts/fix_serial_permissions.sh` |
| 腕部画面是灰度图或视角不对 | 相机标识指到了别的设备（比如 RealSense 的红外流），改用 `/dev/v4l/by-id/` 路径 |
| RealSense 打不开 | 确认没有别的进程占用它；Client 环境里的 `pyrealsense2` 是 2.56.5（LeRobot 限制在 2.57 以下） |
| CUDA out of memory | 有别的模型还占着显存，见 8.3 |
| LingBot 启动时报 `Python.h: No such file` | launcher 里的 `C_INCLUDE_PATH` 没有指向 `envs/pyheaders`，用模板重新生成配置 |
| 第一步动作偏离当前姿态几十上百度 | 坐标系转换问题，见第 0 节第 2 点。检查配置里有没有误加 `joint_frame: current` |
| `uv` 报 `No solution found`（LingBot 环境） | 少了 `--no-deps`，见 3.6 |
| `uv` 下载超时 | `export UV_HTTP_TIMEOUT=1200 UV_CONCURRENT_DOWNLOADS=3` 后重试 |
| Policy Server 和 Router 的连接时断时续 | 无线网卡在 AP 之间漫游后 IP 变了。尽量用有线网，或者把无线连接固定到一个 BSSID |

**只想验证连通性、不想留正式记录**：把 Client 配置里的 `test` 设成 `true` 再启动。这时用的是假机械臂和假画面，Router 上的记录带 Test 标记、不计入排名。验证完改回 `false`。

---

## 11. 哪些东西是机器专属的

换机器时这些必须重新确认，不能照搬：

- `colosseum-client/configs/robot.so101.joint.yaml`：token、串口、相机标识、关节限位；
- `colosseum-policy-server/configs/local-runtime-so101-mixed-joint.yaml`：全是绝对路径，用 6.1 节重新生成，关节限位要重填；
- 机械臂标定（`~/.cache/huggingface/lerobot/calibration/`）；
- Claude Code 登录（`~/.claude`），每台机器各登录一次；
- 所有 `.venv` 和 `envs/claude-cli`：按第 3 节重建，不要拷贝。

**不要提交进任何仓库**（都含 token、密码或 API key，主仓库的 `.gitignore` 已经排除了）：`.env`、`command.md`、`running_command.md`、两个子仓库 `configs/` 下的 `*.yaml`。

---

## 12. 本目录的内容

```
colosseum/
├── README.md                                   本文档
├── patches/
│   ├── colosseum-client-so101.patch                基于上游 c37afee
│   └── colosseum-policy-server-so101.patch         基于上游 698b75d
├── configs/
│   ├── local-runtime-so101-mixed-joint.yaml.template   Policy Server：5 VLA + 3 LLM（正式评测用）
│   ├── robot.so101.joint.yaml.template                 Client：配合上一个
│   ├── local-runtime-so101.yaml.template               Policy Server：只有 5 个 VLA
│   └── robot.so101.yaml.template                       Client：配合上一个
└── tools/
    ├── download_weights.py                    下载 5 个 VLA 的权重、GR00T 基座、Qwen3-VL processor
    ├── joint_limits.py                        从标定文件计算关节限位
    ├── read_joints.py                         读当前关节角
    ├── router_check.py                        Router 连通性和已注册模型（只读）
    ├── hw_check.py                            机械臂驱动自检
    ├── runtime_probe.py                       单个模型走完整本地协议（VLA 和 LLM 都支持）
    ├── fetch_reference_samples.py             下载开环检查用的参考样本
    └── openloop_check.py                      预测动作对比真实动作
```

相关文档：`colosseum-client/docs/so101.md`（驱动的契约和行为）、`colosseum-policy-server/docs/so101.md`（worker、适配器、mixed runtime）、`colosseum-policy-server/docs/inspect-agent.md`（LLM agent）。

### 原机器上验证过什么（2026-10-21）

- 两个子仓库合并上游 2026-10-19 的 `main` 之后的单元测试（Client 347 passed / 1 skipped，Policy Server 307 passed / 1 skipped）；
- 两份补丁在上游基准提交的全新 clone 上应用后，文件树和 `so101-support` HEAD 完全一致；
- **用本文的模板生成配置，8 个模型全部走通了完整的本地协议**（7.3 节的表格），其中 LingBot 用的是 `checkpoints/Qwen3-VL-4B-Instruct`，Claude 用的是 `envs/claude-cli` 里新装的 CLI；
- 合并上游之后，又用真正的 Client 代码（`LocalPolicyClient`，读 `robot.so101.joint.yaml`）连接真正的 Policy Server，只把 Router 换成本地模拟（重放 Router 以前实际下发的 8 个模型规格）。8 个模型都拿到了动作，3 个 LLM 都是 `joint_pos` 关节控制；
- `envs/frozen/` 里 MolmoAct2、pi0.5、LingBot（加 `--no-deps`）三份清单，在空环境里用 `--dry-run` 都能完整解析；G05、Client、Policy Server 三个 `uv.lock` 都和各自的 `pyproject.toml` 一致；
- 这条机械臂的驱动真机自检，以及 Open track 正式评测（含提交和上传）。

### 没有验证过的

- **本文档的步骤没有在第二台机器上从头走过一遍**。第 3 节里各环境的建法是按原机器的实际状态整理的，并且做了依赖解析检查，但完整的下载和安装只在原机器上做过；
- LingBot 的权重仓库已经不公开，`download_weights.py` 能否下载取决于你的 HF 账号有没有权限，多半要走拷贝；
- GR00T 的模型卡要求用训练项目里更新过的 LeRobot GR00T 实现，那份代码没有公开。我们用的是 LeRobot 0.6.1，只喂两路真实相机；正确性只有开环检查作为依据；
- 非 RTX 50 系显卡、其他型号的相机。

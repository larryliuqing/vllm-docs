# vLLM 源码安装与启动指南

## 一、下载 vLLM 源码

```bash
# 克隆源码仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

## 二、安装 Miniconda

```bash
# 下载并安装 Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/.miniconda3

# 初始化 conda
echo 'source ~/.miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
source ~/.bashrc
```

## 三、创建 Python 环境

```bash
# 创建 vllm-cpu 环境（Python 3.12）
conda create -n vllm-cpu python=3.12 -y

# 激活环境
conda activate vllm-cpu
```

## 四、安装依赖

```bash
# 安装 PyTorch CPU 版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 vLLM 依赖
pip install -r requirements/common.txt
pip install -r requirements/cpu.txt
```

## 五、编译源码（✅ 必须步骤）

```bash
cd ~/work/code/vllm-project/vllm
pip install -e . --no-build-isolation
```

### 编译产物

编译完成后，`build/` 目录下会生成以下文件：
- `_C.abi3.so` - 主扩展模块
- `_C_AVX512.abi3.so` - AVX512 优化版本  
- `_C_AVX2.abi3.so` - AVX2 优化版本

## 六、从源码启动模型

### 方式 1：Python API

```bash
conda activate vllm-cpu
cd ~/work/code/vllm-project/vllm
export PYTHONPATH=~/work/code/vllm-project/vllm:$PYTHONPATH

python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='distilgpt2', gpu_memory_utilization=0.2, max_num_seqs=1)
outputs = llm.generate('Hello!', SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
"
```

### 方式 2：CLI 服务模式

```bash
conda activate vllm-cpu
cd ~/work/code/vllm-project/vllm
VLLM_TARGET_DEVICE=cpu python -m vllm.entrypoints.cli.main serve distilgpt2 \
  --gpu-memory-utilization 0.2 \
  --max-num-seqs 1 \
  --port 8000
```

### 方式 3：HTTP API 调用

```bash
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!", "max_tokens": 50}'
```

## 七、关键说明

| 步骤 | 是否必须 | 说明 |
|------|----------|------|
| 下载源码 | ✅ | 获取最新代码 |
| 创建环境 | ✅ | 隔离依赖环境 |
| 安装依赖 | ✅ | 安装 PyTorch 等依赖 |
| **编译源码** | ✅✅ | **必须！** 编译 C++ 扩展模块 |
| 设置环境变量 | ✅ | `VLLM_TARGET_DEVICE=cpu` |

## 八、注意事项

1. **编译时间**：首次编译可能需要 10-30 分钟
2. **内存要求**：建议至少 16GB 内存运行模型
3. **环境变量**：启动前必须设置 `VLLM_TARGET_DEVICE=cpu`
4. **模型选择**：内存有限时建议使用小模型如 `distilgpt2`

## 九、验证安装

```bash
conda activate vllm-cpu
cd ~/work/code/vllm-project/vllm
export PYTHONPATH=~/work/code/vllm-project/vllm:$PYTHONPATH

python -c "
import vllm
print('vLLM version:', vllm.__version__)
print('Module path:', vllm.__file__)
print('Is local build:', '/work/code/vllm-project/vllm' in vllm.__file__)
"
```

### 预期输出

```
vLLM version: 0.20.2rc1.dev367+g0d4d334ea.d20260515
Module path: /home/sam/work/code/vllm-project/vllm/vllm/__init__.py
Is local build: True
```

## 十、常见问题

### Q1：启动时报错 "Failed to infer device type"

**解决方案**：设置环境变量 `VLLM_TARGET_DEVICE=cpu`

```bash
VLLM_TARGET_DEVICE=cpu python -m vllm.entrypoints.cli.main serve ...
```

### Q2：内存不足报错

**解决方案**：降低内存利用率参数

```bash
--gpu-memory-utilization 0.2
```

### Q3：编译失败

**解决方案**：确保安装了必要的编译工具

```bash
sudo apt-get install build-essential cmake ninja-build
```

---

**文档版本**：v1.0  
**创建日期**：2026-05-15  
**适用场景**：WSL Ubuntu 环境下的 vLLM CPU 版本安装
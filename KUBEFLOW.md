# 在 Kubeflow 上运行 zIMDB 训练

本文档介绍如何在 Kubeflow 上使用 4 个 T4 GPU 进行分布式训练。

## 目录

- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [构建 Docker 镜像](#构建-docker-镜像)
- [部署到 Kubeflow](#部署到-kubeflow)
- [配置说明](#配置说明)
- [监控和调试](#监控和调试)
- [故障排除](#故障排除)

## 前提条件

### 1. 集群要求

- Kubernetes 集群（版本 >= 1.21）
- Kubeflow 已安装（版本 >= 1.7）
- Kubeflow Training Operator（支持 PyTorchJob）
- 至少一个具有 4 个 NVIDIA T4 GPU 的节点

### 2. 本地工具

```bash
# 检查工具是否已安装
docker --version
kubectl version --client
```

### 3. 访问权限

- Docker 镜像仓库的推送权限（例如 Docker Hub、Google Container Registry、AWS ECR）
- Kubeflow 集群的 kubectl 访问权限

## 快速开始

### 步骤 1：配置镜像仓库

编辑 YAML 配置文件，替换 `<YOUR_REGISTRY>` 为你的镜像仓库地址：

```bash
# 例如使用 Docker Hub
YOUR_REGISTRY="dockerhub-username"

# 或使用 Google Container Registry
YOUR_REGISTRY="gcr.io/your-project-id"

# 或使用 AWS ECR
YOUR_REGISTRY="123456789.dkr.ecr.us-west-2.amazonaws.com"
```

### 步骤 2：构建并推送镜像

```bash
# 构建镜像
docker build -t ${YOUR_REGISTRY}/zimdb-training:latest .

# 推送镜像
docker push ${YOUR_REGISTRY}/zimdb-training:latest
```

### 步骤 3：更新 YAML 配置

```bash
# 自动替换镜像地址
sed -i "s|<YOUR_REGISTRY>|${YOUR_REGISTRY}|g" kubeflow-pytorchjob-single-node.yaml
```

### 步骤 4：部署到 Kubeflow

```bash
# 创建 PVC 和 PyTorchJob
kubectl apply -f kubeflow-pytorchjob-single-node.yaml

# 查看任务状态
kubectl get pytorchjobs -n kubeflow
```

### 步骤 5：查看训练日志

```bash
# 获取 Pod 名称
POD_NAME=$(kubectl get pods -n kubeflow -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}')

# 查看日志
kubectl logs -f ${POD_NAME} -n kubeflow
```

## 构建 Docker 镜像

### 本地构建

```bash
# 基础构建
docker build -t zimdb-training:latest .

# 指定标签
docker build -t zimdb-training:v1.0.0 .

# 多平台构建（如果需要）
docker buildx build --platform linux/amd64 -t zimdb-training:latest .
```

### 使用 CI/CD

#### GitHub Actions 示例

创建 `.github/workflows/build-push.yaml`：

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    paths:
      - 'train.py'
      - 'Dockerfile'
      - 'pyproject.toml'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/zimdb-training:latest
```

## 部署到 Kubeflow

### 配置文件说明

项目提供了两种部署配置：

#### 1. 单节点 4 GPU 配置（推荐）

**文件**: `kubeflow-pytorchjob-single-node.yaml`

- **适用场景**: 一台服务器上有 4 个 GPU
- **架构**: 1 个 Pod，4 个 GPU
- **通信**: 节点内 GPU 通信，延迟最低
- **命令**: `torchrun --standalone --nproc_per_node=4`

```bash
# 部署
kubectl apply -f kubeflow-pytorchjob-single-node.yaml
```

#### 2. 多节点分布式配置

**文件**: `kubeflow-pytorchjob.yaml`

- **适用场景**: GPU 分布在多台服务器
- **架构**: 4 个 Pod（1 Master + 3 Workers），每个 1 GPU
- **通信**: 跨节点 GPU 通信
- **命令**: `torchrun --nproc_per_node=4 --nnodes=1`

```bash
# 部署
kubectl apply -f kubeflow-pytorchjob.yaml
```

### 检查部署状态

```bash
# 查看 PyTorchJob 状态
kubectl get pytorchjobs -n kubeflow

# 查看 Pod 状态
kubectl get pods -n kubeflow -l app=zimdb-training

# 查看详细信息
kubectl describe pytorchjob zimdb-training-single-node-4gpu -n kubeflow
```

### 查看训练日志

```bash
# 实时查看日志
kubectl logs -f -n kubeflow \
  $(kubectl get pods -n kubeflow -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}')

# 查看最近 100 行
kubectl logs --tail=100 -n kubeflow \
  $(kubectl get pods -n kubeflow -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}')
```

## 配置说明

### 资源配置

#### GPU 配置

```yaml
resources:
  limits:
    nvidia.com/gpu: 4  # 请求 4 个 GPU
    memory: 64Gi       # 内存限制
    cpu: 16            # CPU 限制
  requests:
    nvidia.com/gpu: 4  # GPU 请求
    memory: 32Gi       # 内存请求
    cpu: 8             # CPU 请求
```

#### 节点选择

```yaml
nodeSelector:
  # 确保调度到有 T4 GPU 的节点
  accelerator: nvidia-tesla-t4
  # 或使用自定义标签
  # gpu-type: t4
  # gpu-count: "4"
```

### 存储配置

#### 数据缓存 PVC

用于存储 IMDB 数据集缓存：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zimdb-data-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

#### 模型输出 PVC

用于保存训练好的模型：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zimdb-model-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### 环境变量

```yaml
env:
# NCCL 调试级别
- name: NCCL_DEBUG
  value: INFO  # 可选: INFO, WARN, ERROR

# NCCL 网络接口
- name: NCCL_SOCKET_IFNAME
  value: eth0  # 根据实际网络配置调整

# GPU 可见性
- name: CUDA_VISIBLE_DEVICES
  value: "0,1,2,3"

# 禁用 InfiniBand（如果不可用）
- name: NCCL_IB_DISABLE
  value: "1"

# 启用 P2P 通信（节点内 GPU）
- name: NCCL_P2P_DISABLE
  value: "0"
```

## 监控和调试

### 查看 GPU 使用情况

如果节点上安装了 NVIDIA Device Plugin：

```bash
# 查看节点 GPU 资源
kubectl describe nodes | grep -A 10 "Capacity:" | grep nvidia.com/gpu

# 查看 GPU 使用情况（需要 DCGM Exporter）
kubectl get --raw /apis/metrics.k8s.io/v1beta1/nodes | jq .
```

### 进入 Pod 调试

```bash
# 获取 Pod 名称
POD_NAME=$(kubectl get pods -n kubeflow -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}')

# 进入 Pod
kubectl exec -it ${POD_NAME} -n kubeflow -- /bin/bash

# 在 Pod 内检查 GPU
nvidia-smi

# 检查 PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 查看事件

```bash
# 查看 namespace 中的事件
kubectl get events -n kubeflow --sort-by='.lastTimestamp'

# 查看特定 Pod 的事件
kubectl describe pod ${POD_NAME} -n kubeflow
```

### TensorBoard（可选）

如果需要可视化训练过程，可以集成 TensorBoard：

1. 修改 `train.py` 添加 TensorBoard 日志：

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/mnt/tensorboard')
```

2. 创建 TensorBoard Service：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: tensorboard
  namespace: kubeflow
spec:
  ports:
  - port: 6006
    targetPort: 6006
  selector:
    app: tensorboard
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorboard
  namespace: kubeflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensorboard
  template:
    metadata:
      labels:
        app: tensorboard
    spec:
      containers:
      - name: tensorboard
        image: tensorflow/tensorflow:latest
        command: ["tensorboard"]
        args: ["--logdir=/mnt/tensorboard", "--host=0.0.0.0"]
        ports:
        - containerPort: 6006
        volumeMounts:
        - name: tensorboard-logs
          mountPath: /mnt/tensorboard
      volumes:
      - name: tensorboard-logs
        persistentVolumeClaim:
          claimName: zimdb-tensorboard-pvc
```

## 故障排除

### 问题 1: Pod 一直处于 Pending 状态

**原因**: 没有满足条件的节点

**解决方案**:

```bash
# 检查节点资源
kubectl describe nodes | grep -A 5 "Allocated resources"

# 检查 Pod 事件
kubectl describe pod ${POD_NAME} -n kubeflow

# 调整资源请求或节点选择器
```

### 问题 2: 镜像拉取失败

**原因**: 镜像不存在或没有权限

**解决方案**:

```bash
# 验证镜像是否存在
docker pull ${YOUR_REGISTRY}/zimdb-training:latest

# 创建 ImagePullSecret
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry-server> \
  --docker-username=<your-name> \
  --docker-password=<your-password> \
  -n kubeflow

# 在 YAML 中添加
spec:
  template:
    spec:
      imagePullSecrets:
      - name: regcred
```

### 问题 3: GPU 不可用

**原因**: NVIDIA Device Plugin 未安装或配置错误

**解决方案**:

```bash
# 检查 NVIDIA Device Plugin
kubectl get pods -n kube-system | grep nvidia

# 安装 NVIDIA Device Plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# 验证 GPU 资源
kubectl describe nodes | grep nvidia.com/gpu
```

### 问题 4: NCCL 初始化失败

**原因**: 网络配置或多卡通信问题

**解决方案**:

```bash
# 设置调试环境变量
env:
- name: NCCL_DEBUG
  value: INFO
- name: NCCL_DEBUG_SUBSYS
  value: ALL

# 禁用 InfiniBand
- name: NCCL_IB_DISABLE
  value: "1"

# 指定网络接口
- name: NCCL_SOCKET_IFNAME
  value: eth0
```

### 问题 5: OOM (Out of Memory)

**原因**: GPU 内存不足

**解决方案**:

1. 减小 batch size：

```python
# 在 train.py 中修改
BATCH_SIZE = 32  # 从 64 减小到 32
```

2. 启用梯度检查点：

```python
USE_GRADIENT_CHECKPOINTING = True
```

3. 增加梯度累积步数：

```python
GRADIENT_ACCUMULATION_STEPS = 8
```

### 问题 6: 训练速度慢

**解决方案**:

1. 检查 GPU 利用率：

```bash
kubectl exec -it ${POD_NAME} -n kubeflow -- nvidia-smi dmon -s u
```

2. 优化 DataLoader：

```python
# 增加 num_workers
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
```

3. 启用混合精度训练：

```python
USE_AMP = True
```

## 清理资源

```bash
# 删除 PyTorchJob
kubectl delete pytorchjob zimdb-training-single-node-4gpu -n kubeflow

# 删除 PVC（注意：会删除数据）
kubectl delete pvc zimdb-data-pvc zimdb-model-pvc -n kubeflow

# 查看剩余资源
kubectl get all -n kubeflow -l app=zimdb-training
```

## 最佳实践

### 1. 资源规划

- 预留足够的内存（建议每个 GPU 配置 8-16GB 内存）
- CPU 核心数建议为 GPU 数量的 2-4 倍
- 使用节点亲和性确保调度到合适的节点

### 2. 数据管理

- 使用 PVC 持久化数据集和模型
- 考虑使用 NFS 或对象存储用于大规模数据
- 定期清理训练产生的临时文件

### 3. 监控

- 集成 Prometheus + Grafana 监控 GPU 使用率
- 使用 TensorBoard 可视化训练过程
- 设置告警通知训练失败

### 4. 成本优化

- 使用 Spot/Preemptible 实例降低成本
- 训练完成后及时清理资源
- 考虑使用 Kubeflow Pipelines 自动化流程

## 进阶配置

### 使用自定义训练参数

通过 ConfigMap 传递参数：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-config
  namespace: kubeflow
data:
  config.json: |
    {
      "num_epochs": 5,
      "batch_size": 128,
      "learning_rate": 0.001
    }
```

在 Pod 中挂载：

```yaml
volumeMounts:
- name: config
  mountPath: /config

volumes:
- name: config
  configMap:
    name: training-config
```

### 多阶段训练

创建 Kubeflow Pipeline 实现多阶段训练流程：

```python
from kfp import dsl

@dsl.pipeline(
    name='zIMDB Training Pipeline',
    description='Multi-stage training pipeline'
)
def training_pipeline():
    # 阶段 1: 数据预处理
    preprocess_op = dsl.ContainerOp(...)

    # 阶段 2: 训练
    train_op = dsl.ContainerOp(...)

    # 阶段 3: 评估
    eval_op = dsl.ContainerOp(...)

    train_op.after(preprocess_op)
    eval_op.after(train_op)
```

## 参考资料

- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

## 获取帮助

如果遇到问题，请：

1. 检查 Pod 日志: `kubectl logs <pod-name> -n kubeflow`
2. 查看事件: `kubectl describe pytorchjob <job-name> -n kubeflow`
3. 提交 Issue 到项目仓库
4. 查阅 Kubeflow 社区文档

---

**最后更新**: 2025-10-27
**版本**: 1.0.0

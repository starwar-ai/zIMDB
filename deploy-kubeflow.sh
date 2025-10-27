#!/bin/bash
# Kubeflow 部署脚本 - zIMDB 训练
# 使用单节点 4 GPU 配置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  zIMDB Kubeflow 部署脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查必要工具
echo -e "${YELLOW}检查必要工具...${NC}"
command -v docker >/dev/null 2>&1 || { echo -e "${RED}错误: docker 未安装${NC}" >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}错误: kubectl 未安装${NC}" >&2; exit 1; }
echo -e "${GREEN}✓ 工具检查完成${NC}"
echo ""

# 获取镜像仓库地址
if [ -z "$1" ]; then
    echo -e "${YELLOW}请输入你的 Docker 镜像仓库地址:${NC}"
    echo "例如: dockerhub-username"
    echo "     gcr.io/your-project-id"
    echo "     123456789.dkr.ecr.us-west-2.amazonaws.com"
    read -p "镜像仓库地址: " REGISTRY
else
    REGISTRY=$1
fi

if [ -z "$REGISTRY" ]; then
    echo -e "${RED}错误: 镜像仓库地址不能为空${NC}"
    exit 1
fi

IMAGE_NAME="${REGISTRY}/zimdb-training:latest"
echo -e "${GREEN}镜像名称: ${IMAGE_NAME}${NC}"
echo ""

# 询问是否构建镜像
read -p "是否需要构建并推送 Docker 镜像? (y/n): " BUILD_IMAGE
if [[ "$BUILD_IMAGE" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}构建 Docker 镜像...${NC}"
    docker build -t ${IMAGE_NAME} .
    echo -e "${GREEN}✓ 镜像构建完成${NC}"
    echo ""

    echo -e "${YELLOW}推送镜像到仓库...${NC}"
    docker push ${IMAGE_NAME}
    echo -e "${GREEN}✓ 镜像推送完成${NC}"
    echo ""
fi

# 更新 YAML 配置
echo -e "${YELLOW}更新 YAML 配置文件...${NC}"
YAML_FILE="kubeflow-pytorchjob-single-node.yaml"
TEMP_FILE="${YAML_FILE}.tmp"

# 创建临时文件并替换镜像地址
sed "s|<YOUR_REGISTRY>|${REGISTRY}|g" ${YAML_FILE} > ${TEMP_FILE}

echo -e "${GREEN}✓ 配置文件更新完成${NC}"
echo ""

# 检查 Kubernetes 连接
echo -e "${YELLOW}检查 Kubernetes 集群连接...${NC}"
if kubectl cluster-info >/dev/null 2>&1; then
    echo -e "${GREEN}✓ 成功连接到集群${NC}"
    kubectl cluster-info | head -1
else
    echo -e "${RED}错误: 无法连接到 Kubernetes 集群${NC}"
    echo "请检查 kubectl 配置"
    rm -f ${TEMP_FILE}
    exit 1
fi
echo ""

# 检查 namespace
NAMESPACE="kubeflow"
echo -e "${YELLOW}检查 namespace: ${NAMESPACE}${NC}"
if kubectl get namespace ${NAMESPACE} >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Namespace ${NAMESPACE} 存在${NC}"
else
    echo -e "${YELLOW}Namespace ${NAMESPACE} 不存在，正在创建...${NC}"
    kubectl create namespace ${NAMESPACE}
    echo -e "${GREEN}✓ Namespace 创建完成${NC}"
fi
echo ""

# 询问是否部署
echo -e "${YELLOW}准备部署到 Kubeflow...${NC}"
echo "配置信息:"
echo "  - 镜像: ${IMAGE_NAME}"
echo "  - Namespace: ${NAMESPACE}"
echo "  - GPU 数量: 4"
echo "  - GPU 类型: NVIDIA T4"
echo ""
read -p "确认部署? (y/n): " CONFIRM_DEPLOY

if [[ "$CONFIRM_DEPLOY" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}部署到 Kubeflow...${NC}"
    kubectl apply -f ${TEMP_FILE}
    echo -e "${GREEN}✓ 部署完成${NC}"
    echo ""

    # 等待 Pod 创建
    echo -e "${YELLOW}等待 Pod 创建...${NC}"
    sleep 5

    # 显示状态
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  部署状态${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    echo -e "${YELLOW}PyTorchJob 状态:${NC}"
    kubectl get pytorchjobs -n ${NAMESPACE} -l training-type=single-node-multi-gpu
    echo ""

    echo -e "${YELLOW}Pod 状态:${NC}"
    kubectl get pods -n ${NAMESPACE} -l app=zimdb-training
    echo ""

    # 获取 Pod 名称
    POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=zimdb-training -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -n "$POD_NAME" ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  有用的命令${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "查看训练日志:"
        echo "  kubectl logs -f ${POD_NAME} -n ${NAMESPACE}"
        echo ""
        echo "查看 PyTorchJob 状态:"
        echo "  kubectl describe pytorchjob -n ${NAMESPACE} -l training-type=single-node-multi-gpu"
        echo ""
        echo "进入 Pod 调试:"
        echo "  kubectl exec -it ${POD_NAME} -n ${NAMESPACE} -- /bin/bash"
        echo ""
        echo "查看 GPU 使用:"
        echo "  kubectl exec ${POD_NAME} -n ${NAMESPACE} -- nvidia-smi"
        echo ""
        echo "删除任务:"
        echo "  kubectl delete pytorchjob -n ${NAMESPACE} -l training-type=single-node-multi-gpu"
        echo ""

        # 询问是否查看日志
        read -p "是否查看训练日志? (y/n): " VIEW_LOGS
        if [[ "$VIEW_LOGS" =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}开始监控训练日志 (Ctrl+C 退出)...${NC}"
            echo ""
            kubectl logs -f ${POD_NAME} -n ${NAMESPACE}
        fi
    fi
else
    echo -e "${YELLOW}取消部署${NC}"
fi

# 清理临时文件
rm -f ${TEMP_FILE}

echo ""
echo -e "${GREEN}脚本执行完成！${NC}"

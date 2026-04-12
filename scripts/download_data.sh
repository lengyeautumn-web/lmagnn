#!/bin/bash
# 创建数据目录
mkdir -p data
cd data

# 下载 WN18RR 数据集示例 (这里假设你有一个托管地址或从常用仓库下载)
echo "Downloading WN18RR dataset..."
wget https://github.com/villmow/datasets/raw/master/wordnet/WN18RR.tar.gz
tar -xvf WN18RR.tar.gz
rm WN18RR.tar.gz

echo "Data preparation complete."
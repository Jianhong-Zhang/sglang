nohup /usr/local/bin/etcd/etcd --name node1  --initial-cluster "node1=http://0.0.0.0:2480" --listen-peer-urls http://0.0.0.0:2480 --initial-advertise-peer-urls http://0.0.0.0:2480 --listen-client-urls http://0.0.0.0:2479 --advertise-client-urls http://0.0.0.0:2479 --data-dir /tmp/etcd &

nohup nats-server -js -p 5222 &

# google/gemma-4-E2B-it
# Qwen/Qwen3-0.6B
# google/gemma-4-31B-it-qat-w4a16-ct
NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=0,1 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang   --model-path google/gemma-4-31B-it-qat-w4a16-ct   --page-size 64  --enable-hierarchical-cache --hicache-io-backend kernel_xpu  --hicache-size 20 --max-total-tokens 74000  --hicache-write-policy write_through --tp 2  --hicache-storage-backend nixl   --skip-tokenizer-init --log-level DEBUG

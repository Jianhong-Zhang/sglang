nohup /usr/local/bin/etcd/etcd --name node1  --initial-cluster "node1=http://0.0.0.0:2480" --listen-peer-urls http://0.0.0.0:2480 --initial-advertise-peer-urls http://0.0.0.0:2480 --listen-client-urls http://0.0.0.0:2479 --advertise-client-urls http://0.0.0.0:2479 --data-dir /tmp/etcd &

nohup nats-server -js -p 5222 &

# google/gemma-4-E2B-it
# Qwen/Qwen3-0.6B
# google/gemma-4-31B-it-qat-w4a16-ct

#### basic aggregated commandline ####
# HF_HUB_OFFLINE=1 HF_HUB_CACHE=/mnt/weka/data/litang_pdtest/hf/hub NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang --device xpu --attention-backend intel_xpu   --model-path google/gemma-4-31B-it-qat-w4a16-ct   --page-size 64   --tp 2 --swa-full-tokens-ratio 0.2    --skip-tokenizer-init --log-level DEBUG --enable-request-time-stats-logging --enable-mfu-metrics --enable-metrics

#### agg + hicache ####
# HF_HUB_OFFLINE=1 HF_HUB_CACHE=/mnt/weka/data/litang_pdtest/hf/hub NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang --device xpu --attention-backend intel_xpu --model-path google/gemma-4-31B-it-qat-w4a16-ct  --page-size 64 --tp 2 --swa-full-tokens-ratio 0.2 --skip-tokenizer-init --log-level DEBUG --enable-request-time-stats-logging --enable-mfu-metrics --enable-metrics \
# --enable-hierarchical-cache --hicache-io-backend kernel_xpu  --hicache-size 40  --hicache-write-policy write_through  --hicache-storage-backend nixl  

#### prefill + hicache #####a
export PYTHONPATH=/opt/intel/nixl-xpu:/opt/intel/intel_nixl/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/opt/intel/intel_nixl/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NIXL_PLUGIN_DIR=/opt/intel/intel_nixl/lib/x86_64-linux-gnu/plugins

export no_proxy="${no_proxy},172.26.46.0/22,192.165.123.0/24"
export NO_PROXY="$no_proxy"

HF_HUB_OFFLINE=1 HF_HUB_CACHE=/mnt/weka/data/litang_pdtest/hf/hub NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang --device xpu --attention-backend intel_xpu   --model-path google/gemma-4-31B-it-qat-w4a16-ct   --page-size 64   --tp 2 --swa-full-tokens-ratio 0.2    --skip-tokenizer-init --log-level DEBUG --enable-request-time-stats-logging --enable-mfu-metrics --enable-metrics --hicache-io-backend kernel_xpu  --hicache-size 40  --hicache-write-policy write_through  --hicache-storage-backend nixl \
--enable-metrics --host 0.0.0.0 --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disaggregation-ib-device mlx5_0 \
> hicache_p.log 2>&1

# NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang --device xpu --attention-backend intel_xpu   --model-path google/gemma-4-31B-it-qat-w4a16-ct   --page-size 64  --enable-hierarchical-cache --hicache-io-backend kernel_xpu  --hicache-size 40  --hicache-write-policy write_through --tp 2 --swa-full-tokens-ratio 0.2 --hicache-storage-backend nixl   --skip-tokenizer-init --log-level DEBUG --enable-request-time-stats-logging --enable-mfu-metrics --enable-metrics  > hicache3_100k.log 2>&1
#
#
# NEOReadDebugKeys=1 AllowUnrestrictedSize=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 ZE_AFFINITY_MASK=2,3 NATS_SERVER="nats://127.0.0.1:5222" ETCD_ENDPOINTS="http://127.0.0.1:2479"  DYN_LOG="debug" python -m dynamo.sglang --device xpu --attention-backend intel_xpu   --model-path google/gemma-4-31B-it-qat-w4a16-ct   --page-size 64 --tp 2 --swa-full-tokens-ratio 0.2    --skip-tokenizer-init --log-level DEBUG --enable-request-time-stats-logging --enable-mfu-metrics --enable-metrics  > nohicache.log 2>&1

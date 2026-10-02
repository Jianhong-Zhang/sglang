# CUDA coredump: see which GPU kernel a hung rank is stuck in

Needs `cuda-gdb`, which ships with the CUDA toolkit; make sure it is on `PATH`.

Set these before launching, so you can trigger a GPU coredump on demand while the process hangs:

```bash
export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1
export CUDA_COREDUMP_PIPE="/tmp/cuda_pipe_%h_%p"
export CUDA_COREDUMP_FILE="/tmp/cuda_coredump_%h_%p"
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
```

While the process hangs, find the pipe via `/proc/<pid>/fd/` and write to it to trigger the dump:

```bash
ls /proc/<pid>/fd/ -la 2>/dev/null | grep cuda_pipe
dd if=/dev/zero bs=1M count=1 > /tmp/cuda_pipe_<hostname>_<pid>
```

If you don't need to keep the process alive, `kill -SIGABRT <pid>` also triggers a CUDA coredump, but terminates the
process.

Open it with `cuda-gdb --batch -ex "target cudacore <coredump_file>"`. On load it shows which kernel is stuck, for
example:

```
Opening GPU coredump: <coredump_file>
[Current focus set to CUDA kernel 0, grid <grid>, cluster (4,0,0), block (16,0,0), thread (64,0,0), device 0, sm 0, warp 0, lane 0]
#0  0x<addr> in ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)<<<(24,1,1),(512,1,1)>>> ()
```

A stuck `ncclDevKernel_*` frame means the hang is in a collective, not a compute kernel. Combine it with the py-spy stack
(the Python caller of that collective) to name the collective and its call site. If that caller is an all-gather, for
example, suspect a size mismatch between the TP ranks.

# it is cuda cuda time!!

## Benchmark methodology
- Kernels were tested for saturating performance only, using a randomly generated dataset.

- Runtimes were measured using nsys/ncu. As a result, devices ran at their base clock speed, and all other costs were not taken into account (e.g., additional memory allocation).

- No kernel implements boundary checks, so the number of elements in each dataset was rounded up to the nearest aligned value.

- Given the very non-rigorous methodology, the results are intended only to show the potential of the optimizations used.

## Testing devices
- NVIDIA A100 SXM4 40GB:
  + CUDA: 12.4
  + Driver: 550.54.15 
- NVIDIA Geforce RTX 3060 12GB:
  + CUDA: 12.8
  + Driver: 572.83

## Results

### Merge

#### NVIDIA A100 SXM4
Test data: 100M ints

|            | Thrust (ms) | CCT (ms) | Improvement (%) |
|------------|-------------|----------|-----------------|
| Merge path | 0.087       | 0.056    | 55%             |
| Merge      | 1.15        | 1.15     | 0%              |
| Total      | 1.237       | 1.206    | 2.57%           |

The main optimization here is that the merge path kernel uses a binary-lifting-like binary search, which means there are more common elements among threads within a warp. Although the merge path kernel does not take up much of the total runtime, the significant individual speedup translates into a non-negligible overall improvement.


### Prefix sum

#### NVIDIA A100 SXM4
Test data: 2^30 floats

|       | CUB (ms) | CUB 32 regs (ms) | CCT (ms) | Improvement (%) |
|-------|----------|------------------|----------|-----------------|
| Total | 7.09     | 6.84             | 6.22     | 10%             |

Limiting the CUB scan to 32 registers per thread allows it to run at 100% occupancy.

The optimization here is that one warp handles the lookback while other warps perform the calculation. Should I name it Async Decoupled Lookback?

### Sort
Test data: 2^28 uints
|      | Thrust (ms) | CCT (ms) | Improvement (%) |
|------|-------------|----------|-----------------|
| A100 | 11.08       | 9.62     | 15%             | 
| 3060 | 38.67       | 32.79    | 18%             | 


Main optimizations used:

- ADL, similar to the prefix sum kernel. In this kernel, 16 warps perform the calculations, and 8 warps handle the lookback.

- Each look-back thread is responsible for the count of a digit. Each thread can broadcast the sum of several blocks, allowing for skipping behavior. For example, if 3 bits are used as a flag:

  + Flag `111` means completed.
  + Flag `000` means the result is not ready.
  + Flag `001` means the block holds the sum for that block only.
  + Flag `010` means the block holds the sum for that block and the previous block.
  + Etc.

- And some, probably architecture specific, optimzations. At least they seem to work on Ampere.

To my surprise, the kernel performs equivalently well on less powerful device like the 3060.
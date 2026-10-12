// gpu_vram_check.cu - allocate (almost) all free VRAM on each GPU and write/verify patterns.
// Written for this repo (scripts/diag). Purpose: check that the advertised memory (e.g. 32 GB
// on a modified board) really exists, is not aliased, and returns what was written.
//
// For every pattern ALL chunks are written first and only then ALL chunks are verified, so
// two virtual ranges that alias the same physical memory are caught by the address patterns.
// Patterns (64-bit words, value depends on the word's device address A):
//   0 addr       : A            1 not_addr : ~A
//   2 alt_5555   : 0x5555..55   3 alt_aaaa : 0xAAAA..AA
//   4 rand       : splitmix64(A ^ seed)     5 not_rand : ~splitmix64(A ^ seed)
//
// Refuses to run when `nvidia-smi --query-compute-apps` lists a process on a selected GPU,
// or when that list cannot be read, unless --force is given.
// Reports per GPU: total memory, bytes allocated, bytes verified, mismatches (first ones with
// address / expected / actual), OR of all wrong bits.
//
// Build: nvcc -O2 -arch=sm_89 -o gpu_vram_check gpu_vram_check.cu
// Usage: gpu_vram_check [--devices 0,1,..] [--margin-mib N (512)] [--chunk-mib N (1024)]
//                       [--patterns 0,1,2,3,4,5] [--loops N (1)] [--seed N] [--force]
// Exit: 0 all verified without mismatches, 1 mismatches found, 2 usage/refused,
//       3 no CUDA device or CUDA error.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(call)                                                                                   \
  do {                                                                                             \
    cudaError_t e_ = (call);                                                                       \
    if (e_ != cudaSuccess) {                                                                       \
      fprintf(stderr, "fatal CUDA error %s (%s) at %s:%d: %s\n", cudaGetErrorName(e_),             \
              cudaGetErrorString(e_), __FILE__, __LINE__, #call);                                  \
      exit(3);                                                                                     \
    }                                                                                              \
  } while (0)

struct Mismatch {
  unsigned long long addr, expected, actual;
};

static const char *pname[6] = {"addr", "not_addr", "alt_5555", "alt_aaaa", "rand", "not_rand"};

__host__ __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

__device__ __forceinline__ uint64_t pattern(int kind, uint64_t addr, uint64_t seed) {
  switch (kind) {
    case 0: return addr;
    case 1: return ~addr;
    case 2: return 0x5555555555555555ull;
    case 3: return 0xAAAAAAAAAAAAAAAAull;
    case 4: return splitmix64(addr ^ seed);
    default: return ~splitmix64(addr ^ seed);
  }
}

__global__ void fill(uint64_t *p, size_t n, int kind, uint64_t seed) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = pattern(kind, (uint64_t)(uintptr_t)(p + i), seed);
}

__global__ void verify(const uint64_t *p, size_t n, int kind, uint64_t seed, unsigned long long *nerr,
                       unsigned long long *badbits, Mismatch *log, unsigned int *nlog, unsigned int maxlog) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint64_t a = (uint64_t)(uintptr_t)(p + i);
    uint64_t e = pattern(kind, a, seed), v = p[i];
    if (v != e) {
      atomicAdd(nerr, 1ull);
      atomicOr(badbits, (unsigned long long)(v ^ e));
      unsigned int k = atomicAdd(nlog, 1u);
      if (k < maxlog) log[k] = Mismatch{a, e, v};
    }
  }
}

// ---------------------------------------------------------------- compute-apps guard
struct Proc {
  std::string line;
  unsigned dom = 0, bus = 0, dev = 0;
  bool parsed = false;
};

static bool list_compute_apps(std::vector<Proc> &out, std::string &why) {
  FILE *f = popen("nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv,noheader 2>&1", "r");
  if (!f) { why = "cannot start nvidia-smi"; return false; }
  char buf[1024];
  std::vector<std::string> lines;
  while (fgets(buf, sizeof buf, f)) {
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
    if (!s.empty()) lines.push_back(s);
  }
  int rc = pclose(f);
  if (rc != 0) {
    why = "nvidia-smi --query-compute-apps failed (exit status " + std::to_string(rc) + ")";
    for (auto &l : lines) why += "\n    " + l;
    return false;
  }
  for (auto &l : lines) {
    if (l.find("No running") != std::string::npos) continue;
    Proc p;
    p.line = l;
    size_t c = l.rfind(',');
    if (c != std::string::npos) {
      unsigned dom, bus, dev, fn;
      if (sscanf(l.c_str() + c + 1, " %x:%x:%x.%x", &dom, &bus, &dev, &fn) == 4) {
        p.dom = dom; p.bus = bus; p.dev = dev; p.parsed = true;
      }
    }
    out.push_back(p);
  }
  return true;
}

static void usage(const char *p) {
  fprintf(stderr,
          "usage: %s [--devices 0,1,..] [--margin-mib N (512)] [--chunk-mib N (1024)]\n"
          "          [--patterns 0,1,2,3,4,5] [--loops N (1)] [--seed N] [--force]\n",
          p);
}

static std::vector<int> parse_list(const char *s) {
  std::vector<int> v;
  std::string t(s);
  size_t pos = 0;
  while (pos <= t.size()) {
    size_t c = t.find(',', pos);
    std::string x = t.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    if (!x.empty()) v.push_back(atoi(x.c_str()));
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  return v;
}

int main(int argc, char **argv) {
  std::vector<int> devs, pats = {0, 1, 2, 3, 4, 5};
  size_t margin_mib = 512, chunk_mib = 1024;
  int loops = 1;
  bool force = false;
  uint64_t seed = 0x408032ull;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&]() -> const char * {
      if (i + 1 >= argc) { usage(argv[0]); exit(2); }
      return argv[++i];
    };
    if (a == "--devices") devs = parse_list(need());
    else if (a == "--margin-mib") margin_mib = strtoull(need(), nullptr, 10);
    else if (a == "--chunk-mib") chunk_mib = strtoull(need(), nullptr, 10);
    else if (a == "--patterns") pats = parse_list(need());
    else if (a == "--loops") loops = atoi(need());
    else if (a == "--seed") seed = strtoull(need(), nullptr, 0);
    else if (a == "--force") force = true;
    else { usage(argv[0]); return 2; }
  }
  if (chunk_mib < 16 || loops < 1 || pats.empty()) { usage(argv[0]); return 2; }
  for (int p : pats)
    if (p < 0 || p > 5) { usage(argv[0]); return 2; }

  int n = 0;
  cudaError_t ce = cudaGetDeviceCount(&n);
  if (ce != cudaSuccess || n == 0) {
    fprintf(stderr, "no usable CUDA device: %s\n", ce != cudaSuccess ? cudaGetErrorString(ce) : "count is 0");
    return 3;
  }
  if (devs.empty())
    for (int d = 0; d < n; d++) devs.push_back(d);
  for (int d : devs)
    if (d < 0 || d >= n) { fprintf(stderr, "device %d does not exist (count %d)\n", d, n); return 2; }

  // guard: refuse if other compute processes are on the selected GPUs
  std::vector<Proc> procs;
  std::string why;
  bool listed = list_compute_apps(procs, why);
  if (!listed) {
    fprintf(stderr, "cannot verify that the GPUs are idle: %s\n", why.c_str());
    if (!force) { fprintf(stderr, "refusing to run (use --force to override)\n"); return 2; }
    fprintf(stderr, "--force given: continuing\n");
  } else {
    bool busy = false;
    for (int d : devs) {
      cudaDeviceProp p;
      CK(cudaGetDeviceProperties(&p, d));
      for (auto &pr : procs) {
        bool same = !pr.parsed || ((int)pr.dom == p.pciDomainID && (int)pr.bus == p.pciBusID && (int)pr.dev == p.pciDeviceID);
        if (same) {
          fprintf(stderr, "GPU %d (%04x:%02x:%02x) has a compute process: %s\n", d, p.pciDomainID, p.pciBusID,
                  p.pciDeviceID, pr.line.c_str());
          busy = true;
        }
      }
    }
    if (busy && !force) { fprintf(stderr, "refusing to run (use --force to override)\n"); return 2; }
    if (busy) fprintf(stderr, "--force given: continuing despite running processes\n");
  }

  printf("gpu_vram_check: devices");
  for (int d : devs) printf(" %d", d);
  printf(", margin %zu MiB, chunk %zu MiB, loops %d, seed 0x%llx, patterns", margin_mib, chunk_mib, loops,
         (unsigned long long)seed);
  for (int p : pats) printf(" %s", pname[p]);
  printf("\n");

  const unsigned int maxlog = 64;
  bool any_mismatch = false;
  for (int d : devs) {
    CK(cudaSetDevice(d));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, d));
    char bus[32] = {0};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, d));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    size_t margin = margin_mib << 20;
    size_t target = fr > margin ? fr - margin : 0;
    printf("\nGPU %d %s %s: totalGlobalMem %zu B (%.2f GiB), free %zu B (%.2f GiB), target %.2f GiB\n", d, p.name, bus,
           (size_t)p.totalGlobalMem, p.totalGlobalMem / 1073741824.0, fr, fr / 1073741824.0, target / 1073741824.0);

    std::vector<std::pair<uint64_t *, size_t>> chunks;
    size_t chunk = chunk_mib << 20, allocated = 0;
    while (chunk >= (16u << 20)) {
      if (allocated + chunk > target) { chunk /= 2; continue; }
      void *ptr = nullptr;
      if (cudaMalloc(&ptr, chunk) != cudaSuccess) {
        cudaGetLastError();
        chunk /= 2;
        continue;
      }
      chunks.push_back({(uint64_t *)ptr, chunk});
      allocated += chunk;
    }
    printf("  allocated %zu B (%.2f GiB) in %zu chunks = %.1f%% of totalGlobalMem\n", allocated,
           allocated / 1073741824.0, chunks.size(), 100.0 * allocated / (double)p.totalGlobalMem);

    unsigned long long *d_nerr, *d_bad;
    unsigned int *d_nlog;
    Mismatch *d_log;
    CK(cudaMalloc(&d_nerr, sizeof(unsigned long long)));
    CK(cudaMalloc(&d_bad, sizeof(unsigned long long)));
    CK(cudaMalloc(&d_nlog, sizeof(unsigned int)));
    CK(cudaMalloc(&d_log, sizeof(Mismatch) * maxlog));
    int blocks = p.multiProcessorCount * 8, threads = 256;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    unsigned long long dev_err = 0, dev_bad = 0, verified = 0;
    for (int l = 0; l < loops; l++)
      for (int pk : pats) {
        CK(cudaMemset(d_nerr, 0, sizeof(unsigned long long)));
        CK(cudaMemset(d_bad, 0, sizeof(unsigned long long)));
        CK(cudaMemset(d_nlog, 0, sizeof(unsigned int)));
        uint64_t s = seed + (uint64_t)l;
        CK(cudaEventRecord(e0));
        for (auto &c : chunks) fill<<<blocks, threads>>>(c.first, c.second / 8, pk, s);
        CK(cudaGetLastError());
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float wms = 0, vms = 0;
        CK(cudaEventElapsedTime(&wms, e0, e1));
        CK(cudaEventRecord(e0));
        for (auto &c : chunks) verify<<<blocks, threads>>>(c.first, c.second / 8, pk, s, d_nerr, d_bad, d_log, d_nlog, maxlog);
        CK(cudaGetLastError());
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&vms, e0, e1));
        unsigned long long nerr = 0, bad = 0;
        unsigned int nlog = 0;
        CK(cudaMemcpy(&nerr, d_nerr, sizeof nerr, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&bad, d_bad, sizeof bad, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&nlog, d_nlog, sizeof nlog, cudaMemcpyDeviceToHost));
        verified += allocated;
        dev_err += nerr;
        dev_bad |= bad;
        printf("  loop %d pattern %-8s write %7.1f GB/s verify %7.1f GB/s mismatches %llu\n", l, pname[pk],
               allocated / (wms * 1e-3) / 1e9, allocated / (vms * 1e-3) / 1e9, nerr);
        if (nerr) {
          std::vector<Mismatch> h(maxlog);
          unsigned int k = nlog < maxlog ? nlog : maxlog;
          CK(cudaMemcpy(h.data(), d_log, sizeof(Mismatch) * k, cudaMemcpyDeviceToHost));
          for (unsigned int i = 0; i < k; i++) {
            long chunk_idx = -1;
            unsigned long long off = 0;
            for (size_t c = 0; c < chunks.size(); c++) {
              unsigned long long b = (unsigned long long)(uintptr_t)chunks[c].first;
              if (h[i].addr >= b && h[i].addr < b + chunks[c].second) { chunk_idx = (long)c; off = h[i].addr - b; break; }
            }
            printf("MISMATCH\tdev=%d\tpattern=%s\tchunk=%ld\toffset=0x%llx\taddr=0x%llx\texpected=0x%016llx\tactual=0x%016llx\n",
                   d, pname[pk], chunk_idx, off, h[i].addr, h[i].expected, h[i].actual);
          }
        }
      }
    any_mismatch |= dev_err != 0;
    printf("RESULT\tdev=%d\tbus=%s\tname=%s\ttotal_global_mem=%zu\tfree_before=%zu\tallocated=%zu\tverified_bytes=%llu\tpatterns=%zu\tloops=%d\tmismatches=%llu\tbad_bits=0x%016llx\n",
           d, bus, p.name, (size_t)p.totalGlobalMem, fr, allocated, verified, pats.size(), loops, dev_err, dev_bad);
    for (auto &c : chunks) cudaFree(c.first);
    cudaFree(d_nerr); cudaFree(d_bad); cudaFree(d_nlog); cudaFree(d_log);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  printf("\noverall: %s\n", any_mismatch ? "MISMATCHES FOUND" : "no mismatches");
  return any_mismatch ? 1 : 0;
}

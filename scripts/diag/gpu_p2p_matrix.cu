// gpu_p2p_matrix.cu - GPU inventory, P2P capability and PCIe bandwidth/latency matrix.
// Written for this repo (scripts/diag). CUDA runtime API only; no NVML, no external code.
//
// Measures (GB/s = 1e9 bytes/s; best = fastest of --reps after one warm-up):
//   * device list: name, memory, PCI bus id, compute capability, SM count
//   * cudaDeviceCanAccessPeer and cudaDeviceGetP2PAttribute for every ordered pair
//   * H2D / D2H, pinned and pageable host memory, per GPU
//   * D2D for every ordered pair via cudaMemcpyPeerAsync, BEFORE and AFTER
//     cudaDeviceEnablePeerAccess (each enable call result is reported); diagonal = local copy
//   * bidirectional D2D per pair with peer access (both directions at once)
//   * all GPUs at once: H2D and D2H simultaneously (shared upstream lanes show up as an
//     aggregate well below the sum of the solo numbers)
//   * small-transfer latency (default 4 KiB): H2D, D2H, D2D peer; async copy + stream sync
//
// Nothing is written to any GPU setting. Peer access is enabled only inside this process.
//
// Build: nvcc -O2 -arch=sm_89 -o gpu_p2p_matrix gpu_p2p_matrix.cu
// Usage: gpu_p2p_matrix [--size-mib N] [--reps N] [--lat-bytes N] [--lat-iters N]
//                       [--csv FILE] [--no-pageable]
// Exit: 0 ok, 2 usage, 3 no CUDA device or fatal CUDA error.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

static FILE *g_csv = nullptr;

static std::string csv_field(const std::string &s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string o = "\"";
  for (char c : s) o += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return o + "\"";
}

// kind,src,dst,bytes,metric,value,unit,note
static void row(const std::string &kind, const std::string &src, const std::string &dst, size_t bytes,
                const std::string &metric, const std::string &value, const std::string &unit,
                const std::string &note = "") {
  FILE *f = g_csv ? g_csv : stdout;
  fprintf(f, "%s%s,%s,%s,%zu,%s,%s,%s,%s\n", g_csv ? "" : "CSV,", csv_field(kind).c_str(),
          csv_field(src).c_str(), csv_field(dst).c_str(), bytes, csv_field(metric).c_str(),
          csv_field(value).c_str(), csv_field(unit).c_str(), csv_field(note).c_str());
}

static std::string fmt(double v, const char *f = "%.3f") {
  char b[64];
  snprintf(b, sizeof b, f, v);
  return b;
}

static bool soft(cudaError_t e, const char *what, std::string *msg = nullptr) {
  if (e == cudaSuccess) return true;
  std::string m = std::string(cudaGetErrorName(e)) + ": " + cudaGetErrorString(e);
  fprintf(stderr, "  %s -> %s\n", what, m.c_str());
  if (msg) *msg = m;
  cudaGetLastError();  // clear non-sticky error
  return false;
}

struct Stat {
  double best = 0, median = 0;
  bool ok = false;
  std::string err;
};

static Stat stats_from_ms(std::vector<float> ms, size_t bytes) {
  Stat s;
  if (ms.empty()) return s;
  std::sort(ms.begin(), ms.end());
  double med = (ms.size() % 2) ? ms[ms.size() / 2] : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
  s.best = bytes / (ms.front() * 1e-3) / 1e9;
  s.median = bytes / (med * 1e-3) / 1e9;
  s.ok = true;
  return s;
}

// Times issue() on stream st of device dev with events; one warm-up, then reps.
template <class F>
static Stat time_copy(int dev, cudaStream_t st, size_t bytes, int reps, F issue) {
  Stat s;
  CK(cudaSetDevice(dev));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<float> ms;
  if (soft(issue(), "warm-up copy", &s.err) && soft(cudaStreamSynchronize(st), "warm-up sync", &s.err)) {
    for (int r = 0; r < reps; r++) {
      CK(cudaEventRecord(e0, st));
      if (!soft(issue(), "copy", &s.err)) break;
      CK(cudaEventRecord(e1, st));
      if (!soft(cudaEventSynchronize(e1), "sync", &s.err)) break;
      float t = 0;
      CK(cudaEventElapsedTime(&t, e0, e1));
      ms.push_back(t);
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  std::string err = s.err;
  s = stats_from_ms(ms, bytes);
  if ((int)ms.size() != reps) { s.ok = false; s.err = err.empty() ? "incomplete" : err; }
  return s;
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void print_matrix(const char *title, int n, const std::vector<std::string> &cells) {
  printf("\n%s (row = source GPU, column = destination GPU)\n%6s", title, "");
  for (int j = 0; j < n; j++) printf(" %9s", ("GPU" + std::to_string(j)).c_str());
  printf("\n");
  for (int i = 0; i < n; i++) {
    printf("%6s", ("GPU" + std::to_string(i)).c_str());
    for (int j = 0; j < n; j++) printf(" %9s", cells[i * n + j].c_str());
    printf("\n");
  }
}

static void usage(const char *p) {
  fprintf(stderr,
          "usage: %s [--size-mib N (256)] [--reps N (5)] [--lat-bytes N (4096)] [--lat-iters N (1000)]\n"
          "          [--csv FILE] [--no-pageable]\n",
          p);
}

int main(int argc, char **argv) {
  size_t mib = 256, lat_bytes = 4096;
  int reps = 5, lat_iters = 1000;
  bool pageable = true;
  const char *csv_path = nullptr;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](void) -> const char * {
      if (i + 1 >= argc) { usage(argv[0]); exit(2); }
      return argv[++i];
    };
    if (a == "--size-mib") mib = strtoull(need(), nullptr, 10);
    else if (a == "--reps") reps = atoi(need());
    else if (a == "--lat-bytes") lat_bytes = strtoull(need(), nullptr, 10);
    else if (a == "--lat-iters") lat_iters = atoi(need());
    else if (a == "--csv") csv_path = need();
    else if (a == "--no-pageable") pageable = false;
    else { usage(argv[0]); return 2; }
  }
  if (mib < 1 || reps < 1 || lat_bytes < 1 || lat_iters < 1) { usage(argv[0]); return 2; }
  const size_t bytes = mib << 20;
  if (csv_path) {
    g_csv = fopen(csv_path, "w");
    if (!g_csv) { perror(csv_path); return 2; }
  }
  if (g_csv) fprintf(g_csv, "kind,src,dst,bytes,metric,value,unit,note\n");

  int rt = 0, drv = 0, n = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  cudaError_t ce = cudaGetDeviceCount(&n);
  printf("gpu_p2p_matrix: CUDA runtime %d, driver API %d, CUDA_VISIBLE_DEVICES=%s, CUDA_DEVICE_ORDER=%s\n", rt, drv,
         getenv("CUDA_VISIBLE_DEVICES") ? getenv("CUDA_VISIBLE_DEVICES") : "(unset)",
         getenv("CUDA_DEVICE_ORDER") ? getenv("CUDA_DEVICE_ORDER") : "(unset)");
  row("runtime", "", "", 0, "cuda_runtime_version", std::to_string(rt), "");
  row("runtime", "", "", 0, "cuda_driver_version", std::to_string(drv), "");
  if (ce != cudaSuccess || n == 0) {
    fprintf(stderr, "no usable CUDA device: %s\n", ce != cudaSuccess ? cudaGetErrorString(ce) : "count is 0");
    row("runtime", "", "", 0, "device_count", "0", "", ce != cudaSuccess ? cudaGetErrorName(ce) : "");
    if (g_csv) fclose(g_csv);
    return 3;
  }
  row("runtime", "", "", 0, "device_count", std::to_string(n), "");
  printf("transfer size %zu MiB, reps %d (best and median after 1 warm-up), latency %zu B x %d\n", mib, reps,
         lat_bytes, lat_iters);

  // ---------------------------------------------------------------- devices
  printf("\n%-4s %-34s %-14s %12s %12s %5s %4s\n", "dev", "name", "pci_bus_id", "total_MiB", "free_MiB", "cc",
         "SMs");
  for (int d = 0; d < n; d++) {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, d));
    char bus[32] = {0};
    CK(cudaDeviceGetPCIBusId(bus, sizeof bus, d));
    CK(cudaSetDevice(d));
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    printf("%-4d %-34s %-14s %12.0f %12.0f %3d.%d %4d\n", d, p.name, bus, p.totalGlobalMem / 1048576.0,
           fr / 1048576.0, p.major, p.minor, p.multiProcessorCount);
    std::string ds = std::to_string(d);
    row("device", ds, "", 0, "name", p.name, "");
    row("device", ds, "", 0, "pci_bus_id", bus, "");
    row("device", ds, "", 0, "total_global_mem", std::to_string(p.totalGlobalMem), "bytes");
    row("device", ds, "", 0, "free_mem", std::to_string(fr), "bytes");
    row("device", ds, "", 0, "compute_capability", std::to_string(p.major) + "." + std::to_string(p.minor), "");
    row("device", ds, "", 0, "sm_count", std::to_string(p.multiProcessorCount), "");
  }

  // ---------------------------------------------------------------- P2P capability
  std::vector<std::string> cap(n * n, "-"), rank(n * n, "-");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      int can = 0, acc = 0, atom = 0, perf = 0;
      std::string si = std::to_string(i), sj = std::to_string(j), m;
      if (soft(cudaDeviceCanAccessPeer(&can, i, j), "cudaDeviceCanAccessPeer", &m)) {
        cap[i * n + j] = std::to_string(can);
        row("can_access_peer", si, sj, 0, "value", std::to_string(can), "bool");
      } else {
        cap[i * n + j] = "err";
        row("can_access_peer", si, sj, 0, "value", "error", "", m);
      }
      if (cudaDeviceGetP2PAttribute(&acc, cudaDevP2PAttrAccessSupported, i, j) == cudaSuccess)
        row("p2p_attr", si, sj, 0, "access_supported", std::to_string(acc), "bool");
      if (cudaDeviceGetP2PAttribute(&atom, cudaDevP2PAttrNativeAtomicSupported, i, j) == cudaSuccess)
        row("p2p_attr", si, sj, 0, "native_atomic_supported", std::to_string(atom), "bool");
      if (cudaDeviceGetP2PAttribute(&perf, cudaDevP2PAttrPerformanceRank, i, j) == cudaSuccess) {
        row("p2p_attr", si, sj, 0, "performance_rank", std::to_string(perf), "");
        rank[i * n + j] = std::to_string(perf);
      }
      cudaGetLastError();
    }
  print_matrix("cudaDeviceCanAccessPeer", n, cap);
  print_matrix("cudaDevP2PAttrPerformanceRank", n, rank);

  // ---------------------------------------------------------------- buffers
  std::vector<void *> dA(n), dB(n), hPin(n);
  std::vector<cudaStream_t> st(n);
  for (int d = 0; d < n; d++) {
    CK(cudaSetDevice(d));
    CK(cudaMalloc(&dA[d], bytes));
    CK(cudaMalloc(&dB[d], bytes));
    CK(cudaMemset(dA[d], 0x5a, bytes));
    CK(cudaMemset(dB[d], 0, bytes));
    CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
    CK(cudaMallocHost(&hPin[d], bytes));
    memset(hPin[d], 0xa5, bytes);
  }
  void *hPage = nullptr;
  if (pageable) {
    hPage = malloc(bytes);
    if (!hPage) { fprintf(stderr, "cannot allocate %zu pageable bytes\n", bytes); return 3; }
    memset(hPage, 0x3c, bytes);
  }
  CK(cudaDeviceSynchronize());

  // ---------------------------------------------------------------- host <-> device
  printf("\n%-4s %14s %14s %14s %14s   (GB/s, best / median)\n", "dev", "H2D pinned", "D2H pinned",
         "H2D pageable", "D2H pageable");
  std::vector<double> solo_h2d(n, 0), solo_d2h(n, 0);
  for (int d = 0; d < n; d++) {
    std::string ds = std::to_string(d);
    Stat a = time_copy(d, st[d], bytes, reps,
                       [&] { return cudaMemcpyAsync(dA[d], hPin[d], bytes, cudaMemcpyHostToDevice, st[d]); });
    Stat b = time_copy(d, st[d], bytes, reps,
                       [&] { return cudaMemcpyAsync(hPin[d], dA[d], bytes, cudaMemcpyDeviceToHost, st[d]); });
    Stat c, e;
    if (pageable) {
      c = time_copy(d, st[d], bytes, reps,
                    [&] { return cudaMemcpyAsync(dB[d], hPage, bytes, cudaMemcpyHostToDevice, st[d]); });
      e = time_copy(d, st[d], bytes, reps,
                    [&] { return cudaMemcpyAsync(hPage, dB[d], bytes, cudaMemcpyDeviceToHost, st[d]); });
    }
    solo_h2d[d] = a.best;
    solo_d2h[d] = b.best;
    auto cell = [](const Stat &s) { return s.ok ? fmt(s.best, "%.2f") + "/" + fmt(s.median, "%.2f") : std::string("n/a"); };
    printf("%-4d %14s %14s %14s %14s\n", d, cell(a).c_str(), cell(b).c_str(), pageable ? cell(c).c_str() : "skipped",
           pageable ? cell(e).c_str() : "skipped");
    struct { const char *k; Stat *s; } all[] = {{"h2d_pinned", &a}, {"d2h_pinned", &b}, {"h2d_pageable", &c}, {"d2h_pageable", &e}};
    for (auto &x : all) {
      if (!pageable && strstr(x.k, "pageable")) continue;
      bool h2d = strncmp(x.k, "h2d", 3) == 0;
      if (x.s->ok) {
        row(x.k, h2d ? "host" : ds, h2d ? ds : "host", bytes, "best_GBps", fmt(x.s->best), "GB/s");
        row(x.k, h2d ? "host" : ds, h2d ? ds : "host", bytes, "median_GBps", fmt(x.s->median), "GB/s");
      } else {
        row(x.k, h2d ? "host" : ds, h2d ? ds : "host", bytes, "error", "1", "", x.s->err);
      }
    }
  }

  // ---------------------------------------------------------------- D2D without / with peer access
  auto d2d_matrix = [&](const char *kind, const char *title) {
    std::vector<std::string> cells(n * n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        Stat s = time_copy(i, st[i], bytes, reps,
                           [&] { return cudaMemcpyPeerAsync(dB[j], j, dA[i], i, bytes, st[i]); });
        cells[i * n + j] = s.ok ? fmt(s.best, "%.2f") : std::string("err");
        std::string note = i == j ? "local copy on one device" : "";
        if (s.ok) {
          row(kind, std::to_string(i), std::to_string(j), bytes, "best_GBps", fmt(s.best), "GB/s", note);
          row(kind, std::to_string(i), std::to_string(j), bytes, "median_GBps", fmt(s.median), "GB/s", note);
        } else {
          row(kind, std::to_string(i), std::to_string(j), bytes, "error", "1", "", s.err);
        }
      }
    print_matrix(title, n, cells);
  };
  d2d_matrix("d2d_nopeer", "D2D cudaMemcpyPeerAsync WITHOUT peer access, best GB/s");

  std::vector<std::string> en(n * n, "-");
  std::vector<int> enabled(n * n, 0);
  for (int i = 0; i < n; i++) {
    CK(cudaSetDevice(i));
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
      std::string name = cudaGetErrorName(e);
      if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) enabled[i * n + j] = 1;
      en[i * n + j] = e == cudaSuccess ? "ok" : name.substr(0, 9);
      row("enable_peer", std::to_string(i), std::to_string(j), 0, "status", name, "",
          e == cudaSuccess ? "" : cudaGetErrorString(e));
      cudaGetLastError();
    }
  }
  print_matrix("cudaDeviceEnablePeerAccess result", n, en);
  d2d_matrix("d2d_peer", "D2D cudaMemcpyPeerAsync AFTER enabling peer access, best GB/s");

  // bidirectional, both directions at once
  {
    std::vector<std::string> cells(n * n, "-");
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++) {
        std::vector<double> t;
        std::string err;
        for (int r = 0; r <= reps; r++) {
          CK(cudaSetDevice(i)); CK(cudaDeviceSynchronize());
          CK(cudaSetDevice(j)); CK(cudaDeviceSynchronize());
          double t0 = now_s();
          CK(cudaSetDevice(i));
          bool ok1 = soft(cudaMemcpyPeerAsync(dB[j], j, dA[i], i, bytes, st[i]), "bidir i->j", &err);
          CK(cudaSetDevice(j));
          bool ok2 = soft(cudaMemcpyPeerAsync(dB[i], i, dA[j], j, bytes, st[j]), "bidir j->i", &err);
          bool ok3 = soft(cudaStreamSynchronize(st[i]), "sync", &err) && soft(cudaStreamSynchronize(st[j]), "sync", &err);
          if (!(ok1 && ok2 && ok3)) break;
          if (r > 0) t.push_back(now_s() - t0);
        }
        if ((int)t.size() == reps) {
          double best = 2.0 * bytes / *std::min_element(t.begin(), t.end()) / 1e9;
          cells[i * n + j] = cells[j * n + i] = fmt(best, "%.2f");
          row("d2d_bidir_peer", std::to_string(i), std::to_string(j), 2 * bytes, "best_GBps", fmt(best), "GB/s",
              "sum of both directions");
        } else {
          cells[i * n + j] = cells[j * n + i] = "err";
          row("d2d_bidir_peer", std::to_string(i), std::to_string(j), 2 * bytes, "error", "1", "", err);
        }
      }
    print_matrix("Bidirectional D2D (peer access state as above), best GB/s, sum of both directions", n, cells);
  }

  // ---------------------------------------------------------------- all GPUs simultaneously
  for (int dir = 0; dir < 2; dir++) {
    const char *kind = dir == 0 ? "h2d_all_simultaneous" : "d2h_all_simultaneous";
    std::vector<cudaEvent_t> e0(n), e1(n);
    for (int d = 0; d < n; d++) {
      CK(cudaSetDevice(d));
      CK(cudaEventCreate(&e0[d]));
      CK(cudaEventCreate(&e1[d]));
    }
    double best_agg = 0;
    std::vector<double> best_per(n, 0);
    for (int r = 0; r <= reps; r++) {
      for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); CK(cudaDeviceSynchronize()); }
      double t0 = now_s();
      for (int d = 0; d < n; d++) {
        CK(cudaSetDevice(d));
        CK(cudaEventRecord(e0[d], st[d]));
        if (dir == 0) CK(cudaMemcpyAsync(dA[d], hPin[d], bytes, cudaMemcpyHostToDevice, st[d]));
        else CK(cudaMemcpyAsync(hPin[d], dA[d], bytes, cudaMemcpyDeviceToHost, st[d]));
        CK(cudaEventRecord(e1[d], st[d]));
      }
      for (int d = 0; d < n; d++) { CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d])); }
      double dt = now_s() - t0;
      if (r == 0) continue;
      double agg = (double)n * bytes / dt / 1e9;
      if (agg > best_agg) {
        best_agg = agg;
        for (int d = 0; d < n; d++) {
          float ms = 0;
          CK(cudaEventElapsedTime(&ms, e0[d], e1[d]));
          best_per[d] = bytes / (ms * 1e-3) / 1e9;
        }
      }
    }
    printf("\n%s (pinned): aggregate %.2f GB/s over %d GPUs\n", kind, best_agg, n);
    double sum_solo = 0;
    for (int d = 0; d < n; d++) {
      double solo = dir == 0 ? solo_h2d[d] : solo_d2h[d];
      sum_solo += solo;
      printf("  GPU%d: %.2f GB/s while all run (solo %.2f GB/s, ratio %.2f)\n", d, best_per[d], solo,
             solo > 0 ? best_per[d] / solo : 0.0);
      row(kind, dir == 0 ? "host" : std::to_string(d), dir == 0 ? std::to_string(d) : "host", bytes, "GBps",
          fmt(best_per[d]), "GB/s", "per GPU during the best simultaneous run");
      row(kind, dir == 0 ? "host" : std::to_string(d), dir == 0 ? std::to_string(d) : "host", bytes,
          "ratio_to_solo", fmt(solo > 0 ? best_per[d] / solo : 0.0), "");
    }
    printf("  aggregate / sum of solo = %.2f (1.0 = no shared bottleneck)\n", sum_solo > 0 ? best_agg / sum_solo : 0.0);
    row(kind, "all", "all", (size_t)n * bytes, "aggregate_GBps", fmt(best_agg), "GB/s");
    row(kind, "all", "all", (size_t)n * bytes, "aggregate_over_sum_of_solo", fmt(sum_solo > 0 ? best_agg / sum_solo : 0.0), "");
    for (int d = 0; d < n; d++) { cudaEventDestroy(e0[d]); cudaEventDestroy(e1[d]); }
  }

  // ---------------------------------------------------------------- latency
  printf("\nlatency of one %zu-byte copy (async copy + stream synchronize, mean of %d), microseconds\n", lat_bytes,
         lat_iters);
  auto lat = [&](int dev, cudaStream_t s, auto issue, std::string *err) -> double {
    CK(cudaSetDevice(dev));
    for (int k = 0; k < 10; k++)
      if (!soft(issue(), "latency warm-up", err) || !soft(cudaStreamSynchronize(s), "sync", err)) return -1;
    double t0 = now_s();
    for (int k = 0; k < lat_iters; k++) {
      if (!soft(issue(), "latency copy", err)) return -1;
      if (!soft(cudaStreamSynchronize(s), "sync", err)) return -1;
    }
    return (now_s() - t0) / lat_iters * 1e6;
  };
  for (int d = 0; d < n; d++) {
    std::string err, ds = std::to_string(d);
    double h = lat(d, st[d], [&] { return cudaMemcpyAsync(dA[d], hPin[d], lat_bytes, cudaMemcpyHostToDevice, st[d]); }, &err);
    double g = lat(d, st[d], [&] { return cudaMemcpyAsync(hPin[d], dA[d], lat_bytes, cudaMemcpyDeviceToHost, st[d]); }, &err);
    printf("  GPU%d: H2D %.2f us, D2H %.2f us\n", d, h, g);
    row("lat_h2d_pinned", "host", ds, lat_bytes, "mean_us", fmt(h), "us");
    row("lat_d2h_pinned", ds, "host", lat_bytes, "mean_us", fmt(g), "us");
  }
  {
    std::vector<std::string> cells(n * n, "-");
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        std::string err;
        double v = lat(i, st[i], [&] { return cudaMemcpyPeerAsync(dB[j], j, dA[i], i, lat_bytes, st[i]); }, &err);
        cells[i * n + j] = v >= 0 ? fmt(v, "%.2f") : std::string("err");
        row("lat_d2d_peer", std::to_string(i), std::to_string(j), lat_bytes, v >= 0 ? "mean_us" : "error",
            v >= 0 ? fmt(v) : "1", v >= 0 ? "us" : "", enabled[i * n + j] ? "peer access enabled" : "peer access NOT enabled");
      }
    print_matrix("D2D latency, microseconds", n, cells);
  }

  for (int d = 0; d < n; d++) {
    cudaSetDevice(d);
    cudaFree(dA[d]);
    cudaFree(dB[d]);
    cudaFreeHost(hPin[d]);
    cudaStreamDestroy(st[d]);
  }
  free(hPage);
  if (g_csv) { fclose(g_csv); printf("\nCSV written to %s\n", csv_path); }
  return 0;
}

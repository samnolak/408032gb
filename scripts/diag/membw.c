/*
 * membw.c - small OpenMP memory-bandwidth probe (copy / scale / add / triad / read).
 * Written for this repo (scripts/diag); it is not STREAM and shares no code with it,
 * only the well-known kernel shapes.
 *
 *   copy : c[i] = a[i]               bytes counted per element: 16
 *   scale: b[i] = s * c[i]           16
 *   add  : c[i] = a[i] + b[i]        24
 *   triad: a[i] = b[i] + s * c[i]    24
 *   read : sum += a[i]               8   (read-only; closest to CPU expert GEMV traffic)
 *
 * Bytes are the explicit loads+stores only (write-allocate traffic is not counted).
 * GB/s = 1e9 bytes per second. Each run executes every kernel once; run 0 is a warm-up
 * and is excluded. "best" uses the minimum time, "median" the median time.
 *
 * Build: gcc -O3 -march=native -fopenmp membw.c -o membw
 * Usage: membw [-s MiB_per_array] [-n runs] [-t threads] [-q]
 * Arrays are first touched in parallel with the same static schedule as the kernels, so
 * under numactl / OMP_PROC_BIND each thread's pages land on its own NUMA node.
 * Exit code: 0 ok, 1 validation failed, 2 usage / allocation error.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { K_COPY, K_SCALE, K_ADD, K_TRIAD, K_READ, NK };
static const char *kname[NK] = {"copy", "scale", "add", "triad", "read"};
static const double kbytes[NK] = {16.0, 16.0, 24.0, 24.0, 8.0};

static int cmp_double(const void *x, const void *y) {
  double a = *(const double *)x, b = *(const double *)y;
  return (a > b) - (a < b);
}

static void usage(const char *p) {
  fprintf(stderr,
          "usage: %s [-s MiB_per_array (default 1024)] [-n runs incl. warm-up (default 11, 2..200)]\n"
          "          [-t threads (default: OMP default)] [-q (CSV lines only)]\n", p);
}

int main(int argc, char **argv) {
  double mib = 1024.0;
  int runs = 11, threads = 0, quiet = 0, opt;
  while ((opt = getopt(argc, argv, "s:n:t:qh")) != -1) {
    switch (opt) {
      case 's': mib = atof(optarg); break;
      case 'n': runs = atoi(optarg); break;
      case 't': threads = atoi(optarg); break;
      case 'q': quiet = 1; break;
      default: usage(argv[0]); return 2;
    }
  }
  if (mib < 1.0 || runs < 2 || runs > 200 || threads < 0) { usage(argv[0]); return 2; }
  if (threads > 0) omp_set_num_threads(threads);

  const size_t n = (size_t)(mib * 1024.0 * 1024.0) / sizeof(double);
  const size_t bytes = n * sizeof(double);
  double *a = NULL, *b = NULL, *c = NULL;
  if (posix_memalign((void **)&a, 2u << 20, bytes) || posix_memalign((void **)&b, 2u << 20, bytes) ||
      posix_memalign((void **)&c, 2u << 20, bytes)) {
    fprintf(stderr, "membw: cannot allocate 3 x %zu bytes: %s\n", bytes, strerror(errno));
    return 2;
  }

  int nthr = 1;
  /* CPUs actually used by the worker threads (to verify binding). */
  int ncpu = (int)sysconf(_SC_NPROCESSORS_CONF);
  if (ncpu < 1) ncpu = 1;
  unsigned char *used = calloc((size_t)ncpu + 1, 1);
#pragma omp parallel
  {
#pragma omp single
    nthr = omp_get_num_threads();
    int cpu = sched_getcpu();
    if (used && cpu >= 0 && cpu < ncpu) used[cpu] = 1;
  }

  /* first touch, same static schedule as the kernels */
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }

  const double s = 3.0;
  double *t = malloc(sizeof(double) * (size_t)runs * NK);
  if (!t) { fprintf(stderr, "membw: out of memory\n"); return 2; }
  volatile double sink = 0.0;
  double ea = 1.0, eb = 2.0, ec = 0.0; /* expected element values */
  int sum_ok = 1;

  for (int r = 0; r < runs; r++) {
    double t0, sum = 0.0;
    t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static) reduction(+ : sum)
    for (size_t i = 0; i < n; i++) sum += a[i];
    t[r * NK + K_READ] = omp_get_wtime() - t0;
    sink += sum;
    if (fabs(sum - ea * (double)n) > 1e-6 * fabs(ea * (double)n)) sum_ok = 0;

    t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) c[i] = a[i];
    t[r * NK + K_COPY] = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) b[i] = s * c[i];
    t[r * NK + K_SCALE] = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];
    t[r * NK + K_ADD] = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = b[i] + s * c[i];
    t[r * NK + K_TRIAD] = omp_get_wtime() - t0;

    ec = ea; eb = s * ec; ec = ea + eb; ea = eb + s * ec;
  }

  /* validation: every element must equal the scalar replay (relative 1e-9) */
  double maxrel = 0.0;
#pragma omp parallel for schedule(static) reduction(max : maxrel)
  for (size_t i = 0; i < n; i++) {
    double ra = fabs(a[i] - ea) / fabs(ea), rb = fabs(b[i] - eb) / fabs(eb), rc = fabs(c[i] - ec) / fabs(ec);
    double m = ra > rb ? ra : rb;
    m = m > rc ? m : rc;
    if (m > maxrel) maxrel = m;
  }
  int valid = (maxrel <= 1e-9) && sum_ok;

  const char *bind = getenv("OMP_PROC_BIND"), *places = getenv("OMP_PLACES");
  if (!quiet) {
    printf("membw: %zu doubles per array = %.1f MiB per array, 3 arrays = %.1f MiB total\n", n,
           bytes / 1048576.0, 3.0 * bytes / 1048576.0);
    printf("membw: runs=%d (first is warm-up, excluded), threads=%d, OMP_PROC_BIND=%s, OMP_PLACES=%s\n", runs,
           nthr, bind ? bind : "(unset)", places ? places : "(unset)");
    printf("membw: cpus used by worker threads:");
    int first = 1;
    for (int i = 0; used && i < ncpu; i++) {
      if (!used[i]) continue;
      int j = i;
      while (j + 1 < ncpu && used[j + 1]) j++;
      printf(first ? " %d" : ",%d", i);
      if (j > i) printf("-%d", j);
      first = 0;
      i = j;
    }
    printf("\n");
    printf("%-6s %12s %12s %10s %10s %10s\n", "kernel", "best_GB/s", "median_GB/s", "min_s", "median_s", "max_s");
  }
  double *col = malloc(sizeof(double) * (size_t)runs);
  for (int k = 0; k < NK; k++) {
    int m = 0;
    for (int r = 1; r < runs; r++) col[m++] = t[r * NK + k];
    qsort(col, (size_t)m, sizeof(double), cmp_double);
    double tmin = col[0], tmax = col[m - 1];
    double tmed = (m % 2) ? col[m / 2] : 0.5 * (col[m / 2 - 1] + col[m / 2]);
    double gb = kbytes[k] * (double)n / 1e9;
    if (!quiet)
      printf("%-6s %12.2f %12.2f %10.6f %10.6f %10.6f\n", kname[k], gb / tmin, gb / tmed, tmin, tmed, tmax);
    printf("CSV,%s,%d,%zu,%d,%.3f,%.3f,%.6f,%.6f,%.6f\n", kname[k], nthr, bytes, runs - 1, gb / tmin, gb / tmed,
           tmin, tmed, tmax);
  }
  printf("validation: %s (max relative error %.3g, read-sum %s)\n", valid ? "ok" : "FAILED", maxrel,
         sum_ok ? "ok" : "mismatch");
  if (sink == 42.4242) printf(" \n"); /* keep the reductions alive */
  free(col); free(t); free(used); free(a); free(b); free(c);
  return valid ? 0 : 1;
}

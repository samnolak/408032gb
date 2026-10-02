// Functional test of strata-glm's DiskReader Linux path (408032gb patch 0003), extracted verbatim.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <random>
#include "diskreader.inc"   // extracted from the patched glm_main.cpp by run_diskreader_test.sh
static std::vector<uint8_t> mk(size_t n) { std::vector<uint8_t> v(n); std::mt19937 r(7); for (auto& x : v) x = (uint8_t) r(); return v; }
int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s\n", m); ++fails; } else std::printf("ok: %s\n", m); } while (0)
int main() {
    const size_t N = (size_t) 64 << 20 | 1234;   // 64 MiB + a tail that is not a multiple of 4096
    auto data = mk(N);
    for (const char* f : {"/tmp/drtest/a.bin", "/tmp/drtest/b.bin"}) { FILE* o = std::fopen(f, "wb"); std::fwrite(data.data(), 1, N, o); std::fclose(o); }
    DiskReader r;
    CHECK(r.open({"/tmp/drtest/a.bin", "/tmp/drtest/b.bin"}, {10.0, 7.0}, 4), "open two mirrors");
    CHECK(r.drives() == 2, "two drives");
    const uint64_t blob = 14155776;   // one GLM expert (14.16 MB), at offsets that are not 4 KiB aligned
    void* raw = nullptr;
    if (posix_memalign(&raw, 4096, blob + 2 * 4096)) return 2;
    uint8_t* dst = (uint8_t*) raw;
    for (uint64_t off : {(uint64_t) 0, (uint64_t) 4096 * 3 + 100, (uint64_t) 12345678}) {
        r.start(0, off, blob, dst, true);
        const bool ok = r.wait(0);
        const uint64_t a0 = off / 4096 * 4096;   // the blob lands at dst + off % 4096
        CHECK(ok && std::memcmp(dst + (off - a0), data.data() + off, blob) == 0, ("aligned dst, blob at " + std::to_string(off)).c_str());
    }
    {   // the file's last blob: its aligned window runs past the end
        const uint64_t off = N - 5000, bytes = 5000, a0 = off / 4096 * 4096;
        r.start(1, off, bytes, dst, true);
        CHECK(r.wait(1) && std::memcmp(dst + (off - a0), data.data() + off, bytes) == 0, "tail past end of file");
    }
    {   // a destination O_DIRECT refuses (not aligned): buffered fallback
        uint8_t* un = dst + 8;
        const uint64_t off = 8192;
        r.start(2, off, 100000, un, true);
        CHECK(r.wait(2) && std::memcmp(un, data.data() + off, 100000) == 0, "unaligned destination falls back to buffered");
    }
    {   // low priority + promote, several jobs in flight
        std::vector<uint8_t*> bufs;
        for (int j = 0; j < 6; ++j) { void* p = nullptr; if (posix_memalign(&p, 4096, blob + 4096)) return 2; bufs.push_back((uint8_t*) p); }
        for (int j = 0; j < 6; ++j) r.start(10 + j, (uint64_t) j * 4096 * 1000, blob, bufs[(size_t) j], j % 2 == 0);
        r.promote(11);
        bool all = true;
        for (int j = 0; j < 6; ++j) all &= r.wait(10 + j) && std::memcmp(bufs[(size_t) j], data.data() + (uint64_t) j * 4096 * 1000, blob) == 0;
        CHECK(all, "6 jobs, mixed priority, one promoted");
        CHECK(r.reads(0) > 0 && r.reads(1) > 0, ("both mirrors used: " + std::to_string(r.reads(0)) + " / " + std::to_string(r.reads(1)) + " pieces").c_str());
    }
    {   // a missing file must fail the job, not hang
        DiskReader bad;
        bad.open({"/tmp/drtest/missing.bin"}, {1.0}, 1);
        bad.start(0, 0, 4096, dst, true);
        CHECK(!bad.wait(0), "missing file: job reports failure");
    }
    r.close();
    std::printf("%s (%d failures)\n", fails ? "FAILED" : "ALL OK", fails);
    return fails != 0;
}

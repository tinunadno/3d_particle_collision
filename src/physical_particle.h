#pragma once

#include "particle_system.h"
#include "basic_thread_pool.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

constexpr float impulse_ = 1.0f;

template<typename NumericT>
struct PhysicalParticleSystem {
    mrc::ParticleSystem<NumericT> ps;
    std::vector<sc::utils::Vec<NumericT, 3>> speed;
    std::vector<sc::utils::Vec<NumericT, 3>> acceleration;
    std::vector<NumericT> weight;

    void resize(std::size_t size) {
        ps.positions.resize(size);
        ps.colors.resize(size);
        ps.sizes.resize(size);
        ps.enableRender.resize(size);
        speed.resize(size);
        acceleration.resize(size);
        weight.resize(size);
    }
};

namespace detail {

struct SpatialGrid {
    static constexpr uint64_t EMPTY_KEY = ~uint64_t(0);

    struct Slot {
        uint64_t key = EMPTY_KEY;
        uint32_t start = 0;
        uint32_t count = 0;
    };

    float invCell = 1.0f;
    uint32_t mask = 0;
    std::vector<Slot>     table;
    std::vector<uint32_t> list;      // particle indices, grouped by cell
    std::vector<uint32_t> slotBuf;   // per-insert: which slot it went to

    static uint32_t nextPow2(uint32_t v) {
        --v; v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; return v + 1;
    }

    static uint64_t cellKey(int32_t cx, int32_t cy, int32_t cz) {
        constexpr int32_t OFF = 1 << 20;
        return (uint64_t((cx + OFF) & 0x1FFFFF) << 42)
             | (uint64_t((cy + OFF) & 0x1FFFFF) << 21)
             |  uint64_t((cz + OFF) & 0x1FFFFF);
    }

    uint32_t probe(uint64_t k) const {
        uint32_t s = uint32_t((k * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
        while (table[s].key != EMPTY_KEY && table[s].key != k)
            s = (s + 1) & mask;
        return s;
    }

    template<typename NumericT>
    void build(const std::vector<sc::utils::Vec<NumericT, 3>>& positions,
               const uint32_t* ids, uint32_t count, float cellSize)
    {
        invCell = 1.0f / cellSize;
        uint32_t sz = nextPow2(std::max(count * 2u, 64u));
        mask = sz - 1;
        table.assign(sz, {EMPTY_KEY, 0, 0});
        slotBuf.resize(count);

        for (uint32_t si = 0; si < count; ++si) {
            const auto& p = positions[ids[si]];
            uint64_t k = cellKey(
                static_cast<int32_t>(std::floor(p[0] * invCell)),
                static_cast<int32_t>(std::floor(p[1] * invCell)),
                static_cast<int32_t>(std::floor(p[2] * invCell)));
            uint32_t s = probe(k);
            if (table[s].key == EMPTY_KEY) table[s].key = k;
            ++table[s].count;
            slotBuf[si] = s;
        }

        uint32_t offset = 0;
        for (uint32_t i = 0; i < sz; ++i) {
            if (table[i].key != EMPTY_KEY) {
                table[i].start = offset;
                offset += table[i].count;
                table[i].count = 0;
            }
        }

        list.resize(count);
        for (uint32_t si = 0; si < count; ++si) {
            auto& sl = table[slotBuf[si]];
            list[sl.start + sl.count++] = ids[si];
        }
    }

    template<typename Func>
    void forEachNeighbor(float x, float y, float z, Func&& func) const {
        int32_t cx = static_cast<int32_t>(std::floor(x * invCell));
        int32_t cy = static_cast<int32_t>(std::floor(y * invCell));
        int32_t cz = static_cast<int32_t>(std::floor(z * invCell));
        for (int32_t dz = -1; dz <= 1; ++dz)
            for (int32_t dy = -1; dy <= 1; ++dy)
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    uint32_t s = probe(cellKey(cx + dx, cy + dy, cz + dz));
                    if (table[s].key == EMPTY_KEY) continue;
                    const uint32_t beg = table[s].start;
                    const uint32_t end = beg + table[s].count;
                    for (uint32_t p = beg; p < end; ++p)
                        func(list[p]);
                }
    }
};

} // namespace detail

template<typename NumericT>
void iterate(PhysicalParticleSystem<NumericT>& particles, NumericT dt) {
    static ThreadPool pool;
    static detail::SpatialGrid grid;
    static std::vector<uint32_t> smallIds, largeIds;

    const std::size_t N = particles.speed.size();
    if (N == 0) return;

    pool.runConcurrentTask([&](std::size_t tid, std::size_t tc) {
        const std::size_t chunk = (N + tc - 1) / tc;
        const std::size_t lo = std::min(chunk * tid, N);
        const std::size_t hi = std::min(lo + chunk, N);
        for (std::size_t i = lo; i < hi; ++i) {
            particles.speed[i] = particles.speed[i] + particles.acceleration[i] * dt;
            particles.ps.positions[i] = particles.ps.positions[i] + particles.speed[i] * dt;
        }
    });
    float rMin = std::numeric_limits<float>::max();
    float rMax = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        const float r = particles.ps.sizes[i];
        if (r > 0) rMin = std::min(rMin, r);
        rMax = std::max(rMax, r);
    }
    if (rMin > rMax) rMin = rMax;

    const bool multiScale = (rMax > rMin * 4.0f);
    const float cellSize  = 2.0f * rMin;

    smallIds.clear();
    largeIds.clear();
    smallIds.reserve(N);
    for (uint32_t i = 0; i < static_cast<uint32_t>(N); ++i) {
        if (multiScale && particles.ps.sizes[i] > rMin * 4.0f)
            largeIds.push_back(i);
        else
            smallIds.push_back(i);
    }

    grid.build(particles.ps.positions, smallIds.data(),
               static_cast<uint32_t>(smallIds.size()), cellSize);

    const uint32_t nLarge = static_cast<uint32_t>(largeIds.size());
    pool.runConcurrentTask([&](std::size_t tid, std::size_t tc) {
        const std::size_t chunk = (N + tc - 1) / tc;
        const std::size_t lo = std::min(chunk * tid, N);
        const std::size_t hi = std::min(lo + chunk, N);

        for (std::size_t i = lo; i < hi; ++i) {
            float px = particles.ps.positions[i][0];
            float py = particles.ps.positions[i][1];
            float pz = particles.ps.positions[i][2];
            float vx = particles.speed[i][0];
            float vy = particles.speed[i][1];
            float vz = particles.speed[i][2];
            const float ri    = particles.ps.sizes[i];
            const float wi    = particles.weight[i];
            const float invMi = 1.0f / wi;

            auto collide = [&](uint32_t j) {
                if (static_cast<std::size_t>(j) == i) return;

                const float dx = px - particles.ps.positions[j][0];
                const float dy = py - particles.ps.positions[j][1];
                const float dz = pz - particles.ps.positions[j][2];
                const float d2 = dx * dx + dy * dy + dz * dz;
                const float minDist = ri + particles.ps.sizes[j];

                if (d2 >= minDist * minDist) return;   // early-out (no sqrt)

                const float dist = std::sqrt(d2);
                if (dist < 1e-10f) return;              // degenerate overlap

                const float invDist = 1.0f / dist;
                const float nx = dx * invDist;
                const float ny = dy * invDist;
                const float nz = dz * invDist;

                const float relVn = (vx - particles.speed[j][0]) * nx
                                  + (vy - particles.speed[j][1]) * ny
                                  + (vz - particles.speed[j][2]) * nz;
                if (relVn > 0) return;                  // separating

                const float wj    = particles.weight[j];
                const float invMj = 1.0f / wj;
                const float imp   = -(1.0f + impulse_) * relVn / (invMi + invMj);

                const float s1 = imp * invMi;
                vx += s1 * nx;
                vy += s1 * ny;
                vz += s1 * nz;

                const float s2 = (minDist - dist) * 0.5f * (wj / (wi + wj));
                px += s2 * nx;
                py += s2 * ny;
                pz += s2 * nz;
            };

            const bool isLarge = multiScale && ri > rMin * 4.0f;
            if (!isLarge) {
                grid.forEachNeighbor(px, py, pz, collide);
                for (uint32_t l = 0; l < nLarge; ++l)
                    collide(largeIds[l]);
            } else {
                for (std::size_t j = 0; j < N; ++j)
                    collide(static_cast<uint32_t>(j));
            }

            particles.ps.positions[i] = sc::utils::Vec<NumericT, 3>{px, py, pz};
            particles.speed[i] = sc::utils::Vec<NumericT, 3>{vx, vy, vz};
        }
    });
}

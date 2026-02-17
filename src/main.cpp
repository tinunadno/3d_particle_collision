#include <random>

#include "physical_particle.h"
#include "utils/vec.h"
#include "camera/camera.h"
#include "main_pipeline.h"

struct Roi3{
    sc::utils::Vec<float, 3> center;
    sc::utils::Vec<float, 3> rotation;
    sc::utils::Vec<float, 3> sizes;
    // width height depth
};

void generateParticlesByRoi(
    PhysicalParticleSystem<float>& particles,
    const Roi3& roi,
    float particleSize,
    float weight,
    const sc::utils::Vec<float, 3> color = {},
    const sc::utils::Vec<float, 3> speedOffset = {}
) {
    const float diameter = 2.0f * particleSize;
    const auto partX = std::max<std::size_t>(1, static_cast<std::size_t>(roi.sizes[0] / diameter) + 1);
    const auto partY = std::max<std::size_t>(1, static_cast<std::size_t>(roi.sizes[1] / diameter) + 1);
    const auto partZ = std::max<std::size_t>(1, static_cast<std::size_t>(roi.sizes[2] / diameter) + 1);
    const std::size_t totalParticles = partX * partY * partZ;

    const std::size_t oldSize = particles.ps.count();
    const std::size_t newSize = oldSize + totalParticles;

    particles.resize(newSize);

    std::size_t ptr = oldSize;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform01(0.f, 1.f);

    sc::utils::Vec<float, 3> basePos{
        -static_cast<float>(partX - 1) * diameter / 2.f,
        -static_cast<float>(partY - 1) * diameter / 2.f,
        -static_cast<float>(partZ - 1) * diameter / 2.f
    };
    sc::utils::Vec<float, 3> step{diameter, diameter, diameter};

    for (std::size_t x = 0; x < partX; x++) {
        for (std::size_t y = 0; y < partY; y++) {
            for (std::size_t z = 0; z < partZ; z++) {
                float colorFlick = .3f;
                particles.speed[ptr] = speedOffset;
                particles.weight[ptr] = weight;
                particles.ps.sizes[ptr] = particleSize;
                particles.ps.colors[ptr] = color +
                    sc::utils::Vec<float, 3>{uniform01(gen), uniform01(gen), uniform01(gen)} * colorFlick;
                particles.ps.enableRender[ptr] = true;
                sc::utils::Vec<float, 3> pos =
                    basePos + step * sc::utils::Vec<float, 3>(x, y, z);
                pos = sc::utils::rotateEuler(pos, roi.rotation);
                particles.acceleration[ptr] = sc::utils::Vec<float, 3>{};
                particles.ps.positions[ptr] = pos + roi.center;
                ptr++;
            }
        }
    }
}

void addHeavyOne(PhysicalParticleSystem<float>& particles) {
    particles.speed.emplace_back(0.f, 0.f, 3.f);
    particles.weight.emplace_back(100.f);
    particles.ps.sizes.emplace_back(.5f);
    particles.ps.colors.emplace_back(1.f, 0.f, 0.f);
    particles.ps.enableRender.emplace_back(true);
    particles.ps.positions.emplace_back(0.f, 0.f, -2.f);
    particles.acceleration.emplace_back(0.f, 0.f, 0.f);
}

int main() {

    sc::Camera<float, sc::VecArray> camera;
    camera.pos()[2] = 2.0f;
    camera.setLen(0.3);
    camera.setRes(sc::utils::Vec<float, 2>{1000, 800});

    PhysicalParticleSystem<float> particles;
    mrc::makeCircleBillboard(particles.ps, 8);

    Roi3 roi{
        sc::utils::Vec<float, 3>{.0f, .0f, .0f},
        sc::utils::Vec<float, 3>{.0f, .0f, .0f},
        sc::utils::Vec<float, 3>{1., 1., 1.}
    };

    generateParticlesByRoi(particles, roi, .02f, .0025f,
        sc::utils::Vec<float, 3>{.8, .9, 1.},
        sc::utils::Vec<float, 3>{0, -.1, 0.}
        );
    roi.center[0] -= 2.f;
    roi.rotation = sc::utils::Vec<float, 3>{1.f, 1.f, 0.f};
    generateParticlesByRoi(particles, roi, .02f, .004f,
        sc::utils::Vec<float, 3>{.8, .0, 1.},
        sc::utils::Vec<float, 3>{3., .3, 0}
        );
    addHeavyOne(particles);

    float step = 0.001f;
    bool pause = false;

    auto efmu = [&particles, &step, &pause](std::size_t, std::size_t) {
        if (!pause) iterate(particles, step);
    };

    std::vector<std::pair<std::vector<int>, std::function<void()>>> customKeyHandlers = {
        {{GLFW_KEY_Q}, [&step](){ step *= 1.1f; },},
        {{GLFW_KEY_E}, [&step](){ step *= .9f; },},
        {{GLFW_KEY_LEFT_ALT, GLFW_KEY_P}, [&pause](){ pause = !pause; },},
    };

    mrc::initMrcRender(camera, { }, { },
        efmu, { }, customKeyHandlers,
        sc::utils::Vec<int, 2>{-1,-1}, 60, {}, &particles.ps);

    return 0;
}
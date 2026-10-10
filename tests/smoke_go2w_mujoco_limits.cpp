// Native headless model/config validation; no robot SDK or hardware connection.
#include <mujoco/mujoco.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    char error[1024] = {};
    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!model) { std::cerr << error << '\n'; return 1; }
    mjData* data = mj_makeData(model);
    int result = 0;
    try {
        if (model->nu != 16) throw std::runtime_error("Expected 16 Go2W actuators");
        for (int i = 0; i < model->nu; ++i) {
            const std::string name = mj_id2name(model, mjOBJ_ACTUATOR, i);
            const double limit = name.find("calf") != std::string::npos ? 35.0 : 23.5;
            if (!model->actuator_ctrllimited[i] ||
                std::abs(model->actuator_ctrlrange[2*i] + limit) > 1e-6 ||
                std::abs(model->actuator_ctrlrange[2*i+1] - limit) > 1e-6)
                throw std::runtime_error("Wrong compiled limit: " + name);
        }
        for (const double sign : {-1.0, 1.0}) {
            for (int i = 0; i < model->nu; ++i) data->ctrl[i] = sign * 1000.0;
            mj_forward(model, data);
            for (int i = 0; i < model->nu; ++i) {
                const double expected = model->actuator_ctrlrange[2*i + (sign > 0 ? 1 : 0)];
                if (std::abs(data->actuator_force[i] - expected) > 1e-6)
                    throw std::runtime_error("Actuator force clipping mismatch");
            }
        }
        for (int i = 0; i < model->nu; ++i) data->ctrl[i] = 0.0;
        for (int step = 0; step < 100; ++step) mj_step(model, data);
        for (int i = 0; i < model->nq; ++i)
            if (!std::isfinite(data->qpos[i])) throw std::runtime_error("Nonfinite qpos");
        for (int i = 0; i < model->nv; ++i)
            if (!std::isfinite(data->qvel[i])) throw std::runtime_error("Nonfinite qvel");
        std::cout << "GO2W_MUJOCO_LIMITS_OK actuators=16 steps=100\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        result = 1;
    }
    mj_deleteData(data);
    mj_deleteModel(model);
    return result;
}

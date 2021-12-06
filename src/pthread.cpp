#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <hdist/pthread.hpp>

template<typename ...Args>
void UNUSED(Args &&... args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

int main(int argc, char **argv) {
    int c;
    int gui_flag = 0;
    int iter_u = 1000;
    int set_size = 300;
    int set_stemp = 100;
    int set_btemp = 36;
    int set_tnum = 4;
    opterr = 0;
    while ((c = getopt (argc, argv, "gi:s:t:")) != -1){
        switch (c)
        {
            case 'g':
                gui_flag = 1;
                break;
            case 'i':
                iter_u = atoi(optarg);
                break;
            case 's':
                set_size = atoi(optarg);
                break;
            case 't':
                set_tnum = atoi(optarg);
                break;
            case '?':
                break;
            default:
                break;
        }
    }
    bool first = true;
    bool finished = false;
    static hdist::State current_state, last_state;
    static std::chrono::high_resolution_clock::time_point begin, end;
    static const char* algo_list[2] = { "jacobi", "sor" };
    current_state.room_size = set_size;
    last_state.room_size = set_size;
    current_state.source_temp = set_stemp;
    last_state.source_temp = set_stemp;
    current_state.border_temp = set_btemp;
    last_state.border_temp = set_btemp;
    auto grid = hdist::Grid{
            static_cast<size_t>(current_state.room_size),
            current_state.border_temp,
            current_state.source_temp,
            static_cast<size_t>(current_state.source_x),
            static_cast<size_t>(current_state.source_y)};
    static int iter = 0;
    if(!gui_flag){
        while(1){
            if (first) {
                first = false;
                finished = false;
                begin = std::chrono::high_resolution_clock::now();
            }

            if (!finished) {
                finished = hdist::calculate(current_state, grid, set_tnum) || iter++ == iter_u;
                if (finished) end = std::chrono::high_resolution_clock::now();
            } else {
                if(iter>=iter_u) {printf("iteration finished in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());}
                else printf("stabilized in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
                break;
            }
        }

    }
    else
    {
        graphic::GraphicContext context{"Sequential"};
        context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
            auto io = ImGui::GetIO();
            ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::Begin("Sequential", nullptr,
                        ImGuiWindowFlags_NoMove
                        | ImGuiWindowFlags_NoCollapse
                        | ImGuiWindowFlags_NoTitleBar
                        | ImGuiWindowFlags_NoResize);
            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);
            ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
            ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
            ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 100, "%f");
            ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 100, "%f");
            ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
            ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
            //ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

            if (current_state.algo == hdist::Algorithm::Sor) {
                ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
            }

            if (current_state.room_size != last_state.room_size) { // not yet converge
                grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y)};
                first = true;
            }

            if (current_state != last_state) {
                last_state = current_state;
                finished = false;
            }

            if (first) {
                first = false;
                finished = false;
                begin = std::chrono::high_resolution_clock::now();
            }

            if (!finished) {
                finished = hdist::calculate(current_state, grid, set_tnum) || iter++ == iter_u;
                if (finished) end = std::chrono::high_resolution_clock::now();
            } else {
                if(iter>=iter_u) {ImGui::Text("iteration finished in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());}
                else ImGui::Text("stabilized in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
            }

            const ImVec2 p = ImGui::GetCursorScreenPos();
            float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
            for (size_t i = 0; i < current_state.room_size; ++i) {
                for (size_t j = 0; j < current_state.room_size; ++j) {
                    auto temp = grid[{i, j}];
                    auto color = temp_to_color(temp);
                    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                    y += current_state.block_size;
                }
                x += current_state.block_size;
                y = p.y + current_state.block_size;
            }
            ImGui::End();
        });
    }
}

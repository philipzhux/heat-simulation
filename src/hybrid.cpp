#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hybrid.hpp>
#include <cmath>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#define TERMINATE 2
template<typename ...Args>
void UNUSED(Args &&... args [[maybe_unused]]) {}

ImColor temp_to_color(double temp) {
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

int main(int argc, char **argv) {
    UNUSED(argc, argv);
    MPI_Init(&argc, &argv);
    int rank, proc;
    int c;
    int gui_flag = 0;
    int iter_u = 1000;
    int set_size = 100;
    int set_stemp = 100;
    int set_btemp = 36;
    opterr = 0;
    while ((c = getopt (argc, argv, "gi:s:t:b:")) != -1){
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
                set_stemp = atoi(optarg);
                break;
            case 'b':
                set_btemp = atoi(optarg);
                break;
            case '?':
                break;
            default:
                break;
        }
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int res = MPI_Comm_size(MPI_COMM_WORLD, &proc);
    if (MPI_SUCCESS != res) {
        throw std::runtime_error("failed to get MPI world size");
    }
    do_it(proc,rank,gui_flag,iter_u,set_size,set_stemp,set_btemp);
    MPI_Finalize();
    
}


void do_it(const int & proc, const int & rank, const int & gui_flag, const int & iter_u, const int & set_size,
const int & set_stemp, const int & set_btemp) {
    int first = true;
    int finished = false;
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
            static_cast<size_t>(current_state.source_y),
            proc, rank};
    if(gui_flag){
        if(0 == rank){
            gui(current_state, last_state, begin, end, algo_list,
             first, finished, grid, proc, rank, iter_u);
        }
        else {
            int iter = 0;
            while(true){
                MPI_Bcast(&first, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD); //whether config has been changed
                iter++;
                if(first==TERMINATE) {
                    break;
                }
                if(first){
                    first = false;
                    finished = false;
                    MPI_Bcast(&current_state, sizeof(hdist::State), MPI_BYTE, 0, MPI_COMM_WORLD); //get the new config
                    grid = hdist::Grid{
                        static_cast<size_t>(current_state.room_size),
                        current_state.border_temp,
                        current_state.source_temp,
                        static_cast<size_t>(current_state.source_x),
                        static_cast<size_t>(current_state.source_y),
                        proc,rank};
                }

                if (!finished) {
                    finished = hdist::calculate(current_state, grid);
                    MPI_Allreduce(MPI_IN_PLACE,&finished,sizeof(int),MPI_BYTE,MPI_LAND,MPI_COMM_WORLD);
                } else {
                    // break;
                }

            }
        }
    }
    else {
        int iter = 0;
        while(true){
            if (first) {
                first = false;
                finished = false;
                if(0 == rank) begin = std::chrono::high_resolution_clock::now();
            }
            if (!finished) {
                finished = hdist::calculate(current_state, grid) || iter++ == iter_u;
                MPI_Allreduce(MPI_IN_PLACE,&finished,sizeof(int),MPI_BYTE,MPI_LAND,MPI_COMM_WORLD);
                if (finished && 0 == rank) end = std::chrono::high_resolution_clock::now();
            } else {
                if(0 == rank && iter>=iter_u) {printf("iteration finished in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());}
                else if(0 == rank && finished && iter<iter_u) printf("stabilized in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
                break;
            }
        }
    }
}


void gui(hdist::State& current_state, hdist::State& last_state,
            std::chrono::high_resolution_clock::time_point& begin,
            std::chrono::high_resolution_clock::time_point& end,
            const char *const * algo_list, int& first, int& finished,
            hdist::Grid& grid, const int & proc, const int & rank, const int & iter_u){
    graphic::GraphicContext context{"Assignment 4"};
    bool to_quit = false;
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *) {
        static int iter = 0;
        if(to_quit) {
            printf("Quitting from GUI in 3 seconds\n");
            sleep(3);
            context->quit();
            if(iter>=iter_u) {printf("iteration finished in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());}
            else printf("stabilized in %ld ns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        }
        
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 4", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragInt("Room Size", &current_state.room_size, 10, 200, 1600, "%d");
        ImGui::DragFloat("Block Size", &current_state.block_size, 0.01, 0.1, 10, "%f");
        ImGui::DragFloat("Source Temp", &current_state.source_temp, 0.1, 0, 500, "%f");
        ImGui::DragFloat("Border Temp", &current_state.border_temp, 0.1, 0, 500, "%f");
        ImGui::DragInt("Source X", &current_state.source_x, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragInt("Source Y", &current_state.source_y, 1, 1, current_state.room_size - 2, "%d");
        ImGui::DragFloat("Tolerance", &current_state.tolerance, 0.01, 0.01, 1, "%f");
        ImGui::ListBox("Algorithm", reinterpret_cast<int *>(&current_state.algo), algo_list, 2);

        if (current_state.algo == hdist::Algorithm::Sor) {
            ImGui::DragFloat("Sor Constant", &current_state.sor_constant, 0.01, 0.0, 20.0, "%f");
        }

        if (current_state != last_state) {
            last_state = current_state;
            grid = hdist::Grid{
                    static_cast<size_t>(current_state.room_size),
                    current_state.border_temp,
                    current_state.source_temp,
                    static_cast<size_t>(current_state.source_x),
                    static_cast<size_t>(current_state.source_y),
                    proc,rank};
            first = true;
            finished = false;
        }
            MPI_Bcast(&first, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
            if (first) {
                first = false;
                finished = false;
                MPI_Bcast(&current_state, sizeof(hdist::State), MPI_BYTE, 0, MPI_COMM_WORLD);
                begin = std::chrono::high_resolution_clock::now();
            }

            if (!finished) {
                finished = hdist::calculate(current_state, grid);
                MPI_Allreduce(MPI_IN_PLACE,&finished,sizeof(int),MPI_BYTE,MPI_LAND,MPI_COMM_WORLD);
                finished = finished || (iter++ == iter_u);
                if (finished) end = std::chrono::high_resolution_clock::now();
            } else {
                if(iter<iter_u) {ImGui::Text("stabilized in %ld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());}
                else ImGui::Text("iter_u reached in %ld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
                first = TERMINATE;
                MPI_Bcast(&first, sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
                to_quit = true;

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
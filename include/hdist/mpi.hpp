#pragma once

#include <vector>
#include <mpi.h>
#include <cmath>

namespace hdist {

    enum class Algorithm : int {
        Jacobi = 0,
        Sor = 1
    };

    struct State {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const = default;
    };
    
    struct Alt {
    };

    constexpr static inline Alt alt{};

    struct Grid {
        std::vector<double> data0, data1;
        size_t current_buffer = 0;
        int my_start = 0;
        size_t length;
        int rank,proc;
        int* recvcounts;
        int* displs;
        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y, int proc, int rank)
                : data0(size * size), data1(size * size), length(size), rank(rank), proc(proc) {
            recvcounts = new int[proc];
            displs = new int[proc];
            int count  = 0;
            recvcounts[0] = getLength(length,proc,0)*length;
            displs[0] = 0;
            if(rank) my_start += getLength(length,proc, 0);
            for(int i=1;i<proc;i++){
                count += recvcounts[i-1];
                recvcounts[i] = getLength(length,proc,i)*length;
                displs[i] = count;
                if(i<rank) my_start += getLength(length,proc,i);
            }
            for (size_t i = 0; i < length; ++i) {
                for (size_t j = 0; j < length; ++j) {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1) {
                        this->operator[]({i, j}) = border_temp;
                    } else if (i == x && j == y) {
                        this->operator[]({i, j}) = source_temp;
                    } else {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }
        int get_rank() {
            return rank;
        }
        int get_start() {
            return  my_start;
        }
        int get_length() {
            return getLength(length, proc,rank);
        }
        static inline int getLength(int total, int proc,int r) {
            if(total<proc) return r<total;
            return (total-r)/proc + ((total-r)%proc > 0);
        }
        std::vector<double> &get_current_buffer() {
            return data0;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return data1[std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer() {
            double* recv_buf = data0.data();
            double* send_buf = data1.data()+displs[rank];
            MPI_Allgatherv(send_buf, recvcounts[rank], MPI_DOUBLE, recv_buf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        }
    };

    struct UpdateResult {
        int stable;
        double temp;
    };

    UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state) {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1) {
            result.temp = state.border_temp;
        } else if (i == state.source_x && j == state.source_y) {
            result.temp = state.source_temp;
        } else {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            switch (state.algo) {
                case Algorithm::Jacobi:
                    result.temp = 0.25 * sum;
                    break;
                case Algorithm::Sor:
                    result.temp = grid[{i, j}] + (1.0 / state.sor_constant) * (sum - 4.0 * grid[{i, j}]);
                    break;
            }
        }
        result.stable = std::fabs(grid[{i, j}] - result.temp) < state.tolerance;
        return result;
    }

    int calculate(const State &state, Grid &grid) {
        int stabilized = true;
        size_t my_start = grid.get_start();
        size_t my_length = grid.get_length();
        switch (state.algo) {
            case Algorithm::Jacobi:
                for (size_t i = my_start; i < my_start+my_length; ++i) {
                    for (size_t j = 0; j < state.room_size; ++j) {
                        auto result = update_single(i, j, grid, state);
                        stabilized &= result.stable;
                        grid[{alt, i, j}] = result.temp;
                    }
                }
                grid.switch_buffer();
                break;
            case Algorithm::Sor:
                for (auto k : {0, 1}) {
                    for (size_t i = my_start; i < my_start+my_length; ++i) {
                        for (size_t j = 0; j < state.room_size; j++) {
                            if (k == ((i + j) & 1)) {
                                auto result = update_single(i, j, grid, state);
                                stabilized &= result.stable;
                                grid[{alt, i, j}] = result.temp;
                            } else {
                                grid[{alt, i, j}] = grid[{i, j}];
                            }
                        }
                    }
                    grid.switch_buffer();
                }
        }
        return stabilized;
    };


} // namespace hdist




void gui(hdist::State& current_state, hdist::State& last_state,
            std::chrono::high_resolution_clock::time_point& begin,
            std::chrono::high_resolution_clock::time_point& end,
            const char *const * algo_list, int& first, int& finished,
            hdist::Grid& grid, const int & proc, const int & rank, const int & iter_u);

void do_it(const int & proc, const int & rank, const int & gui_flag, const int & iter_u, const int & set_size,
const int & set_stemp, const int & set_btemp);
#pragma once

#include <vector>
#include <pthread.h>

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
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
                : data0(size * size), data1(size * size), length(size) {
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

        static inline int getLength(int total, int proc,int r) {
            if(total<proc) return r<total;
            return (total-r)/proc + ((total-r)%proc > 0);
        }

        static inline int getStart(int total, int proc,int r) {
            int start = 0;
            for(int i=0;i<r-1;i++) start+=Grid::getLength(total,proc,i);
            return start;
        }
        std::vector<double> &get_current_buffer() {
            if (current_buffer == 0) return data0;
            return data1;
        }

        double &operator[](std::pair<size_t, size_t> index) {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index) {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[
                    std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer() {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult {
        bool stable;
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

    typedef UpdateResult (*__US)(size_t, size_t, Grid&, const State&);
    struct Para {
        int start;
        int length;
        const State* state_ptr;
        __US update_single;
        Grid* grid_ptr;
        bool* stable_temp_ptr;
        int k;
    };
    
    bool calculate(const State &state, Grid &grid, int thread_num) {
        bool stabilized = true;
        Para* para = new Para[thread_num];
        pthread_t* threads = new pthread_t[thread_num];
        bool* statle_temp_ptr = new bool[thread_num];
        //printf("point 1\n");
        switch (state.algo) {
            case Algorithm::Jacobi:
                for(int t=0;t<thread_num;t++){
                    statle_temp_ptr[t] = true;
                    (para+t)->grid_ptr = &grid;
                     (para+t)->state_ptr = &state;
                    (para+t)->stable_temp_ptr = statle_temp_ptr+t;
                    (para+t)->length = Grid::getLength(state.room_size,thread_num,t);
                    (para+t)->start = Grid::getStart(state.room_size,thread_num,t);
                    (para+t)->update_single = &update_single;
                    pthread_create(threads+t,NULL,[](void* ptr)-> void * {
                    Para* para = static_cast<Para*>(ptr);
                    //printf("point 2\n");
                    for (size_t i = para->start; i < para->start+para->length; ++i) {
                        for (size_t j = 0; j < para->state_ptr->room_size; ++j) {
                            auto result = para->update_single(i, j, *(para->grid_ptr), *(para->state_ptr));
                            //printf("point 3\n");
                            *(para->stable_temp_ptr) &= result.stable;
                            para->grid_ptr->operator[]({alt, i, j}) = result.temp;
                        }
                    }
                    return NULL;
                },para+t);}
                grid.switch_buffer();
                break;
            case Algorithm::Sor:
                for (auto k : {0, 1}) {
                    for(int t=0;t<thread_num;t++){
                    (para+t)->grid_ptr = &grid;
                    (para+t)->stable_temp_ptr = statle_temp_ptr+t;
                    (para+t)->length = Grid::getLength(state.room_size,thread_num,t);
                    (para+t)->start = Grid::getStart(state.room_size,thread_num,t);
                    (para+t)->update_single = &update_single;
                    (para+t)->k = k;
                    pthread_create(threads+t,NULL,[](void* ptr)-> void * {
                    Para* para = static_cast<Para*>(ptr);
                    for (size_t i = para->start; i < para->start+para->length; ++i) {
                        for (size_t j = 0; j < para->state_ptr->room_size; ++j){
                            if (para->k == ((i + j) & 1)) {
                                auto result = para->update_single(i, j, *(para->grid_ptr), *(para->state_ptr));
                                *(para->stable_temp_ptr) &= result.stable;
                                para->grid_ptr->operator[]({alt, i, j}) = result.temp;
                            } else {
                                para->grid_ptr->operator[]({alt, i, j}) = para->grid_ptr->operator[]({i, j});
                            }
                        }
                    }
                    return NULL;
                },para+t);}
                    grid.switch_buffer();
                }
        }
        for(int i=0;i<thread_num;i++) {
            //printf("point 4\n");
            pthread_join(threads[i],NULL);
            //printf("point 5\n");
            stabilized &= statle_temp_ptr[i];
            //printf("point 6\n");
        }
        delete[] para;
        delete[] threads;
        delete[] statle_temp_ptr;
        return stabilized;
    };


} // namespace hdist
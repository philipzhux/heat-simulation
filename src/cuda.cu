enum class Algo : int {
    Jacobi = 0,
    Sor = 1
};

struct DGrid
{
    int room_size;
    int source_x;
    int source_y;
    float source_temp;
    float border_temp;
    float tolerance;
    float sorc;
    int algo;
    double *d0;
    double *d1;
    int k;
    int curr_buf;
    __device__ __host__ DGrid(int room_size, int source_x, int source_y, float source_temp,
         float border_temp, float tolerance, float sorc, int algo, double *d0, double *d1, int k,
        int curr_buf):
        room_size{room_size},source_x{source_x},source_y{source_y},source_temp{source_temp},
        border_temp{border_temp},tolerance{tolerance},sorc{sorc},algo{algo},d0{d0},d1{d1},k{k},curr_buff{curr_buff} {

     }
    
     __device__ __host__ DGrid(const DGrid &grid) : room_size{grid.room_size},source_x{grid.source_x},source_y{grid.source_y},source_temp{grid.source_temp},
     border_temp{grid.border_temp},tolerance{grid.tolerance},sorc{grid.sorc},algo{grid.algo},d0{grid.d0},d1{grid.d1},k{grid.k},curr_buff{grid.curr_buff} {}
    __device__ double get_copy(size_t i, size_t j) {
        if (current_buffer) return data1[i * room_size + j];
        else return data0[i * room_size + j];
    }
    __device__ void update_alt(size_t i, size_t j, double temp) {
        if (current_buffer) d0[i * room_size + j] = temp;
        else d1[i * room_size + j] = temp;
    }
    __device__ void update_single(size_t i, size_t j) {
        double temp;
        if (i == 0 || j == 0 || i == room_size - 1 || j == room_size - 1) {
            temp = border_temp;
        } else if (i == source_x && j == source_y) {
            temp = source_temp;
        } else {
            auto sum = get_copy(i+1,j) + get_copy(i-1,j) + get_copy(i,j+1)+ get_copy(i,j-1);
            switch (algo) {
                case Algo::Jacobi:
                    temp = 0.25 * sum;
                    update_alt(i,j,temp);
                    break;
                case Algo::Sor:
                    if (k == ((i + j) & 1)){
                        temp = get_copy(i,j) + (1.0 / sor_constant) * (sum - 4.0 * get_copy(i,j));
                        update_alt(i,j,temp);
                    }
                    else {
                        update_alt(i,j,get_copy(i,j));
                    }
                    break;
            }
        }
    }
    
};
void alloc_init(int size, double *d0, double *d1, double **d0_d, double **d1_d){
    cudaMalloc((void**)d0_d, sizeof(double) * size * size);
    cudaMalloc((void**)d1_d, sizeof(double) * size * size);
    cudaMemcpy(*d0_d, d0, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(*d1_d, d1, sizeof(double) * size * size, cudaMemcpyHostToDevice);
}

void fetch_data(int size, double *d0, double *d1, double *d0_d, double *d1_d){
    cudaMemcpy(d0, d0_d, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d1, d1_d, sizeof(double) * size * size, cudaMemcpyDeviceToHost);
}


__global__
void dev_cal(DGrid grid) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (size_t i = index; i < grid.room_size; i+=stride) {
        for (size_t j = 0; j < grid.room_size; ++j) {
            grid.update_single(i, j);
        }
    }
    
}

void clean_alloc(double* d0, double* d1){
    cudaFree(d0);
    cudaFree(d1);
}

void host_cal(int x, int y, int room_size, int source_x, int source_y, 
    float source_temp, float border_temp, float tolerance, float sorc, int algo, 
    double *d0_d, double *d1_d, int& curr_buf){
        if(!algo){
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                tolerance, sorc, algo, d0_d, d1_d, k, curr_buf});
           curr_buf = !curr_buf;
        }
        else{
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                tolerance, sorc, algo, d0_d, d1_d, k, curr_buf});
           
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                    tolerance, sorc, algo, d0_d, d1_d, !k, !curr_buf});
        }


    }
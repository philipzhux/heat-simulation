// enum class Algo : int {
//     Jacobi = 0,
//     Sor = 1
// };
#define JACOBI 0
#define SOR 1
#include <stdio.h>
__device__ int curr_buf_dev = 0;
int curr_buf = 0;
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
    __device__ __host__ DGrid(int room_size, int source_x, int source_y, float source_temp,
         float border_temp, float tolerance, float sorc, int algo, double *d0, double *d1):
        room_size{room_size},source_x{source_x},source_y{source_y},source_temp{source_temp},
        border_temp{border_temp},tolerance{tolerance},sorc{sorc},algo{algo},d0{d0},d1{d1} {
            
     }
    
     __device__ __host__ DGrid(const DGrid &grid) : room_size{grid.room_size},source_x{grid.source_x},source_y{grid.source_y},source_temp{grid.source_temp},
     border_temp{grid.border_temp},tolerance{grid.tolerance},sorc{grid.sorc},algo{grid.algo},d0{grid.d0},d1{grid.d1} {}
    __device__ double get_copy(size_t i, size_t j) {
        if (curr_buf_dev) return d1[i * room_size + j];
        else return d0[i * room_size + j];
    }
    __device__ void update_alt(size_t i, size_t j, double temp) {
        //printf("updating alt[%d,%d] = %f and curr_buf =%d\n",(int)i,(int)j,temp,curr_buf_dev);
        if (curr_buf_dev) d0[i * room_size + j] = temp;
        else d1[i * room_size + j] = temp;
    }
    __device__ void update_single(size_t i, size_t j, int k) {
        double temp;
        if (i == 0 || j == 0 || i == room_size - 1 || j == room_size - 1) {
            temp = border_temp;
        } else if (i == source_x && j == source_y) {
            temp = source_temp;
        } else {
            auto sum = get_copy(i+1,j) + get_copy(i-1,j) + get_copy(i,j+1)+ get_copy(i,j-1);
            switch (algo) {
                case JACOBI:
                    temp = 0.25 * sum;
                    update_alt(i,j,temp);
                    break;
                case SOR:
                    if (k == ((i + j) & 1)){
                        temp = get_copy(i,j) + (1.0 / sorc) * (sum - 4.0 * get_copy(i,j));
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

__global__ void dev_cal(DGrid grid, int k) {
    // DGrid grid(room_size, source_x, source_y, source_temp, border_temp,
    // tolerance, sorc, algo, d0, d1);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //("%d\n",(int)grid.room_size);
    for (size_t i = index; i < grid.room_size; i+=stride) {
        for (size_t j = 0; j < grid.room_size; ++j) {
            grid.update_single(i, j,k);
        }
    }
    //printf("d0_d[150,160]=%f\n",grid.get_copy(150,160));
    cudaError_t cudaStatus;
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            printf("mykernel launch failed: %s\n",
                    cudaGetErrorString(cudaStatus));
            return;
        }
    
}




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




void clean_alloc(double* d0, double* d1){
    cudaFree(d0);
    cudaFree(d1);
}

void host_cal(int x, int y, int room_size, int source_x, int source_y, 
    float source_temp, float border_temp, float tolerance, float sorc, int algo, 
    double *d0_d, double *d1_d){
        if(!algo){
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                tolerance, sorc, algo, d0_d, d1_d}, 0);
            curr_buf = !curr_buf;
            cudaMemcpyToSymbol(curr_buf_dev, &curr_buf, sizeof(int));
        }
        else{
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                tolerance, sorc, algo, d0_d, d1_d}, 0);
            curr_buf = !curr_buf;
            cudaMemcpyToSymbol(curr_buf_dev, &curr_buf, sizeof(int));
            dev_cal<<<x, y>>>({room_size, source_x, source_y, source_temp, border_temp,
                    tolerance, sorc, algo, d0_d, d1_d}, 1);
            curr_buf = !curr_buf;
            cudaMemcpyToSymbol(curr_buf_dev, &curr_buf, sizeof(int));
        }

    }
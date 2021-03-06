
# Parallel Computing: Heat Simulation
## Table of Contents
- [Parallel Computing: Heat Simulation](#parallel-computing-heat-simulation)
  - [Table of Contents](#table-of-contents)
  - [1. Introduction](#1-introduction)
    - [1.1 Project abstact](#11-project-abstact)
    - [1.2 Heat Simulation](#12-heat-simulation)
      - [The sequential implementation](#the-sequential-implementation)
      - [The parallel implementations](#the-parallel-implementations)
    - [1.3 How to run](#13-how-to-run)
  - [2. Methods](#2-methods)
    - [2.1 Design process](#21-design-process)
    - [2.2 Sequential Implementation](#22-sequential-implementation)
    - [2.3 Job partitions and distributions](#23-job-partitions-and-distributions)
      - [MPI implementaitons](#mpi-implementaitons)
      - [Pthread implementation](#pthread-implementation)
      - [CUDA implementation](#cuda-implementation)
      - [OpenMP implementation](#openmp-implementation)
      - [Hybrid (OpenMP+MPI)](#hybrid-openmpmpi)
    - [2.4 Local Calculation](#24-local-calculation)
    - [2.5 Rusults gathering](#25-rusults-gathering)
      - [MPI implementaitons](#mpi-implementaitons-1)
      - [Pthread implementation](#pthread-implementation-1)
      - [CUDA implementation](#cuda-implementation-1)
      - [OpenMP implementation](#openmp-implementation-1)
      - [Hybrid (OpenMP+MPI)](#hybrid-openmpmpi-1)
    - [2.6 Performance analysis](#26-performance-analysis)
      - [Profiling preparation](#profiling-preparation)
      - [Testing on different number of cores/threads/node](#testing-on-different-number-of-coresthreadsnode)
        - [MPI-like](#mpi-like)
        - [Pthread-like](#pthread-like)
        - [CUDA](#cuda)
      - [Result analysis](#result-analysis)
  - [3. Results](#3-results)
    - [3.1 GUI Results](#31-gui-results)
    - [3.2 Performance analysis](#32-performance-analysis)
      - [All but CUDA](#all-but-cuda)
      - [CUDA Analysis](#cuda-analysis)
  - [4. Conslusion](#4-conslusion)

## 1. Introduction
### 1.1 Project abstact
In the project we are required to implement a parallel program simulating the N-sody movements in a dimentinal plate, with CPU computation implemented in message passing manner via MPI, and in shared-memory manner with Pthread and OpenMP, hybrid manner via MPI combined with OpenMP and GPU computation implemented in Nvidia CUDA.

Compiler specificaitons:
* MPI, Pthread, OpenMP, Hybrid(bonus) implementations:
  ```
  clang version 12.0.1 
  (/llvm/llvm-project/clang fed41342a82f5a3a9201819a82bf7a48313e296b)
  Target: x86_64-unknown-linux-gnu
  Thread model: posix
  ```
* CUDA implementation:
  ```
  ## CUDA compiler:
  > Cuda compilation tools, release 11.4, V11.4.120
  > Build cuda_11.4.r11.4/compiler.30300941_0
  ## Host compiler:
  > g++ (GCC) 4.8.5 20150623 (Red Hat 4.8.5-28)
  ```

The program simulates a heat source distributing and propagating the heat around a 2-dimensinoal plane.

The performance matrixing tests the MPI, Pthread, OpenMP and hybrid implementations on different core numbers ranging from 4 to 64 for the ease of comparison, with the size (length) of the plane ranging from 100 to 800, and the iteration time is fixed to 600.
### 1.2 Heat Simulation
#### The sequential implementation
We can slice the continuous movement of the bodies into discrete time slice, and for the bodies in set ```B(t_1)```, its accleartion will based on the status in the previous time slice ```B(t_0)```, and there can be no dependency among elements in ```B(t_1)```.

Therefore, each time before I compute the acceleration, I will save a copy of the vector of ```bodies```, namely ```snapshot``` (i.e. the previous state), and change the attributes of each body (acceleration, velocity, position) independently in accordance with the content of ```snapshot```.

I contruct a class called Body encapsulating all attributes of a body and related method (which modifies the logic in the given template to make Body class independent). Therefore, the sequential version is a simple iteration:
```C++
std::vector<Body> snapshot = bodies;
for(size_t i = 0; i<bodies.size(); ++i) {
    # I. update accleration of bodies[i]
    # according to snapshot;
    # II. update velocity and position of
    # bodies[i] based on updated attributes.
}
```
Note there is NO DATA DEPENDENCY among different iterations of the for loop, making it very easy to be transformed into the parallel implementations.

#### The parallel implementations

The parallel implementations are highly similar to the sequential version stated above. Regardeless of the implementations, for each working slaves (processes, threads, etc.), they will all: 
* **Receive a full copy of ```snapshot```**
  Either in message passing manner in MPI or shard memory manner in Pthread and OpenMP, or memory copy from host to device in CUDA;

* **Copy the initial state of corresponding ```Body``` object from snapshot to there partition**, **update there partition**
  Do their own shares of the sequential loop above, according to the content in snapshot);

* **Send back their partition**
    Either in message passing manner in MPI or shard memory manner in Pthread and OpenMP, or memory copy from device to host in CUDA);

* All partitions combined consitutdes the bread new **bodies** vector which is the new state of bodies.

### 1.3 How to run
A Makefile is prepare, which further calls cmake. For your convinience, simply go to the build directory and call make:
```bash
cd /path/to/project/build
make
```
In total 5 executables will be generated:
* ```hs_mpi```, ```hs_hybrid```: MPI and MPI + OpenMP (bonus) implementations of the program. 

  Arguments: 
  * ```-g```: enable GUI (default **disabled**)
  * ```-s some_value```: specify the size of room/plane (default 100)
  * ```-i some_value```: specify iteration times i (default 600)




* ```hs_pthread```, ```hs_omp```: Pthread and OpenMP implementations of the program. 

  Arguments: 
  * ```-g```: enable GUI (default **disabled**)
  * ```-t some_value```: specify the **thread number** (default 16)
  * ```-s some_value```: specify the size of room/plane (default 100)
  * ```-i some_value```: specify iteration times i (default 600)




* ```hs_cuda```: CUDA implementation of the program. 

  Arguments: 
  * ```-g```: enable GUI (default **disabled**)
  * ```-x some_value```: specify the **grid dimension** (default 4)
  * ```-y some_value```: specify the **block dimension** (default 64)
  *i.e. the default config will utilized 256 threads accross 4 blocks*
  * ```-s some_value```: specify the size of room/plane (default 100)
  * ```-i some_value```: specify iteration times i (default 600)


Note that the value of these parameters can also be dynamically adjusted in GUI interface if GUI is enabled (by feeding -g argument).
  
An example to run the MPI version in parallel (hybrid is similar):
```bash
cd /path/to/project/build
# non-gui run:
mpirun ./hs_mpi
# gui run:
mpirun ./hs_mpi -g
```
  
An example to run the Pthread version in parallel (OpenMP is similar):
```bash
cd /path/to/project/build
# non-gui run with 8 threads
./hs_pthread -t 8
# gui run with 8 threads
./hs_pthread -g -t 8
```

An example to run CUDA (OpenMP is similar):
```bash
cd /path/to/project/build
# non-gui run with 1x128 block v thread
srun ./hs_cuda -x 1 -y 128
# gui run with default 4x64 block v thread
./hs_cuda -g
```



## 2. Methods
### 2.1 Design process
The design process is consists of in total 5 steps, which are **constructing sequential version**, **partitioning plane pixels (coordinates)**, **distributing partitions**, **implementing local calculations** finally **collecting local results** to formulate the final result.

An illustration of the approach is showned as below:

<figure align="center"  style="width: 100%;">

![fg1](https://i.imgur.com/fDVmBXs.png)

</figure>

Note that this is a uniform design idea suitable for all the implementations, with minor changes only.


Multiple tests are conducted on different core/thread numbers in **[1,4,8,16,32,64]** using:
*  **i) MPI**
  1,4,8,16,32,64 tasks with 1 core per task spanning from 1 to 2 nodes;
*  **ii) Pthread**:
 1,4,8,16,32,64 threads with maximum 32 core only on a single node, including case of over-spawning;
* **iii) OpenMP**:
   1,4,8,16,32,64 threads with maximum 32 core only on a single node, including case of over-spawning;
* **iv) CUDA**:
grid dimension and block dimension selected in [(1,32),(1,64),(1,128),(1,256),(2,128),(4,64)]) 
* **v) Hybrid**:
    1,4,8,16,32,64 tasks with 1 core per task spanning from 1 to 2 nodes, but using OpenMP on the same node, and MPI across the nodes.

with plane size in **[100,500,1000]** repectively for performance analysis and comparison purposes.


### 2.2 Sequential Implementation
In the sequential implementation, two buffers are alternatively used to hold the previous state and new state, and the new state is calculated according to temperature in the old buffer and stored in the new buffer, and switch in turns. As for the temperature calculation, two algorithms, Jacobi and Sor are implemented, basically calculate the temperature of a given state according to the four adjacent pixels' temperature.
### 2.3 Job partitions and distributions
To start contructing the parallel version of the algorithm, we must first partition the job into chunks of tasks and distributes them to each working nodes. Implementations of partitioning and distributions are explored.

#### MPI implementaitons

To partition the planes into serveral rows distributed to working processes as banlance as possible a partition scheme is derived with the idea of ```???(M-i)/n???``` (M being number of rows, i being rank, n being processes number) from discrete mathematics:

```C++
inline int getLength(const int &size, const int &proc, const int &rank){
    //porc-1, rank-1 to exclude master
    if(size<proc) return rank<size;
    return (size-rank)/proc + ((size-rank)%proc > 0); // ceil funct
}
```
So only minor changes are needed to transform the sequential version to the MPI version, for which I embedded into the switch buffer method in the grid:
```
MPI_Allgatherv(send_buf, recvcounts[rank],
MPI_DOUBLE, recv_buf, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
```
Here I adopt the All-to-all approach of MPI to synchronize and combine the result at each iteration.

#### Pthread implementation
To transform the sequential version into the pthread version, modifications are done inside the calculation iteration:
```C++
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
for (size_t i = para->start; i < para->start+para->length; ++i) {
for (size_t j = 0; j < para->state_ptr->room_size; ++j) {
auto result = para->update_single(i, j,*(para->grid_ptr), *(para->state_ptr));
*(para->stable_temp_ptr) &= result.stable;
para->grid_ptr->operator[]({alt, i, j}) = result.temp;
}
}
return NULL;
},para+t);}
```
Us usual, the parameters are packed as struct to be passed as a pointer. For simplicity, the lambda expression with tail return type specification is used to cast to a function pointer.
#### CUDA implementation
In CUDA implementations, an alternative method of partitioning is used, which is basically the simple framework as follows:
```C++
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for(int i=index;i<bodies_num; i+=stride){
    //do own share of work
}
```

Such simple paritioning (kind of round-robin approach) can be adapted to any block and grid dimension configuration, and providing a ralatively balanced partition as well.
As for the distribution, it is rather simple: data are first initialized in the host, copy to the device once, and then the result is copied back multiple times, each time a calculation is conducted. A global device variable curr_buf is used to do the buffer switching on device.

#### OpenMP implementation
The partition and distribution of the bodies in OpenMP is done implicitly by the compiler with only simple pragma involved:
```C++
#pragma omp parallel for shared(..) default(none)
```
#### Hybrid (OpenMP+MPI)
In the hybrid approach, the partitioning among the MPI processes is consistent with the [MPI implementaitons](#mpi-implementaitons), while in each MPI process, the OpenMP approach is adopted (by adding the above pragma to the local for loop).


### 2.4 Local Calculation
The local calculation adopts two algorithms: Jacobi and Sor.
### 2.5 Rusults gathering

#### MPI implementaitons

In MPI implementation, the gathering is embedded in the switch buffer method, which calls Allgatherv
```
MPI_AllGatherv(buffer.data()+offset,...);
```

#### Pthread implementation
The shared memory nature make it easy to collect results. 
A point to note is that the stablized variable may cause a data race condition. To avoid the data race but also avoid the use of mutex to slow down the process, each threads has there own stablized variable and will be reduced on logical OR upon completion.

#### CUDA implementation
In CUDA implementations, the memory allocated by is shared (among threads and blocks) as well (by ```CudaMalloc```). Therefore, we only need to copy bodies data from device back to host.

However, the kernel function call will not block the CPU. Therefore, for the sake of syncronization, ```cudaDeviceSynchronize()``` is called before memcpy:
```C++
cudaDeviceSynchronize();
cudaMemcpy(bodies, device_bodies,size*sizeof(Body),
cudaMemcpyDeviceToHost);
```

#### OpenMP implementation
The collection of the bodies in OpenMP is done implicitly by the compiler as well.

#### Hybrid (OpenMP+MPI)
Basically the same as [MPI implementaitons](#mpi-implementaitons-1), except that the OpenMP pragma is used for the ```for``` loop in each working process.



### 2.6 Performance analysis
The performance anaylysis is conducted in follow steps:
#### Profiling preparation
In the program, the C++ class ```std::chrono::high_resolution_clock``` is used to count the **total running time**, which will be **printed upon completion of the program**.


#### Testing on different number of cores/threads/node

The implementations are divided into three groups:
* **MPI-like**: MPI, Hybrid
* **Pthread-like**: Pthread, OpenMP
* **CUDA**

##### MPI-like
**MPI**: test are conducted with different number of tasks selected in **[1,4,8,16,32,64]**, each task comes with **a single** allocated CPU core, and for the 64 tasks config, two nodes are used.
**Hybrid**: test are conducted with different number of **cores** selected in **[1,4,8,16,32,64]**, each task comes with at most **16** allocted CPU core (so the corresponding task number is **[1,1,1,1,2,4]**), and for the 64 tasks config, two nodes are used.
For example:
For 8 cores config: task#=1; cpus-per-task=8;
For 16 cores config: task#=1; cpus-per-task=16;
For 32 cores config: task#=2; cpus-per-task=16;


##### Pthread-like
Both OpenMP and Pthread implementations will be tested with different number of threads (by feading argument to the program). The task number (i.e. CPU core number as the default cpus-per-task is 1) will be: ```min(thread_num,32)``` and all on a single node. Therefore the case of 64 threads is to experiment the effect of **over-spawning**.

##### CUDA
CUDA is different form the CPU computations above, therefore totally different config will be adopted. Different **(Grid_Dim,Block_Dim)** tuple is tested, selecting from [(1,32),(1,64),(1,128),(1,256),(2,128),(4,64)]


All testing above also span 3 different plane size which are 100, 500, 1000. 

Example of generated scripts (```/pvfsmnt/119010486/script_4/mpi/t8_s500.sh```):
```shell
#!/bin/bash
#SBATCH --account=csc4005
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=1
mkdir -p /pvfsmnt/119010486/proj4/test_results/mpi/
mpirun /pvfsmnt/119010486/proj4/hs_mpi -s 500 >>
/pvfsmnt/119010486/proj4/test_results/mpi/8_500.res
```
Finally, it is submitted also with a script (```/pvfsmnt/119010486/script_4/submit.sh```):
```shell
#!/bin/bash
until [ $(squeue --me | wc -l) -lt 32 ]
do
sleep 3
done
echo "current queue length = $(($(squeue --me | wc -l)-1))"
sbatch /pvfsmnt/119010486/script_4/mpi/t1_s100.sh
sleep 10
until [ $(squeue --me | wc -l) -lt 32 ]
do
sleep 3
done
echo "current queue length = $(($(squeue --me | wc -l)-1))"
sbatch /pvfsmnt/119010486/script_4/mpi/t1_s500.sh
sleep 10
...
```
Simply put, the outcome of these packs of scripts will be results matrix in two dimension **plane size** and **core number/thread number**. Inside each set, there are results from running the five implementations respectively for comparison purposes. Inside each result file, the **total running time** is reported. The whole set of result files are then compressed and downloaded to the local computer.

#### Result analysis
The result files are parsed using ```Python```. ```Pandas``` library is leveraged to organize the performance factors into tables with **Plane Sizes** being the column and **Core/Thread Number** being the row. Serveral graph is plotted as well to visualize the result. The outcome of the optimization is also inspected through test results.


## 3. Results
### 3.1 GUI Results

Running arguments: 
```shell
./hs_pthread -g -t 8
```
Running screenshot: 

![WX20211206-234448@2x](https://i.imgur.com/suV48FM.png)

### 3.2 Performance analysis
#### All but CUDA

The speedup of parallelism is calculated by the running time divided by the running time on single core/thread for each correponding plane size.

By observation, OpenMP performs the best when threads number is less than (or equal) 32 (its multithreading nature restricts it into a single node). Beyond 32 threads/processes, MPI performs better than the hybrid, whereas the difference is very small. Relatively, pthread performs the worst probably because of the implementation overhead is heavier than the OpenMP version, which is built upon the pthread but may optimize in terms of implementation.

Shown below are the illustrations of speedup except CUDA, with x axis being plane size, y axis being speedup, in both small, medium and large set:
<figure align="center">


![WX20211206-234942@2x](https://i.imgur.com/etm1Z9Y.png)

</figure>

We can see that in the small set, the small size greatly restricts the level of parallelism. The greatest speedup is below 12. We can also conclude that the pthread implementation has the greatest overhead of fork and join action, leading to a sharp drop when number of threads increses. In the experiment, MPI is the lightest, and the more pthread is incroduced (hybrid>openmp and pthread), the more overhead there appears.


<figure align="center">

![WX20211206-234956@2x](https://i.imgur.com/8CvxiOz.png)

</figure>

We can see that in the medium set, the bottleneck is moved much higher, and pure MPI implementation still performs the best among the four. It is also noteworthy that the 64 threads config for pthread and openmp is to examine the affect of over-spawning. By observation, the OpenMP implementation has a heavier cost for over-spawning in comparison with pthread.

<figure align="center">

![WX20211206-235009@2x](https://i.imgur.com/zNm77Qs.png)

</figure>

The outcome for the large set is highly similar, if not identical, in shape to the medium set. However, if we observe the largest speedup (by MPI in 64 cores config, clost to 60) in the large set, we can tell that the increse of data size further lift the bottleneck/ceiling of parallism up, which is consistent with the Gustafson's Law.

In thoery, the hyrbid OpenMP+MPI version should outperform the pure MPI version. However, it is not the case in the experiment, which is consistent with many existing research results. It may be because pure MPI achieves lower computational granularity therefore higher spatial locality of particles) and also lower communication granularity.

More horizontal comparisons and analysis are to carried out on **Pthread versus MPI** and **Dynamic Scheduling versus Static Scheduling**.

#### CUDA Analysis

The speedup of CUDA program is examined by comparing with the sequential running time [**on CPU**]. As the number of threads of GPU is very high compared with CPU, therefore the speedup performance greatly depends on the granularity of the data. As a body is an indivisable computing unit, when the plane size is small, the speedup performance is not ideal:
  <figure align="center">

![cuda_small](https://i.imgur.com/LOjlTnr.png)

  </figure>

  
When the size (length) of the plane increases, the maximum speedup has a great leap to above 20, and by observation, when the number of total threads (GridDim*BlockDim) is fixed, the more blocks allocated, the higher the speedup:
  <figure align="center">

![cuda_mid](https://i.imgur.com/upvC8lA.png)

  </figure>

The conslusion above is further substantiated by the speedup performance when the plane size if 1000 (larget set):
  <figure align="center">

![cuda_large](https://i.imgur.com/XAQV3p7.png)

  </figure>

## 4. Conslusion
In the project a parallel heat simulation is implemented and anaylzed. Fiver version in MPI, Pthread, OpenMP, CUDA, MPI and OpenMP Hybrid are implemented.

In total 90 ```(3-sizes * 6-core-configurations * 5-implementations) ```tests are carried out, each implementation testing its performance over 6 different core number from 1 to 64 over the plane size ranging from 100 to 1000. The speed up is examined on the matrix of core numbers and data sizes. The running time/duration is examined. The test is conducted through a series of script files generated with Python.

The result is parsed and visualized by leveraging the Python Pandas library. In terms of CPU computation, MPI exhibits the best speedup performance, whereas the pthread version ranks the last. This is different from the common sense that MPI has a higher overhead, and pthread should be lighter-weighted than the MPI process, which may caused the minor implementaiton differnce of the two program, or the issue of memory locality.

When comparing OpenMP and Pthread, which share the same underlying multihtreaded implementations, OpenMP outperforms Pthread, probably because OpenMP can achieve high efficiency in interparticle contact detection than manual implementation in pthread. However, the over-spawning cost of OpenMP is higher than that of Pthread in the experiments.

The GPU computation, implemented in CUDA, however, has greater number of threads available, and therefore preferes larger number of compuation unit (e.g. higher level of granularity or simply larger number of data). It is also observed that the greater number of blocks (but the same total threads) has positive affect on the speedup performance.

On top of that, the speedup of hybrid OpenMP MPI implementation is not as good as that of the pure MPI, probably due to the fact that pure MPI achieves lower computational granularity therefore higher spatial locality of particles) and also lower communication granularity.
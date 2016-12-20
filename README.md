# cuda-runtime

CUDA runtime support functions

```JavaScript
const CudaMemory = require('cuda-runtime').Memory;
const CudaDriverVersion = require('cuda-runtime').driverVersion();
const CudaRuntimeVersion = require('cuda-runtime').runtimeVersion();

let mem = new CudaMemory(512); // allocate 512 bytes of device memory

mem.set(new Buffer(512)); // memcpy buffer into device memory

let x = new Buffer(512);
mem.get(x); // read back memory from device
```

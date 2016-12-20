const assert = require('assert');
const ffi = require('ffi');
const ref = require('ref');
const ArrayType = require('ref-array');
const StructType = require('ref-struct');
const Pointer = ref.refType;

const voidPtr = Pointer(ref.types.void);
const voidPtrPtr = Pointer(voidPtr);
const intPtr = Pointer(ref.types.void);
const cudaError_t = ref.types.int;

const api = ffi.Library('/usr/local/cuda/lib/libcudart', {
  cudaGetErrorName: ['string', [cudaError_t]],
  cudaGetErrorString: ['string', [cudaError_t]],
  cudaDriverGetVersion: [cudaError_t, [intPtr]],
  cudaRuntimeGetVersion: [cudaError_t, [intPtr]],
  cudaMalloc: [cudaError_t, [voidPtrPtr, 'size_t']],
  cudaFree: [cudaError_t, [voidPtr]],
  cudaMemcpy: [cudaError_t, [voidPtr, voidPtr, 'size_t', 'int']],
});

class CudaError extends Error {
  constructor(err) {
    super(api.cudaGetErrorString(err));
  }
}

function check(err) {
  if (err) {
    throw new CudaError(err);
  }
}

class CudaMemory {
  constructor(size) {
    let ip = ref.alloc(voidPtr);
    check(api.cudaMalloc(ip, size));
    this._ptr = ip.deref();
    this._size = size;
  }

  release() {
    if (this._ptr) {
      check(api.cudaFree(this._ptr));
      this._ptr = null;
    }
  }

  set(buffer) {
    assert(buffer.length == this._size);
    check(api.cudaMemcpy(this._ptr, buffer, this._size, 1));
  }

  get(buffer) {
    assert(buffer.length == this._size);
    check(api.cudaMemcpy(buffer, this._ptr, this._size, 2));
  }
};

module.exports = {
  Error: CudaError,
  Memory: CudaMemory,
  driverVersion: () => {
    let ip = ref.alloc('int');
    check(api.cudaDriverGetVersion(ip));
    return ip.deref();
  },
  runtimeVersion: () => {
    let ip = ref.alloc('int');
    check(api.cudaRuntimeGetVersion(ip));
    return ip.deref();
  },
};

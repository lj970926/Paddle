/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_XPU

#include <memory>
#include <vector>

#include "paddle/phi/backends/xpu/forwards.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/xpu_cuda_stream.h"
#endif

namespace Eigen {
struct DefaultDevice;
}  // namespace Eigen

namespace xpu = baidu::xpu::api;

namespace phi {

#ifdef PADDLE_WITH_XPU
class XPUCUDAStream;
#endif

class DenseTensor;
class XPUContext : public DeviceContext,
                   public TypeInfoTraits<DeviceContext, XPUContext> {
 public:
  XPUContext();

  // is_comm_context = 1 for init comm context with gm_size=1 and l3_size=1
  explicit XPUContext(const XPUPlace&, bool is_comm_context = 0);

  virtual ~XPUContext();

#ifdef PADDLE_WITH_XPU
  /*! \brief  Return XPUCUDAStream in the device context. */
  XPUCUDAStream* xpu_cuda_stream() const;

  // Note that this function is a trick implementation since all 'set' methods
  // are protected by default.
  // clear: whether clear the original CUDAStream or not
  void SetXPUCUDAStream(XPUCUDAStream*, bool clear = true);
#endif

  const Place& GetPlace() const override;

  backends::xpu::XPUVersion xpu_version() const;

  xpu::Context* x_context(int i = 0) const;

  // Return bkcl context.
  xpu::BKCLContext_t bkcl_context() const;
  void SetBkclContext(xpu::BKCLContext_t context);
  void CreateStream(int i = 0);
  void RecordEvent(XPUEvent event, int s) const;
  void StreamWaitEvent(XPUEvent event, int s) const;
  void StreamWaitStream(int wait_stream, int record_stream) const;
  int64_t GetStreamNum() const;
  void AddStashedMemory(int stream, const phi::DenseTensor& tensor);

  void SetEigenDevice(Eigen::DefaultDevice*);
  void SetEigenDevice(std::function<Eigen::DefaultDevice*()>&&);

  // For share external stream.
  void SetStream(void* stream, int i = 0);

  // Wait for all operations completion in the stream.
  void Wait() const override;

 public:
  // NOTE: DeviceContext hold resources. Used in training scenarios.
  // The interface used by the training scene, DeviceContext will initialize
  // all resources and delete them when destructing.
  void Init();

 public:
  // NOTE: External users manage resources. Used in inference scenarios.
  // The Set interface is for inference only, DeviceContext will mark the
  // resource as external, and will not delete any resource when destructing.
  void SetXContext(xpu::Context*, int i = 0);

  void SetL3Cache(int64_t l3_size = 1024, int i = 0);

  void SetXpuVersion(int version);

  void SetRuntimeVersion(int runtime_version);

  void SetDriverVersion(int driver_version);

  Eigen::DefaultDevice* eigen_device() const { return nullptr; }

  XPUStream stream(int i = 0) const;

  static const char* name() { return "XPUContext"; }

 private:
  struct Impl;
  std::vector<std::unique_ptr<Impl>> impls_;

  void CheckValidStreamId(int i) const;
};

// KPS (Kernel PrimitiveS API) needs to exist as a kind of backend,
// because we want to implement a KPS-based kernel and make it run
// on GPU and XPU at the same time, so we need KPSContext when registering
// KPS Kernel. Note: XPU and GPU cannot be compiled at the same time!
#if PADDLE_WITH_XPU_KP
using KPSContext = XPUContext;
#endif

}  // namespace phi

namespace phi {
#if defined(PADDLE_WITH_XPU)
// Currently, XPUPinnedContext is only used to data copying.
class XPUPinnedContext
    : public DeviceContext,
      public phi::TypeInfoTraits<DeviceContext, XPUPinnedContext> {
 public:
  XPUPinnedContext();
  explicit XPUPinnedContext(XPUPinnedPlace place);

  const Place& GetPlace() const override;

  Eigen::DefaultDevice* eigen_device() const;

  static const char* name() { return "XPUPinnedContext"; }

 private:
  XPUPinnedPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};
#endif
}  // namespace phi

#endif

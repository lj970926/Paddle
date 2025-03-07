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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/operators/fused/fusion_lstm_op.h"
#include "paddle/fluid/operators/fused/onednn/fusion_rnn_onednn.h"
#include "paddle/phi/core/expect.h"

namespace paddle {
namespace operators {

using phi::OneDNNContext;
using phi::funcs::OneDNNGetDataType;
using phi::funcs::OneDNNMemDesc;
using phi::funcs::RNNReorderType;
using OneDNNMemoryFormat = dnnl::memory::format_tag;

template <typename T, typename T_out = T>
class LSTMMKLDNNHandler
    : public RNNMKLDNNHandler<T, dnnl::lstm_forward, T_out> {
 public:
  LSTMMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                    const OneDNNContext& dev_ctx,
                    const dnnl::engine onednn_engine,
                    phi::Place cpu_place UNUSED,
                    const phi::DenseTensor* input,
                    const phi::DenseTensor* weight_h,
                    const phi::DenseTensor* h0,
                    const phi::DenseTensor* c0 UNUSED,
                    const bool is_reverse,
                    const int64_t N,
                    const int64_t Ti,
                    const int64_t IC,
                    const int64_t OC,
                    const std::string& unique_name UNUSED)
      : RNNMKLDNNHandler<T, dnnl::lstm_forward, T_out>(
            ctx,
            dev_ctx,
            onednn_engine,
            ctx.GetPlace(),
            input,
            weight_h,
            h0,
            is_reverse,
            N,
            Ti,
            IC,
            OC,
            4,
            ctx.InputName("X") + ctx.InputName("WeightH")) {
    if (unlikely(!this->isCached())) {
      const bool is_INT8 = std::is_same<T, uint8_t>::value;
      const bool use_peepholes = ctx.Attr<bool>("use_peepholes");
      // oneDNN kernel has hardcoded activation functions
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("gate_activation"),
          "sigmoid",
          phi::errors::Unimplemented("oneDNN fusion_lstm supports only "
                                     "sigmoid as a gate activation."));
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("cell_activation"),
          "tanh",
          phi::errors::Unimplemented(
              "oneDNN fusion_lstm supports only tanh as a cell activation."));
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("candidate_activation"),
          "tanh",
          phi::errors::Unimplemented(
              "oneDNN fusion_lstm supports only tanh a candidate activation."));

      // Weights for int8 kernel are of a type s8
      const auto weights_dt =
          is_INT8 ? dnnl::memory::data_type::s8 : OneDNNGetDataType<T>();

      // oneDNN RNN dimensions
      const int64_t D = 1;  // Directions
      const int64_t L = 1;  // Layers (PP supports only 1 stacked layer)
      const int64_t G = 4;  // Number of Gates, 4 for LSTM

      // Create memory descriptors
      auto input_md = OneDNNMemDesc(
          {Ti, N, IC}, OneDNNGetDataType<T>(), OneDNNMemoryFormat::tnc);
      auto weight_x_md =
          OneDNNMemDesc({L, D, IC, G, OC}, weights_dt, OneDNNMemoryFormat::any);
      auto weight_h_md =
          OneDNNMemDesc({L, D, OC, G, OC}, weights_dt, OneDNNMemoryFormat::any);
      auto bias_md = OneDNNMemDesc(
          {L, D, G, OC}, OneDNNGetDataType<float>(), OneDNNMemoryFormat::ldgo);
      auto hidden_md = OneDNNMemDesc(
          {Ti, N, OC}, OneDNNGetDataType<T_out>(), OneDNNMemoryFormat::any);

      auto h0_md = OneDNNMemDesc(
          {L, D, N, OC}, OneDNNGetDataType<T>(), OneDNNMemoryFormat::any);
      auto c0_md = OneDNNMemDesc(
          {L, D, N, OC}, OneDNNGetDataType<float>(), OneDNNMemoryFormat::any);

      // Create LSTM oneDNN primitive
      const auto direction =
          is_reverse ? dnnl::rnn_direction::unidirectional_right2left
                     : dnnl::rnn_direction::unidirectional_left2right;
      if (!use_peepholes) {
        this->AcquireForwardPrimitiveDescriptor(
            this->attr_,
            dnnl::prop_kind::forward_inference,
            direction,
            input_md,
            h0_md,
            c0_md,
            weight_x_md,
            weight_h_md,
            bias_md,
            hidden_md,
            dnnl::memory::desc(),
            dnnl::memory::desc());
      } else {
        auto weight_peephole_md = OneDNNMemDesc({L, D, 3, OC},
                                                OneDNNGetDataType<float>(),
                                                OneDNNMemoryFormat::ldgo);
        this->AcquireForwardPrimitiveDescriptor(
            this->attr_,
            dnnl::prop_kind::forward_inference,
            direction,
            input_md,
            h0_md,
            c0_md,
            weight_x_md,
            weight_h_md,
            weight_peephole_md,
            bias_md,
            hidden_md,
            dnnl::memory::desc(),
            dnnl::memory::desc());
      }
    }
  }

  // PaddlePaddle has different order of weights than oneDNN, so a reorder is
  // needed
  // PaddlePaddle:  {c, i, f, o}
  // oneDNN:        {i, f, c, o}
  template <typename U>
  void ReorderGates(U* weights, int64_t I) {
    size_t inner_block_size = this->OC;
    size_t block_size = inner_block_size * this->G;
    for (size_t i = 0; i < (size_t)I; ++i) {  // NOLINT
      size_t offset = i * block_size;

      U* base_pos = weights + offset;
      std::swap_ranges(base_pos,
                       base_pos + inner_block_size,
                       base_pos + inner_block_size);  // c <-> i
      std::swap_ranges(base_pos + inner_block_size,
                       base_pos + 2 * inner_block_size,
                       base_pos + 2 * inner_block_size);  // c <-> f
    }
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightXMemory(
      const phi::DenseTensor* weight_x) {
    const std::string wx_key = this->memory_key_ + "@weight_x";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wx_key));

    if (!memory_p) {
      auto user_md = OneDNNMemDesc({1, 1, this->IC, this->G, this->OC},
                                   OneDNNGetDataType<U>(),
                                   OneDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      auto* weight_x_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      memcpy(weight_x_data,
             weight_x->data<U>(),
             sizeof(U) * this->IC * this->G * this->OC);

      ReorderGates(weight_x_data, this->IC);

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_layer_desc(), this->engine_);

      auto& astream = OneDNNContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wx_key, memory_p);
    }
    return memory_p;
  }

  template <typename U>
  std::shared_ptr<dnnl::memory> AcquireWeightHMemory(
      const phi::DenseTensor* weight_h) {
    const std::string wh_key = this->memory_key_ + "@weight_h";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wh_key));

    if (!memory_p) {
      auto user_md = OneDNNMemDesc({1, 1, this->OC, this->G, this->OC},
                                   OneDNNGetDataType<U>(),
                                   OneDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      auto* weight_h_data = reinterpret_cast<U*>(user_memory.get_data_handle());
      memcpy(weight_h_data,
             weight_h->data<U>(),
             sizeof(U) * this->OC * this->G * this->OC);

      ReorderGates(weight_h_data, this->OC);

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_iter_desc(), this->engine_);

      auto& astream = OneDNNContext::tls().get_stream();
      dnnl::reorder(user_memory, *memory_p, this->attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wh_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(
      const phi::DenseTensor* bias) {
    const std::string bias_key = this->memory_key_ + "@bias";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(bias_key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->bias_desc(),
                                                this->engine_);
      auto* bias_data = reinterpret_cast<float*>(memory_p->get_data_handle());
      if (bias) {
        const float* user_bias_data =
            bias->data<float>();  // Bias in oneDNN is always float

        memcpy(bias_data, user_bias_data, sizeof(float) * this->G * this->OC);

        ReorderGates(bias_data, 1);
      } else {
        // oneDNN always need bias memory, if it's not provided in PP, let
        // oneDNN allocate memory and set it to 0
        memset(bias_data, 0, sizeof(float) * this->G * this->OC);
      }

      this->dev_ctx_.SetBlob(bias_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquirePeepholeWeights(
      const phi::DenseTensor* bias) {
    const std::string peepholes_key = this->memory_key_ + "@peepholes_weights";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(peepholes_key));

    if (!memory_p) {
      auto user_md = OneDNNMemDesc({1, 1, 3, this->OC},
                                   OneDNNGetDataType<float>(),
                                   OneDNNMemoryFormat::ldgo);
      auto user_memory = dnnl::memory(user_md, this->engine_);
      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_peephole_desc(), this->engine_);
      auto* peephole_weights_data =
          reinterpret_cast<float*>(memory_p->get_data_handle());

      const float* user_bias_data =
          bias->data<float>();  // Bias in oneDNN is always float
      memcpy(peephole_weights_data,
             user_bias_data + 4 * this->OC,
             sizeof(float) * 3 * this->OC);

      this->dev_ctx_.SetBlob(peepholes_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireC0Memory(const phi::DenseTensor* c0) {
    const std::string c0_key = this->memory_key_ + "@c0";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(c0_key));

    if (!memory_p) {
      auto user_c0_memory = dnnl::memory();
      if (c0) {
        user_c0_memory =
            dnnl::memory({{1, 1, this->N, this->OC},
                          OneDNNGetDataType<float>(),
                          OneDNNMemoryFormat::ldnc},
                         this->engine_,
                         phi::funcs::to_void_cast(c0->data<float>()));
      } else {
        user_c0_memory = dnnl::memory({{1, 1, this->N, this->OC},
                                       OneDNNGetDataType<float>(),
                                       OneDNNMemoryFormat::ldnc},
                                      this->engine_);
        memset(user_c0_memory.get_data_handle(),
               0,
               sizeof(float) * this->N * this->OC);
      }
      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->src_iter_c_desc(), this->engine_);

      auto& astream = OneDNNContext::tls().get_stream();
      dnnl::reorder(user_c0_memory, *memory_p)
          .execute(astream, user_c0_memory, *memory_p);

      this->dev_ctx_.SetBlob(c0_key, memory_p);
    }
    return memory_p;
  }
};

template <typename T, typename DeviceContext>
class FusionLSTMMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool is_bf16 = std::is_same<T, phi::dtype::bfloat16>::value;
    const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    // BF16 does not support force output
    if (!is_bf16 && force_fp32_output) {  // NOLINT
      RunKernel<float>(ctx);
    } else {
      RunKernel<T>(ctx);
    }
  }

  template <typename Tout = T>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    // Get Tensors
    const auto* input = ctx.Input<phi::DenseTensor>("X");
    const auto* h0 = ctx.Input<phi::DenseTensor>("H0");
    const auto* c0 = ctx.Input<phi::DenseTensor>("C0");
    const auto* weight_x = ctx.Input<phi::DenseTensor>("WeightX");
    const auto* weight_h = ctx.Input<phi::DenseTensor>("WeightH");
    const auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* hidden = ctx.Output<phi::DenseTensor>("Hidden");
    auto x_dims = input->dims();
    auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                          ? common::flatten_to_2d(x_dims, 1)
                          : x_dims;
    // Get attributes
    const bool is_reverse = ctx.Attr<bool>("is_reverse");
    const bool use_peepholes = ctx.Attr<bool>("use_peepholes");

    // Get tensor dimensions
    const auto x_mat_dims_vec = common::vectorize(x_mat_dims);
    const auto weight_h_dims = common::vectorize(weight_h->dims());
    const auto& input_lod = input->lod()[0];

    // Calculate RNN dimensions
    const int64_t N = input_lod.size() - 1;  // Number of sentences (batches)
    const int64_t Ti =  // Max length of the sentence in a batch
        [&input_lod]() {
          size_t res = 0;
          for (size_t i = 0; i < (input_lod.size() - 1); ++i) {
            res = std::max(res, input_lod[i + 1] - input_lod[i]);
          }
          return res;
        }();
    const int64_t IC = x_mat_dims_vec[1];  // Input channels
    const int64_t OC = weight_h_dims[0];   // Output channels

    LSTMMKLDNNHandler<T, Tout> handler(
        ctx,
        dev_ctx,
        onednn_engine,
        ctx.GetPlace(),
        input,
        weight_h,
        h0,
        c0,
        is_reverse,
        N,
        Ti,
        IC,
        OC,
        ctx.InputName("X") + ctx.InputName("WeightH"));

    auto input_memory_p =
        handler.AcquireInputMemoryWithReorder(input, is_reverse);
    auto c0_memory_p = handler.AcquireC0Memory(c0);

    std::shared_ptr<dnnl::memory> h0_memory_p, weight_h_memory_p,
        weight_x_memory_p;

    if (weight_h->dtype() == phi::DataType::FLOAT32) {
      h0_memory_p = handler.template AcquireH0Memory<float>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<float>(weight_x);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<float>(weight_h);
    } else if (weight_h->dtype() == phi::DataType::BFLOAT16) {
      h0_memory_p = handler.template AcquireH0Memory<phi::dtype::bfloat16>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<phi::dtype::bfloat16>(weight_x);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<phi::dtype::bfloat16>(weight_h);
    } else {
      h0_memory_p = handler.template AcquireH0Memory<uint8_t>(h0);
      weight_x_memory_p =
          handler.template AcquireWeightXMemory<int8_t>(weight_x);
      weight_h_memory_p =
          handler.template AcquireWeightHMemory<int8_t>(weight_h);
    }

    auto bias_memory_p = handler.AcquireBiasMemory(bias);
    auto hidden_onednn_memory_p = handler.AcquireOutputMemory();

    std::unordered_map<int, dnnl::memory> lstm_args = {
        {DNNL_ARG_SRC_LAYER, *input_memory_p},
        {DNNL_ARG_SRC_ITER, *h0_memory_p},
        {DNNL_ARG_SRC_ITER_C, *c0_memory_p},
        {DNNL_ARG_WEIGHTS_LAYER, *weight_x_memory_p},
        {DNNL_ARG_WEIGHTS_ITER, *weight_h_memory_p},
        {DNNL_ARG_BIAS, *bias_memory_p},
        {DNNL_ARG_DST_LAYER, *hidden_onednn_memory_p}};

    if (use_peepholes) {
      auto peephole_weight_p = handler.AcquirePeepholeWeights(bias);
      std::pair<int, dnnl::memory> peepholes_weights(DNNL_ARG_WEIGHTS_PEEPHOLE,
                                                     *peephole_weight_p);
      lstm_args.insert(peepholes_weights);
    }

    auto lstm_forward_p = handler.AcquireForwardPrimitive();

    auto& astream = OneDNNContext::tls().get_stream();
    lstm_forward_p->execute(astream, lstm_args);
    astream.wait();

    auto* hidden_onednn_data = hidden_onednn_memory_p->get_data_handle();
    auto* hidden_data =
        phi::funcs::to_void_cast(hidden->mutable_data<Tout>(ctx.GetPlace()));
    if (handler.is_NTC()) {
      handler.reorderRNNdata(hidden_onednn_data,
                             hidden_data,
                             input_lod,
                             is_reverse,
                             RNNReorderType::NTC_PP);
    } else {
      handler.reorderRNNdata(hidden_onednn_data,
                             hidden_data,
                             input_lod,
                             is_reverse,
                             RNNReorderType::TNC_PP);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(fusion_lstm,
                          OneDNN,
                          ONEDNN,
                          ops::FusionLSTMMKLDNNKernel,
                          float,
                          uint8_t,
                          phi::dtype::bfloat16) {}

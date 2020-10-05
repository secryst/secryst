# ported from https://pytorch.org/docs/master/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_

module Secryst
  class ClipGradNorm < Torch::NN::F
    def self.clip_grad_norm(parameters, max_norm:, norm_type:2)
      parameters = parameters.select {|p| p.grad }
      max_norm = max_norm.to_f
      if parameters.length == 0
        return Torch.tensor(0.0)
      end
      device = parameters[0].grad.device
      if norm_type == Float::INFINITY
        # ... TODO
      else
        total_norm = Numo::Linalg.norm(Numo::NArray.concatenate(parameters.map {|p| Numo::Linalg.norm(p.grad.detach.numo, norm_type)}), norm_type)
      end
      clip_coef = max_norm / (total_norm + 1e-6)
      if clip_coef < 1
        parameters.each {|p| p.grad = p.grad.detach * clip_coef}
      end

      return total_norm
    end
  end
end

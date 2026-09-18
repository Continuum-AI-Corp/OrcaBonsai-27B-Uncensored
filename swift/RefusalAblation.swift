// Reference implementation of the runtime projection for mlx-swift.
//
// The pack ships as `refusal_dir_fp32.bin`: 5120 little-endian Float32 values, no
// header, unit norm, in the plain (unrotated) hidden basis. Apply the projection to the
// output of every module that writes the residual stream -- on this architecture that is
// each layer's `mlp.down_proj`, plus `self_attn.o_proj` on the 16 full-attention layers,
// `linear_attn.out_proj` on the 48 linear-attention layers, and the token embedding:
// 129 sites in total. Wrapping only `o_proj` would miss three quarters of the attention
// writes.
//
// No Hadamard handling belongs here. The rotation is folded into the input dimension of
// each projection and the runtime already compensates on the activation side; the
// vectors projected here are outputs, which live in the plain hidden basis.
//
// Note that PACK-RUNTIME.md states Swift support is layer-level only and full-model
// loading still needs model integration. Since that has to be written anyway, this
// projection adds one dot product and one axpy per residual write.

import Foundation
import MLX

public struct RefusalAblation {
    /// Unit-norm direction in the unrotated hidden basis.
    public let direction: MLXArray
    /// 1.0 matches a full permanent weight edit; lower trades strength for capability.
    public var alpha: Float

    public init(direction: MLXArray, alpha: Float = 1.0) {
        self.direction = direction.asType(.float32)
        self.alpha = alpha
    }

    /// Load `refusal_dir_fp32.bin` -- a bare array of `hiddenSize` Float32 values.
    public init(binaryAt url: URL, hiddenSize: Int = 5120, alpha: Float = 1.0) throws {
        let data = try Data(contentsOf: url)
        guard data.count == hiddenSize * 4 else {
            throw NSError(domain: "RefusalAblation", code: 1, userInfo: [
                NSLocalizedDescriptionKey:
                    "expected \(hiddenSize * 4) bytes, found \(data.count)",
            ])
        }
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
        self.init(direction: MLXArray(values, [hiddenSize]), alpha: alpha)
    }

    /// Remove the component along `direction` from a residual-stream write.
    ///
    /// Accumulate in float32: doing this in float16 leaves enough residue per site that
    /// it compounds across 129 of them.
    public func project(_ y: MLXArray) -> MLXArray {
        guard alpha != 0 else { return y }
        let yf = y.asType(.float32)
        let component = (yf * direction).sum(axis: -1, keepDims: true)
        return (yf - alpha * component * direction).asType(y.dtype)
    }
}

/// Wrap any residual-writing module so its output is projected.
///
///     layer.mlp.downProj = AblatedLinear(inner: layer.mlp.downProj, ablation: ablation)
public final class AblatedLinear: Module, UnaryLayer {
    let inner: UnaryLayer
    let ablation: RefusalAblation

    public init(inner: UnaryLayer, ablation: RefusalAblation) {
        self.inner = inner
        self.ablation = ablation
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        ablation.project(inner(x))
    }
}

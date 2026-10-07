// Yuna addition to the vendored coreai-models package (not part of Apple's upstream sources).

import Accelerate
import Foundation
@preconcurrency import Metal
import os

// MARK: - CPU sampler with MLX-style logit penalties

/// Samples on the CPU after the decode command buffer completes, so the repetition, presence and
/// frequency penalties of `mlx_lm.sample_utils` can see the tokens already sampled.
///
/// Per token: fp16 logits row -> Float32 (vImage) -> penalties over the trailing token window ->
/// temperature, top-p, min-p, top-k and a categorical draw (or argmax at temperature 0).
/// It never builds an MPSGraph, so unlike the GPU composite sampler it costs no GPU heap on an
/// 8 GB iPhone.
///
/// Nothing here loops over the vocabulary in Swift: a scalar pass over 152k floats costs ~26 ms in
/// a -Onone build (Apple's `CompositeSampler` does several), against ~0.1 ms for one `vDSP_maxvi`.
/// Candidates are instead peeled off the row one `vDSP_maxvi` at a time, stopping as soon as the
/// nucleus, min-p or the tail says so, which is a handful of passes for a peaked distribution.
///
/// The next decode reads its input token from `outputBuffer`, which the completion handler writes,
/// so this sampler reports `decodesSerially`: the engine must not encode another step before then.
final class CPULogitSampler: MPSGraphSampler, @unchecked Sendable {
    let vocabSize: Int
    var decodesSerially: Bool { true }
    private let scratch: MTLBuffer?
    // Float32 working rows. Only touched inside `state.withLock`.
    private let row: UnsafeMutablePointer<Float>
    private let tmp: UnsafeMutablePointer<Float>

    private struct State {
        var config: SamplingConfiguration
        var window: [Int32] = []
        var counts: [Int32: Int32] = [:]
        var loggedOnce = false
    }
    private let state: OSAllocatedUnfairLock<State>

    /// Caps on how many tokens one draw may peel. Without top-k, a nucleus or min-p window is
    /// bounded at 256 (it is a few tokens in practice) and plain temperature sampling at 64, the
    /// same default the GPU factory uses; mass below `tailWeight` of the best token is dropped.
    private static let windowCap = 256
    private static let temperatureOnlyCap = 64
    private static let tailWeight: Float = 1e-6

    init(device: MTLDevice, vocabSize: Int, config: SamplingConfiguration) {
        self.vocabSize = vocabSize
        self.scratch = device.makeBuffer(length: 4, options: .storageModeShared)
        self.row = .allocate(capacity: vocabSize)
        self.row.initialize(repeating: 0, count: vocabSize)
        self.tmp = .allocate(capacity: vocabSize)
        self.tmp.initialize(repeating: 0, count: vocabSize)
        self.state = OSAllocatedUnfairLock(initialState: State(config: config))
        print("YunaAVL: CPU logit sampler vocab=\(vocabSize) \(Self.describe(config))")
    }

    deinit {
        row.deallocate()
        tmp.deallocate()
    }

    // MARK: Configuration and penalty window

    private static func describe(_ c: SamplingConfiguration) -> String {
        let topK = c.topK.map { String($0) } ?? "-"
        let topP = c.topP.map { String($0) } ?? "-"
        let minP = c.minP.map { String($0) } ?? "-"
        let window = (c.penaltyContextSize ?? 0) > 0 ? String(c.penaltyContextSize ?? 0) : "all"
        let penalties = "rep=\(c.repetitionPenalty ?? 1) presence=\(c.presencePenalty ?? 0) frequency=\(c.frequencyPenalty ?? 0)"
        return "T=\(c.temperature) topK=\(topK) topP=\(topP) minP=\(minP) \(penalties) window=\(window)"
    }

    func beginGeneration(config: SamplingConfiguration, context: [Int32]) {
        let vocab = Int32(vocabSize)
        state.withLock { s in
            s.config = config
            s.window.removeAll(keepingCapacity: true)
            s.counts.removeAll(keepingCapacity: true)
            guard config.hasLogitPenalties else { return }
            // Pad ids (>= vocab) name image/audio slots, not vocabulary rows.
            let limit = config.penaltyContextSize ?? 0
            let tail = limit > 0 ? context.suffix(limit) : context[...]
            for t in tail where t >= 0 && t < vocab {
                s.window.append(t)
                s.counts[t, default: 0] += 1
            }
        }
    }

    private static func record(_ token: Int32, into s: inout State, vocab: Int32) {
        guard s.config.hasLogitPenalties, token >= 0, token < vocab else { return }
        s.window.append(token)
        s.counts[token, default: 0] += 1
        let limit = s.config.penaltyContextSize ?? 0
        guard limit > 0, s.window.count > limit else { return }
        let old = s.window.removeFirst()
        if let c = s.counts[old] { s.counts[old] = c > 1 ? c - 1 : nil }
    }

    /// `mlx_lm` order: sign-aware repetition penalty first, then the additive presence and
    /// frequency penalties. Once per distinct token, except frequency which scales by its count.
    private func penalize(counts: [Int32: Int32], config: SamplingConfiguration) {
        let repetition = Float(config.repetitionPenalty ?? 1)
        let presence = Float(config.presencePenalty ?? 0)
        let frequency = Float(config.frequencyPenalty ?? 0)
        for (token, count) in counts {
            let i = Int(token)
            var v = row[i]
            if repetition != 1 { v = v < 0 ? v * repetition : v / repetition }
            row[i] = v - (presence + frequency * Float(count))
        }
    }

    // MARK: Sampling

    /// vDSP_maxvi can return the wrong index when the row holds a NaN, so NaN becomes -inf and
    /// +inf the largest finite value. False when no finite logit is left.
    private func sanitize() -> Bool {
        let count = vocabSize
        var sum: Float = 0
        vDSP_sve(row, 1, &sum, vDSP_Length(count))
        guard !sum.isFinite else { return true }
        var finite = 0
        for i in 0..<count {
            let v = row[i]
            if v.isNaN {
                row[i] = -.infinity
            } else if v == .infinity {
                row[i] = .greatestFiniteMagnitude
                finite += 1
            } else if v != -.infinity {
                finite += 1
            }
        }
        return finite > 0
    }

    /// MLX semantics on the penalised row. Temperature 0 is the argmax. Otherwise the row is divided
    /// by the temperature, the nucleus is measured against the whole vocabulary as
    /// `mlx_vlm.top_p_sampling` does (a token stays while the mass ranked above it is below top-p),
    /// min-p is relative to the best token, top-k is a hard cap, and the draw is proportional to the
    /// kept weights.
    private func pick(_ config: SamplingConfiguration) -> Int32 {
        let n = vDSP_Length(vocabSize)
        var best: Float = 0
        var index: vDSP_Length = 0
        guard config.temperature > 0 else {
            vDSP_maxvi(row, 1, &best, &index, n)
            return Int32(index)
        }
        var inverse = Float(1 / config.temperature)
        vDSP_vsmul(row, 1, &inverse, row, 1, n)
        vDSP_maxv(row, 1, &best, n)
        let top = best
        let nucleus = Float(config.topP ?? 1)
        let useNucleus = nucleus < 1
        let minP = Float(config.minP ?? 0)
        let cap = min(
            config.topK ?? (useNucleus || minP > 0 ? Self.windowCap : Self.temperatureOnlyCap), vocabSize)
        var norm: Float = 1
        if useNucleus {
            var shift = -top
            vDSP_vsadd(row, 1, &shift, tmp, 1, n)
            var count = Int32(vocabSize)
            vvexpf(tmp, tmp, &count)
            vDSP_sve(tmp, 1, &norm, n)
        }
        var ids: [Int32] = []
        var weights: [Float] = []
        ids.reserveCapacity(cap)
        weights.reserveCapacity(cap)
        var mass: Float = 0
        for rank in 0..<cap {
            vDSP_maxvi(row, 1, &best, &index, n)
            let w = expf(best - top)
            if rank > 0 && (w < minP || w < Self.tailWeight) { break }
            ids.append(Int32(index))
            weights.append(w)
            mass += w
            row[Int(index)] = -.infinity
            if useNucleus && mass / norm >= nucleus { break }
        }
        var r = Float.random(in: 0..<1) * mass
        for (k, w) in weights.enumerated() {
            r -= w
            if r < 0 { return ids[k] }
        }
        return ids[ids.count - 1]
    }

    func sample(_ buffer: MTLBuffer, byteOffset: Int) -> Int32 {
        let vocab = vocabSize
        let needed = byteOffset + vocab * MemoryLayout<Float16>.size
        guard buffer.length >= needed else {
            print("YunaAVL: CPU logit sampler logits buffer too small (\(buffer.length) < \(needed))")
            return SamplerSentinel.failure
        }
        nonisolated(unsafe) let halves = buffer.contents().advanced(by: byteOffset)
        return state.withLock { s -> Int32 in
            var src = vImage_Buffer(
                data: halves, height: 1, width: vImagePixelCount(vocab),
                rowBytes: vocab * MemoryLayout<Float16>.stride)
            var out = vImage_Buffer(
                data: row, height: 1, width: vImagePixelCount(vocab), rowBytes: vocab * MemoryLayout<Float>.stride)
            guard vImageConvert_Planar16FtoPlanarF(&src, &out, vImage_Flags(kvImageNoFlags)) == kvImageNoError,
                sanitize()
            else {
                print("YunaAVL: CPU logit sampler found no finite logits")
                return SamplerSentinel.failure
            }
            if s.config.hasLogitPenalties { penalize(counts: s.counts, config: s.config) }
            let token = pick(s.config)
            Self.record(token, into: &s, vocab: Int32(vocab))
            if !s.loggedOnce {
                s.loggedOnce = true
                print("YunaAVL: CPU logit sampler first token=\(token) window=\(s.window.count)")
            }
            return token
        }
    }

    // MARK: MPSGraphSampler

    func encode(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        logitsOffset: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32) -> Void
    ) {
        guard let cb = queue.makeCommandBuffer() else {
            completion(SamplerSentinel.failure)
            return
        }
        cb.label = "CPULogit.waitForDecode"
        // A real GPU hazard on the logits buffer, so this command buffer cannot complete before the
        // decode that writes it (an empty command buffer is not a reliable fence).
        if let scratch, logitsOffset < logitsBuffer.length, let blit = cb.makeBlitCommandEncoder() {
            let size = min(4, logitsBuffer.length - logitsOffset)
            if size > 0 {
                blit.copy(from: logitsBuffer, sourceOffset: logitsOffset, to: scratch, destinationOffset: 0, size: size)
            }
            blit.endEncoding()
        }
        nonisolated(unsafe) let logits = logitsBuffer
        nonisolated(unsafe) let output = outputBuffer
        nonisolated(unsafe) let onToken = completion
        cb.addCompletedHandler { [self] buffer in
            if buffer.status == .error {
                print("YunaAVL: CPU logit sampler wait failed: \(String(describing: buffer.error))")
                onToken(SamplerSentinel.failure)
                return
            }
            let token = self.sample(logits, byteOffset: logitsOffset)
            // The next decode reads this slot, so it is written before the engine hears about the token.
            if token >= 0 {
                output.contents().advanced(by: outputOffset).assumingMemoryBound(to: Int32.self).pointee = token
            }
            onToken(token)
        }
        cb.commit()
    }

    func encodeWithSlice(
        to queue: MTLCommandQueue,
        logitsBuffer: MTLBuffer,
        queryLength: Int,
        outputBuffer: MTLBuffer,
        outputOffset: Int,
        completion: @escaping (Int32) -> Void
    ) {
        let last = max(queryLength, 1) - 1
        encode(
            to: queue,
            logitsBuffer: logitsBuffer,
            logitsOffset: last * vocabSize * MemoryLayout<UInt16>.size,
            outputBuffer: outputBuffer,
            outputOffset: outputOffset,
            completion: completion
        )
    }
}

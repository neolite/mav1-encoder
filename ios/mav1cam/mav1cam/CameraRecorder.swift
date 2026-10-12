import AVFoundation
import CoreMedia
import Foundation
import Photos
import UIKit

/// Camera -> mav1 -> .ivf in Documents. NV12/planar input; the UV planes of
/// NV12 buffers are deinterleaved once per frame into scratch buffers.
final class CameraRecorder: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let videoQueue = DispatchQueue(label: "mav1.video", qos: .userInitiated)
    private let videoOutput = AVCaptureVideoDataOutput()

    private var enc: OpaquePointer?
    private var file: FileHandle?
    private var outURL: URL?
    private var movWriter: AVAssetWriter?
    private var movInput: AVAssetWriterInput?
    private var movURL: URL?
    private var av1Desc: CMFormatDescription?
    private var sampleIdx: Int64 = 0
    private var uBuf: UnsafeMutableRawPointer?
    private var vBuf: UnsafeMutableRawPointer?
    private var uvBytes = 0
    private var yBuf: UnsafeMutableRawPointer?
    private var yBytes = 0
    /// 10-bit HDR capture (P010 -> I420-10bit) requested for the next
    /// recording. iPhone 12+ delivers 10-bit when the output format allows it.
    @Published var hdrRequested = false
    /// camera frame rate target: 30 or 60
    @Published var fpsTarget = 30
    private(set) var hdrActive = false

    private var framesSent = 0
    private var packets = 0
    private var bytesWritten = 0
    /// out-fps samples (one per ~0.5s) for the sparkline + min/avg/max
    @Published var fpsHist: [Double] = []
    private var lastSample = 0.0
    private(set) var fpsMin = 0.0, fpsMax = 0.0, fpsAvg = 0.0
    private var startTime = 0.0
    private var lastPacketTime = 0.0

    @Published private(set) var recording = false
    @Published var status = "ready"
    /// last finished .mp4 — share-sheet source
    @Published var lastShareURL: URL?

    // MARK: session

    private let presets: [AVCaptureSession.Preset] = [.hd1280x720, .hd1920x1080, .hd4K3840x2160]

    private var camInput: AVCaptureDeviceInput?
    /// camera selector: [label, position, type]
    let cameras: [(String, AVCaptureDevice.Position, AVCaptureDevice.DeviceType)] = [
        ("0.5x", .back, .builtInUltraWideCamera),
        ("1x", .back, .builtInWideAngleCamera),
        ("5x", .back, .builtInTelephotoCamera),
        ("front", .front, .builtInWideAngleCamera),
    ]

    override init() {
        super.init()
        session.sessionPreset = .hd1920x1080

        if let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
           let input = try? AVCaptureDeviceInput(device: dev),
           session.canAddInput(input) {
            session.addInput(input)
            camInput = input
            tuneDevice(dev)
        } else {
            status = "camera init failed"
            return
        }

        let fmt = videoOutput.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8Planar)
            ? kCVPixelFormatType_420YpCbCr8Planar
            : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: fmt]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(videoOutput) else {
            status = "output init failed"
            return
        }
        session.addOutput(videoOutput)

        if let conn = videoOutput.connection(with: .video), conn.isVideoRotationAngleSupported(90) {
            conn.videoRotationAngle = 90
        }
    }

    /// 30 fps capture + continuous autofocus/exposure (without these the
    /// focus point sticks wherever the device happened to open). The
    /// session preset picks the capture format; don't override it with a
    /// possibly low-res entry from dev.formats.
    private func tuneDevice(_ dev: AVCaptureDevice) {
        try? dev.lockForConfiguration()
        // HDR: pick a 10-bit-capable HDR format so the output can deliver
        // P010. dev.formats also lists tiny preview/HDR entries — pick one
        // matching the preset's resolution, else the largest available.
        let fps = fpsTarget
        if hdrRequested {
            let target = CMVideoFormatDescriptionGetDimensions(dev.activeFormat.formatDescription)
            let hdrFmts = dev.formats.filter {
                $0.isVideoHDRSupported
                    && $0.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= Double(fps) && $0.maxFrameRate >= Double(fps) })
            }
            func size(_ f: AVCaptureDevice.Format) -> Int {
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                return Int(d.width) * Int(d.height)
            }
            if let f = hdrFmts.first(where: { size($0) == Int(target.width) * Int(target.height) })
                     ?? hdrFmts.max(by: { size($0) < size($1) }) {
                dev.activeFormat = f
            }
        } else if fps > 30 {
            // 60fps: need a format that supports it (preset's choice may cap at 30)
            let target = CMVideoFormatDescriptionGetDimensions(dev.activeFormat.formatDescription)
            let cands = dev.formats.filter {
                $0.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= Double(fps) })
            }
            func size(_ f: AVCaptureDevice.Format) -> Int {
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                return Int(d.width) * Int(d.height)
            }
            if let f = cands.first(where: { size($0) == Int(target.width) * Int(target.height) })
                     ?? cands.max(by: { size($0) < size($1) }) {
                dev.activeFormat = f
            }
        }
        let t = CMTime(value: 1, timescale: CMTimeScale(fps))
        if dev.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= Double(fps) && $0.maxFrameRate >= Double(fps) }) {
            dev.activeVideoMinFrameDuration = t
            dev.activeVideoMaxFrameDuration = t
        }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        if dev.isSmoothAutoFocusSupported { dev.isSmoothAutoFocusEnabled = true }
        if dev.isExposureModeSupported(.continuousAutoExposure) { dev.exposureMode = .continuousAutoExposure }
        dev.unlockForConfiguration()
    }

    /// Swap the session's camera input at index `i` of `cameras`.
    func selectCamera(_ i: Int) {
        let (_, pos, type) = cameras[i]
        videoQueue.async {
            guard let dev = AVCaptureDevice.default(type, for: .video, position: pos),
                  let input = try? AVCaptureDeviceInput(device: dev) else { return }
            self.session.beginConfiguration()
            if let old = self.camInput { self.session.removeInput(old) }
            if self.session.canAddInput(input) {
                self.session.addInput(input)
                self.camInput = input
                self.tuneDevice(dev)
            } else if let old = self.camInput {
                self.session.addInput(old)
            }
            self.session.commitConfiguration()
        }
    }

    @MainActor
    func requestAccessAndStart() async {
        let ok = await AVCaptureDevice.requestAccess(for: .video)
        if ok {
            startSession()
        } else {
            status = "camera access denied"
        }
    }

    func startSession() {
        if !session.isRunning { session.startRunning() }
    }

    // MARK: recording

    func start(presetIndex: Int) {
        videoQueue.async { self._start(presetIndex: presetIndex) }
    }

    private func _start(presetIndex: Int) {
        let preset = presets[presetIndex]
        // ask for whatever 10-bit biplanar variant the output actually offers
        let tenBit: OSType? = [
            kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange,
            kCVPixelFormatType_420YpCbCr10BiPlanarFullRange,
        ].first { videoOutput.availableVideoPixelFormatTypes.contains($0) }
        let want10 = hdrRequested && tenBit != nil
        session.beginConfiguration()
        session.sessionPreset = preset
        if let dev = camInput?.device { tuneDevice(dev) }
        // P010 keeps 10-bit pixels through the capture chain on devices that
        // support HDR capture; anything else falls back to 8-bit NV12.
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
            want10 ? tenBit!
                   : videoOutput.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8Planar)
                     ? kCVPixelFormatType_420YpCbCr8Planar
                     : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        session.commitConfiguration()

        sampleIdx = 0
        framesSent = 0; packets = 0; bytesWritten = 0
        fpsHist = []; fpsMin = 0; fpsMax = 0; fpsAvg = 0; lastSample = 0
        startTime = CACurrentMediaTime()
        DispatchQueue.main.async {
            self.recording = true
            self.status = "starting…"
        }
        startSession()
    }

    /// The encoder + output files are created on the FIRST captured frame so
    /// the geometry always matches what the session actually delivers (HDR
    /// formats / presets can produce different dimensions than asked).
    private func lazyCreate(w: Int, h: Int, is10bit: Bool) -> Bool {
        hdrActive = is10bit

        var cfg = Mav1Config()
        mav1_default_config(&cfg, Int32(w), Int32(h))
        cfg.use_metal = 1
        cfg.turbo = 1
        // mirror mav1enc's fast path (restoration alone is ~2.8s/frame on-device)
        cfg.restoration = 0
        cfg.tx_search = 0
        cfg.tx_depth = 0
        cfg.switchable_interp = 0
        cfg.qindex = 100
        cfg.tile_rows_log2 = 1 // 8 tiles: entropy spreads onto E-cores
        cfg.fps_num = Int32(fpsTarget)
        cfg.fps_den = 1
        cfg.keyint = 0
        cfg.rc = MAV1_RC_CQP
        // phone memory: per-engine ME scratch ~197MB + qpel ~100MB at 1080p
        // (~4x at 4K) — 2 engines fit for <=1080p, single engine for 4K
        // phone memory: per-engine ~400MB at 1080p — 3 engines ~=1.2GB; the
        // extra engine lands on the 4 idle efficiency cores
        cfg.threads = w * h >= 3840 * 2160 ? 2 : 4 // 4th engine lands on E-cores too
        if is10bit {
            cfg.bitdepth = 10
            // iPhone captures HDR in HLG (Dolby Vision is a mux-level side
            // channel; the raw frames are HLG-tagged BT.2020 10-bit)
            cfg.color_primaries = 9          // BT.2020
            cfg.transfer_characteristics = 18 // HLG
            cfg.matrix_coefficients = 9      // BT.2020 NCL
            cfg.color_range = 0
            cfg.hdr = 1
            cfg.max_cll = 1000
            cfg.max_fall = 400
        } else {
            cfg.bitdepth = 8
        }

        var errbuf = [CChar](repeating: 0, count: 256)
        enc = errbuf.withUnsafeMutableBufferPointer { mav1_create(&cfg, $0.baseAddress, $0.count) }
        guard enc != nil else {
            DispatchQueue.main.async { self.recording = false; self.status = "mav1_create failed: \(String(cString: errbuf))" }
            return false
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        let url = docs.appendingPathComponent("mav1-\(df.string(from: Date())).ivf")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            mav1_destroy(enc); enc = nil
            DispatchQueue.main.async { self.recording = false; self.status = "file open failed" }
            return false
        }
        file = fh
        outURL = url

        // IVF header (frame count patched at stop)
        var hdr = Data()
        hdr.append(contentsOf: [0x44, 0x4B, 0x49, 0x46]) // DKIF
        appendLE(&hdr, UInt16(0)); appendLE(&hdr, UInt16(32))
        hdr.append(contentsOf: [0x41, 0x56, 0x30, 0x31]) // AV01
        appendLE(&hdr, UInt16(w)); appendLE(&hdr, UInt16(h))
        appendLE(&hdr, UInt32(fpsTarget)); appendLE(&hdr, UInt32(1))
        appendLE(&hdr, UInt32(0)); appendLE(&hdr, UInt32(0))
        fh.write(hdr)

        // .mov mux for Photos: pass-through av01 samples in an MP4 container
        // (the A17+ decoder plays them back in the gallery)
        movURL = url.deletingPathExtension().appendingPathExtension("mp4")
        try? FileManager.default.removeItem(at: movURL!)
        setupMovWriter(width: w, height: h, bitdepth: cfg.bitdepth, enc: enc!)

        DispatchQueue.main.async {
            self.status = "recording \(w)x\(h)\(is10bit ? " HDR10" : "") turbo"
        }
        return true
    }

    func stop() {
        // stop feeding frames immediately so the drain only flushes what's
        // already in flight; the UI goes idle at once
        DispatchQueue.main.async {
            self.recording = false
            self.status = "stopping…"
        }
        videoQueue.async { self._stop() }
    }

    private func _stop() {
        guard let enc else { return }
        // flush
        var planes: [UnsafePointer<UInt8>?] = [nil, nil, nil]
        var strides: [Int32] = [0, 0, 0]
        planes.withUnsafeMutableBufferPointer { p in
            strides.withUnsafeMutableBufferPointer { s in
                mav1_send_frame(enc, nil, s.baseAddress)
            }
        }
        var out: UnsafeMutablePointer<UInt8>?
        var size = 0
        let deadline = CACurrentMediaTime() + 20
        while mav1_receive_packet_wait(enc, &out, &size, 500) == 0 {
            writePacket(out, size: size)
            if CACurrentMediaTime() > deadline { break }
        }
        patchFrameCount()
        try? file?.close()
        file = nil
        finishMov()
        mav1_destroy(enc)
        self.enc = nil
        uBuf.map { free($0) }; uBuf = nil
        vBuf.map { free($0) }; vBuf = nil
        yBuf.map { free($0) }; yBuf = nil
        uvBytes = 0; yBytes = 0

        let name = outURL?.lastPathComponent ?? ""
        DispatchQueue.main.async {
            self.recording = false
            self.status = "saved \(name) (\(self.packets) frames)"
        }
    }

    // MARK: capture

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard recording else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)

        if enc == nil {
            let is10 = fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                    || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange
            guard lazyCreate(w: w, h: h, is10bit: is10) else { return }
        }
        guard let enc, let file else { return }

        var pl: [UnsafePointer<UInt8>?]
        var st: [Int32]

        if fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
            || fmt == kCVPixelFormatType_420YpCbCr10BiPlanarFullRange {
            // P010: uint16 MSB-justified -> LSB-justified I420-10bit.
            let cw = (w + 1) / 2, ch = (h + 1) / 2
            if yBytes < w * h * 2 {
                yBuf?.deallocate()
                yBuf = UnsafeMutableRawPointer.allocate(byteCount: w * h * 2, alignment: 16)
                yBytes = w * h * 2
            }
            if uvBytes < cw * ch * 2 {
                uBuf?.deallocate(); vBuf?.deallocate()
                uBuf = UnsafeMutableRawPointer.allocate(byteCount: cw * ch * 2, alignment: 16)
                vBuf = UnsafeMutableRawPointer.allocate(byteCount: cw * ch * 2, alignment: 16)
                uvBytes = cw * ch * 2
            }
            guard let yb = yBuf, let ub = uBuf, let vb = vBuf,
                  let yp = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
                  let uvp = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { return }
            let ys = CVPixelBufferGetBytesPerRowOfPlane(pb, 0) / 2
            let uvs = CVPixelBufferGetBytesPerRowOfPlane(pb, 1) / 2
            let src = yp.assumingMemoryBound(to: UInt16.self)
            yb.withMemoryRebound(to: UInt16.self, capacity: w * h) { dst in
                for y in 0..<h {
                    let s = src + y * ys, d = dst + y * w
                    for x in 0..<w { d[x] = s[x] >> 6 }
                }
            }
            let uvr = uvp.assumingMemoryBound(to: UInt16.self)
            ub.withMemoryRebound(to: UInt16.self, capacity: cw * ch) { up in
                vb.withMemoryRebound(to: UInt16.self, capacity: cw * ch) { vp in
                    for y in 0..<ch {
                        let s = uvr + y * uvs, d = y * cw
                        for x in 0..<cw {
                            up[d + x] = s[2 * x] >> 6
                            vp[d + x] = s[2 * x + 1] >> 6
                        }
                    }
                }
            }
            pl = [
                UnsafePointer(yb.assumingMemoryBound(to: UInt8.self)),
                UnsafePointer(ub.assumingMemoryBound(to: UInt8.self)),
                UnsafePointer(vb.assumingMemoryBound(to: UInt8.self)),
            ]
            st = [Int32(w * 2), Int32(cw * 2), Int32(cw * 2)]
        } else if fmt == kCVPixelFormatType_420YpCbCr8Planar {
            pl = [
                CVPixelBufferGetBaseAddressOfPlane(pb, 0).map { UnsafePointer($0.assumingMemoryBound(to: UInt8.self)) },
                CVPixelBufferGetBaseAddressOfPlane(pb, 1).map { UnsafePointer($0.assumingMemoryBound(to: UInt8.self)) },
                CVPixelBufferGetBaseAddressOfPlane(pb, 2).map { UnsafePointer($0.assumingMemoryBound(to: UInt8.self)) },
            ]
            st = [
                Int32(CVPixelBufferGetBytesPerRowOfPlane(pb, 0)),
                Int32(CVPixelBufferGetBytesPerRowOfPlane(pb, 1)),
                Int32(CVPixelBufferGetBytesPerRowOfPlane(pb, 2)),
            ]
        } else {
            // NV12: split the interleaved UV plane
            guard let yp = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
                  let uvp = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { return }
            let cw = (w + 1) / 2, ch = (h + 1) / 2, n = cw * ch
            if uvBytes < n {
                uBuf?.deallocate(); vBuf?.deallocate()
                uBuf = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: 16)
                vBuf = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: 16)
                uvBytes = n
            }
            guard let u = uBuf, let v = vBuf else { return }
            let uStride = CVPixelBufferGetBytesPerRowOfPlane(pb, 1)
            let uv = uvp.assumingMemoryBound(to: UInt8.self)
            u.withMemoryRebound(to: UInt8.self, capacity: n) { up in
                v.withMemoryRebound(to: UInt8.self, capacity: n) { vp in
                    for y in 0..<ch {
                        let s = uv + y * uStride, d = y * cw
                        for x in 0..<cw {
                            up[d + x] = s[2 * x]
                            vp[d + x] = s[2 * x + 1]
                        }
                    }
                }
            }
            pl = [
                CVPixelBufferGetBaseAddressOfPlane(pb, 0).map { UnsafePointer($0.assumingMemoryBound(to: UInt8.self)) },
                UnsafePointer(u.assumingMemoryBound(to: UInt8.self)),
                UnsafePointer(v.assumingMemoryBound(to: UInt8.self)),
            ]
            st = [
                Int32(CVPixelBufferGetBytesPerRowOfPlane(pb, 0)),
                Int32(cw),
                Int32(cw),
            ]
        }

        pl.withUnsafeMutableBufferPointer { p in
            st.withUnsafeMutableBufferPointer { s in
                mav1_send_frame(enc, p.baseAddress, s.baseAddress)
            }
        }
        framesSent += 1

        var out: UnsafeMutablePointer<UInt8>?
        var size = 0
        while mav1_receive_packet(enc, &out, &size) == 0 {
            writePacket(out, size: size)
        }
        if true {
            let el = CACurrentMediaTime() - startTime
            let inFps = Double(framesSent) / el
            let outFps = Double(packets) / el
            let mbps = Double(bytesWritten) * 8 / el / 1e6
            let backend = String(cString: mav1_backend_name(enc))
            let pend = mav1_pending_packets(enc)
            var stage = ""
            if let st = mav1_last_stats(enc) {
                stage = String(format: "a%.0f r%.0f f%.0f e%.0fms",
                               st.pointee.t_analysis * 1000, st.pointee.t_recon * 1000,
                               st.pointee.t_filter * 1000, st.pointee.t_entropy * 1000)
            }
            // fps history (~2Hz) for min/avg/max + sparkline
            if el - lastSample >= 0.5 {
                lastSample = el
                fpsHist.append(outFps)
                if fpsHist.count > 240 { fpsHist.removeFirst(fpsHist.count - 240) }
                if let mn = fpsHist.min(), let mx = fpsHist.max() {
                    fpsMin = mn; fpsMax = mx; fpsAvg = fpsHist.reduce(0, +) / Double(fpsHist.count)
                }
            }
            DispatchQueue.main.async {
                self.status = String(format:
                    "%@ | in %df/out %d | %.1fs | %.0f→%.1ffps | %.2fMB %.1fMbps | q%d\n%@\nmin %.0f avg %.0f max %.0f fps",
                    backend, self.framesSent, self.packets, el, inFps, outFps,
                    Double(self.bytesWritten) / 1e6, mbps, pend, stage,
                    self.fpsMin, self.fpsAvg, self.fpsMax)
            }
        }
    }

    // MARK: mov mux (av01 passthrough for Photos)

    private func setupMovWriter(width: Int, height: Int, bitdepth: Int32, enc: OpaquePointer) {
        guard let movURL else { return }
        // AV1 Codec Configuration Record: marker|version, seq_profile|level,
        // tier|bitdepth|subsampling flags, initial delay, then the sequence
        // header OBU as the config payload.
        var av1c = Data([0x81, UInt8(0 << 5 | 40), UInt8(bitdepth > 8 ? 0x4C : 0x0C), 0x00])
        var sh = [UInt8](repeating: 0, count: 256)
        let shLen = mav1_sequence_header(enc, &sh)
        av1c.append(contentsOf: sh[0..<Int(shLen)])
        let atoms: [String: Data] = ["av1C": av1c]
        let ext: [String: Any] = [
            kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms as String: atoms,
        ]
        var fd: CMFormatDescription?
        let rc = CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault, codecType: kCMVideoCodecType_AV1,
            width: Int32(width), height: Int32(height),
            extensions: ext as CFDictionary, formatDescriptionOut: &fd)
        guard rc == noErr, let fd else { return }
        av1Desc = fd

        guard let w = try? AVAssetWriter(outputURL: movURL, fileType: .mp4) else { return }
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: nil, sourceFormatHint: fd)
        input.expectsMediaDataInRealTime = true
        guard w.canAdd(input) else { return }
        w.add(input)
        guard w.startWriting() else { return }
        w.startSession(atSourceTime: .zero)
        movWriter = w
        movInput = input
    }

    private func appendMovSample(_ data: UnsafeMutablePointer<UInt8>, size: Int) {
        guard let inp = movInput, let fd = av1Desc, movWriter?.status == .writing else { return }
        var bb: CMBlockBuffer?
        guard CMBlockBufferCreateEmpty(allocator: kCFAllocatorDefault, capacity: 0, flags: 0, blockBufferOut: &bb) == noErr,
              let bb else { return }
        guard CMBlockBufferAppendMemoryBlock(bb, memoryBlock: nil, length: size, blockAllocator: nil,
                                             customBlockSource: nil, offsetToData: 0, dataLength: size, flags: 0) == noErr,
              CMBlockBufferReplaceDataBytes(with: data, blockBuffer: bb, offsetIntoDestination: 0, dataLength: size) == noErr else { return }
        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CMTimeScale(fpsTarget)),
            presentationTimeStamp: CMTime(value: sampleIdx, timescale: CMTimeScale(fpsTarget)),
            decodeTimeStamp: .invalid)
        var sz = size
        var sb: CMSampleBuffer?
        guard CMSampleBufferCreateReady(allocator: kCFAllocatorDefault, dataBuffer: bb, formatDescription: fd,
                                        sampleCount: 1, sampleTimingEntryCount: 1, sampleTimingArray: &timing,
                                        sampleSizeEntryCount: 1, sampleSizeArray: &sz, sampleBufferOut: &sb) == noErr,
              let sb else { return }
        sampleIdx += 1
        inp.append(sb)
    }

    private func finishMov() {
        guard movWriter != nil else { return }
        let url = movURL
        movInput?.markAsFinished()
        let w = movWriter
        movWriter = nil
        movInput = nil
        av1Desc = nil
        w?.finishWriting { [weak self] in
            guard let self, let url else { return }
            guard w?.status == .completed else {
                DispatchQueue.main.async {
                    self.status = "mp4 failed: \(w?.error?.localizedDescription ?? "\(String(describing: w?.status))")"
                }
                return
            }
            DispatchQueue.main.async { self.lastShareURL = url }
        }
    }

    // MARK: IVF

    private func writePacket(_ data: UnsafeMutablePointer<UInt8>?, size: Int) {
        guard let file, let data else { return }
        var hdr = Data(capacity: 12)
        appendLE(&hdr, UInt32(size))
        appendLE(&hdr, UInt64(packets))
        file.write(hdr)
        file.write(Data(bytes: data, count: size))
        appendMovSample(data, size: size)
        free(data)
        packets += 1
        bytesWritten += size + 12
    }

    private func patchFrameCount() {
        guard let file else { return }
        try? file.seek(toOffset: 24)
        var d = Data(count: 4)
        appendLE(&d, UInt32(packets))
        file.write(d)
    }

    private func appendLE<T: FixedWidthInteger>(_ d: inout Data, _ v: T) {
        var x = v.littleEndian
        withUnsafeBytes(of: &x) { d.append(contentsOf: $0) }
    }
}

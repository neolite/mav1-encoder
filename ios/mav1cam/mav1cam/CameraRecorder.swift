import AVFoundation
import Foundation
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
    private var uBuf: UnsafeMutableRawPointer?
    private var vBuf: UnsafeMutableRawPointer?
    private var uvBytes = 0
    private var yBuf: UnsafeMutableRawPointer?
    private var yBytes = 0
    /// 10-bit HDR capture (P010 -> I420-10bit) requested for the next
    /// recording. iPhone 12+ delivers 10-bit when the output format allows it.
    @Published var hdrRequested = false
    private(set) var hdrActive = false

    private var framesSent = 0
    private var packets = 0
    private var bytesWritten = 0
    private var startTime = 0.0
    private var lastPacketTime = 0.0

    @Published private(set) var recording = false
    @Published var status = "ready"

    // MARK: session

    private let presets: [AVCaptureSession.Preset] = [.hd1280x720, .hd1920x1080, .hd4K3840x2160]

    override init() {
        super.init()
        session.sessionPreset = .hd1920x1080

        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev),
              session.canAddInput(input) else {
            status = "camera init failed"
            return
        }
        session.addInput(input)

        // request 30 fps
        for f in dev.formats where f.videoSupportedFrameRateRanges.contains(where: { $0.minFrameRate <= 30 && $0.maxFrameRate >= 30 }) && CMFormatDescriptionGetMediaSubType(f.formatDescription) == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
            try? dev.lockForConfiguration()
            dev.activeFormat = f
            dev.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            dev.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            dev.unlockForConfiguration()
            break
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
        let want10 = hdrRequested && videoOutput.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange)
        session.beginConfiguration()
        session.sessionPreset = preset
        // P010 keeps 10-bit pixels through the capture chain on devices that
        // support HDR capture; anything else falls back to 8-bit NV12.
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
            want10 ? kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
                   : videoOutput.availableVideoPixelFormatTypes.contains(kCVPixelFormatType_420YpCbCr8Planar)
                     ? kCVPixelFormatType_420YpCbCr8Planar
                     : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        session.commitConfiguration()
        hdrActive = want10

        let dim = preset == .hd4K3840x2160 ? (3840, 2160) : preset == .hd1920x1080 ? (1920, 1080) : (1280, 720)

        var cfg = Mav1Config()
        mav1_default_config(&cfg, Int32(dim.0), Int32(dim.1))
        cfg.use_metal = 1
        cfg.turbo = 1
        cfg.qindex = 100
        cfg.fps_num = 30
        cfg.fps_den = 1
        cfg.keyint = 0
        cfg.rc = MAV1_RC_CQP
        // phone memory: 4 full engines ~1.4GB at 1080p; 2 fit comfortably
        cfg.threads = 2
        if hdrActive {
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
            DispatchQueue.main.async { self.status = "mav1_create failed: \(String(cString: errbuf))" }
            return
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        let url = docs.appendingPathComponent("mav1-\(df.string(from: Date())).ivf")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let fh = try? FileHandle(forWritingTo: url) else {
            mav1_destroy(enc); enc = nil
            DispatchQueue.main.async { self.status = "file open failed" }
            return
        }
        file = fh
        outURL = url

        // IVF header (frame count patched at stop)
        var hdr = Data()
        hdr.append(contentsOf: [0x44, 0x4B, 0x49, 0x46]) // DKIF
        appendLE(&hdr, UInt16(0)); appendLE(&hdr, UInt16(32))
        hdr.append(contentsOf: [0x41, 0x56, 0x30, 0x31]) // AV01
        appendLE(&hdr, UInt16(dim.0)); appendLE(&hdr, UInt16(dim.1))
        appendLE(&hdr, UInt32(30)); appendLE(&hdr, UInt32(1))
        appendLE(&hdr, UInt32(0)); appendLE(&hdr, UInt32(0))
        fh.write(hdr)

        framesSent = 0; packets = 0; bytesWritten = 0
        startTime = CACurrentMediaTime()
        DispatchQueue.main.async {
            self.recording = true
            self.status = "recording \(dim.0)x\(dim.1)\(self.hdrActive ? " HDR10" : "") turbo"
        }
        startSession()
    }

    func stop() {
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
        while mav1_receive_packet_wait(enc, &out, &size, 2000) == 0 {
            writePacket(out, size: size)
        }
        patchFrameCount()
        try? file?.close()
        file = nil
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
        guard recording, let enc, let file else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let fmt = CVPixelBufferGetPixelFormatType(pb)

        var pl: [UnsafePointer<UInt8>?]
        var st: [Int32]

        if fmt == kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange {
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
        if framesSent % 15 == 0 {
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
            DispatchQueue.main.async {
                self.status = String(format:
                    "%@ | in %df/out %d | %.1fs | %.0f→%.1ffps | %.2fMB %.1fMbps | q%d\n%@",
                    backend, self.framesSent, self.packets, el, inFps, outFps,
                    Double(self.bytesWritten) / 1e6, mbps, pend, stage)
            }
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

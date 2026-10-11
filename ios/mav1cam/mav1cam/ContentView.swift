import SwiftUI
import AVFoundation
import UIKit

struct PreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewUIView {
        let v = PreviewUIView()
        v.previewLayer.session = session
        v.previewLayer.videoGravity = .resizeAspectFill
        return v
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {}

    final class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

struct ContentView: View {
    @StateObject private var rec = CameraRecorder()
    @State private var preset = 1 // 0: 720p, 1: 1080p, 2: 4K

    var body: some View {
        ZStack {
            PreviewView(session: rec.session)
                .ignoresSafeArea()

            VStack {
                Picker("Preset", selection: $preset) {
                    Text("720p").tag(0)
                    Text("1080p").tag(1)
                    Text("4K").tag(2)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                .padding(.top, 8)
                .disabled(rec.recording)

                Toggle("HDR 10-bit", isOn: $rec.hdrRequested)
                    .font(.caption.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal)
                    .padding(.top, 4)
                    .disabled(rec.recording)

                Spacer()

                Text(rec.status)
                    .font(.system(.caption, design: .monospaced))
                    .padding(8)
                    .background(.black.opacity(0.55))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.bottom, 4)

                Button(action: {
                    if rec.recording {
                        rec.stop()
                    } else {
                        rec.start(presetIndex: preset)
                    }
                }) {
                    Circle()
                        .strokeBorder(.white, lineWidth: 4)
                        .fill(rec.recording ? .red : .clear)
                        .frame(width: 72, height: 72)
                        .padding(.bottom, 24)
                }
            }
        }
        .preferredColorScheme(.dark)
        .task {
            await rec.requestAccessAndStart()
        }
    }
}

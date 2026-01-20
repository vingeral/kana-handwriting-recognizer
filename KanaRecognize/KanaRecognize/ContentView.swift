import SwiftUI
import PencilKit
import CoreML
import Vision
import UIKit

// 改成你的模型类名
typealias KanaModel = KanaRecognizeV4

private let MODEL_SIDE = 360
private let TOPK = 10
private let INK_WIDTH: CGFloat = 22

struct ContentView: View {
    @State private var canvas = PKCanvasView()
    @State private var hasInk = false

    @State private var topK: [(String, Double)] = []
    @State private var status: String = "Ready."

    private let vnModel: VNCoreMLModel = {
        do {
            let coreML = try KanaModel(configuration: MLModelConfiguration()).model
            return try VNCoreMLModel(for: coreML)
        } catch {
            fatalError("Failed to create VNCoreMLModel: \(error)")
        }
    }()

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                header
                statusBar

                writingArea
                    .padding(.horizontal)

                controlsRow
                    .padding(.horizontal)

                predictionsRow
                    .padding(.horizontal)
                    .padding(.bottom, 10)

                Spacer(minLength: 8)
            }
            .padding(.top, 8)
            .navigationBarTitleDisplayMode(.inline)
            .onAppear { setupCanvas(canvas) }
            .preferredColorScheme(.light)
        }
    }

    // MARK: - UI

    private var header: some View {
        Text("Kana Recognize Demo")
            .font(.system(size: 34, weight: .bold))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
    }

    private var statusBar: some View {
        Text(status)
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            .lineLimit(2)
    }

    private var writingArea: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color.white)
                .overlay(
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(Color.black.opacity(0.10), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.06), radius: 10, x: 0, y: 6)

            PencilCanvasView(canvas: $canvas, hasInk: $hasInk, inkWidth: INK_WIDTH) { drawing in
                if drawing.strokes.isEmpty {
                    topK = []
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 22))

            GeometryReader { g in
                let side = min(g.size.width, g.size.height) * 0.80
                let rect = CGRect(
                    x: (g.size.width - side) / 2,
                    y: (g.size.height - side) / 2,
                    width: side,
                    height: side
                )
                GuideBox(rect: rect)
                    .allowsHitTesting(false)
            }
        }
        .frame(maxWidth: .infinity)
        .aspectRatio(0.95, contentMode: .fit)
    }

    private var controlsRow: some View {
        HStack(spacing: 14) {
            Button("Clear") { clearAll() }
                .buttonStyle(.bordered)
                .controlSize(.large)

            Button {
                recognize()
            } label: {
                Text("Recognize")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!hasInk)
        }
    }

    private var predictionsRow: some View {
        let columns = Array(repeating: GridItem(.flexible(), spacing: 10), count: 5)

        return LazyVGrid(columns: columns, spacing: 10) {
            ForEach(0..<TOPK, id: \.self) { idx in
                let item = idx < topK.count ? topK[idx] : ("—", 0.0)
                PredictionTile(rank: idx + 1, label: item.0, confidence: item.1)
            }
        }
    }

    // MARK: - Actions

    private func clearAll() {
        canvas.drawing = PKDrawing()
        hasInk = false
        topK = []
        status = "Cleared."
    }

    private func recognize() {
        let drawing = canvas.drawing
        guard !drawing.strokes.isEmpty else {
            status = "E_NO_INK: No ink."
            return
        }

        let input = Normalizer.renderNormalizedBW(drawing: drawing, side: MODEL_SIDE)
        guard let cg = input.cgImage else {
            status = "E_CGIMAGE: Failed to get CGImage."
            return
        }

        status = "Running…"
        topK = []

        let request = VNCoreMLRequest(model: vnModel) { req, err in
            DispatchQueue.main.async {
                if let err {
                    self.status = "E_VISION: \(err.localizedDescription)"
                    return
                }

                if let results = req.results as? [VNClassificationObservation], !results.isEmpty {
                    let sorted = results.sorted { $0.confidence > $1.confidence }
                    self.topK = Array(sorted.prefix(TOPK)).map { ($0.identifier, Double($0.confidence)) }

                    if let best = self.topK.first {
                        self.status = "OK: Best \(best.0)  \(String(format: "%.1f%%", best.1 * 100))"
                    } else {
                        self.status = "E_EMPTY: No result."
                    }
                    return
                }

                self.status = "E_OUTPUT: No VNClassificationObservation."
            }
        }

        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cgImage: cg, orientation: .up, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([request]) }
            catch {
                DispatchQueue.main.async {
                    self.status = "E_PREDICT: \(error.localizedDescription)"
                }
            }
        }
    }

    private func setupCanvas(_ canvas: PKCanvasView) {
        canvas.overrideUserInterfaceStyle = .light
        canvas.backgroundColor = .white
        canvas.isOpaque = true

        canvas.isUserInteractionEnabled = true
        canvas.drawingPolicy = .anyInput

        canvas.tool = PKInkingTool(.pen, color: .black, width: INK_WIDTH)

        canvas.alwaysBounceVertical = false
        canvas.alwaysBounceHorizontal = false
        canvas.minimumZoomScale = 1
        canvas.maximumZoomScale = 1
    }
}

// MARK: - Prediction Tile

private struct PredictionTile: View {
    let rank: Int
    let label: String
    let confidence: Double

    var body: some View {
        VStack(spacing: 8) {
            Text(label)
                .font(.system(size: 28, weight: .semibold))
                .frame(maxWidth: .infinity)
                .minimumScaleFactor(0.5)
                .lineLimit(1)

            ProgressView(value: confidence)
                .frame(maxWidth: .infinity)

            Text(String(format: "%.1f%%", confidence * 100))
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity)
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(.black.opacity(0.08), lineWidth: 1))
        .accessibilityLabel("Rank \(rank) \(label) \(Int(confidence * 100)) percent")
    }
}

// MARK: - PencilKit wrapper

struct PencilCanvasView: UIViewRepresentable {
    @Binding var canvas: PKCanvasView
    @Binding var hasInk: Bool
    let inkWidth: CGFloat
    let onDrawingChanged: (PKDrawing) -> Void

    private func applyPreferredTool(to canvas: PKCanvasView) {
        // Tool Picker 有时会把 tool 换回系统上次使用的细笔，这里强制回到我们的粗笔
        canvas.tool = PKInkingTool(.pen, color: .black, width: inkWidth)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(hasInk: $hasInk, onDrawingChanged: onDrawingChanged, reapplyTool: { [inkWidth] canvas in
            canvas.tool = PKInkingTool(.pen, color: .black, width: inkWidth)
        })
    }

    func makeUIView(context: Context) -> PKCanvasView {
        canvas.delegate = context.coordinator
        canvas.isUserInteractionEnabled = true
        canvas.drawingPolicy = .anyInput
        applyPreferredTool(to: canvas)

        DispatchQueue.main.async {
            context.coordinator.attachToolPickerIfPossible(to: canvas)
        }
        return canvas
    }

    func updateUIView(_ uiView: PKCanvasView, context: Context) {
        applyPreferredTool(to: uiView)
        DispatchQueue.main.async {
            context.coordinator.attachToolPickerIfPossible(to: uiView)
        }
    }

    final class Coordinator: NSObject, PKCanvasViewDelegate {
        private var hasInk: Binding<Bool>
        private let onDrawingChanged: (PKDrawing) -> Void
        private let reapplyTool: (PKCanvasView) -> Void

        init(hasInk: Binding<Bool>, onDrawingChanged: @escaping (PKDrawing) -> Void, reapplyTool: @escaping (PKCanvasView) -> Void) {
            self.hasInk = hasInk
            self.onDrawingChanged = onDrawingChanged
            self.reapplyTool = reapplyTool
        }

        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            let has = !canvasView.drawing.strokes.isEmpty
            if hasInk.wrappedValue != has {
                DispatchQueue.main.async { self.hasInk.wrappedValue = has }
            }
            DispatchQueue.main.async { self.onDrawingChanged(canvasView.drawing) }
        }

        func attachToolPickerIfPossible(to canvas: PKCanvasView) {
            guard let window = canvas.window ?? UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .flatMap({ $0.windows })
                .first(where: { $0.isKeyWindow }) else { return }

            let picker = PKToolPicker.shared(for: window)
            picker?.addObserver(canvas)
            canvas.becomeFirstResponder()
            picker?.setVisible(false, forFirstResponder: canvas)
            reapplyTool(canvas)
        }
    }
}

// MARK: - Normalizer (force white bg + black ink)

enum Normalizer {
    static func renderNormalizedBW(drawing: PKDrawing, side: Int) -> UIImage {
        let bounds = drawing.bounds.insetBy(dx: -10, dy: -10)
        let base = drawing.image(from: bounds, scale: 2.0)

        let targetSize = CGSize(width: side, height: side)
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        defer { UIGraphicsEndImageContext() }

        UIColor.white.setFill()
        UIRectFill(CGRect(origin: .zero, size: targetSize))

        let fitMax = CGFloat(side) * 0.82
        let scaleToFit = min(fitMax / max(base.size.width, 1),
                             fitMax / max(base.size.height, 1))
        let w = base.size.width * scaleToFit
        let h = base.size.height * scaleToFit
        let x = (targetSize.width - w) / 2
        let y = (targetSize.height - h) / 2

        base.draw(in: CGRect(x: x, y: y, width: w, height: h))
        let composed = UIGraphicsGetImageFromCurrentImageContext()!
        return forceBlackInkWhiteBG(composed)
    }

    static func forceBlackInkWhiteBG(_ image: UIImage) -> UIImage {
        guard let cg = image.cgImage else { return image }

        let width = cg.width
        let height = cg.height
        let bytesPerRow = width * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        var data = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let ctx = CGContext(
            data: &data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return image }

        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

        let threshold: UInt8 = 245

        for y in 0..<height {
            for x in 0..<width {
                let i = y * bytesPerRow + x * 4
                let r = data[i]
                let g = data[i + 1]
                let b = data[i + 2]
                let a = data[i + 3]

                if a == 0 {
                    data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255
                    continue
                }

                if r < threshold || g < threshold || b < threshold {
                    data[i] = 0; data[i+1] = 0; data[i+2] = 0; data[i+3] = 255
                } else {
                    data[i] = 255; data[i+1] = 255; data[i+2] = 255; data[i+3] = 255
                }
            }
        }

        guard let outCG = ctx.makeImage() else { return image }
        return UIImage(cgImage: outCG, scale: image.scale, orientation: image.imageOrientation)
    }
}

// MARK: - Guide box

struct GuideBox: View {
    let rect: CGRect
    var body: some View {
        Path { p in
            p.addRoundedRect(in: rect, cornerSize: CGSize(width: 12, height: 12))
        }
        .stroke(.black.opacity(0.22), style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
    }
}

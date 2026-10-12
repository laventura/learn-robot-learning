// OCR every page of an image-only PDF (such as the rebuilt lecture slides) with
// macOS's built-in Vision framework. Nothing to install.
//
// Usage: swift tools/ocr_pdf.swift refs/week03_slides.pdf > refs/slides-text/w03.txt
// Output: "=== page N" then the page's text lines, top to bottom.
import Foundation
import PDFKit
import Vision

let args = CommandLine.arguments
guard args.count == 2, let doc = PDFDocument(url: URL(fileURLWithPath: args[1])) else {
    FileHandle.standardError.write("usage: ocr_pdf.swift <pdf>\n".data(using: .utf8)!)
    exit(1)
}

for i in 0..<doc.pageCount {
    guard let page = doc.page(at: i) else { continue }
    // Render at 2x the page size so small slide text stays legible to OCR.
    let box = page.bounds(for: .mediaBox)
    let size = CGSize(width: box.width * 2, height: box.height * 2)
    let image = page.thumbnail(of: size, for: .mediaBox)
    guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { continue }

    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    try? VNImageRequestHandler(cgImage: cg).perform([request])

    // Vision's y axis points up, so sort by descending y to read top to bottom.
    let lines = (request.results ?? [])
        .sorted { $0.boundingBox.minY > $1.boundingBox.minY }
        .compactMap { $0.topCandidates(1).first?.string }
    print("=== page \(i + 1)")
    print(lines.joined(separator: "\n"))
}

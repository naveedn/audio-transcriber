import AVFoundation
import FluidAudio
import Foundation

// MARK: - CLI Entrypoint

@main
struct ParakeetTranscriberCLI {
    static func main() async {
        do {
            let arguments = try Arguments.parse()
            let runner = ParakeetRunner(arguments: arguments)
            let transcription = try await runner.run()
            try transcription.write(to: arguments.outputPath)
        } catch {
            fputs("❌ Parakeet error: \(error.localizedDescription)\n", stderr)
            exit(EXIT_FAILURE)
        }
    }
}

// MARK: - Argument parsing

struct Arguments {
    let audioPath: URL
    let diarizationPath: URL
    let outputPath: URL
    let modelVersion: AsrModelVersion
    let modelsRoot: URL?
    let minSegmentSeconds: Double
    let language: String

    static func parse() throws -> Arguments {
        var audio: String?
        var diarization: String?
        var output: String?
        var modelsRoot: String?
        var modelVersion: AsrModelVersion = .v2
        var minSegmentSeconds = 1.0
        var language = "en"

        let arguments = Array(CommandLine.arguments.dropFirst())
        var index = 0

        func nextValue(after currentIndex: inout Int) throws -> String {
            let nextIndex = currentIndex + 1
            guard nextIndex < arguments.count else {
                throw CLIError.missingValue(arguments[currentIndex])
            }
            currentIndex = nextIndex
            return arguments[nextIndex]
        }

        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--audio":
                audio = try nextValue(after: &index)
            case "--diarization":
                diarization = try nextValue(after: &index)
            case "--output":
                output = try nextValue(after: &index)
            case "--models-root":
                modelsRoot = try nextValue(after: &index)
            case "--model-version":
                let value = try nextValue(after: &index).lowercased()
                if value.contains("v2") || value == "2" {
                    modelVersion = .v2
                } else if value.contains("v3") || value == "3" {
                    modelVersion = .v3
                } else {
                    throw CLIError.invalidValue("--model-version", value)
                }
            case "--min-seconds":
                let value = try nextValue(after: &index)
                guard let parsed = Double(value), parsed > 0 else {
                    throw CLIError.invalidValue("--min-seconds", value)
                }
                minSegmentSeconds = parsed
            case "--language":
                language = try nextValue(after: &index)
            case "--help", "-h":
                Arguments.printUsage()
                exit(EXIT_SUCCESS)
            default:
                throw CLIError.unknownOption(arg)
            }
            index += 1
        }

        guard let audio, let diarization else {
            throw CLIError.missingRequired("--audio/--diarization")
        }

        let audioURL = URL(fileURLWithPath: audio)
        let diarizationURL = URL(fileURLWithPath: diarization)

        let resolvedOutput: URL
        if let output {
            resolvedOutput = URL(fileURLWithPath: output)
        } else {
            resolvedOutput = audioURL
                .deletingPathExtension()
                .appendingPathExtension("parakeet.json")
        }

        return Arguments(
            audioPath: audioURL,
            diarizationPath: diarizationURL,
            outputPath: resolvedOutput,
            modelVersion: modelVersion,
            modelsRoot: modelsRoot.map { URL(fileURLWithPath: $0, isDirectory: true) },
            minSegmentSeconds: minSegmentSeconds,
            language: language
        )
    }

    static func printUsage() {
        let usage = """
        Usage: parakeet-transcriber --audio <wav> --diarization <json> [options]

        Required:
          --audio <path>          WAV file that Senko generated per track
          --diarization <path>    Matching Senko diarization JSON

        Options:
          --output <path>         Destination JSON (defaults to <audio>.parakeet.json)
          --models-root <dir>     Directory containing the downloaded Parakeet repo folder
          --model-version <v2|v3> Choose Parakeet ASR model (default: v2)
          --min-seconds <float>   Skip segments shorter than this many seconds (default: 1.0)
          --language <code>       Language code stored in the output (default: en)
          --help                  Show this help message
        """
        print(usage)
    }

    enum CLIError: LocalizedError {
        case missingRequired(String)
        case missingValue(String)
        case invalidValue(String, String)
        case unknownOption(String)

        var errorDescription: String? {
            switch self {
            case .missingRequired(let flag):
                return "Missing required option(s): \(flag)"
            case .missingValue(let flag):
                return "Missing value for option \(flag)"
            case .invalidValue(let flag, let value):
                return "Invalid value '\(value)' provided for \(flag)"
            case .unknownOption(let option):
                return "Unknown option: \(option)"
            }
        }
    }
}

// MARK: - Runner

struct ParakeetRunner {
    let arguments: Arguments
    private let sampleRate = 16_000

    func run() async throws -> TranscriptionOutput {
        let diarization = try DiarizationPayload.load(from: arguments.diarizationPath)
        let trackName = arguments.audioPath.deletingPathExtension().lastPathComponent
        let diarizedSegments = diarization.segments(withDefaultSpeaker: trackName)

        guard !diarizedSegments.isEmpty else {
            return TranscriptionOutput.empty(
                language: arguments.language,
                modelIdentifier: arguments.modelIdentifier
            )
        }

        let audioSamples = try AudioConverter().resampleAudioFile(arguments.audioPath)
        let modelsDirectory = resolvedModelsDirectory()
        let asrModels = try await AsrModels.downloadAndLoad(
            to: modelsDirectory,
            configuration: nil,
            version: arguments.modelVersion
        )

        let asrManager = AsrManager()
        try await asrManager.initialize(models: asrModels)

        let minSamples = Int(arguments.minSegmentSeconds * Double(sampleRate))
        var segments: [SegmentOutput] = []
        var combinedText: [String] = []

        for (index, segment) in diarizedSegments.enumerated() {
            guard let slice = slice(for: segment, totalSamples: audioSamples.count) else {
                continue
            }

            let length = slice.end - slice.start
            if length < minSamples {
                continue
            }

            let chunk = Array(audioSamples[slice.start..<slice.end])
            let asrResult = try await asrManager.transcribe(chunk, source: .system)

            let trimmed = asrResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            combinedText.append(trimmed)

            // Token-level output can be restored by re-enabling the block below.
            /*
            let tokens: [TokenOutput]? = (asrResult.tokenTimings ?? []).map { timing in
                TokenOutput(
                    token: timing.token,
                    tokenId: timing.tokenId,
                    start: segment.start + timing.startTime,
                    end: segment.start + timing.endTime,
                    confidence: Double(timing.confidence)
                )
            }
            */
            let tokens: [TokenOutput]? = nil

            let outputSegment = SegmentOutput(
                start: segment.start,
                end: segment.end,
                text: trimmed,
                speaker: segment.label(for: trackName),
                segmentId: index,
                confidence: Double(asrResult.confidence),
                tokens: tokens
            )
            segments.append(outputSegment)
        }

        try await asrManager.resetDecoderState()

        return TranscriptionOutput(
            segments: segments,
            text: combinedText.joined(separator: " "),
            language: arguments.language,
            model: arguments.modelIdentifier,
            engine: "parakeet-coreml",
            totalSegments: segments.count
        )
    }

    private func slice(for segment: SpeakerSegment, totalSamples: Int) -> (start: Int, end: Int)? {
        guard segment.end > segment.start else { return nil }
        let startSample = max(0, Int(segment.start * Double(sampleRate)))
        let endSample = min(totalSamples, Int(segment.end * Double(sampleRate)))
        guard endSample > startSample else { return nil }
        return (startSample, endSample)
    }

    private func resolvedModelsDirectory() -> URL {
        guard let modelsRoot = arguments.modelsRoot else {
            return AsrModels.defaultCacheDirectory(for: arguments.modelVersion)
        }

        let lowercased = modelsRoot.lastPathComponent.lowercased()
        if lowercased.contains(arguments.modelVersion.repoFolderName.lowercased()) {
            return modelsRoot
        }

        return modelsRoot.appendingPathComponent(arguments.modelVersion.repoFolderName, isDirectory: true)
    }
}

// MARK: - Diarization payload

struct SpeakerSegment {
    let start: Double
    let end: Double
    let diarizedSpeaker: String?

    func label(for trackName: String) -> String {
        let speaker = diarizedSpeaker?.isEmpty == false ? diarizedSpeaker! : trackName
        if speaker == trackName {
            return speaker
        }
        return "\(trackName):\(speaker)"
    }
}

struct DiarizationPayload: Decodable {
    struct Segment: Decodable {
        let start: Double
        let end: Double
        let speaker: String?
    }

    struct VadSegment: Decodable {
        let start: Double
        let end: Double
    }

    let mergedSegments: [Segment]?
    let vadSegments: [VadSegment]?

    static func load(from url: URL) throws -> DiarizationPayload {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(DiarizationPayload.self, from: data)
    }

    func segments(withDefaultSpeaker trackName: String) -> [SpeakerSegment] {
        if let mergedSegments, !mergedSegments.isEmpty {
            return mergedSegments.map { segment in
                SpeakerSegment(
                    start: segment.start,
                    end: segment.end,
                    diarizedSpeaker: segment.speaker
                )
            }
        }

        guard let vadSegments else { return [] }
        return vadSegments.map { segment in
            SpeakerSegment(start: segment.start, end: segment.end, diarizedSpeaker: trackName)
        }
    }
}

// MARK: - Output payload

struct TokenOutput: Codable {
    let token: String
    let tokenId: Int
    let start: Double
    let end: Double
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case token
        case tokenId = "token_id"
        case start
        case end
        case confidence
    }
}

struct SegmentOutput: Codable {
    let start: Double
    let end: Double
    let text: String
    let speaker: String
    let segmentId: Int
    let confidence: Double
    let tokens: [TokenOutput]?

    enum CodingKeys: String, CodingKey {
        case start
        case end
        case text
        case speaker
        case segmentId = "segment_id"
        case confidence
        case tokens
    }
}

struct TranscriptionOutput: Codable {
    let segments: [SegmentOutput]
    let text: String
    let language: String
    let model: String
    let engine: String
    let totalSegments: Int

    enum CodingKeys: String, CodingKey {
        case segments
        case text
        case language
        case model
        case engine
        case totalSegments = "total_segments"
    }

    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }

    static func empty(language: String, modelIdentifier: String) -> TranscriptionOutput {
        TranscriptionOutput(
            segments: [],
            text: "",
            language: language,
            model: modelIdentifier,
            engine: "parakeet-coreml",
            totalSegments: 0
        )
    }
}

private extension Arguments {
    var modelIdentifier: String {
        modelVersion.repoIdentifier
    }
}

private extension AsrModelVersion {
    var repoIdentifier: String {
        switch self {
        case .v2:
            return Repo.parakeetV2.rawValue
        case .v3:
            return Repo.parakeet.rawValue
        }
    }

    var repoFolderName: String {
        switch self {
        case .v2:
            return Repo.parakeetV2.folderName
        case .v3:
            return Repo.parakeet.folderName
        }
    }
}

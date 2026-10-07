// SwiftUI / iOS G2P — same contract as phonemize.py
// Bundle: data/*.json + lib/{libespeak.dylib,espeak-data,ja}
// espeak_Initialize(..., lib, 0) — lib contains espeak-data. Mode 19. Not ng. Not 51.
// ja: compile OpenJTalk C (MeCab/NJD/JPCommon, no HTS). Dict = lib/ja/.../sys.dic. Not the CPython .so.
// Never ship voices/en, default, en-n/rp/sc/wi/wm, !v, mbrola.
import Foundation

public final class YunaPhonemizer {
	public let symbols: [String]
	public let symbolToId: [String: Int]
	private let marksRe: NSRegularExpression
	private let flagRe: NSRegularExpression
	private let aliases: [String: String]
	private let voices: [String: String]
	private let files: [String: String]
	private var voice: String?

	public init(bundle: URL) throws {
		let data = bundle.appendingPathComponent("data")
		let symJSON = try Self.json(data.appendingPathComponent("symbols.json"))
		self.symbols = symJSON["symbols"] as! [String]
		precondition(symbols.count == 177)
		self.symbolToId = Dictionary(uniqueKeysWithValues: symbols.enumerated().map { ($1, $0) })
		let punct = try Self.json(data.appendingPathComponent("punct.json"))
		let markStr = punct["default_marks"] as! String
		self.marksRe = try NSRegularExpression(pattern: "(\\s*[\(NSRegularExpression.escapedPattern(for: markStr))]+\\s*)+")
		self.flagRe = try NSRegularExpression(pattern: punct["language_switch_re"] as! String)
		let langs = try Self.json(data.appendingPathComponent("langs.json"))
		self.voices = langs["voices"] as! [String: String]
		self.files = langs["files"] as? [String: String] ?? [:]
		self.aliases = langs["aliases"] as! [String: String]
		let parent = bundle.appendingPathComponent("lib").path
		if espeak_Initialize(2, 0, parent, 0) <= 0 { throw NSError(domain: "yuna", code: 1, userInfo: [NSLocalizedDescriptionKey: "espeak_Initialize failed"]) }
	}

	deinit { espeak_Terminate() }

	public func canon(_ lang: String) -> String { aliases[lang] ?? lang }

	public func phonemesRaw(_ text: String, lang: String) -> String {
		let lid = canon(lang)
		if lid == "ja" { return text } // OpenJTalk C frontend + ja_rewrite.json — not in this file
		return g2p(text.trimmingCharacters(in: .whitespacesAndNewlines), lang: lid).trimmingCharacters(in: .whitespacesAndNewlines)
	}

	/// Mirrors fold_phones in phonemize.py: two g2p spellings that duplicate a symbol already in
	/// the table are folded onto it instead of taking slots of their own. "--" is the em dash in
	/// Japanese rows (always a run of two; en-us and ru spell the same pause "—"), and "^" is the
	/// Russian soft sign leaking out as ASCII where the table has "ʲ". Keep in step with Python
	/// or Core AI and PyTorch will tokenise the same text differently.
	public func foldPhones(_ ipa: String) -> String {
		var out = ipa.replacingOccurrences(of: "^", with: "\u{02B2}")
		while out.contains("--") { out = out.replacingOccurrences(of: "--", with: "\u{2014}") }
		return out.replacingOccurrences(of: "-", with: "\u{2014}")
	}

	public func cleanedIds(_ ipa: String) -> [Int] {
		foldPhones(ipa).unicodeScalars.compactMap { symbolToId[String($0)] }
	}

	private func g2p(_ text: String, lang: String) -> String {
		setVoice(lang)
		if marksRe.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) == nil { return post(raw(text)) }
		let (chunks, marks) = preserve(text)
		let ph = chunks.map { post(raw($0)) }
		return restore(ph, marks)
	}

	private func setVoice(_ code: String) {
		if voice == code { return }
		guard let name = voices[code] else { return }
		for cand in [name, files[code] ?? "", code] where !cand.isEmpty {
			if cand.withCString({ espeak_SetVoiceByName($0) }) == 0 { voice = code; return }
		}
	}

	private func raw(_ text: String) -> String {
		var bytes = Array(text.utf8) + [0]
		return bytes.withUnsafeMutableBufferPointer { buf in
			var ptr: UnsafePointer<CChar>? = UnsafePointer(buf.baseAddress)
			var parts: [String] = []
			while ptr != nil {
				if let ph = espeak_TextToPhonemes(&ptr, 1, 19) { parts.append(String(cString: ph)) }
			}
			return parts.joined(separator: " ")
		}
	}

	private func post(_ line: String) -> String {
		var s = line.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\n", with: " ")
		while s.contains("  ") { s = s.replacingOccurrences(of: "  ", with: " ") }
		s = s.replacingOccurrences(of: "_+", with: "_", options: .regularExpression)
		s = s.replacingOccurrences(of: "_ ", with: " ")
		if s.contains("(") { s = flagRe.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "") }
		if s.isEmpty { return "" }
		return s.split(separator: " ", omittingEmptySubsequences: false).map { $0.replacingOccurrences(of: "_", with: "") + " " }.joined()
	}

	private struct Mark { var mark: String; var position: String } // B I E A

	private func preserve(_ line: String) -> ([String], [Mark]) {
		let ns = line as NSString
		let matches = marksRe.matches(in: line, range: NSRange(location: 0, length: ns.length))
		if matches.isEmpty { return ([line], []) }
		if matches.count == 1 && ns.substring(with: matches[0].range) == line { return ([], [Mark(mark: line, position: "A")]) }
		var marks: [Mark] = []
		for (i, m) in matches.enumerated() {
			let g = ns.substring(with: m.range)
			var pos = "I"
			if i == 0 && line.hasPrefix(g) { pos = "B" }
			else if i == matches.count - 1 && line.hasSuffix(g) { pos = "E" }
			marks.append(Mark(mark: g, position: pos))
		}
		var rest = line
		var chunks: [String] = []
		for m in marks {
			let parts = rest.components(separatedBy: m.mark)
			chunks.append(parts[0])
			rest = parts.dropFirst().joined(separator: m.mark)
		}
		chunks.append(rest)
		return (chunks.filter { !$0.isEmpty }, marks)
	}

	private func restore(_ phonemes: [String], _ marks: [Mark]) -> String {
		var text = phonemes
		var marks = marks
		var out: [String] = []
		var pos = 0
		while !text.isEmpty || !marks.isEmpty {
			if marks.isEmpty {
				for var line in text { if !line.hasSuffix(" ") { line += " " }; out.append(line) }
				text.removeAll()
			} else if text.isEmpty {
				out.append(marks.map(\.mark).joined())
				marks.removeAll()
			} else if marks[0].position == "B" {
				if text[0].hasSuffix(" ") { text[0].removeLast() }
				text[0] = marks.removeFirst().mark + text[0]
			} else if marks[0].position == "E" {
				if text[0].hasSuffix(" ") { text[0].removeLast() }
				let mark = marks.removeFirst().mark
				out.append(text.removeFirst() + mark + (mark.hasSuffix(" ") ? "" : " "))
				pos += 1
			} else if marks[0].position == "A" {
				let mark = marks.removeFirst().mark
				out.append(mark + (mark.hasSuffix(" ") ? "" : " "))
				pos += 1
			} else {
				if text[0].hasSuffix(" ") { text[0].removeLast() }
				if text.count == 1 { text[0] += marks.removeFirst().mark }
				else {
					let first = text.removeFirst()
					text[0] = first + marks.removeFirst().mark + text[0]
				}
			}
		}
		return out.joined()
	}

	private static func json(_ url: URL) throws -> [String: Any] {
		try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
	}
}

@_silgen_name("espeak_Initialize") func espeak_Initialize(_ output: Int32, _ buflen: Int32, _ path: UnsafePointer<CChar>?, _ options: Int32) -> Int32
@_silgen_name("espeak_SetVoiceByName") func espeak_SetVoiceByName(_ name: UnsafePointer<CChar>?) -> Int32
@_silgen_name("espeak_TextToPhonemes") func espeak_TextToPhonemes(_ textptr: UnsafeMutablePointer<UnsafePointer<CChar>?>, _ textmode: Int32, _ phonememode: Int32) -> UnsafePointer<CChar>?
@_silgen_name("espeak_Terminate") func espeak_Terminate()

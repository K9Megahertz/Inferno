#include <infernotokenizer/bpetokenizer.h>
#include <iostream>
#include <fstream>
#include <sstream>
namespace InfernoTokenizer {



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function Initialize()
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	bool BPETokenizer::Initialize(const TokenizerConfig& config) {

		load_merges(config.merges_file);
		load_vocab(config.vocab_file);


		
		m_special_tokens["<|endoftext|>"] = 60257;
		m_id_to_special[60257] = "<|endoftext|>";

		m_special_tokens["<|user|>"] = 60258;
		m_id_to_special[60258] = "<|user|>";

		m_special_tokens["<|assistant|>"] = 60259;
		m_id_to_special[60259] = "<|assistant|>";
			

		return true;
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function find_special_at()
//
//  Check whether any special token begins at the given position.
//
//  Example:
//
//      text = "hello<|endoftext|>"
//
//                    ^
//                   pos
//
//  If found:
//
//      matched_special = "<|endoftext|>"
//      special_id      = 50256
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	bool BPETokenizer::find_special_at(const std::string& text, size_t pos, std::string& matched_special, uint32_t& special_id) const {

		matched_special.clear();

		special_id = 0;

		// Check every registered special token.
		for (const auto& [special, id] : m_special_tokens) {

			// Ignore empty strings.
			if (special.empty()) {
				continue;
			}

			// Not enough remaining bytes to match.
			if (pos + special.size() > text.size()) {
				continue;
			}

			// Compare raw text against special token string.
			if (text.compare(pos, special.size(), special) == 0) {

				// Prefer longest match if multiple overlap.
				if (special.size() > matched_special.size()) {
					matched_special = special;
					special_id = id;
				}
			}
		}

		// True if we found a match.
		return !matched_special.empty();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function encode_normal_text()
//
//  This is the OLD encode logic.
//
//  This function handles ordinary text ONLY.
//
//  Steps:
//
//      1. Pretokenize text into pieces
//      2. Convert each piece into byte IDs
//      3. Apply BPE merges
//      4. Append final token IDs to output
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void BPETokenizer::encode_normal_text(const std::string& text, std::vector<uint32_t>& ret) {

		// Nothing to process.
		if (text.empty()) {
			return;
		}

		// Split text into pretokenized pieces.
		//
		// Example:
		//
		//      " hello world"
		//
		// might become:
		//
		//      [" hello", " world"]
		//
		std::vector<std::string> pieces = tok.split(text);

		// Process each piece independently.
		for (const std::string& piece : pieces) {

			// Convert raw bytes into initial token stream.
			//
			// Initially every byte is its own token.
			//
			// Example:
			//
			//      "hi"
			//
			// becomes:
			//
			//      [104, 105]
			//
			std::vector<uint32_t> piecebytes;

			for (unsigned char c : piece) {
				piecebytes.push_back(static_cast<uint32_t>(c));
			}

			// Repeatedly apply the best-ranked merge.
			while (true) {

				// Lowest merge rank found so far.
				uint32_t bestrank = std::numeric_limits<uint32_t>::max();

				// Token ID produced by the best merge.
				uint32_t besttoken = 0;

				// Position of the best merge.
				int bestindex = -1;

				// Scan all adjacent pairs.
				//
				// Example:
				//
				//      [104, 101, 108, 108, 111]
				//
				// checks:
				//
				//      (104,101)
				//      (101,108)
				//      (108,108)
				//      (108,111)
				//
				for (size_t i = 0; i + 1 < piecebytes.size(); i++) {

					uint32_t a = piecebytes[i];
					uint32_t b = piecebytes[i + 1];

					// Pack pair into 64-bit key.
					uint64_t pairval =
						(uint64_t(a) << 32) | uint64_t(b);

					// Check whether this pair has a merge rule.
					auto it = m_mergemap.find(pairval);

					if (it != m_mergemap.end()) {

						const MergeEntry& entry = it->second;

						// Lower rank = higher priority merge.
						if (entry.rank < bestrank) {
							bestrank = entry.rank;
							besttoken = entry.token;
							bestindex = static_cast<int>(i);
						}
					}
				}

				// No more merges found.
				//
				// Piece is fully encoded.
				//
				if (bestindex == -1) {
					break;
				}

				// Apply the merge.
				//
				// Example:
				//
				//      [104,101]
				//
				// becomes:
				//
				//      [256]
				//
				piecebytes[bestindex] = besttoken;

				// Remove the second token of the merged pair.
				piecebytes.erase(piecebytes.begin() + bestindex + 1);
			}

			// Append final encoded tokens to output stream.
			ret.insert(ret.end(), piecebytes.begin(), piecebytes.end());
		}
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function encode()
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function encode()
//
//  This is the main entry point for converting raw text into token IDs.
//
//  The important thing here is that we scan for special tokens FIRST,
//  before sending normal text through the PreTokenizer + BPE logic.
//
//  Example:
//
//      "hello <|endoftext|> world"
//
//  becomes:
//
//      BPE("hello ")
//      endoftext_id
//      BPE(" world")
//
//  The literal string "<|endoftext|>" never goes through the pretokenizer.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::vector<uint32_t> BPETokenizer::encode(const std::string& text) {

		// Final output token stream.
		std::vector<uint32_t> ret;

		// Temporary buffer for collecting normal text.
		//
		// We accumulate ordinary text here until we hit a special token.
		//
		// Example:
		//
		//      "hello world<|endoftext|>next"
		//
		// normal_buffer grows as:
		//
		//      "h"
		//      "he"
		//      "hel"
		//      ...
		//      "hello world"
		//
		std::string normal_buffer;

		// Current position in the raw input text.
		size_t i = 0;

		// Walk through the raw input text one byte at a time.
		while (i < text.size()) {

			// If we find a special token here,
			// this string will hold the matched token text.
			//
			// Example:
			//
			//      "<|endoftext|>"
			//
			std::string matched_special;

			// ID associated with the matched special token.
			//
			// Example:
			//
			//      50256
			//
			uint32_t special_id = 0;

			// Check whether a special token starts at this position.
			//
			// Example:
			//
			//      text = "hello<|endoftext|>"
			//                    ^
			//                    i
			//
			if (find_special_at(text, i, matched_special, special_id)) {

				// We found a special token.
				//
				// First, flush all accumulated normal text through the
				// normal BPE encoding pipeline.
				//
				// Example:
				//
				//      normal_buffer = "hello"
				//
				// gets tokenized normally.
				//
				encode_normal_text(normal_buffer, ret);

				// Clear the buffer now that we have processed it.
				normal_buffer.clear();

				// Emit the special token ID directly.
				//
				// IMPORTANT:
				//
				// The literal string "<|endoftext|>" NEVER goes through
				// the pretokenizer or BPE merge logic.
				//
				// Instead, we directly insert its reserved token ID.
				//
				ret.push_back(special_id);

				// Skip past the literal special-token text in the input.
				//
				// Example:
				//
				//      "<|endoftext|>"
				//
				// move i forward by the full token length.
				//
				i += matched_special.size();
			}
			else {

				// No special token starts here.
				//
				// This is ordinary text, so append it to the normal buffer.
				//
				normal_buffer.push_back(text[i]);

				// Advance to next byte.
				i++;
			}
		}

		// Process any remaining normal text left at the end.
		//
		// Example:
		//
		//      "hello<|endoftext|>world"
		//
		// after the loop:
		//
		//      normal_buffer = "world"
		//
		encode_normal_text(normal_buffer, ret);

		return ret;
	}

	/*std::vector<uint32_t> BPETokenizer::encode(const std::string& text) {

		std::vector<uint32_t> ret{};

		std::vector<std::string> pieces = tok.split(text);


		//loop through every piece in our list
		for (std::string piece : pieces) {

			//std::cout << "Working on piece: " << piece << std::endl;


			//convert to bytes
			std::vector<uint32_t> piecebytes;
			for (unsigned char c : piece) {
				piecebytes.push_back((uint32_t)c);
			}



			while (true) {

				uint32_t bestrank = std::numeric_limits<uint32_t>::max();
				uint64_t besttoken = 0;
				int bestindex = -1;

				for (int i = 0; i < piecebytes.size() - 1; i++) {
					uint32_t a = piecebytes[i];
					uint32_t b = piecebytes[i+1];
					//std::cout << a << ", " << b << std::endl;
					
					uint64_t pairval = (uint64_t)a << 32 | (uint64_t)b;

					auto it = m_mergemap.find(pairval);

					if (it != m_mergemap.end()) {
						const MergeEntry& entry = it->second;

						if (entry.rank < bestrank) {
							bestrank = entry.rank;
							besttoken = entry.token;
							bestindex = i;
						}
					}
				}

				//didnt find any matches, we done
				if (bestindex == -1)
					break;  //break out of while


				//we have now found the best
				piecebytes[bestindex] = besttoken;
				piecebytes.erase(piecebytes.begin() + bestindex + 1);


			}

			ret.insert(ret.end(), piecebytes.begin(), piecebytes.end());


		}
		
		return ret;

	}*/	



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function decode()
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) {
		std::string out;

		for (uint32_t t : tokens) {

			// First check whether this token is a special token.
			auto special_it = m_id_to_special.find(t);

			if (special_it != m_id_to_special.end()) {
				out += special_it->second;
				continue;
			}

			// Otherwise decode as a normal BPE/byte token.
			if (t >= m_vocablist.size()) {
				throw std::runtime_error("BPETokenizer::decode invalid token id");
			}

			out += m_vocablist[t];
		}

		return out;
	}

	std::string BPETokenizer::decode(const std::vector<int>& tokens) {
		std::vector<uint32_t> converted;
		converted.reserve(tokens.size());

		for (int t : tokens) {
			if (t < 0) {
				throw std::runtime_error("BPETokenizer::decode negative token id");
			}

			converted.push_back(static_cast<uint32_t>(t));
		}

		return decode(converted);
	}

	std::string BPETokenizer::decode(uint32_t token) {
		std::vector<uint32_t> tokens;
		tokens.push_back(token);
		return decode(tokens);
	}

	


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function load_merges()
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void BPETokenizer::load_merges(const std::string& filename) {

		std::ifstream in(filename);

		if (!in) {
			std::cerr << "Failed to open merges file: " << filename << "\n";
			return;
		}

		uint32_t a, b;

		uint32_t rank = 0;
		uint32_t token = 256;

		while (in >> a >> b) {

			uint64_t key = (uint64_t(a) << 32) | uint64_t(b);

			m_mergemap[key] = { token, rank };

			token++;
			rank++;
		}
	}



	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//
	//  Function load_vocab()
	//
	//
	//
	//
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void BPETokenizer::load_vocab(const std::string& filename) {

		std::string line;
		std::ifstream file(filename);

		if (!file) {
			std::cerr << "Failed to open vocab file: " << filename << "\n";
			return;
		}

		while (std::getline(file, line)) {
			std::istringstream iss(line);

			uint32_t token_id;
			char colon;

			iss >> token_id >> colon;

			if (token_id >= m_vocablist.size())
				m_vocablist.resize(token_id + 1);

			std::string token_str;
			uint32_t val;

			while (iss >> val) {
				token_str += static_cast<char>(val);
			}

			m_vocablist[token_id] = token_str;
		}
	}


}
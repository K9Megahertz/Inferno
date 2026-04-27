#include <pretokenizer/pretokenizer.h>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace Tokenizer {

    class PreTokenizer::Impl {


    public:

        
        void process_string(const std::string& string);
        void flush_current_piece();
        void reset();
        //const std::unordered_map<std::string, uint64_t>& result() const {
        const std::vector<std::string>& pieces() const {
            //return m_piece_freqs;
            return m_pieces;
        }

    private:

        static bool is_space_byte(unsigned char b);
        static bool is_ascii_letter(unsigned char b);
        static bool is_ascii_digit(unsigned char b);
        static bool is_word_like_byte(unsigned char b);

        void emit_piece(const std::string& piece);
        void start_new_piece_with_pending(unsigned char b);
        void process_byte(unsigned char b);

        // This map stores piece -> count
        // Example:
        // "dog"   -> 12
        // " world"-> 8
        // ","     -> 25
        //std::unordered_map<std::string, uint64_t> m_piece_freqs;
        std::vector<std::string> m_pieces;

        // This stores whitespace that we have seen but have not attached yet.
        // Example:
        // if we read two spaces before "dog", this might temporarily hold "  "
        std::string m_pending_spaces;

        // This is the token we are currently building.
        // Example:
        // while reading the bytes for "dog", this grows as:
        // "d" -> "do" -> "dog"
        std::string m_current_piece;



    };
    
  

    PreTokenizer::PreTokenizer() : m_impl(std::make_unique<Impl>()) {

    }

    PreTokenizer::~PreTokenizer() = default;    

    

    std::vector<std::string> PreTokenizer::split(const std::string& text) {
        m_impl->reset();
        m_impl->process_string(text);
        m_impl->flush_current_piece();
        return m_impl->pieces();
    }  


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void PreTokenizer::Impl::reset() {
        m_pieces.clear();
        m_pending_spaces.clear();
        m_current_piece.clear();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  Return true if the byte is one of the whitespace bytes we care about.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    bool PreTokenizer::Impl::is_space_byte(unsigned char b) {
        return b == ' ' || b == '\t' || b == '\n' || b == '\r';
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  Return true if the byte is an ASCII letter.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    bool PreTokenizer::Impl::is_ascii_letter(unsigned char b) {
        return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z');
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  Return true if the byte is an ASCII digit.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    bool PreTokenizer::Impl::is_ascii_digit(unsigned char b) {
        return (b >= '0' && b <= '9');
    }

    

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //
    //  Decide whether this byte should be considered part of a "word-like" run.
    //
    //  For this simple version, word-like means:
    //  - ASCII letters
    //  - ASCII digits
    //  - underscore
    //  - any byte >= 128
    //
    //  That last rule is a simple trick:
    //  UTF-8 bytes for non-ASCII text are >= 128, so this keeps them together
    //  as part of the same run instead of splitting them apart immediately.
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool PreTokenizer::Impl::is_word_like_byte(unsigned char b) {
        return is_ascii_letter(b) || is_ascii_digit(b) || b == '_' || b >= 128;
    }



    void PreTokenizer::Impl::process_string(const std::string& string) {
        for (unsigned char c : string) {            
            process_byte(c);
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  Add one completed piece to the frequency table.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    void PreTokenizer::Impl::emit_piece(const std::string& piece) {

        // Ignore empty strings.
        if (!piece.empty()) {

            // Increment the count for this piece.
            //++m_piece_freqs[piece];
            m_pieces.push_back(piece);
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  If we are currently building a piece, emit it and clear it.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    void PreTokenizer::Impl::flush_current_piece() {

        // Only emit if there is something to emit.
        if (!m_current_piece.empty()) {

            // Add the piece to the frequency table.
            emit_piece(m_current_piece);

            // Reset current piece to empty.
            m_current_piece.clear();
        }
    }



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //
    //  Start a new token using any pending spaces plus the current byte.
    //
    //  Example:
    //  pending spaces = "  "
    //  new byte = 'd'
    //  result current piece = "  d"
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    
    void PreTokenizer::Impl::start_new_piece_with_pending(unsigned char b) {

        // Make sure current piece starts empty.
        m_current_piece.clear();

        // Attach any spaces we saved earlier.
        m_current_piece += m_pending_spaces;

        // Clear pending spaces now that they have been used.
        m_pending_spaces.clear();

        // Append the current byte to start the new piece.
        m_current_piece.push_back(static_cast<char>(b));
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Function train()
    //  Core tokenizer logic for one byte.
    //
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    void PreTokenizer::Impl::process_byte(unsigned char b) {

        // ------------------------------------------------------------
        // Case 1: whitespace
        // ------------------------------------------------------------
        if (is_space_byte(b)) {

            // If we were in the middle of building a token, finish it now.
            // Whitespace ends the current token.
            flush_current_piece();

            // Save this whitespace so it can be attached to the next token.
            m_pending_spaces.push_back(static_cast<char>(b));

            // Done handling this byte.
            return;
        }

        // ------------------------------------------------------------
        // Case 2: word-like byte
        // ------------------------------------------------------------
        if (is_word_like_byte(b)) {

            // If there is no current token yet, start one.
            if (m_current_piece.empty()) {

                // Start new token with any pending spaces attached.
                start_new_piece_with_pending(b);
            }
            else {

                // Look at the last byte currently in the token.
                unsigned char last = static_cast<unsigned char>(m_current_piece.back());

                // If the last byte was also word-like, keep extending this token.
                if (is_word_like_byte(last)) {

                    // Append this byte to the current token.
                    m_current_piece.push_back(static_cast<char>(b));
                }
                else {

                    // Otherwise, the token type changed.
                    // Finish the old token first.
                    flush_current_piece();

                    // Then start a new one with pending spaces.
                    start_new_piece_with_pending(b);
                }
            }

            // Done handling this byte.
            return;
        }

        // ------------------------------------------------------------
        // Case 3: punctuation / symbol / anything else
        // ------------------------------------------------------------

        // If we were building a token, end it before handling punctuation.
        flush_current_piece();

        // Create a token for this punctuation/symbol byte.
        std::string piece;

        // Attach any saved spaces to it first.
        piece += m_pending_spaces;

        // Clear pending spaces now that they are being used.
        m_pending_spaces.clear();

        // Add the punctuation/symbol byte itself.
        piece.push_back(static_cast<char>(b));

        // Emit this punctuation token immediately.
        emit_piece(piece);
    }

}
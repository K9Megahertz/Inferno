#include "token.h"
#pragma once
class Tokenizer {




public:

	Tokenizer(std::istream& stream, const unsigned long filesize);
	~Tokenizer() = default;

	bool skip_whitespace();
	Token* parse_whitespace();	
	void advance();

	Token *get_next_token();

private:

	std::vector<unsigned char> currentchars;
	unsigned long cursorpos = 0;	
	std::istream& stream;
	const unsigned long streamlength;


};
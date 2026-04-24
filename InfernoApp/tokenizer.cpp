#include "tokenizer.h"
#include <fstream>
#include <cctype>


Tokenizer::Tokenizer(std::istream& s, const unsigned long filesize) : stream(s), streamlength(filesize) {

	char c;
	stream.get(c);                                        //get first character in the stream
	currentchars.push_back(static_cast<unsigned char>(c));
}

bool Tokenizer::skip_whitespace() {

	bool skipped = false;
	while (currentchars[0] == ' ' || currentchars[0] == 10 || currentchars[0] == 13 || currentchars[0] == '\t') {   //skip any whitespace characters
		//advance();                                                                     //advance to next character		
		skipped = true;
	}
	return skipped;

}



Token* Tokenizer::get_next_token() {

	while (currentchars[0] != '\0') {


		if (std::isspace(currentchars[0])) {
			return parse_whitespace();

		}	



	}

}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function parse_whitespace()
//  
//  
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Token* Tokenizer::parse_whitespace() {
	
	std::vector<unsigned char> s;
	while (std::isspace(currentchars[0]))
	{
		s.push_back(currentchars[0]);		                                                      //add character to string
		advance();                                                                        //advance to next character
	}

	return new Token(Token::TokenType::TOKEN_WHITESPACE, s);                          //return the token

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Function advance()
// 
// 
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Tokenizer::advance() {

	if (currentchars[0] != '\0' && cursorpos < streamlength) {                   //make sure were not past the end of the file
		cursorpos++;                                                            //increment the index
		currentchars.clear();
		char c;
		stream.get(c);                                             //get next character in the stream
		currentchars.push_back(static_cast<unsigned char>(c));		
	}
	else {
		currentchars.clear();
		currentchars.push_back('\0');
	}
}
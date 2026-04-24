#pragma once
#include <string>
#include <vector>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Token
//
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Token
{

public:

	enum class TokenType {
		TOKEN_NULL,
		TOKEN_WHITESPACE,
		TOKEN_ID,
		TOKEN_EQUALS,
		TOKEN_STRING,
		TOKEN_SEMI,
		TOKEN_LPAREN,
		TOKEN_RPAREN,
		TOKEN_LBRACE,
		TOKEN_RBRACE,
		TOKEN_LBRACKET,
		TOKEN_RBRACKET,
		TOKEN_COLON,
		TOKEN_COMMA,
		TOKEN_FORWARD_SLASH,
		TOKEN_BACKWARD_SLASH,
		TOKEN_POUND,
		TOKEN_LT,
		TOKEN_GT,
		TOKEN_ASTERISK,
		TOKEN_MINUS,
		TOKEN_PLUS,
		TOKEN_EXCLAMATION_MARK,
		TOKEN_PERIOD,
		TOKEN_QUESTION_MARK,
		TOKEN_CARET,
		TOKEN_AT,
		TOKEN_PERCENT,
		TOKEN_AMPERSAND,
		TOKEN_PIPE,
		TOKEN_CHAR,
		TOKEN_INT,
		TOKEN_DOLLAR,
		TOKEN_TILDE,
		TOKEN_BACKTICK,
		TOKEN_SINGLE_QUOTE,
		TOKEN_DOUBLE_QUOTE,
		TOKEN_UTF8,
		TOKEN_EOF
	};

	Token(Token::TokenType type, std::vector<unsigned char> currentchars);
	Token();
	~Token();


	TokenType m_type;
	std::vector<unsigned char> m_contents;

	static std::string toString(Token::TokenType tt);


};

